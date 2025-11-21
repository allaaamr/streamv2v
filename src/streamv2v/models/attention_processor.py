# attention_processor.py
# Cached attention processors:
# - Self-attn: keep a rolling cache of K/V (and optionally outputs) across frames; concat with current.
# - Cross-attn: cache prompt-context K/V once per unique context and reuse across frames.

from collections import deque
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import xformers.ops as xops  # type: ignore
except Exception:
    xops = None


def _context_signature(x: torch.Tensor) -> Tuple[Tuple[int, ...], torch.dtype, torch.device, float]:
    with torch.no_grad():
        return (tuple(x.shape), x.dtype, x.device, float(x.mean().detach().float().item()))


class _CachedBase(nn.Module):
    def __init__(
        self,
        name: str = "",
        use_feature_injection: bool = False,
        feature_injection_strength: float = 0.8,
        feature_similarity_threshold: float = 0.98,
        interval: int = 4,
        max_frames: int = 1,
        use_tome_cache: bool = False,
        tome_metric: str = "keys",
        tome_ratio: float = 0.5,
        use_grid: bool = False,
        cache_cross_attention: bool = True,
        **kwargs,  # ignore extra args for compatibility
    ):
        super().__init__()
        self.name = name
        self.interval = max(1, int(interval))
        self.max_frames = max(1, int(max_frames))
        self.use_feature_injection = bool(use_feature_injection)
        self.feature_injection_strength = float(feature_injection_strength)
        self.feature_similarity_threshold = float(feature_similarity_threshold)
        self.use_tome_cache = bool(use_tome_cache)
        self.tome_metric = tome_metric
        self.tome_ratio = float(tome_ratio)
        self.use_grid = bool(use_grid)
        self.cache_cross_attention = bool(cache_cross_attention)

        # Rolling self-attn caches
        self.cached_key = deque(maxlen=self.max_frames)
        self.cached_value = deque(maxlen=self.max_frames)
        self.cached_output = deque(maxlen=self.max_frames) if self.use_feature_injection else None

        # Pre-concatenated views (fast-path)
        self._cat_k: Optional[torch.Tensor] = None
        self._cat_v: Optional[torch.Tensor] = None
        self._cat_out: Optional[torch.Tensor] = None

        # Cross-attn single context cache
        self._cross_k: Optional[torch.Tensor] = None
        self._cross_v: Optional[torch.Tensor] = None
        self._cross_sig: Optional[Tuple] = None

        self._frame_id: int = 0

    def _refresh_cat_views(self):
        if len(self.cached_key) > 0:
            self._cat_k = torch.cat(list(self.cached_key), dim=1)
            self._cat_v = torch.cat(list(self.cached_value), dim=1)
            if self.use_feature_injection and self.cached_output and len(self.cached_output) > 0:
                self._cat_out = torch.cat(list(self.cached_output), dim=1)
            else:
                self._cat_out = None
        else:
            self._cat_k = self._cat_v = self._cat_out = None

    def _maybe_push_self_cache(self, k_cur: torch.Tensor, v_cur: torch.Tensor, out_cur: Optional[torch.Tensor]):
        if (self._frame_id % self.interval) == 0:
            self.cached_key.append(k_cur)
            self.cached_value.append(v_cur)
            if self.use_feature_injection and self.cached_output is not None and out_cur is not None:
                self.cached_output.append(out_cur)
            self._refresh_cat_views()

    def reset_cross_cache(self):
        self._cross_k = None
        self._cross_v = None
        self._cross_sig = None

    # nn.Module.forward wrapper (diffusers calls this object directly)
    def forward(
        self,
        attn: nn.Module,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out = self._attention_core(attn, hidden_states, encoder_hidden_states, attention_mask)
        self._frame_id += 1
        return out

    # Implemented in subclasses
    def _attention_core(
        self,
        attn: nn.Module,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        raise NotImplementedError


# ---------------- SDPA (PyTorch) variant ----------------
class CachedSTAttnProcessor2_0(_CachedBase):
    def _attention_core(
        self,
        attn: nn.Module,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        residual = hidden_states

        # Q
        query = attn.to_q(hidden_states)
        if getattr(attn, "norm_q", None) is not None:
            query = attn.norm_q(query)

        # K,V
        is_cross = encoder_hidden_states is not None
        if is_cross:
            ctx = encoder_hidden_states
            if self.cache_cross_attention:
                sig = _context_signature(ctx)
                if self._cross_k is None or self._cross_sig != sig:
                    key = attn.to_k(ctx)
                    value = attn.to_v(ctx)
                    if getattr(attn, "norm_k", None): key = attn.norm_k(key)
                    if getattr(attn, "norm_v", None): value = attn.norm_v(value)
                    self._cross_k, self._cross_v, self._cross_sig = key, value, sig
                else:
                    key, value = self._cross_k, self._cross_v
            else:
                key = attn.to_k(ctx); value = attn.to_v(ctx)
                if getattr(attn, "norm_k", None): key = attn.norm_k(key)
                if getattr(attn, "norm_v", None): value = attn.norm_v(value)
            cached_out = None
        else:
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
            if getattr(attn, "norm_k", None): key = attn.norm_k(key)
            if getattr(attn, "norm_v", None): value = attn.norm_v(value)
            if self._cat_k is not None:
                key = torch.cat((key, self._cat_k), dim=1)
                value = torch.cat((value, self._cat_v), dim=1)
            cached_out = hidden_states if self.use_feature_injection else None

        # heads->batch
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # mask (SDPA expects [B*H, Q, K])
        if attention_mask is not None:
            bsz, q_len, _ = hidden_states.shape
            k_len = key.shape[1]
            attention_mask = attn.prepare_attention_mask(attention_mask, k_len, bsz)
            if attention_mask is not None:
                attention_mask = attention_mask.view(bsz, attn.heads, 1, k_len)
                attention_mask = attention_mask.expand(-1, -1, q_len, -1)
                attention_mask = attention_mask.reshape(bsz * attn.heads, q_len, k_len)

        out = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False, scale=attn.scale
        )
        out = attn.batch_to_head_dim(out)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)

        if not is_cross:
            # Recompute small per-frame k/v to push (no history) to avoid slicing
            k_cur = attn.to_k(residual if getattr(attn, "norm_k", None) is None else getattr(attn, "norm_k")(residual))
            v_cur = attn.to_v(residual if getattr(attn, "norm_v", None) is None else getattr(attn, "norm_v")(residual))
            self._maybe_push_self_cache(k_cur, v_cur, cached_out)

        return out


# ---------------- xFormers variant ----------------
class CachedSTXFormersAttnProcessor(_CachedBase):
    def _attention_core(
        self,
        attn: nn.Module,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        residual = hidden_states

        # Q
        query = attn.to_q(hidden_states)
        if getattr(attn, "norm_q", None) is not None:
            query = attn.norm_q(query)

        # K,V
        is_cross = encoder_hidden_states is not None
        if is_cross:
            # breakpoint()
            ctx = encoder_hidden_states
            if self.cache_cross_attention:
                sig = _context_signature(ctx)
                if self._cross_k is None or self._cross_sig != sig:
                    key = attn.to_k(ctx)
                    value = attn.to_v(ctx)
                    if getattr(attn, "norm_k", None): key = attn.norm_k(key)
                    if getattr(attn, "norm_v", None): value = attn.norm_v(value)
                    self._cross_k, self._cross_v, self._cross_sig = key, value, sig
                else:
                    key, value = self._cross_k, self._cross_v
            else:
                key = attn.to_k(ctx); value = attn.to_v(ctx)
                if getattr(attn, "norm_k", None): key = attn.norm_k(key)
                if getattr(attn, "norm_v", None): value = attn.norm_v(value)
            cached_out = None
        else:
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
            if getattr(attn, "norm_k", None): key = attn.norm_k(key)
            if getattr(attn, "norm_v", None): value = attn.norm_v(value)
            if self._cat_k is not None:
                key = torch.cat((key, self._cat_k), dim=1)
                value = torch.cat((value, self._cat_v), dim=1)
            cached_out = hidden_states if self.use_feature_injection else None

        # heads->batch
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # xFormers MEA (or SDPA fallback)
        if xops is not None:
            attn_bias = None
            if attention_mask is not None:
                bsz, q_len, _ = hidden_states.shape
                k_len = key.shape[1]
                attention_mask = attn.prepare_attention_mask(attention_mask, k_len, bsz)
                if attention_mask is not None:
                    attention_mask = attention_mask.view(bsz, attn.heads, 1, k_len)
                    attention_mask = attention_mask.expand(-1, -1, q_len, -1)
                    attn_bias = attention_mask.reshape(bsz * attn.heads, q_len, k_len)
            out = xops.memory_efficient_attention(query, key, value, attn_bias=attn_bias, p=0.0, scale=attn.scale)
        else:
            if attention_mask is not None:
                bsz, q_len, _ = hidden_states.shape
                k_len = key.shape[1]
                attention_mask = attn.prepare_attention_mask(attention_mask, k_len, bsz)
                if attention_mask is not None:
                    attention_mask = attention_mask.view(bsz, attn.heads, 1, k_len)
                    attention_mask = attention_mask.expand(-1, -1, q_len, -1)
                    attention_mask = attention_mask.reshape(bsz * attn.heads, q_len, k_len)
            out = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False, scale=attn.scale
            )

        out = attn.batch_to_head_dim(out)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)

        if not is_cross:
            k_cur = attn.to_k(residual if getattr(attn, "norm_k", None) is None else getattr(attn, "norm_k")(residual))
            v_cur = attn.to_v(residual if getattr(attn, "norm_v", None) is None else getattr(attn, "norm_v")(residual))
            self._maybe_push_self_cache(k_cur, v_cur, cached_out)

        return out
