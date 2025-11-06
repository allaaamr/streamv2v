import os
import sys
import time
import argparse
from typing import Literal

import torch
from torchvision.io import read_video, write_video
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.wrapper import StreamV2VWrapper  # noqa: E402

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


torch.cuda.empty_cache()
torch.cuda.ipc_collect()
def parse_args():
    p = argparse.ArgumentParser("StreamV2V video editing (argparse version)")
    # Required-ish I/O
    p.add_argument("--input_path", type=str, default="./src.mp4", help="Path to input video")
    p.add_argument("--prompt", type=str, default="Virat kohli is giving a talk", help="Editing prompt")
    p.add_argument("--output_path", type=str, default=os.path.join(CURRENT_DIR, "outputs", "edited.mp4"),
                   help="Exact save path for the output video (directories will be created)")

    # Model and sampling
    p.add_argument("--model_id", type=str, default="Jiali/stable-diffusion-1.5")
    p.add_argument("--scale", type=float, default=1.0, help="Spatial scale factor for H,W")
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--diffusion_steps", type=int, default=4)
    p.add_argument("--noise_strength", type=float, default=0.4)

    # Performance/acceleration
    p.add_argument("--acceleration", type=str, choices=["none", "xformers", "tensorrt"], default="xformers")
    p.add_argument("--use_denoising_batch", action="store_true", default=True)
    p.add_argument("--no_use_denoising_batch", dest="use_denoising_batch", action="store_false")
    p.add_argument("--use_cached_attn", action="store_true", default=True)
    p.add_argument("--no_use_cached_attn", dest="use_cached_attn", action="store_false")
    p.add_argument("--use_feature_injection", action="store_true", default=True)
    p.add_argument("--no_use_feature_injection", dest="use_feature_injection", action="store_false")
    p.add_argument("--feature_injection_strength", type=float, default=0.8)
    p.add_argument("--feature_similarity_threshold", type=float, default=0.98)
    p.add_argument("--cache_interval", type=int, default=4)
    p.add_argument("--cache_maxframes", type=int, default=1)
    p.add_argument("--use_tome_cache", action="store_true", default=True)
    p.add_argument("--no_use_tome_cache", dest="use_tome_cache", action="store_false")
    p.add_argument("--do_add_noise", action="store_true", default=True)
    p.add_argument("--no_do_add_noise", dest="do_add_noise", action="store_false")
    p.add_argument("--enable_similar_image_filter", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=2)

    return p.parse_args()


def main(
    input_path: str,
    prompt: str,
    output_path: str,
    model_id: str,
    scale: float,
    guidance_scale: float,
    diffusion_steps: int,
    noise_strength: float,
    acceleration: Literal["none", "xformers", "tensorrt"],
    use_denoising_batch: bool,
    use_cached_attn: bool,
    use_feature_injection: bool,
    feature_injection_strength: float,
    feature_similarity_threshold: float,
    cache_interval: int,
    cache_maxframes: int,
    use_tome_cache: bool,
    do_add_noise: bool,
    enable_similar_image_filter: bool,
    seed: int,
):
    # --- ensure output directory exists ---
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    print("A")
    # --- read input video ---
    video_tensor, _, info = read_video(input_path)  # [T,H,W,C], uint8
    fps = float(info["video_fps"])
    video = video_tensor.float() / 255.0  # [0,1] float
    print("B")
    # --- compute scaled dims (keep divisible by 8 for diffusion safety) ---
    in_h, in_w = int(video.shape[1]), int(video.shape[2])
    h = max(8, int(in_h * scale))
    w = max(8, int(in_w * scale))
    h = (h // 8) * 8
    w = (w // 8) * 8
    print("C")
    # --- build timesteps schedule ---
    steps = max(1, diffusion_steps)
    init_step = int(50 * (1.0 - noise_strength))
    interval = max(1, int(50 * noise_strength) // steps)
    t_index_list = [init_step + i * interval for i in range(steps)]
    print("D")
    # --- init wrapper ---
    stream = StreamV2VWrapper(
        model_id_or_path=model_id,
        mode="img2img",
        t_index_list=t_index_list,
        frame_buffer_size=1,
        width=w,
        height=h,
        warmup=10,
        acceleration=acceleration,
        do_add_noise=do_add_noise,
        output_type="pt",
        enable_similar_image_filter=enable_similar_image_filter,
        similar_image_filter_threshold=0.98,
        use_denoising_batch=use_denoising_batch,
        use_cached_attn=use_cached_attn,
        use_feature_injection=use_feature_injection,
        feature_injection_strength=feature_injection_strength,
        feature_similarity_threshold=feature_similarity_threshold,
        cache_interval=cache_interval,
        cache_maxframes=cache_maxframes,
        use_tome_cache=use_tome_cache,
        seed=seed,
    )
    stream.prepare(prompt=prompt, num_inference_steps=50, guidance_scale=guidance_scale)
    print("E")
    # --- optional LoRAs by prompt keywords ---
    if any(word in prompt for word in ["pixelart", "pixel art", "Pixel art", "PixArFK"]):
        stream.stream.load_lora("./lora_weights/PixelArtRedmond15V-PixelArt-PIXARFK.safetensors", adapter_name="pixelart")
        stream.stream.pipe.set_adapters(["lcm", "pixelart"], adapter_weights=[1.0, 1.0])
        print("Use LORA: pixelart")
    elif any(word in prompt for word in ["lowpoly", "low poly", "Low poly"]):
        stream.stream.load_lora("./lora_weights/low_poly.safetensors", adapter_name="lowpoly")
        stream.stream.pipe.set_adapters(["lcm", "lowpoly"], adapter_weights=[1.0, 1.0])
        print("Use LORA: lowpoly")
    elif any(word in prompt for word in ["Claymation", "claymation"]):
        stream.stream.load_lora("./lora_weights/Claymation.safetensors", adapter_name="claymation")
        stream.stream.pipe.set_adapters(["lcm", "claymation"], adapter_weights=[1.0, 1.0])
        print("Use LORA: claymation")
    elif any(word in prompt for word in ["crayons", "Crayons", "crayons doodle", "Crayons doodle"]):
        stream.stream.load_lora("./lora_weights/doodle.safetensors", adapter_name="crayons")
        stream.stream.pipe.set_adapters(["lcm", "crayons"], adapter_weights=[1.0, 1.0])
        print("Use LORA: crayons")
    elif any(word in prompt for word in ["sketch", "Sketch", "pencil drawing", "Pencil drawing"]):
        stream.stream.load_lora("./lora_weights/Sketch_offcolor.safetensors", adapter_name="sketch")
        stream.stream.pipe.set_adapters(["lcm", "sketch"], adapter_weights=[1.0, 1.0])
        print("Use LORA: sketch")
    elif any(word in prompt for word in ["oil painting", "Oil painting"]):
        stream.stream.load_lora("./lora_weights/bichu-v0612.safetensors", adapter_name="oilpainting")
        stream.stream.pipe.set_adapters(["lcm", "oilpainting"], adapter_weights=[1.0, 1.0])
        print("Use LORA: oilpainting")

    # --- warmup on first frame ---
    out_video = torch.zeros(video.shape[0], h, w, 3, dtype=torch.float32)
    for _ in range(stream.batch_size):
        stream(image=video[0].permute(2, 0, 1))
    print("F")
    # --- inference loop ---
    times = []
    for i in tqdm(range(video.shape[0])):
        t0 = time.time()
        out_img = stream(video[i].permute(2, 0, 1))  # CHW [0,1]
        out_video[i] = out_img.permute(1, 2, 0)      # HWC
        times.append(time.time() - t0)

    if len(times) > 20:
        print(f"Avg/frame (skip first 20): {sum(times[20:]) / len(times[20:]):.4f}s")

    # # --- save output exactly where requested ---
    from fractions import Fraction

    out_video = out_video * 255
    fps = Fraction(float(fps)).limit_denominator()
    write_video(output_path, out_video, fps=fps, video_codec="libx264")
    # --- save output exactly where requested ---
    # import imageio.v2 as iio
    # from fractions import Fraction

    # # ensure uint8 on CPU
    # # (if you kept a full tensor, convert it once; if you stream frames, do inside the loop)
    # fps = Fraction(float(fps)).limit_denominator()
    # writer = iio.get_writer(output_path, fps=float(fps), codec='libx264', quality=8)

    # for i in range(out_video.shape[0]):
    #     frame = out_video[i].clamp(0, 255).to(torch.uint8).cpu().numpy()
    #     writer.append_data(frame)

    # writer.close()



if __name__ == "__main__":
    args = parse_args()
    main(
        input_path=args.input_path,
        prompt=args.prompt,
        output_path=args.output_path,
        model_id=args.model_id,
        scale=args.scale,
        guidance_scale=args.guidance_scale,
        diffusion_steps=args.diffusion_steps,
        noise_strength=args.noise_strength,
        acceleration=args.acceleration,
        use_denoising_batch=args.use_denoising_batch,
        use_cached_attn=args.use_cached_attn,
        use_feature_injection=args.use_feature_injection,
        feature_injection_strength=args.feature_injection_strength,
        feature_similarity_threshold=args.feature_similarity_threshold,
        cache_interval=args.cache_interval,
        cache_maxframes=args.cache_maxframes,
        use_tome_cache=args.use_tome_cache,
        do_add_noise=args.do_add_noise,
        enable_similar_image_filter=args.enable_similar_image_filter,
        seed=args.seed,
    )
