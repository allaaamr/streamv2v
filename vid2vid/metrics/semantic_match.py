import argparse
import torch
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


def extract_frames(video_path, frame_interval=10):
    """
    Extract frames from the video every `frame_interval` frames.
    Returns a list of PIL images.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        frame_count += 1

    cap.release()
    return frames


def compute_clip_score(video_path, prompt, frame_interval=10, device=None):
    """
    Compute semantic match between a video and a text prompt using Hugging Face CLIP.
    Returns average and max similarity across sampled frames.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load CLIP model and processor
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()

    # Extract frames
    frames = extract_frames(video_path, frame_interval)
    print(f"✅ Extracted {len(frames)} frames from {video_path}")

    similarities = []
    for frame in tqdm(frames, desc="Computing CLIP similarity"):
        inputs = processor(text=[prompt], images=frame, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # logits_per_image is the similarity between image and text
            sim = outputs.logits_per_image.item()
            similarities.append(sim)

    avg_score = np.mean(similarities)
    max_score = np.max(similarities)

    print("\n🧾 Semantic Match Scores:")
    print(f"→ Average similarity: {avg_score:.4f}")
    print(f"→ Max similarity:     {max_score:.4f}")
    return avg_score, max_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute semantic match score between video and text prompt using Hugging Face CLIP")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video file")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt to compare")
    parser.add_argument("--interval", type=int, default=10, help="Frame sampling interval (default: 10)")
    args = parser.parse_args()

    compute_clip_score(args.video_path, args.prompt, args.interval)
