from PIL import Image
import json 
import os 
import torch 
import cv2

from transformers import AutoProcessor, CLIPVisionModel
from transformers import CLIPProcessor, CLIPModel


device = "cuda"

# Function to load JSON data from a file
def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

# Function to create a dictionary with vid_name as keys and prompt as values
def create_vid_prompt_dict(json_data):
    vid_prompt_dict = {}
    for item in json_data:
        vid_name = item.get('vid_name', '')
        prompt = item.get('prompt', '')
        vid_prompt_dict[vid_name] = prompt
    return vid_prompt_dict

import torch
import cv2
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

def compute_clip_score(video_path, prompt, frame_interval=1, device="cuda"):
    """
    Compute CLIP score for a video against a text prompt.
    
    Args:
        video_path: Path to the video file
        prompt: Text prompt to compare against
        frame_interval: Process every Nth frame (default: 1 for all frames)
        device: Device to run model on (default: "cuda")
    
    Returns:
        tuple: (text_score, consistency_score)
            - text_score: Average cosine similarity between prompt and video frames
            - consistency_score: Average cosine similarity between consecutive frames
    """
    # Load model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model = model.to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    
    video_embs = []
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")
    
    frame_count = 0
    
    # Process video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Only process every Nth frame based on frame_interval
            if frame_count % frame_interval == 0:
                # Convert the BGR frame captured by cv2 to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Convert the numpy array frame to PIL Image
                image = Image.fromarray(frame_rgb)
                
                # Process with CLIP
                with torch.no_grad():
                    inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    outputs = model(**inputs)
                    image_embeds = outputs.image_embeds
                    text_embeds = outputs.text_embeds
                
                video_embs.append(image_embeds)
            
            frame_count += 1
        else:
            # Break the loop if no frames are returned (end of video)
            break
    
    # Release the video capture object
    cap.release()
    
    if len(video_embs) == 0:
        raise ValueError(f"No frames were processed from video: {video_path}")
    
    video_embs = torch.cat(video_embs, dim=0)  # (T, D)
    
    # Compute text-to-video score
    text_score = cos(text_embeds, video_embs)  # (1, T)
    text_score = text_score.mean().cpu().item()
    
    # Compute temporal consistency score (consecutive frames)
    if len(video_embs) > 1:
        emb1 = video_embs[:-1]  # (N, D)
        emb2 = video_embs[1:]   # (N, D)
        consistency_score = cos(emb1, emb2)  # (N,)
        consistency_score = consistency_score.mean().cpu().item()
    else:
        consistency_score = 1.0  # Single frame, perfect consistency
    
    return text_score, consistency_score

if __name__ == "__main__":

    file_path = 'user_study_upload/eval.json'  # Path to your JSON file
    with open(file_path, 'r') as file:
        json_data = json.load(file)
    video_maps = create_vid_prompt_dict(json_data)

    method_name = 'StreamV2V'
    edit_video_dir = f"user_study_upload/{method_name}"
    video_names = list(video_maps.keys())

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model = model.to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    consistency_score = []
    prompt_score = []


    out_json = {}

    for v in video_names:
        try:
            out_json[v] = {}
            prompt = video_maps[v]
            video_path = os.path.join(edit_video_dir, f"{v}.mp4")
            video_embs = []

            # Open the video file
            cap = cv2.VideoCapture(video_path)

            # Check if video opened successfully
            if not cap.isOpened():
                print("Error opening video file")
            # Process video frames
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    # Convert the BGR frame captured by cv2 to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Convert the numpy array frame to PIL Image
                    image = Image.fromarray(frame_rgb)
                    
                    # Your existing processing code
                    with torch.no_grad():
                        inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True)
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        outputs = model(**inputs)

                        image_embeds = outputs.image_embeds
                        text_embeds = outputs.text_embeds
                    video_embs.append(image_embeds)
                else:
                    # Break the loop if no frames are returned (end of video)
                    break

            # Release the video capture object
            cap.release()
            video_embs = torch.cat(video_embs, dim=0)   # (T, 768)
            # 
            text_score = cos(text_embeds, video_embs)    # (1, T)
            text_score = text_score.mean().cpu().item()
            prompt_score.append(text_score)

            # two continue frames cos similarity
            emb1 = video_embs[:-1]  # (N, 768)
            emb2 = video_embs[1:]   # (N, 768)
            score = cos(emb1, emb2) # (N,)
            score = score.mean().cpu().item()

            consistency_score.append(score)
            out_json[v][prompt] = score
            print(v, prompt, score)
        except:
            print(f'{v} does not exist!')

    print("Number of videos ", len(prompt_score))
    print("Avg consistency score ", sum(consistency_score) / len(consistency_score))
    # print("Avg prompt score ", sum(prompt_score) / len(prompt_score))

    json.dump(out_json, open(f"{method_name}.clipscore", "w"), sort_keys=True, indent=4)