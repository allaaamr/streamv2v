"""
StreamV2V Noise Level Ablation Study
Tests different noise_strength values across all videos from batch script
"""

import os
import subprocess
import json
import time
from collections import defaultdict
from pathlib import Path
from torchvision.models.optical_flow import raft_large

# Noise levels to test
NOISE_LEVELS = [ 0.2, 0.4, 0.6, 0.8, 1]

# Output directory
OUTPUT_DIR = "./ablation_outputs"

# Video definitions (from your batch script)
VIDEOS = [
    ("./source_video/soccerball.mp4", 1, "a soccer ball is sitting in the grass next to a tree, in flat 2d anime"),
    ("./source_video/dog-agility.mp4", 2, "a dog running through a field of poles, in flat 2d anime"),
    ("./source_video/gold-fish.mp4", 3, "goldfish in aquarium, in flat 2d anime"),
    ("./source_video/tennis.mp4", 4, "a man holding a tennis racket on a tennis court, in flat 2d anime"),
    ("./source_video/walking.mp4", 5, "a man taking a selfie with a woman on a street, in flat 2d anime"),
    ("./source_video/tennis-vest.mp4", 6, "a man holding a tennis racket on a tennis court, in flat 2d anime"),
    ("./source_video/flamingo.mp4", 7, "a group of flamingos standing in the water, in flat 2d anime"),
    ("./source_video/sheep.mp4", 8, "a group of sheep and a baby lamb in a field, in flat 2d anime"),
    ("./source_video/guitar-violin.mp4", 9, "two men playing violin and guitar in a room, in flat 2d anime"),
    ("./source_video/pigs.mp4", 10, "A Chinese ink painting of a group of pigs are standing on the ground"),
    ("./source_video/gold-fish.mp4", 11, "A Chinese ink painting of goldfish in aquarium"),
    ("./source_video/mallard-water.mp4", 12, "A Chinese ink painting of a duck swimming in a pond with some trees"),
    ("./source_video/guitar-violin.mp4", 13, "A Chinese ink painting of two men playing violin and guitar in a room"),
    ("./source_video/pigs.mp4", 14, "A group of cows are standing on the ground."),
    ("./source_video/dog-agility.mp4", 15, "A Corgi running through a field of poles."),
    ("./source_video/dog-agility.mp4", 16, "A Husky running through a field of poles."),
    ("./source_video/mallard-water.mp4", 17, "A flamingo swimming in a pond with some trees."),
    ("./source_video/mallard-water.mp4", 18, "A swan swimming in a pond with some trees."),
    ("./source_video/dogs-jump.mp4", 19, "A woman is standing in a field with two tigers."),
    ("./source_video/dogs-jump.mp4", 20, "A woman is standing in a field with two leopards."),
    ("./source_video/dogs-jump.mp4", 21, "A woman is standing in a field with two dogs in front of the Egyptian pyramids."),
    ("./source_video/gold-fish.mp4", 22, "Goldfish in an aquarium surrounded by colorful coral reefs."),
    ("./source_video/sheep.mp4", 23, "A group of cows and a baby calf in a field."),
    ("./source_video/breakdance-flare.mp4", 24, "An impressionist painting of a man doing a handstand on the street"),
    ("./source_video/mallard-water.mp4", 25, "An impressionist painting of a duck swimming in a pond with some trees"),
    ("./source_video/breakdance.mp4", 26, "An impressionist painting of a man doing a handstand on a brick wall"),
    ("./source_video/sheep.mp4", 27, "An impressionist painting of a group of sheep and a baby lamb in a field"),
    ("./source_video/guitar-violin.mp4", 28, "An impressionist painting of two men playing violin and guitar in a room"),
    ("./source_video/dancing.mp4", 29, "An impressionist painting of a group of people dancing in the street"),
    ("./source_video/soccerball.mp4", 30, "An oil painting of a soccer ball is sitting in the grass next to a tree"),
    ("./source_video/dog-agility.mp4", 31, "An oil painting of a dog running through a field of poles"),
    ("./source_video/gold-fish.mp4", 32, "An oil painting of goldfish in aquarium"),
    ("./source_video/salsa.mp4", 33, "An oil painting of a group of people dancing on a street"),
    ("./source_video/soccerball.mp4", 34, "A pixel art of a soccer ball is sitting in the grass next to a tree"),
    ("./source_video/pigs.mp4", 35, "A pixel art of a group of pigs are standing on the ground"),
    ("./source_video/dog-agility.mp4", 36, "A pixel art of a dog running through a field of poles"),
    ("./source_video/elephant.mp4", 37, "A pixel art of a large elephant standing in a dirt area"),
    ("./source_video/breakdance-flare.mp4", 38, "A pixel art of a man doing a handstand on the street"),
    ("./source_video/gold-fish.mp4", 39, "A pixel art of goldfish in aquarium"),
    ("./source_video/lindy-hop.mp4", 40, "A pixel art of a group of people dancing in a dance studio"),
    ("./source_video/flamingo.mp4", 41, "A pixel art of a group of flamingos standing in the water"),
    ("./source_video/breakdance.mp4", 42, "A pixel art of a man doing a handstand on a brick wall"),
    ("./source_video/sheep.mp4", 43, "A pixel art of a group of sheep and a baby lamb in a field"),
    ("./source_video/guitar-violin.mp4", 44, "A pixel art of two men playing violin and guitar in a room"),
    ("./source_video/guitar-violin.mp4", 45, "A pencil sketch of two men playing violin and guitar in a room"),
    ("./source_video/dog-agility.mp4", 46, "Ukiyo-e Art - a dog running through a field of poles"),
    ("./source_video/gold-fish.mp4", 47, "Ukiyo-e Art - goldfish in aquarium"),
    ("./source_video/tennis.mp4", 48, "Ukiyo-e Art - a man holding a tennis racket on a tennis court"),
    ("./source_video/tennis-vest.mp4", 49, "Ukiyo-e Art - a man holding a tennis racket on a tennis court"),
    ("./source_video/flamingo.mp4", 50, "Ukiyo-e Art - a group of flamingos standing in the water"),
    ("./source_video/guitar-violin.mp4", 51, "Ukiyo-e Art - two men playing violin and guitar in a room"),
    ("./source_video/dancing.mp4", 52, "Ukiyo-e Art - a group of people dancing in the street"),
    ("./source_video/salsa.mp4", 53, "Ukiyo-e Art - a group of people dancing on a street"),
    ("./source_video/soccerball.mp4", 54, "A Van Gogh style painting of a soccer ball is sitting in the grass next to a tree"),
    ("./source_video/pigs.mp4", 55, "A Van Gogh style painting of a group of pigs are standing on the ground"),
    ("./source_video/dog-agility.mp4", 56, "A Van Gogh style painting of a dog running through a field of poles"),
    ("./source_video/breakdance-flare.mp4", 57, "A Van Gogh style painting of a man doing a handstand on the street"),
    ("./source_video/tennis.mp4", 58, "A Van Gogh style painting of a man holding a tennis racket on a tennis court"),
    ("./source_video/walking.mp4", 59, "A Van Gogh style painting of a man taking a selfie with a woman on a street"),
    ("./source_video/mallard-water.mp4", 60, "A Van Gogh style painting of a duck swimming in a pond with some trees"),
    ("./source_video/lindy-hop.mp4", 61, "A Van Gogh style painting of a group of people dancing in a dance studio"),
    ("./source_video/tennis-vest.mp4", 62, "A Van Gogh style painting of a man holding a tennis racket on a tennis court"),
    ("./source_video/flamingo.mp4", 63, "A Van Gogh style painting of a group of flamingos standing in the water"),
    ("./source_video/breakdance.mp4", 64, "A Van Gogh style painting of a man doing a handstand on a brick wall"),
    ("./source_video/dogs-jump.mp4", 65, "A Van Gogh style painting of a woman is standing in a field with two dogs"),
    ("./source_video/guitar-violin.mp4", 66, "A Van Gogh style painting of two men playing violin and guitar in a room"),
    ("./source_video/salsa.mp4", 67, "A Van Gogh style painting of a group of people dancing on a street"),
]



# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_video_name(video_path):
    """Extract video name from path."""
    return Path(video_path).stem


def run_generation(input_path, prompt, output_path, noise_strength):
    """Run video generation with specific noise level."""
    cmd = [
        "python", "vid2vid/main.py",
        "--input_path", input_path,
        "--prompt", prompt,
        "--output_path", output_path,
        "--noise_strength", str(noise_strength)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("    ⏰ Timeout")
        return False
    except Exception as e:
        print(f"    ❌ Error: {e}")
        return False


def compute_warp_error(ref_video_path, edit_video_path):
    """Compute warp error using existing warp_error.py script."""
    cmd = [
        "python", "-c",
        f"""
    import sys
    sys.path.append('.')
    from vid2vid.metrics.warp_error import calculate_warp_error_video, raft_large
    import torch

    model = raft_large(pretrained=True, progress=False).to('cuda').eval()
    error = calculate_warp_error_video(model, '{ref_video_path}', '{edit_video_path}')
    print(f'{{error:.6f}}')
    """
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            return float(result.stdout.strip())
        else:
            return None
    except:
        return None


def compute_clip_score(video_path, prompt):
    """Compute CLIP score using existing clip_score.py script."""
    cmd = [
        "python", "-c",
        f"""
        import sys
        sys.path.append('.')
        from vid2vid.metrics.clip_score import compute_clip_score

        avg, max_s = compute_clip_score('{video_path}', '{prompt}', frame_interval=10)
        print(f'{{avg:.6f}},{{max_s:.6f}}')
        """
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            avg, max_s = result.stdout.strip().split(',')
            return float(avg), float(max_s)
        else:
            return None, None
    except:
        return None, None


# ============================================================================
#  ABLATION 
# ============================================================================

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Results storage
    results = defaultdict(lambda: defaultdict(dict))
    
    print("=" * 80)
    print("🔬 STREAMV2V NOISE LEVEL ABLATION STUDY")
    print("=" * 80)
    print(f"Videos: {len(VIDEOS)}")
    print(f"Noise levels: {NOISE_LEVELS}")
    print(f"Total experiments: {len(VIDEOS) * len(NOISE_LEVELS)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80)
    print()
    
    # # Load RAFT model once for warp error computation
    # print("🔧 Loading RAFT model for warp error evaluation...")
    # import torch
    
    raft_model = raft_large(pretrained=True, progress=False).to('cuda').eval()
    print("✅ Model loaded\n")
    
    # # # Load CLIP model once
    # # print("🔧 Loading CLIP model for semantic evaluation...")
    # # from transformers import CLIPProcessor, CLIPModel
    # # clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda").eval()
    # # clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    # # print("✅ Model loaded\n")
    
    total_experiments = len(VIDEOS) * len(NOISE_LEVELS)
    current_experiment = 0
    # Main loop
    current_dir = os.path.dirname(__file__)
    for video_path, c, prompt in VIDEOS:
        video_name = get_video_name(video_path)
        video_path = os.path.join(current_dir, '..', 'vid2vid','source_video', f"{video_name}.mp4")
        video_path = os.path.abspath(video_path)
        
        
        print(f"\n{'='*80}")
        print(f"📹 VIDEO: {video_name}")
        print(f"{'='*80}")
        
        for noise in NOISE_LEVELS:
            current_experiment += 1
            
            print(f"\n[{current_experiment}/{total_experiments}] 🎯 Noise Level: {noise}")
            print("-" * 60)
            
            # Define output path
            output_name = f"{video_name}_{c}_noise{noise}.mp4"
            output_path = os.path.join(OUTPUT_DIR, output_name)
            
            # Generate video
            if os.path.exists(output_path):
                print(f"  ⏭️  Video exists, skipping generation")
            else:
                print(f"  🎬 Generating...")
                try:
                    run_generation(video_path, prompt, output_path, noise)
                except Exception as e:
                        print(f"    ❌ Error: {e}")
                        print(f"  ❌ Generation failed")
                        results[noise][video_name] = {
                            'warp_error': None,
                            'text_score_avg': None,
                            'consistency_score_avg': None,
                            'status': 'failed'
                        }
                        return False

            # Evaluate
            print(f"  📊 Evaluating...")
            
            # Warp error
            print(f"    → Computing warp error...", end=" ", flush=True)
            from metrics.warp_error import calculate_warp_error_video
            warp_error = calculate_warp_error_video(raft_model, video_path, output_path)
            print(f"✓")
            
            # CLIP score
            print(f"    → Computing CLIP score...", end=" ", flush=True)
            from metrics.clip_score import compute_clip_score
            text_score, consistency_score = compute_clip_score(output_path, prompt, frame_interval=10, device="cuda")
            print(f"✓")
            
            # Store results
            video_name_res = f"{video_name}_{prompt}"
            results[noise][video_name_res] = {
                'warp_error': warp_error,
                'text_score': text_score,
                'consistency_score': consistency_score,
                'status': 'success'
            }
            
            # Print results
            print(f"\n  📈 Results:")
            print(f"    Warp Error: {warp_error:.6f}")
            print(f"    Text Score:   {text_score:.6f}")
            print(f"    Consistency Score:   {consistency_score:.6f}")
    
    # ========================================================================
    # SUMMARY STATISTICS
    # ========================================================================
    
    print("\n\n" + "=" * 80)
    print("📊 SUMMARY: AVERAGE METRICS ACROSS ALL VIDEOS")
    print("=" * 80)
    
    summary_table = []
    
    for noise in sorted(results.keys()):
        # Collect valid metrics
        warp_errors = [r['warp_error'] for r in results[noise].values() ]
                    #    if r['warp_error'] is not None]
        text_avg = [r['text_score'] for r in results[noise].values() ]
                    #  if r['clip_avg'] is not None]
        consistency_avg = [r['consistency_score'] for r in results[noise].values() ]
                    #  if r['clip_max'] is not None]
        
        # Compute averages
        avg_warp = sum(warp_errors) / len(warp_errors) if warp_errors else None
        avg_text_score = sum(text_avg) / len(text_avg) if text_avg else None
        avg_consistency_score = sum(consistency_avg) / len(consistency_avg) if consistency_avg else None
        
        summary_table.append({
            'noise': noise,
            'avg_warp_error': avg_warp,
            'avg_text_score': avg_text_score,
            'avg_consistency_score': avg_consistency_score,
            'num_success': len(warp_errors)
        })
    
    # Print summary table
    print()
    print(f"{'Noise':<10} {'Avg Warp Error':<20} {'Avg Text Score':<20} {'Avg Consistency Score':<20} {'Success':<10}")
    print("-" * 80)
    
    for row in summary_table:
        noise = row['noise']
        warp = f"{row['avg_warp_error']:.6f}" if row['avg_warp_error'] is not None else "N/A"
        clip_a = f"{row['avg_text_score']:.6f}" if row['avg_text_score'] is not None else "N/A"
        clip_m = f"{row['avg_consistency_score']:.6f}" if row['avg_consistency_score'] is not None else "N/A"
        success = f"{row['num_success']}/{len(VIDEOS)}"
        
        print(f"{noise:<10.2f} {warp:<20} {clip_a:<20} {clip_m:<20} {success:<10}")
    
    # Find best configuration
    # print("\n" + "=" * 80)
    # print("🏆 BEST CONFIGURATIONS")
    # print("=" * 80)
    
    # valid_rows = [r for r in summary_table if r['avg_warp_error'] is not None]
    # if valid_rows:
    #     best_warp = min(valid_rows, key=lambda x: x['avg_warp_error'])
    #     print(f"\n✨ Lowest Warp Error (best temporal consistency):")
    #     print(f"   Noise Level: {best_warp['noise']}")
    #     print(f"   Avg Warp Error: {best_warp['avg_warp_error']:.6f}")
    #     print(f"   Avg Text Score: {best_warp['avg_text_score']:.6f}")
    
    # valid_rows_clip = [r for r in summary_table if r['avg_text_score'] is not None]
    # if valid_rows_clip:
    #     best_clip = max(valid_rows_clip, key=lambda x: x['avg_clip_avg'])
    #     print(f"\n✨ Highest CLIP Score:")
    #     print(f"   Noise Level: {best_clip['noise']}")
    #     print(f"   Avg Warp Error: {best_clip['avg_warp_error']:.6f}")
    #     print(f"   Avg CLIP Score: {best_clip['avg_clip_avg']:.6f}")
    
    # Save results to JSON
    results_file = os.path.join(OUTPUT_DIR, "ablation_results.json")
    with open(results_file, 'w') as f:
        # Convert defaultdict to regular dict for JSON serialization
        json_results = {
            str(noise): dict(videos) 
            for noise, videos in results.items()
        }
        json.dump(json_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    print("\n" + "=" * 80)
    print("✅ ABLATION STUDY COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()