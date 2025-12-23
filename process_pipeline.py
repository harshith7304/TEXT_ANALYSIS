import os
import argparse
import json
import shutil
from pathlib import Path

# Import stages
from text_detector_craft import CraftTextDetector
from gemini_text_analysis import analyze_all_crops
from pipeline_utils import normalize_analysis_data
from profiler import Profiler

# Import helper from test_craft to manage run directories
from test_craft import get_next_run_dir, process_single_image, process_batch

def merge_results(craft_result, gemini_result):
    """
    Merge CRAFT detection results with Gemini analysis.
    MATCHING LOGIC: Uses the 'id' from CRAFT and 'crop' filename from Gemini.
    Gemini crop filenames are usually 'region_{id}.png'.
    """
    merged_regions = []
    
    # Create a lookup for Gemini results by region ID
    gemini_map = {}
    for item in gemini_result:
        filename = item['crop'] # e.g. region_1.png
        # Extract ID
        try:
            # Assuming format 'region_X.png'
            region_id = int(filename.split('_')[1].split('.')[0])
            gemini_map[region_id] = item['analysis']
        except (IndexError, ValueError):
            print(f"Warning: Could not parse ID from filename {filename}")
            continue

    # Merge
    for region in craft_result['text_regions']:
        rid = region['id']
        merged_region = region.copy()
        
        if rid in gemini_map:
            merged_region['analysis'] = gemini_map[rid]
        else:
            merged_region['analysis'] = None
            
        merged_regions.append(merged_region)
        
    final_output = craft_result.copy()
    final_output['text_regions'] = merged_regions
    return final_output

def main():
    parser = argparse.ArgumentParser(description="Unified Text Segmentation & Analysis Pipeline")
    parser.add_argument("--image", type=str, help="Path to single image")
    parser.add_argument("--folder", type=str, help="Path to folder of images")
    parser.add_argument("--output_dir", type=str, default="pipeline_outputs", help="Directory for pipeline results")
    parser.add_argument("--cuda", action="store_true", help="Use GPU for CRAFT")
    
    args = parser.parse_args()
    
    if not args.image and not args.folder:
        parser.error("Please provide --image or --folder")
        
    profiler = Profiler()
    profiler.start("Pipeline Initialization")
    
    # Setup Output Dir
    run_dir = get_next_run_dir(args.output_dir)
    print(f"Pipeline Run Directory: {run_dir}")
    
    # Initialize Detector
    detector = CraftTextDetector(
        text_threshold=0.7,
        link_threshold=0.4,
        cuda=args.cuda,
        merge_lines=True # Use line merging for better text blocks
    )
    profiler.stop("Pipeline Initialization")

    # Determine images to process
    images_to_process = []
    if args.image:
        images_to_process.append(Path(args.image))
    elif args.folder:
        folder = Path(args.folder)
        extensions = ['.png', '.jpg', '.jpeg', '.webp']
        for ext in extensions:
            images_to_process.extend(folder.glob(f"*{ext}"))
            images_to_process.extend(folder.glob(f"*{ext.upper()}"))
            
    print(f"Found {len(images_to_process)} images to process.")

    # --- PROCESS LOOP ---
    pipeline_results = []
    
    profiler.start(f"Total Processing ({len(images_to_process)} images)")
    
    for img_path in images_to_process:
        print(f"\n--- Processing {img_path.name} ---")
        img_profiler = Profiler() # Sub-profiler for per-image stats
        
        # 1. SEGMENTATION (CRAFT)
        img_profiler.start("Stage 1: Segmentation (CRAFT)")
        # We use process_single_image from test_craft because it handles saving crops/json structure we need
        # It saves to run_dir/image_stem/
        craft_result = process_single_image(str(img_path), detector, results_base=run_dir)
        img_profiler.stop("Stage 1: Segmentation (CRAFT)")
        
        # Identify crop directory
        # test_craft saves to: base / image_stem / crops
        crop_dir = run_dir / img_path.stem / "crops"
        
        if not crop_dir.exists() or not list(crop_dir.glob("*")):
            print(f"No text detected or crops not saved for {img_path.name}")
            continue

        # 2. ANALYSIS (GEMINI)
        img_profiler.start("Stage 2: Analysis (Gemini)")
        gemini_result = analyze_all_crops(str(crop_dir))
        img_profiler.stop("Stage 2: Analysis (Gemini)")
        
        # 3. NORMALIZATION (FONTS)
        img_profiler.start("Stage 3: Normalization")
        normalized_gemini_result = normalize_analysis_data(gemini_result)
        img_profiler.stop("Stage 3: Normalization")
        
        # 4. MERGE & SAVE
        merged_data = merge_results(craft_result, normalized_gemini_result)
        
        # Save merged JSON
        final_json_path = run_dir / img_path.stem / f"{img_path.stem}_full_analysis.json"
        with open(final_json_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, indent=2)
            
        print(f"Saved full analysis to: {final_json_path}")
        
        pipeline_results.append({
            "image": str(img_path),
            "result_file": str(final_json_path),
            "timings": img_profiler.events
        })
        
        # Print sub-report for this image
        img_profiler.report()

    profiler.stop(f"Total Processing ({len(images_to_process)} images)")
    
    # Final Report
    profiler.report()
    
    # Save a summary log
    summary_path = run_dir / "pipeline_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            "total_images": len(images_to_process),
            "global_timings": profiler.events,
            "details": pipeline_results
        }, f, indent=2)
    print(f"Pipeline summary saved to {summary_path}")

if __name__ == "__main__":
    main()
