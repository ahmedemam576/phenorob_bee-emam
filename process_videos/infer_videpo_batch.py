import os
import cv2
import glob
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from ultralytics import YOLO
import shutil

def tile_frame(frame, tile_size):
    """Splits a frame into tiles of specified size and returns a list of (tile, index, x, y)."""
    tiles = []
    h, w, _ = frame.shape
    tile_idx = 0

    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tile = frame[y:y + tile_size, x:x + tile_size]
            
            # Ensure tile has the correct size (avoid partial tiles)
            if tile.shape[0] == tile_size and tile.shape[1] == tile_size:
                tiles.append((tile, tile_idx, x, y))
                tile_idx += 1
    return tiles

def process_video_with_batch_inference(video_path, output_folder, tile_size, model, conf_threshold, batch_size=16):
    """
    Processes a video by:
    1. Extracting frames at 1 FPS
    2. Tiling each frame
    3. Running batch inference on tiles
    4. Saving only tiles with detections
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Should be 1 FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_seconds = total_frames // fps  # Each frame corresponds to 1 second

    slice_output_dir = os.path.join(output_folder, video_name)
    os.makedirs(slice_output_dir, exist_ok=True)  # Ensure output folder exists
    os.makedirs("/tmp/tile_batch", exist_ok=True)  # Temp directory for batch processing

    for second in range(total_seconds):
        frame_idx = second * fps  # 1 FPS means frame index == second
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        
        # Tile the frame
        tiles = tile_frame(frame, tile_size)
        
        # Process tiles in batches
        for i in range(0, len(tiles), batch_size):
            batch_tiles = tiles[i:i+batch_size]
            
            # Prepare batch for inference
            batch_paths = []
            batch_info = []  # Store tile information for later use
            
            for j, (tile, tile_idx, x, y) in enumerate(batch_tiles):
                tile_filename = f"{video_name}_{second}_tile{tile_idx}.jpg"
                temp_tile_path = os.path.join("/tmp/tile_batch", f"batch_{i}_{j}_{tile_filename}")
                
                # Save tile temporarily for inference
                cv2.imwrite(temp_tile_path, tile)
                batch_paths.append(temp_tile_path)
                batch_info.append((tile_filename, tile_idx, second))
            
            # Run batch inference on GPU
            results = model(batch_paths, conf=conf_threshold, device="cuda:1", batch=len(batch_paths))
            
            # Process results
            for k, result in enumerate(results):
                tile_filename, tile_idx, sec = batch_info[k]
                tile_path = batch_paths[k]
                
                # Check if there are any detections
                has_valid_boxes = any(det.conf >= conf_threshold for det in result.boxes)
                
                if has_valid_boxes:
                    # Save the tile (only if it has detections)
                    output_tile_path = os.path.join(slice_output_dir, tile_filename)
                    # os.rename(tile_path, output_tile_path)  # Move instead of copy for efficiency
                    # use copy better
                    shutil.copy(tile_path, output_tile_path)
                    
                    # Save annotations if needed
                    ann_filename = f"{video_name}_{sec}_tile{tile_idx}.txt"
                    ann_path = os.path.join(slice_output_dir, ann_filename)
                    
                    # Create YOLO format annotation file
                    with open(ann_path, 'w') as f:
                        for box in result.boxes:
                            if box.conf >= conf_threshold:
                                # Get class, normalized coordinates (YOLO format)
                                cls = int(box.cls.item())
                                x_center, y_center, width, height = box.xywhn[0].tolist()
                                
                                # Write in YOLO format: class x_center y_center width height
                                f.write(f"{cls} {x_center} {y_center} {width} {height} {box.conf}\n")
               
                os.remove(tile_path)
            
    cap.release()
    
    # Clean up temporary directory
    for file in os.listdir("/tmp/tile_batch"):
        os.remove(os.path.join("/tmp/tile_batch", file))

def main():
    """Main function to process all videos in a folder with batch inference."""
    
    # Paths
    # Paths
    # video_folder = "/scratch/s52melba/videos_1fps/20230622_1fps"

    video_folder_list = [
        '/scratch/s52melba/videos_1fps/20230624_1fps',
        '/scratch/s52melba/videos_1fps/20230626_1fps',
        '/scratch/s52melba/videos_1fps/20230628_1fps',
        '/scratch/s52melba/videos_1fps/20230629_1fps',
        '/scratch/s52melba/videos_1fps/20230630_1fps',
        '/scratch/s52melba/videos_1fps/20230708_1fps'
    ]
    output_folder = "/scratch/s52melba/infer_output_all"
    model_path = "/scratch/s52melba/phenorob_bee/runs/detect/train2/weights/best.pt"
    tile_size = 256
    conf_threshold = 0.5
    batch_size = 32  # Adjust based on your GPU memory
    model = YOLO(model_path)
    
    # Load YOLO model once with GPU
    
    os.makedirs(output_folder, exist_ok=True)  # Ensure output folder exists
    
    for video_folder in video_folder_list:
        # Get video files
        video_files = glob.glob(os.path.join(video_folder, "*.MOV"))
        
        # Create a partial function with the model and other parameters
        process_func = partial(
            process_video_with_batch_inference, 
            output_folder=output_folder, 
            tile_size=tile_size, 
            model=model, 
            conf_threshold=conf_threshold,
            batch_size=batch_size
        )
        
        # Process videos sequentially (the model will use GPU for batch processing)
        for video_path in tqdm(video_files, desc="Processing Videos"):
            process_func(video_path)
            
    print(f"Processing complete. Results saved to: {output_folder}")
                                        
if __name__ == "__main__":
    main()

# tmux new -s infer_session
# python3 phenorob_bee/process_videos/infer_videpo_batch.py
# Ctrl + B, then D   # to exit
# tmux attach -t infer_session
