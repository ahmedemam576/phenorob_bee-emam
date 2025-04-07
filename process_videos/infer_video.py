import os
import cv2
import glob
import numpy as np
import shutil
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from ultralytics import YOLO

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

def process_video_with_inference(video_path, output_folder, tile_size, model, conf_threshold):
    """
    Processes a video by:
    1. Extracting frames at 1 FPS
    2. Tiling each frame
    3. Running inference on each tile
    4. Saving only tiles with detections
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Should be 1 FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_seconds = total_frames // fps  # Each frame corresponds to 1 second

    slice_output_dir = os.path.join(output_folder, video_name)
    os.makedirs(slice_output_dir, exist_ok=True)  # Ensure output folder exists

    for second in range(total_seconds):
        frame_idx = second * fps  # 1 FPS means frame index == second
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        
        # Tile the frame manually
        tiles = tile_frame(frame, tile_size)
        
        # Process each tile with the model
        for tile, tile_idx, x, y in tiles:
            tile_filename = f"{video_name}_{second}_tile{tile_idx}.jpg"
            temp_tile_path = os.path.join("/tmp", tile_filename)  # Save temporarily
            # Save tile temporarily for inference
            cv2.imwrite(temp_tile_path, tile)
            
            # Run inference
            results = model(temp_tile_path, conf=conf_threshold, device="cuda:1")
            
            # Check if there are any detections with confidence >= threshold
            has_valid_boxes = any(det.conf >= conf_threshold for det in results[0].boxes)

            if has_valid_boxes:
                # Save the tile (only if it has detections)
                tile_path = os.path.join(slice_output_dir, tile_filename)
                shutil.copy(temp_tile_path, tile_path)
                
                # Save annotations if needed
                ann_filename = f"{video_name}_{second}_tile{tile_idx}.txt"
                ann_path = os.path.join(slice_output_dir, ann_filename)
                
                # Create YOLO format annotation file
                with open(ann_path, 'w') as f:
                    for box in results[0].boxes:
                        if box.conf >= conf_threshold:
                            # Get class, normalized coordinates (YOLO format)
                            cls = int(box.cls.item())
                            x_center, y_center, width, height = box.xywhn[0].tolist()
                            
                            # Write in YOLO format: class x_center y_center width height
                            f.write(f"{cls} {x_center} {y_center} {width} {height} {box.conf}\n")
            
            # Clean up temporary file
            os.remove(temp_tile_path)

    cap.release()

def main():
    """Main function to process all videos in a folder with inference."""
    
    # Paths
    video_folder = "/scratch/s52melba/videos_1fps/20230619_1fps"
    output_folder = "/scratch/s52melba/infer_output_all"
    model_path = "/scratch/s52melba/phenorob_bee/runs/detect/train2/weights/best.pt"
    tile_size = 256
    conf_threshold = 0.3
    
    # Load YOLO model once (outside the parallel processing to avoid loading it multiple times)
    model = YOLO(model_path)
    
    os.makedirs(output_folder, exist_ok=True)  # Ensure output folder exists
    
    # Get video files
    video_files = glob.glob(os.path.join(video_folder, "*.MOV"))
    
    # Create a partial function with the model and other parameters
    process_func = partial(
        process_video_with_inference, 
        output_folder=output_folder, 
        tile_size=tile_size, 
        model=model, 
        conf_threshold=conf_threshold
    )
    
    # Process videos sequentially (safer with GPU models)
    for video_path in tqdm(video_files, desc="Processing Videos"):
        process_func(video_path)
        
    print(f"Processing complete. Results saved to: {output_folder}")

if __name__ == "__main__":
    main()