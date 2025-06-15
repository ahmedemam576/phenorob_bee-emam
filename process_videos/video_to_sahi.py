import os
import cv2
import glob
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial

def tile_frame(frame, tile_size):
    """Splits a frame into 256x256 tiles and returns a list of (tile, index)."""
    tiles = []
    h, w, _ = frame.shape
    tile_idx = 0

    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tile = frame[y:y + tile_size, x:x + tile_size]
            
            # Ensure tile has the correct size (avoid partial tiles)
            if tile.shape[0] == tile_size and tile.shape[1] == tile_size:
                tiles.append((tile, tile_idx))
                tile_idx += 1
    return tiles

def process_video(video_path, output_folder, tile_size):
    """Extracts frames at 1 FPS, tiles them manually, and saves tiles with correct naming."""
    
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
        
        # Save each tile
        for tile, tile_idx in tiles:
            tile_filename = f"{video_name}_{second}_tile{tile_idx}.jpg"
            tile_path = os.path.join(slice_output_dir, tile_filename)
            cv2.imwrite(tile_path, tile)

    cap.release()

def main():
    """Main function to process all videos in a folder using multiprocessing."""
    
    # Paths
    video_folder = "/scratch/s52melba/videos_1fps/zz_temp"
    output_folder = "/scratch/s52melba/20230619_1fps_tiled"
    tile_size = 256
    num_workers = min(4, os.cpu_count())  # Adjust number of parallel workers

    os.makedirs(output_folder, exist_ok=True)  # Ensure output folder exists
    
    # Get video files
    video_files = glob.glob(os.path.join(video_folder, "*.MOV"))

    # Process videos in parallel using functools.partial to avoid pickling issues
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        process_func = partial(process_video, output_folder=output_folder, tile_size=tile_size)
        list(tqdm(executor.map(process_func, video_files), total=len(video_files), desc="Processing Videos"))

    print("Processing complete.")

if __name__ == "__main__":
    main()
