import os
import cv2
import glob
import numpy as np
from tqdm import tqdm
import re
from collections import defaultdict

def parse_tile_filename(tile_filename):
    """
    Parse tile filename to extract video info and tile position.
    Expected format: {video_name}_{second}_tile{tile_idx}.jpg
    Example: 20230708_plot1_18.28_2_tile3.jpg
    """
    # Remove extension
    base_name = os.path.splitext(tile_filename)[0]
    
    # Pattern to match: video_name_second_tileN
    # We need to be careful because video names can contain underscores
    pattern = r'(.+)_(\d+)_tile(\d+)$'
    match = re.match(pattern, base_name)
    
    if match:
        video_name = match.group(1)
        second = int(match.group(2))
        tile_idx = int(match.group(3))
        return video_name, second, tile_idx
    else:
        raise ValueError(f"Could not parse tile filename: {tile_filename}")

def get_tile_position(tile_idx, frame_width, frame_height, tile_size):
    """
    Calculate tile position (x, y) from tile index.
    This should match the tiling logic from your original script.
    """
    tiles_per_row = frame_width // tile_size
    tiles_per_col = frame_height // tile_size
    
    row = tile_idx // tiles_per_row
    col = tile_idx % tiles_per_row
    
    x = col * tile_size
    y = row * tile_size
    
    return x, y

def convert_tile_coords_to_full_image(tile_labels, tile_x, tile_y, tile_size, full_width, full_height):
    """
    Convert YOLO coordinates from tile space to full image space.
    
    Args:
        tile_labels: List of (class, x_center, y_center, width, height, conf) in tile normalized coords
        tile_x, tile_y: Top-left position of tile in full image
        tile_size: Size of the tile
        full_width, full_height: Dimensions of full image
    
    Returns:
        List of (class, x_center, y_center, width, height, conf) in full image normalized coords
    """
    full_image_labels = []
    
    for label in tile_labels:
        cls, x_center_tile, y_center_tile, width_tile, height_tile, conf = label
        
        # Convert from tile normalized coordinates to tile pixel coordinates
        x_center_pixel = x_center_tile * tile_size
        y_center_pixel = y_center_tile * tile_size
        width_pixel = width_tile * tile_size
        height_pixel = height_tile * tile_size
        
        # Convert to full image pixel coordinates
        x_center_full_pixel = tile_x + x_center_pixel
        y_center_full_pixel = tile_y + y_center_pixel
        
        # Convert to full image normalized coordinates
        x_center_full_norm = x_center_full_pixel / full_width
        y_center_full_norm = y_center_full_pixel / full_height
        width_full_norm = width_pixel / full_width
        height_full_norm = height_pixel / full_height
        
        full_image_labels.append((cls, x_center_full_norm, y_center_full_norm, width_full_norm, height_full_norm, conf))
    
    return full_image_labels

def load_yolo_labels(label_file):
    """Load YOLO format labels from file."""
    labels = []
    if os.path.exists(label_file):
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    conf = float(parts[5]) if len(parts) > 5 else 1.0
                    labels.append((cls, x_center, y_center, width, height, conf))
    return labels

def save_yolo_labels(labels, output_file):
    """Save YOLO format labels to file."""
    with open(output_file, 'w') as f:
        for cls, x_center, y_center, width, height, conf in labels:
            f.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {conf:.6f}\n")

def extract_frame_from_video(video_path, second, output_path):
    """Extract a specific frame (at given second) from video."""
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    frame_idx = second * fps
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    
    if ret:
        cv2.imwrite(output_path, frame)
        height, width = frame.shape[:2]
        cap.release()
        return width, height
    else:
        cap.release()
        return None, None

def process_dataset_split(dataset_folder, split_name, videos_root, output_folder, tile_size=256):
    """
    Process a dataset split (train/val/test) and reconstruct full image labels.
    
    Args:
        dataset_folder: Path to dataset folder containing train/val/test
        split_name: 'train', 'val', or 'test'
        videos_root: Root folder containing video subfolders
        output_folder: Output folder for reconstructed full images and labels
        tile_size: Size of tiles used during inference
    """
    split_path = os.path.join(dataset_folder, split_name)
    if not os.path.exists(split_path):
        print(f"Split {split_name} not found in {dataset_folder}")
        return
    
    # Get all tile images
    tile_images = glob.glob(os.path.join(split_path, "*.jpg"))
    
    # Group tiles by video and second
    video_second_groups = defaultdict(list)
    
    for tile_image in tile_images:
        tile_filename = os.path.basename(tile_image)
        try:
            video_name, second, tile_idx = parse_tile_filename(tile_filename)
            video_second_groups[(video_name, second)].append((tile_image, tile_idx))
        except ValueError as e:
            print(f"Warning: {e}")
            continue
    
    # Create output directories
    output_split_path = os.path.join(output_folder, split_name)
    os.makedirs(output_split_path, exist_ok=True)
    
    # Process each video-second combination
    for (video_name, second), tiles in tqdm(video_second_groups.items(), 
                                          desc=f"Processing {split_name} full images"):
        
        # Find the corresponding video file
        video_file = None
        for video_folder in os.listdir(videos_root):
            video_folder_path = os.path.join(videos_root, video_folder)
            if os.path.isdir(video_folder_path):
                potential_video = os.path.join(video_folder_path, f"{video_name}.MOV")
                if os.path.exists(potential_video):
                    video_file = potential_video
                    break
        
        if video_file is None:
            print(f"Warning: Could not find video file for {video_name}")
            continue
        
        # Extract the frame
        full_image_name = f"{video_name}_{second}.jpg"
        full_image_path = os.path.join(output_split_path, full_image_name)
        
        frame_width, frame_height = extract_frame_from_video(video_file, second, full_image_path)
        
        if frame_width is None or frame_height is None:
            print(f"Warning: Could not extract frame from {video_file} at second {second}")
            continue
        
        # Collect all labels from tiles
        all_full_image_labels = []
        
        for tile_image, tile_idx in tiles:
            # Load tile labels
            tile_label_file = os.path.splitext(tile_image)[0] + ".txt"
            tile_labels = load_yolo_labels(tile_label_file)
            
            if not tile_labels:
                continue
            
            # Get tile position
            tile_x, tile_y = get_tile_position(tile_idx, frame_width, frame_height, tile_size)
            
            # Convert tile coordinates to full image coordinates
            full_image_labels = convert_tile_coords_to_full_image(
                tile_labels, tile_x, tile_y, tile_size, frame_width, frame_height
            )
            
            all_full_image_labels.extend(full_image_labels)
        
        # Save full image labels
        if all_full_image_labels:
            full_label_path = os.path.join(output_split_path, f"{video_name}_{second}.txt")
            save_yolo_labels(all_full_image_labels, full_label_path)

def main():
    """Main function to reconstruct full image dataset from tiles."""
    
    # Configuration
    dataset_folder = "/scratch/s52melba/yolo_v5_mixed_ds_no_empty_images"  # Folder containing train/val/test
    videos_root = "/scratch/s52melba/videos_1fps"  # Root folder with video subfolders
    output_folder = "/scratch/s52melba/full_img_labelled"  # Output folder for full images
    tile_size = 256  # Same tile size used during inference
    
    # Process each split
    for split_name in ['train', 'val', 'test']:
        print(f"\n=== Processing {split_name} split ===")
        process_dataset_split(
            dataset_folder=dataset_folder,
            split_name=split_name,
            videos_root=videos_root,
            output_folder=output_folder,
            tile_size=tile_size
        )
    
    print(f"\nReconstruction complete! Full image dataset saved to: {output_folder}")

if __name__ == "__main__":
    main()