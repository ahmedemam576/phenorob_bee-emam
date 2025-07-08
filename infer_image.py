import os
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import shutil
import glob

def tile_frame(frame, tile_size):
    """Splits a frame into tiles of specified size and returns a list of (tile, tile_idx, x, y)."""
    tiles = []
    h, w, _ = frame.shape
    tile_idx = 0
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tile = frame[y:y + tile_size, x:x + tile_size]
            if tile.shape[0] == tile_size and tile.shape[1] == tile_size:
                tiles.append((tile, tile_idx, x, y))
                tile_idx += 1
    return tiles

def process_images_with_batch_inference(image_paths, output_folder, tile_size, model, conf_threshold, batch_size=16):
    os.makedirs(output_folder, exist_ok=True)
    tile_output_folder = os.path.join(output_folder, "tiles")
    fullimg_output_folder = os.path.join(output_folder, "full_images")
    os.makedirs(tile_output_folder, exist_ok=True)
    os.makedirs(fullimg_output_folder, exist_ok=True)
    os.makedirs("/tmp/tile_batch", exist_ok=True)

    for img_path in tqdm(image_paths, desc="Processing Images"):
        image_name = os.path.splitext(os.path.basename(img_path))[0]
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Failed to read {img_path}")
            continue
        
        tiles = tile_frame(frame, tile_size)
        full_img_boxes = []  # to collect all detections in full-image coordinates

        for i in range(0, len(tiles), batch_size):
            batch_tiles = tiles[i:i+batch_size]
            batch_paths = []
            batch_info = []

            for j, (tile, tile_idx, x_offset, y_offset) in enumerate(batch_tiles):
                temp_tile_path = os.path.join("/tmp/tile_batch", f"{image_name}_tile{i}_{j}.jpg")
                cv2.imwrite(temp_tile_path, tile)
                batch_paths.append(temp_tile_path)
                batch_info.append((tile, tile_idx, x_offset, y_offset, temp_tile_path))

            results = model(batch_paths, conf=conf_threshold, device="cuda:0", batch=len(batch_paths))

            for k, result in enumerate(results):
                tile, tile_idx, x_off, y_off, tile_path = batch_info[k]

                if any(box.conf >= conf_threshold for box in result.boxes):
                    # Save tile
                    tile_filename = f"{image_name}_tile{tile_idx}.jpg"
                    shutil.copy(tile_path, os.path.join(tile_output_folder, tile_filename))

                    # Save tile annotations
                    ann_path = os.path.join(tile_output_folder, tile_filename.replace(".jpg", ".txt"))
                    with open(ann_path, 'w') as f:
                        for box in result.boxes:
                            if box.conf >= conf_threshold:
                                cls = int(box.cls.item())
                                x_center, y_center, w, h = box.xywhn[0].tolist()

                                f.write(f"{cls} {x_center} {y_center} {w} {h}\n")

                                # Convert back to full image coordinates for drawing
                                abs_x_center = x_center * tile.shape[1] + x_off
                                abs_y_center = y_center * tile.shape[0] + y_off
                                abs_w = w * tile.shape[1]
                                abs_h = h * tile.shape[0]

                                x1 = int(abs_x_center - abs_w / 2)
                                y1 = int(abs_y_center - abs_h / 2)
                                x2 = int(abs_x_center + abs_w / 2)
                                y2 = int(abs_y_center + abs_h / 2)
                                full_img_boxes.append((x1, y1, x2, y2, cls, box.conf))

                os.remove(tile_path)

        # Save full image with boxes if any tile had detections
        if full_img_boxes:
            for (x1, y1, x2, y2, cls, conf) in full_img_boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{cls}"
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            out_fullimg_path = os.path.join(fullimg_output_folder, f"{image_name}.jpg")
            cv2.imwrite(out_fullimg_path, frame)

def main():
    # Example usage
    image_paths = glob.glob("/scratch/s52melba/phenorob_bee/infer_temp/*.jpg")  # Update this path
    output_folder = "/scratch/s52melba/phenorob_bee/infer_temp_output"                 # Update this path
    model_path = "/scratch/s52melba/bee-detection/yolov12l-v2-run13/weights/best.pt"                   # Update this path
    tile_size = 256
    conf_threshold = 0.5
    batch_size = 2

    model = YOLO(model_path)
    process_images_with_batch_inference(
        image_paths=image_paths,
        output_folder=output_folder,
        tile_size=tile_size,
        model=model,
        conf_threshold=conf_threshold,
        batch_size=batch_size
    )

if __name__ == "__main__":
    main()
