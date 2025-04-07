import os
import cv2
import shutil
from ultralytics import YOLO
from tqdm import tqdm

# Define paths
input_folder = "/scratch/s52melba/20230619_1fps_tiled/20230619_plot2_12.33"
output_folder = "/scratch/s52melba/infer_output"
model_path = "/scratch/s52melba/phenorob_bee/runs/detect/train2/weights/best.pt"  # Change to your preferred model (yolov8s.pt, yolov8m.pt, etc.)
conf_threshold = 0.5

# Load YOLO model
model = YOLO(model_path)

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

# Process images
for img_name in tqdm(os.listdir(input_folder)):
    img_path = os.path.join(input_folder, img_name)
    
    # Skip non-JPG images
    if not img_name.lower().endswith(".jpg"):
        continue

    # Run inference
    results = model(img_path, conf=conf_threshold, device="cuda:0")
    
    # Check if there are any detections with confidence >= threshold
    has_valid_boxes = any(det.conf >= conf_threshold for det in results[0].boxes)

    if has_valid_boxes:
        shutil.copy(img_path, os.path.join(output_folder, img_name))

print("Filtering complete. Check:", output_folder)
