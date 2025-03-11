import os
import cv2
import random
import matplotlib.pyplot as plt

# Path to dataset
train_folder = "/scratch/s52melba/dataset_yolo_sahi/train"
output_folder = "/scratch/s52melba/out_vis"

# Class names (Modify if you have class names, otherwise indices will be shown)
class_names = {0: "Class 1", 1: "Class 2", 2: "Class 3"}  # Example

# Get all image files
image_files = [f for f in os.listdir(train_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

def visualize_image(image_path, txt_path):
    """Load an image, draw bounding boxes from YOLO annotation, and display it."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib
    
    h, w, _ = image.shape  # Get image dimensions

    with open(txt_path, "r") as f:
        lines = f.readlines()

    if not lines:
        return  # Skip empty annotations

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue  # Skip invalid annotations

        class_id, x_center, y_center, bbox_width, bbox_height = map(float, parts)

        # Convert YOLO format (normalized) to pixel values
        x1 = int((x_center - bbox_width / 2) * w)
        y1 = int((y_center - bbox_height / 2) * h)
        x2 = int((x_center + bbox_width / 2) * w)
        y2 = int((y_center + bbox_height / 2) * h)

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue box

        # Label (if class names exist)
        label = class_names.get(int(class_id), f"Class {int(class_id)}")
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Show image with bounding boxes
    return image
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.axis("off")
    plt.show()

# Process images
counter =0
for img in image_files:
    txt_file = os.path.join(train_folder, img.rsplit(".", 1)[0] + ".txt")
    
    if os.path.exists(txt_file):
        image = visualize_image(os.path.join(train_folder, img), txt_file)
        if image is not None:
            output_path = os.path.join(output_folder, img)
            cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            print(f"Processed: {img}")
            counter += 1
            if counter == 10:
                break
