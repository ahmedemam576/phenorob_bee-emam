import os
import shutil
import random

def convert_to_yolo_format(dataset_folder, output_folder):
    # Define paths
    train_input = os.path.join(dataset_folder, "train")
    val_input = os.path.join(dataset_folder, "val")
    
    train_output_img = os.path.join(output_folder, "images", "train")
    train_output_lbl = os.path.join(output_folder, "labels", "train")
    val_output_img = os.path.join(output_folder, "images", "val")
    val_output_lbl = os.path.join(output_folder, "labels", "val")
    
    # Create YOLOv5 folder structure
    for path in [train_output_img, train_output_lbl, val_output_img, val_output_lbl]:
        os.makedirs(path, exist_ok=True)

    def process_split(input_folder, output_img, output_lbl):
        for file in os.listdir(input_folder):
            if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
                img_path = os.path.join(input_folder, file)
                ann_path = os.path.join(input_folder, os.path.splitext(file)[0] + ".txt")

                # Check if annotation exists
                if os.path.exists(ann_path):
                    with open(ann_path, "r") as f:
                        lines = f.readlines()
                    
                    if len(lines) == 0 and random.random() > 0.1:
                        # Skip images with empty annotations 50% of the time
                        continue
                    
                    # Convert annotation to YOLO format
                    yolo_annotations = []
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue  # Skip malformed lines
                        
                        class_id = int(parts[0])
                        x_min, y_min, x_max, y_max = map(float, parts[1:])
                        
                        # Convert to YOLO format (normalized x_center, y_center, width, height)
                        x_center = (x_min + x_max) / 2
                        y_center = (y_min + y_max) / 2
                        width = x_max - x_min
                        height = y_max - y_min
                        
                        yolo_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")
                    
                    # Save converted annotation
                    with open(os.path.join(output_lbl, os.path.basename(ann_path)), "w") as f:
                        f.write("\n".join(yolo_annotations))
                    
                    # Copy image to the new dataset
                    shutil.copy(img_path, os.path.join(output_img, os.path.basename(img_path)))

    # Process train and val splits
    process_split(train_input, train_output_img, train_output_lbl)
    process_split(val_input, val_output_img, val_output_lbl)

    # Create dataset.yaml
    dataset_yaml = f"""train: {os.path.abspath(train_output_img)}
val: {os.path.abspath(val_output_img)}

nc: 1  # Change this if you have more classes
names: ['object']  # Change class names accordingly
"""

    with open(os.path.join(output_folder, "dataset.yaml"), "w") as f:
        f.write(dataset_yaml)

    print("Dataset conversion completed successfully!")

# Example usage:
dataset_folder = "/scratch/s52melba/dataset_yolo_sahi"
output_folder = "/scratch/s52melba/yolo_final_dataset"
convert_to_yolo_format(dataset_folder, output_folder)
