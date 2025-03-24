import json
import os
import shutil
import random

# Paths
coco_json_path = "/scratch/s52melba/coco_sahi_256/coco_sahi_coco.json"
images_dir = "/scratch/s52melba/coco_sahi_256/"
output_dir = "/scratch/s52melba/coco_sahi_256_train_val"
train_ratio = 0.80  # 80% train, 20% val

# Create output folders
train_img_dir = os.path.join(output_dir, "train")
val_img_dir = os.path.join(output_dir, "val")
os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(val_img_dir, exist_ok=True)

# Load COCO JSON
with open(coco_json_path, "r") as f:
    coco_data = json.load(f)

# Shuffle and split images
random.shuffle(coco_data["images"])
split_idx = int(len(coco_data["images"]) * train_ratio)
train_images = coco_data["images"][:split_idx]
val_images = coco_data["images"][split_idx:]

# Helper: Move images
def move_images(images, target_dir):
    for img in images:
        img_path = os.path.join(images_dir, img["file_name"])
        if os.path.exists(img_path):
            # shutil.move(img_path, os.path.join(target_dir, img["file_name"]))
            # SHUTIL COPY 
            shutil.copy(img_path, os.path.join(target_dir, img["file_name"]))
        else:
            print("Image not found:", img_path)

print("Moving images...")
print("Train images:", len(train_images))
print("First train image:", train_images[0]["file_name"])
print("Val images:", len(val_images))
# Move images
move_images(train_images, train_img_dir)
move_images(val_images, val_img_dir)

# Filter annotations
def filter_annotations(images, annotations):
    image_ids = {img["id"] for img in images}
    return [ann for ann in annotations if ann["image_id"] in image_ids]

train_annotations = filter_annotations(train_images, coco_data["annotations"])
val_annotations = filter_annotations(val_images, coco_data["annotations"])

# Save new COCO JSON files
train_json = {"images": train_images, "annotations": train_annotations, "categories": coco_data["categories"]}
val_json = {"images": val_images, "annotations": val_annotations, "categories": coco_data["categories"]}

with open(os.path.join(output_dir, "train.json"), "w") as f:
    json.dump(train_json, f, indent=4)
with open(os.path.join(output_dir, "val.json"), "w") as f:
    json.dump(val_json, f, indent=4)

print("Dataset split completed!")
