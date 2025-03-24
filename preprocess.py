import os
import random

# Path to your train folder
train_folder = "/scratch/s52melba/dataset_yolo_sahi_256/train"

# Get all image files
image_files = [f for f in os.listdir(train_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Separate images into annotated and background (empty annotation file)
annotated_images = []
background_images = []

for img in image_files:
    txt_file = os.path.join(train_folder, img.rsplit(".", 1)[0] + ".txt")
    
    if os.path.exists(txt_file):
        with open(txt_file, "r") as f:
            content = f.read().strip()
        
        if content:  
            annotated_images.append(img)  # Image has boxes
        else:
            background_images.append(img)  # Empty annotation file (background image)

# Randomly select 20% of background images to keep
# keep_background_images = set(random.sample(background_images, int(len(background_images) * 0.15)))
keep_background_images = set(random.sample(background_images, int(len(annotated_images) * 0.9)))
print(f"Annotated images: {len(annotated_images)}")
print(f"Background images: {len(background_images)}")
print(f"Keep background images: {len(keep_background_images)}")
if (len(background_images)/len(annotated_images)) > 0.8:
    raise ValueError("Too many background images, please check the dataset.")

# Process files
for img in background_images:
    if img not in keep_background_images:
        img_path = os.path.join(train_folder, img)
        txt_path = os.path.join(train_folder, img.rsplit(".", 1)[0] + ".txt")
        
        os.remove(img_path)  # Delete the image
        os.remove(txt_path)  # Delete the empty annotation file

# print("Processing complete. Kept all annotated images and 5% of background images.")
print("Processing complete.")
