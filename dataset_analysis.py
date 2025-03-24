import os

# Set the path to your dataset folder
dataset_path = "/scratch/s52melba/dataset_yolo_sahi/train"

# Get all files in the dataset folder
files = os.listdir(dataset_path)

# Separate image files and annotation files
image_extensions = {".jpg", ".png", ".jpeg"}
images = [f for f in files if os.path.splitext(f)[1].lower() in image_extensions]
annotations = {os.path.splitext(f)[0]: f for f in files if f.endswith(".txt")}

# Initialize counts
total_images = len(images)
images_with_objects = 0
images_without_objects = 0
total_objects = 0

# Process each image
for img in images:
    base_name = os.path.splitext(img)[0]  # Get filename without extension
    annotation_file = annotations.get(base_name)
    
    if annotation_file:
        with open(os.path.join(dataset_path, annotation_file), "r") as f:
            lines = f.readlines()
            num_objects = len(lines)
            total_objects += num_objects
            
            if num_objects > 0:
                images_with_objects += 1
            else:
                images_without_objects += 1
    else:
        images_without_objects += 1  # No annotation file means no objects

# Print results
print(f"Total number of images: {total_images}")
print(f"Total images with objects: {images_with_objects}")
print(f"Total images without objects: {images_without_objects}")
print(f"Total number of objects: {total_objects}")
