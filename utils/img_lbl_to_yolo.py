import os
import shutil

# Original dataset paths
dataset_dir = '/scratch/s52melba/yolo_dataset_v2/'
merged_dir = '/scratch/s52melba/yolo_dataset_v2_ready/'

# Original dataset paths
# dataset_dir = '/path/to/your/dataset'
train_img_dir = os.path.join(dataset_dir, 'train', 'images')
train_label_dir = os.path.join(dataset_dir, 'train', 'labels')
val_img_dir = os.path.join(dataset_dir, 'val', 'images')
val_label_dir = os.path.join(dataset_dir, 'val', 'labels')

# New merged dataset path
# merged_dir = '/path/to/your/merged_dataset'

# Create train and val output dirs in the merged folder
merged_train_img_dir = os.path.join(merged_dir, 'train')
merged_train_label_dir = os.path.join(merged_dir, 'train')
merged_val_img_dir = os.path.join(merged_dir, 'val')
merged_val_label_dir = os.path.join(merged_dir, 'val')

# Create output directories
os.makedirs(merged_train_img_dir, exist_ok=True)
os.makedirs(merged_train_label_dir, exist_ok=True)
os.makedirs(merged_val_img_dir, exist_ok=True)
os.makedirs(merged_val_label_dir, exist_ok=True)

# Helper to copy files
def copy_all_files(src_dir, dst_dir, extensions=None):
    for file in os.listdir(src_dir):
        if extensions and not file.lower().endswith(extensions):
            continue
        shutil.copy(os.path.join(src_dir, file), os.path.join(dst_dir, file))

# Copy original train files
copy_all_files(train_img_dir, merged_train_img_dir, extensions=('.jpg', '.jpeg', '.png'))
copy_all_files(train_label_dir, merged_train_label_dir, extensions=('.txt',))

# Copy original val files
copy_all_files(val_img_dir, merged_val_img_dir, extensions=('.jpg', '.jpeg', '.png'))
copy_all_files(val_label_dir, merged_val_label_dir, extensions=('.txt',))

print("Merged dataset created successfully at:", merged_dir)
