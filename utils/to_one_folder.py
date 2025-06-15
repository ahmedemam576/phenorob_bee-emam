import os
import shutil

# Source directory containing subfolders
src_root = '/scratch/s52melba/infer_output_all'

# Destination directory where all files will be copied
dst_folder = '/scratch/s52melba/infer_output_all_one_folder'
os.makedirs(dst_folder, exist_ok=True)

# Walk through all subdirectories
for root, dirs, files in os.walk(src_root):
    for file in files:
        if file.endswith(('.jpg', '.txt')):
            src_file = os.path.join(root, file)
            dst_file = os.path.join(dst_folder, file)

            # If file with same name exists, rename to avoid conflict
            if os.path.exists(dst_file):
                base, ext = os.path.splitext(file)
                counter = 1
                while os.path.exists(dst_file):
                    dst_file = os.path.join(dst_folder, f"{base}_{counter}{ext}")
                    counter += 1

            shutil.copy2(src_file, dst_file)

print("Copying complete.")
