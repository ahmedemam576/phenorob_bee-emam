import os

def count_jpeg_files(folder_path):
    jpeg_extensions = ['.jpg', '.jpeg', '.jpe','png']
    jpeg_count = 0

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in jpeg_extensions):
                jpeg_count += 1

    return jpeg_count

if __name__ == "__main__":
    # folder_path = input("Enter the folder path: ")
    folder_path = '/scratch/s52melba/dataset_yolo_sahi'
    # folder_path = '/scratch/s52melba/coco_sahi_train_val'
    if os.path.isdir(folder_path):
        count = count_jpeg_files(folder_path)
        print(f"Number of JPEG files in '{folder_path}': {count}")
    else:
        print("Invalid folder path.")