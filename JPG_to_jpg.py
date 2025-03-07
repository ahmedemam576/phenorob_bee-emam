import os

def rename_jpg_to_lowercase(folder_path):
    # Walk through the directory and its subdirectories
    counter = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Check if the file has a .JPG extension
            if file.endswith('.JPG'):
                # Construct the full file path
                old_file_path = os.path.join(root, file)
                # Create the new file name with .jpg extension
                new_file_path = os.path.join(root, file[:-4] + '.jpg')
                
                # Rename the file
                os.rename(old_file_path, new_file_path)
                # print(f'Renamed: {old_file_path} -> {new_file_path}')
                counter += 1
    print(f"Renamed {counter} files from .JPG to .jpg")

if __name__ == "__main__":
    # Specify the folder path here
    # folder_path = input("Enter the folder path: ")
    folder_path = '/scratch/s52melba/dataset'
    
    # Call the function to rename .JPG files to .jpg
    rename_jpg_to_lowercase(folder_path)