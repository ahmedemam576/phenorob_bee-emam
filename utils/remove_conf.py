import os
import re

def process_yolo_annotations(folder_path):
    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            
            # Read all lines from the file
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Process each line to remove confidence score
            new_lines = []
            for line in lines:
                # Split the line into components
                parts = line.strip().split()
                
                # Keep only the first 5 elements (class + bbox coordinates)
                if len(parts) > 5:
                    new_line = ' '.join(parts[:5]) + '\n'
                    new_lines.append(new_line)
                else:
                    # If the line doesn't have confidence, keep it as is
                    new_lines.append(line)
            
            with open(file_path, 'w') as f:
                f.writelines(new_lines)

if __name__ == '__main__':
    folder_path = '/scratch/s52melba/infer_output_all_one_folder'
    process_yolo_annotations(folder_path)
    print(f"Processed all .txt files in {folder_path}")