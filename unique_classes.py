import os

def count_classes(annotation_path):
    """
    Counts the number of unique classes in YOLO annotation files.

    Args:
        annotation_path (str): Path to the directory containing YOLO annotation files.

    Returns:
        int: Number of unique classes.
        list: List of unique class IDs.
    """
    class_ids = set()  # Use a set to store unique class IDs

    # Iterate through all annotation files in the directory
    for filename in os.listdir(annotation_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(annotation_path, filename)
            with open(file_path, "r") as file:
                for line in file:
                    # Each line in a YOLO annotation file is formatted as:
                    # class_id x_center y_center width height
                    class_id = int(line.split()[0])  # Extract the class ID
                    class_ids.add(class_id)  # Add the class ID to the set

    # Return the number of unique classes and the list of class IDs
    return len(class_ids), sorted(class_ids)

# Example usage
if __name__ == "__main__":
    annotation_path = "/home/s52melba/dataset_bee_annotated/val/labels"  # Replace with your annotation directory path
    num_classes, class_ids = count_classes(annotation_path)
    print(f"Number of unique classes: {num_classes}")
    print(f"Class IDs: {class_ids}")