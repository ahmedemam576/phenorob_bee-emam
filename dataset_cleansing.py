import os

def check_missing_labels_and_images(images_folder, labels_folder, remove_files=False):
    # Get list of image files and label files
    image_files = set([f.split('.')[0] for f in os.listdir(images_folder) if f.endswith('.JPG') or f.endswith('.jpg')])
    label_files = set([f.split('.')[0] for f in os.listdir(labels_folder) if f.endswith('.txt')])

    print(f"Number of images: {len(image_files)}")
    print(f"Number of labels: {len(label_files)}")
 
    images_without_labels = image_files - label_files

    # Find labels with no images
    labels_without_images = label_files - image_files

    # Print the results
    print(f"Number of images without labels: {len(images_without_labels)}")
    print(f"Number of labels without images: {len(labels_without_images)}")

    # Remove images without labels
    if images_without_labels and remove_files:
        print("Removing images without labels...")
        for image in images_without_labels:
            # Remove both .jpg and .png (if they exist)
            for ext in ['.jpg', '.png']:
                image_path = os.path.join(images_folder, image + ext)
                if os.path.exists(image_path):
                    os.remove(image_path)
                    print(f"Removed: {image_path}")

    # Remove labels without images
    if labels_without_images and remove_files:
        print("Removing labels without images...")
        for label in labels_without_images:
            label_path = os.path.join(labels_folder, label + '.txt')
            if os.path.exists(label_path):
                os.remove(label_path)
                print(f"Removed: {label_path}")


if __name__ == "__main__":
    # Path to the images and labels folders
    images_folder = "/scratch/s52melba/dataset/val/images"
    labels_folder = "/scratch/s52melba/dataset/val/labels"

    # Set `remove_files=True` to remove files without corresponding pairs
    remove_files = False  # Set to False if you only want to check without removing files

    # Check for missing labels and images
    check_missing_labels_and_images(images_folder, labels_folder, remove_files)