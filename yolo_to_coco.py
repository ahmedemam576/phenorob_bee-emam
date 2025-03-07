import fiftyone as fo

# Path to your YOLO dataset
yolo_dataset_dir = "/scratch/s52melba/dataset"

# Load the YOLO dataset
dataset = fo.Dataset.from_dir(
    dataset_dir=yolo_dataset_dir,
    dataset_type=fo.types.YOLOv5Dataset,
    label_field="ground_truth",
)

# Path to save the COCO dataset
coco_dataset_dir = "/scratch/s52melba/coco_dataset"

# Export the dataset to COCO format
dataset.export(
    export_dir=coco_dataset_dir,
    dataset_type=fo.types.COCODetectionDataset,
    label_field="ground_truth",
)

print(f"COCO dataset saved to {coco_dataset_dir}")