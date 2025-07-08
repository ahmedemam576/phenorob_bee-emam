import json

# Dummy "info" field
info = {
    "description": "bee_detection",
    "version": "1.0",
    "year": 2025,
    "contributor": "auto_script",
    "date_created": "2025-07-01"
}

# Dummy "licenses" field
licenses = [
    {
        "id": 1,
        "name": "Unknown",
        "url": "https://example.com/license"
    }
]

# Paths to annotation files
coco_paths = [
    "/scratch/s52melba/yolo_v5_mixed_ds_no_empty_images_coco/train/_annotations.coco.json",
    "/scratch/s52melba/yolo_v5_mixed_ds_no_empty_images_coco/valid/_annotations.coco.json",
    "/scratch/s52melba/yolo_v5_mixed_ds_no_empty_images_coco/test/_annotations.coco.json",
]

for path in coco_paths:
    with open(path, "r") as f:
        data = json.load(f)

    # Add or overwrite 'info' and 'licenses'
    data["info"] = info
    data["licenses"] = licenses

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Added 'info' and 'licenses' to: {path}")
