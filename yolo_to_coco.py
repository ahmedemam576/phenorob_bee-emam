import os
import json
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def convert_yolo_to_coco(yolo_root, output_dir):
    yolo_root = Path(yolo_root)
    output_dir = Path(output_dir)

    splits = ['train', 'val', 'test']
    categories = set()

    # First pass to gather all class ids
    for split in splits:
        split_path = yolo_root / split
        for label_file in split_path.glob("*.txt"):
            with open(label_file, "r") as f:
                for line in f:
                    if line.strip() == "":
                        continue
                    class_id = int(line.strip().split()[0])
                    categories.add(class_id)

    categories = sorted(list(categories))
    category_map = {id: {"id": id, "name": str(id)} for id in categories}

    for split in splits:
        print(f"\nProcessing split: {split}")
        split_path = yolo_root / split
        out_images_dir = output_dir / split
        out_images_dir.mkdir(parents=True, exist_ok=True)

        coco = {
            "images": [],
            "annotations": [],
            "categories": list(category_map.values())
        }

        annotation_id = 0
        image_id = 0

        for image_file in tqdm(split_path.glob("*.*")):
            if not image_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                continue

            label_file = image_file.with_suffix(".txt")
            if not label_file.exists():
                continue

            try:
                with Image.open(image_file) as img:
                    width, height = img.size
            except:
                print(f"Warning: Cannot open image {image_file}")
                continue

            # Add image entry
            image_entry = {
                "id": image_id,
                "file_name": image_file.name,
                "width": width,
                "height": height
            }
            coco["images"].append(image_entry)

            # Add annotations
            with open(label_file, "r") as f:
                for line in f:
                    if line.strip() == "":
                        continue
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    class_id = int(parts[0])
                    x_center, y_center, w, h = map(float, parts[1:5])
                    # Convert YOLO format to COCO bbox format
                    x = (x_center - w / 2) * width
                    y = (y_center - h / 2) * height
                    w *= width
                    h *= height

                    coco["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": class_id,
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "iscrowd": 0
                    })
                    annotation_id += 1

            # Copy image to output folder
            shutil.copy(image_file, out_images_dir / image_file.name)
            image_id += 1

        # Save COCO JSON
        ann_path = output_dir / f"annotations_{split}.json"
        with open(ann_path, "w") as f:
            json.dump(coco, f, indent=2)

        print(f"Saved COCO JSON to {ann_path}")


if __name__ == "__main__":
    yolo_ds_path = '/scratch/s52melba/yolo_v5_mixed_ds_no_empty_images'
    output_dir = '/scratch/s52melba/yolo_v5_mixed_ds_no_empty_images_coco'
    convert_yolo_to_coco(yolo_ds_path, output_dir)
