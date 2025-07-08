import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from pycocotools.coco import COCO
from rfdetr import RFDETRBase

# ----------------------------
# Config
# ----------------------------
CLASSES = ["nothing", "honey bee", "bumble bee", "unidentified"]
NUM_CLASSES = len(CLASSES)

VAL_JSON = "/scratch/s52melba/yolo_v5_mixed_ds_no_empty_images_coco/valid/_annotations.coco.json"
VAL_IMG_DIR = "/scratch/s52melba/yolo_v5_mixed_ds_no_empty_images_coco/valid"
CHECKPOINT_PATH = "/scratch/s52melba/phenorob_bee/detr_output_v2/checkpoint_best_regular.pth"
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
OUTPUT_DIR = "output_visuals"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# Load model
# ----------------------------
model = RFDETRBase(pretrain_weights=CHECKPOINT_PATH, device="cuda")

# ----------------------------
# Helper: Draw box with label
# ----------------------------
def draw_boxes(image, boxes, labels, color=(0, 255, 0)):
    img = np.array(image.copy())
    for (x1, y1, x2, y2), label in zip(boxes, labels):
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        if label:
            cv2.putText(img, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1, cv2.LINE_AA)
    return img

# ----------------------------
# Load COCO validation set
# ----------------------------
coco = COCO(VAL_JSON)
img_ids = coco.getImgIds()

print("Generating visualizations...")
for img_id in tqdm(img_ids):
    img_info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(VAL_IMG_DIR, img_info["file_name"])
    image_pil = Image.open(img_path).convert("RGB")

    # ----------- Ground Truth -----------
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)

    gt_boxes = []
    gt_labels = []
    for ann in anns:
        x, y, w, h = ann["bbox"]
        cls_id = ann["category_id"]
        gt_boxes.append([x, y, x + w, y + h])
        gt_labels.append(CLASSES[cls_id])

    gt_img = draw_boxes(image_pil, gt_boxes, gt_labels, color=(0, 255, 0))  # Green

    # ----------- Prediction -----------
    predictions = model.predict(image_pil, threshold=CONFIDENCE_THRESHOLD)
    pred_boxes = predictions.xyxy
    pred_classes = predictions.class_id
    pred_scores = predictions.confidence

    pred_labels = [
        f"{CLASSES[c]} {s:.2f}" for c, s in zip(pred_classes, pred_scores)
    ]
    pred_img = draw_boxes(image_pil, pred_boxes, pred_labels, color=(0, 0, 255))  # Red

    # ----------- Combine & Save -----------
    combined = np.hstack((gt_img, pred_img))
    save_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(img_info['file_name'])[0]}_compare.jpg")
    cv2.imwrite(save_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

print(f"✅ All visualizations saved to: {OUTPUT_DIR}")
