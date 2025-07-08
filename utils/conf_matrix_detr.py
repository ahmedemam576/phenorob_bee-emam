import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from rfdetr import RFDETRBase
import supervision as sv

# ----------------------------
# Config
# ----------------------------
CLASSES = ["nothing", "honey bee", "bumble bee", "unidentified"]
NUM_CLASSES = len(CLASSES)

#/scratch/s52melba/yolo_v5_mixed_ds_no_empty_images_coco/valid
VAL_JSON = "/scratch/s52melba/yolo_v5_mixed_ds_no_empty_images_coco/valid/_annotations.coco.json"
VAL_IMG_DIR = "/scratch/s52melba/yolo_v5_mixed_ds_no_empty_images_coco/valid"
CHECKPOINT_PATH = "/scratch/s52melba/phenorob_bee/detr_output_v2/checkpoint_best_regular.pth"
OUTPUT_PATH = "conf_matrix.png"
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5



# ----------------------------
# Load model
# ----------------------------
model = RFDETRBase(pretrain_weights=CHECKPOINT_PATH)



# ----------------------------
# Helper: IoU
# ----------------------------
def compute_iou(boxA, boxB):
    # boxA: (N, 4), boxB: (M, 4)
    ious = np.zeros((boxA.shape[0], boxB.shape[0]))
    for i, a in enumerate(boxA):
        xA1, yA1, xA2, yA2 = a
        areaA = (xA2 - xA1) * (yA2 - yA1)
        for j, b in enumerate(boxB):
            xB1, yB1, xB2, yB2 = b
            areaB = (xB2 - xB1) * (yB2 - yB1)

            x1 = max(xA1, xB1)
            y1 = max(yA1, yB1)
            x2 = min(xA2, xB2)
            y2 = min(yA2, yB2)

            interW = max(0, x2 - x1)
            interH = max(0, y2 - y1)
            inter = interW * interH
            union = areaA + areaB - inter
            iou = inter / union if union > 0 else 0
            ious[i, j] = iou
    return ious


# ----------------------------
# Prepare COCO
# ----------------------------
coco = COCO(VAL_JSON)
img_ids = coco.getImgIds()
gt_all = []
pred_all = []

print("Running inference and matching predictions...")
for img_id in tqdm(img_ids):
    img_info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(VAL_IMG_DIR, img_info["file_name"])
    image = Image.open(img_path).convert("RGB")

    # Ground truth
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    if len(anns) == 0:
        continue

    gt_boxes = [ann["bbox"] for ann in anns]
    gt_classes = [ann["category_id"] for ann in anns]

    # Convert xywh to xyxy
    gt_xyxy = []
    for box in gt_boxes:
        x, y, w, h = box
        gt_xyxy.append([x, y, x + w, y + h])
    gt_xyxy = np.array(gt_xyxy)

    # Predict
    pred = model.predict(image, threshold=CONFIDENCE_THRESHOLD)
    pred_xyxy = pred.xyxy
    pred_classes = pred.class_id

    if len(pred_xyxy) == 0:
        # no prediction → all GT are FN (skip for now), or count as "nothing"
        continue

    iou_matrix = compute_iou(gt_xyxy, pred_xyxy)
    gt_matched = set()
    pred_matched = set()

    for gt_idx in range(len(gt_xyxy)):
        ious = iou_matrix[gt_idx]
        pred_idx = np.argmax(ious)
        if ious[pred_idx] >= IOU_THRESHOLD and pred_idx not in pred_matched:
            gt_all.append(gt_classes[gt_idx])
            pred_all.append(pred_classes[pred_idx])
            gt_matched.add(gt_idx)
            pred_matched.add(pred_idx)

    # False positives
    for pred_idx in range(len(pred_classes)):
        if pred_idx not in pred_matched:
            gt_all.append(0)  # "nothing"
            pred_all.append(pred_classes[pred_idx])

# ----------------------------
# Confusion Matrix
# ----------------------------
cm = confusion_matrix(gt_all, pred_all, labels=list(range(NUM_CLASSES)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)

fig, ax = plt.subplots(figsize=(7, 7))
disp.plot(ax=ax, cmap="Blues", values_format="d")
plt.title("RF-DETR Confusion Matrix (with FPs)")
plt.tight_layout()
plt.savefig(OUTPUT_PATH)
plt.close()

print(f"✅ Confusion matrix saved to {OUTPUT_PATH}")