import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd
import torch
from typing import List, Tuple, Dict
import concurrent.futures
from threading import Lock

from rfdetr import RFDETRBase
import supervision as sv

# ----------------------------
# Config
# ----------------------------
CLASSES = ["nothing", "honey bee", "bumble bee", "unidentified"]
NUM_CLASSES = len(CLASSES)

# Paths
VAL_JSON = "/home/s52melba/dataset_rtdetr_format/test/_annotations.coco.json"
VAL_IMG_DIR = "/home/s52melba/dataset_rtdetr_format/test"
CHECKPOINT_PATH = "/scratch/s52melba/detr_train_2/checkpoint_best_total.pth"
OUTPUT_FOLDER = "/scratch/s52melba/analysis_output_test"  # Output folder for all plots and results
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
BATCH_SIZE = 16  # Number of images to process concurrently (not true batching)
NUM_WORKERS = 4  # Number of worker threads for parallel processing

# Create output directory
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Publication-ready matplotlib settings
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'text.usetex': False,  # Set to True if you have LaTeX installed
    'axes.linewidth': 1.2,
    'grid.alpha': 0.3
})

# Enhanced colorblind-friendly palettes
# Using scientifically-validated colorblind-safe palettes

# Option 1: Paul Tol's colorblind-safe palette (muted)
PAUL_TOL_MUTED = ['#332288', '#88CCEE', '#44AA99', '#117733', '#999933', '#DDCC77', 
                  '#CC6677', '#882255', '#AA4499']

# Option 2: Wong's colorblind-safe palette
WONG_PALETTE = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442', 
                '#56B4E9', '#E69F00', '#000000']

# Option 3: Okabe-Ito colorblind-safe palette (most recommended for scientific publications)
OKABE_ITO = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', 
             '#D55E00', '#CC79A7', '#999999']

# Using Okabe-Ito as default (change this to WONG_PALETTE or PAUL_TOL_MUTED if preferred)
COLORBLIND_COLORS = OKABE_ITO

# Color palette for confusion matrices (sequential colormap)
# Using 'cividis' which is colorblind-friendly, or 'viridis'
CONFUSION_MATRIX_CMAP = 'cividis'  # Alternative: 'viridis', 'plasma', 'magma'

# ----------------------------
# Load model with proper RF-DETR initialization
# ----------------------------
print("Loading RF-DETR model...")
print(f"Loading checkpoint from: {CHECKPOINT_PATH}")

# Initialize RF-DETR model with custom checkpoint
model = RFDETRBase(pretrain_weights=CHECKPOINT_PATH)

# Optimize model for faster inference (RF-DETR specific optimization)
print("Optimizing model for inference...")
try:
    model.optimize_for_inference()
    print("✅ Model optimization successful!")
except Exception as e:
    print(f"⚠️  Model optimization failed: {e}")
    print("Proceeding without optimization...")

print("Model loaded successfully!")

# ----------------------------
# Helper: IoU
# ----------------------------
def compute_iou(boxA, boxB):
    """Compute IoU between two sets of boxes."""
    if boxA.size == 0 or boxB.size == 0:
        return np.zeros((boxA.shape[0], boxB.shape[0]))
    
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
# Parallel Processing Functions for RF-DETR
# ----------------------------
def process_single_image(args):
    """Process a single image with RF-DETR model."""
    img_id, img_path, gt_data = args
    
    try:
        # Load image
        image = Image.open(img_path).convert("RGB")
        
        # RF-DETR inference
        detections = model.predict(image, threshold=CONFIDENCE_THRESHOLD)
        
        # Extract prediction data
        pred_xyxy = detections.xyxy if len(detections.xyxy) > 0 else np.array([]).reshape(0, 4)
        pred_classes = detections.class_id if hasattr(detections, 'class_id') and len(detections.class_id) > 0 else np.array([])
        pred_confidences = detections.confidence if hasattr(detections, 'confidence') and len(detections.confidence) > 0 else np.array([])
        
        return {
            'img_id': img_id,
            'success': True,
            'gt_data': gt_data,
            'pred_xyxy': pred_xyxy,
            'pred_classes': pred_classes,
            'pred_confidences': pred_confidences,
            'error': None
        }
        
    except Exception as e:
        return {
            'img_id': img_id,
            'success': False,
            'gt_data': gt_data,
            'pred_xyxy': np.array([]).reshape(0, 4),
            'pred_classes': np.array([]),
            'pred_confidences': np.array([]),
            'error': str(e)
        }

def process_predictions_results(results, detection_stats, per_class_stats, gt_all, pred_all, confidence_scores):
    """Process the results from parallel inference."""
    
    for result in results:
        gt_xyxy, gt_classes = result['gt_data']
        
        if not result['success']:
            print(f"Error processing image {result['img_id']}: {result['error']}")
            detection_stats['num_pred'].append(0)
            if len(gt_xyxy) > 0:
                for gt_class in gt_classes:
                    per_class_stats[CLASSES[gt_class]]['fn'] += 1
            continue
        
        pred_xyxy = result['pred_xyxy']
        pred_classes = result['pred_classes']
        pred_confidences = result['pred_confidences']
        
        # Store detection statistics
        detection_stats['num_pred'].append(len(pred_xyxy))
        
        if len(gt_xyxy) == 0 and len(pred_xyxy) == 0:
            continue
        elif len(pred_xyxy) == 0:
            # All GT are FN
            for gt_class in gt_classes:
                per_class_stats[CLASSES[gt_class]]['fn'] += 1
            continue
        elif len(gt_xyxy) == 0:
            # All predictions are FP
            for j, pred_class in enumerate(pred_classes):
                gt_all.append(0)  # "nothing"
                pred_all.append(pred_class)
                confidence_scores.append(pred_confidences[j] if len(pred_confidences) > j else 0.5)
                per_class_stats[CLASSES[pred_class]]['fp'] += 1
            continue

        # Matching logic
        iou_matrix = compute_iou(gt_xyxy, pred_xyxy)
        gt_matched = set()
        pred_matched = set()

        # Match predictions to ground truth
        for gt_idx in range(len(gt_xyxy)):
            if len(pred_xyxy) == 0:
                continue
            ious = iou_matrix[gt_idx]
            pred_idx = np.argmax(ious)
            if ious[pred_idx] >= IOU_THRESHOLD and pred_idx not in pred_matched:
                gt_all.append(gt_classes[gt_idx])
                pred_all.append(pred_classes[pred_idx])
                confidence_scores.append(pred_confidences[pred_idx] if len(pred_confidences) > pred_idx else 0.5)
                gt_matched.add(gt_idx)
                pred_matched.add(pred_idx)
                
                # Update stats
                if gt_classes[gt_idx] == pred_classes[pred_idx]:
                    per_class_stats[CLASSES[gt_classes[gt_idx]]]['tp'] += 1
                else:
                    per_class_stats[CLASSES[gt_classes[gt_idx]]]['fn'] += 1
                    per_class_stats[CLASSES[pred_classes[pred_idx]]]['fp'] += 1

        # Unmatched ground truth (False negatives)
        for gt_idx in range(len(gt_classes)):
            if gt_idx not in gt_matched:
                per_class_stats[CLASSES[gt_classes[gt_idx]]]['fn'] += 1

        # Unmatched predictions (False positives)
        for pred_idx in range(len(pred_classes)):
            if pred_idx not in pred_matched:
                gt_all.append(0)  # "nothing"
                pred_all.append(pred_classes[pred_idx])
                confidence_scores.append(pred_confidences[pred_idx] if len(pred_confidences) > pred_idx else 0.5)
                per_class_stats[CLASSES[pred_classes[pred_idx]]]['fp'] += 1

# ----------------------------
# Data Collection with Parallel Processing
# ----------------------------
coco = COCO(VAL_JSON)
img_ids = coco.getImgIds()
gt_all = []
pred_all = []
confidence_scores = []
detection_stats = defaultdict(list)
per_class_stats = {cls: {'tp': 0, 'fp': 0, 'fn': 0} for cls in CLASSES}

print(f"Running parallel inference (batch size: {BATCH_SIZE}, workers: {NUM_WORKERS}) and collecting statistics...")
print(f"Total images to process: {len(img_ids)}")

# Prepare all tasks
all_tasks = []
for img_id in img_ids:
    img_info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(VAL_IMG_DIR, img_info["file_name"])
    
    # Ground truth
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    
    gt_boxes = []
    gt_classes = []
    
    for ann in anns:
        gt_boxes.append(ann["bbox"])
        gt_classes.append(ann["category_id"])

    if len(gt_boxes) == 0:
        gt_xyxy = np.array([]).reshape(0, 4)
    else:
        # Convert xywh to xyxy
        gt_xyxy = []
        for box in gt_boxes:
            x, y, w, h = box
            gt_xyxy.append([x, y, x + w, y + h])
        gt_xyxy = np.array(gt_xyxy)
    
    # Store GT statistics
    detection_stats['num_gt'].append(len(gt_xyxy))
    
    all_tasks.append((img_id, img_path, (gt_xyxy, gt_classes)))

# Process images in parallel batches
print("Processing images with parallel inference...")
results_queue = []

# Process in batches to manage memory
for batch_start in tqdm(range(0, len(all_tasks), BATCH_SIZE), desc="Processing batches"):
    batch_end = min(batch_start + BATCH_SIZE, len(all_tasks))
    batch_tasks = all_tasks[batch_start:batch_end]
    
    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(NUM_WORKERS, len(batch_tasks))) as executor:
        batch_results = list(executor.map(process_single_image, batch_tasks))
    
    # Process results immediately to manage memory
    process_predictions_results(batch_results, detection_stats, per_class_stats, 
                              gt_all, pred_all, confidence_scores)

# ----------------------------
# 1. Confusion Matrices (Raw and Normalized)
# ----------------------------
print("Creating publication-ready confusion matrices...")

# Compute confusion matrices
cm = confusion_matrix(gt_all, pred_all, labels=list(range(NUM_CLASSES)))
cm_normalized = confusion_matrix(gt_all, pred_all, labels=list(range(NUM_CLASSES)), normalize='true')

# Function to create confusion matrix plot
def create_confusion_matrix_plot(matrix, title, filename_base, fmt='.2f', vmax=None):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use colorblind-friendly colormap
    im = ax.imshow(matrix, interpolation='nearest', cmap=CONFUSION_MATRIX_CMAP, 
                   alpha=0.9, vmin=0, vmax=vmax)
    
    # Add colorbar with better formatting
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar_label = 'Proportion' if 'Normalized' in title else 'Number of Samples'
    cbar.set_label(cbar_label, rotation=270, labelpad=20, fontsize=14)
    
    # Set ticks and labels
    tick_marks = np.arange(len(CLASSES))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(CLASSES, rotation=45, ha='right')
    ax.set_yticklabels(CLASSES)
    
    # Add text annotations with better contrast
    thresh = matrix.max() / 2.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if fmt == 'd':
                text = format(int(value), 'd')
            else:
                text = format(value, fmt)
            
            ax.text(j, i, text,
                    ha="center", va="center",
                    color="white" if value > thresh else "black",
                    fontsize=14, fontweight='bold')
    
    # Labels and title
    ax.set_ylabel('True Label', fontsize=16, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=16, fontweight='bold')
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
    
    # Add grid for better readability
    ax.set_xticks(np.arange(len(CLASSES)+1)-.5, minor=True)
    ax.set_yticks(np.arange(len(CLASSES)+1)-.5, minor=True)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, f'{filename_base}.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUTPUT_FOLDER, f'{filename_base}.pdf'), 
                bbox_inches='tight', facecolor='white')
    plt.close()

# Create raw counts confusion matrix
create_confusion_matrix_plot(
    cm, 
    f'RF-DETR Confusion Matrix - Raw Counts\n(IoU ≥ {IOU_THRESHOLD}, Confidence ≥ {CONFIDENCE_THRESHOLD})',
    'confusion_matrix_raw_counts',
    fmt='d'
)

# Create normalized confusion matrix
create_confusion_matrix_plot(
    cm_normalized,
    f'RF-DETR Confusion Matrix - Normalized\n(IoU ≥ {IOU_THRESHOLD}, Confidence ≥ {CONFIDENCE_THRESHOLD})',
    'confusion_matrix_normalized',
    fmt='.3f',
    vmax=1.0
)

# ----------------------------
# 2. Per-Class Performance Metrics
# ----------------------------
print("Creating per-class performance analysis...")

# Calculate metrics for each class
metrics_data = []
for class_name in CLASSES:
    stats = per_class_stats[class_name]
    tp, fp, fn = stats['tp'], stats['fp'], stats['fn']
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics_data.append({
        'Class': class_name,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': tp + fn
    })

metrics_df = pd.DataFrame(metrics_data)

# Create grouped bar chart with enhanced colors
fig, ax = plt.subplots(figsize=(12, 8))

x = np.arange(len(CLASSES))
width = 0.25

bars1 = ax.bar(x - width, metrics_df['Precision'], width, label='Precision', 
               color=COLORBLIND_COLORS[0], alpha=0.85, edgecolor='black', linewidth=1.0)
bars2 = ax.bar(x, metrics_df['Recall'], width, label='Recall', 
               color=COLORBLIND_COLORS[1], alpha=0.85, edgecolor='black', linewidth=1.0)
bars3 = ax.bar(x + width, metrics_df['F1-Score'], width, label='F1-Score', 
               color=COLORBLIND_COLORS[2], alpha=0.85, edgecolor='black', linewidth=1.0)

# Add value labels on bars
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom',
                   fontsize=10, fontweight='bold')

add_value_labels(bars1)
add_value_labels(bars2)
add_value_labels(bars3)

ax.set_xlabel('Classes', fontsize=14, fontweight='bold')
ax.set_ylabel('Score', fontsize=14, fontweight='bold')
ax.set_title(f'Per-Class Performance Metrics\n(Confidence ≥ {CONFIDENCE_THRESHOLD}, IoU ≥ {IOU_THRESHOLD})', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(CLASSES, rotation=45, ha='right')
ax.legend(loc='upper right', frameon=True, shadow=True, fancybox=True)
ax.set_ylim(0, 1.1)
ax.grid(True, alpha=0.3, axis='y')
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, 'per_class_metrics.png'), 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(os.path.join(OUTPUT_FOLDER, 'per_class_metrics.pdf'), 
            bbox_inches='tight', facecolor='white')
plt.close()

# ----------------------------
# 3. Detection Statistics
# ----------------------------
print("Creating detection statistics visualization...")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Distribution of ground truth objects per image
ax1.hist(detection_stats['num_gt'], bins=20, alpha=0.85, color=COLORBLIND_COLORS[3], 
         edgecolor='black', linewidth=1.0)
ax1.set_xlabel('Number of Ground Truth Objects', fontweight='bold')
ax1.set_ylabel('Frequency', fontweight='bold')
ax1.set_title('Distribution of GT Objects per Image', fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_axisbelow(True)

# Distribution of predicted objects per image
ax2.hist(detection_stats['num_pred'], bins=20, alpha=0.85, color=COLORBLIND_COLORS[4], 
         edgecolor='black', linewidth=1.0)
ax2.set_xlabel('Number of Predicted Objects', fontweight='bold')
ax2.set_ylabel('Frequency', fontweight='bold')
ax2.set_title('Distribution of Predicted Objects per Image', fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_axisbelow(True)

# Confidence score distribution
if confidence_scores:
    ax3.hist(confidence_scores, bins=30, alpha=0.85, color=COLORBLIND_COLORS[5], 
             edgecolor='black', linewidth=1.0)
    ax3.axvline(CONFIDENCE_THRESHOLD, color='red', linestyle='--', linewidth=2.5, 
                label=f'Threshold = {CONFIDENCE_THRESHOLD}')
    ax3.set_xlabel('Confidence Score', fontweight='bold')
    ax3.set_ylabel('Frequency', fontweight='bold')
    ax3.set_title('Distribution of Prediction Confidence Scores', fontweight='bold')
    ax3.legend(frameon=True, shadow=True)
    ax3.grid(True, alpha=0.3)
    ax3.set_axisbelow(True)

# Class distribution
class_counts = [sum(1 for x in gt_all if x == i) for i in range(NUM_CLASSES)]
bars = ax4.bar(CLASSES, class_counts, color=COLORBLIND_COLORS[:NUM_CLASSES], 
               alpha=0.85, edgecolor='black', linewidth=1.0)
ax4.set_xlabel('Classes', fontweight='bold')
ax4.set_ylabel('Count', fontweight='bold')
ax4.set_title('Ground Truth Class Distribution', fontweight='bold')
ax4.tick_params(axis='x', rotation=45)
ax4.grid(True, alpha=0.3, axis='y')
ax4.set_axisbelow(True)

# Add value labels on bars
for bar, count in zip(bars, class_counts):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(class_counts)*0.01,
             str(count), ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.suptitle(f'RF-DETR Detection Statistics (Parallel Processing: {BATCH_SIZE})', 
             fontsize=18, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, 'detection_statistics.png'), 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(os.path.join(OUTPUT_FOLDER, 'detection_statistics.pdf'), 
            bbox_inches='tight', facecolor='white')
plt.close()

# ----------------------------
# 4. Save detailed results
# ----------------------------
print("Saving detailed results...")

# Save metrics table
metrics_df.to_csv(os.path.join(OUTPUT_FOLDER, 'per_class_metrics.csv'), index=False)

# Save both confusion matrices as CSV
cm_df = pd.DataFrame(cm, index=CLASSES, columns=CLASSES)
cm_df.to_csv(os.path.join(OUTPUT_FOLDER, 'confusion_matrix_raw.csv'))

cm_normalized_df = pd.DataFrame(cm_normalized, index=CLASSES, columns=CLASSES)
cm_normalized_df.to_csv(os.path.join(OUTPUT_FOLDER, 'confusion_matrix_normalized.csv'))

# Save classification report
if gt_all and pred_all:
    report = classification_report(gt_all, pred_all, target_names=CLASSES, 
                                 labels=list(range(NUM_CLASSES)), output_dict=True)
    with open(os.path.join(OUTPUT_FOLDER, 'classification_report.json'), 'w') as f:
        json.dump(report, f, indent=2)

# Save summary statistics
summary_stats = {
    'total_images': len(img_ids),
    'total_predictions': len(pred_all),
    'total_ground_truth': len([x for x in gt_all if x != 0]),
    'confidence_threshold': CONFIDENCE_THRESHOLD,
    'iou_threshold': IOU_THRESHOLD,
    'parallel_batch_size': BATCH_SIZE,
    'num_workers': NUM_WORKERS,
    'average_gt_per_image': np.mean(detection_stats['num_gt']),
    'average_pred_per_image': np.mean(detection_stats['num_pred']),
    'class_distribution': {CLASSES[i]: class_counts[i] for i in range(NUM_CLASSES)},
    'color_palette_used': 'Okabe-Ito (colorblind-safe)',
    'confusion_matrix_colormap': CONFUSION_MATRIX_CMAP
}

with open(os.path.join(OUTPUT_FOLDER, 'summary_statistics.json'), 'w') as f:
    json.dump(summary_stats, f, indent=2)

print(f"\n✅ Analysis complete! All results saved to '{OUTPUT_FOLDER}' folder:")
print(f"   📊 Raw counts confusion matrix: confusion_matrix_raw_counts.png/pdf")
print(f"   📊 Normalized confusion matrix: confusion_matrix_normalized.png/pdf")
print(f"   📈 Per-class metrics: per_class_metrics.png/pdf")
print(f"   📉 Detection statistics: detection_statistics.png/pdf")
print(f"   📋 CSV files: confusion_matrix_raw.csv, confusion_matrix_normalized.csv, per_class_metrics.csv")
print(f"   📝 Detailed classification report and summary statistics")

# Print summary to console
print(f"\n📊 SUMMARY STATISTICS:")
print(f"   Parallel processing batch size: {BATCH_SIZE}")
print(f"   Number of workers: {NUM_WORKERS}")
print(f"   Total images processed: {len(img_ids)}")
print(f"   Total detections: {len(pred_all)}")
print(f"   Average GT objects per image: {np.mean(detection_stats['num_gt']):.2f}")
print(f"   Average predictions per image: {np.mean(detection_stats['num_pred']):.2f}")
print(f"\n📈 OVERALL PERFORMANCE:")
if gt_all and pred_all:
    overall_accuracy = sum(1 for i, j in zip(gt_all, pred_all) if i == j) / len(gt_all)
    print(f"   Overall accuracy: {overall_accuracy:.3f}")

print(f"\n🎨 Using Okabe-Ito colorblind-friendly palette for publication-quality figures!")
print(f"🎨 Confusion matrices use '{CONFUSION_MATRIX_CMAP}' colormap (colorblind-safe)")
print(f"🚀 Parallel inference with {BATCH_SIZE} concurrent images completed successfully!")