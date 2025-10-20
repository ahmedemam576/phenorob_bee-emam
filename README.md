---

## 📦 Model Weights & Dataset

Model weights and dataset are available here:  
🔗 https://uni-bonn.sciebo.de/s/RQFNDTtHEg6rYDc

---

## 🧪 Inference

To run inference, use the script:  
`phenorob_bee/infer_image_rfdetr.py`

---

## 📊 Evaluation

To compute metrics, use the script:  
`utils/all_metrics.py`

---

## ⚙️ Environment & Training

The environment and training setup are the same as in the original RF-DETR repository:  
🔗 https://github.com/roboflow/rf-detr


## 📁 File Reference — What Each File Does

---

### 🗂️ Folders

#### 📦 `detr_output/`
Stores outputs generated from **DETR** or **RF-DETR** models — such as predictions, logs, and exported results from training and inference.

#### 🎥 `process_videos/`
Contains helper scripts for **creating datasets from raw videos**.  
These scripts extract frames, standardize filenames, and prepare clean input for labeling or training.

#### 📊 `runs/detect/train2/`
Holds **experiment outputs** (YOLO format), including training logs, metrics, predictions, and trained model checkpoints.

#### 🧰 `utils/`
General-purpose utility scripts used throughout the project.  
Includes tools like `utils/all_metrics.py`, which calculates **precision**, **recall**, and **mAP** from predictions and ground truth.

---

### 🧾 Files

#### ⚙️ `.gitignore`
Lists files and directories ignored by Git (e.g., checkpoints, temporary data, logs, or virtual environments).

#### 📘 `README.md`
The **original temporary README**.  
Links to the dataset and model weights (Sciebo), and explains how to run inference with `infer_image_rfdetr.py`, evaluate with `utils/all_metrics.py`, and follow the RF-DETR setup.

#### 🖼️ `JPG_to_jpg.py`
Renames images from `.JPG` to `.jpg` to ensure consistency, especially on Linux or case-sensitive systems.

#### 🔄 `coco_to_yolo.py`
Converts **COCO JSON annotations** into **YOLO text labels**.  
Useful for switching datasets from COCO to YOLO format.

#### ✂️ `coco_train_val_split.py`
Splits a single COCO annotation file into **train**, **validation**, and optionally **test** subsets.  
Automates dataset partitioning for reproducibility.

#### 🧩 `create_sahi_dataset.py`
Builds a **SAHI (Slicing Aided Hyper Inference)** dataset by dividing large images into overlapping tiles.  
Improves detection performance on **small objects like bees**.

#### 📈 `dataset_analysis.py`
Performs **statistical analysis** on the dataset.  
Calculates class frequencies, bounding box size distributions, and image counts per class.

#### 🧹 `dataset_cleansing.py`
Cleans and validates datasets by removing broken files, fixing label mismatches, and standardizing structure.

#### 🎯 `det_with_tracking.py`
Performs **detection and tracking** on a single video or live stream.  
Assigns unique IDs to bees across frames and generates annotated output videos for further analysis.

#### 📂 `det_with_tracking_folder.py`
Runs **detection + tracking** for an entire folder of videos or image sequences.  
Connects detections across frames and saves results in organized subfolders.  
Also used for **data analysis per folder**, providing aggregated performance summaries and behavioral insights across datasets.

#### 🪪 `dummy_coco_info.py`
Creates placeholder metadata (“info” section) for COCO-format annotation files when full dataset details aren’t available.

#### 🧠 `infer_image.py`
Runs **inference on a single image** using a YOLO or DETR model.  
Saves prediction results and a visualization with bounding boxes.

#### 🧠 `infer_image_rfdetr.py`
Performs **RF-DETR single-image inference**.  
Reproduces the workflow described in the original README and visualizes detections from a pretrained RF-DETR model.

#### 🚫 `no_images.py`
Finds label files that reference missing images.  
Prevents training or validation crashes caused by incomplete datasets.

#### ⚙️ `preprocess.py`
Handles **preprocessing steps** such as resizing, renaming, normalization, and reorganizing images or labels before training.

#### 🧩 `tiled_labeled_to_full_img.py`
Merges annotations from **tiled images** (after SAHI slicing) back to coordinates in the **full original image**.

#### 🧠 `train.py`
Main **YOLO training script**.  
Loads configurations, datasets, and models, and saves all results to the `runs/` directory.

#### 🤖 `train_detr.py`
Main **DETR / RF-DETR training script**.  
Trains a transformer-based detection model using dataset and hyperparameter YAML configurations.

#### 🧮 `train_large.py`
A specialized **large-scale or multi-GPU training** script.  
Uses higher image resolutions, larger batch sizes, and longer schedules for better accuracy on big datasets.

#### 📊 `train_val_comparison.py`
Compares **training vs validation** performance (loss, mAP, precision) to detect overfitting or underfitting trends.

#### 🧾 `unique_classes.py`
Lists all **unique class IDs and names** in your dataset.  
Useful for verifying that labels match class definitions.

#### 🔁 `yolo_to_coco.py`
Converts **YOLO-format labels or predictions** into **COCO JSON format**.  
Used for evaluating YOLO outputs with COCO-based tools.

#### 🧮 `yolo_val_script.py`
Runs **YOLO validation** and computes metrics such as precision, recall, and mAP.  
Can export validation results to COCO JSON for use with `utils/all_metrics.py`.

---
