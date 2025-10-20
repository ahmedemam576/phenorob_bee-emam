

## 📁 File Reference (What Each File Does)

### Folders

#### `detr_output/`
Stores outputs generated from DETR or RF-DETR models — such as model predictions, logs, or exported results from training and inference.

#### `process_videos/`
Contains helper scripts for creating datasets from raw videos.  
These scripts extract frames, standardize naming, and prepare clean input for labeling or training.

#### `runs/detect/train2/`
Holds experiment outputs (in Ultralytics/YOLO format), including training logs, performance metrics, predictions, and trained model checkpoints.

#### `utils/`
General-purpose utility scripts used across the project.  
Includes metric computation tools such as `utils/all_metrics.py`, which calculates precision, recall, and mAP values from prediction and ground-truth data.

---

### Files

#### `.gitignore`
Lists files and directories ignored by Git (e.g., checkpoints, temporary data, logs, or virtual environments).

#### `README.md`
The original temporary README.  
It links to the dataset and model weights (hosted on Sciebo), explains how to run inference with `infer_image_rfdetr.py`, evaluate using `utils/all_metrics.py`, and follow the RF-DETR environment setup.

#### `JPG_to_jpg.py`
Renames image files from `.JPG` to `.jpg` to ensure consistency in datasets (important for Linux and case-sensitive systems).

#### `coco_to_yolo.py`
Converts COCO-formatted annotation JSON files into YOLO text label files.  
Useful for switching datasets from COCO to YOLO training.

#### `coco_train_val_split.py`
Splits a single COCO annotation file into train, validation, and possibly test subsets.  
Used to create reproducible dataset splits automatically.

#### `create_sahi_dataset.py`
Prepares a **SAHI** (Slicing Aided Hyper Inference) dataset by dividing images into overlapping tiles.  
This helps detect small objects like bees more accurately during training and inference.

#### `dataset_analysis.py`
Performs statistical analysis on the dataset.  
Generates information such as class frequency, bounding box size distribution, and image counts per category — helping assess data quality and balance.

#### `dataset_cleansing.py`
Cleans and validates the dataset.  
Removes broken or empty files, fixes naming inconsistencies, and ensures all images and labels match correctly.

#### `det_with_tracking.py`
Performs detection and tracking on a **single video or live stream**.  
It detects bees in each frame and uses a lightweight tracker to assign consistent IDs to bees across frames.  
Creates an annotated output video and optional per-frame data for further analysis.

#### `det_with_tracking_folder.py`
Processes **all videos or image sequences in a folder**, running detection and tracking for each subfolder.  
It links detections across frames, assigns tracking IDs, and saves results in structured folders.  
Additionally, it is used for **data analysis at the folder level**, enabling batch performance summaries and pattern observation across multiple datasets.

#### `dummy_coco_info.py`
Creates a minimal COCO-format “info” section (dataset metadata).  
Useful when you need a valid COCO file but don’t have complete information such as dataset name or version.

#### `infer_image.py`
Runs inference on a single image using a trained model (YOLO or DETR).  
Outputs detection results and a visualization of bounding boxes.  
Used for quick checks of model performance on individual images.

#### `infer_image_rfdetr.py`
Performs single-image inference using an **RF-DETR** model.  
Reproduces the setup and process referenced in the original README for visualizing detections from a pretrained RF-DETR checkpoint.

#### `no_images.py`
Detects missing images in datasets (e.g., label files without corresponding images).  
Prevents training or validation from failing due to incomplete data.

#### `preprocess.py`
Handles dataset preprocessing steps such as resizing, renaming, normalization, and folder organization before training.

#### `tiled_labeled_to_full_img.py`
Converts annotations made on image tiles (e.g., after SAHI slicing) back to coordinates of the full original image.  
Ensures label consistency between tiled and full-resolution datasets.

#### `train.py`
Main **YOLO training script**.  
Handles configuration, dataset loading, and model training using the Ultralytics framework, with outputs stored in `runs/`.

#### `train_detr.py`
Main **DETR/RF-DETR training script**.  
Trains transformer-based detection models using COCO or YOLO datasets, based on a configuration YAML file.

#### `train_large.py`
A specialized training setup for **large-scale** or multi-GPU training.  
Uses higher image resolution, larger batch sizes, and longer training duration to maximize accuracy on big datasets.

#### `train_val_comparison.py`
Compares metrics between training and validation runs (e.g., loss, mAP, precision).  
Helps identify overfitting or underfitting and visualize performance trends over time.

#### `unique_classes.py`
Lists all unique class IDs and names in the dataset.  
Useful for verifying that label files match the intended class definitions.

#### `yolo_to_coco.py`
Converts YOLO-format labels or predictions back into COCO JSON format.  
Essential when you want to evaluate YOLO results using COCO-based metric scripts.

#### `yolo_val_script.py`
Runs validation on YOLO models and computes metrics such as precision, recall, and mAP.  
It can also export the validation results to COCO JSON format for comparison with ground truth using `utils/all_metrics.py`.

---
