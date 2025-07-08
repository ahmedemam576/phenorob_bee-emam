from ultralytics import YOLO

# Load a YOLOv8 model (choose one of: 'yolov8n.pt', 'yolov8s.pt', etc. or a custom .pt model)
# model = YOLO("/scratch/s52melba/bee-detection/yolov12l-v2-run13/weights/best.pt") that's the best one so far
model = YOLO("/scratch/s52melba/bee-detection/yolov12l_train_mixed_ds/weights/best.pt")  # or your custom trained model: "runs/detect/train/weights/best.pt"

# Load a dataset defined by a YOLO-format YAML file (specify path to your dataset config)
# The YAML file must have `train:`, `val:`, and `nc:` (number of classes)
data_path = "/scratch/s52melba/yolo_v5_mixed_ds/data.yml"

# Run validation
results = model.val(data=data_path, split='train', save_conf=True, device='1')

# Access the confusion matrix
confusion_matrix = results.confusion_matrix

# Print the confusion matrix
print("Confusion Matrix:")
print(confusion_matrix.matrix)

# Optional: visualize it
confusion_matrix.plot(save_dir="/scratch/s52melba/phenorob_bee/")
