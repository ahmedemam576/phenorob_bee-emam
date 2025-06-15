# import supervision as sv
# from ultralytics import YOLO
# import numpy as np

# dataset = sv.DetectionDataset.from_yolo(images_directory_path= '/home/s52melba/dataset_bee_annotated/val/images',
#                                         annotations_directory_path= '/home/s52melba/dataset_bee_annotated/val/labels',
#                                         data_yaml_path='/home/s52melba/dataset_bee_annotated/data.yaml')


# def callback(image: np.ndarray) -> sv.Detections:
#     result = model(image)[0]
#     return sv.Detections.from_ultralytics(result)


from ultralytics import YOLO
import os
os.environ["COMET_API_KEY"] = "P8W7VEZekEpLpMT9j82D2i3FA"
model = YOLO('/scratch/s52melba/phenorob_bee/yolov12l.pt')

results = model.train(
  data=f'/scratch/s52melba/yolo_dataset_v2_ready/data.yml',
  epochs=200,
  batch=8,
  # dropout=0.2,
  plots=True,
  flipud=0.2,
  mosaic=0.2,
  mixup=0.2,
  crop_fraction=0.2,
  device = "1",
  project='bee-detection',      # name of the project in Comet
  name='yolov12l-v2-run1',      # optional run name
  tracker='comet' 
)
