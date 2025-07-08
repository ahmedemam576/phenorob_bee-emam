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
model = YOLO('/scratch/s52melba/bee-detection/yolov12l-v2-run13/weights/best.pt')

results = model.train(
  data=f'/scratch/s52melba/yolo_v5_mixed_ds/data.yml',
  optimizer='AdamW',
  epochs=200,
  save_period=40,
  batch=12,
  dropout=0.2,
  plots=True,
  flipud=0.2,
  mosaic=0.2,
  crop_fraction=0.2,
  amp=False,
  close_mosaic=50,
  degrees=40,
  shear=10,
  mixup=0.0,
  device = "0,2,3",
  project='bee-detection',      # name of the project in Comet
  name='yolov12l_train_mixed_ds',      # optional run name
  tracker='comet' 
)
