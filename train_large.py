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

model = YOLO('/scratch/s52melba/phenorob_bee/yolov12l.pt')

results = model.train(
  data=f'/scratch/s52melba/dataset_yolo_sahi_256/data.yml',
  epochs=200,
  batch=8,
  # dropout=0.2,
  plots=True,
  flipud=0.2,
  mosaic=0.2,
  mixup=0.2,
  crop_fraction=0.2,
  device = "1",
)