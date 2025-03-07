import supervision as sv
from ultralytics import YOLO
import numpy as np

dataset = sv.DetectionDataset.from_yolo(images_directory_path= '/home/s52melba/dataset_bee_annotated/val/images',
                                        annotations_directory_path= '/home/s52melba/dataset_bee_annotated/val/labels',
                                        data_yaml_path='/home/s52melba/dataset_bee_annotated/data.yaml')

model = YOLO('yolo12s.pt')

def callback(image: np.ndarray) -> sv.Detections:
    result = model(image)[0]
    return sv.Detections.from_ultralytics(result)