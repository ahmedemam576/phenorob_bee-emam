from sahi.utils.coco import Coco, export_coco_as_yolov5

train_img_dir = '/scratch/s52melba/coco_sahi_256_train_val/train'
val_img_dir = '/scratch/s52melba/coco_sahi_256_train_val/val'

train_json_file = '/scratch/s52melba/coco_sahi_256_train_val/train.json'
val_json_file = '/scratch/s52melba/coco_sahi_256_train_val/val.json'

# init Coco object
train_coco = Coco.from_coco_dict_or_path(train_json_file, image_dir=train_img_dir)
val_coco = Coco.from_coco_dict_or_path(val_json_file, image_dir=val_img_dir)

# export converted YoloV5 formatted dataset into given output_dir with given train/val split
data_yml_path = export_coco_as_yolov5(
  output_dir="/scratch/s52melba/dataset_yolo_sahi_256",
  train_coco=train_coco,
  val_coco=val_coco
)