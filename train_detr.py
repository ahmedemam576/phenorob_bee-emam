from rfdetr import RFDETRBase
import os

model = RFDETRBase()
output_dir = '/scratch/s52melba/phenorob_bee/detr_output_v2'
# if exist raise error and stop
if os.path.exists(output_dir):
    raise FileExistsError(f"Output directory {output_dir} already exists. Please remove it or choose a different name.")
# Create the output directory
os.makedirs(output_dir, exist_ok=True)

model.train(dataset_dir='/scratch/s52melba/yolo_v5_mixed_ds_no_empty_images_coco',
             epochs=30, batch_size=3, grad_accum_steps=4, lr=1e-4, output_dir=output_dir,
             resume="/scratch/s52melba/phenorob_bee/detr_output/checkpoint_best_total.pth",
             tensorboard=True
             )

# import torch

# if torch.cuda.is_available():
#     num_gpus = torch.cuda.device_count()
#     print(f"✅ {num_gpus} GPU(s) available:")
#     for i in range(num_gpus):
#         print(f"  [{i}] {torch.cuda.get_device_name(i)}")
#     # init a tensor and send to gpu
#     device = torch.device("cuda:0")
#     tensor = torch.randn(2, 2).to(device)
#     print(f"Tensor on GPU: {tensor}")
# else:
#     print("❌ No CUDA-compatible GPU found.")


# import json
# import os

# # Define categories
# categories = [
#     {"id": 0, "name": "nothing", "supercategory": "none"},
#     {"id": 1, "name": "honey_bee", "supercategory": "none"},
#     {"id": 2, "name": "bumble_bee", "supercategory": "none"},
#     {"id": 3, "name": "unidentified", "supercategory": "none"},
# ]

# # Paths to each annotation file
# coco_paths = [
#     "/scratch/s52melba/yolo_v5_mixed_ds_no_empty_images_coco/train/_annotations.coco.json",
#     "/scratch/s52melba/yolo_v5_mixed_ds_no_empty_images_coco/valid/_annotations.coco.json",
#     "/scratch/s52melba/yolo_v5_mixed_ds_no_empty_images_coco/test/_annotations.coco.json",
# ]

# for path in coco_paths:
#     with open(path, "r") as f:
#         data = json.load(f)

#     # Replace or add the categories
#     data["categories"] = categories

#     with open(path, "w") as f:
#         json.dump(data, f, indent=2)

#     print(f"Updated categories in: {path}")
