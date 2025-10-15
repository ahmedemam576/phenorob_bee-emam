# from rfdetr import RFDETRBase
# import os

# model = RFDETRBase()
# output_dir = '/home/s52melba/phenorob_bee/detr_output'
# # if exist raise error and stop
# if os.path.exists(output_dir):
#     raise FileExistsError(f"Output directory {output_dir} already exists. Please remove it or choose a different name.")
# # Create the output directory
# os.makedirs(output_dir, exist_ok=True)

# model.train(dataset_dir='/home/s52melba/dataset_rtdetr_format',
#              epochs=100, batch_size=32, grad_accum_steps=2, lr=1e-4, output_dir=output_dir,
#             #  resume="/scratch/s52melba/phenorob_bee/detr_output/checkpoint_best_total.pth",
#              tensorboard=True
#              )



from rfdetr import RFDETRBase
import os

model = RFDETRBase()
output_dir = '/scratch/s52melba/detr_train_2'

if os.path.exists(output_dir):
    raise FileExistsError(f"Output directory {output_dir} already exists.")
os.makedirs(output_dir, exist_ok=True)

model.train(
    dataset_dir='/home/s52melba/dataset_rtdetr_format',
    epochs=150,  # Increase epochs since using lower LR
    batch_size=32,
    grad_accum_steps=2,
    lr=1e-5,  # Reduced learning rate (most important change)
    lr_encoder=5e-6,  # Even lower LR for encoder
    output_dir=output_dir,
    
    # Resume from best checkpoint
    resume="/home/s52melba/phenorob_bee/detr_output/checkpoint_best_total.pth",
    
    # Regularization (confirmed parameters)
    weight_decay=1e-4,
    use_ema=True,
    
    # Early stopping to prevent overfitting
    early_stopping=True,
    early_stopping_patience=15,
    early_stopping_min_delta=0.001,
    early_stopping_use_ema=True,
    
    # Logging
    tensorboard=True,
    checkpoint_interval=5  # Save checkpoints every 5 epochs
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
