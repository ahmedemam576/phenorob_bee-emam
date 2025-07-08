import os
import numpy as np

def check_labels(folder):
    for file in os.listdir(folder):
        if file.endswith('.txt'):
            with open(os.path.join(folder, file), 'r') as f:
                for line in f:
                    try:
                        cls, x, y, w, h = map(float, line.strip().split())
                        if not (0 < w < 1 and 0 < h < 1 and 0 < x < 1 and 0 < y < 1):
                            print(f"⚠️ Invalid bbox in {file}: w={w}, h={h}")
                    except:
                        print(f"❌ Malformed line in {file}: {line.strip()}")

check_labels('/scratch/s52melba/yolo_ds_v3/train')
print('-' * 200)
check_labels('/scratch/s52melba/yolo_ds_v3/val')
print('-' * 200)
check_labels('/scratch/s52melba/yolo_ds_v3/test')
