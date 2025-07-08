import os
import yaml
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def load_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def get_image_paths(image_dir):
    return sorted(glob.glob(os.path.join(image_dir, '*.jpg')) + 
                  glob.glob(os.path.join(image_dir, '*.png')))

def get_label_paths(label_dir):
    return sorted(glob.glob(os.path.join(label_dir, '*.txt')))

def get_image_stats(image_paths):
    hsv_hist = np.zeros((3, 256))
    brightness = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None: continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness.append(np.mean(gray))

        # HSV histograms
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        for i in range(3):
            hist = cv2.calcHist([hsv], [i], None, [256], [0, 256]).flatten()
            hsv_hist[i] += hist
    return hsv_hist, brightness

def get_label_stats(label_paths):
    class_count = Counter()
    objects_per_img = []
    for path in label_paths:
        with open(path) as f:
            lines = f.readlines()
            objects_per_img.append(len(lines))
            for line in lines:
                cls = int(line.split()[0])
                class_count[cls] += 1
    return class_count, objects_per_img

def plot_yolo_dataset_summary(yaml_path, save_path='dataset_summary_hsv.png'):
    data = load_yaml(yaml_path)
    splits = {'Train': data['train'], 'Val': data['val']}
    
    fig, axes = plt.subplots(3, 2, figsize=(18, 12))
    fig.suptitle('YOLO Dataset HSV & Object Analysis: Train vs Val', fontsize=20)

    for i, (split_name, path) in enumerate(splits.items()):
        image_dir = path if os.path.isdir(path) else os.path.join(os.path.dirname(yaml_path), path)
        label_dir = image_dir.replace('images', 'labels')

        image_paths = get_image_paths(image_dir)
        label_paths = get_label_paths(label_dir)

        hsv_hist, brightness = get_image_stats(image_paths)
        class_count, objects_per_img = get_label_stats(label_paths)

        classes = sorted(set(class_count.keys()))
        class_vals = [class_count[c] for c in classes]

        # Object count per image
        axes[0][i].hist(objects_per_img, bins=20, color='orange')
        axes[0][i].set_title(f'{split_name} Objects per Image')
        axes[0][i].set_xlabel('Objects')
        axes[0][i].set_ylabel('Frequency')

        # Class distribution
        axes[1][i].bar(classes, class_vals, color='green')
        axes[1][i].set_title(f'{split_name} Class Distribution')
        axes[1][i].set_xlabel('Class ID')
        axes[1][i].set_ylabel('Count')

        # HSV Histograms (Value only shown here as brightness proxy)
        axes[2][i].plot(hsv_hist[0], label='Hue', color='blue')
        axes[2][i].plot(hsv_hist[1], label='Saturation', color='purple')
        axes[2][i].plot(hsv_hist[2], label='Value', color='black')
        axes[2][i].set_title(f'{split_name} HSV Histogram')
        axes[2][i].set_xlim(0, 256)
        axes[2][i].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    print(f"✅ Summary saved to: {save_path}")



def main():
    plot_yolo_dataset_summary('/scratch/s52melba/yolo_dataset_v2_ready/data.yml','/scratch/s52melba/yolo_dataset_v2_ready/dataset_summary.png')


if __name__ == "__main__":
    main()