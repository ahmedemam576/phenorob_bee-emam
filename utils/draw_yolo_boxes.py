import os
import cv2
import argparse

def parse_yolo_annotation_line(line):
    # Handles this format: class x_center y_center width height tensor([conf], device='cuda:0')
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    cls = int(parts[0])
    x_center = float(parts[1])
    y_center = float(parts[2])
    width = float(parts[3])
    height = float(parts[4])
    return cls, x_center, y_center, width, height

def draw_boxes_on_image(img_path, txt_path):
    image = cv2.imread(img_path)
    h, w, _ = image.shape
    with open(txt_path, 'r') as f:
        for line in f:
            parsed = parse_yolo_annotation_line(line)
            if parsed is None:
                continue
            cls, x_center, y_center, box_width, box_height = parsed

            # Convert normalized to absolute pixel values
            x1 = int((x_center - box_width / 2) * w)
            y1 = int((y_center - box_height / 2) * h)
            x2 = int((x_center + box_width / 2) * w)
            y2 = int((y_center + box_height / 2) * h)

            # Draw rectangle and class label
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, str(cls), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return image

def main(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(input_folder):
        if file.endswith(".jpg") or file.endswith(".png"):
            img_path = os.path.join(input_folder, file)
            txt_path = os.path.join(input_folder, os.path.splitext(file)[0] + ".txt")

            if not os.path.exists(txt_path):
                continue

            result_image = draw_boxes_on_image(img_path, txt_path)
            output_path = os.path.join(output_folder, file)
            cv2.imwrite(output_path, result_image)

    print(f"Processed images saved in {output_folder}")

if __name__ == "__main__":
    input_folder = '/scratch/s52melba/infer_images_output/tiles'
    output_folder = '/scratch/s52melba/infer_images_output/drawn'
    main(input_folder, output_folder)
