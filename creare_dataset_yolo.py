import os
import cv2
import random
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov10n.pt")
vehicle_classes = ['car', 'bus', 'truck', 'motorcycle', 'bicycle', 'train']

class_name_to_id = {
    "Audi_A3": 0, "Audi_A4": 1, "Audi_A6": 2, "BMW_Seria_3": 3, "BMW_Seria_5": 4,
    "Dacia_Duster": 5, "Dacia_Logan": 6, "Dacia_Sandero": 7, "Ford_Focus": 8,
    "Mercedes_C-Class": 9, "Renault_Captur": 10, "Renault_Clio": 11, "Renault_Megane": 12,
    "Skoda_Fabia": 13, "Skoda_Octavia": 14, "Skoda_Superb": 15, "Toyota_Corolla": 16,
    "Toyota_Yaris": 17, "Volkswagen_Golf": 18, "Volkswagen_Passat": 19, "Volkswagen_Polo": 20
}

base_dir = 'car_dataset'
image_exts = ('.jpg', '.jpeg', '.png')
train_ratio = 0.8

image_train_dir = os.path.join(base_dir, 'images', 'train')
image_val_dir = os.path.join(base_dir, 'images', 'val')
label_train_dir = os.path.join(base_dir, 'labels', 'train')
label_val_dir = os.path.join(base_dir, 'labels', 'val')

for d in [image_train_dir, image_val_dir, label_train_dir, label_val_dir]:
    os.makedirs(d, exist_ok=True)

class_folders = [f for f in os.listdir(base_dir)
                 if os.path.isdir(os.path.join(base_dir, f)) and f not in ['images', 'labels']]
class_folders.sort()

counters = {class_name_to_id[name]: 1 for name in class_folders}

def save_image_and_label(img, bbox, cls_id, img_out_dir, lbl_out_dir, counters):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    xc = ((x1 + x2) / 2) / w
    yc = ((y1 + y2) / 2) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h

    count = counters[cls_id]
    base_name = f"{cls_id}_{count:04d}"
    counters[cls_id] += 1

    label_path = os.path.join(lbl_out_dir, f"{base_name}.txt")
    with open(label_path, 'w') as f:
        f.write(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

    img_path_out = os.path.join(img_out_dir, f"{base_name}.jpg")
    cv2.imwrite(img_path_out, img)

def rotate_image_and_box(img, bbox, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated_img = cv2.warpAffine(img, M, (w, h))

    x1, y1, x2, y2 = bbox
    points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    ones = np.ones((4, 1))
    points_ones = np.hstack([points, ones])
    rotated_points = M.dot(points_ones.T).T

    rx1, ry1 = rotated_points.min(axis=0)
    rx2, ry2 = rotated_points.max(axis=0)
    rx1 = np.clip(rx1, 0, w)
    ry1 = np.clip(ry1, 0, h)
    rx2 = np.clip(rx2, 0, w)
    ry2 = np.clip(ry2, 0, h)

    return rotated_img, (rx1, ry1, rx2, ry2)

for class_name in class_folders:
    class_id = class_name_to_id[class_name]
    class_folder = os.path.join(base_dir, class_name)
    images = [f for f in os.listdir(class_folder) if f.lower().endswith(image_exts)]

    random.shuffle(images)
    train_cutoff = int(len(images) * train_ratio)
    train_imgs = images[:train_cutoff]
    val_imgs = images[train_cutoff:]

    os.makedirs(os.path.join(image_train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(image_val_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(label_train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(label_val_dir, class_name), exist_ok=True)

    for split, img_list, img_out_dir, lbl_out_dir in [
        ("train", train_imgs, os.path.join(image_train_dir, class_name), os.path.join(label_train_dir, class_name)),
        ("val", val_imgs, os.path.join(image_val_dir, class_name), os.path.join(label_val_dir, class_name))
    ]:
        for img_file in img_list:
            src_img_path = os.path.join(class_folder, img_file)
            img = cv2.imread(src_img_path)
            h, w = img.shape[:2]

            results = model(img)[0]

            max_box = None
            max_area = 0
            for box in results.boxes:
                cls_id_pred = int(box.cls[0])
                class_name_detected = model.names[cls_id_pred]
                if class_name_detected not in vehicle_classes:
                    continue
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    max_box = (x1, y1, x2, y2)

            if max_box:
                save_image_and_label(img, max_box, class_id, img_out_dir, lbl_out_dir, counters)

                img_flipped = cv2.flip(img, 1)
                x1, y1, x2, y2 = max_box
                x1f = w - x2
                x2f = w - x1
                save_image_and_label(img_flipped, (x1f, y1, x2f, y2), class_id, img_out_dir, lbl_out_dir, counters)

                for angle in [-15, 15]:
                    img_rot, rot_box = rotate_image_and_box(img, max_box, angle)
                    save_image_and_label(img_rot, rot_box, class_id, img_out_dir, lbl_out_dir, counters)
            else:
                print(f"Nicio masina detectata in {img_file}, se sare.")
