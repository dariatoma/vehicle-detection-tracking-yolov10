import os
import random
import shutil

base_image_dir = "car_dataset/images"
base_label_dir = "car_dataset/labels"

train_image_dir = os.path.join(base_image_dir, "train")
val_image_dir = os.path.join(base_image_dir, "val")
train_label_dir = os.path.join(base_label_dir, "train")
val_label_dir = os.path.join(base_label_dir, "val")

os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(val_image_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

all_images = [f for f in os.listdir(base_image_dir) if f.lower().endswith('.jpg')]
random.shuffle(all_images)

split_idx = int(0.8 * len(all_images))
train_images = all_images[:split_idx]
val_images = all_images[split_idx:]

def move_files(image_list, img_dest, lbl_dest):
    for image_file in image_list:
        base_name = os.path.splitext(image_file)[0]
        label_file = base_name + '.txt'

        src_img = os.path.join(base_image_dir, image_file)
        src_lbl = os.path.join(base_label_dir, label_file)

        dst_img = os.path.join(img_dest, image_file)
        dst_lbl = os.path.join(lbl_dest, label_file)

        if os.path.exists(src_img):
            shutil.move(src_img, dst_img)
        if os.path.exists(src_lbl):
            shutil.move(src_lbl, dst_lbl)

move_files(train_images, train_image_dir, train_label_dir)
move_files(val_images, val_image_dir, val_label_dir)

print(f"Total fișiere: {len(all_images)}")
print(f"Train: {len(train_images)}")
print(f"Val: {len(val_images)}")
