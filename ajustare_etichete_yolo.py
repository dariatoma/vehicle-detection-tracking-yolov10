import cv2
import os

def calculate_yolo_coords(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Nu pot citi imaginea {img_path}")
        return None

    _, thresh = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"Niciun contur găsit în {img_path}")
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    img_h, img_w = img.shape[:2]
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    width = w / img_w
    height = h / img_h

    return x_center, y_center, width, height

def write_label_file(txt_path, class_id, coords):
    if coords is None:
        with open(txt_path, 'w') as f:
            pass
        return

    x_center, y_center, width, height = coords
    with open(txt_path, 'w') as f:
        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

base_dir = "/content/proiect_daria/PythonProject/car_dataset"
base_image_dir = os.path.join(base_dir, "images")
base_label_dir = os.path.join(base_dir, "labels")
base_dirs = ['train', 'val']

class_name_to_id = {
    "Audi_A3": 0,
    "Audi_A4": 1,
    "Audi_A6": 2,
    "BMW_Seria_3": 3,
    "BMW_Seria_5": 4,
    "Dacia_Duster": 5,
    "Dacia_Logan": 6,
    "Dacia_Sandero": 7,
    "Ford_Focus": 8,
    "Mercedes_C-Class": 9,
    "Renault_Captur": 10,
    "Renault_Clio": 11,
    "Renault_Megane": 12,
    "Skoda_Fabia": 13,
    "Skoda_Octavia": 14,
    "Skoda_Superb": 15,
    "Toyota_Corolla": 16,
    "Toyota_Yaris": 17,
    "Volkswagen_Golf": 18,
    "Volkswagen_Passat": 19,
    "Volkswagen_Polo": 20
}

for split in base_dirs:
    img_split_dir = os.path.join(base_image_dir, split)
    label_split_dir = os.path.join(base_label_dir, split)

    for class_name in os.listdir(img_split_dir):
        class_img_dir = os.path.join(img_split_dir, class_name)
        class_label_dir = os.path.join(label_split_dir, class_name)

        if not os.path.isdir(class_img_dir):
            continue

        class_id = class_name_to_id.get(class_name)
        if class_id is None:
            print(f"Clasa {class_name} nu are un ID definit în class_name_to_id")
            continue

        os.makedirs(class_label_dir, exist_ok=True)

        for img_file in os.listdir(class_img_dir):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            img_path = os.path.join(class_img_dir, img_file)
            txt_file = os.path.splitext(img_file)[0] + ".txt"
            txt_path = os.path.join(class_label_dir, txt_file)

            coords = calculate_yolo_coords(img_path)
            write_label_file(txt_path, class_id, coords)



class_folders = [f for f in os.listdir(base_dir)
                 if os.path.isdir(os.path.join(base_dir, f)) and f not in ['images', 'labels']]
class_folders.sort()


data_yaml_path = os.path.join(base_dir, 'data.yaml')
with open(data_yaml_path, 'w') as f:
    f.write(f"path: {os.path.abspath(base_dir)}\n")
    f.write("train: images/train\n")
    f.write("val: images/val\n")
    f.write(f"nc: {len(class_folders)}\n")
    class_names_str = "[" + ", ".join(f"'{name}'" for name in class_folders) + "]"
    f.write(f"names: {class_names_str}\n")

