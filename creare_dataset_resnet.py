from ultralytics import YOLO
import cv2
import os
import glob
from tqdm import tqdm

model = YOLO('yolov10n.pt')

classes = [
    'Audi_A3', 'Audi_A4', 'Audi_A6', 'BMW_Seria_3', 'BMW_Seria_5', 'Dacia_Duster',
    'Dacia_Logan', 'Dacia_Sandero', 'Ford_Focus', 'Mercedes_C-Class', 'Renault_Captur',
    'Renault_Clio', 'Renault_Megane', 'Skoda_Fabia', 'Skoda_Octavia', 'Skoda_Superb',
    'Toyota_Corolla', 'Toyota_Yaris', 'Volkswagen_Golf', 'Volkswagen_Passat', 'Volkswagen_Polo'
]

vehicle_classes = classes

train_images_dir = 'car_dataset/images/train'
val_images_dir = 'car_dataset/images/val'


output_dir = 'cropped_dataset'

for class_name in classes:
    os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)

def detect_and_crop_largest_vehicle(img_path, conf_threshold=0.25):
    image = cv2.imread(img_path)
    results = model(image)

    max_area = 0
    max_box = None

    if results and results[0].boxes is not None:
        for box in results[0].boxes.data:
            x1, y1, x2, y2, conf, cls = box[:6]

            # Filtru pe confidence
            if conf < conf_threshold:
                continue

            cls = int(cls)
            class_name = results[0].names.get(cls, f'class_{cls}')

            if class_name not in vehicle_classes:
                continue

            # Calculeaza aria bbox
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                max_box = (int(x1), int(y1), int(x2), int(y2))

    if max_box is None:
        h, w = image.shape[:2]
        return image, (0, 0, w, h)

    x1, y1, x2, y2 = max_box
    cropped = image[y1:y2, x1:x2]

    return cropped, max_box

for images_dir in [train_images_dir, val_images_dir]:
    image_files = glob.glob(os.path.join(images_dir, '*.jpg'))

    print(f'Processing {images_dir}, {len(image_files)} images...')

    for image_path in tqdm(image_files):
        image_filename = os.path.basename(image_path)

        try:
            idClasa = int(image_filename.split('_')[0])
        except ValueError:
            print(f'[WARNING] Skipping image with invalid filename format: {image_filename}')
            continue

        class_name = classes[idClasa]

        cropped_img, max_box = detect_and_crop_largest_vehicle(image_path)

        if cropped_img is None or cropped_img.size == 0:
            print(f'[WARNING] Invalid crop for image: {image_filename}')
            continue

        output_path = os.path.join(output_dir, class_name, image_filename)
        cv2.imwrite(output_path, cropped_img)

print('All images cropped and saved.')