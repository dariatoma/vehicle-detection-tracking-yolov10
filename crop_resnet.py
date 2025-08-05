import os
import cv2
import glob
from tqdm import tqdm
import numpy as np

def auto_crop(image, padding_ratio=0.05, min_area_ratio=0.05,
              min_width_ratio=0.4, min_height_ratio=0.4,
              max_w_h_ratio=2.0, border_ignore=10):

    h_img, w_img = image.shape[:2]
    img_area = h_img * w_img


    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    edges = cv2.Canny(thresh, 50, 150)


    edges[:border_ignore, :] = 0
    edges[-border_ignore:, :] = 0
    edges[:, :border_ignore] = 0
    edges[:, -border_ignore:] = 0

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return image

    valid_contours = [c for c in contours if cv2.contourArea(c) > img_area * min_area_ratio]
    if len(valid_contours) == 0:
        return image

    largest_contour = max(valid_contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(largest_contour)
    w_h_ratio = w / h

    if (w >= w_img * min_width_ratio and
            h >= h_img * min_height_ratio and
            1 / max_w_h_ratio < w_h_ratio < max_w_h_ratio):
        pad_x = int(w * padding_ratio)
        pad_y = int(h * padding_ratio)

        x1 = max(x - pad_x, 0)
        y1 = max(y - pad_y, 0)
        x2 = min(x + w + pad_x, w_img)
        y2 = min(y + h + pad_y, h_img)

        cropped = image[y1:y2, x1:x2]
        return cropped

    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = box.astype(int)

    x_coords = box[:, 0]
    y_coords = box[:, 1]
    x1 = max(int(np.min(x_coords)), 0)
    x2 = min(int(np.max(x_coords)), w_img)
    y1 = max(int(np.min(y_coords)), 0)
    y2 = min(int(np.max(y_coords)), h_img)

    w_rot = x2 - x1
    h_rot = y2 - y1
    pad_x = int(w_rot * padding_ratio)
    pad_y = int(h_rot * padding_ratio)

    x1 = max(x1 - pad_x, 0)
    y1 = max(y1 - pad_y, 0)
    x2 = min(x2 + pad_x, w_img)
    y2 = min(y2 + pad_y, h_img)

    cropped = image[y1:y2, x1:x2]
    return cropped

cropped_dataset_root = 'cropped_dataset'

for class_name in os.listdir(cropped_dataset_root):
    class_dir = os.path.join(cropped_dataset_root, class_name)
    if not os.path.isdir(class_dir):
        continue

    image_files = glob.glob(os.path.join(class_dir, '*.jpg'))

    print(f'Processing class {class_name}, {len(image_files)} images...')

    for image_path in tqdm(image_files):
        image_filename = os.path.basename(image_path)
        image = cv2.imread(image_path)

        # Safety check
        if image is None:
            print(f'[WARNING] Failed to load image: {image_path}')
            continue

        # Auto-crop
        cropped_image = auto_crop(image,
                                  padding_ratio=0.05,
                                  min_area_ratio=0.05,
                                  min_width_ratio=0.4,
                                  min_height_ratio=0.4,
                                  max_w_h_ratio=2.0,
                                  border_ignore=10)

        cv2.imwrite(image_path, cropped_image)

print('All images auto-cropped and saved in cropped_dataset.')
