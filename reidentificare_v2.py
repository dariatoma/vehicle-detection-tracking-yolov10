import os
import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import numpy as np
from PIL import Image
from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class_names = [
    'Audi_A3', 'Audi_A4', 'Audi_A6', 'BMW_Seria_3', 'BMW_Seria_5',
    'Dacia_Duster', 'Dacia_Logan', 'Dacia_Sandero', 'Ford_Focus',
    'Mercedes_C-Class', 'Renault_Captur', 'Renault_Clio', 'Renault_Megane',
    'Skoda_Fabia', 'Skoda_Octavia', 'Skoda_Superb', 'Toyota_Corolla',
    'Toyota_Yaris', 'Volkswagen_Golf', 'Volkswagen_Passat', 'Volkswagen_Polo'
]


model_yolo = YOLO("runs/detect/train3/weights/best.pt")
vehicle_classes = list(model_yolo.names.values())


resnet = models.resnet50(weights=None)
resnet.fc = nn.Linear(resnet.fc.in_features, len(class_names))
resnet.load_state_dict(torch.load('best_resnet50.pth', map_location=device))
resnet = resnet.to(device)
resnet.eval()

feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
feature_extractor.eval()


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


instances_db = []  # Global buffer cu toate instanțele
next_track_id = 0

MAX_MISSED_FRAMES = 200
SIM_THRESHOLD = 0.80


def search_by_image(image_path, class_id_filter=None, top_k=5):
    pil_image = Image.open(image_path).convert("RGB")
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = feature_extractor(input_tensor)
        features = features.view(features.size(0), -1).cpu().numpy()[0]

    if class_id_filter is not None:
        candidates = [inst for inst in instances_db if inst['class_id'] == class_id_filter]
    else:
        candidates = instances_db

    if len(candidates) == 0:
        print("Nu există instanțe în baza de date.")
        return []

    existing_descriptors = np.array([inst['features'] for inst in candidates])
    similarities = cosine_similarity([features], existing_descriptors)[0]

    top_indices = np.argsort(similarities)[::-1][:top_k]
    results = []
    for idx in top_indices:
        inst = candidates[idx]
        sim_score = similarities[idx]
        results.append({
            'track_id': inst['track_id'],
            'class_id': inst['class_id'],
            'class_name': class_names[inst['class_id']],
            'similarity': sim_score,
            'first_seen_frame': inst['frame_idx'],
            'last_seen_frame': inst['last_seen'],
            'history_frames': inst['history_frames']
        })

    return results

video_path = '/home/daria/Desktop/vidlicenta/video1.mov'

cap = cv2.VideoCapture(video_path)

output_path = '/home/daria/Desktop/output/output_final_vid1.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model_yolo(frame)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    confidences = results.boxes.conf.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        conf = confidences[i]
        cls = int(classes[i])
        class_name = model_yolo.names.get(cls, f'class_{cls}')

        if class_name not in vehicle_classes or conf < 0.4:
            continue

        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0:
            continue

        try:
            # Transform pt ResNet
            pil_image = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            input_tensor = transform(pil_image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = resnet(input_tensor)
                probs = torch.softmax(output, dim=1)
                top_prob, top_class = torch.max(probs, 1)
                label_class_id = top_class.item()
                label_class_name = class_names[label_class_id]

            with torch.no_grad():
                features = feature_extractor(input_tensor)
                features = features.view(features.size(0), -1).cpu().numpy()[0]  # np array [2048]

            candidates = [inst for inst in instances_db if inst['class_id'] == label_class_id]

            matched = False
            current_track_id = None

            if len(candidates) > 0:
                existing_descriptors = np.array([inst['features'] for inst in candidates])
                similarities = cosine_similarity([features], existing_descriptors)[0]

                best_match_idx = np.argmax(similarities)
                best_match_score = similarities[best_match_idx]

                if best_match_score > SIM_THRESHOLD:
                    current_track_id = candidates[best_match_idx]['track_id']
                    candidates[best_match_idx]['last_seen'] = frame_idx  # update last_seen
                    if candidates[best_match_idx]['history_frames'][-1] != frame_idx:
                        candidates[best_match_idx]['history_frames'].append(frame_idx)
                    matched = True

            if not matched:
                current_track_id = next_track_id
                next_track_id += 1
                instance_entry = {
                    'features': features,
                    'track_id': current_track_id,
                    'frame_idx': frame_idx,
                    'last_seen': frame_idx,
                    'class_id': label_class_id,
                    'history_frames': [frame_idx]
                }
                instances_db.append(instance_entry)

            label_text = f'{label_class_name} - ID {current_track_id}'
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        except Exception as e:
            print(f'Error on crop/classify: {e}')
            continue

    instances_db = [inst for inst in instances_db if (frame_idx - inst['last_seen']) < MAX_MISSED_FRAMES]

    out.write(frame)

    frame_idx += 1
    if frame_idx % 30 == 0:
        print(f'Frame {frame_idx} processed.')

cap.release()
out.release()
cv2.destroyAllWindows()
print(f'\n Video salvat cu succes: {output_path}')

image_results = search_by_image('/home/daria/Desktop/input/test_search.jpg', class_id_filter=6)
for res in image_results:
    print(f"Track ID: {res['track_id']}, Clasa: {res['class_name']}, Similaritate: {res['similarity']:.2f}")
    print(f"First seen: frame {res['first_seen_frame']}, Last seen: frame {res['last_seen_frame']}")
    print(f"History frames: {res['history_frames']}")
    print('---')
