import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from PIL import Image
import numpy as np

class_names = ['Audi_A3', 'Audi_A4', 'Audi_A6', 'BMW_Seria_3', 'BMW_Seria_5',
               'Dacia_Duster', 'Dacia_Logan', 'Dacia_Sandero', 'Ford_Focus',
               'Mercedes_C-Class', 'Renault_Captur', 'Renault_Clio', 'Renault_Megane',
               'Skoda_Fabia', 'Skoda_Octavia', 'Skoda_Superb', 'Toyota_Corolla',
               'Toyota_Yaris', 'Volkswagen_Golf', 'Volkswagen_Passat', 'Volkswagen_Polo']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

resnet = models.resnet50(weights=None)
resnet.fc = nn.Linear(resnet.fc.in_features, len(class_names))
resnet.load_state_dict(torch.load('best_resnet50.pth', map_location=device))
resnet = resnet.to(device)
resnet.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

from ultralytics import YOLO
model_yolo = YOLO("runs/detect/train3/weights/best.pt")

video_path = '/home/daria/Desktop/input/bunfata.mov'
cap = cv2.VideoCapture(video_path)

output_path = '/home/daria/Desktop/output/video_final.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # frame = cv2.rotate(frame, cv2.ROTATE_180)   # pentru bunbun29.mov

    results = model_yolo(frame)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    confidences = results.boxes.conf.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        conf = confidences[i]

        if conf < 0.4:
            continue

        cropped = frame[y1:y2, x1:x2]

        try:
            pil_image = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            input_tensor = transform(pil_image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = resnet(input_tensor)
                probs = torch.softmax(output, dim=1)
                top_prob, top_class = torch.max(probs, 1)
                label = class_names[top_class.item()]
                score = top_prob.item()

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            text = f'{label} ({score*100:.1f}%)'
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        except Exception as e:
            print(f'[WARNING] Error on crop/classify: {e}')
            continue

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()

print(f'\n Video salvat: {output_path}')