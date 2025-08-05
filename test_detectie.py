import os
import cv2
import time
import numpy as np
from ultralytics import YOLO

class CentroidTrackerV2:
    def __init__(self, maxDisappeared=2, maxDistance=50):
        self.nextObjectID = 0
        self.objects = dict()
        self.disappeared = dict()
        self.maxDisappeared = maxDisappeared
        self.maxDistance = maxDistance

    def register(self, centroid, class_name):
        self.objects[self.nextObjectID] = (centroid, class_name)
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, detections):
        input_centroids = []
        input_classes = []

        for (x1, y1, x2, y2, class_name) in detections:
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            input_centroids.append((cX, cY))
            input_classes.append(class_name)

        if len(input_centroids) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        if len(self.objects) == 0:
            for i, centroid in enumerate(input_centroids):
                self.register(centroid, input_classes[i])
            return self.objects

        objectIDs = list(self.objects.keys())
        objectCentroids = [self.objects[objectID][0] for objectID in objectIDs]

        D = np.linalg.norm(np.array(objectCentroids)[:, np.newaxis] - np.array(input_centroids), axis=2)

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        usedRows = set()
        usedCols = set()

        for (row, col) in zip(rows, cols):
            if row in usedRows or col in usedCols:
                continue
            if D[row, col] > self.maxDistance:
                continue

            objectID = objectIDs[row]
            self.objects[objectID] = (input_centroids[col], input_classes[col])
            self.disappeared[objectID] = 0

            usedRows.add(row)
            usedCols.add(col)

        unusedRows = set(range(0, len(objectCentroids))).difference(usedRows)
        for row in unusedRows:
            objectID = objectIDs[row]
            self.disappeared[objectID] += 1
            if self.disappeared[objectID] > self.maxDisappeared:
                self.deregister(objectID)

        unusedCols = set(range(0, len(input_centroids))).difference(usedCols)
        for col in unusedCols:
            self.register(input_centroids[col], input_classes[col])

        return self.objects


model = YOLO("runs/detect/train3/weights/best.pt")
vehicle_classes = list(model.names.values())
print(f" Vehicle classes: {vehicle_classes}")

input_folder = '/home/daria/Desktop/input'
output_folder = '/home/daria/Desktop/output'
os.makedirs(output_folder, exist_ok=True)


video_files = [f for f in os.listdir(input_folder) if f.endswith(('.mp4', '.mov', '.avi'))]
print(f"Found {len(video_files)} video files to process.\n")

video_rotation_map = {
    'bunbun29.mov': True,    # primul video rotit
    'bunfata.mov': False     # al doilea video NU rotit
}

# Loop pe video-uri
for video_name in video_files:
    input_video = os.path.join(input_folder, video_name)
    video_base_name = os.path.splitext(video_name)[0]
    output_video = os.path.join(output_folder, f'{video_base_name}_tracked.mp4')

    rotate_frame = video_rotation_map.get(video_name, False)

    print(f"Processing {video_name}...")

    cap = cv2.VideoCapture(input_video)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    tracker = CentroidTrackerV2(maxDisappeared=2, maxDistance=50)

    frame_idx = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if rotate_frame:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        results = model(frame, conf=0.4, iou=0.5)[0]

        detections = []

        if results.boxes is not None:
            for box in results.boxes.data:
                x1, y1, x2, y2, conf, cls = box[:6]
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                conf = float(conf)
                cls = int(cls)
                class_name = model.names.get(cls, f'class_{cls}')

                if class_name not in vehicle_classes:
                    continue

                detections.append((x1, y1, x2, y2, class_name))

        objects = tracker.update(detections)

        for (objectID, (centroid, class_name)) in objects.items():
            for (x1, y1, x2, y2, c_name) in detections:
                cX = int((x1 + x2) / 2.0)
                cY = int((y1 + y2) / 2.0)
                if abs(cX - centroid[0]) < 5 and abs(cY - centroid[1]) < 5:
                    label = f'ID {objectID} | {class_name}'

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame, label, (x1, y1 - 10), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        out.write(frame)

        frame_idx += 1
        if frame_idx % 30 == 0:
            elapsed = time.time() - start_time
            print(f"Frame {frame_idx}/{frame_count} | Time elapsed: {elapsed:.1f}s")

    cap.release()
    out.release()
    print(f"Output saved to {output_video}\n")

print("All videos processed!")
