from ultralytics import YOLO

model = YOLO('yolov10n.pt')
model.train(data='/content/proiect_daria/PythonProject/car_dataset/data.yaml', epochs=60, imgsz=640, batch=16)