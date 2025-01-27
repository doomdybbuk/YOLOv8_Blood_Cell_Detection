from ultralytics import YOLO
import os


DATA_YAML_PATH = r"C:\Users\Manan Niklank Jain\OneDrive\Desktop\YOLOv8_Blood_Cell_Detection\data\processed\data.yaml"

if not os.path.exists(DATA_YAML_PATH):
    raise FileNotFoundError(f"Data YAML file not found at: {DATA_YAML_PATH}")

model = YOLO("yolov8n.pt")

results = model.train(
    data=DATA_YAML_PATH,
    epochs=100,
    imgsz=640,
    batch=16,
    name="blood_cell_detection",
    project="blood_cell_detection"
)