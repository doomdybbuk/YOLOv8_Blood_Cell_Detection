from ultralytics import YOLO
import os


# Define the path to the data YAML file
# DATA_YAML_PATH = r"C:\Users\Manan Niklank Jain\OneDrive\Desktop\YOLOv8_Blood_Cell_Detection\data\processed\data.yaml"
DATA_YAML_PATH = "../data/processed/data.yaml"

# Check if the data YAML file exists at the specified path
if not os.path.exists(DATA_YAML_PATH):
    # Raise a FileNotFoundError if the file does not exist
    raise FileNotFoundError(f"Data YAML file not found at: {DATA_YAML_PATH}")

# Initialize a YOLO model with the weights from the "yolov8n.pt" file
model = YOLO("yolov8n.pt")

# Train the YOLO model on the data specified in the data YAML file
results = model.train(
    # Path to the data YAML file
    data=DATA_YAML_PATH,
    # Number of epochs to train the model for
    epochs=125,
    # Image size for training
    imgsz=640,
    # Batch size for training
    batch=16,
    # Name of the training run
    name="blood_cell_detection",
    # Project name for the training run
    project="blood_cell_detection"
)