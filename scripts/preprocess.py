import os
import cv2
import random
import xml.etree.ElementTree as ET
from shutil import copyfile

# Configuration
random.seed(42)
DATASET_PATH = r"C:\Users\Manan Niklank Jain\OneDrive\Desktop\YOLOv8_Blood_Cell_Detection\data\raw"
OUTPUT_PATH = os.path.join(os.path.dirname(DATASET_PATH), "processed")
SPLITS = {"train": 0.7, "val": 0.2, "test": 0.1}
IMG_SIZE = (640, 640)

# Create directories
for split in SPLITS:
    os.makedirs(os.path.join(OUTPUT_PATH, "images", split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_PATH, "labels", split), exist_ok=True)

# Get all files
image_files = [f for f in os.listdir(os.path.join(DATASET_PATH, "JPEGImages")) if f.endswith(".jpg")]
random.shuffle(image_files)

# Split files
total = len(image_files)
train_count = int(total * SPLITS["train"])
val_count = int(total * SPLITS["val"])

splits = {
    "train": image_files[:train_count],
    "val": image_files[train_count:train_count+val_count],
    "test": image_files[train_count+val_count:]
}

# Class mapping
classes = {"RBC": 0, "WBC": 1, "Platelets": 2}

# Process each split
for split, files in splits.items():
    for file in files:
        # Process image
        img_path = os.path.join(DATASET_PATH, "JPEGImages", file)
        img = cv2.imread(img_path)
        img = cv2.resize(img, IMG_SIZE)
        cv2.imwrite(os.path.join(OUTPUT_PATH, "images", split, file), img)
        
        # Process annotations
        xml_path = os.path.join(DATASET_PATH, "Annotations", file.replace(".jpg", ".xml"))
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        label_file = file.replace(".jpg", ".txt")
        with open(os.path.join(OUTPUT_PATH, "labels", split, label_file), "w") as f:
            for obj in root.findall("object"):
                cls_name = obj.find("name").text
                if cls_name not in classes:
                    continue
                
                bbox = obj.find("bndbox")
                xmin = int(bbox.find("xmin").text)
                ymin = int(bbox.find("ymin").text)
                xmax = int(bbox.find("xmax").text)
                ymax = int(bbox.find("ymax").text)
                
                # Convert to YOLO format
                img_w = int(root.find("size").find("width").text)
                img_h = int(root.find("size").find("height").text)
                
                x_center = ((xmin + xmax) / 2) / img_w
                y_center = ((ymin + ymax) / 2) / img_h
                width = (xmax - xmin) / img_w
                height = (ymax - ymin) / img_h
                
                f.write(f"{classes[cls_name]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

print("Preprocessing completed!")