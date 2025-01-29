# Import necessary libraries for image processing and file operations
import os  # Import the os module for interacting with the operating system
import cv2  # Import the cv2 module for image processing
import random  # Import the random module for shuffling the dataset
import xml.etree.ElementTree as ET  # Import the ET module for parsing XML files
from shutil import copyfile  # Import the copyfile function for copying files
import numpy as np  # Import the np module for numerical operations

# Configuration for the dataset and output paths
random.seed(42)  # Set the random seed for reproducibility
DATASET_PATH = r"C:\Users\Manan Niklank Jain\OneDrive\Desktop\YOLOv8_Blood_Cell_Detection\data\raw"  # Define the path to the raw dataset
OUTPUT_PATH = r"C:\Users\Manan Niklank Jain\OneDrive\Desktop\YOLOv8_Blood_Cell_Detection\data\processed"  # Define the path to the processed dataset
SPLIT_RATIOS = {"train": 0.7, "val": 0.2, "test": 0.1}  # Define the split ratios for the dataset
IMG_SIZE = (640, 640)  # Define the image size for resizing

# Create a class mapping for the blood cell types
classes = {"RBC": 0, "WBC": 1, "Platelets": 2}  # Define the class mapping

# Create output directories for the processed dataset
os.makedirs(OUTPUT_PATH, exist_ok=True)  # Create the output directory if it doesn't exist
for split in SPLIT_RATIOS.keys():  # Iterate over the split ratios
    os.makedirs(os.path.join(OUTPUT_PATH, "images", split), exist_ok=True)  # Create the image directory for each split
    os.makedirs(os.path.join(OUTPUT_PATH, "labels", split), exist_ok=True)  # Create the label directory for each split

# Get the list of all image files in the raw dataset
image_files = [f for f in os.listdir(os.path.join(DATASET_PATH, "JPEGImages")) if f.endswith(".jpg")]  # Get the list of image files
random.shuffle(image_files)  # Shuffle the list of image files

# Split the dataset into training, validation, and testing sets
total_files = len(image_files)  # Get the total number of files
train_count = int(total_files * SPLIT_RATIOS["train"])  # Calculate the number of training files
val_count = int(total_files * SPLIT_RATIOS["val"])  # Calculate the number of validation files
test_count = total_files - train_count - val_count  # Calculate the number of testing files

splits = {  # Define the splits
    "train": image_files[:train_count],  # Define the training split
    "val": image_files[train_count:train_count+val_count],  # Define the validation split
    "test": image_files[train_count+val_count:]  # Define the testing split
}

# Process each split
for split, files in splits.items():  # Iterate over the splits
    for file in files:  # Iterate over the files in each split
        # Process the image
        img_path = os.path.join(DATASET_PATH, "JPEGImages", file)  # Get the image path
        img = cv2.imread(img_path)  # Read the image
        original_height, original_width = img.shape[:2]  # Get the original image dimensions
        target_size = 640  # Define the target image size

        # Resize the image with aspect ratio maintained
        scale = min(target_size / original_width, target_size / original_height)  # Calculate the scaling factor
        new_width = int(original_width * scale)  # Calculate the new width
        new_height = int(original_height * scale)  # Calculate the new height
        resized_img = cv2.resize(img, (new_width, new_height))  # Resize the image

        # Pad the image to make it square
        padded_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)  # Create a padded image
        dx = (target_size - new_width) // 2  # Calculate the x offset
        dy = (target_size - new_height) // 2  # Calculate the y offset
        padded_img[dy:dy+new_height, dx:dx+new_width] = resized_img  # Pad the image
        cv2.imwrite(os.path.join(OUTPUT_PATH, "images", split, file), padded_img)  # Write the padded image

        # Process the annotations
        xml_path = os.path.join(DATASET_PATH, "Annotations", file.replace(".jpg", ".xml"))  # Get the annotation path
        tree = ET.parse(xml_path)  # Parse the annotation XML
        root = tree.getroot()  # Get the root element

        original_width_xml = int(root.find("size/width").text)  # Get the original width from the annotation
        original_height_xml = int(root.find("size/height").text)  # Get the original height from the annotation

        label_file = file.replace(".jpg", ".txt")  # Get the label file name
        with open(os.path.join(OUTPUT_PATH, "labels", split, label_file), "w") as f:  # Open the label file
            for obj in root.findall("object"):  # Iterate over the objects in the annotation
                cls_name = obj.find("name").text  # Get the class name
                if cls_name not in classes:  # Check if the class is valid
                    continue  # Skip if the class is not valid

                bbox = obj.find("bndbox")  # Get the bounding box
                xmin = int(bbox.find("xmin").text)  # Get the x minimum
                ymin = int(bbox.find("ymin").text)  # Get the y minimum
                xmax = int(bbox.find("xmax").text)  # Get the x maximum
                ymax = int(bbox.find("ymax").text)  # Get the y maximum

                # Adjust for scaling and padding
                scale = min(target_size / original_width_xml, target_size / original_height_xml)  # Calculate the scaling factor
                new_width_scaled = original_width_xml * scale  # Calculate the new width
                new_height_scaled = original_height_xml * scale  # Calculate the new height
                dx = (target_size - new_width_scaled) // 2  # Calculate the x offset
                dy_pad = (target_size - new_height_scaled) // 2  # Calculate the y offset

                scaled_xmin = xmin * scale  # Scale the x minimum
                scaled_xmax = xmax * scale  # Scale the x maximum
                scaled_ymin = ymin * scale  # Scale the y minimum
                scaled_ymax = ymax * scale  # Scale the y maximum

                padded_xmin = scaled_xmin + dx  # Pad the x minimum
                padded_xmax = scaled_xmax + dx  # Pad the x maximum
                padded_ymin = scaled_ymin + dy_pad  # Pad the y minimum
                padded_ymax = scaled_ymax + dy_pad  # Pad the y maximum

                # Convert to YOLO format
                x_center = ((padded_xmin + padded_xmax) / 2) / target_size  # Calculate the x center
                y_center = ((padded_ymin + padded_ymax) / 2) / target_size  # Calculate the y center
                width = (padded_xmax - padded_xmin) / target_size  # Calculate the width
                height = (padded_ymax - padded_ymin) / target_size  # Calculate the height

                f.write(f"{classes[cls_name]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")  # Write the label

# Generate data.yaml
data_yaml = f"""path: {OUTPUT_PATH}
train: images/train
val: images/val
test: images/test

names:
  0: RBC
  1: WBC
  2: Platelets
"""
with open(os.path.join(OUTPUT_PATH, "data.yaml"), "w") as f:  # Open the data.yaml file
    f.write(data_yaml)  # Write the data.yaml file

print("Preprocessing completed!")  # Print the completion message