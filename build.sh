#!/bin/bash
set -o errexit

# Install dependencies
pip install -r requirements.txt

# Install gdown if not installed
pip install gdown

# Create necessary directories
mkdir -p scripts/blood_cell_detection/blood_cell_detection7/weights

# Download best.pt from Google Drive
gdown --id 1c2VAdw1AQnDg4mlK8l_e_NthdA1Jg_G3 -O scripts/blood_cell_detection/blood_cell_detection7/weights/best.pt
