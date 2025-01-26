from flask import Flask, render_template, request, send_from_directory
import os
import cv2
from ultralytics import YOLO

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "../app/static/uploads"
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}

# Load trained model
model = YOLO("../scripts/yolov8n.pt", task="detect")

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded", 400
            
        file = request.files["file"]
        if file.filename == "":
            return "No selected file", 400
            
        if file and allowed_file(file.filename):
            filename = file.filename
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            
            # Perform inference
            results = model.predict(filepath, imgsz=640)
            
            # Process results
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls_id = int(box.cls)
                    conf = float(box.conf)
                    class_name = model.names[cls_id]
                    detections.append({
                        "class": class_name,
                        "confidence": f"{conf:.2f}"
                    })
            
            # Save annotated image
            annotated_img = results[0].plot()
            annotated_filename = f"annotated_{filename}"
            annotated_path = os.path.join(app.config["UPLOAD_FOLDER"], annotated_filename)
            cv2.imwrite(annotated_path, cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
            
            return render_template("result.html",
                                 original=filename,
                                 annotated=annotated_filename,
                                 detections=detections)
    
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)