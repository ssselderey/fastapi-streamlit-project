from ultralytics import YOLO
import numpy as np

class YOLOFaceDetector:
    def __init__(self, model_path=r"C:\Users\327\Desktop\homeworks phase 3\fastapi-streamlit-project\backend\weights\best.pt"):
        self.model = YOLO(model_path)

    def predict(self, img):
        # inference
        results = self.model(img)

        res = results[0]  # first image result

        detections = []
        for box in res.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            detections.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "confidence": conf,
                "class": cls
            })

        return detections
