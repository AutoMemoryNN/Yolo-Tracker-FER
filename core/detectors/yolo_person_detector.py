import numpy as np
from ultralytics import YOLO
from core.detectors.types import PersonDetector


class YOLOPersonDetector(PersonDetector):
    def __init__(self, model_path="\\models\\yolov8n.pt", conf=0.5):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        results = self.model.predict(frame, conf=self.conf, verbose=False)
        boxes = []

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                if cls_id == 0:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    boxes.append((int(x1), int(y1), int(x2), int(y2)))

        return boxes
