from huggingface_hub import hf_hub_download
from ultralytics import YOLO

from core.detectors.types import FaceDetector


class YOLOFaceDetector(FaceDetector):
    def __init__(self, model_path="\\models\\hug_yolov11n.pt", conf=0.5):
        # Cargamos modelo YOLO
        self.model = YOLO(
            hf_hub_download(
                repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt"
            )
        )
        self.conf = conf

    def detect(self, frame):
        """
        Recibe un frame (numpy array BGR) y devuelve las bounding boxes de caras detectadas.
        Por ahora detecta 'person' y devuelve sus coordenadas, más adelante afinamos para cara/mano.
        """
        results = self.model.predict(frame, conf=self.conf)
        boxes = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                if cls_id == 0:  # 'person' en COCO dataset
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    boxes.append((x1, y1, x2, y2))
        return boxes
