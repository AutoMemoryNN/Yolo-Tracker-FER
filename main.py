from core.Vision import Vision
from core.detectors.deep_emotion_detector import DeepFaceEmotionDetector
from core.detectors.fer_emotion_detector import FerEmotionDetector
from core.detectors.hug_yolo_face_detector import YOLOFaceDetector
from core.detectors.yolo_person_detector import YOLOPersonDetector


if __name__ == "__main__":
    vision = Vision(
        person_detector=YOLOPersonDetector(),
        face_detector=YOLOFaceDetector(),
        emotion_detector=FerEmotionDetector(),
        camera_n=0,
    )
    vision.show_detected_persons()
