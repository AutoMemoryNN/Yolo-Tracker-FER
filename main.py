from core.Vision import Vision
from core.detectors.emotion_detector import FerEmotionDetector
from core.detectors.face_detector import YOLOFaceDetector
from core.detectors.person_detector import YOLOPersonDetector


if __name__ == "__main__":
    vision = Vision(
        person_detector=YOLOPersonDetector(),
        face_detector=YOLOFaceDetector(),
        emotion_detector=FerEmotionDetector(),
        camera_n=0,
    )
    vision.show_detected_persons()
