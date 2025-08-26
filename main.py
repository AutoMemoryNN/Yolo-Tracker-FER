import sys
import threading
from core.Animator import Animator, MyWindow
from core.Interpreter import Interpreter
from core.Vision import Vision
from core.detectors.deep_emotion_detector import DeepFaceEmotionDetector
from core.detectors.fer_emotion_detector import FerEmotionDetector
from core.detectors.hug_yolo_face_detector import YOLOFaceDetector
from core.detectors.yolo_person_detector import YOLOPersonDetector
from core.trackers.centroid_track import CentroidTracker
from PySide6.QtWidgets import QApplication

if __name__ == "__main__":
    vision = Vision(
        person_detector=YOLOPersonDetector(),
        face_detector=YOLOFaceDetector(),
        emotion_detector=FerEmotionDetector(),
        tracker=CentroidTracker(),
        camera_n=0,
    )

    interpreter = Interpreter(vision)
    animator = Animator(interpreter)
    animator.start_windows()
