from core.detectors.types import EmotionDetector, FaceDetector


class Vision:
    """Handle all vision-related tasks, interpretation and processing of visual data."""

    def __init__(self, face_detector: FaceDetector, emotion_detector: EmotionDetector):
        self.face_detector = face_detector
        self.emotion_detector = emotion_detector
