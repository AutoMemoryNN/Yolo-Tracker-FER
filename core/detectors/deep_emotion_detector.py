from deepface import DeepFace
import numpy as np
from core.detectors.types import EmotionDetector, EmotionResult


class DeepFaceEmotionDetector(EmotionDetector):
    def predict(self, image: np.ndarray) -> EmotionResult:
        try:
            result = DeepFace.analyze(
                image, actions=["emotion"], enforce_detection=False
            )
            emotion = result[0]["dominant_emotion"]
            score = result[0]["emotion"][emotion] / 100.0  # DeepFace da %
            return {"emotion": emotion, "score": score}
        except Exception:
            return {"emotion": None, "score": 0.0}
