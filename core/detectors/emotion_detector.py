from fer import FER
import numpy as np


class EmotionDetector:
    def __init__(self, backend="opencv"):
        use_mtcnn = backend == "mtcnn"
        self.detector = FER(mtcnn=use_mtcnn)

    def predict(self, image: np.ndarray) -> dict:
        result = self.detector.top_emotion(image)
        if result is None:
            return {"emotion": None, "score": 0.0}
        emotion, score = result
        return {"emotion": emotion, "score": score}
