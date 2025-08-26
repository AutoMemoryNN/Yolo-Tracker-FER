from fer import FER  # type: ignore
import numpy as np

from core.detectors.types import EmotionDetector, EmotionResult


class FerEmotionDetector(EmotionDetector):
    def __init__(self, backend: str = "opencv") -> None:
        use_mtcnn = backend == "mtcnn"
        self.detector = FER(mtcnn=use_mtcnn)

    def predict(self, image: np.ndarray) -> EmotionResult:
        result = self.detector.top_emotion(image)
        if result is None:
            return {"emotion": None, "score": 0.0}
        emotion, score = result
        return {"emotion": emotion, "score": score}

    def get_available_emotions(self) -> list[str]:
        # Hardcoded list of available emotions
        return ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
