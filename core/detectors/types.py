from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypedDict

import numpy as np


class EmotionResult(TypedDict):
    emotion: str | None
    score: float


@dataclass
class Person:
    id: int
    bbox: tuple[int, int, int, int]
    emotion: EmotionResult | None


class FaceDetector(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        pass


class EmotionDetector(ABC):
    @abstractmethod
    def predict(self, image: np.ndarray) -> EmotionResult:
        pass
