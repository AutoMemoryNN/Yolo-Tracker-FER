import cv2
import numpy as np
import time
from core.detectors.types import Person
from core.Vision import Vision


class Interpreter:
    """Owns a Vision instance; orchestrates loop and rendering.

    Visualization config lives as class-level globals. Update these attributes
    to configure how to display without changing the constructor.
    """

    # Global visualization flags
    SHOW_BBOX: bool = True
    SHOW_EMOTION: bool = True
    SHOW_TIME: bool = True
    WINDOW_NAME: str = "Detected Persons"
    DISPLAY: bool = True

    # ROI options forwarded to Vision.process_frame
    PERCENTAGE_PADDING: float = 0.1
    SQUARE_ROI: bool = True

    def __init__(self, vision: Vision) -> None:
        self.vision = vision
        self.fps = vision.get_fps()

    def render(self, frame: np.ndarray, persons: list[Person]) -> None:
        for person in persons:
            x1, y1, x2, y2 = map(int, person.face_bbox)
            if self.SHOW_BBOX:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if self.SHOW_EMOTION:
                label = "Unknown"
                if person.emotion and person.emotion.get("emotion") is not None:
                    emo = str(person.emotion.get("emotion"))
                    score = float(person.emotion.get("score", 0.0))
                    label = f"{emo} {score:.2f}"
                cv2.putText(
                    frame,
                    label,
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )

        if self.SHOW_TIME and hasattr(self.vision, "fps"):
            time_text = f"FPS: {self.vision.fps:.2f}"
            cv2.putText(
                frame,
                time_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        cv2.imshow(self.WINDOW_NAME, frame)

    def interpret(self) -> None:
        """Main loop: grab frames, run process_frame, then render based on flags."""
        try:
            while True:
                ret, frame = self.vision.camera.read()
                if not ret:
                    continue

                persons = self.vision.process_frame(
                    frame,
                    percentage_padding=self.PERCENTAGE_PADDING,
                    square=self.SQUARE_ROI,
                )

                if self.DISPLAY:
                    self.render(frame, persons)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
        finally:
            if self.vision.camera is not None:
                self.vision.camera.release()
            cv2.destroyAllWindows()
