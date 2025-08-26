import time
import cv2
from core.detectors.types import (
    EmotionDetector,
    FaceDetector,
    Person,
    PersonDetector,
    EmotionResult,
)
from core.trackers.types import Tracker

import os

os.environ["YOLO_VERBOSE"] = "False"


class Vision:
    """Handle all vision-related tasks, interpretation and processing of visual data."""

    last_time: float = 0.0

    def __init__(
        self,
        person_detector: PersonDetector,
        face_detector: FaceDetector,
        emotion_detector: EmotionDetector,
        tracker: Tracker | None = None,
        camera_n: int = 0,
    ):
        self.person_detector = person_detector
        self.face_detector = face_detector
        self.emotion_detector = emotion_detector
        self.tracker = tracker
        self.face_detected: list[Person] = []
        self.camera = cv2.VideoCapture(camera_n)

    def _extract_roi(
        self,
        frame: cv2.typing.MatLike,
        bbox: tuple[int, int, int, int],
        percentage_padding: float = 0.0,
        square: bool = False,
    ) -> tuple[cv2.typing.MatLike, tuple[int, int, int, int]]:
        """Return a cropped ROI and adjusted bbox with optional padding and square shape.

        - percentage_padding is applied relative to the bbox size (width/height).
        - If square=True, we expand to a square around the bbox center using
          max(width, height) and then apply padding.
        - Always clamps to frame bounds.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)

        if square:
            side = max(bw, bh)
            side = int(round(side * (1.0 + percentage_padding)))
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            half = side / 2.0
            nx1 = int(round(cx - half))
            ny1 = int(round(cy - half))
            nx2 = int(round(cx + half))
            ny2 = int(round(cy + half))
        else:
            pad_w = int(round(bw * percentage_padding / 2.0))
            pad_h = int(round(bh * percentage_padding / 2.0))
            nx1 = x1 - pad_w
            ny1 = y1 - pad_h
            nx2 = x2 + pad_w
            ny2 = y2 + pad_h

        # Clamp to frame bounds
        nx1 = max(0, nx1)
        ny1 = max(0, ny1)
        nx2 = min(w, nx2)
        ny2 = min(h, ny2)

        roi = frame[ny1:ny2, nx1:nx2]
        return roi, (nx1, ny1, nx2, ny2)

    def process_frame(
        self,
        frame: cv2.typing.MatLike | None = None,
        percentage_padding: float = 0.0,
        square: bool = False,
        use_tracking: bool = True,
    ) -> list[Person]:
        """Process a frame and update persons_detected.

        If frame is None, capture one from the camera. Returns the list of
        detected persons for the processed frame.
        """
        start_time = time.time()

        if frame is None:
            ret, frame = self.camera.read()
            if not ret:
                self.face_detected = []
                return []

        # Detect faces
        face_bboxes = self.face_detector.detect(frame)
        _ = self.person_detector.detect(frame)  # reserve

        # Update tracker with detected faces
        if use_tracking and self.tracker:
            # Update tracker and get assignments
            self.tracker.update(face_bboxes)
            assignments = (
                self.tracker.get_assignments(face_bboxes)
                if hasattr(self.tracker, "get_assignments")
                else {}
            )
        else:
            assignments = {i: i for i in range(len(face_bboxes))}

        # Sequential emotion prediction (no threads)
        current_persons: list[Person] = []

        for detection_idx, bbox in enumerate(face_bboxes):
            face_roi, adj_bbox = self._extract_roi(
                frame,
                bbox,
                percentage_padding=percentage_padding,
                square=square,
            )

            # Get tracking ID for this detection
            tracking_id = assignments.get(detection_idx, detection_idx)

            # Default emotion result
            emotion_result: EmotionResult | None = None

            if face_roi.size > 0:
                try:
                    # Predict emotion sequentially
                    emotion_result = self.emotion_detector.predict(face_roi.copy())
                except Exception:
                    emotion_result = None

            # Append person with computed emotion
            current_persons.append(
                Person(id=tracking_id, face_bbox=adj_bbox, emotion=emotion_result)
            )

        self.face_detected = current_persons
        self.last_time = time.time() - start_time

        return current_persons

    def get_fps(self) -> float:
        """Get the processing FPS based on last frame processing time."""
        return 1.0 / self.last_time if self.last_time > 0 else 0.0

    def release(self) -> None:
        """Release camera resources."""
        if self.camera:
            self.camera.release()
