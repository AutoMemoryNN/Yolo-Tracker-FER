import cv2
from core.detectors.types import EmotionDetector, FaceDetector, Person


class Vision:
    """Handle all vision-related tasks, interpretation and processing of visual data."""

    def __init__(
        self,
        face_detector: FaceDetector,
        emotion_detector: EmotionDetector,
        camera_n: int = 0,
    ):
        self.face_detector = face_detector
        self.emotion_detector = emotion_detector
        self.persons_detected: list[Person] = []
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
    ) -> list[Person]:
        """Process a frame and update persons_detected.

        If frame is None, capture one from the camera. Returns the list of
        detected persons for the processed frame.
        """
        if frame is None:
            ret, frame = self.camera.read()
            if not ret:
                self.persons_detected = []
                return []

        face_bboxes = self.face_detector.detect(frame)

        current_persons: list[Person] = []

        person_id = 0
        for bbox in face_bboxes:
            face_roi, _ = self._extract_roi(
                frame,
                bbox,
                percentage_padding=percentage_padding,
                square=square,
            )

            emotion_result = (
                self.emotion_detector.predict(face_roi) if face_roi.size > 0 else None
            )
            person = Person(id=person_id, bbox=bbox, emotion=emotion_result)
            current_persons.append(person)
            person_id += 1

        self.persons_detected = current_persons
        return current_persons

    def show_detected_persons(self) -> None:
        """display video with current detections only.

        Draws boxes and emotions for the current frame only so old boxes
        disappear naturally, minimizing lag.
        Press 'q' to close the window.
        """
        try:
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    continue

                persons = self.process_frame(frame)

                for person in persons:
                    x1, y1, x2, y2 = map(int, person.bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    emotion_text = (
                        str(person.emotion)
                        if person.emotion and person.emotion.get("emotion") is not None
                        else "Unknown"
                    )
                    cv2.putText(
                        frame,
                        emotion_text,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2,
                    )

                cv2.imshow("Detected Persons", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
        finally:
            if self.camera is not None:
                self.camera.release()
            cv2.destroyAllWindows()
