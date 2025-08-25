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

    def process_frame(self, frame: cv2.typing.MatLike | None = None) -> list[Person]:
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
            x1, y1, x2, y2 = map(int, bbox)
            h, w = frame.shape[:2]
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(w, x2), min(h, y2)
            face = frame[y1c:y2c, x1c:x2c]
            emotion_result = (
                self.emotion_detector.predict(face) if face.size > 0 else None
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
