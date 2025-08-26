from collections import deque, defaultdict
from typing import Dict, List, Optional
import cv2
import numpy as np
from core.detectors.types import Person
from core.Vision import Vision


class Interpreter:
    """Owns a Vision instance; orchestrates loop and rendering."""

    # Global visualization flags
    SHOW_BBOX: bool = True
    SHOW_EMOTION: bool = True
    SHOW_ID: bool = True
    SHOW_FPS: bool = True
    WINDOW_NAME: str = "Detected Persons"
    DISPLAY: bool = True

    # ROI options forwarded to Vision.process_frame
    PERCENTAGE_PADDING: float = 0.1
    SQUARE_ROI: bool = True

    MAX_PERSONS_IN_RECORD: int = 50

    def __init__(self, vision: Vision) -> None:
        self.vision = vision
        self.fps = vision.get_fps()
        self.persons_records: deque[List[Person]] = deque(maxlen=50)
        self.avg_persons_emotion: dict[int, dict[str, float]] = {}

    def update_avg_emotions(self) -> None:
        """Calcula el promedio de emociones por persona (requiere cola llena)."""
        if (
            self.persons_records.maxlen is None
            or len(self.persons_records) < self.persons_records.maxlen
        ):
            # The queue is not full yet, do nothing
            return

        # Temporary dictionary to accumulate emotions values and counts of occurrences by person
        sums: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        counts: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # Iterate persons and accumulate their emotions over MAX_PERSONS_IN_RECORD frames
        for persons in self.persons_records:
            for p in persons:
                if p.emotion and p.emotion.get("emotion"):
                    emotion_value = p.emotion["emotion"]
                    if emotion_value is not None:
                        emo_name = emotion_value.lower()
                        score = float(p.emotion.get("score", 0.0))
                        sums[p.id][emo_name] += score
                        counts[p.id][emo_name] += 1

        # Calculate averages
        self.avg_persons_emotion.clear()
        for pid, emo_sums in sums.items():
            result: Dict[str, float] = {}
            for emo_name, total in emo_sums.items():
                n = counts[pid][emo_name]
                if n > 0:
                    result[emo_name] = total / n
            self.avg_persons_emotion[pid] = result
            print(f"ID {pid} - Promedios: {result}")

    def get_top_person(self, emotion: str) -> Optional[int]:
        """Returns the ID of the person with the highest average in a given emotion."""
        if not self.avg_persons_emotion:
            return None

        emotion = emotion.lower()
        top_id = None
        top_score = -1.0
        for pid, emo_dict in self.avg_persons_emotion.items():
            if emotion in emo_dict and emo_dict[emotion] > top_score:
                top_id = pid
                top_score = emo_dict[emotion]

        return top_id

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
                    label = f"{emo}{score:.2f}"
                cv2.putText(
                    frame,
                    label,
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_ITALIC,
                    0.6,
                    (0, 255, 0),
                    2,
                )
            if self.SHOW_ID:
                cv2.putText(
                    frame,
                    f"ID: {person.id}",
                    (x1, max(0, y1 - 30)),
                    cv2.FONT_ITALIC,
                    0.6,
                    (0, 255, 0),
                    2,
                )

        if self.SHOW_FPS and self.vision.get_fps() is not None:
            time_text = f"FPS: {self.vision.get_fps():.2f}"
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

                self.persons_records.append(persons)
                self.update_avg_emotions()

                # Safe lookup by ID in current frame (IDs are not list indices)
                persons_by_id = {p.id: p for p in persons}

                top_happy_id = self.get_top_person("happy")
                if top_happy_id is None:
                    print("more happy: no top yet (no 'happy' data)")
                elif top_happy_id in persons_by_id:
                    print(f"more happy {persons_by_id[top_happy_id]}")
                else:
                    print(f"more happy: top ID {top_happy_id} not visible this frame")

                top_sad_id = self.get_top_person("sad")
                if top_sad_id is None:
                    print("more sad: no top yet (no 'sad' data)")
                elif top_sad_id in persons_by_id:
                    print(f"more sad {persons_by_id[top_sad_id]}")
                else:
                    print(f"more sad: top ID {top_sad_id} not visible this frame")

                if self.DISPLAY:
                    self.render(frame, persons)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
        finally:
            if self.vision.camera is not None:
                self.vision.camera.release()
            cv2.destroyAllWindows()
