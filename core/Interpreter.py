from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time
import cv2
import numpy as np
from core.detectors.types import Person
from core.Vision import Vision


@dataclass
class TopEmotional:
    """Class to hold the top emotional scores for each person."""

    person_id: int
    emotion: str
    score: float
    # Do not include ROI content in equality to avoid numpy broadcasting errors
    frame: np.ndarray = field(compare=False)


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
    TOPS_UPDATE_TICKS: int = 50
    EMOTION_THRESHOLD: float = 0.82

    def __init__(self, vision: Vision) -> None:
        self.vision = vision
        self.fps = vision.get_fps()
        self.persons_records: deque[List[Person]] = deque(maxlen=50)
        self.avg_persons_emotion: dict[int, dict[str, float]] = {}
        self.current_frame: Optional[np.ndarray] = None
        self.last_persons: List[Person] = []  # snapshot de personas para el display

        self.persons_by_id: Dict[int, Person] = {}
        self.more_emotional_top: Dict[str, Optional[TopEmotional]] = {}
        self.threshold_top: Dict[str, Optional[TopEmotional]] = {}
        self._tick_counter: int = 0

    def update_avg_emotions(self) -> None:
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
            if self._tick_counter % 100 == 0:
                print(f"ID {pid} - Avg: {result}")

    def get_top_person(self, emotion: str) -> Optional[int]:
        if not self.avg_persons_emotion:
            return None

        emotion = emotion.lower()
        top_id = None
        top_score = -1.0

        for pid, emo_dict in self.avg_persons_emotion.items():
            if emotion in emo_dict and emo_dict[emotion] > top_score:
                top_id = pid
                top_score = emo_dict[emotion]

        # If there is no top or the current person is not visible, clear and exit
        if top_id is None:
            self.more_emotional_top[emotion] = None
            return None

        # Need a current frame and a visible person in this frame to snapshot
        if self.current_frame is None or top_id not in self.persons_by_id:
            self.more_emotional_top[emotion] = None
            return None

        person_frame = self._save_top_roi(
            self.current_frame, self.persons_by_id[top_id], emotion
        )

        if person_frame is None:
            self.more_emotional_top[emotion] = None
            return None

        self.more_emotional_top[emotion] = TopEmotional(
            person_id=top_id,
            emotion=emotion,
            score=top_score,
            frame=person_frame,
        )

        return top_id

    def update_top_persons_by_threshold(self, persons: List[Person]) -> None:
        for p in persons:
            if not p.emotion or not p.emotion.get("emotion"):
                continue

            emotion_value = p.emotion["emotion"]
            if emotion_value is None:
                continue

            emo_name = emotion_value.lower()
            score = float(p.emotion.get("score", 0.0))

            if score >= self.EMOTION_THRESHOLD:
                # Check if we already have a top for this emotion
                current_top = self.threshold_top.get(emo_name)

                # Only update if we don't have a top for this emotion or if the new score is higher
                if current_top is None or score > current_top.score:
                    if self.current_frame is None:
                        continue

                    roi, _ = self.vision._extract_roi(
                        self.current_frame,
                        p.face_bbox,
                        percentage_padding=self.PERCENTAGE_PADDING,
                        square=self.SQUARE_ROI,
                    )

                    if roi.size > 0:
                        self.threshold_top[emo_name] = TopEmotional(
                            person_id=p.id,
                            emotion=emo_name,
                            score=score,
                            frame=roi,
                        )

    def _save_top_roi(
        self, frame: np.ndarray, person: Person, emotion: str
    ) -> Optional[np.ndarray]:
        roi, _ = self.vision._extract_roi(
            frame,
            person.face_bbox,
            percentage_padding=self.PERCENTAGE_PADDING,
            square=self.SQUARE_ROI,
        )
        if roi.size == 0:
            return None

        return roi

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
        """Main loop: render at ~30 FPS and process every N ms; boxes can go at 3-5 FPS."""
        process_dt = 1.0 / 5.0  # ~5 FPS for logic; increase to 1/3.0 for ~3 FPS
        last_proc = 0.0
        try:
            while True:
                ret, frame = self.vision.camera.read()
                if not ret:
                    continue

                self.current_frame = frame
                now = time.monotonic()

                # process at slower rate
                do_process = (now - last_proc) >= process_dt
                if do_process:
                    persons = self.vision.process_frame(
                        frame,
                        percentage_padding=self.PERCENTAGE_PADDING,
                        square=self.SQUARE_ROI,
                    )

                    # update snapshots only when there's new processing
                    self.persons_by_id.clear()
                    self.persons_by_id.update({p.id: p for p in persons})
                    self.last_persons = persons[:]  # lightweight copy

                    self.persons_records.append(persons)
                    self.update_avg_emotions()
                    self.update_top_persons_by_threshold(persons)

                    last_proc = now

                # always render using the last available result
                if self.DISPLAY:
                    disp = frame.copy()
                    self.render(disp, self.last_persons)

                self._tick_counter += 1

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
        finally:
            if self.vision.camera is not None:
                self.vision.camera.release()
            cv2.destroyAllWindows()
