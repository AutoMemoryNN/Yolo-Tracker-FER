import numpy as np
from typing import Dict, List, Set
from core.trackers.types import Rect, TrackMap, Tracker


class CentroidTracker(Tracker):
    """Centroid-based object tracker that assigns consistent IDs to detected faces."""

    def __init__(self, max_disappeared: int = 20) -> None:
        self.next_id: int = 0
        self.objects: Dict[int, np.ndarray] = {}  # id -> centroid
        self.disappeared: Dict[int, int] = {}  # id -> frames missing
        self.max_disappeared: int = max_disappeared
        self.new_faces_in_frame: Set[int] = (
            set()
        )  # Track new faces detected in current frame

    def register(self, centroid: np.ndarray) -> int:
        """Register a new object and return its ID."""
        object_id = self.next_id
        self.objects[object_id] = centroid
        self.disappeared[object_id] = 0
        self.new_faces_in_frame.add(object_id)  # Mark as new face
        self.next_id += 1
        print(f"🆕 NUEVO ROSTRO DETECTADO - Asignado ID: {object_id}")
        return object_id

    def deregister(self, object_id: int) -> None:
        """Remove an object from tracking."""
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.disappeared:
            del self.disappeared[object_id]

    def update(self, rects: List[Rect]) -> TrackMap:
        """Update tracker with new detections and return current object centroids."""
        # Clear new faces from previous frame
        self.new_faces_in_frame.clear()

        # If no detections, mark existing objects as disappeared
        if len(rects) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return self.objects

        # Compute input centroids from bounding boxes
        input_centroids: np.ndarray = np.zeros((len(rects), 2), dtype="int")
        for i, (x1, y1, x2, y2) in enumerate(rects):
            cX: int = int((x1 + x2) / 2.0)
            cY: int = int((y1 + y2) / 2.0)
            input_centroids[i] = (cX, cY)

        # If no existing objects, register all input centroids
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
        else:
            # Match existing objects with new detections
            object_ids: List[int] = list(self.objects.keys())
            object_centroids: List[np.ndarray] = list(self.objects.values())

            # Compute distance matrix between existing and new centroids
            D: np.ndarray = np.linalg.norm(
                np.array(object_centroids)[:, None] - input_centroids[None, :], axis=2
            )

            # Hungarian algorithm approximation for assignment
            rows: np.ndarray = D.min(axis=1).argsort()
            cols: np.ndarray = D.argmin(axis=1)[rows]

            used_rows: Set[int] = set()
            used_cols: Set[int] = set()

            # Update matched objects
            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                object_id: int = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            # Handle unmatched existing objects
            unused_rows: Set[int] = set(range(D.shape[0])) - used_rows
            for row in unused_rows:
                unused_object_id: int = object_ids[row]
                self.disappeared[unused_object_id] += 1
                if self.disappeared[unused_object_id] > self.max_disappeared:
                    self.deregister(unused_object_id)

            # Register new detections
            unused_cols: Set[int] = set(range(D.shape[1])) - used_cols
            for col in unused_cols:
                self.register(input_centroids[col])

        return self.objects

    def is_new_face(self, tracking_id: int) -> bool:
        """Check if a tracking ID corresponds to a newly detected face in current frame."""
        return tracking_id in self.new_faces_in_frame

    def get_new_faces_count(self) -> int:
        """Get number of new faces detected in current frame."""
        return len(self.new_faces_in_frame)

    def get_assignments(self, rects: List[Rect]) -> Dict[int, int]:
        """Get mapping from detection index to tracking ID."""
        if len(rects) == 0:
            return {}

        assignments = {}
        input_centroids: np.ndarray = np.zeros((len(rects), 2), dtype="int")
        for i, (x1, y1, x2, y2) in enumerate(rects):
            cX: int = int((x1 + x2) / 2.0)
            cY: int = int((y1 + y2) / 2.0)
            input_centroids[i] = (cX, cY)

        if len(self.objects) == 0:
            return assignments

        object_ids: List[int] = list(self.objects.keys())
        object_centroids: List[np.ndarray] = list(self.objects.values())

        # Find closest centroid for each detection
        for i, input_centroid in enumerate(input_centroids):
            distances = [
                np.linalg.norm(input_centroid - obj_centroid)
                for obj_centroid in object_centroids
            ]
            min_idx = np.argmin(distances)
            assignments[i] = object_ids[min_idx]

        return assignments
