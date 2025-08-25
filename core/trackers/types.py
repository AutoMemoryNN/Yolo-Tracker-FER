from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, TypedDict

import numpy as np


# Basic aliases for tracker IO types
Rect = Tuple[int, int, int, int]  # (x1, y1, x2, y2)
Centroid = np.ndarray  # shape (2,), dtype int
TrackMap = Dict[int, Centroid]  # id -> centroid


class TrackState(TypedDict, total=False):
    """Optional richer state for a track.

    Not used by CentroidTracker directly, but available for other trackers.
    """

    id: int
    bbox: Rect
    centroid: Tuple[int, int]


class Tracker(ABC):
    """Abstract tracker interface for frame-by-frame updates."""

    @abstractmethod
    def update(self, rects: List[Rect]) -> TrackMap:
        """Update tracker with current detections and return current tracks.

        rects: list of (x1, y1, x2, y2) detections for the current frame.
        Returns a mapping of track_id to centroid.
        """
        raise NotImplementedError

    @abstractmethod
    def register(self, centroid: Centroid) -> None:
        """Register a new track given a centroid."""
        raise NotImplementedError

    @abstractmethod
    def deregister(self, object_id: int) -> None:
        """Remove a track by id when it has disappeared for too long."""
        raise NotImplementedError
