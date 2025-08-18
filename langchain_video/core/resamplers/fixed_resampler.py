from typing import List
from .base import VideoResampler

import numpy as np

class FixedCountResampler(VideoResampler):
    """Resamples frames to a fixed count, evenly spaced."""
    def __init__(self, count: int = 30):
        self.count = count

    def resample(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        if not frames or self.count <= 0:
            return []
        total = len(frames)
        if total <= self.count:
            return frames

        step = total / self.count
        indices = [int(step * i) for i in range(self.count)]
        return [frames[i] for i in indices]