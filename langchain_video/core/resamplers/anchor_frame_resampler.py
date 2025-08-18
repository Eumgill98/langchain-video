from typing import List
from .base import VideoResampler

import numpy as np

class AnchorFrameResampler(VideoResampler):
    """Extracts anchor frames: first, middle, and last frame of a video clip."""

    def resample(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        if not frames:
            return []
        total = len(frames)
        first = frames[0]
        middle = frames[total // 2]
        last = frames[-1]
        return [first, middle, last]