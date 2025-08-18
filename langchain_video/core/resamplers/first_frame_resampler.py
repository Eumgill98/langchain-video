from typing import List
from .base import VideoResampler

import numpy as np

class FirstFrameResampler(VideoResampler):
    """Extract the first frame of a video clip as a thumbnail."""

    def resample(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        if not frames:
            return []
        return [frames[0]]
