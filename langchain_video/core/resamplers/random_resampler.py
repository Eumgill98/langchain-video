from typing import List
from .base import VideoResampler

import numpy as np

class RandomResampler(VideoResampler):
    """Randomly samples frames from a video clip."""
    def __init__(self, num_samples: int = 30):
        self.num_samples = num_samples

    def resample(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        if not frames or self.num_samples <= 0:
            return []
        total = len(frames)
        if self.num_samples >= total:
            return frames
        indices = self.rng.choice(total, size=self.num_samples, replace=False)
        return [frames[i] for i in sorted(indices)]