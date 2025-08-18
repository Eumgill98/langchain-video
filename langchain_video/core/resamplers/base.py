from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np

class  VideoResampler(ABC):
    """Video Frame Resampler abstract interface"""
    def __call__(
        self,
        frames: List[np.ndarray]
    ) -> List[np.ndarray]:
        
        if not frames:
            return []
        return (self.resample(frames))
    
    @abstractmethod
    def resample(
        self,
        frames: List[np.ndarray]
    ) -> List[np.ndarray]:
        msg = f"`resample` has not been implementd for {self.__class__.__name__} "
        raise NotImplementedError(msg)


