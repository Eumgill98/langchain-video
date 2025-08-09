from __future__ import annotations

from abc import ABC, abstractmethod, Any
from collections.abc import Sequence
from langchain_video.core.loaders.base import MultiModalBlob 
from langchain_core.runnables.config import run_in_executor

class BaseMultiModalBlobTransformer(ABC):
    """Abstract base class for multimodal blob transformation.
    
    A multimodal blob transformation takes a sequence of multimodal blob 
    and returns a sequence of transformed multimodal blobs
    """

    @abstractmethod
    def transform_blobs(
        self,
        blobs: Sequence[MultiModalBlob],
        **kwargs: Any
    ) -> Sequence[MultiModalBlob]:
        """Transform a list of blobs.

        Args:
            blobs: A sequence of Blobs to be transformed.

        Returns:
            A sequence of transformed Blobs.
        """
    
    async def atransform_blobs(
        self,
        blobs: Sequence[MultiModalBlob],
        **kwargs: Any
    ) -> Sequence[MultiModalBlob]:
        """Asynchronously transform a list of blobs.
        
        Args:
            blobs: A sequence of Blobs to be transformed.

        Returns:
            A sequence of transformed Blos.
        """
        return await run_in_executor(
            None, self.transform_blob, blobs, **kwargs
        )