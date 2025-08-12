from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union, List
from pathlib import Path

from langchain_video.core.blobs import VideoBlob, ImageBlob, AudioBlob, PathLike
from collections.abc import Iterable

MultiModalBlob = Union[VideoBlob, ImageBlob, AudioBlob]

class BaseMultiModalBlobLoader(ABC):
    """Abstact interface for Multimodal blobs loaders implementation.
    
    Implementer should be able to load raw content from a storage system according
    to some criteria and return the raw content lazily as a stream of Multimodal blobs.
    """
    
    def _convert2paths(
        self,
        paths: Union[PathLike, List[Union[str, Path]]],
    ) -> List[Path]:
        """Convert input path(s) to a list of pathlib.Path objects."""
        return [p if isinstance(p, Path) else Path(p) for p in paths]

    def __iter__(self):
        """Make directly iterable."""
        return self.yield_blobs()
    
    def __getitem__(self, index):
        blobs = list(self.yield_blobs())
        return blobs[index]

    @abstractmethod
    def yield_blobs(
        self,
    ) -> Iterable[MultiModalBlob]:
        """A lazy loader for raw data represented by MultiModalBlobs object.

        Returns:
            A generator over MultiModalBlobs
        """