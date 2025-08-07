from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union

from langchain_video.core.blobs import VideoBlob, ImageBlob, AudioBlob

if TYPE_CHECKING:
    from collections.abc import Iterable

MultiModalBlob = Union[VideoBlob, ImageBlob, AudioBlob]

class BaseMultiModalBlobLoader(ABC):
    """Abstact interface for Multimodal blobs loaders implementation.
    
    Implementer should be able to load raw content from a storage system according
    to some criteria and return the raw content lazily as a stream of Multimodal blobs.
    """
    @abstractmethod
    def yield_blobs(
        self,
    ) -> Iterable[MultiModalBlob]:
        """A lazy loader for raw data represented by MultiModalBlobs object.

        Returns:
            A generator over MultiModalBlobs
        """