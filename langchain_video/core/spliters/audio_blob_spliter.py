from __future__ import annotations

from typing import List
from collections.abc import Sequence, Iterable
from .base import BaseMultiModalBlobTransformer
from langchain_video.core.blobs import AudioBlob

class AudioBlobSpliter(BaseMultiModalBlobTransformer):
    """AudioBlob Spliter"""
    def __init__(
        self,
        chunk_size: int = 100,
        chunk_overlap: int = 20,
    ):
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
        if (chunk_overlap < 0):
            raise ValueError(f"chunk_overlap must be >= 0, got {chunk_overlap}")
        if (chunk_overlap > chunk_size):
            raise ValueError(f"Got a larger chunk overlap ({chunk_overlap}) than chunk size ({chunk_size}), should be smller")

        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
    
    def split_blobs(self, blobs: Iterable[AudioBlob]) -> List[AudioBlob]:
        """
        Split each AudioBlob in the iterable into smaller Audio chunks based on sample ranges.

        Args:
            blobs (Iterable[AudioBlob]): An iterable of AudioBlob instances to be split.

        Returns:
            List[AudioBlob]: A list of AudioBlob chunks, each containing a subset of samples from the originals,
                            with optional overlapping samples between chunks as specified by chunk_size and chunk_overlap.
        """
        result: List[AudioBlob] = []
        for blob in blobs:
            total_samples = blob.total_samples

            start = 0
            while start < total_samples:
                end = min(start + self._chunk_size, total_samples)
                chunk_blob = blob.get_audio_subclip_by_sample(start, end)

                if chunk_blob.metadata is None:
                    chunk_blob.metadata = {}
                chunk_blob.metadata["chunk_range"] = (start, end)

                result.append(chunk_blob)
                start += self._chunk_size - self._chunk_overlap
        return result
    
    def transform_blobs(self, blobs: Sequence[AudioBlob], **kwargs) -> Sequence[AudioBlob]:
        """Transform sequence of blobs by splitting them."""
        return self.split_blobs(list(blobs))
