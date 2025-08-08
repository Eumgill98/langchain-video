from __future__ import annotations

from typing import List
from collections.abc import Sequence, Iterable
from .base import BaseMultiModalBlobTransformer
from langchain_video.core.blobs import VideoBlob

class VideoBlobSpliter(BaseMultiModalBlobTransformer):
    """VideoBlob Spliter"""
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

    def split_blobs(self, blobs: Iterable[VideoBlob]) -> List[VideoBlob]:
        """
        Split each VideoBlob in the iterable into smaller VideoBlob chunks based on frame ranges.

        Args:
            blobs (Iterable[VideoBlob]): An iterable of VideoBlob instances to be split.

        Returns:
            List[VideoBlob]: A list of VideoBlob chunks, each containing a subset of frames from the originals,
                            with optional overlapping frames between chunks as specified by chunk_size and chunk_overlap.
        """
        result: List[VideoBlob] = []
        for blob in blobs:
            total_frames = blob.total_frames

            start = 0
            while start < total_frames:
                end = min(start + self._chunk_size, total_frames)
                chunk_blob = blob.get_frame_subclip_by_frame(start, end)

                if chunk_blob.metadata is None:
                    chunk_blob.metadata = {}
                chunk_blob.metadata["chunk_range"] = (start, end)

                result.append(chunk_blob)
                start += self._chunk_size - self._chunk_overlap
        return result

    def transform_blobs(self, blobs: Sequence[VideoBlob], **kwargs) -> Sequence[VideoBlob]:
        """Transform sequence of blobs by splitting them."""
        return self.split_blobs(list(blobs))
