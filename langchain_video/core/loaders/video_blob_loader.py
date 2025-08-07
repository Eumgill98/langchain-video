from __future__ import annotations
from .base import BaseMultiModalBlobLoader

from langchain_video.core.blobs import VideoBlob, PathLike

from typing import Union, List, Optional
from pathlib import Path

if TYPE_CHECKING:
    from collections.abc import Iterable

class VideoBlobLoader(BaseMultiModalBlobLoader):
    """Video blob loader implementation."""

    def __init__(
        self,
        paths: Union[PathLike, List[Union[str, Path]]],
        *,
        glob: str = "**/*",
        suffixes: Optional[List[str]] = None
    ):
        if isinstance(paths, PathLike):
            paths = [paths]

        self.paths = self._convert2paths(paths)
        self.suffixes = suffixes or [".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"]

        self.suffixes = [
            suffix.lower() if suffix.startswith('.') else f'.{suffix.lower()}' 
            for suffix in self.suffixes
        ]

    def yield_blobs(self) -> Iterable[VideoBlob]:
        """A lazy loader for video data represented by VideoBlob objects.

        Returns:
            A generator over VideoBlobs
        """
        for path in self.paths:
            if not path.is_file():
                raise FileNotFoundError(f"Video file is not found {path}.")
            if path.suffix.lower() not in self.suffixes:
                raise ValueError(f"Unsupported Video file extension: {path.suffix}. Supported: {self.suffixes}")
            yield VideoBlob.from_path(path)