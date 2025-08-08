from __future__ import annotations
from .base import BaseMultiModalBlobLoader

from langchain_video.core.blobs import AudioBlob, PathLike

from typing import Union, List, Optional
from pathlib import Path

if TYPE_CHECKING:
    from collections.abc import Iterable

class AudioBlobLoader(BaseMultiModalBlobLoader):
    """Audio blob loader implementation."""

    def __init__(
        self,
        paths: Union[PathLike, List[Union[str, Path]]],
        *,
        suffixes: Optional[List[str]] = None
    ):
        if isinstance(paths, PathLike):
            paths = [paths]

        self.paths = self._convert2paths(paths)
        self.suffixes = suffixes or [".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma"]

        self.suffixes = [
            suffix.lower() if suffix.startswith('.') else f'.{suffix.lower()}' 
            for suffix in self.suffixes
        ]

    def yield_blobs(self) -> Iterable[AudioBlob]:
        """A lazy loader for audio data represented by AudioBlob objects.

        Returns:
            A generator over AudioBlobs
        """
        for path in self.paths:
            if not path.is_file():
                raise FileNotFoundError(f"Audio file is not found: {path}.")
            if path.suffix.lower() not in self.suffixes:
                raise ValueError(f"Unsupported Audio file extension: {path.suffix}. Supported: {self.suffixes}")
            yield AudioBlob.from_path(path)