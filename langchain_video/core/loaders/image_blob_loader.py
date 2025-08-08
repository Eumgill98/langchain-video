from __future__ import annotations
from .base import BaseMultiModalBlobLoader

from langchain_video.core.blobs import ImageBlob, PathLike

from typing import Union, List, Optional
from pathlib import Path
from collections.abc import Iterable

class ImageBlobLoader(BaseMultiModalBlobLoader):
    """Image blob loader implemetation."""

    def __init__(
        self,
        paths: Union[PathLike, List[Union[str, Path]]],
        *,
        suffixes: Optional[List[str]] = None
    ):
        if isinstance(paths, PathLike):
            paths = [paths]

        self.paths = self._convert2paths(paths)
        self.suffixes = suffixes or [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".svg"]

        self.suffixes = [
            suffix.lower() if suffix.startswith('.') else f'.{suffix.lower()}' 
            for suffix in self.suffixes
        ]
    def yield_blobs(self) -> Iterable[ImageBlob]:
        """A lazy loader for image data represented by ImageBlob objects.
        
        Returns:
            A generator over ImageBlobs
        """
        for path in self.paths:
            if not path.is_file():
                raise FileNotFoundError(f"Image file is not found: {path}")
            if path.suffix.lower() not in self.suffixes:
                raise ValueError(f"Unsupported Image file extenstion: {path.suffix}. Supported: {self.suffixes}")
            yield ImageBlob.from_path(path)