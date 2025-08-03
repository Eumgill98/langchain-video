from __future__ import annotations

from typing import Optional, Union, List, Dict, Any
from pathlib import PurePath, Path
from langchain_core.documents.base import BaseMedia
import numpy as np
import mimetypes
import cv2

PathLike = Union[str, PurePath]

class ImageBlob(BaseMedia):
    """ImageBlob represents raw image data by either reference or value.

    Provides an interface to materialize the image blob in different representations 
    (e.g., pixel array, color space, metadata), and helps to decouple the development of 
    image data loaders from the downstream processing or analysis of the image content.
    """

    # Core data
    data: Optional[List[np.ndarray]] = None
    path: Optional[PathLike] = None
    mimetype: Optional[str] = None

    # Image data
    _color_space: Optional[str] = None
    _height: Optional[int] = None
    _width: Optional[int] = None
    _channels: Optional[int] = None

    @property
    def color_space(self) -> Optional[str]:
        """Image color space"""
        return self._color_space
    
    @property
    def height(self) -> Optional[int]:
        """Height of image"""
        return self._height
    
    @property
    def width(self) -> Optional[int]:
        """Width of image"""
        return self._width
    
    @property
    def resolution(self) -> Optional[tuple[int, int]]:
        """Image resolution as (width, height) tuple."""
        if self.width is not None and self.height is not None:
            return (self.width, self.height)
        return None
    
    @property
    def channels(self) -> Optional[int]:
        """Channel of image"""
        return self._channels
    
    def _load_image_metadata(self) -> None:
        """Load image metadata using OpenCV"""
        if self.path is None:
            raise ValueError("Cannot load metadata: image path is not set.")
        
        path_obj = Path(self.path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Image file not found: {self.path}")
        
        img = cv2.imread(str(self.path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"Cannot read image file: {self.path}")
        
        self._height, self._width = img.shape[:2]
        self._channels = 1 if len(img.shape) == 2 else img.shape[2]

        self._color_space = "BGR"

    def as_image(self) -> Optional[np.ndarray]:
        """Extract image data as numpy array

        Returns:
            Image data as numpy array
        """
        if self.data is not None:
            img = self.data.copy()
            return (img)
        
        if self.path is None:
            raise ValueError("Cannot extract image: no image data or path available")
        
        path_obj = Path(self.path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Image file not found: {self.path}")
        img = cv2.imread(str(self.path), cv2.IMREAD_UNCHANGED)

        return img

    def get_image_info(self) -> Dict[str, Any]:
        """Get image information as a dictionary"""
        return {
            'color_space': self.color_space,
            'height': self.height,
            'width': self.width,
            'channels': self.channels,
            'image_data_loaded': self.data is not None
        }
    
    @classmethod
    def from_path(
        cls,
        path: PathLike,
        *,
        mime_type: Optional[str] = None,
        guess_type: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ImageBlob:
        """Create ImageBlob from file path.

        Args:
            path: Path to image file
            mime_type: MIME type override
            guess_type: Whether to guess MIME type from file extension
            metadata: Additional metadata

        Returns:
            ImageBlob instance
        """
        if mime_type is None and guess_type:
            mimetype = mimetypes.guess_type(path)[0] if guess_type else None
        else:
            mimetype = mime_type
        
        instance = cls(
            data=None,
            mimetype=mimetype,
            path=path,
            metadata=metadata if metadata is not None else {},
        )
        instance._load_image_metadata()
        return instance
    
    @classmethod
    def from_image(
        cls,
        image: np.ndarray,
        *,
        mime_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ImageBlob:
        """Create ImageBlob from image data

        Args:
            image: Numpy arrays of image
            metadata: Additional metadata

        Returns:
            ImageBlob instance
        """
        if not image:
            raise ValueError("Image cannot be empty.")
        
        height, width = image.shape[:2]
        channels = 1 if image.ndim == 2 else image.shape[2]
        color_space = "Grayscale" if channels == 1 else ("BGRA" if channels == 4 else "BGR")

        return cls(
            data=image,
            mimetypes=mime_type,
            metadata=metadata if metadata is not None else {},
            _height=height,
            _width=width,
            _color_space=color_space,
            _channels=channels
        )