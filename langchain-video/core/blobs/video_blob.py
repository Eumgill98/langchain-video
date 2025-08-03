from __future__ import annotations

from typing import Optional, Union, List, Iterator, Literal, Dict, Any
from pathlib import PurePath
from langchain_core.documents.base import BaseMedia
import numpy as np
import mimetypes
import cv2

PathLike = Union[str, PurePath]
SamplingStrategy = Literal["uniform", "random", "first", "last", "all"]

class VideoBlob(BaseMedia):
    """VideoBlob represents raw video data by either reference or value.

    Provides an interface to materialize the video blob in different representations 
    (e.g., frames, audio track, metadata), and helps to decouple the development of 
    video data loaders from the downstream processing or analysis of the video content.
    """

    # Core data
    data: Optional[List[np.ndarray]] = None
    path: Optional[PathLike] = None
    mimetype: Optional[str] = None

    _codec: Optional[str] = None
    _total_frames: Optional[int] = None
    _fps: Optional[float] = None
    _height: Optional[int] = None
    _width: Optional[int] = None
    _duration_sec: Optional[float] = None

    @property
    def codec(self) -> Optional[str]:
        """Video codec information."""
        return self._codec
    
    @property
    def total_frames(self) -> Optional[str]:
        """Total number of frames in the video."""
        return self._total_frames

    @property
    def fps(self) -> Optional[str]:
        """Frames per second of the video."""
        return self._fps
    
    @property
    def height(self) -> Optional[int]:
        """Height of video frames in pixels."""
        return self._height
    
    @property
    def width(self) -> Optional[int]:
        """Width of video frames in pixels."""
        return self._width
    
    @property
    def duration_sec(self) -> Optional[float]:
        """Duration of the video in seconds."""
        return self.duration_sec
    
    @property
    def resolution(self) -> Optional[tuple[int, int]]:
        """Video resolution as (width, height) tuple."""
        if self.width is not None and self.height is not None:
            return (self.width, self.height)
        return None

    def _load_video_metadata(self) -> None:
        """Load video metatdata using OpenCV"""
        if self.path is None:
            raise ValueError("Cannot load metadata: video path is not set.")
        
        if not self.path.exists():
            raise FileNotFoundError(f"Video file not found: {self.path}")
        
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {self.path}")
        
        try:
            self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._fps = cap.get(cv2.CAP_PROP_FPS)
            self._height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            
            if self._fps > 0:
                self._duration_sec = self._total_frames / self._fps
            
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            self._codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            
        finally:
            cap.release()

    def as_frames(
        self,
        max_frames: Optional[int] = None,
        sampling_strategy: SamplingStrategy = "all"
    ) -> List[np.ndarray]:
        """Materialize video as a list of frame arrays.
        
        Args:
            max_frames: Maximum number of frames to extract (ignored if sampling_strategy="all")
            sampling_strategy: How to sample frames ("uniform", "random", "first", "last", "all")

        Returns:
            List of frames as numpy arrays
        """
        if self.data is not None:
            return self._apply_sampling_to_frames(self.data, max_frames, sampling_strategy)
        
        if self.path is None:
            raise ValueError("Cannot extract frames: no video data or path available")

        frames = self._extract_frames_from_file(max_frames, sampling_strategy)
        if sampling_strategy == "all" and max_frames is None:
            self.data = frames
        return frames
    
    def as_frame_iterator(
        self,
        batch_size: int = 10,
        max_frames: Optional[int] = None
    ) -> Iterator[List[np.ndarray]]:
        """Materialize video as an iterator of frame batches for memory efficiency.

        Args:
            batch_size: Number of frames per batch
            max_frames: Maximum total frames to process

        Yields:
            Batches of frames as lists of numpy arrays
        """
        if self.data is not None:
            frames_to_process = self.data[:max_frames] if max_frames else self.data
            
            for i in range(0, len(frames_to_process), batch_size):
                yield frames_to_process[i:i + batch_size]
            return
        
        if self.path is None:
            raise ValueError("Cannot iterate frames: no video data or path available")

        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {self.path}")

        try:
            batch = []
            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                batch.append(frame)
                frame_count += 1

                if len(batch) >= batch_size:
                    yield batch
                    batch = []

                if max_frames is not None and frame_count >= max_frames:
                    break

            # Yield remaining frames
            if batch:
                yield batch

        finally:
            cap.release()

    def _extract_frames_from_file(
        self,
        max_frames: Optional[int],
        sampling_strategy: SamplingStrategy
    ) -> List[np.ndarray]:
        """Extract frames from video file based on sampling strategy."""
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {self.path}")

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_indices = self._calculate_frame_indices(
                total_frames, max_frames, sampling_strategy
            )

            frames = []
            current_frame = 0
            target_idx = 0

            while target_idx < len(frame_indices) and current_frame < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                if current_frame == frame_indices[target_idx]:
                    frames.append(frame)
                    target_idx += 1

                current_frame += 1

            return frames

        finally:
            cap.release()

    def _calculate_frame_indices(
        self,
        total_frames: int,
        max_frames: Optional[int],
        sampling_strategy: SamplingStrategy
    ) -> List[int]:
        """Calculate frame indices based on sampling strategy."""
        if sampling_strategy == "all":
            return list(range(total_frames))
            
        if max_frames is None:
            return list(range(total_frames))

        max_frames = min(max_frames, total_frames)

        if sampling_strategy == "uniform":
            if max_frames >= total_frames:
                return list(range(total_frames))
            step = total_frames / max_frames
            return [int(i * step) for i in range(max_frames)]

        elif sampling_strategy == "first":
            return list(range(min(max_frames, total_frames)))

        elif sampling_strategy == "last":
            start = max(0, total_frames - max_frames)
            return list(range(start, total_frames))

        elif sampling_strategy == "random":
            indices = np.random.choice(total_frames, size=max_frames, replace=False)
            return sorted(indices.tolist())

        else:
            raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")

    def _apply_sampling_to_frames(
        self,
        frames: List[np.ndarray],
        max_frames: Optional[int],
        sampling_strategy: SamplingStrategy
    ) -> List[np.ndarray]:
        """Apply sampling strategy to already loaded frames."""
        if sampling_strategy == "all":
            return frames
            
        if max_frames is None or len(frames) <= max_frames:
            return frames

        indices = self._calculate_frame_indices(len(frames), max_frames, sampling_strategy)
        return [frames[i] for i in indices]

    @classmethod
    def from_path(
        cls,
        path: PathLike,
        *,
        mime_type: Optional[str] = None,
        guess_type: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> VideoBlob:
        """Create VideoBlob from file path.
        
        Args:
            path: Path to video file
            mime_type: MIME type override
            guess_type: Whether to guess MIME type from file extension
            metadata: Additional meatadata

        Returns:
            VideoBlob instance
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
        instance._load_video_metadata()
        return instance

    @classmethod
    def from_frames(
        cls,
        frames: List[np.ndarray],
        *,
        mime_type: Optional[str] = None, 
        fps: Optional[float] = None,
        codec: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> VideoBlob:
        """Create VideoBlob from frame data
        
        Args:
            frames: List of frame arrays
            fps: Frames per second (optional)
            metadata: Additional metadata

        Returns:
            VideoBlob instance
        """
        if not frames:
            raise ValueError("Frame list cannot be empty.")
        
        height, width = frames[0].shape[:2]

        return cls(
            data=frames,
            mimetype=mime_type,
            metadata=metadata if metadata is not None else {},
            _total_frames=len(frames),
            _fps=fps,
            _codec=codec,
            _height=height,
            _width=width,
            _duration_sec=len(frames) / fps if fps and fps > 0 else None,
        )
        