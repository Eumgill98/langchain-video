from __future__ import annotations

from typing import Optional, Union, List, Literal, Dict, Any
from pathlib import PurePath, Path
from langchain_core.documents.base import BaseMedia
import numpy as np
import mimetypes
import cv2
import ffmpeg

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
    audio_data: Optional[List[np.ndarray]] = None
    path: Optional[PathLike] = None
    mimetype: Optional[str] = None

    # Video data
    _codec: Optional[str] = None
    _total_frames: Optional[int] = None
    _fps: Optional[float] = None
    _height: Optional[int] = None
    _width: Optional[int] = None
    _duration_sec: Optional[float] = None

    # Audio data
    _audio_codec: Optional[str] = None
    _sample_rate: Optional[int] = None
    _audio_channels: Optional[int] = None
    _audio_bitrate: Optional[int] = None
    _has_audio: Optional[bool] = None

    @property
    def codec(self) -> Optional[str]:
        """Video codec information."""
        return self._codec
    
    @property
    def total_frames(self) -> Optional[int]:
        """Total number of frames in the video."""
        return self._total_frames

    @property
    def fps(self) -> Optional[float]:
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
        return self._duration_sec
    
    @property
    def resolution(self) -> Optional[tuple[int, int]]:
        """Video resolution as (width, height) tuple."""
        if self.width is not None and self.height is not None:
            return (self.width, self.height)
        return None
    
    @property
    def audio_codec(self) -> Optional[str]:
        """Audio Codec information."""
        return self._audio_codec
    
    @property
    def sample_rate(self) -> Optional[int]:
        """Sample rate of audio"""
        return self._sample_rate
    
    @property
    def audio_channels(self) -> Optional[int]:
        """Audio channels"""
        return self._audio_channels

    @property
    def audio_bitrate(self) -> Optional[int]:
        """Audio bitrate"""
        return self._audio_bitrate
    
    @property
    def has_audio(self) -> Optional[bool]:
        """Check if video has audio"""
        return self._has_audio

    def _load_video_metadata(self) -> None:
        """Load video metadata using OpenCV and audio metadata using FFmpeg"""
        if self.path is None:
            raise ValueError("Cannot load metadata: video path is not set.")
        
        path_obj = Path(self.path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Video file not found: {self.path}")
        
        # Load video metadata using OpenCV
        self._load_video_metadata_opencv()
        
        # Load audio metadata using FFmpeg
        self._load_audio_metadata_ffmpeg()

    def _load_video_metadata_opencv(self) -> None:
        """Load video metadata using OpenCV"""
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

    def _load_audio_metadata_ffmpeg(self) -> None:
        """Load audio metadata using ffmpeg-python"""
        try:
            probe = ffmpeg.probe(str(self.path))
            
            audio_streams = [
                stream for stream in probe['streams']
                if stream['codec_type'] == 'audio'
            ]
            
            if audio_streams:
                self._has_audio = True
                audio_stream = audio_streams[0]
                
                self._audio_codec = audio_stream.get('codec_name')
                self._sample_rate = int(audio_stream.get('sample_rate', 0)) or None
                self._audio_channels = int(audio_stream.get('channels', 0)) or None
                self._audio_bitrate = int(audio_stream.get('bit_rate', 0)) or None
            else:
                self._has_audio = False
                self._audio_codec = None
                self._sample_rate = None
                self._audio_channels = None
                self._audio_bitrate = None
                
        except ffmpeg.Error as e:
            raise RuntimeError(f"FFprobe failed: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load audio metadata: {e}")

    def as_audios(self, mono: bool = False) -> Optional[np.ndarray]:
        """Extract audio data as numpy array with original sample rate.
        
        Args:
            mono: Convert to mono if True
            
        Returns:
            Audio data as numpy array at original sample rate:
            - Mono: shape (n_samples,)
            - Stereo/Multi: shape (n_samples, n_channels)
            - None if no audio
        """
        if self.audio_data is not None:
            audio = self.audio_data.copy()

            if mono and audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            
            return audio
        
        if self.path is None:
            raise ValueError("Cannot extract audio: no audio data or path available")
        
        if not self.has_audio:
            return None

        return self._extract_audio_from_file(mono)

    def _extract_audio_from_file(self, mono: bool = False) -> Optional[np.ndarray]:
        """Extract audio from video file using ffmpeg-python with original sample rate"""
        try:
            stream = ffmpeg.input(str(self.path))
            audio = stream.audio
            
            options = {}
            if mono:
                options['ac'] = 1
            
            out = ffmpeg.output(
                audio,
                'pipe:',
                format='f32le',
                acodec='pcm_f32le',
                **options
            )
            
            stdout, _ = ffmpeg.run(out, pipe_stdout=True, pipe_stderr=True, quiet=True)
            
            audio_data = np.frombuffer(stdout, dtype=np.float32)

            if not mono and self.audio_channels and self.audio_channels > 1:
                audio_data = audio_data.reshape(-1, self.audio_channels)
            
            return audio_data
            
        except ffmpeg.Error as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            raise RuntimeError(f"FFmpeg audio extraction failed: {error_msg}")
        except Exception as e:
            raise RuntimeError(f"Audio extraction failed: {e}")

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
        return frames
    
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

    def get_audio_info(self) -> Dict[str, Any]:
        """Get audio information as a dictionary"""
        return {
            'has_audio': self.has_audio,
            'audio_codec': self.audio_codec,
            'sample_rate': self.sample_rate,
            'audio_channels': self.audio_channels,
            'audio_bitrate': self.audio_bitrate,
            'audio_data_loaded': self.audio_data is not None
        }

    def get_video_info(self) -> Dict[str, Any]:
        """Get video information as a dictionary"""
        return {
            'codec': self.codec,
            'total_frames': self.total_frames,
            'fps': self.fps,
            'resolution': self.resolution,
            'duration_sec': self.duration_sec,
            'video_data_loaded': self.data is not None
        }

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
    
    @classmethod
    def from_frames_and_audio(
        cls,
        frames: List[np.ndarray],
        *,
        mime_type: Optional[str] = None, 
        fps: Optional[float] = None,
        codec: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        audios: Optional[np.ndarray] = None,
        audio_codec: Optional[str] = None,
        sample_rate: Optional[int] = None,
        audio_channels: Optional[int] = None,
        audio_bitrate: Optional[int] = None,
        has_audio: Optional[bool] = None,
    ) -> VideoBlob:
        """Create VideoBlob from frame data and audio data
        
        Args:
            frames: List of frame arrays
            fps: Frames per second (optional)
            codec: Video codec (optional)
            metadata: Additional metadata
            audios: Audio data as numpy array (optional)
            audio_codec: Audio codec (optional)
            sample_rate: Audio sample rate (optional)
            audio_channels: Number of audio channels (optional)
            audio_bitrate: Audio bitrate (optional)
            has_audio: Whether the video has audio (optional)

        Returns:
            VideoBlob instance
        """
        if not frames:
            raise ValueError("Frame list cannot be empty.")
        
        height, width = frames[0].shape[:2]

        if audios is not None:
            has_audio = True
            if audio_channels is None:
                audio_channels = 1 if audios.ndim == 1 else audios.shape[1]

        return cls(
            data=frames,
            audio_data=audios,
            mimetype=mime_type,
            metadata=metadata if metadata is not None else {},
            _total_frames=len(frames),
            _fps=fps,
            _codec=codec,
            _height=height,
            _width=width,
            _duration_sec=len(frames) / fps if fps and fps > 0 else None,
            _audio_codec=audio_codec,
            _sample_rate=sample_rate,
            _audio_channels=audio_channels,
            _audio_bitrate=audio_bitrate,
            _has_audio=has_audio,
        )