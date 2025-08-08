from __future__ import annotations
from .base import PathLike, SamplingStrategy

from typing import Optional, List, Dict, Any
from pathlib import Path
from pydantic import ConfigDict
from langchain_core.documents.base import BaseMedia
import numpy as np
import mimetypes
import cv2
import ffmpeg

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
    codec: Optional[str] = None
    total_frames: Optional[int] = None
    fps: Optional[float] = None
    height: Optional[int] = None
    width: Optional[int] = None
    duration_sec: Optional[float] = None

    # Audio data
    audio_codec: Optional[str] = None
    sample_rate: Optional[int] = None
    audio_channels: Optional[int] = None
    audio_bitrate: Optional[int] = None
    has_audio: Optional[bool] = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )
    
    @property
    def resolution(self) -> Optional[tuple[int, int]]:
        """Video resolution as (width, height) tuple."""
        if self.width is not None and self.height is not None:
            return (self.width, self.height)
        return None

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
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            
            if self.fps > 0:
                self.duration_sec = self.total_frames / self.fps
            
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            self.codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            
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
                self.has_audio = True
                audio_stream = audio_streams[0]
                
                self.audio_codec = audio_stream.get('codec_name')
                self.sample_rate = int(audio_stream.get('sample_rate', 0)) or None
                self.audio_channels = int(audio_stream.get('channels', 0)) or None
                self.audio_bitrate = int(audio_stream.get('bit_rate', 0)) or None
            else:
                self.has_audio = False
                self.audio_codec = None
                self.sample_rate = None
                self.audio_channels = None
                self.audio_bitrate = None
                
        except ffmpeg.Error as e:
            raise RuntimeError(f"FFprobe failed: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load audio metadata: {e}")

    def as_audios(self, mono: bool = False) -> Optional[List[np.ndarray]]:
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
    
    def get_duration(self) -> Optional[float]:
        return self.duration_sec
    
    def get_frame_count(self) -> Optional[int]:
        return self.total_frames

    def _frame2sec(self, frame: int):
        if self.fps is None:
            raise ValueError("fps (frames per second) is not defined.")
        return frame / self.fps

    def get_frame_subclip(
        self,
        start_sec: float,
        end_sec: float
    )  -> VideoBlob:
        """
        Extract a video subclip by time (seconds) and return a new VideoBlob instance.
        
        Returns:
            new VideoBlob
        """ 
        if self.fps is None:
            raise ValueError("FPS is required to extract frame subclip by time.")
        
        frames = self.data if self.data is not None else self.as_frames()
        
        start_frame = int(start_sec * self.fps)
        end_frame = int(end_sec * self.fps)

        # extract audio data 
        audios = self._get_audio_subclip(start_sec, end_sec)
        
        return VideoBlob.from_frames_and_audio(
            frames=frames[start_frame:end_frame],
            mime_type=self.mimetype,
            fps=self.fps,
            codec=self.codec,
            metadata=self.metadata,
            path=self.path,
            audios=audios,
            audio_codec=self.audio_codec,
            sample_rate=self.sample_rate,
            audio_channels=self.audio_channels,
            audio_bitrate=self.audio_bitrate,
            has_audio=self.has_audio,
        )
    
    def get_frame_subclip_by_frame(
        self,
        start_frame: int,
        end_frame: int
    ) -> VideoBlob:
        """
        Extract a video subclip by frame and return a new VideoBlob instance.
        
        Returns:
            new VideoBlob
        """
        frames = self.data if self.data is not None else self.as_frames()

        # extract audios
        audios = self._get_audio_subclip(self._frame2sec(start_frame), self._frame2sec(end_frame))

        return VideoBlob.from_frames_and_audio(
            frames=frames[start_frame:end_frame],
            mime_type=self.mimetype,
            fps=self.fps,
            codec=self.codec,
            metadata=self.metadata,
            path=self.path,
            audios=audios,
            audio_codec=self.audio_codec,
            sample_rate=self.sample_rate,
            audio_channels=self.audio_channels,
            audio_bitrate=self.audio_bitrate,
            has_audio=self.has_audio,
        )

    def _get_audio_subclip(
        self,
        start_sec: float,
        end_sec: float,
    ) -> Optional[List[np.ndarray]]:
        """
        Helper method to extract audio subclip by time (seconds) and returns List of np.ndarray
        
        Returns:
            List of np.ndarray representing the audio samples in the given time range,
            or None if no audio data exists.
        """

        if not self.has_audio:
            return None
        
        if self.sample_rate is None:
            raise ValueError("sample rate is required to extract audio subclip by time.")

        audios = self.audio_data if self.audio_data is not None else self.as_audios()

        start_sample = int(start_sec * self.sample_rate)
        end_sample = int(end_sec * self.sample_rate)

        return audios[start_sample : end_sample]
    
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
        path: Optional[str] = None,
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
            path=path,
            mimetype=mime_type,
            metadata=metadata if metadata is not None else {},
            total_frames=len(frames),
            fps=fps,
            codec=codec,
            height=height,
            width=width,
            duration_sec=len(frames) / fps if fps and fps > 0 else None,
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
        path: Optional[str] = None,
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
            total_frames=len(frames),
            fps=fps,
            codec=codec,
            height=height,
            width=width,
            duration_sec=len(frames) / fps if fps and fps > 0 else None,
            audio_codec=audio_codec,
            sample_rate=sample_rate,
            audio_channels=audio_channels,
            audio_bitrate=audio_bitrate,
            has_audio=has_audio,
            path=path,
        )
    