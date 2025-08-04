from __future__ import annotations

from typing import Optional, Union, List, Literal, Dict, Any
from pathlib import PurePath, Path
from langchain_core.documents.base import BaseMedia
import numpy as np
import mimetypes
import soundfile as sf

PathLike = Union[str, PurePath]

class AudioBlob(BaseMedia):
    """AudioBlob represents raw audio data by either reference or value.

    Provides an interface to materialize the audio blob in different representations 
    and helps to decouple the development of audio data loaders from the downstream 
    processing or analysis of the audio content.
    """

    # Core data
    data: Optional[np.ndarray] = None
    path: Optional[PathLike] = None
    mimetype: Optional[str] = None

    # Audio data
    _codec: Optional[str] = None
    _sample_rate: Optional[int] = None
    _channels: Optional[int] = None
    _bitrate: Optional[int] = None
    _duration_sec: Optional[float] = None
    _total_samples: Optional[int] = None
    _bit_depth: Optional[int] = None

    @property
    def codec(self) -> Optional[str]:
        """Audio codec information."""
        return self._codec
    
    @property
    def sample_rate(self) -> Optional[int]:
        """Sample rate of the audio in Hz."""
        return self._sample_rate

    @property
    def channels(self) -> Optional[int]:
        """Number of audio channels."""
        return self._channels
    
    @property
    def bitrate(self) -> Optional[int]:
        """Bitrate of the audio in bits per second."""
        return self._bitrate
    
    @property
    def duration_sec(self) -> Optional[float]:
        """Duration of the audio in seconds."""
        return self._duration_sec
    
    @property
    def total_samples(self) -> Optional[int]:
        """Total number of audio samples."""
        return self._total_samples
    
    @property
    def bit_depth(self) -> Optional[int]:
        """Bit depth of the audio."""
        return self._bit_depth
    
    @property
    def is_mono(self) -> Optional[bool]:
        """Check if audio is mono."""
        return self.channels == 1 if self.channels is not None else None
    
    @property
    def is_stereo(self) -> Optional[bool]:
        """Check if audio is stereo."""
        return self.channels == 2 if self.channels is not None else None

    def _load_audio_metadata(self) -> None:
        """Load audio metadata using soundfile"""
        if self.path is None:
            raise ValueError("Cannot load metadata: audio path is not set.")
        
        path_obj = Path(self.path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Audio file not found: {self.path}")
        try:
            info = sf.info(str(self.path))
            
            self._sample_rate = info.samplerate
            self._channels = info.channels
            self._total_samples = info.frames
            self._duration_sec = info.duration
            self._codec = info.format
            self._bit_depth = getattr(info.subtype_info, 'bits_per_sample', None)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load audio metadata: {e}")
        
    def as_audio(self, mono: bool = False) -> np.ndarray:
        """Get audio data as numpy array.

        Args:
            mono: Convert to mono if True
            
        Returns:
            Audio data as numpy array:
            - Mono: shape (n_samples,)
            - Multi-channel: shape (n_samples, n_channels)
        """
        if self.data is not None:
            audio = self.data.copy()

            if mono and audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            
            return audio
        
        if self.path is None:
            raise ValueError("Cannot extract audio: no audio data or path available")

        return self._extract_audio_from_file(mono)
    
    def _extract_audio_from_file(self, mono: bool = False) -> np.ndarray:
        """Extract audio from file using soundfile with original sample rate"""
        try:
            audio, sr = sf.read(str(self.path), dtype='float32')
            
            if mono and audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            
            return audio
            
        except Exception as e:
            raise RuntimeError(f"Audio extraction failed: {e}")

    def get_audio_info(self) -> Dict[str, Any]:
        """Get audio information as a dictionary"""
        return {
            'codec': self.codec,
            'sample_rate': self.sample_rate,
            'channels': self.channels,
            'bitrate': self.bitrate,
            'duration_sec': self.duration_sec,
            'total_samples': self.total_samples,
            'bit_depth': self.bit_depth,
            'is_mono': self.is_mono,
            'is_stereo': self.is_stereo,
            'audio_data_loaded': self.data is not None
        }
    
    @classmethod
    def from_path(
        cls,
        path: PathLike,
        *,
        mime_type: Optional[str] = None,
        guess_type: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AudioBlob:
        """Create AudioBlob from file path.
        
        Args:
            path: Path to audio file
            mime_type: MIME type override
            guess_type: Whether to guess MIME type from file extension
            metadata: Additional metadata

        Returns:
            AudioBlob instance
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
        instance._load_audio_metadata()
        return instance
    
    @classmethod
    def from_audio(
        cls,
        audio: np.ndarray,
        sample_rate: int,
        *,
        mime_type: Optional[str] = None,
        codec: Optional[str] = None,
        bitrate: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AudioBlob:
        """Create AudioBlob from numpy array.
        
        Args:
            data: Audio data as numpy array
            sample_rate: Sample rate in Hz
            mime_type: MIME type (optional)
            codec: Audio codec (optional)
            bitrate: Audio bitrate (optional)
            metadata: Additional metadata

        Returns:
            AudioBlob instance
        """
        if audio.ndim == 0 or (audio.ndim > 2):
            raise ValueError("Audio data must be 1D (mono) or 2D (multi-channel)")
        
        if sample_rate <= 0:
            raise ValueError("Sample rate must be greater than 0")

        channels = 1 if audio.ndim == 1 else audio.shape[1]
        total_samples = len(audio)
        duration_sec = total_samples / sample_rate

        return cls(
            data=audio,
            mimetype=mime_type,
            metadata=metadata if metadata is not None else {},
            _sample_rate=sample_rate,
            _channels=channels,
            _codec=codec,
            _bitrate=bitrate,
            _total_samples=total_samples,
            _duration_sec=duration_sec,
        )