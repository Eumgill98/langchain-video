from PIL import Image
import io

from typing import List, Tuple, Any, Callable
import numpy as np

import torch
import torchaudio

from langchain_video.core.embeddings import TextEmbeddings, ImageEmbeddings, AudioEmbeddings

from langchain_video.embeddings.imagebind.source import data
from langchain_video.embeddings.imagebind.source.models import imagebind_model
from langchain_video.embeddings.imagebind.source.models.imagebind_model import ModalityType


class ImageBindEmbeddings(TextEmbeddings, ImageEmbeddings, AudioEmbeddings):
    """ImageBind embedding model from 'ImageBind: One Embedding Space To Bind Them All'.
    
    This class supports text, image and audio embedding.
    
    Original Repo: https://github.com/facebookresearch/ImageBind
    
    Example:
        .. code-block:: python

            from lagnchain_video.embeddings.imagebind.imagebind import ImageBindEmbeddings

            # initiate embedding model
            embedding_model = ImageBindEmbeddings(device="cuda") # or "cpu"

            # single text
            text_embedding = embedding_model.embed_query_text(text) # text should be string

            # multiple text
            text_embeddings = embedding_model.embed_documents(texts) # list of texts
            
            # single image
            image_embedding = embedding_model.embed_query_image(image) # image can be file_path, np.ndarray, etc.

            # multiple image
            image_embeddings = embedding_model.embed_images(images) # list of images

            # single audio
            # audio input must be a tuple: (file_path or np.ndarray or etc, sampling_rate or None)
            audio_embedding = embedding_model.embed_query_audio(audio)

            # multiple audio
            audio_embeddings = embedding_model.embed_audios(audios) # list of audios
    """
    def __init__(self, device: str, **kwargs: Any):
        super().__init__(**kwargs)

        self.device = device
        self._client = imagebind_model.imagebind_huge(pretrained=True)
        self._client.eval()
        self._client.to(self.device)

    def _process_images(self, images: Any) -> List[Image.Image]:
        """Process single or multiple images to the correct format for ImageBind."""
        # Ensure the input is always a list for a consistent loop.
        if not isinstance(images, list):
            images = [images]

        processed_images = []
        for img in images:
            if isinstance(img, str):
                # If string, assume it's a file path
                img = Image.open(img).convert('RGB')
            elif isinstance(img, bytes):
                # If bytes, load from bytes
                img = Image.open(io.BytesIO(img)).convert('RGB')
            elif isinstance(img, np.ndarray):
                # If numpy array, convert to PIL Image
                img = Image.fromarray(img).convert('RGB')
            elif hasattr(img, 'mode'):
                # If PIL Image, ensure RGB
                img = img.convert('RGB')
            else:
                # Try to convert to PIL Image if possible
                try:
                    img = Image.fromarray(np.array(img)).convert('RGB')
                except Exception:
                    raise ValueError(f"Unable to process image of type: {type(img)}")
            
            processed_images.append(img)

        return processed_images
    
    def _process_audios(self, audios: Any) -> List[Tuple[torch.Tensor, Any]]:
        """Process single or multiple audios to the correct format for ImageBind."""
        # Ensure the input is always a list for a consistent loop.
        if not isinstance(audios, list):
            audios = [audios]

        processed_audios = []
        for audio, sr in audios:
            if isinstance(audio, str):
                # If string, assume it's a file path
                audio_tensor, new_sr = torchaudio.load(audio)
            elif isinstance(audio, bytes):
                # If bytes, load from bytes
                audio_tensor, new_sr = torchaudio.load(io.BytesIO(audio))
            elif isinstance(audio, np.ndarray):
                # If numpy array, convert to torch.tensor
                if audio.ndim > 1 and audio.shape[0] < audio.shape[1]:
                    audio_tensor, new_sr = torch.from_numpy(audio), sr
                else:
                    audio_tensor, new_sr = torch.from_numpy(audio).T, sr
            elif isinstance(audio, torch.Tensor):
                # If torch.Tensor, use original data
                audio_tensor, new_sr = audio, sr
            else:
                # Try to convert to torch.Tensor if possible
                try:
                    audio = np.array(audio)
                    if audio.ndim > 1 and audio.shape[0] < audio.shape[1]:
                        audio_tensor, new_sr = torch.from_numpy(audio), sr
                    else:
                        audio_tensor, new_sr = torch.from_numpy(audio).T, sr
                except Exception:
                    raise ValueError(f"Unable to process audio type: {type(audio)}")
            
            # validate dimension
            if audio_tensor.ndim > 2:
                raise ValueError(f"Dimension should be 1D or 2D. Current Dim: {audio_tensor.ndim}")
            
            processed_audios.append((audio_tensor, new_sr))

        return processed_audios

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs.

        Args:
            texts: List of text to embed.

        Returns:
            List of embeddings.
        """
        # make input with multiple texts
        documents = {
            ModalityType.TEXT: data.transform_text_data(texts, self.device)
        }

        # make embeddings
        with torch.no_grad():
            embeddings = self._client(documents)
        
        return embeddings[ModalityType.TEXT].tolist()

    def embed_query_text(self, text: str) -> List[float]:
        """Embed query text.

        Args:
            text: Text to embed.

        Returns:
            Embedding.
        """
        # Ensure the input is always a list for transformation.
        if not isinstance(text, list):
            text = [text]

        # make input with single query text
        query_text = {
            ModalityType.TEXT: data.transform_text_data(text, self.device)
        }

        # make embedding
        with torch.no_grad():
            embedding = self._client(query_text)
        
        return embedding[ModalityType.TEXT].tolist()

    def embed_images(self, images: Any) -> List[List[float]]:
        """Embed multiple images.

        Args:
            images: List of image to embed.

        Returns:
            List of embeddings.
        """
        # convert type of multiple images from Any to PIL.Image.Image
        processed_images = self._process_images(images)

        # make input with multiple images
        input_images = {
            ModalityType.VISION: data.transform_vision_data(processed_images, self.device)
        }

        # make embeddings
        with torch.no_grad():
            embeddings = self._client(input_images)
        
        return embeddings[ModalityType.VISION].tolist()

    def embed_query_image(self, image: Any) -> List[float]:
        """Embed query image.

        Args:
            image: Image to embed.

        Returns:
            Embedding.
        """
        # convert type of single query image from Any to PIL.Image.Image
        processed_image = self._process_images(image)

        # make input with single query image
        query_image = {
            ModalityType.VISION: data.transform_vision_data(processed_image, self.device)
        }

        # make embedding
        with torch.no_grad():
            embedding = self._client(query_image)
        
        return embedding[ModalityType.VISION].tolist()

    def embed_audios(self, audios: Any) -> List[List[float]]:
        """Embed multiple audios.

        Args:
            audios: List of audio to embed.

        Returns:
            List of embeddings.
        """
        # convert type of multiple audios from Any to torch.Tensor
        processed_audios = self._process_audios(audios)

        # make input with multiple audios
        input_audios = {
            ModalityType.AUDIO: data.transform_audio_data(processed_audios, self.device)
        }

        # make embeddings
        with torch.no_grad():
            embeddings = self._client(input_audios)
        
        return embeddings[ModalityType.AUDIO].tolist()

    def embed_query_audio(self, audio: Any) -> List[float]:
        """Embed query audio.

        Args:
            audio: Audio to embed.

        Returns:
            Embedding.
        """
        # convert type of single query audio from Any to torch.Tensor
        processed_audio = self._process_audios(audio)

        # make input with single query audio
        query_audio = {
            ModalityType.AUDIO: data.transform_audio_data(processed_audio, self.device)
        }

        # make embedding
        with torch.no_grad():
            embedding = self._client(query_audio)
        
        return embedding[ModalityType.AUDIO].tolist()
    
    def embed_videos(self, videos: Any, func: Callable=torch.mean, **func_kwargs: dict) -> List[List[float]]:
        """Embed multiple videos.

        Args:
            videos: List of videos to embed.

        Returns:
            List of embeddings.
        """
        embeddings = []       
        for video in videos:
            embeddings.append(
                self.embed_query_video(video, func, func_kwargs)
            )
        
        return embeddings

    def embed_query_video(self, video: Any, func: Callable=torch.mean, **func_kwargs: dict) -> List[float]:
        """Embed query video (== multiple images).

        Args:
            video: Video to embed.

        Returns:
            Embedding.
        """
        # convert type of single video from Any to PIL.Image.Image
        processed_video = self._process_images(video)

        # make input with single video
        query_video = {
            ModalityType.VISION: data.transform_vision_data(processed_video, self.device)
        }

        # make embedding
        with torch.no_grad():
            embedding = self._client(query_video)[ModalityType.VISION]
        
        # applying callable method (default: torch.mean)
        embedding = func(embedding, **func_kwargs)
        
        return embedding.tolist()