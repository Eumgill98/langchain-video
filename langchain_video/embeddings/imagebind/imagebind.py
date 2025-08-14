from PIL import Image
import io

from typing import List, Tuple, Any
import numpy as np

import torch
import torchaudio

from langchain_video.core.embeddings.base import TextEmbeddings, ImageEmbeddings, AudioEmbeddings
from pydantic import BaseModel

from langchain_video.embeddings.imagebind.source import data
from langchain_video.embeddings.imagebind.source.models import imagebind_model
from langchain_video.embeddings.imagebind.source.models.imagebind_model import ModalityType


class ImageBindEmbeddings(BaseModel, TextEmbeddings, ImageEmbeddings, AudioEmbeddings):
    """ImageBind embedding model from 'ImageBind: One Embedding Space To Bind Them All'.
    
    This class supports text, image and audio.
    
    To use, you should have the following requirements.
    - https://github.com/facebookresearch/ImageBind?tab=readme-ov-file (ImageBind GitHub Repository)
    - install requirements.txt (at ImageBind GitHub Repository)

    
    Example: (TODO: 수정)
        .. code-block:: python

            from langchain_video.core.embeddings.imagebind import ImageBindEmbeddings

            embeddings = ImageBindEmbeddings()

            model_name = "sentence-transformers/clip-ViT-B-32"
            model_kwargs = {'device': 'cpu'}
            encode_kwargs = {'normalize_embeddings': False}
            
            hf = HuggingFaceMultiModalEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
            
            # Text embedding
            text_emb = hf.embed_text("Hello world")
            
            # Image embedding
            from PIL import Image
            image = Image.open("image.jpg")
            image_emb = hf.embed_image(image)
    """
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self._client = imagebind_model.imagebind_huge(pretrained=True)
        self._client.eval()
        self._client.to(self.device)

    def _process_images(self, images: List[Any]) -> List[Image.Image]:
        """Process images to the correct format for ImageBind."""
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
    
    def _process_audios(self, audios: List[Tuple[Any, Any]]) -> List[Tuple[torch.Tensor, Any]]:
        """Process audios to the correct format for ImageBind."""
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

    # TODO:
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs.

        Args:
            texts: List of text to embed.

        Returns:
            List of embeddings.
        """

    def embed_query_text(self, text: List[str]) -> List[float]:
        """Embed query text.

        Args:
            text: Text to embed.

        Returns:
            Embedding.
        """
        # make query text
        query_text = {
            ModalityType.TEXT: data.load_and_transform_text(text, self.device)
        }

        # make embeddings
        with torch.no_grad():
            embeddings = self._client(query_text)
        
        return embeddings[ModalityType.TEXT].tolist()

    # TODO:
    def embed_images(self, images: List[np.ndarray]) -> List[List[float]]:
        """Embed multiple images.

        Args:
            images: List of image to embed.

        Returns:
            List of embeddings.
        """

    def embed_query_image(self, image: List[Any]) -> List[float]:
        """Embed query image.

        Args:
            image: Image to embed.

        Returns:
            Embedding.
        """
        # convert Any to PIL.Image
        processed_images = self._process_images(image)

        # make query image
        query_image = {
            ModalityType.VISION: data.load_and_transform_vision_data(processed_images, self.device)
        }

        # make embeddings
        with torch.no_grad():
            embeddings = self._client(query_image)
        
        return embeddings[ModalityType.VISION].tolist()

    # TODO:
    def embed_audios(self, audios: List[Any]) -> List[List[float]]:
        """Embed multiple audios.

        Args:
            audios: List of audio to embed.

        Returns:
            List of embeddings.
        """

    def embed_query_audio(self, audio: List[Tuple[Any, Any]]) -> List[float]:
        """Embed query audio.

        Args:
            audio: Audio to embed.

        Returns:
            Embedding.
        """
        # convert Any to torch.Tensor
        processed_audios = self._process_audios(audio)

        # make query image
        query_audio = {
            ModalityType.AUDIO: data.load_and_transform_audio_data(processed_audios, self.device)
        }

        # make embeddings
        with torch.no_grad():
            embeddings = self._client(query_audio)
        
        return embeddings[ModalityType.AUDIO].tolist()