from __future__ import annotations
from typing import Any, Optional, Union, List
import io
import numpy as np
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, ConfigDict, Field
from langchain_huggingface.utils.import_utils import (
    IMPORT_ERROR,
    is_ipex_available,
    is_optimum_intel_available,
    is_optimum_intel_version,
)

DEFAULT_MULTIMODAL_MODEL_NAME = "sentence-transformers/clip-ViT-B-32"
_MIN_OPTIMUM_VERSION = "1.22"

class HuggingFaceMultiModalEmbeddings(BaseModel, Embeddings):
    """HuggingFace sentence_transformers multimodal embedding models.
    
    This class supports text, image using sentence_transformers. (not supported audio yet)
    
    To use, you should have the ``sentence_transformers`` python package installed.
    
    Example:
        .. code-block:: python
        
            from langchain_video.core.embeddings.huggingface import HuggingFaceMultiModalEmbeddings
            
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
    model_name: str = Field(default=DEFAULT_MULTIMODAL_MODEL_NAME, alias="model")
    """Model name to use."""
    
    cache_folder: Optional[str] = None
    """Path to store models.
    Can be also set by SENTENCE_TRANSFORMERS_HOME environment variable."""
    
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass to the Sentence Transformer model, such as `device`,
    `prompts`, `default_prompt_name`, `revision`, `trust_remote_code`, or `token`.
    See also the Sentence Transformer documentation: https://sbert.net/docs/package_reference/SentenceTransformer.html#sentence_transformers.SentenceTransformer"""
    
    encode_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass when calling the `encode` method for the documents of
    the Sentence Transformer model, such as `prompt_name`, `prompt`, `batch_size`,
    `precision`, `normalize_embeddings`, and more.
    See also the Sentence Transformer documentation: https://sbert.net/docs/package_reference/SentenceTransformer.html#sentence_transformers.SentenceTransformer.encode"""
    
    query_encode_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass when calling the `encode` method for the query of
    the Sentence Transformer model, such as `prompt_name`, `prompt`, `batch_size`,
    `precision`, `normalize_embeddings`, and more.
    See also the Sentence Transformer documentation: https://sbert.net/docs/package_reference/SentenceTransformer.html#sentence_transformers.SentenceTransformer.encode"""
    
    image_encode_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass when calling the `encode` method for images."""
    
    multi_process: bool = False
    """Run encode() on multiple GPUs."""
    
    show_progress: bool = False
    """Whether to show a progress bar."""
    
    def __init__(self, **kwargs: Any):
        """Initialize the sentence_transformer."""
        super().__init__(**kwargs)
        
        try:
            import sentence_transformers  # type: ignore[import]
        except ImportError as exc:
            msg = (
                "Could not import sentence_transformers python package. "
                "Please install it with `pip install sentence-transformers`."
            )
            raise ImportError(msg) from exc
        
        if self.model_kwargs.get("backend", "torch") == "ipex":
            if not is_optimum_intel_available() or not is_ipex_available():
                msg = f"Backend: ipex {IMPORT_ERROR.format('optimum[ipex]')}"
                raise ImportError(msg)
            if is_optimum_intel_version("<", _MIN_OPTIMUM_VERSION):
                msg = (
                    f"Backend: ipex requires optimum-intel>="
                    f"{_MIN_OPTIMUM_VERSION}. You can install it with pip: "
                    "`pip install --upgrade --upgrade-strategy eager "
                    "`optimum[ipex]`."
                )
                raise ImportError(msg)
            from optimum.intel import IPEXSentenceTransformer  # type: ignore[import]
            model_cls = IPEXSentenceTransformer
        else:
            model_cls = sentence_transformers.SentenceTransformer
        
        self._client = model_cls(
            self.model_name, cache_folder=self.cache_folder, **self.model_kwargs
        )
    
    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=(),
        populate_by_name=True,
    )
    
    def _embed_text(self, texts: list[str], encode_kwargs: dict[str, Any]) -> list[list[float]]:
        """Embed texts using the sentence transformer model.
        
        Args:
            texts: The list of texts to embed.
            encode_kwargs: Keyword arguments to pass when calling the `encode` method for the texts.
        
        Returns:
            List of embeddings, one for each text.
        """
        import sentence_transformers
        
        texts = [x.replace("\n", " ") for x in texts]
        
        if self.multi_process:
            pool = self._client.start_multi_process_pool()
            embeddings = self._client.encode_multi_process(texts, pool)
            sentence_transformers.SentenceTransformer.stop_multi_process_pool(pool)
        else:
            embeddings = self._client.encode(
                texts,
                show_progress_bar=self.show_progress,
                **encode_kwargs,
            )
        
        if isinstance(embeddings, list):
            msg = (
                "Expected embeddings to be a Tensor or a numpy array, "
                "got a list instead."
            )
            raise TypeError(msg)
        
        return embeddings.tolist()
    
    def _process_images(self, images: List[Any]) -> List[Any]:
        """Process images to the correct format for sentence_transformers."""
        from PIL import Image
        
        processed_images = []
        
        for img in images:
            if isinstance(img, str):
                # If string, assume it's a file path
                img = Image.open(img).convert('RGB')
            elif isinstance(img, bytes):
                # If bytes, load from bytes
                img = Image.open(io.BytesIO(img)).convert('RGB')
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
    
    def _embed_image(self, images: List[Any], encode_kwargs: dict[str, Any]) -> list[list[float]]:
        """Embed images using the sentence transformer model. (if image embedding is available)
        
        Args:
            images: The list of images to embed.
            encode_kwargs: Keyword arguments to pass when calling the `encode` method for the images.
        
        Returns:
            List of embeddings, one for each image.
        """
        import sentence_transformers
        
        processed_images = self._process_images(images)
        
        if self.multi_process:
            pool = self._client.start_multi_process_pool()
            embeddings = self._client.encode_multi_process(processed_images, pool)
            sentence_transformers.SentenceTransformer.stop_multi_process_pool(pool)
        else:
            embeddings = self._client.encode(
                processed_images,
                show_progress_bar=self.show_progress,
                **encode_kwargs,
            )
        
        if isinstance(embeddings, list):
            msg = (
                "Expected embeddings to be a Tensor or a numpy array, "
                "got a list instead."
            )
            raise TypeError(msg)
        
        return embeddings.tolist()
    
    # Text embedding methods
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.
        
        Args:
            texts: The list of texts to embed.
        
        Returns:
            List of embeddings, one for each text.
        """
        return self._embed_text(texts, self.encode_kwargs)
    
    def embed_query(self, text: str) -> list[float]:
        """Compute query embeddings using a HuggingFace transformer model.
        
        Args:
            text: The text to embed.
        
        Returns:
            Embeddings for the text.
        """
        embed_kwargs = (
            self.query_encode_kwargs
            if len(self.query_encode_kwargs) > 0
            else self.encode_kwargs
        )
        return self._embed_text([text], embed_kwargs)[0]
    
    # Image embedding methods
    def embed_image(self, image: Union[Any, List[Any]]) -> Union[List[float], List[List[float]]]:
        """Embed image(s) using the sentence transformer model.
        
        Args:
            image: The image or list of images to embed.
                Can be file paths, PIL Images, numpy arrays, or bytes.
        
        Returns:
            Embedding(s) for the image(s).
        """
        encode_kwargs = (
            self.image_encode_kwargs
            if len(self.image_encode_kwargs) > 0
            else self.encode_kwargs
        )
        
        if not isinstance(image, list):
            return self._embed_image([image], encode_kwargs)[0]
        else:
            return self._embed_image(image, encode_kwargs)
    
    def embed_images(self, images: List[Any]) -> List[List[float]]:
        """Embed multiple images using the sentence transformer model.
        
        Args:
            images: The list of images to embed.
                Can be file paths, PIL Images, numpy arrays, or bytes.
        
        Returns:
            List of embeddings, one for each image.
        """
        return self.embed_image(images)