"""Interface for embedding models."""
from abc import ABC, abstractmethod
from typing import List

import numpy as np

from langchain_core.runnables.config import run_in_executor


class TextEmbeddings(ABC):
    """
    This is an interface meant for implementing text embedding models.

    Text embedding models are used to map text to a vector (a point in n-dimensional
    space).

    Texts that are similar will usually be mapped to points that are close to each
    other in this space. The exact details of what's considered "similar" and how
    "distance" is measured in this space are dependent on the specific embedding model.

    This abstraction contains a method for embedding a list of documents and a method
    for embedding a query text. The embedding of a query text is expected to be a single
    vector, while the embedding of a list of documents is expected to be a list of
    vectors.

    Usually the query embedding is identical to the document embedding, but the
    abstraction allows treating them independently.

    In addition to the synchronous methods, this interface also provides asynchronous
    versions of the methods.

    By default, the asynchronous methods are implemented using the synchronous methods;
    however, implementations may choose to override the asynchronous methods with
    an async native implementation for performance reasons.
    """

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs.

        Args:
            texts: List of text to embed.

        Returns:
            List of embeddings.
        """

    @abstractmethod
    def embed_query_text(self, text: str) -> List[float]:
        """Embed query text.

        Args:
            text: Text to embed.

        Returns:
            Embedding.
        """

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous Embed search docs.

        Args:
            texts: List of text to embed.

        Returns:
            List of embeddings.
        """
        return await run_in_executor(None, self.embed_documents, texts)

    async def aembed_query_text(self, text: str) -> List[float]:
        """Asynchronous Embed query text.

        Args:
            text: Text to embed.

        Returns:
            Embedding.
        """
        return await run_in_executor(None, self.embed_query_text, text)


class ImageEmbeddings(ABC):
    """
    This is an interface meant for implementing image embedding models.

    Image embedding models are used to map image to a vector (a point in n-dimensional
    space).

    Images that are similar will usually be mapped to points that are close to each
    other in this space. The exact details of what's considered "similar" and how
    "distance" is measured in this space are dependent on the specific embedding model.

    This abstraction contains a method for embedding a list of documents and a method
    for embedding a query image. The embedding of a query image is expected to be a single
    vector, while the embedding of a list of documents is expected to be a list of
    vectors.

    Usually the query embedding is identical to the document embedding, but the
    abstraction allows treating them independently.

    In addition to the synchronous methods, this interface also provides asynchronous
    versions of the methods.

    By default, the asynchronous methods are implemented using the synchronous methods;
    however, implementations may choose to override the asynchronous methods with
    an async native implementation for performance reasons.
    """

    @abstractmethod
    def embed_images(self, images: List[np.ndarray]) -> List[List[float]]:
        """Embed multiple images.

        Args:
            images: List of image to embed.

        Returns:
            List of embeddings.
        """

    @abstractmethod
    def embed_query_image(self, image: np.ndarray) -> List[float]:
        """Embed query image.

        Args:
            image: Image to embed.

        Returns:
            Embedding.
        """
    
    async def aembed_images(self, images: List[np.ndarray]) -> List[List[float]]:
        """Asynchronous Embed multiple images.

        Args:
            images: List of image to embed.

        Returns:
            List of embeddings.
        """
        return await run_in_executor(None, self.embed_images, images)

    async def aembed_query_image(self, image: np.ndarray) -> List[float]:
        """Asynchronous Embed query image.

        Args:
            image: Image to embed.

        Returns:
            Embedding.
        """
        return await run_in_executor(None, self.embed_query_image, image)


class AudioEmbeddings(ABC):
    """
    This is an interface meant for implementing audio embedding models.

    Audio embedding models are used to map audio to a vector (a point in n-dimensional
    space).

    Audios that are similar will usually be mapped to points that are close to each
    other in this space. The exact details of what's considered "similar" and how
    "distance" is measured in this space are dependent on the specific embedding model.

    This abstraction contains a method for embedding a list of documents and a method
    for embedding a query audio. The embedding of a query audio is expected to be a single
    vector, while the embedding of a list of documents is expected to be a list of
    vectors.

    Usually the query embedding is identical to the document embedding, but the
    abstraction allows treating them independently.

    In addition to the synchronous methods, this interface also provides asynchronous
    versions of the methods.

    By default, the asynchronous methods are implemented using the synchronous methods;
    however, implementations may choose to override the asynchronous methods with
    an async native implementation for performance reasons.
    """

    @abstractmethod
    def embed_audios(self, audios: List[np.ndarray]) -> List[List[float]]:
        """Embed multiple audios.

        Args:
            audios: List of audio to embed.

        Returns:
            List of embeddings.
        """

    @abstractmethod
    def embed_query_audio(self, audio: np.ndarray) -> List[float]:
        """Embed query audio.

        Args:
            audio: Audio to embed.

        Returns:
            Embedding.
        """

    async def aembed_audios(self, audios: List[np.ndarray]) -> List[List[float]]:
        """Asynchronous Embed multiple audios.

        Args:
            audios: List of audio to embed.

        Returns:
            List of embeddings.
        """
        return await run_in_executor(None, self.embed_audios, audios)

    async def aembed_query_audio(self, audio: np.ndarray) -> List[float]:
        """Asynchronous Embed query audio.

        Args:
            audio: Audio to embed.

        Returns:
            Embedding.
        """
        return await run_in_executor(None, self.embed_query_audio, audio)