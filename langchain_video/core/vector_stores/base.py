from __future__ import annotations

from abc import abstractmethod
from typing import List, Optional, Any, Union

from langchain_core.vectorstores import VectorStore
from langchain.docstore.document import Document
from langchain_video.core.blobs import ImageBlob, VideoBlob, AudioBlob, MultiModalBlob
from langchain_video.core.embeddings import MultiModalEmbeddings

import numpy as np

class MultiModalVectorStore(VectorStore):
    """nterface for multimodal vector store."""
    def __init__(self, embedding_function:MultiModalEmbeddings, resampler = None, **kwargs):
        super().__init__()
        self.embedding_function = embedding_function
        self.resampler = resampler
    
    @abstractmethod
    def add_images(
        self,
        images: List[ImageBlob],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any
    ) -> List[str]:
        """add image to vector store"""
        msg = f"`add_images` has not been implemented for {self.__class__.__name__} "
        raise NotImplementedError(msg)
    
    @abstractmethod
    def add_videos(
        self,
        videos: List[VideoBlob],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any
    ) -> List[str]:
        """add video to vector store"""
        msg = f"`add_videos` has not been implemented for {self.__class__.__name__} "
        raise NotImplementedError(msg)
    
    @abstractmethod
    def add_audios(
        self,
        audios: List[AudioBlob],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any
    ) -> List[str]:
        """add audios to vector store"""
        msg = f"`add_audios` has not been implemented for {self.__class__.__name__} "
        raise NotImplementedError(msg)
    
    @abstractmethod
    def similarity_search_by_image(
        self,
        image: Union[np.ndarray, ImageBlob],
        k: int = 4,
        **kwargs: Any
    ) -> List[MultiModalBlob]:
        """search by image"""
        msg = f"`similarity_search_by_image` has not been implemented for {self.__class__.__name__} "
        raise NotImplementedError(msg)
    
    @abstractmethod
    def similarity_search_by_video(
        self,
        video: Union[List[np.ndarray], VideoBlob],
        k: int = 4,
        **kwargs: Any
    ) -> List[Document]:
        """search by video"""
        msg = f"`similarity_search_by_video` has not been implemented for {self.__class__.__name__} "
        raise NotImplementedError(msg)
    
    @abstractmethod
    def similarity_search_by_audio(
        self,
        audio: Union[List[np.ndarray], AudioBlob],
        k: int = 4,
        **kwargs: Any
    ) -> List[Document]:
        """search by audio"""
        msg = f"`imilarity_search_by_audio` has not been implemented for {self.__class__.__name__} "
        raise NotImplementedError(msg)
    
    @classmethod
    @abstractmethod 
    def from_images(
        cls,
        images: List[ImageBlob],
        embedding: MultiModalEmbeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any
    ) -> "MultiModalVectorStore":
        """Make vector store by images data"""
        msg = f"`from_images` has not been implemented for {cls.__name__}"
        raise NotImplementedError(msg)
    
    @classmethod
    @abstractmethod
    def from_videos(
        cls,
        videos: List[VideoBlob], 
        embedding: MultiModalEmbeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any
    ) -> "MultiModalVectorStore":
        """Make vector store by video data"""
        msg = f"`from_videos` has not been implemented for {cls.__name__}"
        raise NotImplementedError(msg)
    
    @classmethod
    @abstractmethod
    def from_audios(
        cls,
        audios: List[AudioBlob],
        embedding: MultiModalEmbeddings,
        metadatas: Optional[List[dict]] = None, 
        **kwargs: Any
    ) -> "MultiModalVectorStore":
        """Make vector store by audio data"""
        msg = f"`from_audios` has not been implemented for {cls.__name__}"
        raise NotImplementedError(msg)
