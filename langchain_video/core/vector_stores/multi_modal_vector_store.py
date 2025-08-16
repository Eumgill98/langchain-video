from __future__ import annotations

from typing import List, Dict, Optional, Union
from langchain_video.core.blobs import AudioBlob, VideoBlob, ImageBlob
from langchain_video.core.blobs import MultiModalBlob
from langchain_video.core.embeddings import MultiModalEmbeddings

from langchain_core.vectorstores import VectorStore

import numpy as np

class MultiModalVectorStore:
    def __init__(
        self,
        embedding: MultiModalEmbeddings,
        vector_store: VectorStore
    ):
        self.embedding = embedding
        self.vector_store = vector_store

    def _embed_blob(
        self,
        blob: MultiModalBlob,
    ) -> Optional[np.ndarray]:
        """Make Embeddings by blob"""
        if isinstance(blob, AudioBlob):
            return (self.embedding.embed_query_audio(blob.as_audio()))
        elif isinstance(blob, VideoBlob):
            return (self.embedding.embed_videos(blob.as_frames()))
        elif isinstance(blob, ImageBlob):
            return (self.embedding.embed_query_image(blob.as_image()))
        return None

    def _extract_metadata(
        self,
        blob: MultiModalBlob,
    ) -> Dict:
        """Extract Blob metadata"""

        if isinstance(blob, AudioBlob):
            return {
                'path': blob.path,
                'start_sample': blob.start_sample,
                'end_sample': blob.end_sample,
                'mimetype': blob.mimetype,
                'codec': blob.codec,
                'sample_rate': blob.sample_rate,
                'channels': blob.channels,
                'bitrate': blob.bitrate,
                'duration_sec': blob.duration_sec,
                'total_samples': blob.total_samples,
                'bit_depth': blob.bit_depth,
            }
        elif isinstance(blob, VideoBlob):
            return {
                'path': blob.path,
                'start_frame': blob.start_frame,
                'end_frame': blob.end_frame,
                'start_sample': blob.start_sample,
                'end_sample': blob.end_sample,
                'mimetype': blob.mimetype,
                'codec': blob.codec,
                'total_frames': blob.total_frames,
                'total_samples': blob.total_samples,
                'height': blob.height,
                'width': blob.width,
                'durations_sec': blob.duration_sec,
                'fps': blob.fps,
                'audio_codec': blob.audio_codec,
                'sample_rate': blob.sample_rate,
                'audio_channels': blob.audio_channels,
                'audio_bitrate': blob.audio_bitrate,
            }
        elif isinstance(blob, ImageBlob):
            return {
                'path': blob.path,
                'mimetype': blob.mimetype,
                'color_space': blob.color_space,
                'height': blob.height,
                'width': blob.width,
                'channels': blob.channels,
            }
        return {}

    def add_data(
        self,
        blobs: List[MultiModalBlob]
    ):
        """Add vector to vector store"""
        vectors = []
        metadatas = []

        for blob in blobs:
            embed = self._embed_blob(blob)
            if embed is not None:
                vectors.append(embed)
                metadatas.append(self._extract_metadata(blob))

        # add vector 
        if hasattr(self.vector_store, "add_vectors"):
            self.vector_store.add_vectors(vectors=vectors, metadatas=metadatas)
        elif hasattr(self.vector_store, "add_texts"):
            dummy_texts = ["[MultiModal Embedding]" for _ in vectors]
            self.vector_store.add_texts(dummy_texts, metadatas=metadatas, embeddings=vectors)
        else:
            raise NotImplementedError(f"This Vector Store does not support vector addition")
                
    def search(
        self,
        data: Union[str, MultiModalBlob],
        k:int = 4,
    ):
        """Search"""
        if isinstance(data, str):
            data_embed = self.embedding.embed_text(data)
        elif isinstance(data, MultiModalBlob):
            data_embed = self._embed_blob(data)
        elif isinstance(data, np.ndarray):
            data_embed = data

        if hasattr(self.vector_store, "search_vectors"):
            return self.vector_store.search_vectors(data_embed, k)
        elif hasattr(self.vector_store, "similarity_search_by_vector"):
            return self.vector_store.similarity_search_by_vector(data_embed, k)
        else:
            raise NotImplementedError(f"This Vector Store does not support vector search")