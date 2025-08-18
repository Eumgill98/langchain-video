from __future__ import annotations

import uuid
import numpy as np
import pickle
import torch

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path
from tqdm import tqdm

from langchain.docstore.base import Docstore
from langchain.docstore.document import Document
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.vectorstores.utils import maximal_marginal_relevance
from langchain_video.core.blobs import ImageBlob, VideoBlob, AudioBlob
from langchain_video.core.embeddings import MultiModalEmbeddings

from .base import MultiModalVectorStore

try:
    import faiss
except ImportError:
    raise ImportError(
        "Could not import faiss python package. "
        "Please install it with `pip install faiss-cpu` or `pip install faiss-gpu`."
    )

def dependable_faiss_import() -> Any:
    """Import faiss if available, otherwise raise error."""
    try:
        import faiss
    except ImportError:
        raise ImportError(
            "Could not import faiss python package. "
            "Please install it with `pip install faiss-cpu` or `pip install faiss-gpu`."
        )
    return faiss

class MultiModalFAISS(MultiModalVectorStore):
    """FAISS-based multimodal vector store."""

    def __init__(
        self,
        embedding_function: MultiModalEmbeddings,
        index: faiss.Index,
        docstore: Docstore,
        index_to_docstore_id: Dict[int, str],
        relevance_score_fn: Optional[Callable[[float], float]] = None,
        normalize_L2: bool = False,
        distance_strategy: str = "EUCLIDEAN_DISTANCE",
        resampler = None,
        **kwargs: Any,
    ):
        """Initialize with necessary components."""
        super().__init__(embedding_function, resampler, **kwargs)
        self.index = index
        self.docstore = docstore
        self.index_to_docstore_id = index_to_docstore_id
        self.relevance_score_fn = relevance_score_fn
        self.normalize_L2 = normalize_L2
        self.distance_strategy = distance_strategy

    def __add(
        self,
        embeddings: List[List[float]],
        metadatas: Optional[List[dict]] = None,
        documents: Optional[List[Document]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add the given embeddings to the vectorstore."""
        if not embeddings:
            return []
        
        embeddings_np = np.array(embeddings, dtype=np.float32)
        
        if self.normalize_L2:
            faiss.normalize_L2(embeddings_np)

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in embeddings]

        if documents is None:
            documents = []
            for i, embedding in enumerate(embeddings):
                metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
                doc = Document(page_content="", metadata=metadata)
                documents.append(doc)

        self.docstore.add({doc_id: doc for doc_id, doc in zip(ids, documents)})

        starting_len = self.index.ntotal
        self.index.add(embeddings_np)

        for i, doc_id in enumerate(ids):
            self.index_to_docstore_id[starting_len + i] = doc_id

        return ids
    
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add texts to the vectorstore."""
        embeddings = [self.embedding_function.embed_query_text(text) for text in texts]

        documents = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
            metadata.update({"content_type": "text"})
            doc = Document(page_content=text, metadata=metadata)
            documents.append(doc)

        return self.__add(embeddings, metadatas, documents, ids, **kwargs)
    
    def add_images(
        self,
        images: List[ImageBlob],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add images to the vectorstore."""
        embeddings = [self.embedding_function.embed_query_image(img.as_iamges()) for img in images]

        documents = []
        for i, img in enumerate(images):
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
            metadata.update({
                "content_type": "image",
                "path": getattr(img, 'path', None),
                "format": getattr(img, 'mimetype', None)
            })
            page_content = f"[IMAGE:{getattr(img, 'path', f'image_{i}')}]"
            doc = Document(page_content=page_content, metadata=metadata)
            documents.append(doc)

        return self.__add(embeddings, metadatas, documents, ids, **kwargs)
    
    def add_videos(
        self,
        videos: List[VideoBlob],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        func: Callable=torch.mean,
        dim: int = 0,
        **kwargs: Any,
    ) -> List[str]:
        """Add videos to the vectorstore."""

        verbose = kwargs.get('verbose', False)
        video_iterator = tqdm(videos, desc="Video Embedding Process", disable=not verbose) if verbose else videos
        embeddings = []
        for vid in video_iterator:
            frames = vid.as_frames()
            
            if self.resampler is not None:
                frames = self.resampler(frames)
            
            embedding = self.embedding_function.embed_query_video(frames, func=func, dim=dim)
            embeddings.append(embedding)

        documents = []
        for i, vid in enumerate(videos):
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
            metadata.update({
                "content_type": "video",
                "path": getattr(vid, 'path', None),
                "format": getattr(vid, 'mimetype', None),
                "start_frame": getattr(vid, 'start_frame', None),
                "end_frame": getattr(vid, 'end_frame', None)
            })
            page_content = f"[VIDEO:{getattr(vid, 'path', f'video_{i}')}, ({metadata['start_frame']}:{metadata['end_frame']})]"
            doc = Document(page_content=page_content, metadata=metadata)
            documents.append(doc)

        return self.__add(embeddings, metadatas, documents, ids, **kwargs)
    
    def add_audios(
        self,
        audios: List[AudioBlob],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add audios to the vectorstore."""
        embeddings = [self.embedding_function.embed_query_audio(audio) for audio in audios]

        documents = []
        for i, audio in enumerate(audios):
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
            metadata.update({
                "content_type": "audio",
                "path": getattr(audio, 'path', None),
                "format": getattr(audio, 'mimetype', None),
                "start_sample": getattr(audio, 'start_sample', None),
                "end_sample": getattr(audio, 'end_sample', None)
            })
            page_content = f"[AUDIO:{getattr(audio, 'path', f'audio_{i}')}, ({metadata['start_sample']}:{metadata['end_sample']})]"
            doc = Document(page_content=page_content, metadata=metadata)
            documents.append(doc)

        return self.__add(embeddings, metadatas, documents, ids, **kwargs)

    def __search(
        self,
        query_embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        fetch_k: int = 20,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Search for similar documents with scores."""
        if self.index.ntotal == 0:
            return []

        query_embedding_np = np.array([query_embedding], dtype=np.float32)
        
        if self.normalize_L2:
            faiss.normalize_L2(query_embedding_np)

        scores, indices = self.index.search(query_embedding_np, fetch_k)
        
        docs_and_scores = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:
                continue
                
            doc_id = self.index_to_docstore_id.get(idx)
            if doc_id is None:
                continue
                
            doc = self.docstore.search(doc_id)
            if doc is None:
                continue

            if filter:
                doc_metadata = doc.metadata or {}
                if not all(doc_metadata.get(key) == value for key, value in filter.items()):
                    continue

            if self.relevance_score_fn:
                score = self.relevance_score_fn(score)
            
            docs_and_scores.append((doc, score))
            
            if len(docs_and_scores) >= k:
                break

        return docs_and_scores
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        fetch_k: int = 20,
        **kwargs: Any,
    ) -> List[Document]:
        """Search for similar documents using text query."""
        query_embedding = self.embedding_function.embed_query_text(query)
        print(query_embedding)
        print(len(query_embedding))
        docs_and_scores = self.__search(query_embedding, k, filter, fetch_k, **kwargs)
        return [doc for doc, _ in docs_and_scores]
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        fetch_k: int = 20,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Search for similar documents with scores using text query."""
        query_embedding = self.embedding_function.embed_query_text(query)
        return self.__search(query_embedding, k, filter, fetch_k, **kwargs)
    
    def similarity_search_by_image(
        self,
        image: Union[np.ndarray, ImageBlob],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        fetch_k: int = 20,
        **kwargs: Any,
    ) -> List[Document]:
        """Search for similar content using image query."""
        if isinstance(image, ImageBlob):
            query_embedding = self.embedding_function.embed_query_image(image)
        else:
            query_embedding = image
            
        docs_and_scores = self.__search(query_embedding, k, filter, fetch_k, **kwargs)
        return [doc for doc, _ in docs_and_scores]
    
    def similarity_search_by_video(
        self,
        video: Union[List[np.ndarray], VideoBlob],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        fetch_k: int = 20,
        func: Callable = torch.mean,
        dim: int = 0,
        **kwargs: Any,
    ) -> List[Document]:
        """Search for similar content using video query."""
        if isinstance(video, VideoBlob):
            frames = video.as_frames()
        else:
            frames = video
        
        if self.resampler is not None:
            frames = self.resampler(frames)
        
        query_embedding = self.embedding_function.embed_query_video(frames, func=func, dim=dim)
        docs_and_scores = self.__search(query_embedding, k, filter, fetch_k, **kwargs)
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_by_audio(
        self,
        audio: Union[List[np.ndarray], AudioBlob],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        fetch_k: int = 20,
        **kwargs: Any,
    ) -> List[Document]:
        """Search for similar content using audio query."""
        if isinstance(audio, AudioBlob):
            query_embedding = audio
        else:
            query_embedding = self.embedding_function.embed_query_audio(audio)

        docs_and_scores = self.__search(query_embedding, k, filter, fetch_k, **kwargs)
        return [doc for doc, _ in docs_and_scores]
    
    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance."""
        embedding = self.embedding_function.embed_query_text(query)
        return self.max_marginal_relevance_search_by_vector(
            embedding, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, filter=filter, **kwargs
        )
    
    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance."""
        docs_and_scores = self.__search(embedding, fetch_k, filter, fetch_k, **kwargs)
        
        if len(docs_and_scores) == 0:
            return []
        
        embeddings = []
        docs = []
        for doc, _ in docs_and_scores:
            doc_id = None
            for idx, stored_id in self.index_to_docstore_id.items():
                if self.docstore.search(stored_id) == doc:
                    doc_id = idx
                    break
            
            if doc_id is not None:
                doc_embedding = self.index.reconstruct(doc_id)
                embeddings.append(doc_embedding)
                docs.append(doc)
        
        if not embeddings:
            return [doc for doc, _ in docs_and_scores[:k]]
        
        mmr_selected = maximal_marginal_relevance(
            np.array(embedding), embeddings, k=k, lambda_mult=lambda_mult
        )
        return [docs[i] for i in mmr_selected]
    
    def delete(
        self,
        ids: Optional[List[str]] = None,
        **kwargs: Any
    ) -> Optional[bool]:
        """Delete by document IDs."""
        if ids is None:
            return None

        for doc_id in ids:
            self.docstore.delete([doc_id])

        indices_to_remove = []
        for idx, doc_id in list(self.index_to_docstore_id.items()):
            if doc_id in ids:
                indices_to_remove.append(idx)
                del self.index_to_docstore_id[idx]

        return True
    
    def save_local(self, folder_path: str, index_name: str = "index") -> None:
        """Save FAISS index, docstore, and index_to_docstore_id to disk."""
        path = Path(folder_path)
        path.mkdir(exist_ok=True, parents=True)

        faiss.write_index(self.index, str(path / f"{index_name}.faiss"))

        with open(path / f"{index_name}.pkl", "wb") as f:
            pickle.dump((self.docstore, self.index_to_docstore_id), f)

    @classmethod
    def load_local(
        cls,
        folder_path: str,
        embeddings: MultiModalEmbeddings,
        index_name: str = "index",
        **kwargs: Any,
    ) -> MultiModalFAISS:
        """Load FAISS index, docstore, and index_to_docstore_id from disk."""
        path = Path(folder_path)

        index = faiss.read_index(str(path / f"{index_name}.faiss"))

        with open(path / f"{index_name}.pkl", "rb") as f:
            docstore, index_to_docstore_id = pickle.load(f)

        return cls(embeddings, index, docstore, index_to_docstore_id, **kwargs)
    
    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: MultiModalEmbeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> MultiModalFAISS:
        """Construct FAISS wrapper from raw documents."""
        faiss = dependable_faiss_import()
        
        text_vector = embedding.embed_query_text(texts[0])
        index = faiss.IndexFlatL2(len(text_vector))
        
        docstore = InMemoryDocstore()
        index_to_docstore_id = {}

        vecstore = cls(
            embedding_function=embedding,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
            **kwargs,
        )

        vecstore.add_texts(texts, metadatas=metadatas, ids=ids)
        return vecstore

    @classmethod
    def from_images(
        cls,
        images: List[ImageBlob],
        embedding: MultiModalEmbeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> MultiModalFAISS:
        """Construct FAISS wrapper from images."""
        faiss = dependable_faiss_import()
        
        image_vector = embedding.embed_query_image(images[0].as_image())
        index = faiss.IndexFlatL2(len(image_vector))
        
        docstore = InMemoryDocstore()
        index_to_docstore_id = {}

        vecstore = cls(
            embedding_function=embedding,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
            **kwargs,
        )

        vecstore.add_images(images, metadatas=metadatas, ids=ids)
        return vecstore

    @classmethod
    def from_videos(
        cls,
        videos: List[VideoBlob],
        embedding: MultiModalEmbeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        resampler = None,
        func: Callable = torch.mean,
        dim: int = 0,
        **kwargs: Any,
    ) -> MultiModalFAISS:
        """Construct FAISS wrapper from videos."""
        faiss = dependable_faiss_import()
        
        frames = videos[0].as_frames()
        if resampler is not None:
            frames = resampler(frames)
        video_vec = embedding.embed_query_video(frames, func=func, dim=dim)
        index = faiss.IndexFlatL2(len(video_vec))

        print(f"Len of Embedding : [{len(video_vec)}]")
        
        docstore = InMemoryDocstore()
        index_to_docstore_id = {}

        vecstore = cls(
            embedding_function=embedding,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
            resampler=resampler,
            **kwargs,
        )

        vecstore.add_videos(videos, metadatas=metadatas, ids=ids, func=func, dim=dim, **kwargs)
        return vecstore

    @classmethod
    def from_audios(
        cls,
        audios: List[AudioBlob],
        embedding: MultiModalEmbeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> MultiModalFAISS:
        """Construct FAISS wrapper from audios."""
        faiss = dependable_faiss_import()
        
        audio_vector = embedding.embed_query_audio(audios[0].as_audio())
        index = faiss.IndexFlatL2(len(audio_vector))

        docstore = InMemoryDocstore()
        index_to_docstore_id = {}

        vecstore = cls(
            embedding_function=embedding,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
            **kwargs,
        )

        vecstore.add_audios(audios, metadatas=metadatas, ids=ids)
        return vecstore