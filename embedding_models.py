"""
Embedding models ABC for QGpT corpus builder.
All embedding models must inherit from EmbeddingModel.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any

import numpy as np


class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""
    
    def __init__(self):
        self._dimension = None
        self._name = None
        self._embedding_type = "dense"
    
    @abstractmethod
    def encode(self, texts: list[str]) -> Dict[str, Any]:
        """Encode texts to vectors.
        
        Returns:
            Dict with 'dense_vecs' or 'sparse_vecs' key
        """
        pass
    
    @property
    def dimension(self) -> int:
        """Vector dimension. Return -1 for variable dimension (sparse)."""
        return self._dimension
    
    @property
    def name(self) -> str:
        """Model identifier for filenames."""
        return self._name
    
    @property
    def embedding_type(self) -> str:
        """Type: 'dense' or 'sparse'."""
        return self._embedding_type


class BGE_M3_Flag(EmbeddingModel):
    """BGE-M3 using FlagEmbedding implementation."""
    
    def __init__(self, batch_size: int = 64):
        super().__init__()
        from FlagEmbedding import BGEM3FlagModel
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device=device, max_length=8192)
        self._dimension = 1024
        self._name = "bge_m3_flag"
        self._batch_size = batch_size
        
    def encode(self, texts: list[str]) -> Dict[str, Any]:
        result = self._model.encode(texts, batch_size=self._batch_size)
        return {"dense_vecs": result['dense_vecs']}


class BGE_M3_Milvus(EmbeddingModel):
    """BGE-M3 using Milvus hybrid implementation."""
    
    def __init__(self, batch_size: int = 64):
        super().__init__()
        from pymilvus.model.hybrid import BGEM3EmbeddingFunction
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = BGEM3EmbeddingFunction(use_fp16=True, device=device, batch_size=batch_size, max_length=8192)
        self._dimension = 1024
        self._name = "bge_m3_milvus"
        self._batch_size = batch_size
        
    def encode(self, texts: list[str]) -> Dict[str, Any]:
        result = self._model(texts)
        return {"dense_vecs": result['dense']}


def _mean_pool_token_embeddings(token_embeddings):
    """Extract document vector from token-level embeddings via mean pooling."""
    return token_embeddings[0].mean(axis=0)


class JinaColBERT_V2(EmbeddingModel):
    """Jina ColBERT v2 model using Pylate."""
    
    def __init__(self, batch_size: int = 64):
        super().__init__()
        from pylate import models
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = models.ColBERT(
            model_name_or_path="jinaai/jina-colbert-v2",
            query_prefix="[QueryMarker]",
            document_prefix="[DocumentMarker]",
            attend_to_expansion_tokens=True,
            trust_remote_code=True,
            device=device,
        )
        self._dimension = 128
        self._name = "jina_colbert_v2"
        self._batch_size = batch_size
        
    def encode(self, texts: list[str]) -> Dict[str, Any]:
        # Process in batches for memory efficiency
        embeddings = []
        for i in range(0, len(texts), self._batch_size):
            batch_texts = texts[i:i + self._batch_size]
            batch_embeddings = [
                _mean_pool_token_embeddings(self._model.encode([text], is_query=False))
                for text in batch_texts
            ]
            embeddings.extend(batch_embeddings)
        return {"dense_vecs": np.array(embeddings)}