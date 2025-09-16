"""
Embedding models ABC for QGpT corpus builder.
All embedding models must inherit from EmbeddingModel.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List

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
    
    def __init__(self, batch_size: int = 32):
        super().__init__()
        from FlagEmbedding import BGEM3FlagModel
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device=device, max_length=8192)
        self._dimension = 1024
        self._name = "bge_m3_flag"
        self._batch_size = batch_size
        
    def encode(self, texts: list[str]) -> Dict[str, Any]:
        result = self._model.encode(texts, batch_size=self._batch_size, max_length=8192)
        return {"dense_vecs": result['dense_vecs']}


class JinaColBERT_V2(EmbeddingModel):
    """Jina ColBERT v2 model using Pylate."""
    
    def __init__(self, batch_size: int = 32):
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
            document_length=8192,  # 就是這個參數
        )
        self._dimension = 128
        self._name = "jina_colbert_v2"
        self._batch_size = batch_size
        
    def encode(self, texts: list[str]) -> Dict[str, Any]:
        # 直接用官方 API，它內建批次處理
        token_embeddings = self._model.encode(
            texts, 
            batch_size=self._batch_size,
            is_query=False,
            show_progress_bar=False
        )
        # 對每個文件的 token embeddings 做 mean pooling
        pooled_embeddings = [embed.mean(axis=0) for embed in token_embeddings]
        return {"dense_vecs": np.array(pooled_embeddings)}

    
MODELS = {
    "bge_m3_flag": BGE_M3_Flag,
    "jina_colbert_v2": JinaColBERT_V2,
}