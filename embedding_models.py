"""
Embedding models ABC for QGpT corpus builder.
All embedding models must inherit from EmbeddingModel.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any

import numpy as np


class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""
    
    @abstractmethod
    def encode(self, texts: list[str]) -> Dict[str, Any]:
        """Encode texts to vectors.
        
        Returns:
            Dict with 'dense_vecs' or 'sparse_vecs' key
        """
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector dimension. Return -1 for variable dimension (sparse)."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Model identifier for filenames."""
        pass
    
    @property
    @abstractmethod
    def embedding_type(self) -> str:
        """Type: 'dense' or 'sparse'."""
        pass


class BGE_M3_Flag(EmbeddingModel):
    """BGE-M3 using FlagEmbedding implementation."""
    
    def __init__(self):
        from FlagEmbedding import BGEM3FlagModel
        self._model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
        
    def encode(self, texts: list[str]) -> Dict[str, Any]:
        result = self._model.encode(texts)
        return {"dense_vecs": result['dense_vecs']}
    
    @property
    def dimension(self) -> int:
        return 1024
    
    @property
    def name(self) -> str:
        return "bge_m3_flag"
    
    @property
    def embedding_type(self) -> str:
        return "dense"


class BGE_M3_Milvus(EmbeddingModel):
    """BGE-M3 using Milvus hybrid implementation."""
    
    def __init__(self):
        from pymilvus.model.hybrid import BGEM3EmbeddingFunction
        self._model = BGEM3EmbeddingFunction(use_fp16=True)
        
    def encode(self, texts: list[str]) -> Dict[str, Any]:
        result = self._model(texts)
        return {"dense_vecs": result['dense']}
    
    @property
    def dimension(self) -> int:
        return 1024
    
    @property
    def name(self) -> str:
        return "bge_m3_milvus"
    
    @property
    def embedding_type(self) -> str:
        return "dense"


class JinaColBERT_V2(EmbeddingModel):
    """Jina ColBERT v2 model using Pylate."""
    
    def __init__(self):
        from pylate import models
        self._model = models.ColBERT(
            model_name_or_path="jinaai/jina-colbert-v2",
            query_prefix="[QueryMarker]",
            document_prefix="[DocumentMarker]",
            attend_to_expansion_tokens=True,
            trust_remote_code=True,
        )
        
    def encode(self, texts: list[str]) -> Dict[str, Any]:
        # Pylate ColBERT encode 返回 token-level embeddings
        # 需要做 mean pooling 得到文件向量
        embeddings = []
        for text in texts:
            # 使用 document_prefix 編碼
            token_embeddings = self._model.encode([text], is_query=False)
            # 對 token 維度做 mean pooling
            doc_embedding = token_embeddings[0].mean(axis=0)  # shape: [hidden_dim]
            embeddings.append(doc_embedding)
        
        import numpy as np
        return {"dense_vecs": np.array(embeddings)}
    
    @property
    def dimension(self) -> int:
        return 128  # Jina ColBERT v2 dimension
    
    @property
    def name(self) -> str:
        return "jina_colbert_v2"
    
    @property
    def embedding_type(self) -> str:
        return "dense"