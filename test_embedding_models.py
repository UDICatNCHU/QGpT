#!/usr/bin/env python3
"""
Test script for embedding models batch inference.
Tests all three models with batch size 16.
"""

import time

import numpy as np
from loguru import logger

from embedding_models import BGE_M3_Flag, BGE_M3_Milvus, JinaColBERT_V2


def get_test_sentences() -> list[str]:
    """Generate 32 test sentences."""
    return [
        "NLP is very funny",
        "Today is friday", 
        "Python programming language is great",
        "Machine learning revolutionizes data analysis",
        "The weather today seems quite pleasant",
        "Python programming language is versatile",
        "Coffee helps me stay awake during coding",
        "Vector embeddings capture semantic meaning",
        "Database indexing improves query performance",
        "Deep learning models require substantial computation",
        "Natural language processing enables AI understanding",
        "Cloud computing provides scalable infrastructure", 
        "Software engineering practices ensure code quality",
        "Data science combines statistics and programming",
        "Artificial intelligence transforms various industries",
        "Open source projects drive innovation forward",
        "User experience design focuses on human needs",
        "Information retrieval systems help find relevant content",
        "Distributed systems handle large-scale workloads",
        "Algorithm optimization reduces computational complexity",
        "Web development frameworks accelerate building applications",
        "Computer vision analyzes and interprets images",
        "Cybersecurity protects against digital threats",
        "Mobile applications provide convenient user access",
        "Data visualization makes complex information understandable",
        "Version control systems track code changes",
        "API design enables system integration",
        "Performance monitoring identifies bottlenecks",
        "Code documentation facilitates team collaboration",
        "Testing strategies ensure software reliability",
        "Continuous integration automates development workflows",
        "Error handling improves system robustness"
    ]


def batch_inference_test(model, texts: list[str], batch_size: int = 16) -> dict:
    """Test batch inference for a model."""
    logger.info(f"Testing {model.name} (dim={model.dimension}, type={model.embedding_type})")
    
    # Process in batches
    all_vectors = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        logger.info(f"  Batch {i//batch_size + 1}: processing {len(batch)} texts...")
        result = model.encode(batch)
        vectors = result['dense_vecs']
        all_vectors.extend(vectors)
            
    return {
        'model_name': model.name,
        'total_vectors': len(all_vectors),
        'dimension': model.dimension,
        'vector_shape': np.array(all_vectors[0]).shape if all_vectors else None
    }


def test_single_model(model_name: str = "bge-flag"):
    """Test single model for quick debugging."""
    model_map = {
        "bge-flag": BGE_M3_Flag,
        "bge-milvus": BGE_M3_Milvus,
        "jina-colbert": JinaColBERT_V2
    }
    
    if model_name not in model_map:
        logger.error(f"Unknown model: {model_name}")
        logger.error(f"Available: {list(model_map.keys())}")
        return
        
    model = model_map[model_name]()
    texts = get_test_sentences()
    
    batch_inference_test(model, texts, batch_size=16)


if __name__ == "__main__":
    test_single_model("bge-flag")
    time.sleep(30)
    test_single_model("bge-milvus")
    time.sleep(30)
    test_single_model("jina-colbert")