#!/usr/bin/env python3
"""
QGpT: Improved Table Retrieval with Question Generation from Partial Tables
Multi-Model Corpus Embedding Builder

This script builds vector embeddings for table corpora using multiple embedding models.
It processes table data from JSON files and creates model-specific vector databases.

Usage:
    python corpus_embedding_builder.py [corpus_file_path] --model [model_name]
    python corpus_embedding_builder.py --list         # List available corpora
    python corpus_embedding_builder.py --list-models  # List available models
    python corpus_embedding_builder.py --all          # Build all corpora with default model
    python corpus_embedding_builder.py --folder [path] # Build all files in folder recursively

Author: QGpT Research Team
Repository: https://github.com/UDICatNCHU/QGpT
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import List, Dict
from loguru import logger
from pymilvus import MilvusClient
import os
from concurrent.futures import ProcessPoolExecutor
import torch
import multiprocessing as mp

# Suppress package warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from embedding_models import BGE_M3_Flag, BGE_M3_Milvus, JinaColBERT_V2
from utils import (
    load_json_dataset, 
    preprocess_text, 
    extract_corpus_name_from_path,
    get_corpus_files,
    validate_corpus_structure,
    generate_db_name
)

# Constants - eliminate magic numbers
MAX_TEXT_LENGTH = 1000  # Milvus storage optimization
MAX_DB_NAME_LENGTH = 32  # milvus_lite limitation
SUPPORTED_EXTENSIONS = {'.json'}  # Supported file types

# Available models configuration
AVAILABLE_MODELS = {
    "bge_flag": BGE_M3_Flag,
    "bge_milvus": BGE_M3_Milvus,
    "jina_colbert": JinaColBERT_V2
}


def create_safe_db_path(model_name: str, corpus_path: Path) -> Path:
    """Create database path that respects milvus_lite constraints"""
    corpus_name = extract_corpus_name_from_path(str(corpus_path))
    base_name = generate_db_name(corpus_name)
    db_path = Path(f"db/{model_name}/{base_name}")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


def normalize_vector(vec) -> List[float]:
    """Convert various vector formats to float32 list"""
    if hasattr(vec, 'astype'):  # numpy array
        return vec.astype('float32').tolist()
    if isinstance(vec, (list, tuple)):
        return list(vec)
    # Let Milvus handle other cases and report errors
    return vec


def extract_document_data(item: Dict) -> tuple[str, Dict]:
    """Extract preprocessed text and metadata from corpus item."""
    return (
        preprocess_text(item['Text']),
        {
            'id': item['id'],
            'filename': item.get('FileName', ''),
            'sheet_name': item.get('SheetName', '')
        }
    )


def build_corpus_embedding(corpus_path: Path, model_name: str, force_rebuild: bool = False, gpu_id: int = 0) -> bool:
    """Build embedding for corpus file with specified model."""
    if model_name not in AVAILABLE_MODELS:
        available = ", ".join(AVAILABLE_MODELS.keys())
        logger.error(f"Unknown model: {model_name}. Available: {available}")
        return False
     
    # Load and validate data
    try:
        data = load_json_dataset(str(corpus_path))
    except Exception as e:
        logger.error(f"Failed to load {corpus_path}: {e}")
        return False
    
    if not validate_corpus_structure(data):
        logger.error(f"Invalid corpus structure in file: {corpus_path}")
        return False
        
    # Set GPU device for this process
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Initialize model with batch size
    try:
        model = AVAILABLE_MODELS[model_name](batch_size=64)
        
    except Exception as e:
        logger.error(f"Failed to initialize model {model_name} on GPU {gpu_id}: {e}")
        return False
    
    # Setup database path
    db_path = create_safe_db_path(model_name, corpus_path)
    collection_name = "corpus"

    # Check if database exists
    if db_path.exists() and not force_rebuild:
        logger.info(f"Database exists: {db_path}, skipping (use --force to rebuild)")
        return True
    
    # Extract documents and metadata  
    try:
        documents, metadata = zip(*[extract_document_data(item) for item in data])
        documents, metadata = list(documents), list(metadata)
    except Exception as e:
        logger.error(f"Failed to preprocess documents: {e}")
        return False
    
    try:
        result = model.encode(documents)
        vectors = result['dense_vecs']
        logger.info(f"Generated {len(vectors)} vectors of dimension {len(vectors[0])}")
        
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        return False
    
    # Setup and populate database
    try:
        _setup_milvus_database(db_path, collection_name, model.dimension, documents, vectors, metadata)
    except Exception as e:
        logger.error(f"Failed to setup database: {e}")
        return False
    
    logger.success(f"Successfully built embeddings for '{corpus_path.name}' at {db_path} using {model.name}")
    return True


def _setup_milvus_database(db_path: Path, collection_name: str, dimension: int, 
                          documents: List[str], vectors: List, metadata: List[Dict]) -> None:
    """Setup Milvus database and insert data"""
    client = MilvusClient(str(db_path))
    
    # Drop existing collection if exists
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)
    
    # Create new collection
    actual_dimension = dimension if dimension > 0 else len(vectors[0])
    client.create_collection(collection_name, dimension=actual_dimension)
    
    # Prepare and insert data
    insert_data = []
    for i, (doc, vec, meta) in enumerate(zip(documents, vectors, metadata)):
        insert_data.append({
            "id": i,
            "vector": normalize_vector(vec),
            "text": doc[:MAX_TEXT_LENGTH],
            "original_id": meta['id'],
            "filename": meta['filename'],
            "sheet_name": meta['sheet_name']
        })
    
    client.insert(collection_name, insert_data)


def find_corpus_files_in_folder(folder_path: Path) -> List[Path]:
    """Recursively find all corpus files in folder"""
    if not folder_path.is_dir():
        raise ValueError(f"Not a directory: {folder_path}")
    
    corpus_files = []
    for file_path in folder_path.rglob("*"):
        if file_path.is_file() and file_path.suffix in SUPPORTED_EXTENSIONS:
            corpus_files.append(file_path)
    
    return sorted(corpus_files)


def _build_single_corpus_worker(args_tuple) -> bool:
    """Worker function for parallel processing"""
    corpus_info, model_name, force_rebuild, gpu_id = args_tuple
    return build_corpus_embedding(Path(corpus_info['path']), model_name, force_rebuild, gpu_id)


def build_all_corpora(args) -> None:
    """Build embeddings for all available corpora using parallel processing"""
    corpus_files = get_corpus_files()
    if not corpus_files:
        logger.warning("No corpus files found")
        return
    
    # Determine number of GPUs and workers
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
    max_workers = min(gpu_count, len(corpus_files))
        
    # Prepare work arguments with GPU assignment
    work_args = []
    for i, corpus_info in enumerate(corpus_files):
        gpu_id = i % gpu_count if torch.cuda.is_available() else 0
        work_args.append((corpus_info, args.model, args.force, gpu_id))
    
    # Execute in parallel with spawn method for CUDA compatibility
    success_count = 0
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context('spawn')) as executor:
        results = list(executor.map(_build_single_corpus_worker, work_args))
        success_count = sum(results)
    
    logger.success(f"Completed: {success_count}/{len(corpus_files)} successful")


def _build_folder_corpus_worker(args_tuple) -> bool:
    """Worker function for folder parallel processing"""
    corpus_file, model_name, force_rebuild, gpu_id = args_tuple
    return build_corpus_embedding(corpus_file, model_name, force_rebuild, gpu_id)


def build_folder_corpora(args) -> None:
    """Build embeddings for all corpus files in folder using parallel processing"""
    folder_path = Path(args.folder)
    
    try:
        corpus_files = find_corpus_files_in_folder(folder_path)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)
    
    if not corpus_files:
        logger.warning(f"No corpus files found in {folder_path}")
        return
    
    # Determine number of GPUs and workers
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
    max_workers = min(gpu_count, len(corpus_files))
    
    # Prepare work arguments with GPU assignment
    work_args = []
    for i, corpus_file in enumerate(corpus_files):
        gpu_id = i % gpu_count if torch.cuda.is_available() else 0
        work_args.append((corpus_file, args.model, args.force, gpu_id))
    
    # Execute in parallel with spawn method for CUDA compatibility
    success_count = 0
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context('spawn')) as executor:
        results = list(executor.map(_build_folder_corpus_worker, work_args))
        success_count = sum(results)
    
    logger.success(f"Completed folder processing: {success_count}/{len(corpus_files)} successful")


def build_single_corpus(args) -> None:
    """Build embedding for single corpus file"""
    corpus_path = Path(args.corpus_path)
    
    if not corpus_path.exists():
        logger.error(f"File not found: {args.corpus_path}")
        sys.exit(1)
    
    if corpus_path.suffix not in SUPPORTED_EXTENSIONS:
        logger.error(f"Unsupported file type: {corpus_path.suffix}. Supported: {', '.join(SUPPORTED_EXTENSIONS)}")
        sys.exit(1)
    
    if not build_corpus_embedding(corpus_path, args.model, args.force):
        sys.exit(1)


def parse_arguments() -> argparse.Namespace:
    """Parse and validate command line arguments"""
    parser = argparse.ArgumentParser(
        description='QGpT multi-model corpus embedding builder',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build single corpus
  python corpus_embedding_builder.py corpus.json --model bge_flag
  
  # Build all files in folder recursively
  python corpus_embedding_builder.py --folder /path/to/corpora --model bge_milvus
  
  # List available corpora and models
  python corpus_embedding_builder.py --list
  python corpus_embedding_builder.py --list-models
  
  # Build all corpora with default model
  python corpus_embedding_builder.py --all --model bge_milvus
  
  # Force rebuild existing databases
  python corpus_embedding_builder.py --all --force
        """
    )
    
    parser.add_argument('corpus_path', nargs='?', help='Corpus file path')
    parser.add_argument('--model', default='bge_flag', 
                       choices=list(AVAILABLE_MODELS.keys()),
                       help='Embedding model to use')
    parser.add_argument('--list', action='store_true', help='List available corpora')
    parser.add_argument('--list-models', action='store_true', help='List available models')
    parser.add_argument('--all', action='store_true', help='Build all corpora')
    parser.add_argument('--folder', help='Build all corpus files in folder recursively')
    parser.add_argument('--force', action='store_true', help='Force rebuild existing databases')
    
    return parser.parse_args()


def determine_action(args) -> str:
    """Determine which action to take based on arguments"""
    if args.list:
        return 'list_corpora'
    if args.list_models:
        return 'list_models'
    if args.all:
        return 'build_all'
    if args.folder:
        return 'build_folder'
    if args.corpus_path:
        return 'build_single'
    return 'show_help'


# Command routing table
COMMAND_HANDLERS = {
    'build_all': build_all_corpora,
    'build_folder': build_folder_corpora,
    'build_single': build_single_corpus,
    'show_help': lambda _: parse_arguments().print_help()
}


def main() -> None:
    """Main entry point with table-driven command routing"""
    args = parse_arguments()
    action = determine_action(args)
    COMMAND_HANDLERS[action](args)


if __name__ == "__main__":
    main()