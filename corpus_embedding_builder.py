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
import hashlib
import sys
from pathlib import Path
from typing import List, Dict
from loguru import logger
from pymilvus import MilvusClient

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


def build_corpus_embedding(corpus_path: Path, model_name: str, force_rebuild: bool = False) -> None:
    """Build embedding for corpus file with specified model."""
    if model_name not in AVAILABLE_MODELS:
        available = ", ".join(AVAILABLE_MODELS.keys())
        logger.error(f"Unknown model: {model_name}. Available: {available}")
        raise ValueError(f"Unknown model: {model_name}")   
     
    # Load and validate data
    logger.info(f"Loading corpus data from: {corpus_path}")
    data = load_json_dataset(str(corpus_path))
    
    if not validate_corpus_structure(data):
        logger.error(f"Invalid corpus structure in file: {corpus_path}")
        raise ValueError(f"Invalid corpus structure in file: {corpus_path}")
    
    logger.info(f"Loaded {len(data)} documents")
    
    # Initialize model
    logger.info(f"Initializing model: {model_name}")
    model = AVAILABLE_MODELS[model_name]()
    
    # Setup database path
    db_path = create_safe_db_path(model_name, corpus_path)
    collection_name = "corpus"
    
    logger.info(f"Database: {db_path}")
    logger.info(f"Collection: {collection_name}")
    logger.info(f"Model dimension: {model.dimension}")
    
    # Check if database exists
    if db_path.exists() and not force_rebuild:
        logger.info(f"Database exists: {db_path}, skipping (use --force to rebuild)")
        return
    
    # Preprocess documents
    logger.info("Preprocessing documents...")
    documents = []
    metadata = []
    
    for item in data:
        clean_text = preprocess_text(item['Text'])
        documents.append(clean_text)
        
        metadata.append({
            'id': item['id'],
            'filename': item.get('FileName', ''),
            'sheet_name': item.get('SheetName', '')
        })
    
    # Generate embeddings
    logger.info("Generating embeddings...")
    result = model.encode(documents)
    vectors = result['dense_vecs']
    logger.info(f"Generated {len(vectors)} vectors of dimension {len(vectors[0])}")
    
    # Setup and populate database
    _setup_milvus_database(db_path, collection_name, model.dimension, documents, vectors, metadata)
    
    logger.success(f"Successfully built embeddings for '{corpus_path.name}' using {model.name}")
    logger.info(f"Database file: {db_path}")
    logger.info(f"Records: {len(documents)}")


def _setup_milvus_database(db_path: Path, collection_name: str, dimension: int, 
                          documents: List[str], vectors: List, metadata: List[Dict]) -> None:
    """Setup Milvus database and insert data"""
    logger.info("Setting up vector database...")
    client = MilvusClient(str(db_path))
    
    # Drop existing collection if exists
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)
        logger.info("Dropped existing collection")
    
    # Create new collection
    actual_dimension = dimension if dimension > 0 else len(vectors[0])
    client.create_collection(collection_name, dimension=actual_dimension)
    logger.info(f"Created collection with dimension {actual_dimension}")
    
    # Prepare and insert data
    logger.info("Preparing insert data...")
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
    
    logger.info("Inserting data into database...")
    client.insert(collection_name, insert_data)


def build_corpus_embedding_safe(corpus_path: Path, model_name: str, force: bool) -> bool:
    """Safe wrapper that returns success/failure without throwing"""
    try:
        build_corpus_embedding(corpus_path, model_name, force)
        return True
    
    except (ValueError, FileNotFoundError) as e:
        logger.error(f"Configuration error for {corpus_path}: {e}")
        return False
    
    except Exception as e:
        logger.error(f"Unexpected error processing {corpus_path}: {e}")
        return False


def find_corpus_files_in_folder(folder_path: Path) -> List[Path]:
    """Recursively find all corpus files in folder"""
    if not folder_path.is_dir():
        raise ValueError(f"Not a directory: {folder_path}")
    
    corpus_files = []
    for file_path in folder_path.rglob("*"):
        if file_path.is_file() and file_path.suffix in SUPPORTED_EXTENSIONS:
            corpus_files.append(file_path)
    
    return sorted(corpus_files)


def handle_list_commands(args) -> bool:
    """Handle --list and --list-models flags"""
    if args.list:
        corpus_files = get_corpus_files()
        if corpus_files:
            logger.info("Available corpora:")
            for corpus in corpus_files:
                logger.info(f"  - {corpus['name']}: {corpus['path']}")
        else:
            logger.warning("No corpus files found")
        return True
    
    if args.list_models:
        logger.info("Available models:")
        for model_name in AVAILABLE_MODELS.keys():
            logger.info(f"  - {model_name}")
        return True
    
    return False


def handle_build_all(args) -> None:
    """Handle --all flag"""
    corpus_files = get_corpus_files()
    if not corpus_files:
        logger.warning("No corpus files found")
        return
    
    logger.info(f"Building embeddings for {len(corpus_files)} corpora using {args.model}")
    
    success_count = 0
    for i, corpus_info in enumerate(corpus_files, 1):
        logger.info(f"Processing corpus {i}/{len(corpus_files)}: {corpus_info['name']}")
        if build_corpus_embedding_safe(Path(corpus_info['path']), args.model, args.force):
            success_count += 1
    
    logger.success(f"Completed: {success_count}/{len(corpus_files)} successful")


def handle_build_folder(args) -> None:
    """Handle --folder flag"""
    folder_path = Path(args.folder)
    
    try:
        corpus_files = find_corpus_files_in_folder(folder_path)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)
    
    if not corpus_files:
        logger.warning(f"No corpus files found in {folder_path}")
        return
    
    logger.info(f"Found {len(corpus_files)} corpus files in {folder_path}")
    logger.info(f"Building embeddings using model: {args.model}")
    
    success_count = 0
    for i, corpus_file in enumerate(corpus_files, 1):
        relative_path = corpus_file.relative_to(folder_path)
        logger.info(f"Processing {i}/{len(corpus_files)}: {relative_path}")
        
        if build_corpus_embedding_safe(corpus_file, args.model, args.force):
            success_count += 1
    
    logger.success(f"Completed folder processing: {success_count}/{len(corpus_files)} successful")


def handle_single_corpus(args) -> None:
    """Handle single corpus processing"""
    corpus_path = Path(args.corpus_path)
    
    if not corpus_path.exists():
        logger.error(f"File not found: {args.corpus_path}")
        sys.exit(1)
    
    if corpus_path.suffix not in SUPPORTED_EXTENSIONS:
        logger.error(f"Unsupported file type: {corpus_path.suffix}. Supported: {', '.join(SUPPORTED_EXTENSIONS)}")
        sys.exit(1)
    
    if not build_corpus_embedding_safe(corpus_path, args.model, args.force):
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


def main() -> None:
    """Main entry point with clear command routing"""
    args = parse_arguments()
    
    # Handle list commands first
    if handle_list_commands(args):
        return
    
    # Route to appropriate handler
    if args.all:
        handle_build_all(args)
    elif args.folder:
        handle_build_folder(args)
    elif args.corpus_path:
        handle_single_corpus(args)
    else:
        # No action specified - show help
        parse_arguments().print_help()


if __name__ == "__main__":
    main()