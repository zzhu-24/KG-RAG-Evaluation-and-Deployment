#!/usr/bin/env python3
"""
F1QA LightRAG Document Status Viewing Script
Display status statistics and detailed information for all documents
"""

import os
import logging
import argparse
from datetime import datetime
from typing import Dict

from lightrag import LightRAG
from lightrag.llm.hf import hf_model_complete, hf_embed
from lightrag.utils import EmbeddingFunc
from lightrag.base import DocStatus
from transformers import AutoModel, AutoTokenizer
import torch

import asyncio
import nest_asyncio

nest_asyncio.apply()

# Create log directory
LOG_DIR = "./log"
os.makedirs(LOG_DIR, exist_ok=True)

# Generate log filename with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(LOG_DIR, f"f1qa_status_{timestamp}.log")

# Configure logging: output to both file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)
logger.info("Log file: %s", log_file)

# Detect and configure GPU device
device = "cuda" if torch.cuda.is_available() else "cpu"

WORKING_DIR = "./cache/f1qa_demo"


async def initialize_rag():
    """Initialize LightRAG"""
    logger.info("Initializing LightRAG with working directory: %s", WORKING_DIR)
    logger.info(f"Using device: {device}")
    
    # Load embedding model to GPU
    logger.info("Loading embedding model to GPU...")
    embed_tokenizer = AutoTokenizer.from_pretrained(
        "/home/infres/zzhu-24/large_files/bge-m3"
    )
    embed_model = AutoModel.from_pretrained(
        "/home/infres/zzhu-24/large_files/bge-m3",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    logger.info(f"Embedding model loaded to: {next(embed_model.parameters()).device}")
    
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=hf_model_complete,
        llm_model_name="Qwen/Qwen2.5-3B-Instruct",
        embedding_func=EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=5000,
            func=lambda texts: hf_embed(
                texts,
                tokenizer=embed_tokenizer,
                embed_model=embed_model,
            ),
        ),
    )

    logger.info("Initializing storages...")
    await rag.initialize_storages()
    logger.info("Storages initialized.")
    return rag


async def print_document_status(rag: LightRAG, show_details: bool = False, status_filter: str = None):
    """
    Print document status statistics and detailed information
    
    Args:
        rag: LightRAG instance
        show_details: Whether to show detailed information
        status_filter: Only show documents with specific status (pending, processing, processed, failed, preprocessed)
    """
    logger.info("=" * 80)
    logger.info("Document Status Statistics")
    logger.info("=" * 80)
    
    # Get documents for all statuses
    statuses = [
        DocStatus.PENDING,
        DocStatus.PROCESSING,
        DocStatus.PREPROCESSED,
        DocStatus.PROCESSED,
        DocStatus.FAILED,
    ]
    
    # If status filter is specified, only get that status
    if status_filter:
        status_filter_lower = status_filter.lower()
        status_map = {
            "pending": DocStatus.PENDING,
            "processing": DocStatus.PROCESSING,
            "preprocessed": DocStatus.PREPROCESSED,
            "processed": DocStatus.PROCESSED,
            "failed": DocStatus.FAILED,
        }
        if status_filter_lower in status_map:
            statuses = [status_map[status_filter_lower]]
        else:
            logger.error(f"Invalid status filter: {status_filter}. Valid values: pending, processing, preprocessed, processed, failed")
            return
    
    # Get documents for all statuses
    all_docs = {}
    status_counts = {}
    
    for status in statuses:
        docs = await rag.doc_status.get_docs_by_status(status)
        all_docs[status] = docs
        status_counts[status] = len(docs)
    
    # Display statistics
    total_docs = sum(status_counts.values())
    logger.info(f"\nTotal documents: {total_docs}")
    logger.info("\nDocument count by status:")
    logger.info(f"  - PENDING: {status_counts.get(DocStatus.PENDING, 0)}")
    logger.info(f"  - PROCESSING: {status_counts.get(DocStatus.PROCESSING, 0)}")
    logger.info(f"  - PREPROCESSED: {status_counts.get(DocStatus.PREPROCESSED, 0)}")
    logger.info(f"  - PROCESSED: {status_counts.get(DocStatus.PROCESSED, 0)}")
    logger.info(f"  - FAILED: {status_counts.get(DocStatus.FAILED, 0)}")
    
    # Display detailed information
    if show_details:
        logger.info("\n" + "=" * 80)
        logger.info("Document Detailed Information")
        logger.info("=" * 80)
        
        for status in statuses:
            docs = all_docs[status]
            if not docs:
                continue
            
            logger.info(f"\n{'=' * 80}")
            logger.info(f"Status: {status.upper()} ({len(docs)} documents)")
            logger.info("=" * 80)
            
            # Sort by update time
            sorted_docs = sorted(
                docs.items(),
                key=lambda x: x[1].updated_at if hasattr(x[1], 'updated_at') else "",
                reverse=True
            )
            
            for idx, (doc_id, doc_status) in enumerate(sorted_docs, 1):
                logger.info(f"\n[{idx}] Document ID: {doc_id}")
                logger.info(f"    File path: {doc_status.file_path[:100]}")
                logger.info(f"    Content summary: {doc_status.content_summary[:100]}...")
                logger.info(f"    Content length: {doc_status.content_length}")
                logger.info(f"    Created at: {doc_status.created_at}")
                logger.info(f"    Updated at: {doc_status.updated_at}")
                
                if hasattr(doc_status, 'chunks_count') and doc_status.chunks_count is not None:
                    logger.info(f"    Chunks count: {doc_status.chunks_count}")
                
                if hasattr(doc_status, 'track_id') and doc_status.track_id:
                    logger.info(f"    Track ID: {doc_status.track_id}")
                
                if status == DocStatus.FAILED:
                    error_msg = getattr(doc_status, 'error_msg', None)
                    if error_msg:
                        logger.info(f"    Error message: {error_msg[:200]}")
                
                if hasattr(doc_status, 'metadata') and doc_status.metadata:
                    logger.info(f"    Metadata: {doc_status.metadata}")
    
    # Display error message summary for failed documents
    failed_docs = all_docs.get(DocStatus.FAILED, {})
    if failed_docs:
        logger.info("\n" + "=" * 80)
        logger.info("Failed Documents Error Message Summary")
        logger.info("=" * 80)
        for doc_id, doc_status in list(failed_docs.items())[:10]:  # Only show first 10
            error_msg = getattr(doc_status, 'error_msg', 'Unknown error')
            logger.info(f"\nDocument: {doc_status.file_path[:80]}")
            logger.info(f"  Error: {error_msg[:200]}")
        
        if len(failed_docs) > 10:
            logger.info(f"\n... {len(failed_docs) - 10} more failed documents not shown")


async def main_async():
    """Async main function"""
    parser = argparse.ArgumentParser(description="F1QA LightRAG Document Status Viewing Script")
    parser.add_argument(
        "--show-details",
        action="store_true",
        help="Show detailed information (including detailed information for each document)"
    )
    parser.add_argument(
        "--status",
        type=str,
        default=None,
        choices=["pending", "processing", "preprocessed", "processed", "failed"],
        help="Only show documents with specific status"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize RAG
        rag = await initialize_rag()
        
        # Print document status
        await print_document_status(rag, show_details=args.show_details, status_filter=args.status)
        
        logger.info("\n" + "=" * 80)
        logger.info("Status viewing completed")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Program execution failed: {e}", exc_info=True)
        return 1
    
    return 0


def main():
    """Main function wrapper"""
    return asyncio.run(main_async())


if __name__ == "__main__":
    exit(main())
