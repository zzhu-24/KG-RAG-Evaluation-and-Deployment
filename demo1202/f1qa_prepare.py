#!/usr/bin/env python3
"""
F1QA LightRAG Data Preparation Script
Responsible for initializing LightRAG and inserting news data
Separated from query script to reduce memory usage
"""

import os
import json
import logging
import argparse
from datetime import datetime
from typing import List, Dict
from pathlib import Path

from lightrag import LightRAG
from lightrag.llm.hf import hf_model_complete, hf_embed
from lightrag.utils import EmbeddingFunc
from lightrag.base import DocStatus
from lightrag.kg.shared_storage import get_namespace_data, initialize_pipeline_status
from transformers import AutoModel, AutoTokenizer
import torch

import asyncio
import nest_asyncio
import time

nest_asyncio.apply()

# Create log directory
LOG_DIR = "./log"
os.makedirs(LOG_DIR, exist_ok=True)

# Generate log filename with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(LOG_DIR, f"f1qa_prepare_{timestamp}.log")

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
if torch.cuda.is_available():
    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA version: {torch.version.cuda}")
    logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    logger.warning("No GPU detected, will use CPU")

WORKING_DIR = "./cache/f1qa_demo"
os.makedirs(WORKING_DIR, exist_ok=True)

# Default data path
DEFAULT_NEWS_FILE = "./demo1202/F1QA/news/news_example.jsonl"


def load_news_from_jsonl(jsonl_path: str) -> List[Dict]:
    """
    Load news data from jsonl file
    
    Args:
        jsonl_path: jsonl file path
        
    Returns:
        List of news, each news is a dictionary
    """
    news_list = []
    news_path = Path(jsonl_path)
    
    if not news_path.exists():
        raise FileNotFoundError(f"News file not found: {news_path}")
    
    logger.info(f"Loading news file: {news_path}")
    
    with open(news_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                news_item = json.loads(line)
                # Validate required fields
                if "content" in news_item or "title" in news_item:
                    news_list.append(news_item)
                else:
                    logger.warning(f"Line {line_num} missing required fields: {line[:100]}...")
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num} JSON parsing failed: {e}")
                continue
    
    logger.info(f"Successfully loaded {len(news_list)} news items")
    return news_list


def format_news_for_insertion(news_item: Dict) -> str:
    """
    Format news item as text for insertion into LightRAG
    
    Args:
        news_item: News dictionary
        
    Returns:
        Formatted text
    """
    parts = []
    
    # Add title
    if "title" in news_item and news_item["title"]:
        parts.append(f"Title: {news_item['title']}")
    
    # Add published time
    if "published_time" in news_item and news_item["published_time"]:
        parts.append(f"Published time: {news_item['published_time']}")
    
    # Add summary
    if "summary" in news_item and news_item["summary"]:
        parts.append(f"Summary: {news_item['summary']}")
    
    # Add content (prefer content, use summary if not available)
    if "content" in news_item and news_item["content"]:
        parts.append(f"Content: {news_item['content']}")
    elif "summary" in news_item and news_item["summary"]:
        parts.append(f"Content: {news_item['summary']}")
    
    # Add URL (as metadata)
    if "url" in news_item and news_item["url"]:
        parts.append(f"Source: {news_item['url']}")
    
    return "\n".join(parts)


async def initialize_rag():
    """Initialize LightRAG for data preparation"""
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
    # Ensure pipeline_status is initialized
    await initialize_pipeline_status(workspace=rag.workspace)
    logger.info("Storages initialized.")
    return rag


async def insert_news_to_rag(rag: LightRAG, news_list: List[Dict], force_update: bool = False, skip_failed: bool = False):
    """
    Insert news data into LightRAG with incremental update support
    
    Args:
        rag: LightRAG instance
        news_list: List of news
        force_update: If True, re-insert even if document already exists (default: False)
        skip_failed: If True, skip documents with FAILED status (default: False)
    """
    logger.info(f"Starting to process {len(news_list)} news items...")
    
    # Check existing documents and filter out already processed ones
    texts = []
    file_paths = []
    skipped_count = 0
    skipped_failed_count = 0
    new_count = 0
    reprocess_count = 0
    
    for idx, news_item in enumerate(news_list, 1):
        try:
            # Generate file_path (use URL as primary identifier, fallback to title)
            file_path = news_item.get("url", f"news_{idx}_{news_item.get('title', 'unknown')[:50]}")
            
            # Check if document already exists
            existing_doc = await rag.doc_status.get_doc_by_file_path(file_path)
            
            if existing_doc:
                status = existing_doc.get("status", "unknown")
                
                # Skip processed documents (unless force_update)
                if status == "processed" and not force_update:
                    skipped_count += 1
                    if skipped_count <= 5 or skipped_count % 50 == 0:
                        logger.debug(f"Skipping already processed news: {file_path[:80]}... (status: {status})")
                    continue
                
                # Skip failed documents if skip_failed is True (unless force_update)
                if status == "failed" and skip_failed and not force_update:
                    skipped_failed_count += 1
                    error_msg = existing_doc.get("error_msg", "Unknown error")
                    if skipped_failed_count <= 5 or skipped_failed_count % 50 == 0:
                        logger.warning(f"Skipping failed news: {file_path[:80]}... (error: {error_msg[:100]})")
                    continue
                
                # Re-process other statuses or when force_update is True
                reprocess_count += 1
                if status == "failed":
                    error_msg = existing_doc.get("error_msg", "Unknown error")
                    logger.info(f"Re-processing failed news: {file_path[:80]}... (previous error: {error_msg[:100]})")
                elif status != "processed":
                    logger.info(f"Re-processing news with status '{status}': {file_path[:80]}...")
                else:
                    logger.info(f"Force re-processing news: {file_path[:80]}...")
            else:
                new_count += 1
            
            # Format news text
            news_text = format_news_for_insertion(news_item)
            texts.append(news_text)
            file_paths.append(file_path)
            
        except Exception as e:
            logger.error(f"Error processing news item {idx}: {e}")
            continue
    
    # Log summary
    logger.info("=" * 60)
    logger.info("Document Check Results:")
    logger.info(f"  - New documents: {new_count}")
    logger.info(f"  - Need reprocessing: {reprocess_count}")
    logger.info(f"  - Skipped (processed): {skipped_count}")
    if skip_failed:
        logger.info(f"  - Skipped (failed): {skipped_failed_count}")
    logger.info(f"  - Total to insert: {len(texts)}")
    logger.info("=" * 60)
    
    # Batch insertion
    if texts:
        try:
            logger.info(f"Batch inserting {len(texts)} news documents...")
            track_id = rag.insert(
                input=texts,
                file_paths=file_paths,
            )
            logger.info(f"Insertion task submitted, track_id: {track_id}")
            logger.info("Note: Insertion is asynchronous and may take some time to complete")
        except Exception as e:
            logger.error(f"Error during batch insertion: {e}")
            # If batch insertion fails, try inserting one by one
            logger.info("Trying to insert one by one...")
            for idx, (text, file_path) in enumerate(zip(texts, file_paths), 1):
                try:
                    rag.insert(
                        input=text,
                        file_paths=file_path,
                    )
                    if idx % 10 == 0:
                        logger.info(f"Inserted {idx}/{len(texts)} news documents")
                except Exception as e2:
                    logger.error(f"Error inserting news document {idx}: {e2}")
                    continue
    else:
        logger.info("No new documents to insert")
    
    logger.info(f"Successfully submitted {len(texts)} news documents to LightRAG")


async def show_processing_status(rag: LightRAG, check_interval: int = 5, max_checks: int = None):
    """
    Show processing progress
    
    Args:
        rag: LightRAG instance
        check_interval: Check interval (seconds)
        max_checks: Maximum number of checks, None means infinite checks until completion
    """
    check_count = 0
    last_message = ""
    
    logger.info("=" * 60)
    logger.info("Starting to monitor processing progress...")
    logger.info("=" * 60)
    
    while True:
        try:
            # Get pipeline status
            pipeline_status = await get_namespace_data("pipeline_status", workspace=rag.workspace)
            
            # Get document status statistics
            pending_docs = await rag.doc_status.get_docs_by_status(DocStatus.PENDING)
            processing_docs = await rag.doc_status.get_docs_by_status(DocStatus.PROCESSING)
            processed_docs = await rag.doc_status.get_docs_by_status(DocStatus.PROCESSED)
            failed_docs = await rag.doc_status.get_docs_by_status(DocStatus.FAILED)
            
            total_docs = len(pending_docs) + len(processing_docs) + len(processed_docs) + len(failed_docs)
            
            # Display status
            busy = pipeline_status.get("busy", False)
            latest_message = pipeline_status.get("latest_message", "")
            job_name = pipeline_status.get("job_name", "-")
            cur_batch = pipeline_status.get("cur_batch", 0)
            batchs = pipeline_status.get("batchs", 0)
            
            # Only output when status changes
            if latest_message != last_message or check_count == 0:
                logger.info("\n" + "=" * 60)
                logger.info(f"Processing status: {'Busy' if busy else 'Idle'}")
                logger.info(f"Job name: {job_name}")
                if batchs > 0:
                    logger.info(f"Batch progress: {cur_batch}/{batchs}")
                logger.info(f"Document statistics:")
                logger.info(f"  - Pending: {len(pending_docs)}")
                logger.info(f"  - Processing: {len(processing_docs)}")
                logger.info(f"  - Processed: {len(processed_docs)}")
                logger.info(f"  - Failed: {len(failed_docs)}")
                logger.info(f"  - Total: {total_docs}")
                if latest_message:
                    logger.info(f"Latest message: {latest_message}")
                logger.info("=" * 60)
                last_message = latest_message
            
            # Check if completed
            if not busy and len(pending_docs) == 0 and len(processing_docs) == 0:
                if len(processed_docs) > 0 or len(failed_docs) > 0:
                    logger.info("\n" + "=" * 60)
                    logger.info("Processing completed!")
                    logger.info(f"Successfully processed: {len(processed_docs)} documents")
                    if len(failed_docs) > 0:
                        logger.warning(f"Failed: {len(failed_docs)} documents")
                    logger.info("=" * 60)
                    break
            
            # Check maximum check count
            check_count += 1
            if max_checks and check_count >= max_checks:
                logger.info(f"\nReached maximum check count ({max_checks}), stopping monitoring")
                break
            
            # Wait for next check
            await asyncio.sleep(check_interval)
            
        except Exception as e:
            logger.error(f"Error checking status: {e}")
            await asyncio.sleep(check_interval)
            continue


async def main_async():
    """Async main function for data preparation"""
    parser = argparse.ArgumentParser(description="F1QA LightRAG Data Preparation Script (supports incremental updates)")
    parser.add_argument(
        "--news-file",
        type=str,
        default=DEFAULT_NEWS_FILE,
        help=f"News jsonl file path (default: {DEFAULT_NEWS_FILE})"
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Show processing progress (default: False)"
    )
    parser.add_argument(
        "--force-update",
        action="store_true",
        help="Force re-insert even if document already exists (default: False, incremental update)"
    )
    parser.add_argument(
        "--skip-failed",
        action="store_true",
        help="Skip documents with FAILED status (default: False, will retry failed documents)"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize RAG
        logger.info("=" * 60)
        logger.info("Initializing LightRAG")
        logger.info("=" * 60)
        rag = await initialize_rag()
        
        # Load and insert news data
        logger.info("=" * 60)
        logger.info("Loading news data")
        logger.info("=" * 60)
        news_list = load_news_from_jsonl(args.news_file)
        
        logger.info("=" * 60)
        if args.force_update:
            logger.info("Force update mode: Will re-insert all news data into LightRAG")
        else:
            logger.info("Incremental update mode: Only insert new or unprocessed news data")
        if args.skip_failed:
            logger.info("Skip failed documents mode: Will skip documents with FAILED status")
        logger.info("=" * 60)
        await insert_news_to_rag(rag, news_list, force_update=args.force_update, skip_failed=args.skip_failed)
        
        # Show processing progress (if enabled)
        if args.show_progress:
            logger.info("\nWaiting for processing to start...")
            await asyncio.sleep(2)  # Wait for processing to start
            await show_processing_status(rag, check_interval=5)
        else:
            logger.info("Tip: Use --show-progress parameter to view detailed processing progress")
            logger.info("Data insertion task has been submitted, processing will continue in the background")
            logger.info("You can use f1qa_query.py script to query (data may still be processing)")
        
        logger.info("=" * 60)
        logger.info("Data preparation completed")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Program execution failed: {e}", exc_info=True)
        return 1
    
    return 0


def main():
    """Main function wrapper"""
    return asyncio.run(main_async())


if __name__ == "__main__":
    exit(main())
