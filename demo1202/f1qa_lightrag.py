#!/usr/bin/env python3
"""
F1QA LightRAG Script
Load data from jsonl format news files to LightRAG and answer questions
"""

import os
import json
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path

from lightrag import LightRAG, QueryParam
from lightrag.llm.hf import hf_model_complete, hf_embed
from lightrag.utils import EmbeddingFunc
from lightrag.base import DocStatus
from lightrag.kg.shared_storage import get_namespace_data
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
log_file = os.path.join(LOG_DIR, f"f1qa_lightrag_{timestamp}.log")

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


def load_questions(questions_file: Optional[str] = None) -> List[str]:
    """
    Load question list from file
    
    Args:
        questions_file: Question file path (one question per line)
        
    Returns:
        List of questions
    """
    if questions_file is None:
        return []
    
    questions_path = Path(questions_file)
    if not questions_path.exists():
        logger.warning(f"Question file does not exist: {questions_path}")
        return []
    
    questions = []
    with open(questions_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(line)
    
    logger.info(f"Loaded {len(questions)} questions from file")
    return questions


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


def insert_news_to_rag(rag: LightRAG, news_list: List[Dict]):
    """
    Insert news data into LightRAG
    
    Args:
        rag: LightRAG instance
        news_list: List of news
    """
    logger.info(f"Starting to insert {len(news_list)} news items into LightRAG...")
    
    # Prepare data for batch insertion
    texts = []
    file_paths = []
    
    for idx, news_item in enumerate(news_list, 1):
        try:
            # Format news text
            news_text = format_news_for_insertion(news_item)
            texts.append(news_text)
            
            # Use URL as file_path, or use title if no URL
            file_path = news_item.get("url", f"news_{idx}_{news_item.get('title', 'unknown')[:50]}")
            file_paths.append(file_path)
            
        except Exception as e:
            logger.error(f"Error processing news item {idx}: {e}")
            continue
    
    # Batch insertion
    if texts:
        try:
            logger.info(f"Batch inserting {len(texts)} news items...")
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
                        logger.info(f"Inserted {idx}/{len(texts)} news items")
                except Exception as e2:
                    logger.error(f"Error inserting news item {idx}: {e2}")
                    continue
    
    logger.info(f"Successfully submitted {len(texts)} news items to LightRAG")


async def show_processing_status(rag: LightRAG, check_interval: int = 5, max_checks: int = None):
    """
    显示处理进度
    
    Args:
        rag: LightRAG instance
        check_interval: 检查间隔（秒）
        max_checks: 最大检查次数，None表示无限检查直到完成
    """
    from lightrag.kg.shared_storage import get_namespace_data
    
    check_count = 0
    last_message = ""
    
    logger.info("=" * 60)
    logger.info("开始监控处理进度...")
    logger.info("=" * 60)
    
    while True:
        try:
            # 获取pipeline状态
            pipeline_status = await get_namespace_data("pipeline_status", workspace=rag.workspace)
            
            # 获取文档状态统计
            pending_docs = await rag.doc_status.get_docs_by_status(DocStatus.PENDING)
            processing_docs = await rag.doc_status.get_docs_by_status(DocStatus.PROCESSING)
            processed_docs = await rag.doc_status.get_docs_by_status(DocStatus.PROCESSED)
            failed_docs = await rag.doc_status.get_docs_by_status(DocStatus.FAILED)
            
            total_docs = len(pending_docs) + len(processing_docs) + len(processed_docs) + len(failed_docs)
            
            # 显示状态
            busy = pipeline_status.get("busy", False)
            latest_message = pipeline_status.get("latest_message", "")
            job_name = pipeline_status.get("job_name", "-")
            cur_batch = pipeline_status.get("cur_batch", 0)
            batchs = pipeline_status.get("batchs", 0)
            
            # 只在状态变化时输出
            if latest_message != last_message or check_count == 0:
                logger.info("\n" + "=" * 60)
                logger.info(f"处理状态: {'进行中' if busy else '空闲'}")
                logger.info(f"任务名称: {job_name}")
                if batchs > 0:
                    logger.info(f"批次进度: {cur_batch}/{batchs}")
                logger.info(f"文档统计:")
                logger.info(f"  - 待处理 (PENDING): {len(pending_docs)}")
                logger.info(f"  - 处理中 (PROCESSING): {len(processing_docs)}")
                logger.info(f"  - 已完成 (PROCESSED): {len(processed_docs)}")
                logger.info(f"  - 失败 (FAILED): {len(failed_docs)}")
                logger.info(f"  - 总计: {total_docs}")
                if latest_message:
                    logger.info(f"最新消息: {latest_message}")
                logger.info("=" * 60)
                last_message = latest_message
            
            # 检查是否完成
            if not busy and len(pending_docs) == 0 and len(processing_docs) == 0:
                if len(processed_docs) > 0 or len(failed_docs) > 0:
                    logger.info("\n" + "=" * 60)
                    logger.info("处理完成！")
                    logger.info(f"成功处理: {len(processed_docs)} 个文档")
                    if len(failed_docs) > 0:
                        logger.warning(f"失败: {len(failed_docs)} 个文档")
                    logger.info("=" * 60)
                    break
            
            # 检查最大检查次数
            check_count += 1
            if max_checks and check_count >= max_checks:
                logger.info(f"\n已达到最大检查次数 ({max_checks})，停止监控")
                break
            
            # 等待下次检查
            await asyncio.sleep(check_interval)
            
        except Exception as e:
            logger.error(f"检查状态时出错: {e}")
            await asyncio.sleep(check_interval)
            continue


def query_rag(rag: LightRAG, question: str, mode: str = "hybrid") -> str:
    """
    Answer question using LightRAG
    
    Args:
        rag: LightRAG instance
        question: Question
        mode: Query mode ("naive", "local", "global", "hybrid")
        
    Returns:
        Answer
    """
    try:
        answer = rag.query(question, param=QueryParam(mode=mode))
        return answer
    except Exception as e:
        logger.error(f"Error querying question: {e}")
        return f"Error: {str(e)}"


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="F1QA LightRAG Q&A System")
    parser.add_argument(
        "--news-file",
        type=str,
        default=DEFAULT_NEWS_FILE,
        help=f"News jsonl file path (default: {DEFAULT_NEWS_FILE})"
    )
    parser.add_argument(
        "--questions-file",
        type=str,
        default=None,
        help="Question file path (one question per line, optional)"
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Single question (if provided, will directly answer this question)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="hybrid",
        choices=["naive", "local", "global", "hybrid"],
        help="Query mode (default: hybrid)"
    )
    parser.add_argument(
        "--skip-insert",
        action="store_true",
        help="Skip news insertion step (if data is already inserted)"
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Show processing progress (default: False)"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize RAG
        rag = asyncio.run(initialize_rag())
        
        # 确保pipeline_status已初始化
        from lightrag.kg.shared_storage import initialize_pipeline_status
        asyncio.run(initialize_pipeline_status(workspace=rag.workspace))
        
        # Load and insert news data
        if not args.skip_insert:
            logger.info("=" * 60)
            logger.info("Loading news data")
            logger.info("=" * 60)
            news_list = load_news_from_jsonl(args.news_file)
            
            logger.info("=" * 60)
            logger.info("Inserting news into LightRAG")
            logger.info("=" * 60)
            insert_news_to_rag(rag, news_list)
            
            # 显示处理进度（如果启用）
            if args.show_progress:
                logger.info("\n等待处理开始...")
                time.sleep(2)  # 等待处理开始
                asyncio.run(show_processing_status(rag, check_interval=5))
            else:
                logger.info("提示: 使用 --show-progress 参数可以查看详细处理进度")
        else:
            logger.info("Skipping news insertion step")
        
        # Process questions
        questions = []
        
        if args.question:
            # Single question mode
            questions = [args.question]
        elif args.questions_file:
            # Load questions from file
            questions = load_questions(args.questions_file)
        else:
            # Interactive mode
            logger.info("=" * 60)
            logger.info("Entering interactive Q&A mode (type 'quit' or 'exit' to exit)")
            logger.info("=" * 60)
            
            while True:
                try:
                    question = input("\nPlease enter your question: ").strip()
                    if not question:
                        continue
                    if question.lower() in ["quit", "exit", "q"]:
                        break
                    
                    logger.info(f"Question: {question}")
                    answer = query_rag(rag, question, mode=args.mode)
                    print(f"\nAnswer: {answer}\n")
                    
                except KeyboardInterrupt:
                    logger.info("\n\nExiting interactive mode")
                    break
                except Exception as e:
                    logger.error(f"Error processing question: {e}")
                    continue
            
            return 0
        
        # Batch process questions
        if questions:
            logger.info("=" * 60)
            logger.info(f"Starting to answer {len(questions)} questions (mode: {args.mode})")
            logger.info("=" * 60)
            
            for idx, question in enumerate(questions, 1):
                logger.info(f"\n[{idx}/{len(questions)}] Question: {question}")
                answer = query_rag(rag, question, mode=args.mode)
                print(f"Answer: {answer}\n")
        
        logger.info("=" * 60)
        logger.info("Completed")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Program execution failed: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

