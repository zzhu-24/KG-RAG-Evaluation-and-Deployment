import os
import logging
from datetime import datetime

from lightrag import LightRAG, QueryParam
from lightrag.llm.hf import hf_model_complete, hf_embed
from lightrag.utils import EmbeddingFunc
from transformers import AutoModel, AutoTokenizer
import torch

import asyncio
import nest_asyncio

nest_asyncio.apply()

# 创建日志目录
LOG_DIR = "./log"
os.makedirs(LOG_DIR, exist_ok=True)

# 生成带时间戳的日志文件名
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(LOG_DIR, f"lightrag_hf_demo_{timestamp}.log")

# 配置日志：同时输出到文件和控制台
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

WORKING_DIR = "./cache/paper_demo"
os.makedirs(WORKING_DIR, exist_ok=True)

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def initialize_rag():
    logger.info("Initializing LightRAG with working directory: %s", WORKING_DIR)
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=hf_model_complete,
        llm_model_name="Qwen/Qwen2.5-1.5B-Instruct",
        embedding_func=EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=5000,
            func=lambda texts: hf_embed(
                texts,
                tokenizer=AutoTokenizer.from_pretrained(
                    "/home/infres/zzhu-24/large_files/bge-m3"
                ),
                embed_model=AutoModel.from_pretrained(
                    "/home/infres/zzhu-24/large_files/bge-m3"
                ),
            ),
        ),
    )

    logger.info("Initializing storages...")
    await rag.initialize_storages()  # Auto-initializes pipeline_status
    logger.info("Storages initialized.")
    return rag


def main():
    rag = asyncio.run(initialize_rag())

    with open("./paper.txt", "r", encoding="utf-8") as f:
        paper_content = f.read()
        logger.info("Inserting paper content (%d chars)...", len(paper_content))
        rag.insert(paper_content)
        logger.info("Insertion complete.")

    # Perform naive search
    question = "What are the main research themes and contributions in this paper?"
    logger.info("Running naive query...")
    logger.info(
        "Naive result: %s", rag.query(question, param=QueryParam(mode="naive"))
    )

    # Perform local search
    logger.info("Running local query...")
    logger.info(
        "Local result: %s", rag.query(question, param=QueryParam(mode="local"))
    )

    # Perform global search
    logger.info("Running global query...")
    logger.info(
        "Global result: %s", rag.query(question, param=QueryParam(mode="global"))
    )

    # Perform hybrid search
    logger.info("Running hybrid query...")
    logger.info(
        "Hybrid result: %s", rag.query(question, param=QueryParam(mode="hybrid"))
    )


if __name__ == "__main__":
    main()
