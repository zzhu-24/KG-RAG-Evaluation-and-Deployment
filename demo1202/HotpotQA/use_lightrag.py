"""
LightRAG Evaluation on HotpotQA (Multi-Mode)
============================================

- Dataset: HotpotQA (distractor)
- Retrieval modes: naive / local / global / hybrid
- Metrics: Exact Match (EM), F1
- Framework: Official LightRAG (HKUDS)


"""

import os
import json
import re
import string
import logging
from pathlib import Path
from collections import Counter
from typing import List, Dict

import torch
import asyncio
import nest_asyncio

from lightrag import LightRAG, QueryParam
from lightrag.llm.hf import hf_model_complete, hf_embed
from lightrag.utils import EmbeddingFunc
from transformers import AutoTokenizer, AutoModel

nest_asyncio.apply()

# ======================================================
# Logging
# ======================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ======================================================
# Config
# ======================================================

DATA_PATH = "./data/hotpot_dev_distractor_v1.json"
WORKING_DIR = "./cache/hotpotqa_lightrag"

LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"
embed_model_name = "BAAI/bge-m3"

MAX_SAMPLES = 500        # None for full dev set
RETRIEVAL_MODES = ["naive", "local", "global", "hybrid"]

os.makedirs(WORKING_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# ======================================================
# HotpotQA utilities
# ======================================================

def load_hotpot(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_documents(hotpot_data: List[Dict]) -> Dict[str, str]:
    """
    Extract unique documents from HotpotQA
    title -> concatenated text
    """
    documents = {}

    for example in hotpot_data:
        for title, sentences in example["context"]:
            if title not in documents:
                documents[title] = " ".join(sentences)

    return documents


# ======================================================
# HotpotQA official-style evaluation
# ======================================================

def normalize_answer(s: str) -> str:
    def lower(text): return text.lower()
    def remove_punc(text): return "".join(c for c in text if c not in string.punctuation)
    def remove_articles(text): return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text): return " ".join(text.split())

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()

    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# ======================================================
# Initialize LightRAG (official API)
# ======================================================

async def initialize_rag() -> LightRAG:
    logger.info("Initializing LightRAG")

    embed_tokenizer = AutoTokenizer.from_pretrained(embed_model_name)
    embed_model = AutoModel.from_pretrained(
        embed_model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)


    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=hf_model_complete,
        llm_model_name=LLM_MODEL,
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

    await rag.initialize_storages()
    return rag


# ======================================================
# Main pipeline
# ======================================================

def main():
    logger.info("Loading HotpotQA dataset")
    hotpot = load_hotpot(DATA_PATH)

    if MAX_SAMPLES:
        hotpot = hotpot[:MAX_SAMPLES]

    logger.info("Extracting unique documents")
    documents = extract_documents(hotpot)
    logger.info(f"Unique documents: {len(documents)}")

    logger.info("Initializing LightRAG")
    rag = asyncio.run(initialize_rag())

    # --------------------------------------------------
    # Insert documents (only once)
    # --------------------------------------------------
    logger.info("Inserting documents into LightRAG")
    for title, text in documents.items():
        rag.insert(
            text,
            ids=f"HotpotQA::{title}"
        )
    logger.info("Document insertion completed")

    # --------------------------------------------------
    # Evaluate for each retrieval mode
    # --------------------------------------------------
    summary = {}

    for mode in RETRIEVAL_MODES:
        logger.info("=" * 60)
        logger.info(f"Evaluating retrieval mode: {mode.upper()}")
        logger.info("=" * 60)

        em_total, f1_total = 0.0, 0.0
        n = 0

        for example in hotpot:
            question = example["question"]
            gold_answer = example["answer"]

            try:
                pred = rag.query(
                    question,
                    param=QueryParam(mode=mode)
                )
            except Exception as e:
                logger.error(f"Query failed ({mode}): {e}")
                continue

            em_total += exact_match(pred, gold_answer)
            f1_total += f1_score(pred, gold_answer)
            n += 1

            if n % 50 == 0:
                logger.info(
                    f"[{mode}] {n} samples | EM={em_total/n:.3f} | F1={f1_total/n:.3f}"
                )

        em = em_total / n if n else 0.0
        f1 = f1_total / n if n else 0.0

        summary[mode] = {
            "samples": n,
            "EM": em,
            "F1": f1
        }

        logger.info(f"[{mode.upper()}] FINAL | EM={em:.4f} | F1={f1:.4f}")

    # --------------------------------------------------
    # Print final summary (paper-ready)
    # --------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("FINAL SUMMARY — HotpotQA × LightRAG")
    logger.info("=" * 70)
    for mode, res in summary.items():
        logger.info(
            f"{mode.upper():<8} | "
            f"EM = {res['EM']:.4f} | "
            f"F1 = {res['F1']:.4f} | "
            f"N = {res['samples']}"
        )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
