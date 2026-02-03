import os
import logging
import json
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple

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
log_file = os.path.join(LOG_DIR, f"metaqa_lightrag_{timestamp}.log")

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

# 检测并配置GPU设备
device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    logger.info(f"使用GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA版本: {torch.version.cuda}")
    logger.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    logger.warning("未检测到GPU，将使用CPU")

WORKING_DIR = "./cache/metaqa_demo"
os.makedirs(WORKING_DIR, exist_ok=True)

# MetaQA数据路径（从项目根目录运行）
METAQA_DIR = "./demo1202/MetaQA"
KB_FILE = os.path.join(METAQA_DIR, "kb.txt")
QA_TEST_FILE = os.path.join(METAQA_DIR, "2-hop/vanilla/qa_test.txt")

llm_model_name = "Qwen/Qwen2.5-3B-Instruct"
embed_model_name = "/home/infres/zzhu-24/large_files/bge-m3"

logger.info(f"Using LLM model: {llm_model_name}")
logger.info(f"Using embedding model: {embed_model_name}")
logger.info(f"Using working directory: {WORKING_DIR}")

def load_kg_triples(kb_path: str) -> List[Tuple[str, str, str]]:
    """
    从MetaQA格式的kb.txt文件加载知识图谱三元组
    格式: subject|predicate|object
    """
    triples = []
    with open(kb_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) >= 3:
                s, p, o = parts[0].strip(), parts[1].strip(), parts[2].strip()
                triples.append((s, p, o))
            else:
                logger.warning(f"Line {line_num} has invalid format: {line}")
    logger.info(f"Loaded {len(triples)} triples from {kb_path}")
    return triples


def load_qa_pairs(qa_path: str) -> List[Tuple[str, str]]:
    """
    从MetaQA格式的qa文件加载问题-答案对
    格式: question\tanswer (答案可能包含多个，用|分隔)
    """
    qa_pairs = []
    with open(qa_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            if "\t" in line:
                parts = line.split("\t", 1)
                question = parts[0].strip()
                answer = parts[1].strip() if len(parts) > 1 else ""
                qa_pairs.append((question, answer))
            else:
                logger.warning(f"Line {line_num} has invalid format: {line}")
    logger.info(f"Loaded {len(qa_pairs)} QA pairs from {qa_path}")
    return qa_pairs


def convert_triples_to_custom_kg(triples: List[Tuple[str, str, str]]) -> Dict:
    """
    将MetaQA三元组转换为LightRAG的insert_custom_kg格式
    """
    # 收集所有实体及其信息
    entities_dict: Dict[str, Dict] = {}
    relationships = []
    
    # 统计每个实体出现的次数和关系类型，用于生成描述
    entity_relations: Dict[str, List[str]] = defaultdict(list)
    entity_types: Dict[str, set] = defaultdict(set)
    
    for s, p, o in triples:
        # 收集实体
        if s not in entities_dict:
            entities_dict[s] = {
                "entity_name": s,
                "entity_type": "UNKNOWN",
                "description": f"Entity: {s}",
                "source_id": "metaqa_kb",
                "file_path": "demo1202/MetaQA/kb.txt",
            }
        if o not in entities_dict:
            entities_dict[o] = {
                "entity_name": o,
                "entity_type": "UNKNOWN",
                "description": f"Entity: {o}",
                "source_id": "metaqa_kb",
                "file_path": "demo1202/MetaQA/kb.txt",
            }
        
        # 记录关系类型
        entity_relations[s].append(p)
        entity_relations[o].append(p)
        
        # 根据关系类型推断实体类型
        if p in ["directed_by", "written_by", "starred_actors", "has_genre", "release_year", "in_language", "has_tags"]:
            entity_types[s].add("Movie")
        if p == "directed_by":
            entity_types[o].add("Person")
        elif p == "written_by":
            entity_types[o].add("Person")
        elif p == "starred_actors":
            entity_types[o].add("Person")
        elif p == "has_genre":
            entity_types[o].add("Genre")
        
        # 创建关系
        relationships.append({
            "src_id": s,
            "tgt_id": o,
            "description": f"{s} {p} {o}",
            "keywords": p,
            "weight": 1.0,
            "source_id": "metaqa_kb",
            "file_path": "demo1202/MetaQA/kb.txt",
        })
    
    # 更新实体描述和类型
    for entity_name, entity_data in entities_dict.items():
        relations = entity_relations.get(entity_name, [])
        if relations:
            unique_relations = list(set(relations))[:10]  # 限制描述长度
            entity_data["description"] = f"{entity_name} (relations: {', '.join(unique_relations)})"
        
        # 设置实体类型（优先使用推断的类型）
        if entity_name in entity_types:
            types = entity_types[entity_name]
            if len(types) == 1:
                entity_data["entity_type"] = list(types)[0]
            elif "Movie" in types:
                entity_data["entity_type"] = "Movie"
            elif "Person" in types:
                entity_data["entity_type"] = "Person"
    
    entities = list(entities_dict.values())
    
    logger.info(f"Converted to {len(entities)} entities and {len(relationships)} relationships")
    
    return {
        "entities": entities,
        "relationships": relationships,
        "chunks": [],  # MetaQA是纯KG，不需要chunks
    }


async def initialize_rag():
    """初始化LightRAG"""
    logger.info("Initializing LightRAG with working directory: %s", WORKING_DIR)
    logger.info(f"使用设备: {device}")
    
    # 加载embedding模型到GPU
    logger.info("加载embedding模型到GPU...")
    embed_tokenizer = AutoTokenizer.from_pretrained(embed_model_name)
    embed_model = AutoModel.from_pretrained(embed_model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device)
    logger.info(f"Embedding模型已加载到: {next(embed_model.parameters()).device}")
    
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=hf_model_complete,
        llm_model_name=llm_model_name,
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


def normalize_answer(answer: str) -> str:
    """标准化答案字符串用于比较"""
    return answer.lower().replace(" ", "").strip()


def check_answer(predicted: str, gold: str) -> bool:
    """
    检查预测答案是否正确
    支持多个答案（用|分隔）
    """
    pred_normalized = normalize_answer(predicted)
    gold_normalized = normalize_answer(gold)
    
    # 如果gold包含多个答案（用|分隔），检查pred是否包含任一答案
    if "|" in gold:
        gold_answers = [normalize_answer(a) for a in gold.split("|")]
        return any(gold_answer in pred_normalized for gold_answer in gold_answers)
    else:
        # 检查gold是否在pred中，或者pred是否在gold中
        return gold_normalized in pred_normalized or pred_normalized in gold_normalized


def evaluate(rag: LightRAG, qa_pairs: List[Tuple[str, str]], mode: str = "global", max_samples: int = None):
    """
    评估LightRAG在MetaQA数据集上的表现
    
    Args:
        rag: LightRAG实例
        qa_pairs: 问题-答案对列表
        mode: 查询模式 ("naive", "local", "global", "hybrid")
        max_samples: 最大评估样本数（None表示全部）
    """
    logger.info(f"Starting evaluation with mode={mode}, total QA pairs: {len(qa_pairs)}")
    
    if max_samples:
        qa_pairs = qa_pairs[:max_samples]
        logger.info(f"Limited to {max_samples} samples for evaluation")
    
    correct = 0
    results = []
    
    for idx, (question, gold_answer) in enumerate(qa_pairs, 1):
        try:
            # 执行查询
            predicted = rag.query(question, param=QueryParam(mode=mode))
            
            # 检查答案是否正确
            is_correct = check_answer(predicted, gold_answer)
            if is_correct:
                correct += 1
            
            results.append({
                "question": question,
                "gold_answer": gold_answer,
                "predicted": predicted,
                "correct": is_correct,
            })
            
            if idx % 100 == 0:
                logger.info(f"Processed {idx}/{len(qa_pairs)} questions, current accuracy: {correct/idx:.4f}")
            
        except Exception as e:
            logger.error(f"Error processing question {idx}: {question}, error: {e}")
            results.append({
                "question": question,
                "gold_answer": gold_answer,
                "predicted": "",
                "correct": False,
                "error": str(e),
            })
    
    accuracy = correct / len(qa_pairs) if qa_pairs else 0.0
    
    logger.info("=" * 60)
    logger.info(f"Evaluation Results (mode={mode}):")
    logger.info(f"Total questions: {len(qa_pairs)}")
    logger.info(f"Correct answers: {correct}")
    logger.info(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    logger.info("=" * 60)
    
    return {
        "mode": mode,
        "total": len(qa_pairs),
        "correct": correct,
        "accuracy": accuracy,
        "results": results,
    }


def main():
    """主函数"""
    # 初始化RAG
    rag = asyncio.run(initialize_rag())
    
    # 加载知识图谱
    logger.info("Loading knowledge graph from %s", KB_FILE)
    triples = load_kg_triples(KB_FILE)
    
    # 转换为LightRAG格式
    logger.info("Converting triples to LightRAG format...")
    custom_kg = convert_triples_to_custom_kg(triples)
    
    # 插入知识图谱
    logger.info("Inserting knowledge graph into LightRAG...")
    logger.info(f"  - Entities: {len(custom_kg['entities'])}")
    logger.info(f"  - Relationships: {len(custom_kg['relationships'])}")
    rag.insert_custom_kg(custom_kg)
    logger.info("Knowledge graph inserted successfully.")
    
    # 加载QA数据
    logger.info("Loading QA pairs from %s", QA_TEST_FILE)
    qa_pairs = load_qa_pairs(QA_TEST_FILE)
    
    # 评估不同模式
    evaluation_results = {}
    
    # # 评估global模式（KG-based retrieval）
    # logger.info("\n" + "=" * 60)
    # logger.info("Evaluating with GLOBAL mode (KG-based retrieval)")
    # logger.info("=" * 60)
    # eval_result_global = evaluate(rag, qa_pairs, mode="global", max_samples=None)
    # evaluation_results["global"] = eval_result_global
    
    # # 评估hybrid模式
    # logger.info("\n" + "=" * 60)
    # logger.info("Evaluating with HYBRID mode")
    # logger.info("=" * 60)
    # eval_result_hybrid = evaluate(rag, qa_pairs, mode="hybrid", max_samples=None)
    # evaluation_results["hybrid"] = eval_result_hybrid
    
    # 评估local模式
    logger.info("\n" + "=" * 60)
    logger.info("Evaluating with LOCAL mode")
    logger.info("=" * 60)
    eval_result_local = evaluate(rag, qa_pairs, mode="local", max_samples=None)
    evaluation_results["local"] = eval_result_local
    
    # 保存评估结果
    results_file = os.path.join(WORKING_DIR, f"evaluation_results_{timestamp}.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
    logger.info(f"\nEvaluation results saved to: {results_file}")
    
    # 打印总结
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    for mode, result in evaluation_results.items():
        logger.info(f"{mode.upper()} mode: Accuracy = {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

