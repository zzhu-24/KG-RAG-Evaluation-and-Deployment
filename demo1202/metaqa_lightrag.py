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

# #region agent log
DEBUG_LOG_PATH = "/home/infres/zzhu-24/kg-rag/LightRAG/.cursor/debug.log"
def _debug_log(location, message, data, hypothesis_id):
    try:
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            log_entry = {
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": hypothesis_id,
                "location": location,
                "message": message,
                "data": data,
                "timestamp": int(datetime.now().timestamp() * 1000)
            }
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except: pass
# #endregion

# Create log directory
LOG_DIR = "./log"
os.makedirs(LOG_DIR, exist_ok=True)

# Generate log filename with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(LOG_DIR, f"metaqa_lightrag_{timestamp}.log")

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

# MetaQA data path (run from project root directory)
METAQA_DIR = "./demo1202/MetaQA"
KB_FILE = os.path.join(METAQA_DIR, "kb.txt")
QA_TEST_FILE = os.path.join(METAQA_DIR, "2-hop/vanilla/qa_test.txt")

# llm_model_name="Qwen/Qwen2.5-1.5B-Instruct"
# llm_model_name="Qwen/Qwen3-4B-Thinking-2507-FP8"
# llm_model_name="Qwen/Qwen3-4B-Instruct-2507"
llm_model_name="Qwen/Qwen2.5-3B-Instruct"
embed_model_name="/home/infres/zzhu-24/large_files/bge-m3"
logger.info(f"Using LLM model: {llm_model_name}")
logger.info(f"Using embedding model: {embed_model_name}")

def get_model_safe_name(model_name: str) -> str:
    """Convert model name to a filesystem-safe directory name"""
    # Replace slashes and other special characters with underscores
    safe_name = model_name.replace("/", "_").replace("\\", "_")
    safe_name = safe_name.replace(":", "_").replace(" ", "_")
    # Remove any other potentially problematic characters
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in safe_name)
    return safe_name

# Shared directory for knowledge graph data (entities, relationships, graph)
# This allows different LLM models to share the same knowledge graph
SHARED_KG_DIR = "./cache/metaqa_demo_shared"
os.makedirs(SHARED_KG_DIR, exist_ok=True)

# Model-specific directory for LLM query cache
# Each model has its own cache to avoid conflicts
model_dir_name = get_model_safe_name(llm_model_name)
MODEL_CACHE_DIR = f"./cache/metaqa_demo_{model_dir_name}"
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# Working directory: use model-specific directory for LLM cache isolation
WORKING_DIR = MODEL_CACHE_DIR
logger.info(f"Using shared KG directory: {SHARED_KG_DIR}")
logger.info(f"Using model-specific cache directory: {MODEL_CACHE_DIR}")
logger.info(f"Using working directory: {WORKING_DIR}")

def sync_kg_data_from_shared(shared_dir: str, working_dir: str):
    """
    Create symbolic links from shared directory to working directory.
    Graph data files are shared via links, but LLM cache is model-specific.
    This saves disk space by avoiding file duplication.
    
    Graph data files:
    - vdb_entities.json (vector storage for entities)
    - vdb_relationships.json (vector storage for relationships)
    - vdb_chunks.json (vector storage for chunks)
    - graph_chunk_entity_relation.graphml or kg.json (graph storage)
    """
    kg_files = [
        "vdb_entities.json",
        "vdb_relationships.json", 
        "vdb_chunks.json",
        "graph_chunk_entity_relation.graphml",
        "kg.json",
    ]
    
    # Convert to absolute paths for reliable symlink creation
    shared_dir = os.path.abspath(shared_dir)
    working_dir = os.path.abspath(working_dir)
    
    for filename in kg_files:
        shared_path = os.path.join(shared_dir, filename)
        working_path = os.path.join(working_dir, filename)
        
        # If file exists in shared dir but not in working dir, create symlink
        if os.path.exists(shared_path) and not os.path.exists(working_path):
            logger.info(f"Creating symlink for {filename} from shared directory to working directory")
            # Calculate relative path for symlink
            rel_path = os.path.relpath(shared_path, working_dir)
            os.symlink(rel_path, working_path)
        # If file exists in working dir but not in shared dir, move it to shared and create symlink
        elif os.path.exists(working_path) and not os.path.exists(shared_path):
            # Check if it's already a symlink
            if os.path.islink(working_path):
                logger.warning(f"{filename} in working dir is already a symlink but target doesn't exist, removing broken link")
                os.remove(working_path)
            else:
                logger.info(f"Moving {filename} from working directory to shared directory and creating symlink")
                import shutil
                shutil.move(working_path, shared_path)
                rel_path = os.path.relpath(shared_path, working_dir)
                os.symlink(rel_path, working_path)
        # If both exist, check if working is already a symlink pointing to shared
        elif os.path.exists(shared_path) and os.path.exists(working_path):
            if os.path.islink(working_path):
                # Check if symlink points to the correct target
                link_target = os.readlink(working_path)
                abs_link_target = os.path.abspath(os.path.join(os.path.dirname(working_path), link_target))
                if abs_link_target != os.path.abspath(shared_path):
                    logger.info(f"Updating symlink for {filename} to point to shared directory")
                    os.remove(working_path)
                    rel_path = os.path.relpath(shared_path, working_dir)
                    os.symlink(rel_path, working_path)
                # else: symlink is already correct, do nothing
            else:
                # Working path is a regular file, replace with symlink
                # Use the newer one (prefer shared if same timestamp)
                shared_mtime = os.path.getmtime(shared_path)
                working_mtime = os.path.getmtime(working_path)
                if shared_mtime >= working_mtime:
                    logger.info(f"Replacing {filename} in working directory with symlink to shared (shared is newer or same)")
                    os.remove(working_path)
                    rel_path = os.path.relpath(shared_path, working_dir)
                    os.symlink(rel_path, working_path)
                else:
                    logger.info(f"Moving newer {filename} from working directory to shared and creating symlink")
                    import shutil
                    os.remove(shared_path)
                    shutil.move(working_path, shared_path)
                    rel_path = os.path.relpath(shared_path, working_dir)
                    os.symlink(rel_path, working_path)

def sync_kg_data_to_shared(shared_dir: str, working_dir: str):
    """
    Ensure knowledge graph data in shared directory is up to date.
    Since we use symlinks, files in working directory are actually links to shared directory,
    so this function mainly ensures the symlinks are properly set up.
    """
    kg_files = [
        "vdb_entities.json",
        "vdb_relationships.json",
        "vdb_chunks.json", 
        "graph_chunk_entity_relation.graphml",
        "kg.json",
    ]
    
    # Convert to absolute paths
    shared_dir = os.path.abspath(shared_dir)
    working_dir = os.path.abspath(working_dir)
    
    for filename in kg_files:
        working_path = os.path.join(working_dir, filename)
        shared_path = os.path.join(shared_dir, filename)
        
        # If file exists in working dir
        if os.path.exists(working_path):
            if os.path.islink(working_path):
                # Already a symlink, check if it points to shared
                link_target = os.readlink(working_path)
                abs_link_target = os.path.abspath(os.path.join(os.path.dirname(working_path), link_target))
                if abs_link_target != os.path.abspath(shared_path):
                    logger.debug(f"Updating symlink for {filename} to point to shared directory")
                    os.remove(working_path)
                    rel_path = os.path.relpath(shared_path, working_dir)
                    os.symlink(rel_path, working_path)
                # else: symlink is correct, no action needed
            else:
                # Regular file exists, move to shared and create symlink
                logger.debug(f"Moving {filename} to shared directory and creating symlink")
                import shutil
                if os.path.exists(shared_path):
                    # If shared exists, check which is newer
                    if os.path.getmtime(working_path) > os.path.getmtime(shared_path):
                        os.remove(shared_path)
                        shutil.move(working_path, shared_path)
                    else:
                        os.remove(working_path)
                else:
                    shutil.move(working_path, shared_path)
                rel_path = os.path.relpath(shared_path, working_dir)
                os.symlink(rel_path, working_path)

def load_kg_triples(kb_path: str) -> List[Tuple[str, str, str]]:
    """
    Load knowledge graph triples from MetaQA format kb.txt file
    Format: subject|predicate|object
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
    Load question-answer pairs from MetaQA format qa file
    Format: question\tanswer (answer may contain multiple, separated by |)
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
    Convert MetaQA triples to LightRAG's insert_custom_kg format
    """
    # Collect all entities and their information
    entities_dict: Dict[str, Dict] = {}
    relationships = []
    
    # Count occurrences and relation types for each entity to generate descriptions
    entity_relations: Dict[str, List[str]] = defaultdict(list)
    entity_types: Dict[str, set] = defaultdict(set)
    
    for s, p, o in triples:
        # Collect entities
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
        
        # Record relation types
        entity_relations[s].append(p)
        entity_relations[o].append(p)
        
        # Infer entity types based on relation types
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
        
        # Create relationships
        relationships.append({
            "src_id": s,
            "tgt_id": o,
            "description": f"{s} {p} {o}",
            "keywords": p,
            "weight": 1.0,
            "source_id": "metaqa_kb",
            "file_path": "demo1202/MetaQA/kb.txt",
        })
    
    # Update entity descriptions and types
    for entity_name, entity_data in entities_dict.items():
        relations = entity_relations.get(entity_name, [])
        if relations:
            unique_relations = list(set(relations))[:10]  # Limit description length
            entity_data["description"] = f"{entity_name} (relations: {', '.join(unique_relations)})"
        
        # Set entity type (prefer inferred types)
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
        "chunks": [],  # MetaQA is pure KG, no chunks needed
    }


async def initialize_rag():
    """Initialize LightRAG"""
    # #region agent log
    _debug_log("metaqa_lightrag.py:191", "initialize_rag entry", {"working_dir": WORKING_DIR, "device": device}, "C")
    # #endregion
    logger.info("Initializing LightRAG with working directory: %s", WORKING_DIR)
    logger.info(f"Using device: {device}")
    
    # Load embedding model to GPU
    logger.info("Loading embedding model to GPU...")
    # #region agent log
    _debug_log("metaqa_lightrag.py:198", "Before loading embedding model", {"embed_model_name": embed_model_name, "cuda_available": torch.cuda.is_available()}, "C")
    # #endregion
    embed_tokenizer = AutoTokenizer.from_pretrained(embed_model_name)
    embed_model = AutoModel.from_pretrained(embed_model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device)
    # #region agent log
    _debug_log("metaqa_lightrag.py:200", "After loading embedding model", {"model_device": str(next(embed_model.parameters()).device)}, "C")
    # #endregion
    logger.info(f"Embedding model loaded to: {next(embed_model.parameters()).device}")
    
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

    # Sync knowledge graph data from shared directory before initializing storages
    # This allows different LLM models to share the same knowledge graph
    sync_kg_data_from_shared(SHARED_KG_DIR, WORKING_DIR)
    
    logger.info("Initializing storages...")
    # #region agent log
    _debug_log("metaqa_lightrag.py:217", "Before initialize_storages", {"working_dir": WORKING_DIR}, "A")
    # Check for storage files
    storage_files = []
    for fname in ["vdb_entities.json", "vdb_relationships.json", "vdb_chunks.json", "kg.json", "doc_status.json"]:
        fpath = os.path.join(WORKING_DIR, fname)
        if os.path.exists(fpath):
            size = os.path.getsize(fpath)
            storage_files.append({"name": fname, "size": size, "exists": True})
        else:
            storage_files.append({"name": fname, "exists": False})
    _debug_log("metaqa_lightrag.py:218", "Storage files check", {"files": storage_files}, "A")
    # #endregion
    try:
        await rag.initialize_storages()
        # #region agent log
        _debug_log("metaqa_lightrag.py:230", "After initialize_storages success", {}, "A")
        # #endregion
        logger.info("Storages initialized.")
    except json.JSONDecodeError as e:
        # #region agent log
        _debug_log("metaqa_lightrag.py:234", "JSON decode error in initialize_storages", {"error": str(e), "error_type": type(e).__name__, "error_msg": str(e), "error_args": e.args if hasattr(e, 'args') else []}, "A")
        # #endregion
        raise
    except Exception as e:
        # #region agent log
        _debug_log("metaqa_lightrag.py:238", "Other error in initialize_storages", {"error": str(e), "error_type": type(e).__name__}, "A")
        # #endregion
        raise
    # #region agent log
    _debug_log("metaqa_lightrag.py:220", "initialize_rag exit", {}, "C")
    # #endregion
    return rag


def normalize_answer(answer: str | None) -> str:
    """Normalize answer string for comparison"""
    # #region agent log
    _debug_log("metaqa_lightrag.py:223", "normalize_answer entry", {"answer_type": type(answer).__name__, "answer_is_none": answer is None, "answer_repr": repr(answer) if answer is not None else "None"}, "B")
    # #endregion
    if answer is None:
        # #region agent log
        _debug_log("metaqa_lightrag.py:226", "normalize_answer handling None", {}, "B")
        # #endregion
        return ""
    result = str(answer).lower().replace(" ", "").strip()
    # #region agent log
    _debug_log("metaqa_lightrag.py:228", "normalize_answer exit", {"result": result}, "B")
    # #endregion
    return result


def check_answer(predicted: str | None, gold: str) -> bool:
    """
    Check if predicted answer is correct
    Supports multiple answers (separated by |)
    """
    pred_normalized = normalize_answer(predicted)
    gold_normalized = normalize_answer(gold)
    
    # If gold contains multiple answers (separated by |), check if pred contains any of them
    if "|" in gold:
        gold_answers = [normalize_answer(a) for a in gold.split("|")]
        return any(gold_answer in pred_normalized for gold_answer in gold_answers)
    else:
        # Check if gold is in pred, or pred is in gold
        return gold_normalized in pred_normalized or pred_normalized in gold_normalized


def evaluate(rag: LightRAG, qa_pairs: List[Tuple[str, str]], mode: str = "global", max_samples: int = None):
    """
    Evaluate LightRAG performance on MetaQA dataset
    
    Args:
        rag: LightRAG instance
        qa_pairs: List of question-answer pairs
        mode: Query mode ("naive", "local", "global", "hybrid")
        max_samples: Maximum number of evaluation samples (None means all)
    """
    logger.info(f"Starting evaluation with mode={mode}, total QA pairs: {len(qa_pairs)}")
    
    if max_samples:
        qa_pairs = qa_pairs[:max_samples]
        logger.info(f"Limited to {max_samples} samples for evaluation")
    
    correct = 0
    results = []
    
    for idx, (question, gold_answer) in enumerate(qa_pairs, 1):
        try:
            # Execute query
            # #region agent log
            _debug_log("metaqa_lightrag.py:267", "Before rag.query", {"idx": idx, "question": question[:100]}, "B")
            # #endregion
            predicted = rag.query(question, param=QueryParam(mode=mode))
            # #region agent log
            _debug_log("metaqa_lightrag.py:270", "After rag.query", {"idx": idx, "predicted_type": type(predicted).__name__, "predicted_is_none": predicted is None, "predicted_repr": repr(predicted) if predicted is not None else "None"}, "B")
            # #endregion
            
            # Check if answer is correct
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
    """Main function"""
    # Initialize RAG
    rag = asyncio.run(initialize_rag())
    
    # Load knowledge graph
    logger.info("Loading knowledge graph from %s", KB_FILE)
    triples = load_kg_triples(KB_FILE)
    
    # Convert to LightRAG format
    logger.info("Converting triples to LightRAG format...")
    custom_kg = convert_triples_to_custom_kg(triples)
    
    # Insert knowledge graph
    logger.info("Inserting knowledge graph into LightRAG...")
    logger.info(f"  - Entities: {len(custom_kg['entities'])}")
    logger.info(f"  - Relationships: {len(custom_kg['relationships'])}")
    rag.insert_custom_kg(custom_kg)
    logger.info("Knowledge graph inserted successfully.")
    
    # Sync knowledge graph data to shared directory after insertion
    # This allows other LLM models to use the same knowledge graph
    sync_kg_data_to_shared(SHARED_KG_DIR, WORKING_DIR)
    logger.info("Knowledge graph data synced to shared directory.")
    
    # Load QA data
    logger.info("Loading QA pairs from %s", QA_TEST_FILE)
    qa_pairs = load_qa_pairs(QA_TEST_FILE)
    
    # Evaluate different modes
    evaluation_results = {}
    
    # Evaluate global mode (KG-based retrieval)
    logger.info("\n" + "=" * 60)
    logger.info("Evaluating with GLOBAL mode (KG-based retrieval)")
    logger.info("=" * 60)
    eval_result_global = evaluate(rag, qa_pairs, mode="global", max_samples=None)
    evaluation_results["global"] = eval_result_global
    
    # Evaluate hybrid mode
    logger.info("\n" + "=" * 60)
    logger.info("Evaluating with HYBRID mode")
    logger.info("=" * 60)
    eval_result_hybrid = evaluate(rag, qa_pairs, mode="hybrid", max_samples=None)
    evaluation_results["hybrid"] = eval_result_hybrid
    
    # # Evaluate local mode
    # logger.info("\n" + "=" * 60)
    # logger.info("Evaluating with LOCAL mode")
    # logger.info("=" * 60)
    # eval_result_local = evaluate(rag, qa_pairs, mode="local", max_samples=None)
    # evaluation_results["local"] = eval_result_local
    
    # Save evaluation results
    results_file = os.path.join(WORKING_DIR, f"evaluation_results_{timestamp}.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
    logger.info(f"\nEvaluation results saved to: {results_file}")
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    for mode, result in evaluation_results.items():
        logger.info(f"{mode.upper()} mode: Accuracy = {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

