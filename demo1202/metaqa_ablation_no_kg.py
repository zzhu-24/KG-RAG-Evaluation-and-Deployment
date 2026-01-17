import os
import logging
import json
from datetime import datetime
from typing import List, Tuple

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import asyncio
import nest_asyncio

nest_asyncio.apply()

# Create log directory
LOG_DIR = "./log"
os.makedirs(LOG_DIR, exist_ok=True)

# Generate log filename with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(LOG_DIR, f"metaqa_ablation_no_kg_{timestamp}.log")

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
QA_TEST_FILE = os.path.join(METAQA_DIR, "2-hop/vanilla/qa_test.txt")

# LLM model configuration (same as metaqa_lightrag.py)
llm_model_name = "Qwen/Qwen2.5-3B-Instruct"
logger.info(f"Using LLM model: {llm_model_name}")

# Model-specific directory for results
def get_model_safe_name(model_name: str) -> str:
    """Convert model name to a filesystem-safe directory name"""
    safe_name = model_name.replace("/", "_").replace("\\", "_")
    safe_name = safe_name.replace(":", "_").replace(" ", "_")
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in safe_name)
    return safe_name

model_dir_name = get_model_safe_name(llm_model_name)
RESULTS_DIR = f"./cache/metaqa_ablation_{model_dir_name}"
os.makedirs(RESULTS_DIR, exist_ok=True)
logger.info(f"Using results directory: {RESULTS_DIR}")


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


async def initialize_llm():
    """Initialize LLM model and tokenizer"""
    logger.info("Loading LLM model...")
    logger.info(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        llm_model_name,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        llm_model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if device == "cpu":
        model = model.to(device)
    
    logger.info(f"LLM model loaded to: {next(model.parameters()).device}")
    return model, tokenizer


# System prompt template (same as LightRAG but without KG context)
# This ensures fair comparison by using the same prompt structure
ABLATION_SYSTEM_PROMPT = """---Role---

You are an expert Question Answering (QA) assistant for the MetaQA dataset.

---Task Setting---

This is an ablation study: the Knowledge Graph (KG) and any retrieved documents are NOT available.
You must answer the user question using ONLY your training knowledge and reasoning.
Do NOT assume access to any external tools, databases, or retrieval.

---Goal---

Given a question, produce the best possible short answer in the style expected by MetaQA evaluation.

---Instructions---

1. Understand the Question
- Identify the question type (e.g., actor, director, writer, genre, release year, language, country).
- Determine whether the answer expects:
  - a single entity/name, or
  - multiple entities/names.

2. Answer Generation Rules
- Provide ONLY the final answer, without explanations or intermediate reasoning.
- If multiple answers are required, output them as a comma-separated list (no extra text).
- Use the canonical entity naming style (proper nouns, original movie/person names).
- Do NOT fabricate citations, sources, or references.

3. Uncertainty Handling
- If the answer cannot be determined with high confidence, output exactly:
unknown

4. Formatting
- Output must be plain text ONLY.
- Do NOT use Markdown headings, bullet points, or references.
- Do NOT add any prefix like "Answer:".

---Context---

No context is provided for this ablation setting."""



async def query_llm_direct(model, tokenizer, question: str) -> str:
    """
    Query LLM directly without any KG/RAG context
    This is the ablation baseline - pure LLM QA
    Uses the same system prompt structure as LightRAG for fair comparison
    """
    # Build system prompt (same format as LightRAG but with empty context)
    system_prompt = ABLATION_SYSTEM_PROMPT
    
    # Format messages with system prompt and user query (same as LightRAG)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    
    # Apply chat template (same as LightRAG's hf_model_if_cache)
    try:
        input_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Fallback: try to handle system prompt manually
        try:
            # Some models don't support system role, merge into user message
            merged_content = f"<system>{system_prompt}</system>\n{question}"
            messages_fallback = [{"role": "user", "content": merged_content}]
            input_prompt = tokenizer.apply_chat_template(
                messages_fallback, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Final fallback: simple format
            input_prompt = f"{system_prompt}\n\n---User Query---\n{question}"
    
    # Tokenize and generate (same parameters as LightRAG's hf_model_if_cache)
    input_ids = tokenizer(
        input_prompt, return_tensors="pt", padding=True, truncation=True
    ).to(device)
    
    inputs = {k: v.to(model.device) for k, v in input_ids.items()}
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=512,  # Same as LightRAG
            num_return_sequences=1,
            early_stopping=True,
        )
    
    # Decode only the generated part (excluding input)
    response_text = tokenizer.decode(
        output[0][len(inputs["input_ids"][0]):],
        skip_special_tokens=True
    ).strip()
    
    return response_text


def normalize_answer(answer: str | None) -> str:
    """Normalize answer string for comparison"""
    if answer is None:
        return ""
    result = str(answer).lower().replace(" ", "").strip()
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


async def evaluate_no_kg(model, tokenizer, qa_pairs: List[Tuple[str, str]], max_samples: int = None):
    """
    Evaluate pure LLM (no KG) performance on MetaQA dataset
    
    Args:
        model: LLM model
        tokenizer: Tokenizer
        qa_pairs: List of question-answer pairs
        max_samples: Maximum number of evaluation samples (None means all)
    """
    logger.info(f"Starting ablation evaluation (no KG), total QA pairs: {len(qa_pairs)}")
    
    if max_samples:
        qa_pairs = qa_pairs[:max_samples]
        logger.info(f"Limited to {max_samples} samples for evaluation")
    
    correct = 0
    results = []
    
    for idx, (question, gold_answer) in enumerate(qa_pairs, 1):
        try:
            # Query LLM directly without KG
            predicted = await query_llm_direct(model, tokenizer, question)
            
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
    logger.info("Ablation Evaluation Results (No KG - Pure LLM):")
    logger.info(f"Total questions: {len(qa_pairs)}")
    logger.info(f"Correct answers: {correct}")
    logger.info(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    logger.info("=" * 60)
    
    return {
        "mode": "ablation_no_kg",
        "total": len(qa_pairs),
        "correct": correct,
        "accuracy": accuracy,
        "results": results,
    }


async def main():
    """Main function"""
    # Initialize LLM
    model, tokenizer = await initialize_llm()
    
    # Load QA data
    logger.info("Loading QA pairs from %s", QA_TEST_FILE)
    qa_pairs = load_qa_pairs(QA_TEST_FILE)
    
    # Evaluate without KG (ablation baseline)
    logger.info("\n" + "=" * 60)
    logger.info("Evaluating ABLATION BASELINE (No KG - Pure LLM)")
    logger.info("=" * 60)
    eval_result = await evaluate_no_kg(model, tokenizer, qa_pairs, max_samples=None)
    
    # Save evaluation results
    results_file = os.path.join(RESULTS_DIR, f"ablation_results_{timestamp}.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(eval_result, f, indent=2, ensure_ascii=False)
    logger.info(f"\nEvaluation results saved to: {results_file}")
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("ABLATION EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Mode: {eval_result['mode']}")
    logger.info(f"Total questions: {eval_result['total']}")
    logger.info(f"Correct answers: {eval_result['correct']}")
    logger.info(f"Accuracy: {eval_result['accuracy']:.4f} ({eval_result['accuracy']*100:.2f}%)")
    logger.info("=" * 60)
    logger.info("\nNote: This is the ablation baseline without KG.")
    logger.info("Compare this with metaqa_lightrag.py results to see KG impact.")


if __name__ == "__main__":
    asyncio.run(main())
