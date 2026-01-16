#!/usr/bin/env python3
"""
F1QA LightRAG Query Script
Only responsible for querying, does not perform data insertion
Separated from preparation script to reduce memory usage
"""

import os
import logging
import argparse
from datetime import datetime
from typing import List, Optional
from pathlib import Path

from lightrag import LightRAG, QueryParam
from lightrag.llm.hf import hf_model_complete, hf_embed
from lightrag.utils import EmbeddingFunc
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
log_file = os.path.join(LOG_DIR, f"f1qa_query_{timestamp}.log")

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
    # Clear GPU cache
    torch.cuda.empty_cache()
    logger.info("GPU cache cleared")
else:
    logger.warning("No GPU detected, will use CPU")

WORKING_DIR = "./cache/f1qa_demo"

# Default data path
DEFAULT_QUESTIONS_FILE = None


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
    """Initialize LightRAG for querying only"""
    logger.info("Initializing LightRAG for querying with working directory: %s", WORKING_DIR)
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
    
    # Clear GPU cache
    if device == "cuda":
        torch.cuda.empty_cache()
    
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
    
    # Clear GPU cache
    if device == "cuda":
        torch.cuda.empty_cache()
        logger.info("GPU cache cleared after initialization")
    
    return rag


def print_entities_and_relations(data_result: dict, top_k: int = None):
    """
    Print retrieved entities and relations information
    
    Args:
        data_result: Result dictionary returned by query_data
        top_k: Print top k entities and relations (None means print all)
    """
    if not data_result or data_result.get("status") != "success":
        return
    
    data = data_result.get("data", {})
    entities = data.get("entities", [])
    relationships = data.get("relationships", [])
    
    # Determine print count
    entity_count = len(entities)
    relation_count = len(relationships)
    print_entity_count = min(top_k, entity_count) if top_k else entity_count
    print_relation_count = min(top_k, relation_count) if top_k else relation_count
    
    # Print entity information
    if entities:
        print("\n" + "=" * 80)
        print(f"📊 Retrieved Entities (Total: {entity_count}, Showing top {print_entity_count})")
        print("=" * 80)
        for idx, entity in enumerate(entities[:print_entity_count], 1):
            print(f"\n[Entity {idx}]")
            print(f"  Name: {entity.get('entity_name', 'N/A')}")
            print(f"  Type: {entity.get('entity_type', 'N/A')}")
            description = entity.get('description', 'N/A')
            # Truncate if description is too long
            if len(description) > 500:
                description = description[:500] + "..."
            print(f"  Description: {description}")
            print(f"  Source file: {entity.get('file_path', 'N/A')}")
            print(f"  Created at: {entity.get('created_at', 'N/A')}")
            if entity.get('reference_id'):
                print(f"  Reference ID: {entity.get('reference_id')}")
        if entity_count > print_entity_count:
            print(f"\n  ... {entity_count - print_entity_count} more entities not shown")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("📊 Retrieved Entities: None")
        print("=" * 80)
    
    # Print relation information
    if relationships:
        print("\n" + "=" * 80)
        print(f"🔗 Retrieved Relations (Total: {relation_count}, Showing top {print_relation_count})")
        print("=" * 80)
        for idx, relation in enumerate(relationships[:print_relation_count], 1):
            print(f"\n[Relation {idx}]")
            print(f"  Entity 1: {relation.get('src_id', 'N/A')}")
            print(f"  Entity 2: {relation.get('tgt_id', 'N/A')}")
            description = relation.get('description', 'N/A')
            # Truncate if description is too long
            if len(description) > 500:
                description = description[:500] + "..."
            print(f"  Description: {description}")
            if relation.get('keywords'):
                print(f"  Keywords: {relation.get('keywords')}")
            if relation.get('weight') is not None:
                print(f"  Weight: {relation.get('weight')}")
            print(f"  Source file: {relation.get('file_path', 'N/A')}")
            print(f"  Created at: {relation.get('created_at', 'N/A')}")
            if relation.get('reference_id'):
                print(f"  Reference ID: {relation.get('reference_id')}")
        if relation_count > print_relation_count:
            print(f"\n  ... {relation_count - print_relation_count} more relations not shown")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("🔗 Retrieved Relations: None")
        print("=" * 80)


def query_rag(rag: LightRAG, question: str, mode: str = "hybrid", show_entities: bool = True, **query_kwargs) -> tuple[str, bool]:
    """
    Answer question using LightRAG
    
    Args:
        rag: LightRAG instance
        question: Question
        mode: Query mode ("naive", "local", "global", "hybrid")
        show_entities: Whether to show retrieved entities and relations
        **query_kwargs: Additional query parameters (top_k, chunk_top_k, etc.)
        
    Returns:
        Tuple of (answer, is_oom_error): Answer string and whether it's an OOM error
    """
    try:
        # Clear GPU cache
        if device == "cuda":
            torch.cuda.empty_cache()
        
        param = QueryParam(mode=mode, **query_kwargs)
        
        # If showing entities is enabled, get structured data first
        if show_entities:
            try:
                print("\n🔍 Retrieving related entities and relations...")
                data_result = asyncio.run(rag.aquery_data(question, param=param))
                
                # Get top_k parameter (for display count)
                top_k = query_kwargs.get("top_k", param.top_k)
                
                # Print entity and relation information
                print_entities_and_relations(data_result, top_k=top_k)
            except Exception as e:
                logger.warning(f"Error getting entity and relation information: {e}")
                print(f"⚠️  Unable to display entity and relation information: {e}")
        
        # Get answer
        print("\n💭 Generating answer...")
        answer = rag.query(question, param=param)
        
        # Clear GPU cache
        if device == "cuda":
            torch.cuda.empty_cache()
        
        return answer, False
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error querying question: {e}")
        # Clear GPU cache
        if device == "cuda":
            torch.cuda.empty_cache()
        
        # Detect if it's a CUDA OOM error
        is_oom = "CUDA out of memory" in error_msg or "out of memory" in error_msg.lower()
        return f"Error: {error_msg}", is_oom


def print_help():
    """Print interactive command help"""
    help_text = """
Available commands:
  /help, /h          - Show this help message
  /mode <mode>       - Switch query mode (naive/local/global/hybrid)
  /topk <num>        - Set top_k parameter
  /chunktopk <num>   - Set chunk_top_k parameter
  /maxtokens <num>   - Set max_total_tokens parameter
  /maxentity <num>   - Set max_entity_tokens parameter
  /maxrelation <num> - Set max_relation_tokens parameter
  /lowmem            - Quickly set low memory mode (reduce all parameters)
  /params            - Show current query parameters
  /clear             - Clear screen
  /quit, /exit, /q   - Exit program

Examples:
  /mode hybrid
  /topk 30
  /chunktopk 15
  /maxtokens 4000
  /lowmem            - Quickly set low memory mode
"""
    print(help_text)


def print_current_params(mode, query_kwargs):
    """Print current query parameters"""
    print("\n" + "=" * 60)
    print("Current Query Parameters:")
    print(f"  Mode: {mode}")
    for key, value in sorted(query_kwargs.items()):
        print(f"  {key}: {value}")
    if not query_kwargs:
        print("  (Using LightRAG default values)")
    print("=" * 60 + "\n")


def suggest_low_memory_params():
    """Suggest parameters for low memory mode"""
    return {
        "top_k": 20,
        "chunk_top_k": 10,
        "max_total_tokens": 8000,
        "max_entity_tokens": 3000,
        "max_relation_tokens": 3000,
    }


def interactive_mode(rag: LightRAG, initial_mode: str = "hybrid", initial_kwargs: dict = None, show_entities: bool = True):
    """
    Interactive Q&A mode
    
    Args:
        rag: LightRAG instance
        initial_mode: Initial query mode
        initial_kwargs: Initial query parameters
        show_entities: Whether to show retrieved entities and relations
    """
    if initial_kwargs is None:
        initial_kwargs = {}
    
    current_mode = initial_mode
    current_kwargs = initial_kwargs.copy()
    
    print("\n" + "=" * 60)
    print("F1QA LightRAG Interactive Q&A System")
    print("=" * 60)
    print("Enter a question to start querying, or /help to see available commands")
    print("=" * 60)
    print_current_params(current_mode, current_kwargs)
    
    while True:
        try:
            # Get user input
            user_input = input("\n💬 Please enter a question or command: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith("/"):
                parts = user_input.split()
                command = parts[0].lower()
                
                if command in ["/help", "/h"]:
                    print_help()
                    continue
                
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith("/"):
                parts = user_input.split()
                command = parts[0].lower()
                
                if command in ["/help", "/h"]:
                    print_help()
                    continue
                
                elif command in ["/quit", "/exit", "/q"]:
                    print("\n👋 Goodbye!")
                    break
                
                elif command == "/mode" and len(parts) > 1:
                    new_mode = parts[1].lower()
                    if new_mode in ["naive", "local", "global", "hybrid"]:
                        current_mode = new_mode
                        print(f"✅ Query mode switched to: {current_mode}")
                        print_current_params(current_mode, current_kwargs)
                    else:
                        print(f"❌ Invalid mode: {new_mode}")
                        print("Available modes: naive, local, global, hybrid")
                    continue
                
                elif command == "/topk" and len(parts) > 1:
                    try:
                        top_k = int(parts[1])
                        current_kwargs["top_k"] = top_k
                        print(f"✅ top_k set to: {top_k}")
                        print_current_params(current_mode, current_kwargs)
                    except ValueError:
                        print(f"❌ Invalid number: {parts[1]}")
                    continue
                
                elif command == "/chunktopk" and len(parts) > 1:
                    try:
                        chunk_top_k = int(parts[1])
                        current_kwargs["chunk_top_k"] = chunk_top_k
                        print(f"✅ chunk_top_k set to: {chunk_top_k}")
                        print_current_params(current_mode, current_kwargs)
                    except ValueError:
                        print(f"❌ Invalid number: {parts[1]}")
                    continue
                
                elif command == "/maxtokens" and len(parts) > 1:
                    try:
                        max_tokens = int(parts[1])
                        current_kwargs["max_total_tokens"] = max_tokens
                        print(f"✅ max_total_tokens set to: {max_tokens}")
                        print_current_params(current_mode, current_kwargs)
                    except ValueError:
                        print(f"❌ Invalid number: {parts[1]}")
                    continue
                
                elif command == "/maxentity" and len(parts) > 1:
                    try:
                        max_entity = int(parts[1])
                        current_kwargs["max_entity_tokens"] = max_entity
                        print(f"✅ max_entity_tokens set to: {max_entity}")
                        print_current_params(current_mode, current_kwargs)
                    except ValueError:
                        print(f"❌ Invalid number: {parts[1]}")
                    continue
                
                elif command == "/maxrelation" and len(parts) > 1:
                    try:
                        max_relation = int(parts[1])
                        current_kwargs["max_relation_tokens"] = max_relation
                        print(f"✅ max_relation_tokens set to: {max_relation}")
                        print_current_params(current_mode, current_kwargs)
                    except ValueError:
                        print(f"❌ Invalid number: {parts[1]}")
                    continue
                
                elif command == "/lowmem":
                    low_mem_params = suggest_low_memory_params()
                    current_kwargs.update(low_mem_params)
                    print("✅ Switched to low memory mode:")
                    print_current_params(current_mode, current_kwargs)
                    continue
                
                elif command == "/params":
                    print_current_params(current_mode, current_kwargs)
                    continue
                
                elif command == "/clear":
                    os.system("clear" if os.name != "nt" else "cls")
                    print("\n" + "=" * 60)
                    print("F1QA LightRAG Interactive Q&A System")
                    print("=" * 60)
                    continue
                
                else:
                    print(f"❌ Unknown command: {command}")
                    print("Enter /help to see available commands")
                    continue
            
            # Handle question query
            question = user_input
            logger.info(f"Question: {question}")
            
            print("\n🔍 Querying, please wait...")
            try:
                answer, is_oom = query_rag(rag, question, mode=current_mode, show_entities=show_entities, **current_kwargs)
                print("\n" + "=" * 60)
                print("📝 Answer:")
                print("=" * 60)
                print(answer)
                print("=" * 60)
                
                # If OOM error, provide suggestions
                if is_oom:
                    print("\n" + "⚠️ " * 30)
                    print("CUDA out of memory error detected!")
                    print("\n💡 Suggestions:")
                    print("  1. Use /lowmem command to quickly switch to low memory mode")
                    print("  2. Or manually reduce parameters:")
                    print("     /topk 20")
                    print("     /chunktopk 10")
                    print("     /maxtokens 8000")
                    print("     /maxentity 3000")
                    print("     /maxrelation 3000")
                    print("\nCurrent parameters:")
                    print_current_params(current_mode, current_kwargs)
                    print("⚠️ " * 30 + "\n")
            except Exception as e:
                print(f"\n❌ Query error: {e}")
                logger.error(f"Error processing question: {e}", exc_info=True)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except EOFError:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error processing input: {e}", exc_info=True)
            print(f"\n❌ Error occurred: {e}")


def main():
    """Main function for querying"""
    parser = argparse.ArgumentParser(
        description="F1QA LightRAG Query Script (default: interactive mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (default)
  python f1qa_query.py
  
  # Single question
  python f1qa_query.py --question "what happened to Guanyu ZHOU"
  
  # Read questions from file
  python f1qa_query.py --questions-file questions.txt
  
  # Specify query parameters
  python f1qa_query.py --mode hybrid --top-k 30 --chunk-top-k 15
        """
    )
    parser.add_argument(
        "--questions-file",
        type=str,
        default=DEFAULT_QUESTIONS_FILE,
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
        "--top-k",
        type=int,
        default=None,
        help="Top K entities/relations to retrieve (default: use LightRAG default)"
    )
    parser.add_argument(
        "--chunk-top-k",
        type=int,
        default=None,
        help="Top K chunks to retrieve (default: use LightRAG default)"
    )
    parser.add_argument(
        "--max-total-tokens",
        type=int,
        default=None,
        help="Maximum total tokens for context (default: use LightRAG default)"
    )
    parser.add_argument(
        "--max-entity-tokens",
        type=int,
        default=None,
        help="Maximum entity tokens (default: use LightRAG default)"
    )
    parser.add_argument(
        "--max-relation-tokens",
        type=int,
        default=None,
        help="Maximum relation tokens (default: use LightRAG default)"
    )
    parser.add_argument(
        "--low-mem",
        action="store_true",
        help="Use low memory mode (reduces all parameters)"
    )
    parser.add_argument(
        "--no-show-entities",
        action="store_true",
        help="Don't show retrieved entities and relations (default: show)"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize RAG
        logger.info("=" * 60)
        logger.info("Initializing LightRAG (query mode only)")
        logger.info("=" * 60)
        rag = asyncio.run(initialize_rag())
        
        # Prepare query parameters
        query_kwargs = {}
        if args.low_mem:
            # Use low memory mode
            low_mem_params = suggest_low_memory_params()
            query_kwargs.update(low_mem_params)
            logger.info("Using low memory mode")
        else:
            if args.top_k is not None:
                query_kwargs["top_k"] = args.top_k
            if args.chunk_top_k is not None:
                query_kwargs["chunk_top_k"] = args.chunk_top_k
            if args.max_total_tokens is not None:
                query_kwargs["max_total_tokens"] = args.max_total_tokens
            if args.max_entity_tokens is not None:
                query_kwargs["max_entity_tokens"] = args.max_entity_tokens
            if args.max_relation_tokens is not None:
                query_kwargs["max_relation_tokens"] = args.max_relation_tokens
        
        # Process questions
        if args.question:
            # Single question mode
            logger.info(f"Question: {args.question}")
            show_entities = not args.no_show_entities
            answer, is_oom = query_rag(rag, args.question, mode=args.mode, show_entities=show_entities, **query_kwargs)
            print(f"\nAnswer: {answer}\n")
            if is_oom:
                print("⚠️  Out of memory error detected, suggest reducing query parameters or using /lowmem command")
        elif args.questions_file:
            # Load questions from file
            questions = load_questions(args.questions_file)
            if questions:
                logger.info("=" * 60)
                logger.info(f"Starting to answer {len(questions)} questions (mode: {args.mode})")
                logger.info("=" * 60)
                
                for idx, question in enumerate(questions, 1):
                    logger.info(f"\n[{idx}/{len(questions)}] Question: {question}")
                    show_entities = not args.no_show_entities
                    answer, is_oom = query_rag(rag, question, mode=args.mode, show_entities=show_entities, **query_kwargs)
                    print(f"Answer: {answer}\n")
                    if is_oom:
                        print("⚠️  Out of memory error detected, suggest reducing query parameters")
            else:
                logger.warning("No questions found, entering interactive mode")
                show_entities = not args.no_show_entities
                interactive_mode(rag, initial_mode=args.mode, initial_kwargs=query_kwargs, show_entities=show_entities)
        else:
            # Interactive mode (default)
            show_entities = not args.no_show_entities
            interactive_mode(rag, initial_mode=args.mode, initial_kwargs=query_kwargs, show_entities=show_entities)
        
        logger.info("=" * 60)
        logger.info("Query completed")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Program execution failed: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
