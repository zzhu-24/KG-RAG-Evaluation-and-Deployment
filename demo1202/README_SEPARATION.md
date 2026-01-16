# F1QA LightRAG Script Separation Guide

## Overview

To reduce memory usage and optimize workflow, the original `f1qa_lightrag.py` has been split into two independent scripts:

1. **`f1qa_prepare.py`** - Data preparation script (initialization + data insertion)
2. **`f1qa_query.py`** - Query script (query only)

## Memory Usage Analysis

### Content loaded in memory by the original script (`f1qa_lightrag.py`):

1. **Embedding Model** (bge-m3)
   - Location: GPU
   - Size: Approximately 500MB-1GB (depending on precision)
   - Purpose: Text vectorization

2. **LLM Model** (Qwen/Qwen2.5-3B-Instruct)
   - Location: GPU (loaded on first call)
   - Size: Approximately 6GB (FP16)
   - Purpose: Answer generation
   - Note: Uses `@lru_cache` for caching, loaded on first call

3. **LightRAG Instance**
   - Vector database (entities, relations, document chunks)
   - Graph database (knowledge graph)
   - Document status storage
   - Cache storage

### Advantages After Separation

- **Preparation Phase**: Only requires Embedding model + LLM model (for entity/relation extraction)
- **Query Phase**: Only requires Embedding model + LLM model (for answer generation)
- **Can run separately**: After preparation is complete, you can close the preparation script and only run the query script
- **Reduce memory peaks**: Avoid loading all components simultaneously

## Usage

### 1. Data Preparation Phase

```bash
# Basic usage (no progress display)
python demo1202/f1qa_prepare.py --news-file ./demo1202/F1QA/news/news_example.jsonl

# Show detailed processing progress
python demo1202/f1qa_prepare.py --news-file ./demo1202/F1QA/news/news_example.jsonl --show-progress
```

**Features:**
- Initialize LightRAG
- Load Embedding model
- Insert news data
- Wait for processing to complete (if using `--show-progress`)

**Notes:**
- Data insertion is asynchronous, processing continues in the background even after script exits
- Can use `--show-progress` to monitor processing progress
- After processing completes, data is saved in `./cache/f1qa_demo/` directory

### 2. Query Phase

```bash
# Interactive query
python demo1202/f1qa_query.py

# Single question
python demo1202/f1qa_query.py --question "what happened to Guanyu ZHOU"

# Read questions from file
python demo1202/f1qa_query.py --questions-file questions.txt

# Specify query mode and parameters (reduce memory usage)
python demo1202/f1qa_query.py \
    --question "what happened to Guanyu ZHOU" \
    --mode hybrid \
    --top-k 30 \
    --chunk-top-k 15 \
    --max-total-tokens 4000
```

**Features:**
- Initialize LightRAG (only load necessary components)
- Execute queries
- Automatically clean GPU cache

**Query Parameter Description:**
- `--mode`: Query mode (naive/local/global/hybrid)
- `--top-k`: Number of entities/relations to retrieve (default: LightRAG default)
- `--chunk-top-k`: Number of document chunks to retrieve (default: LightRAG default)
- `--max-total-tokens`: Maximum number of tokens in context (default: LightRAG default)

**Memory Optimization Suggestions:**
- If encountering CUDA OOM errors, you can reduce `--top-k`, `--chunk-top-k`, and `--max-total-tokens`
- Query script automatically cleans GPU cache before and after each query

## Workflow Examples

### Complete Workflow

```bash
# Step 1: Prepare data
python demo1202/f1qa_prepare.py \
    --news-file ./demo1202/F1QA/news/news_example.jsonl \
    --show-progress

# Wait for processing to complete...

# Step 2: Perform queries (can run in different terminal/session)
python demo1202/f1qa_query.py --question "what happened to Guanyu ZHOU"
```

### Query Only (Data Already Prepared)

```bash
# If data is already prepared, directly run the query script
python demo1202/f1qa_query.py
```

## Memory Optimization Tips

1. **Reduce Query Parameters**:
   ```bash
   python demo1202/f1qa_query.py \
       --question "your question" \
       --top-k 20 \
       --chunk-top-k 10 \
       --max-total-tokens 3000
   ```

2. **Use Smaller Models**:
   - Modify `llm_model_name` in `f1qa_query.py` to a smaller model
   - For example: `"Qwen/Qwen2.5-1.5B-Instruct"`

3. **Set Environment Variables** (reduce memory fragmentation):
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   ```

4. **Batch Queries**:
   - Don't query too many questions at once
   - Script automatically cleans GPU cache after each query

## File Structure

```
demo1202/
├── f1qa_lightrag.py      # Original script (split)
├── f1qa_prepare.py       # Data preparation script (new)
├── f1qa_query.py          # Query script (new)
└── README_SEPARATION.md   # This documentation
```

## Notes

1. **Working Directory**: Both scripts use the same working directory `./cache/f1qa_demo/`
2. **Model Path**: Ensure Embedding model path is correct: `/home/infres/zzhu-24/large_files/bge-m3`
3. **GPU Memory**: If encountering OOM during queries, reduce query parameters or use smaller models
4. **Data Consistency**: Ensure data preparation is complete (or at least partially complete) before querying

## Troubleshooting

### CUDA OOM Error

If encountering CUDA out-of-memory errors:

1. Reduce query parameters:
   ```bash
   python demo1202/f1qa_query.py \
       --question "your question" \
       --top-k 20 \
       --chunk-top-k 10
   ```

2. Clear GPU cache:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

3. Restart Python process (release all GPU memory)

### Data Not Found

If prompted that data is not found during queries:

1. Ensure `f1qa_prepare.py` has been run and data insertion is complete
2. Check if working directory `./cache/f1qa_demo/` exists
3. Check if data is still being processed (use `--show-progress` to view)
