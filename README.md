# LightRAG Experiments: MetaQA and F1QA

This project is based on and adapted from [LightRAG](https://github.com/xxx/yyy).
We thank the original authors and contributors for their work.

This project contains two Knowledge Graph Question Answering (KGQA) experiments based on the LightRAG framework:

1. **MetaQA** - Knowledge graph question answering evaluation on the MetaQA dataset
2. **F1QA** - Question answering system based on F1 news data

## Project Structure

```
demo1202/
├── MetaQA/                       # MetaQA dataset and experiment code
│   ├── kb.txt                   # Knowledge graph triple data (expected, add manually)
│   ├── 1-hop/                   # 1-hop question dataset (expected, add manually)
│   ├── 2-hop/                   # 2-hop question dataset (expected, add manually)
│   ├── 3-hop/                   # 3-hop question dataset (expected, add manually)
│   ├── entity/                  # Entity cache (generated, not in repository)
│   └── analyze_metaqa.py        # Data analysis script
├── F1QA/                        # F1QA experiment code and data
│   ├── fetch_f1_news.py         # F1 news scraping script
│   ├── README.md                # F1QA documentation
│   ├── news/                    # News data
│   │   ├── news_example.jsonl   # Example news data
│   │   └── f1_news.jsonl        # F1 news data (fetched or example)
│   └── sources.json             # Data source configuration
├── metaqa_lightrag.py           # MetaQA main experiment script
├── f1qa_prepare.py              # F1QA data preparation script
├── f1qa_query.py                # F1QA query script
├── f1qa_status.py               # F1QA status viewing script
├── f1qa_lightrag.py             # F1QA complete workflow script (split version)
├── print_graphml_example.py     # GraphML example output
├── README_metaqa.md             # MetaQA experiment documentation
└── TOKEN_LIMITS_EXPLANATION.md  # Token limits explanation
```

## Requirements

- Python 3.8+
- CUDA (recommended for GPU acceleration)
- At least 8GB VRAM (16GB+ recommended)

## Installation

```bash
# Install basic dependencies
pip install -r requirements.txt

# Or install main dependencies separately
pip install lightrag-hku
pip install transformers torch
pip install beautifulsoup4 requests tqdm
pip install nest-asyncio
```

## MetaQA Experiment

### Introduction

MetaQA is a knowledge graph question answering dataset containing movie-related knowledge graphs and question-answer pairs. This experiment uses LightRAG to evaluate on the MetaQA dataset.

### Data Download

**Important**: MetaQA dataset files are not included in this repository. You need to download them separately.

1. Download the MetaQA dataset from the [official repository](https://github.com/yuyuz/MetaQA)
2. Extract the dataset and place the files in the following structure:
   ```
   demo1202/MetaQA/
   ├── kb.txt                    # Knowledge graph triples
   ├── 1-hop/
   │   ├── vanilla/
   │   │   ├── qa_train.txt
   │   │   ├── qa_dev.txt
   │   │   └── qa_test.txt
   │   └── ntm/
   │       ├── qa_train.txt
   │       ├── qa_dev.txt
   │       └── qa_test.txt
   ├── 2-hop/                    # Similar structure
   └── 3-hop/                    # Similar structure
   ```

### Data Format

- **Knowledge Graph** (`MetaQA/kb.txt`): `subject|predicate|object` format
- **Question-Answer Pairs** (`MetaQA/*-hop/vanilla/qa_test.txt`): `question\tanswer` format, answers may contain multiple values (separated by `|`)

### Usage

```bash
# Run from project root directory
python demo1202/metaqa_lightrag.py
```

### Output

- **Log files**: `log/metaqa_lightrag_YYYYMMDD_HHMMSS.log`
- **Evaluation results**: `cache/metaqa_demo/evaluation_results_YYYYMMDD_HHMMSS.json`

### Evaluation Modes

Supports three query modes:
- **global**: Global retrieval based on knowledge graph
- **hybrid**: Hybrid retrieval mode (vector + graph)
- **local**: Local retrieval mode

For detailed information, please refer to [README_metaqa.md](README_metaqa.md)

## F1QA Experiment

### Introduction

F1QA is a question answering system based on F1 news data, demonstrating the application of LightRAG in real-world scenarios.

### Data Preparation

#### 1. Fetch News Data

```bash
# Scrape F1 news (optional, example data already available)
python demo1202/F1QA/fetch_f1_news.py
```

#### 2. Prepare Data and Insert into LightRAG

```bash
# Use example data
python demo1202/f1qa_prepare.py --news-file ./demo1202/F1QA/news/news_example.jsonl

# Show processing progress
python demo1202/f1qa_prepare.py --news-file ./demo1202/F1QA/news/news_example.jsonl --show-progress
```

### Querying

```bash
# Interactive query
python demo1202/f1qa_query.py

# Single question
python demo1202/f1qa_query.py --question "what happened to Guanyu ZHOU"

# Read questions from file
python demo1202/f1qa_query.py --questions-file questions.txt

# Specify query parameters (reduce memory usage)
python demo1202/f1qa_query.py \
    --question "what happened to Guanyu ZHOU" \
    --mode hybrid \
    --top-k 30 \
    --chunk-top-k 15 \
    --max-total-tokens 4000
```

### View Status

```bash
# View document processing status
python demo1202/f1qa_status.py
```

## Model Configuration

### Default Configuration

- **LLM**: `Qwen/Qwen2.5-1.5B-Instruct` (MetaQA) / `Qwen/Qwen2.5-3B-Instruct` (F1QA)
- **Embedding**: BGE-M3 (path needs to be configured according to actual situation)

### Modifying Model Configuration

Edit the `initialize_rag()` function in the corresponding script file:

```python
async def initialize_rag():
    # Modify model path and configuration
    embedding_model_path = "your_embedding_model_path"
    llm_model_name = "your_llm_model_name"
    # ...
```

## Utility Scripts

### Print GraphML Example

```bash
python demo1202/print_graphml_example.py
```

## Output Directory Description

- `log/` - Log files (auto-generated, added to .gitignore)
- `cache/` - LightRAG cache data (auto-generated, added to .gitignore)
  - `cache/metaqa_demo/` - MetaQA experiment cache
  - `cache/f1qa_demo/` - F1QA experiment cache

## Notes

1. **Model Path**: Need to modify the model path in scripts according to actual situation
2. **GPU Memory**: If encountering CUDA OOM errors, you can reduce query parameters (`--top-k`, `--chunk-top-k`, `--max-total-tokens`)
3. **Data Files**: Large data files (such as complete F1 news data) are not included in the repository, only example files are kept
4. **Incremental Updates**: F1QA supports incremental updates, already processed documents will be automatically skipped

## License

This project is based on the LightRAG framework, please refer to LightRAG's license.

## Acknowledgements

This project is based on and adapted from [LightRAG](https://github.com/xxx/yyy).
We thank the original authors and contributors for their work.

## Related Links

- [LightRAG Official Repository](https://github.com/HKUDS/LightRAG)
- [MetaQA Dataset](https://github.com/yuyuz/MetaQA)
