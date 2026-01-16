# MetaQA LightRAG Demo

This is a demo using LightRAG to evaluate Knowledge Graph Question Answering (KGQA) on the MetaQA dataset.

## Features

- **Direct KG Insertion**: Reads triples from MetaQA's kb.txt file and directly inserts them into LightRAG's knowledge graph (without text insertion)
- **Multi-mode Evaluation**: Supports evaluation in three query modes: global, hybrid, and local
- **Complete Evaluation Metrics**: Calculates accuracy and saves detailed results

## Data Format

### Knowledge Graph Format (kb.txt)
```
subject|predicate|object
```
Example:
```
Kismet|directed_by|William Dieterle
Kismet|starred_actors|Marlene Dietrich
```

### QA Format (qa_test.txt)
```
question\tanswer
```
Answers may contain multiple values, separated by `|`:
```
what films did [Michelle Trachtenberg] star in	Inspector Gadget|Black Christmas|Ice Princess
```

## Usage

1. Ensure LightRAG and related dependencies are installed:
```bash
pip install lightrag-hku
pip install transformers torch
```

2. Ensure MetaQA data files exist:
   - `MetaQA/kb.txt` - Knowledge graph triples
   - `MetaQA/1-hop/vanilla/qa_test.txt` - Test set QA pairs

3. Run the demo (from project root directory):
```bash
# Run from project root directory
python demo1202/metaqa_lightrag.py
```

## Output

- **Log files**: `log/metaqa_lightrag_YYYYMMDD_HHMMSS.log`
- **Evaluation results**: `cache/metaqa_demo/evaluation_results_YYYYMMDD_HHMMSS.json`
- **Console output**: Real-time display of evaluation progress and final accuracy

## Evaluation Metrics

- **Accuracy**: The proportion of predicted answers matching the standard answers
- Supports multi-answer matching (answers separated by `|`)
- Answers are normalized (lowercase, whitespace removed)

## Configuration

Model configuration used in the code:
- **LLM**: Qwen/Qwen2.5-1.5B-Instruct
- **Embedding**: BGE-M3 (path: `/home/infres/zzhu-24/large_files/bge-m3`)

To modify model configuration, edit the `initialize_rag()` function in `metaqa_lightrag.py`.

## Evaluation Mode Description

- **global**: Global retrieval based on knowledge graph
- **hybrid**: Hybrid retrieval mode (vector + graph)
- **local**: Local retrieval mode

By default, all three modes are evaluated, and results are saved in a JSON file.
