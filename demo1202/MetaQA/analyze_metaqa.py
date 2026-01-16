#!/usr/bin/env python3
"""
MetaQA Dataset Analysis Script

This script collects comprehensive statistics about the MetaQA dataset
and generates an English report.
"""

import os
import json
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set
from datetime import datetime
import statistics

# MetaQA数据路径
METAQA_DIR = "./demo1202/MetaQA"
KB_FILE = os.path.join(METAQA_DIR, "kb.txt")
OUTPUT_DIR = "./demo1202/analysis_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_kg_triples(kb_path: str) -> List[Tuple[str, str, str]]:
    """Load knowledge graph triples from MetaQA format."""
    triples = []
    with open(kb_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) >= 3:
                s, p, o = parts[0].strip(), parts[1].strip(), parts[2].strip()
                triples.append((s, p, o))
    return triples


def load_qa_pairs(qa_path: str) -> List[Tuple[str, str]]:
    """Load question-answer pairs from MetaQA format."""
    qa_pairs = []
    with open(qa_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "\t" in line:
                parts = line.split("\t", 1)
                question = parts[0].strip()
                answer = parts[1].strip() if len(parts) > 1 else ""
                qa_pairs.append((question, answer))
    return qa_pairs


def analyze_knowledge_graph(kb_path: str) -> Dict:
    """Analyze the knowledge graph statistics."""
    print("Analyzing knowledge graph...")
    triples = load_kg_triples(kb_path)
    
    entities = set()
    predicates = set()
    predicate_counts = Counter()
    entity_degrees = defaultdict(int)
    
    for s, p, o in triples:
        entities.add(s)
        entities.add(o)
        predicates.add(p)
        predicate_counts[p] += 1
        entity_degrees[s] += 1
        entity_degrees[o] += 1
    
    # Calculate entity degree statistics
    degrees = list(entity_degrees.values())
    
    stats = {
        "total_triples": len(triples),
        "unique_entities": len(entities),
        "unique_predicates": len(predicates),
        "predicate_distribution": dict(predicate_counts.most_common()),
        "entity_degree_stats": {
            "min": min(degrees) if degrees else 0,
            "max": max(degrees) if degrees else 0,
            "mean": statistics.mean(degrees) if degrees else 0,
            "median": statistics.median(degrees) if degrees else 0,
        },
        "avg_triples_per_entity": len(triples) / len(entities) if entities else 0,
    }
    
    return stats


def analyze_qa_dataset(qa_path: str) -> Dict:
    """Analyze QA dataset statistics."""
    qa_pairs = load_qa_pairs(qa_path)
    
    question_lengths = []
    answer_counts = []
    unique_questions = set()
    unique_answers = set()
    multi_answer_count = 0
    
    for question, answer in qa_pairs:
        question_lengths.append(len(question.split()))
        unique_questions.add(question.lower())
        
        if "|" in answer:
            multi_answer_count += 1
            answers = answer.split("|")
            answer_counts.append(len(answers))
            unique_answers.update(a.strip().lower() for a in answers)
        else:
            answer_counts.append(1)
            unique_answers.add(answer.strip().lower())
    
    stats = {
        "total_qa_pairs": len(qa_pairs),
        "unique_questions": len(unique_questions),
        "unique_answers": len(unique_answers),
        "multi_answer_pairs": multi_answer_count,
        "multi_answer_ratio": multi_answer_count / len(qa_pairs) if qa_pairs else 0,
        "question_length_stats": {
            "min": min(question_lengths) if question_lengths else 0,
            "max": max(question_lengths) if question_lengths else 0,
            "mean": statistics.mean(question_lengths) if question_lengths else 0,
            "median": statistics.median(question_lengths) if question_lengths else 0,
        },
        "answer_count_stats": {
            "min": min(answer_counts) if answer_counts else 0,
            "max": max(answer_counts) if answer_counts else 0,
            "mean": statistics.mean(answer_counts) if answer_counts else 0,
            "median": statistics.median(answer_counts) if answer_counts else 0,
        },
    }
    
    return stats


def collect_all_statistics() -> Dict:
    """Collect all statistics from the MetaQA dataset."""
    print("Collecting MetaQA dataset statistics...")
    
    stats = {
        "knowledge_graph": {},
        "qa_datasets": {},
        "summary": {},
    }
    
    # Analyze knowledge graph
    if os.path.exists(KB_FILE):
        stats["knowledge_graph"] = analyze_knowledge_graph(KB_FILE)
    else:
        print(f"Warning: {KB_FILE} not found")
    
    # Analyze QA datasets
    hops = ["1-hop", "2-hop", "3-hop"]
    splits = ["train", "dev", "test"]
    variants = ["vanilla", "ntm"]
    
    qa_stats = {}
    total_qa_pairs = 0
    
    for hop in hops:
        qa_stats[hop] = {}
        
        # Analyze vanilla variant
        for split in splits:
            qa_path = os.path.join(METAQA_DIR, hop, "vanilla", f"qa_{split}.txt")
            if os.path.exists(qa_path):
                qa_stats[hop][f"vanilla_{split}"] = analyze_qa_dataset(qa_path)
                total_qa_pairs += qa_stats[hop][f"vanilla_{split}"]["total_qa_pairs"]
        
        # Analyze ntm variant
        for split in splits:
            qa_path = os.path.join(METAQA_DIR, hop, "ntm", f"qa_{split}.txt")
            if os.path.exists(qa_path):
                qa_stats[hop][f"ntm_{split}"] = analyze_qa_dataset(qa_path)
    
    stats["qa_datasets"] = qa_stats
    stats["summary"]["total_qa_pairs"] = total_qa_pairs
    
    return stats


def generate_report(stats: Dict) -> str:
    """Generate an English report about the MetaQA dataset."""
    
    kg = stats.get("knowledge_graph", {})
    qa_datasets = stats.get("qa_datasets", {})
    
    report = []
    report.append("=" * 80)
    report.append("MetaQA Dataset Analysis Report")
    report.append("=" * 80)
    report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # 1. Overview
    report.append("1. DATASET OVERVIEW")
    report.append("-" * 80)
    report.append("MetaQA is a large-scale question answering dataset over a knowledge graph.")
    report.append("The dataset contains questions that require 1-hop, 2-hop, or 3-hop reasoning")
    report.append("over a knowledge graph extracted from the WikiMovies dataset.")
    report.append("")
    
    # 2. Knowledge Graph Statistics
    report.append("2. KNOWLEDGE GRAPH STATISTICS")
    report.append("-" * 80)
    if kg:
        report.append(f"Total Triples: {kg.get('total_triples', 0):,}")
        report.append(f"Unique Entities: {kg.get('unique_entities', 0):,}")
        report.append(f"Unique Predicates (Relation Types): {kg.get('unique_predicates', 0):,}")
        report.append(f"Average Triples per Entity: {kg.get('avg_triples_per_entity', 0):.2f}")
        report.append("")
        
        # Entity degree statistics
        degree_stats = kg.get("entity_degree_stats", {})
        report.append("Entity Degree Statistics:")
        report.append(f"  - Minimum degree: {degree_stats.get('min', 0)}")
        report.append(f"  - Maximum degree: {degree_stats.get('max', 0)}")
        report.append(f"  - Mean degree: {degree_stats.get('mean', 0):.2f}")
        report.append(f"  - Median degree: {degree_stats.get('median', 0):.2f}")
        report.append("")
        
        # Predicate distribution
        pred_dist = kg.get("predicate_distribution", {})
        if pred_dist:
            report.append("Top 10 Most Frequent Predicates:")
            for i, (pred, count) in enumerate(list(pred_dist.items())[:10], 1):
                report.append(f"  {i:2d}. {pred:30s}: {count:6,} occurrences")
        report.append("")
    else:
        report.append("Knowledge graph statistics not available.")
        report.append("")
    
    # 3. QA Dataset Statistics
    report.append("3. QUESTION-ANSWER DATASET STATISTICS")
    report.append("-" * 80)
    
    for hop in ["1-hop", "2-hop", "3-hop"]:
        if hop not in qa_datasets:
            continue
            
        report.append(f"\n3.{int(hop[0])} {hop.upper()} Questions")
        report.append("-" * 80)
        
        hop_stats = qa_datasets[hop]
        
        # Vanilla variant
        if any(k.startswith("vanilla_") for k in hop_stats.keys()):
            report.append(f"\n{hop} - Vanilla Variant:")
            for split in ["train", "dev", "test"]:
                key = f"vanilla_{split}"
                if key in hop_stats:
                    s = hop_stats[key]
                    report.append(f"  {split.upper():6s}: {s.get('total_qa_pairs', 0):6,} QA pairs")
                    report.append(f"           Unique questions: {s.get('unique_questions', 0):6,}")
                    report.append(f"           Unique answers: {s.get('unique_answers', 0):6,}")
                    report.append(f"           Multi-answer pairs: {s.get('multi_answer_pairs', 0):6,} ({s.get('multi_answer_ratio', 0)*100:.1f}%)")
                    
                    q_len = s.get("question_length_stats", {})
                    report.append(f"           Question length: {q_len.get('mean', 0):.1f} words (avg), {q_len.get('median', 0):.1f} words (median)")
                    
                    a_count = s.get("answer_count_stats", {})
                    report.append(f"           Answers per question: {a_count.get('mean', 0):.2f} (avg), {a_count.get('max', 0)} (max)")
        
        # NTM variant
        if any(k.startswith("ntm_") for k in hop_stats.keys()):
            report.append(f"\n{hop} - NTM Variant:")
            for split in ["train", "dev", "test"]:
                key = f"ntm_{split}"
                if key in hop_stats:
                    s = hop_stats[key]
                    report.append(f"  {split.upper():6s}: {s.get('total_qa_pairs', 0):6,} QA pairs")
    
    # 4. Dataset Characteristics
    report.append("\n4. DATASET CHARACTERISTICS")
    report.append("-" * 80)
    report.append("Key Features:")
    report.append("  - Multi-hop reasoning: Questions require 1, 2, or 3 hops in the knowledge graph")
    report.append("  - Multiple answers: Many questions have multiple correct answers")
    report.append("  - Entity-centric: Questions focus on entities and their relationships")
    report.append("  - Movie domain: Knowledge graph extracted from WikiMovies dataset")
    report.append("")
    
    # 5. Usage Notes
    report.append("5. USAGE NOTES")
    report.append("-" * 80)
    report.append("The dataset is commonly used for:")
    report.append("  - Evaluating knowledge graph question answering systems")
    report.append("  - Testing multi-hop reasoning capabilities")
    report.append("  - Benchmarking retrieval-augmented generation (RAG) systems")
    report.append("  - Comparing different query strategies (naive, local, global, hybrid)")
    report.append("")
    report.append("Typical evaluation metrics:")
    report.append("  - Accuracy: Percentage of questions answered correctly")
    report.append("  - Hits@1: Percentage of questions where the top-1 answer is correct")
    report.append("")
    
    # 6. File Structure
    report.append("6. FILE STRUCTURE")
    report.append("-" * 80)
    report.append("MetaQA/")
    report.append("  ├── kb.txt                    # Knowledge graph triples (subject|predicate|object)")
    report.append("  ├── 1-hop/")
    report.append("  │   ├── vanilla/              # Standard question format")
    report.append("  │   │   ├── qa_train.txt")
    report.append("  │   │   ├── qa_dev.txt")
    report.append("  │   │   └── qa_test.txt")
    report.append("  │   └── ntm/                  # Neural topic model variant")
    report.append("  ├── 2-hop/                    # Same structure as 1-hop")
    report.append("  └── 3-hop/                    # Same structure as 1-hop")
    report.append("")
    
    report.append("=" * 80)
    
    return "\n".join(report)


def main():
    """Main function to collect statistics and generate report."""
    print("Starting MetaQA dataset analysis...")
    
    # Collect statistics
    stats = collect_all_statistics()
    
    # Save statistics to JSON
    stats_file = os.path.join(OUTPUT_DIR, "metaqa_statistics.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"Statistics saved to: {stats_file}")
    
    # Generate and save report
    report = generate_report(stats)
    report_file = os.path.join(OUTPUT_DIR, "metaqa_report.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report saved to: {report_file}")
    
    # Print report to console
    print("\n" + report)
    
    print(f"\nAnalysis complete! Output files saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()


