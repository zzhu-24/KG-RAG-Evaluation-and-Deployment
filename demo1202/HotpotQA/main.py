"""
Main entry point for LightRAG-style system.

Usage:
- Run indexing once
- Then run queries multiple times

This file orchestrates the pipeline, it does NOT implement logic.
"""

from pathlib import Path
import json

# ===== CONFIGURATION =====

DATA_PATH = "./data/hotpot_dev_distractor_v1.json"
WORKDIR = Path("processed")

LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
EMBED_MODEL = "BAAI/bge-m3"

TOP_K_ENTITY = 5
TOP_K_RELATION = 5


# ===== INDEXING PIPELINE =====

def build_index():
    print("\n[LightRAG] Step 1: Chunking")
    from data_preprocessing_chunking import main as chunking_main
    chunking_main()

    print("\n[LightRAG] Step 2: Entity & Relation Extraction")
    from entity_relation_extraction import main as extraction_main
    extraction_main()

    print("\n[LightRAG] Step 3: LLM Profiling")
    from llm_profiling import (
        collect_entity_mentions,
        collect_relation_mentions,
        load_chunks,
        profile_entities,
        profile_relations
    )
    from retrieval import AnswerLLM

    kg_records = [json.loads(l) for l in open(WORKDIR / "kg.jsonl")]
    chunk_text = load_chunks(WORKDIR / "chunks.jsonl")

    entity_mentions = collect_entity_mentions(kg_records)
    relation_mentions = collect_relation_mentions(kg_records)

    llm = AnswerLLM(LLM_MODEL)

    nodes = profile_entities(entity_mentions, chunk_text, llm)
    edges = profile_relations(relation_mentions, chunk_text, llm)

    print("\n[LightRAG] Step 4: Deduplication")
    from deduplication import deduplicate_entities, deduplicate_relations

    node_index = deduplicate_entities(nodes)
    edge_index = deduplicate_relations(edges)

    WORKDIR.mkdir(parents=True, exist_ok=True)

    with open(WORKDIR / "nodes.jsonl", "w") as f:
        for n in node_index.values():
            f.write(json.dumps(n) + "\n")

    with open(WORKDIR / "edges.jsonl", "w") as f:
        for e in edge_index.values():
            f.write(json.dumps(e) + "\n")

    print("\n[LightRAG] Step 5: Knowledge Graph Construction")
    from building_graph import build_knowledge_graph, save_graph

    G = build_knowledge_graph(
        list(node_index.values()),
        list(edge_index.values())
    )
    save_graph(G, WORKDIR / "knowledge_graph.graphml")

    print("\n[LightRAG] Step 6: Embedding & Vector Indexing")
    from embedding import main as embedding_main
    embedding_main()

    print("\n[LightRAG] Indexing complete.\n")


# ===== QUERY PIPELINE =====

def query_system(question: str):
    from retrieval import (
        answer_query,
        load_faiss,
        load_graph,
        load_jsonl,
        AnswerLLM
    )
    from sentence_transformers import SentenceTransformer

    embed_model = SentenceTransformer(EMBED_MODEL)
    llm = AnswerLLM(LLM_MODEL)

    entity_index = load_faiss(WORKDIR / "entity.index")
    relation_index = load_faiss(WORKDIR / "relation.index")

    entity_meta = load_jsonl(WORKDIR / "entity_meta.json")
    relation_meta = load_jsonl(WORKDIR / "relation_meta.json")

    graph = load_graph(WORKDIR / "knowledge_graph.graphml")

    answer = answer_query(
        question,
        embed_model,
        entity_index,
        entity_meta,
        relation_index,
        relation_meta,
        graph,
        llm,
        TOP_K_ENTITY,
        TOP_K_RELATION
    )

    return answer


# ===== MAIN =====

def main():
    """
    Toggle indexing and querying here.
    """

    # Run ONCE
    build_index()

    # Run MANY times
    question = "Who was the first governor of the Utah Territory?"
    answer = query_system(question)

    print("\nQuestion:")
    print(question)
    print("\nAnswer:")
    print(answer)


if __name__ == "__main__":
    main()
