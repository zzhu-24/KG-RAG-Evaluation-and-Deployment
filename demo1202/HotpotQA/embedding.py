from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from pathlib import Path

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

def embed_texts(model, texts):
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    return np.array(embeddings).astype("float32")

def build_entity_index(nodes, model):
    texts = [n["description"] for n in nodes]
    vectors = embed_texts(model, texts)

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    return index, vectors

def build_relation_index(edges, model):
    texts = [e["description"] for e in edges]
    vectors = embed_texts(model, texts)

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    return index, vectors


def save_index(index, path):
    faiss.write_index(index, str(path))

def save_metadata(profiles, path):
    with open(path, "w") as f:
        for p in profiles:
            f.write(json.dumps(p) + "\n")

def main():
    nodes = load_jsonl("processed/nodes.jsonl")
    edges = load_jsonl("processed/edges.jsonl")

    model = SentenceTransformer("BAAI/bge-m3")

    entity_index, _ = build_entity_index(nodes, model)
    relation_index, _ = build_relation_index(edges, model)

    save_index(entity_index, "processed/entity.index")
    save_index(relation_index, "processed/relation.index")

    save_metadata(nodes, "processed/entity_meta.json")
    save_metadata(edges, "processed/relation_meta.json")

    print("Embedding & indexing complete.")


if __name__ == "__main__":
    main()

