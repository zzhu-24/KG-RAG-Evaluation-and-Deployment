from collections import defaultdict
import json

def collect_entity_mentions(kg_records):
    entity_to_chunks = defaultdict(list)

    for rec in kg_records:
        chunk_id = rec["chunk_id"]
        for ent in rec["entities"]:
            entity_to_chunks[ent].append(chunk_id)

    return entity_to_chunks

def collect_relation_mentions(kg_records):
    rel_to_chunks = defaultdict(list)

    for rec in kg_records:
        chunk_id = rec["chunk_id"]
        for r in rec["relations"]:
            key = (r["head"], r["relation"], r["tail"])
            rel_to_chunks[key].append(chunk_id)

    return rel_to_chunks

def load_chunks(path):
    chunk_text = {}
    with open(path) as f:
        for line in f:
            c = json.loads(line)
            cid = f"{c['doc_title']}::{c['chunk_id']}"
            chunk_text[cid] = c["text"]
    return chunk_text


def build_entity_profile_prompt(entity, texts):
    joined = "\n".join(texts[:5])  # limit evidence

    return f"""
    You are summarizing an entity for knowledge graph indexing.

    Entity name: {entity}

    Based on the following evidence texts, write a concise, factual description
    of the entity (2–3 sentences max).

    Evidence:
    {joined}

    Description:
    """
def build_relation_profile_prompt(head, rel, tail, texts):
    joined = "\n".join(texts[:5])

    return f"""
    You are summarizing a semantic relationship for knowledge graph indexing.

    Relationship:
    {head} — {rel} → {tail}

    Based on the following evidence texts, write a concise description
    of this relationship (1–2 sentences).

    Evidence:
    {joined}

    Description:
    """

def profile_entities(entity_to_chunks, chunk_text, extractor):
    nodes = []

    for entity, chunk_ids in entity_to_chunks.items():
        texts = [chunk_text[cid] for cid in chunk_ids if cid in chunk_text]

        prompt = build_entity_profile_prompt(entity, texts)
        desc = extractor.extract(prompt, max_new_tokens=128)

        nodes.append({
            "entity": entity,
            "description": desc.strip(),
            "source_chunks": chunk_ids
        })

    return nodes

def profile_relations(rel_to_chunks, chunk_text, extractor):
    edges = []

    for (h, r, t), chunk_ids in rel_to_chunks.items():
        texts = [chunk_text[cid] for cid in chunk_ids if cid in chunk_text]

        prompt = build_relation_profile_prompt(h, r, t, texts)
        desc = extractor.extract(prompt, max_new_tokens=128)

        edges.append({
            "head": h,
            "relation": r,
            "tail": t,
            "description": desc.strip(),
            "source_chunks": chunk_ids
        })

    return edges

