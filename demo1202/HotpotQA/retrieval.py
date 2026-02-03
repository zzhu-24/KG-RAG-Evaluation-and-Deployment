import json
import faiss
import numpy as np
import networkx as nx
from transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

def load_faiss(path):
    return faiss.read_index(path)

def load_graph(path):
    return nx.read_graphml(path)


def embed_query(model, query):
    vec = model.encode(
        [query],
        normalize_embeddings=True
    )
    return np.array(vec).astype("float32")

def retrieve_entities(index, metadata, query_vec, top_k=5):
    scores, ids = index.search(query_vec, top_k)
    return [metadata[i] for i in ids[0]]

def retrieve_relations(index, metadata, query_vec, top_k=5):
    scores, ids = index.search(query_vec, top_k)
    return [metadata[i] for i in ids[0]]

# local retrieval expansion from entities
def expand_from_entities(G, entities):
    context = []

    for ent in entities:
        node_id = ent["id"]

        if node_id not in G:
            continue

        # Node description
        context.append(G.nodes[node_id]["description"])

        # 1-hop neighbors
        for nbr in G.successors(node_id):
            edge_data = G[node_id][nbr]
            context.append(edge_data["description"])
            context.append(G.nodes[nbr]["description"])

    return context

# global retrieval from all relations
def expand_from_relations(G, relations):
    context = []

    for rel in relations:
        h, t = rel["head"], rel["tail"]

        if G.has_edge(h, t):
            edge_data = G[h][t]
            context.append(edge_data["description"])
            context.append(G.nodes[h]["description"])
            context.append(G.nodes[t]["description"])

    return context

def expand_from_relations(G, relations):
    context = []

    for rel in relations:
        h, t = rel["head"], rel["tail"]

        if G.has_edge(h, t):
            edge_data = G[h][t]
            context.append(edge_data["description"])
            context.append(G.nodes[h]["description"])
            context.append(G.nodes[t]["description"])

    return context

def assemble_context(entity_ctx, relation_ctx, max_length=3000):
    full_context = entity_ctx + relation_ctx
    text = "\n".join(full_context)
    return text[:max_length]


class AnswerLLM:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )

    def generate(self, prompt, max_new_tokens=256):
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

def build_answer_prompt(query, context):
    return f"""
    You are a question answering system using retrieved knowledge.

    Question:
    {query}

    Context:
    {context}

    Answer:
    """

def answer_query(
    query,
    embed_model,
    entity_index,
    entity_meta,
    relation_index,
    relation_meta,
    graph,
    llm,
    top_k_entity=5,
    top_k_relation=5
):
    # Embed query
    query_vec = embed_query(embed_model, query)

    # Dual-level retrieval
    entities = retrieve_entities(entity_index, entity_meta, query_vec, top_k_entity)
    relations = retrieve_relations(relation_index, relation_meta, query_vec, top_k_relation)

    # Graph expansion
    entity_context = expand_from_entities(graph, entities)
    relation_context = expand_from_relations(graph, relations)

    # Context assembly
    context = assemble_context(entity_context, relation_context)

    # Answer generation
    prompt = build_answer_prompt(query, context)
    return llm.generate(prompt)



