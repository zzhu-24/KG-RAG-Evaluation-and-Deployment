import networkx as nx
import json
from pathlib import Path


def load_nodes(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_edges(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

def build_knowledge_graph(nodes, edges):
    G = nx.DiGraph()

    # Add nodes
    for n in nodes:
        G.add_node(
            n["id"],
            type="entity",
            description=n["description"],
            source_chunks=n["source_chunks"]
        )

    # Add edges
    for e in edges:
        if not G.has_node(e["head"]) or not G.has_node(e["tail"]):
            continue
        G.add_edge(
            e["head"],
            e["tail"],
            relation=e["relation"],
            description=e["description"],
            source_chunks=e["source_chunks"]
        )

    return G

def save_graph(G, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(G, output_path)

def main():
    nodes_path = Path("processed/nodes.jsonl")
    edges_path = Path("processed/edges.jsonl")
    graph_path = Path("processed/knowledge_graph.graphml")

    nodes = load_nodes(nodes_path)
    edges = load_edges(edges_path)

    G = build_knowledge_graph(nodes, edges)

    print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    save_graph(G, graph_path)


if __name__ == "__main__":
    main()
