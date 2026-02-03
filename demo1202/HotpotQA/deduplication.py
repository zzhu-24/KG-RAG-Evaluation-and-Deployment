def deduplicate_entities(nodes):
    """
    nodes: list of entity profiles
    Returns: dict(entity_name -> entity_object)
    """
    entity_index = {}

    for n in nodes:
        name = n["entity"].strip()

        if name not in entity_index:
            entity_index[name] = {
                "id": name,
                "description": n["description"],
                "source_chunks": set(n["source_chunks"])
            }
        else:
            # merge evidence
            entity_index[name]["source_chunks"].update(n["source_chunks"])

    # convert sets to lists
    for e in entity_index.values():
        e["source_chunks"] = list(e["source_chunks"])

    return entity_index

def deduplicate_relations(edges):
    """
    edges: list of relation profiles
    Returns: dict(relation_id -> relation_object)
    """
    relation_index = {}

    for e in edges:
        h = e["head"].strip()
        r = e["relation"].strip()
        t = e["tail"].strip()

        rel_id = f"{h}||{r}||{t}"

        if rel_id not in relation_index:
            relation_index[rel_id] = {
                "id": rel_id,
                "head": h,
                "relation": r,
                "tail": t,
                "description": e["description"],
                "source_chunks": set(e["source_chunks"])
            }
        else:
            relation_index[rel_id]["source_chunks"].update(e["source_chunks"])

    for r in relation_index.values():
        r["source_chunks"] = list(r["source_chunks"])

    return relation_index
