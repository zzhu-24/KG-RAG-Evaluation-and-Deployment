# LightRAG Cache Data Format Documentation

## Overview

This document describes the data formats and generation processes for various cache files in LightRAG, including entity and relationship vector databases, GraphML files, and JSON export formats.

## Table of Contents

1. [Entity Cache Generation Flow](#entity-cache-generation-flow)
2. [Relationship Cache Generation Flow](#relationship-cache-generation-flow)
3. [GraphML File Format](#graphml-file-format)
4. [Graph Data JSON Format](#graph-data-json-format)
5. [Data Structure Comparison](#data-structure-comparison)

---

## Entity Cache Generation Flow

### 1. Entry Point

**File**: `demo1202/metaqa_lightrag.py`

```python
# Line 404
rag.insert_custom_kg(custom_kg)
```

This calls the `insert_custom_kg` method of `LightRAG` with custom knowledge graph data.

### 2. Insert Entities into Vector Database

**File**: `lightrag/lightrag.py`

**Method**: `ainsert_custom_kg()` (Lines 2355-2367)

Key code snippet:
```python
# Insert entities into vector storage with consistent format
data_for_vdb = {
    compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
        "content": dp["entity_name"] + "\n" + dp["description"],
        "entity_name": dp["entity_name"],
        "source_id": dp["source_id"],
        "description": dp["description"],
        "entity_type": dp["entity_type"],
        "file_path": dp.get("file_path", "custom_kg"),
    }
    for dp in all_entities_data
}
await self.entities_vdb.upsert(data_for_vdb)
```

This step:
- Generates unique IDs for each entity (using `compute_mdhash_id` with prefix `ent-`)
- Prepares entity data including `content` (entity name + description), `entity_name`, `source_id`, etc.
- Calls `entities_vdb.upsert()` to insert data into the vector database

### 3. Vector Database Storage Implementation

**File**: `lightrag/kg/nano_vector_db_impl.py`

**Class**: `NanoVectorDBStorage`

**Method**: `upsert()` (Lines 95-142)

#### 3.1 Prepare Data
```python
# Lines 107-114
current_time = int(time.time())
list_data = [
    {
        "__id__": k,
        "__created_at__": current_time,
        **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
    }
    for k, v in data.items()
]
contents = [v["content"] for v in data.values()]
```

#### 3.2 Generate Vector Embeddings
```python
# Lines 116-123
batches = [
    contents[i : i + self._max_batch_size]
    for i in range(0, len(contents), self._max_batch_size)
]

embedding_tasks = [self.embedding_func(batch) for batch in batches]
embeddings_list = await asyncio.gather(*embedding_tasks)
embeddings = np.concatenate(embeddings_list)
```

Uses the configured embedding function (e.g., `hf_embed` configured in `metaqa_lightrag.py`) to vectorize entity content.

#### 3.3 Compress and Encode Vectors
```python
# Lines 128-132
for i, d in enumerate(list_data):
    # Compress vector using Float16 + zlib + Base64 for storage optimization
    vector_f16 = embeddings[i].astype(np.float32)
    compressed_vector = zlib.compress(vector_f16.tobytes())
    encoded_vector = base64.b64encode(compressed_vector).decode("utf-8")
    d["vector"] = encoded_vector
    d["__vector__"] = embeddings[i]
```

Vector compression process:
1. Convert float32 vectors to float16 (reduces storage space)
2. Compress using zlib
3. Encode as base64 string

#### 3.4 Insert into NanoVectorDB
```python
# Lines 134-135
client = await self._get_client()
results = client.upsert(datas=list_data)
```

Calls `NanoVectorDB.upsert()` to add data to the in-memory vector database.

### 4. Save to Disk

**File**: `lightrag/kg/nano_vector_db_impl.py`

**Method**: `index_done_callback()` (Lines 272-305)

```python
# Lines 290-298
async with self._storage_lock:
    try:
        # Save data to disk
        self._client.save()  # Calls NanoVectorDB's save method
        # Notify other processes that data has been updated
        await set_all_update_flags(self.namespace, workspace=self.workspace)
        self.storage_updated.value = False
        return True
    except Exception as e:
        logger.error(f"[{self.workspace}] Error saving data for {self.namespace}: {e}")
        return False
```

`self._client.save()` calls the `NanoVectorDB` library's `save()` method to persist in-memory data to a JSON file.

### 5. File Path

**File**: `lightrag/kg/nano_vector_db_impl.py`

```python
# Lines 42-56
working_dir = self.global_config["working_dir"]
if self.workspace:
    workspace_dir = os.path.join(working_dir, self.workspace)
else:
    workspace_dir = working_dir

self._client_file_name = os.path.join(
    workspace_dir, f"vdb_{self.namespace}.json"
)
```

For `metaqa_demo`:
- `working_dir` = `"./cache"`
- `workspace` = `"metaqa_demo"`
- `namespace` = `"entities"`
- Final file path = `"./cache/metaqa_demo/vdb_entities.json"`

### Entity Data Structure

The JSON file structure:

```json
{
  "embedding_dim": 1024,
  "data": [
    {
      "__id__": "ent-xxx",
      "__created_at__": 1767875356,
      "content": "EntityName\nDescription",
      "entity_name": "EntityName",
      "source_id": "UNKNOWN",
      "file_path": "demo1202/MetaQA/kb.txt",
      "vector": "base64-encoded compressed vector string"
    },
    ...
  ],
  "matrix": "base64-encoded vector matrix of all entities"
}
```

---

## Relationship Cache Generation Flow

### 1. Entry Point

**File**: `demo1202/metaqa_lightrag.py`

```python
# Line 404
rag.insert_custom_kg(custom_kg)
```

Same entry point as entities, using the `insert_custom_kg` method.

### 2. Insert Relationships into Vector Database

**File**: `lightrag/lightrag.py`

**Method**: `ainsert_custom_kg()` (Lines 2369-2383)

Key code snippet:
```python
# Insert relationships into vector storage with consistent format
data_for_vdb = {
    compute_mdhash_id(dp["src_id"] + dp["tgt_id"], prefix="rel-"): {
        "src_id": dp["src_id"],
        "tgt_id": dp["tgt_id"],
        "source_id": dp["source_id"],
        "content": f"{dp['keywords']}\t{dp['src_id']}\n{dp['tgt_id']}\n{dp['description']}",
        "keywords": dp["keywords"],
        "description": dp["description"],
        "weight": dp["weight"],
        "file_path": dp.get("file_path", "custom_kg"),
    }
    for dp in all_relationships_data
}
await self.relationships_vdb.upsert(data_for_vdb)
```

This step:
- Generates unique IDs for each relationship (using `compute_mdhash_id(src_id + tgt_id)` with prefix `rel-`)
- **Important**: Before generating IDs, ensures `src_id` and `tgt_id` are sorted lexicographically (smaller first) for consistency
- Prepares relationship data including:
  - `content`: Format `{keywords}\t{src_id}\n{tgt_id}\n{description}`
  - `src_id`: Source entity ID
  - `tgt_id`: Target entity ID
  - `keywords`: Relationship keywords
  - `description`: Relationship description
  - `weight`: Relationship weight
  - `source_id`: Source chunk ID
  - `file_path`: File path
- Calls `relationships_vdb.upsert()` to insert data into the vector database

### 3. Relationship ID Consistency Handling

**File**: `lightrag/operate.py`

**Method**: `_merge_edges_then_upsert()` (Lines 2361-2393)

Key processing:
```python
# Lines 2361-2363
# Sort src_id and tgt_id to ensure consistent ordering (smaller string first)
if src_id > tgt_id:
    src_id, tgt_id = tgt_id, src_id
```

Before inserting relationships, ensures `src_id` and `tgt_id` are sorted lexicographically, so the same ID is generated whether the relationship is inserted as A→B or B→A.

### 4. Delete Old Relationship Records

**File**: `lightrag/operate.py`

**Method**: `_merge_edges_then_upsert()` (Lines 2365-2373)

```python
# Lines 2365-2373
if relationships_vdb is not None:
    rel_vdb_id = compute_mdhash_id(src_id + tgt_id, prefix="rel-")
    rel_vdb_id_reverse = compute_mdhash_id(tgt_id + src_id, prefix="rel-")
    try:
        await relationships_vdb.delete([rel_vdb_id, rel_vdb_id_reverse])
    except Exception as e:
        logger.debug(
            f"Could not delete old relationship vector records {rel_vdb_id}, {rel_vdb_id_reverse}: {e}"
        )
```

Before inserting new relationships, attempts to delete any existing old records (both forward and reverse) to ensure data consistency.

### 5. Vector Database Storage Implementation

**File**: `lightrag/kg/nano_vector_db_impl.py`

**Class**: `NanoVectorDBStorage`

**Method**: `upsert()` (Lines 95-142)

Relationship data uses the same vectorization process as entities:

1. **Prepare Data**: Add `__id__` and `__created_at__` fields
2. **Generate Vector Embeddings**: Use embedding function to vectorize the `content` field
3. **Compress and Encode Vectors**: Float16 + zlib + Base64 triple compression
4. **Insert into NanoVectorDB**: Call `client.upsert(datas=list_data)`

### 6. Save to Disk

Same saving mechanism as entities (see Entity Cache Generation Flow section 4).

### 7. File Path

For `metaqa_demo`:
- `working_dir` = `"./cache"`
- `workspace` = `"metaqa_demo"`
- `namespace` = `"relationships"`
- Final file path = `"./cache/metaqa_demo/vdb_relationships.json"`

### Relationship Data Structure

The JSON file structure:

```json
{
  "embedding_dim": 1024,
  "data": [
    {
      "__id__": "rel-xxx",
      "__created_at__": 1767875356,
      "content": "keywords\tsrc_entity\ntgt_entity\ndescription",
      "src_id": "src_entity",
      "tgt_id": "tgt_entity",
      "source_id": "chunk-xxx",
      "keywords": "relation_keywords",
      "description": "relation_description",
      "weight": 1.0,
      "file_path": "demo1202/MetaQA/kb.txt",
      "vector": "base64-encoded compressed vector string"
    },
    ...
  ],
  "matrix": "base64-encoded vector matrix of all relationships"
}
```

### Content Field Format

Relationship `content` field uses a special format:
```
{keywords}\t{src_id}\n{tgt_id}\n{description}
```

Example:
```
directed_by	Kismet
William Dieterle
Kismet directed_by William Dieterle
```

This format facilitates:
1. Including complete relationship semantic information during vectorization
2. Matching relationship types, entity pairs, and descriptions during retrieval

### Relationship vs Entity Differences

| Feature | Entity | Relationship |
|---------|--------|--------------|
| ID Prefix | `ent-` | `rel-` |
| ID Generation | `compute_mdhash_id(entity_name, prefix="ent-")` | `compute_mdhash_id(src_id + tgt_id, prefix="rel-")` |
| ID Sorting | Not required | Required (src_id and tgt_id sorted lexicographically) |
| Content Format | `{entity_name}\n{description}` | `{keywords}\t{src_id}\n{tgt_id}\n{description}` |
| Additional Fields | `entity_name`, `entity_type` | `src_id`, `tgt_id`, `keywords`, `weight` |
| Delete Old Records | Not required | Required (delete forward and reverse records) |

---

## GraphML File Format

### Overview

GraphML files are automatically generated by LightRAG when using `NetworkXStorage` (the default graph storage). They contain all node and edge information including descriptions, entity types, and metadata.

### File Location

**File**: `cache/{workspace}/graph_chunk_entity_relation.graphml`

For `metaqa_demo`:
- Path: `./cache/metaqa_demo/graph_chunk_entity_relation.graphml`
- Size: ~48.39 MB
- Nodes: 43,234
- Edges: 124,680

### GraphML Structure

```xml
<?xml version='1.0' encoding='utf-8'?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns" ...>
  <!-- Key definitions for nodes -->
  <key id="d0" for="node" attr.name="entity_id" attr.type="string"/>
  <key id="d1" for="node" attr.name="entity_type" attr.type="string"/>
  <key id="d2" for="node" attr.name="description" attr.type="string"/>
  <key id="d3" for="node" attr.name="source_id" attr.type="string"/>
  <key id="d4" for="node" attr.name="file_path" attr.type="string"/>
  <key id="d5" for="node" attr.name="created_at" attr.type="long"/>
  
  <!-- Key definitions for edges -->
  <key id="d6" for="edge" attr.name="weight" attr.type="double"/>
  <key id="d7" for="edge" attr.name="description" attr.type="string"/>
  <key id="d8" for="edge" attr.name="keywords" attr.type="string"/>
  <key id="d9" for="edge" attr.name="source_id" attr.type="string"/>
  <key id="d10" for="edge" attr.name="file_path" attr.type="string"/>
  <key id="d11" for="edge" attr.name="created_at" attr.type="long"/>
  
  <graph edgedefault="undirected">
    <!-- Nodes -->
    <node id="EntityName">
      <data key="d0">EntityID</data>
      <data key="d1">EntityType</data>
      <data key="d2">Description</data>
      <data key="d3">SourceChunkID</data>
      <data key="d4">FilePath</data>
      <data key="d5">CreatedAtTimestamp</data>
    </node>
    
    <!-- Edges -->
    <edge source="SourceEntity" target="TargetEntity">
      <data key="d6">Weight</data>
      <data key="d7">Description</data>
      <data key="d8">Keywords</data>
      <data key="d9">SourceChunkID</data>
      <data key="d10">FilePath</data>
      <data key="d11">CreatedAtTimestamp</data>
    </edge>
  </graph>
</graphml>
```

### Node Fields

| Key ID | Field Name | Type | Description | Example |
|--------|------------|------|-------------|---------|
| d0 | entity_id | string | Entity ID (same as node ID) | "Kismet" |
| d1 | entity_type | string | Entity type | "Movie", "Person", "UNKNOWN", "Genre" |
| d2 | description | string | Entity description | "Kismet (relations: has_tags, in_language, ...)" |
| d3 | source_id | string | Source chunk ID | "UNKNOWN" |
| d4 | file_path | string | File path | "demo1202/MetaQA/kb.txt" |
| d5 | created_at | long | Creation timestamp | 1767875330 |

### Edge Fields

| Key ID | Field Name | Type | Description | Example |
|--------|------------|------|-------------|---------|
| d6 | weight | double | Relationship weight | 1.0 |
| d7 | description | string | Relationship description | "Kismet directed_by William Dieterle" |
| d8 | keywords | string | Relationship keywords | "directed_by" |
| d9 | source_id | string | Source chunk ID | "UNKNOWN" |
| d10 | file_path | string | File path | "demo1202/MetaQA/kb.txt" |
| d11 | created_at | long | Creation timestamp | 1767875333 |

### Example Data

**Node Example**:
```xml
<node id="Kismet">
  <data key="d0">Kismet</data>
  <data key="d1">Movie</data>
  <data key="d2">Kismet (relations: has_tags, in_language, written_by, starred_actors, directed_by, release_year)</data>
  <data key="d3">UNKNOWN</data>
  <data key="d4">demo1202/MetaQA/kb.txt</data>
  <data key="d5">1767875330</data>
</node>
```

**Edge Example**:
```xml
<edge source="Kismet" target="William Dieterle">
  <data key="d6">1.0</data>
  <data key="d7">Kismet directed_by William Dieterle</data>
  <data key="d8">directed_by</data>
  <data key="d9">UNKNOWN</data>
  <data key="d10">demo1202/MetaQA/kb.txt</data>
  <data key="d11">1767875333</data>
</edge>
```

### Entity Type Distribution (metaqa_demo)

- **Person**: 21,704 (50.2%)
- **Movie**: 16,427 (38.0%)
- **UNKNOWN**: 5,083 (11.8%)
- **Genre**: 20 (0.05%)

### How GraphML is Generated

**File**: `lightrag/kg/networkx_impl.py`

1. **Insert Node** (Lines 132-140):
```python
async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
    graph = await self._get_graph()
    graph.add_node(node_id, **node_data)  # Stores description, entity_type, etc. as node attributes
```

2. **Insert Edge** (Lines 142-152):
```python
async def upsert_edge(self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]) -> None:
    graph = await self._get_graph()
    graph.add_edge(source_node_id, target_node_id, **edge_data)  # Stores description, keywords, etc. as edge attributes
```

3. **Save to Disk** (Lines 503-535):
```python
async def index_done_callback(self) -> bool:
    # ...
    NetworkXStorage.write_nx_graph(self._graph, self._graphml_xml_file, self.workspace)
    # Calls nx.write_graphml() to save as GraphML format
```

---

## Graph Data JSON Format

### Overview

`graph_data.json` is **not automatically generated** by LightRAG. It is created by manually running a conversion script that extracts data from GraphML files for visualization purposes (e.g., importing into Neo4j).

### Generation Script

**File**: `examples/graph_visual_with_neo4j.py`

### Conversion Process

1. **Read GraphML File**:
```python
def xml_to_json(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    namespace = {"": "http://graphml.graphdrawing.org/xmlns"}
    
    data = {"nodes": [], "edges": []}
    
    # Extract nodes
    for node in root.findall(".//node", namespace):
        node_data = {
            "id": node.get("id").strip('"'),
            "entity_type": node.find("./data[@key='d1']", namespace).text.strip('"'),
            "description": node.find("./data[@key='d2']", namespace).text,
            "source_id": node.find("./data[@key='d3']", namespace).text,
        }
        data["nodes"].append(node_data)
    
    # Extract edges
    for edge in root.findall(".//edge", namespace):
        edge_data = {
            "source": edge.get("source").strip('"'),
            "target": edge.get("target").strip('"'),
            "weight": float(edge.find("./data[@key='d6']", namespace).text),
            "description": edge.find("./data[@key='d7']", namespace).text,
            "keywords": edge.find("./data[@key='d8']", namespace).text,
            "source_id": edge.find("./data[@key='d9']", namespace).text,
        }
        data["edges"].append(edge_data)
    
    return data
```

2. **Save as JSON**:
```python
def convert_xml_to_json(xml_path, output_path):
    json_data = xml_to_json(xml_path)
    if json_data:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
```

### JSON Structure

```json
{
  "nodes": [
    {
      "id": "EntityName",
      "entity_type": "Movie",
      "description": "Entity description",
      "source_id": "chunk-xxx"
    },
    ...
  ],
  "edges": [
    {
      "source": "SourceEntity",
      "target": "TargetEntity",
      "weight": 1.0,
      "description": "Relationship description",
      "keywords": "relation_keywords",
      "source_id": "chunk-xxx"
    },
    ...
  ]
}
```

### Why metaqa_demo Doesn't Have graph_data.json

1. `metaqa_demo` has the `graph_chunk_entity_relation.graphml` file (contains all information)
2. The conversion script `examples/graph_visual_with_neo4j.py` was not run
3. Therefore, `graph_data.json` was not generated

### How to Generate graph_data.json for metaqa_demo

You can run the conversion script:

```python
# Modify WORKING_DIR in examples/graph_visual_with_neo4j.py
WORKING_DIR = "./cache/metaqa_demo"  # Change to metaqa_demo

# Then run
python examples/graph_visual_with_neo4j.py
```

Or use a Python script directly:

```python
import os
import json
import xml.etree.ElementTree as ET

def convert_graphml_to_json(graphml_path, json_path):
    tree = ET.parse(graphml_path)
    root = tree.getroot()
    namespace = {"": "http://graphml.graphdrawing.org/xmlns"}
    
    data = {"nodes": [], "edges": []}
    
    # Extract nodes
    for node in root.findall(".//node", namespace):
        node_data = {
            "id": node.get("id").strip('"'),
            "entity_type": node.find("./data[@key='d1']", namespace).text.strip('"') if node.find("./data[@key='d1']", namespace) is not None else "",
            "description": node.find("./data[@key='d2']", namespace).text if node.find("./data[@key='d2']", namespace) is not None else "",
            "source_id": node.find("./data[@key='d3']", namespace).text if node.find("./data[@key='d3']", namespace) is not None else "",
        }
        data["nodes"].append(node_data)
    
    # Extract edges
    for edge in root.findall(".//edge", namespace):
        edge_data = {
            "source": edge.get("source").strip('"'),
            "target": edge.get("target").strip('"'),
            "weight": float(edge.find("./data[@key='d6']", namespace).text) if edge.find("./data[@key='d6']", namespace) is not None else 0.0,
            "description": edge.find("./data[@key='d7']", namespace).text if edge.find("./data[@key='d7']", namespace) is not None else "",
            "keywords": edge.find("./data[@key='d8']", namespace).text if edge.find("./data[@key='d8']", namespace) is not None else "",
            "source_id": edge.find("./data[@key='d9']", namespace).text if edge.find("./data[@key='d9']", namespace) is not None else "",
        }
        data["edges"].append(edge_data)
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Conversion complete: {json_path}")

# Usage
convert_graphml_to_json(
    "./cache/metaqa_demo/graph_chunk_entity_relation.graphml",
    "./cache/metaqa_demo/graph_data.json"
)
```

---

## Data Structure Comparison

### Storage Locations

| Data Type | File Path | Format | Auto-Generated |
|-----------|-----------|--------|----------------|
| Entity Vectors | `cache/{workspace}/vdb_entities.json` | JSON | Yes |
| Relationship Vectors | `cache/{workspace}/vdb_relationships.json` | JSON | Yes |
| Graph Data | `cache/{workspace}/graph_chunk_entity_relation.graphml` | GraphML/XML | Yes |
| Graph Data (JSON) | `cache/{workspace}/graph_data.json` | JSON | No (manual conversion) |

### Description Source

#### For `insert_custom_kg` (metaqa_demo case)

**File**: `lightrag/lightrag.py` Lines 2260-2288

```python
for entity_data in custom_kg.get("entities", []):
    entity_name = entity_data["entity_name"]
    entity_type = entity_data.get("entity_type", "UNKNOWN")
    description = entity_data.get("description", "No description provided")  # From input data
```

In `metaqa_lightrag.py` (Lines 123-206), the `convert_triples_to_custom_kg` function generates simple descriptions:
```python
entity_data["description"] = f"{entity_name} (relations: {', '.join(unique_relations)})"
```

#### For Document-Extracted Entities (paper_demo case)

**File**: `lightrag/operate.py` Lines 1593-1847

More detailed descriptions are generated through LLM:
- Uses LLM to merge entity information from multiple chunks
- Generates more complete descriptions
- Infers entity types

### Key Components

1. **NanoVectorDB**: Lightweight vector database library responsible for vector indexing and storage
2. **Embedding Function**: Configured in `metaqa_lightrag.py` as `hf_embed`, using BGE-M3 model
3. **Vector Compression**: Float16 + zlib + Base64 triple compression, significantly reducing storage space
4. **Relationship Consistency**: Ensures relationship ID consistency by sorting src_id and tgt_id

### Related Code Locations

#### Entity Insertion
- Entry: `demo1202/metaqa_lightrag.py:404` - `rag.insert_custom_kg(custom_kg)`
- Insert Logic: `lightrag/lightrag.py:2355-2367` - Prepare data and call `upsert`
- Vectorization: `lightrag/kg/nano_vector_db_impl.py:95-142` - `upsert()` method
- Save: `lightrag/kg/nano_vector_db_impl.py:272-305` - `index_done_callback()` method

#### Relationship Insertion
- Entry: `demo1202/metaqa_lightrag.py:404` - `rag.insert_custom_kg(custom_kg)`
- Insert Logic: `lightrag/lightrag.py:2369-2383` - Prepare data and call `upsert`
- Relationship Merge: `lightrag/operate.py:1871-2395` - `_merge_edges_then_upsert()` method
- Relationship Rebuild: `lightrag/operate.py:1314-1572` - `_rebuild_single_relationship()` method

#### Graph Storage
- Node Insert: `lightrag/kg/networkx_impl.py:132-140` - `upsert_node()` method
- Edge Insert: `lightrag/kg/networkx_impl.py:142-152` - `upsert_edge()` method
- Save: `lightrag/kg/networkx_impl.py:503-535` - `index_done_callback()` method

#### GraphML to JSON Conversion
- Script: `examples/graph_visual_with_neo4j.py` - Conversion script

---

## Summary

1. **GraphML Files**: Automatically generated by LightRAG, containing all node and edge attributes (description, entity_type, etc.)
2. **graph_data.json**: Requires manual running of a conversion script, used for visualization (e.g., importing into Neo4j)
3. **metaqa_demo doesn't have graph_data.json**: Because the conversion script was not run, but the GraphML file contains all information
4. **Description Sources**:
   - `insert_custom_kg`: From the input `custom_kg` data
   - Document extraction: Generated through LLM with more detailed descriptions

All information is stored in GraphML files, and `graph_data.json` is just a conversion format for easier visualization.

