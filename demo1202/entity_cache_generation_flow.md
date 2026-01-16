# Entity 和 Relationship Cache 生成流程说明

## 概述

本文档说明 entity 和 relationship 数据是如何生成并保存到 cache 文件中的：
- Entity 数据: `cache/metaqa_demo/vdb_entities.json`
- Relationship 数据: `cache/metaqa_demo/vdb_relationships.json`

## 数据流程

### 1. 入口脚本
**文件**: `demo1202/metaqa_lightrag.py`

```python
# 第404行
rag.insert_custom_kg(custom_kg)
```

这里调用了 `LightRAG` 的 `insert_custom_kg` 方法，传入自定义知识图谱数据。

### 2. 插入实体到向量数据库
**文件**: `lightrag/lightrag.py`

**方法**: `ainsert_custom_kg()` (第2217-2380行)

关键代码片段：
```python
# 第2355-2367行
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

这里：
- 为每个实体生成唯一的ID（使用 `compute_mdhash_id`，前缀为 `ent-`）
- 准备实体数据，包括 `content`（实体名称 + 描述）、`entity_name`、`source_id` 等
- 调用 `entities_vdb.upsert()` 将数据插入向量数据库

### 3. 向量数据库存储实现
**文件**: `lightrag/kg/nano_vector_db_impl.py`

**类**: `NanoVectorDBStorage`

**方法**: `upsert()` (第95-142行)

关键处理步骤：

#### 3.1 准备数据
```python
# 第107-114行
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

#### 3.2 生成向量嵌入
```python
# 第116-123行
batches = [
    contents[i : i + self._max_batch_size]
    for i in range(0, len(contents), self._max_batch_size)
]

embedding_tasks = [self.embedding_func(batch) for batch in batches]
embeddings_list = await asyncio.gather(*embedding_tasks)
embeddings = np.concatenate(embeddings_list)
```

使用配置的 embedding 函数（在 `metaqa_lightrag.py` 中配置的 `hf_embed`）对实体内容进行向量化。

#### 3.3 压缩和编码向量
```python
# 第128-132行
for i, d in enumerate(list_data):
    # Compress vector using Float16 + zlib + Base64 for storage optimization
    vector_f16 = embeddings[i].astype(np.float32)
    compressed_vector = zlib.compress(vector_f16.tobytes())
    encoded_vector = base64.b64encode(compressed_vector).decode("utf-8")
    d["vector"] = encoded_vector
    d["__vector__"] = embeddings[i]
```

向量压缩流程：
1. 将 float32 向量转换为 float16（减少存储空间）
2. 使用 zlib 压缩
3. 使用 base64 编码为字符串

#### 3.4 插入到 NanoVectorDB
```python
# 第134-135行
client = await self._get_client()
results = client.upsert(datas=list_data)
```

调用 `NanoVectorDB` 的 `upsert` 方法将数据添加到内存中的向量数据库。

### 4. 保存到磁盘
**文件**: `lightrag/kg/nano_vector_db_impl.py`

**方法**: `index_done_callback()` (第272-305行)

```python
# 第290-298行
async with self._storage_lock:
    try:
        # Save data to disk
        self._client.save()  # 调用 NanoVectorDB 的 save 方法
        # Notify other processes that data has been updated
        await set_all_update_flags(self.namespace, workspace=self.workspace)
        self.storage_updated.value = False
        return True
    except Exception as e:
        logger.error(f"[{self.workspace}] Error saving data for {self.namespace}: {e}")
        return False
```

`self._client.save()` 会调用 `NanoVectorDB` 库的 `save()` 方法，将内存中的数据保存到 JSON 文件。

### 5. 文件路径
**文件**: `lightrag/kg/nano_vector_db_impl.py`

```python
# 第42-56行
working_dir = self.global_config["working_dir"]
if self.workspace:
    workspace_dir = os.path.join(working_dir, self.workspace)
else:
    workspace_dir = working_dir

self._client_file_name = os.path.join(
    workspace_dir, f"vdb_{self.namespace}.json"
)
```

对于 `metaqa_demo`：
- `working_dir` = `"./cache"`
- `workspace` = `"metaqa_demo"`
- `namespace` = `"entities"`
- 最终文件路径 = `"./cache/metaqa_demo/vdb_entities.json"`

## 数据结构

保存到 JSON 文件的数据结构：

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
      "vector": "base64编码的压缩向量字符串"
    },
    ...
  ],
  "matrix": "base64编码的所有向量矩阵"
}
```

## 关键组件

1. **NanoVectorDB**: 轻量级向量数据库库，负责向量索引和存储
2. **Embedding 函数**: 在 `metaqa_lightrag.py` 中配置的 `hf_embed`，使用 BGE-M3 模型
3. **向量压缩**: Float16 + zlib + Base64 三重压缩，大幅减少存储空间

## 相关代码位置

- 入口: `demo1202/metaqa_lightrag.py:404` - `rag.insert_custom_kg(custom_kg)`
- 插入逻辑: `lightrag/lightrag.py:2355-2367` - 准备数据并调用 `upsert`
- 向量化: `lightrag/kg/nano_vector_db_impl.py:95-142` - `upsert()` 方法
- 保存: `lightrag/kg/nano_vector_db_impl.py:272-305` - `index_done_callback()` 方法

---

# Relationship Cache 生成流程说明

## 概述

Relationship（关系）数据的生成和保存流程与 Entity 类似，但有一些特殊处理。

## 数据流程

### 1. 入口脚本
**文件**: `demo1202/metaqa_lightrag.py`

```python
# 第404行
rag.insert_custom_kg(custom_kg)
```

与 Entity 使用相同的入口，通过 `insert_custom_kg` 方法插入知识图谱数据。

### 2. 插入关系到向量数据库
**文件**: `lightrag/lightrag.py`

**方法**: `ainsert_custom_kg()` (第2369-2383行)

关键代码片段：
```python
# 第2369-2383行
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

这里：
- 为每个关系生成唯一的ID（使用 `compute_mdhash_id(src_id + tgt_id)`，前缀为 `rel-`）
- **重要**: 在生成ID前，会确保 `src_id` 和 `tgt_id` 按字典序排序（较小的在前），以保证一致性
- 准备关系数据，包括：
  - `content`: 格式为 `{keywords}\t{src_id}\n{tgt_id}\n{description}`
  - `src_id`: 源实体ID
  - `tgt_id`: 目标实体ID
  - `keywords`: 关系关键词
  - `description`: 关系描述
  - `weight`: 关系权重
  - `source_id`: 来源chunk ID
  - `file_path`: 文件路径
- 调用 `relationships_vdb.upsert()` 将数据插入向量数据库

### 3. 关系ID的一致性处理

**文件**: `lightrag/operate.py`

**方法**: `_merge_edges_then_upsert()` (第2361-2393行)

关键处理：
```python
# 第2361-2363行
# Sort src_id and tgt_id to ensure consistent ordering (smaller string first)
if src_id > tgt_id:
    src_id, tgt_id = tgt_id, src_id
```

在插入关系之前，会确保 `src_id` 和 `tgt_id` 按字典序排序，这样无论关系是从 A→B 还是 B→A 插入，都会生成相同的ID。

### 4. 删除旧关系记录

**文件**: `lightrag/operate.py`

**方法**: `_merge_edges_then_upsert()` (第2365-2373行)

```python
# 第2365-2373行
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

在插入新关系之前，会尝试删除可能存在的旧记录（包括正向和反向），以确保数据一致性。

### 5. 向量数据库存储实现

**文件**: `lightrag/kg/nano_vector_db_impl.py`

**类**: `NanoVectorDBStorage`

**方法**: `upsert()` (第95-142行)

Relationship 数据使用与 Entity 相同的向量化流程：

1. **准备数据**: 添加 `__id__` 和 `__created_at__` 字段
2. **生成向量嵌入**: 使用 embedding 函数对 `content` 字段进行向量化
3. **压缩和编码向量**: Float16 + zlib + Base64 三重压缩
4. **插入到 NanoVectorDB**: 调用 `client.upsert(datas=list_data)`

### 6. 保存到磁盘

**文件**: `lightrag/kg/nano_vector_db_impl.py`

**方法**: `index_done_callback()` (第272-305行)

与 Entity 使用相同的保存机制：
```python
# 第290-298行
async with self._storage_lock:
    try:
        # Save data to disk
        self._client.save()  # 调用 NanoVectorDB 的 save 方法
        # Notify other processes that data has been updated
        await set_all_update_flags(self.namespace, workspace=self.workspace)
        self.storage_updated.value = False
        return True
    except Exception as e:
        logger.error(f"[{self.workspace}] Error saving data for {self.namespace}: {e}")
        return False
```

### 7. 文件路径

**文件**: `lightrag/kg/nano_vector_db_impl.py`

```python
# 第42-56行
working_dir = self.global_config["working_dir"]
if self.workspace:
    workspace_dir = os.path.join(working_dir, self.workspace)
else:
    workspace_dir = working_dir

self._client_file_name = os.path.join(
    workspace_dir, f"vdb_{self.namespace}.json"
)
```

对于 `metaqa_demo`：
- `working_dir` = `"./cache"`
- `workspace` = `"metaqa_demo"`
- `namespace` = `"relationships"`
- 最终文件路径 = `"./cache/metaqa_demo/vdb_relationships.json"`

## Relationship 数据结构

保存到 JSON 文件的数据结构：

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
      "vector": "base64编码的压缩向量字符串"
    },
    ...
  ],
  "matrix": "base64编码的所有向量矩阵"
}
```

### Content 字段格式说明

Relationship 的 `content` 字段采用特殊格式：
```
{keywords}\t{src_id}\n{tgt_id}\n{description}
```

例如：
```
directed_by	Kismet
William Dieterle
Kismet directed_by William Dieterle
```

这种格式便于：
1. 向量化时包含完整的关系语义信息
2. 检索时能够匹配关系类型、实体对和描述

## Relationship 与 Entity 的区别

| 特性 | Entity | Relationship |
|------|--------|--------------|
| ID 前缀 | `ent-` | `rel-` |
| ID 生成方式 | `compute_mdhash_id(entity_name, prefix="ent-")` | `compute_mdhash_id(src_id + tgt_id, prefix="rel-")` |
| ID 排序 | 不需要 | 需要（src_id 和 tgt_id 按字典序排序） |
| Content 格式 | `{entity_name}\n{description}` | `{keywords}\t{src_id}\n{tgt_id}\n{description}` |
| 额外字段 | `entity_name`, `entity_type` | `src_id`, `tgt_id`, `keywords`, `weight` |
| 删除旧记录 | 不需要 | 需要（删除正向和反向记录） |

## 关键组件

1. **NanoVectorDB**: 轻量级向量数据库库，负责向量索引和存储
2. **Embedding 函数**: 在 `metaqa_lightrag.py` 中配置的 `hf_embed`，使用 BGE-M3 模型
3. **向量压缩**: Float16 + zlib + Base64 三重压缩，大幅减少存储空间
4. **关系一致性**: 通过排序 src_id 和 tgt_id 确保关系ID的一致性

## 相关代码位置

### Relationship 插入
- 入口: `demo1202/metaqa_lightrag.py:404` - `rag.insert_custom_kg(custom_kg)`
- 插入逻辑: `lightrag/lightrag.py:2369-2383` - 准备数据并调用 `upsert`
- 关系合并: `lightrag/operate.py:1871-2395` - `_merge_edges_then_upsert()` 方法
- 关系重建: `lightrag/operate.py:1314-1572` - `_rebuild_single_relationship()` 方法

### 向量化和保存
- 向量化: `lightrag/kg/nano_vector_db_impl.py:95-142` - `upsert()` 方法
- 保存: `lightrag/kg/nano_vector_db_impl.py:272-305` - `index_done_callback()` 方法

