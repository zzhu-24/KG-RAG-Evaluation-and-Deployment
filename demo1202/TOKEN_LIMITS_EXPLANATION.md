# LightRAG Token 限制参数详解

## 概述

本文档详细解释 LightRAG 中三个关键 Token 限制参数的含义和工作机制：
- `max_entity_tokens`
- `max_relation_tokens`
- `max_total_tokens`

---

## 1. 参数含义

### 1.1 `max_entity_tokens`（默认：6000）

**含义：所有实体上下文的总 Token 数限制**

- ❌ **不是**：实体数量的限制
- ❌ **不是**：单个实体的 Token 数
- ✅ **是**：所有实体信息序列化后的总 Token 数

### 1.2 `max_relation_tokens`（默认：8000）

**含义：所有关系上下文的总 Token 数限制**

- ❌ **不是**：关系数量的限制
- ❌ **不是**：单个关系的 Token 数
- ✅ **是**：所有关系信息序列化后的总 Token 数

### 1.3 `max_total_tokens`（默认：30000）

**含义：整个查询上下文的总 Token 数限制**

包括：
- System Prompt（系统提示词）
- Entities Context（实体上下文）
- Relations Context（关系上下文）
- Text Chunks（文本块）
- Query（查询文本）
- Buffer（安全缓冲，200 tokens）

---

## 2. 工作流程详解

### 2.1 Stage 1: 搜索阶段

**检索实体和关系：**
```python
# 示例：查询 "what happened to Guanyu ZHOU"
Local query: 40 entities, 25 relations
Global query: 50 entities, 40 relations
Raw search results: 84 entities, 57 relations
```

**说明：**
- 系统先检索到 84 个相关实体和 57 个关系
- 这些是**候选结果**，还没有应用 Token 限制

### 2.2 Stage 2: 截断阶段（Token Truncation）

#### 2.2.1 实体上下文构建

**步骤 1：将每个实体转换为 JSON 对象**

```python
# 每个实体包含以下字段
{
    "entity": "Guanyu ZHOU",           # 实体名称
    "type": "Person",                  # 实体类型
    "description": "周冠宇是一名中国F1车手...",  # 实体描述（已合并）
    "created_at": "2024-01-12 10:00:00",  # 创建时间
    "file_path": "news1.jsonl,news2.jsonl"  # 来源文件
}
```

**关键点：**
- `description` 字段是**已经合并后的摘要**（见第 3 节）
- 如果实体出现在多个文档中，描述会在数据插入阶段合并

**步骤 2：序列化所有实体**

```python
# 将所有实体序列化为 JSON 字符串
entities_str = "\n".join(
    json.dumps(entity, ensure_ascii=False) 
    for entity in entities_context
)
```

**步骤 3：计算 Token 数并截断**

```python
# 计算总 token 数
total_tokens = len(tokenizer.encode(entities_str))

# 如果超过 max_entity_tokens，从末尾截断
if total_tokens > max_entity_tokens:
    # 使用 truncate_list_by_token_size 函数
    # 从列表末尾开始移除实体，直到总 token 数 <= max_entity_tokens
    entities_context = truncate_list_by_token_size(
        entities_context,
        key=lambda x: json.dumps(x),
        max_token_size=max_entity_tokens,
        tokenizer=tokenizer
    )
```

**截断逻辑：**
```python
def truncate_list_by_token_size(list_data, key, max_token_size, tokenizer):
    tokens = 0
    for i, data in enumerate(list_data):
        tokens += len(tokenizer.encode(key(data)))
        if tokens > max_token_size:
            return list_data[:i]  # 返回前面的元素，丢弃后面的
    return list_data  # 如果没超过限制，返回全部
```

**示例：**
```
检索到 84 个实体，总 token 数 = 12,000
max_entity_tokens = 6,000

截断过程：
  实体 1-30: 累计 5,800 tokens ✅ 保留
  实体 31: 累计 6,100 tokens ❌ 超过限制
  结果：保留前 30 个实体，丢弃后 54 个实体

After truncation: 30 entities (从 84 减少到 30)
```

#### 2.2.2 关系上下文构建

关系使用相同的截断逻辑：

```python
# 每个关系包含
{
    "entity1": "Guanyu ZHOU",
    "entity2": "Alfa Romeo",
    "description": "周冠宇是阿尔法·罗密欧车队的车手...",
    "created_at": "2024-01-12 10:00:00",
    "file_path": "news1.jsonl"
}
```

**截断示例：**
```
检索到 57 个关系，总 token 数 = 10,000
max_relation_tokens = 8,000

结果：保留前 45 个关系，丢弃后 12 个关系
After truncation: 45 relations (从 57 减少到 45)
```

### 2.3 Stage 3: 构建最终上下文

#### 2.3.1 动态 Token 分配

```python
# 计算各部分 token 开销
sys_prompt_tokens = len(tokenizer.encode(system_prompt))      # ~500
kg_context_tokens = len(tokenizer.encode(entities + relations))  # ~15,000
query_tokens = len(tokenizer.encode(query))                   # ~20
buffer_tokens = 200  # 安全缓冲

# 计算可用于 chunks 的 token 数
available_chunk_tokens = max_total_tokens - (
    sys_prompt_tokens + kg_context_tokens + query_tokens + buffer_tokens
)

# 示例：
# max_total_tokens = 30,000
# available_chunk_tokens = 30,000 - (500 + 15,000 + 20 + 200) = 14,280
```

#### 2.3.2 Chunk Token 截断

```python
# 对文本块应用 token 截断
truncated_chunks = await process_chunks_unified(
    unique_chunks=merged_chunks,  # 合并后的块（可能很多）
    chunk_token_limit=available_chunk_tokens,  # 动态计算的限制
    ...
)
```

**最终结果：**
```
Final context: 30 entities, 45 relations, 20 chunks
```

---

## 3. Entity 多上下文处理机制

### 3.1 问题场景

**场景：** 一个实体（如 "Guanyu ZHOU"）出现在多个文档块中

```
Chunk 1: "周冠宇在2024年加入阿尔法·罗密欧车队..."
Chunk 2: "周冠宇在2023年获得最佳新秀奖..."
Chunk 3: "周冠宇在2024年阿布扎比站获得第10名..."
```

### 3.2 数据插入阶段的处理

**步骤 1：收集所有描述**

```python
# 系统会收集该实体在所有 chunk 中的描述
description_list = [
    "周冠宇在2024年加入阿尔法·罗密欧车队...",
    "周冠宇在2023年获得最佳新秀奖...",
    "周冠宇在2024年阿布扎比站获得第10名...",
]
```

**步骤 2：Map-Reduce 摘要**

```python
async def _handle_entity_relation_summary(
    entity_name: "Guanyu ZHOU",
    description_list: [...],  # 多个描述
    ...
):
    """
    使用 Map-Reduce 策略合并多个描述：
    
    1. 如果描述数量少且总 token 数小：
       → 直接拼接，不使用 LLM
    
    2. 如果描述数量多或总 token 数大：
       → 使用 LLM 进行摘要合并
       
    3. 如果描述非常多：
       → 先分组，每组摘要，再递归摘要
    """
```

**步骤 3：存储合并后的描述**

```python
# 最终实体存储
{
    "entity_name": "Guanyu ZHOU",
    "description": "周冠宇是中国F1车手，2024年加入阿尔法·罗密欧车队，2023年获得最佳新秀奖，在2024年阿布扎比站获得第10名...",  # 已合并
    "entity_type": "Person",
    "source_ids": ["chunk-1", "chunk-2", "chunk-3"],  # 保留来源
    ...
}
```

### 3.3 查询阶段的处理

**在查询阶段，每个实体只有一个 `description` 字段：**

```python
# 查询时，实体已经是合并后的状态
{
    "entity": "Guanyu ZHOU",
    "type": "Person",
    "description": "周冠宇是中国F1车手，2024年加入阿尔法·罗密欧车队...",  # 已合并
    ...
}
```

**说明：**
- ✅ **不需要在查询时取舍**：描述已经在数据插入阶段合并
- ✅ **保留来源信息**：`source_ids` 字段记录所有来源 chunk
- ✅ **智能摘要**：如果描述太多，系统会自动使用 LLM 摘要

### 3.4 取舍策略

**如果实体描述太长，系统会：**

1. **自动摘要**：使用 LLM 将多个描述合并为一个简洁的描述
2. **保留关键信息**：摘要会保留最重要的信息
3. **保留来源**：`source_ids` 和 `file_path` 字段记录所有来源

**配置参数：**
```python
# 触发摘要的条件
force_llm_summary_on_merge = 8  # 描述数量超过 8 个时触发摘要
summary_max_tokens = 1200        # 摘要的最大 token 数
summary_context_size = 12000     # 摘要上下文的最大 token 数
```

---

## 4. 实际示例

### 4.1 查询示例

**查询：** "what happened to Guanyu ZHOU"

**Stage 1: 搜索**
```
检索到：84 entities, 57 relations
```

**Stage 2: 截断**
```
实体总 token: 12,000 > max_entity_tokens (6,000)
→ 截断后：30 entities

关系总 token: 10,000 > max_relation_tokens (8,000)
→ 截断后：45 relations
```

**Stage 3: 构建上下文**
```
计算各部分 token：
  System Prompt: 500
  Entities: 6,000
  Relations: 8,000
  Query: 20
  Buffer: 200
  Total used: 14,720

可用 chunk tokens: 30,000 - 14,720 = 15,280

最终：30 entities, 45 relations, 20 chunks
```

### 4.2 Token 分配示例

```
┌─────────────────────────────────────────┐
│ max_total_tokens = 30,000              │
├─────────────────────────────────────────┤
│ System Prompt:        500 tokens       │
│ Entities Context:   6,000 tokens       │
│ Relations Context:  8,000 tokens      │
│ Query:                 20 tokens       │
│ Buffer:               200 tokens       │
├─────────────────────────────────────────┤
│ Subtotal:          14,720 tokens       │
├─────────────────────────────────────────┤
│ Available for Chunks: 15,280 tokens   │
└─────────────────────────────────────────┘
```

---

## 5. 关键要点总结

### 5.1 Token 限制的含义

| 参数 | 限制对象 | 截断方式 |
|------|---------|---------|
| `max_entity_tokens` | **所有实体**的总 token 数 | 从列表末尾移除实体 |
| `max_relation_tokens` | **所有关系**的总 token 数 | 从列表末尾移除关系 |
| `max_total_tokens` | **整个上下文**的总 token 数 | 动态分配各部分，剩余给 chunks |

### 5.2 Entity 多上下文处理

✅ **数据插入阶段**：
- 自动收集实体在所有 chunk 中的描述
- 使用 Map-Reduce 策略合并描述
- 如果描述太多，使用 LLM 摘要

✅ **查询阶段**：
- 每个实体只有一个合并后的 `description`
- 不需要手动取舍
- 来源信息保存在 `source_ids` 和 `file_path` 字段

### 5.3 截断顺序

**优先级（从高到低）：**
1. **System Prompt** - 必须保留
2. **Query** - 必须保留
3. **Entities** - 按 `max_entity_tokens` 截断
4. **Relations** - 按 `max_relation_tokens` 截断
5. **Chunks** - 使用剩余 token 数

**截断策略：**
- 实体和关系：从列表**末尾**移除（保留前面的，丢弃后面的）
- 文本块：按相关性排序后截断

---

## 6. 优化建议

### 6.1 如果遇到信息丢失

**问题：** 重要实体被截断（在列表末尾）

**解决方案：**
1. **增加 `max_entity_tokens`**：从 6000 增加到 8000 或 10000
2. **减少检索数量**：减小 `top_k` 参数，只检索最相关的实体
3. **使用更精确的检索**：提高 `cosine_threshold`，只检索高相似度的实体

### 6.2 如果遇到显存不足

**问题：** CUDA OOM 错误

**解决方案：**
1. **减小 `max_total_tokens`**：从 30000 减少到 15000 或 8000
2. **减小 `max_entity_tokens` 和 `max_relation_tokens`**：相应减小
3. **减小 `top_k` 和 `chunk_top_k`**：减少检索数量

### 6.3 推荐配置

**平衡配置（推荐）：**
```python
{
    "top_k": 30,
    "chunk_top_k": 15,
    "max_total_tokens": 15000,
    "max_entity_tokens": 4000,
    "max_relation_tokens": 5000,
}
```

**低显存配置：**
```python
{
    "top_k": 20,
    "chunk_top_k": 10,
    "max_total_tokens": 8000,
    "max_entity_tokens": 3000,
    "max_relation_tokens": 3000,
}
```

---

## 附录：代码位置

- **截断函数**：`lightrag/operate.py::_apply_token_truncation()`
- **实体合并**：`lightrag/operate.py::_handle_entity_relation_summary()`
- **Token 计算**：`lightrag/utils.py::truncate_list_by_token_size()`
- **上下文构建**：`lightrag/operate.py::_build_context_str()`

---

**文档生成时间**：2026-01-12  
**适用版本**：LightRAG (最新版本)
