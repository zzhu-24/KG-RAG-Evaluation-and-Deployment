# LightRAG 并发处理机制说明

## 同时处理的文件数如何确定

### 1. 核心参数：`max_parallel_insert`

同时处理的文件数由 **`max_parallel_insert`** 参数控制。

#### 1.1 默认值
- **默认值**：`2`（同时处理 2 个文件）
- 定义位置：`lightrag/constants.py`
  ```python
  DEFAULT_MAX_PARALLEL_INSERT = 2
  ```

#### 1.2 配置方式

**方式一：环境变量（推荐）**
```bash
export MAX_PARALLEL_INSERT=3
python demo1202/f1qa_prepare.py --news-file ./demo1202/F1QA/news/f1_news.jsonl
```

**方式二：在代码中设置**
```python
rag = LightRAG(
    working_dir=WORKING_DIR,
    max_parallel_insert=3,  # 设置同时处理的文件数
    # ... 其他参数
)
```

#### 1.3 代码实现位置

在 `lightrag/lightrag.py` 中：

```python
# 第 374-377 行：参数定义
max_parallel_insert: int = field(
    default=int(os.getenv("MAX_PARALLEL_INSERT", DEFAULT_MAX_PARALLEL_INSERT))
)
"""Maximum number of parallel insert operations."""

# 第 1732 行：使用信号量限制并发
semaphore = asyncio.Semaphore(self.max_parallel_insert)

# 第 1752 行：在文档处理函数中使用
async with semaphore:
    # 处理单个文档
    ...
```

### 2. 推荐配置值

根据官方文档，推荐配置如下：

| LLM 并发数 (MAX_ASYNC) | 推荐文件并发数 (MAX_PARALLEL_INSERT) | 说明 |
|----------------------|----------------------------------|------|
| 4 | 2 | 默认配置 |
| 8 | 2-3 | 中等并发 |
| 12 | 3-4 | 高并发 |
| 16 | 4-5 | 很高并发 |
| 20+ | 5-10 | 极高并发（最大不超过 10） |

**计算公式**：
```
MAX_PARALLEL_INSERT = MAX_ASYNC / 3 到 MAX_ASYNC / 4
范围：2 ≤ MAX_PARALLEL_INSERT ≤ 10
```

**示例**：
- 如果 `MAX_ASYNC = 12`，则 `MAX_PARALLEL_INSERT = 3` 或 `4`
- 如果 `MAX_ASYNC = 8`，则 `MAX_PARALLEL_INSERT = 2` 或 `3`

### 3. 为什么不能设置太大？

#### 3.1 资源竞争
- **内存占用**：每个文件处理需要缓存大量数据（实体、关系、分块等）
- **磁盘 I/O**：文件级别的提交会增加磁盘写入压力
- **LLM 资源**：虽然 LLM 有全局并发控制，但过多文件会增加队列等待时间

#### 3.2 合并阶段冲突
- 多个文件同时处理时，实体和关系的合并可能产生命名冲突
- 过多的并发文件会增加冲突概率，降低合并效率

#### 3.3 错误恢复成本
- LightRAG 按文件提交处理结果
- 如果系统错误，所有中间阶段的文档都需要重新处理
- 并发文件数越多，错误恢复成本越高

### 4. 多层级并发控制架构

LightRAG 使用多层级并发控制策略：

```
┌─────────────────────────────────────────────────────────┐
│ 文档级别并发控制 (max_parallel_insert = 2)              │
│ ┌─────────────┐  ┌─────────────┐                        │
│ │  文档 A      │  │  文档 B      │                        │
│ └──────┬──────┘  └──────┬──────┘                        │
│        │                │                                │
│        ▼                ▼                                │
│ ┌──────────────────────────────────────┐               │
│ │ 分块级别并发控制 (llm_model_max_async) │               │
│ │ 文档 A: 4 个分块并发处理              │               │
│ │ 文档 B: 4 个分块并发处理              │               │
│ └──────────────┬─────────────────────────┘               │
│                │                                          │
│                ▼                                          │
│ ┌──────────────────────────────────────┐                │
│ │ LLM 全局并发控制 (llm_model_max_async)│                │
│ │ 所有请求共享一个优先级队列            │                │
│ │ 优先级：查询 > 合并 > 提取            │                │
│ └──────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────┘
```

#### 4.1 理论并发计算

**分块级别理论并发数**：
```
理论分块并发数 = max_parallel_insert × llm_model_max_async
```

**示例**：
- `max_parallel_insert = 2`
- `llm_model_max_async = 4`
- 理论分块并发数 = 2 × 4 = **8 个分块同时处理**

**注意**：实际 LLM 请求并发数仍受 `llm_model_max_async` 限制（全局限制为 4）

### 5. 实际使用建议

#### 5.1 根据 LLM 能力调整

**本地部署 LLM**：
- 根据 LLM 服务的并发能力设置 `MAX_ASYNC`
- 推荐：`MAX_ASYNC = LLM服务并发能力`
- 然后：`MAX_PARALLEL_INSERT = MAX_ASYNC / 3`

**API 提供商**：
- 根据 API 限制设置 `MAX_ASYNC`
- 如果请求被拒绝，LightRAG 会自动重试 3 次
- 观察日志，如果频繁重试，说明 `MAX_ASYNC` 设置过高

#### 5.2 根据系统资源调整

**内存充足**：
- 可以适当增加 `MAX_PARALLEL_INSERT`（但不超过 10）

**内存有限**：
- 保持默认值 2
- 或降低到 1（串行处理）

**磁盘 I/O 受限**：
- 保持较小的 `MAX_PARALLEL_INSERT`
- 因为每个文件处理完成后需要写入磁盘

#### 5.3 性能优化建议

1. **优先调整 `MAX_ASYNC`**
   - 这是性能瓶颈的主要因素
   - 根据 LLM 能力设置

2. **然后调整 `MAX_PARALLEL_INSERT`**
   - 设置为 `MAX_ASYNC / 3` 到 `MAX_ASYNC / 4`
   - 范围：2-10

3. **监控和调优**
   - 观察处理速度
   - 检查是否有 LLM 请求重试
   - 根据实际情况微调

### 6. 配置示例

#### 示例 1：默认配置（适合大多数场景）
```bash
# 不设置环境变量，使用默认值
# MAX_ASYNC = 4 (默认)
# MAX_PARALLEL_INSERT = 2 (默认)
python demo1202/f1qa_prepare.py --news-file ./demo1202/F1QA/news/f1_news.jsonl
```

#### 示例 2：中等并发配置
```bash
export MAX_ASYNC=8
export MAX_PARALLEL_INSERT=3
python demo1202/f1qa_prepare.py --news-file ./demo1202/F1QA/news/f1_news.jsonl
```

#### 示例 3：高并发配置（需要强大的 LLM 服务）
```bash
export MAX_ASYNC=12
export MAX_PARALLEL_INSERT=4
python demo1202/f1qa_prepare.py --news-file ./demo1202/F1QA/news/f1_news.jsonl
```

#### 示例 4：低资源环境（串行处理）
```bash
export MAX_ASYNC=2
export MAX_PARALLEL_INSERT=1
python demo1202/f1qa_prepare.py --news-file ./demo1202/F1QA/news/f1_news.jsonl
```

### 7. 验证配置

可以通过以下方式验证配置是否生效：

#### 7.1 查看日志
处理过程中会显示：
```
INFO: Processing 232 document(s)
INFO: Extracting stage 1/232: <file_path>
INFO: Extracting stage 2/232: <file_path>
```

如果 `MAX_PARALLEL_INSERT = 2`，你会看到最多 2 个文件同时进入提取阶段。

#### 7.2 监控资源使用
- **CPU/GPU 使用率**：应该稳定在合理范围
- **内存使用**：不应该持续增长（说明没有内存泄漏）
- **磁盘 I/O**：应该有规律的写入模式

### 8. 常见问题

#### Q1: 为什么设置 `MAX_PARALLEL_INSERT=10` 后速度没有提升？
**A**: 因为 LLM 的并发能力是瓶颈。即使同时处理 10 个文件，LLM 也只能同时处理 `MAX_ASYNC` 个请求。过多的文件并发只会增加等待时间，不会提升速度。

#### Q2: 如何知道 `MAX_ASYNC` 是否设置过高？
**A**: 观察日志中是否有 LLM 请求重试的警告。如果频繁重试，说明超过了 LLM 服务的并发限制。

#### Q3: 可以设置 `MAX_PARALLEL_INSERT=1` 吗？
**A**: 可以，但会串行处理文件，速度较慢。除非系统资源非常有限，否则不推荐。

#### Q4: 为什么推荐值是 `MAX_ASYNC / 3`？
**A**: 这是经验值，平衡了并发效率和资源消耗。过少会浪费 LLM 并发能力，过多会增加冲突和资源竞争。

### 9. 总结

- **同时处理的文件数**由 `MAX_PARALLEL_INSERT` 控制
- **默认值**是 2
- **推荐范围**：2-10
- **推荐公式**：`MAX_PARALLEL_INSERT = MAX_ASYNC / 3` 到 `MAX_ASYNC / 4`
- **配置方式**：环境变量 `MAX_PARALLEL_INSERT` 或代码中设置
- **关键点**：不要设置过大，2-5 通常是最优范围
