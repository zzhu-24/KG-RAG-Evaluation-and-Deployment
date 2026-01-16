# F1QA Prepare 脚本流程说明

## 概述

`f1qa_prepare.py` 是 F1QA LightRAG 系统的数据准备脚本，负责将新闻数据插入到 LightRAG 知识图谱系统中。该脚本支持增量更新，可以智能地跳过已处理的文档，避免重复处理。

## 主要功能

1. **增量更新支持**：自动检测已存在的文档，只处理新文档或未完成的文档
2. **失败文档处理**：可选择跳过或重新处理失败的文档
3. **批量插入**：支持批量插入文档，提高处理效率
4. **进度监控**：可选的处理进度实时监控功能

## 完整流程

### 1. 初始化阶段

#### 1.1 日志系统初始化
- 创建日志目录 `./log`
- 生成带时间戳的日志文件：`f1qa_prepare_YYYYMMDD_HHMMSS.log`
- 配置日志输出到文件和控制台

#### 1.2 设备检测
- 自动检测 GPU 是否可用
- 如果可用，显示 GPU 信息（名称、CUDA 版本、显存大小）
- 否则使用 CPU（会显示警告）

#### 1.3 工作目录设置
- 工作目录：`./cache/f1qa_demo`
- 自动创建目录（如果不存在）

### 2. 数据加载阶段

#### 2.1 加载新闻数据 (`load_news_from_jsonl`)
```
输入：JSONL 文件路径
处理：
  - 逐行读取 JSONL 文件
  - 解析每行的 JSON 数据
  - 验证必需字段（content 或 title）
  - 过滤无效数据（空行、JSON 解析失败、缺少必需字段）
输出：新闻列表（List[Dict]）
```

**数据格式要求**：
- 每行一个 JSON 对象
- 必须包含 `content` 或 `title` 字段
- 可选字段：`url`, `published_time`, `summary`

#### 2.2 数据格式化 (`format_news_for_insertion`)
```
输入：新闻字典
处理：
  - 提取标题（Title）
  - 提取发布时间（Published time）
  - 提取摘要（Summary）
  - 提取内容（Content，优先使用 content，否则使用 summary）
  - 提取 URL（Source）
输出：格式化的文本字符串
```

**格式化后的文本结构**：
```
Title: <标题>
Published time: <发布时间>
Summary: <摘要>
Content: <内容>
Source: <URL>
```

### 3. LightRAG 初始化阶段

#### 3.1 模型加载 (`initialize_rag`)
```
步骤：
  1. 加载嵌入模型（bge-m3）
     - 模型路径：/home/infres/zzhu-24/large_files/bge-m3
     - GPU 模式：使用 float16 精度
     - CPU 模式：使用 float32 精度
     - 将模型移动到指定设备（GPU/CPU）
  
  2. 创建 LightRAG 实例
     - 工作目录：./cache/f1qa_demo
     - LLM 模型：Qwen/Qwen2.5-3B-Instruct
     - 嵌入函数：使用 bge-m3 模型
     - 嵌入维度：1024
     - 最大 token 数：5000
  
  3. 初始化存储
     - 初始化所有存储组件
     - 初始化 pipeline_status（处理管道状态）
```

### 4. 增量更新检查阶段

#### 4.1 文档存在性检查 (`insert_news_to_rag`)
```
对每个新闻项：
  1. 生成文件路径标识符
     - 优先使用 URL
     - 如果没有 URL，使用 "news_{索引}_{标题前50字符}"
  
  2. 检查文档是否已存在
     - 通过 file_path 查询 doc_status 存储
     - 获取文档状态（如果存在）
  
  3. 决策逻辑：
     ┌─────────────────────────────────────────┐
     │ 文档已存在？                            │
     └──────────────┬──────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
      是│                       │否
        │                       │
        ▼                       ▼
   ┌─────────────┐         ┌─────────────┐
   │ 状态检查    │         │ 新文档      │
   └──────┬──────┘         │ new_count++ │
          │                └─────────────┘
     ┌────┴────┐
     │         │
  processed?  failed?
     │         │
     │    ┌────┴────┐
     │    │         │
     │ skip_failed? force_update?
     │    │         │
     │    │    ┌────┴────┐
     │    │    │         │
     │    │  是│         │否
     │    │    │         │
     │    │    ▼         ▼
     │    │ 跳过      重新处理
     │    │            reprocess_count++
     │    │
     │    ▼
     │ 跳过
     │ skipped_failed_count++
     │
     ▼
  跳过
  skipped_count++
```

**状态处理规则**：

| 文档状态 | force_update=False | force_update=True | skip_failed=True |
|---------|-------------------|-------------------|------------------|
| `processed` | 跳过 | 重新处理 | 跳过 |
| `failed` | 重新处理 | 重新处理 | 跳过 |
| `pending` | 重新处理 | 重新处理 | 重新处理 |
| `processing` | 重新处理 | 重新处理 | 重新处理 |
| 不存在 | 插入 | 插入 | 插入 |

#### 4.2 统计信息输出
```
文档检查结果:
  - 新文档: <数量>
  - 需要重新处理: <数量>
  - 已跳过 (已处理): <数量>
  - 已跳过 (失败): <数量>  (如果 skip_failed=True)
  - 待插入总数: <数量>
```

### 5. 文档插入阶段

#### 5.1 批量插入（优先）
```
尝试：
  rag.insert(
    input=texts,          # 文本列表
    file_paths=file_paths # 文件路径列表
  )
  
成功：
  - 返回 track_id
  - 记录插入任务已提交
  - 提示：插入是异步进行的
  
失败：
  - 记录错误
  - 降级到逐个插入模式
```

#### 5.2 逐个插入（降级方案）
```
如果批量插入失败：
  for 每个文档:
    try:
      rag.insert(input=text, file_paths=file_path)
      if idx % 10 == 0:
        记录进度
    except:
      记录错误，继续下一个
```

**注意**：`rag.insert()` 内部会：
1. 调用 `apipeline_enqueue_documents()` 将文档加入队列
2. 调用 `apipeline_process_enqueue_documents()` 开始处理

### 6. 进度监控阶段（可选）

#### 6.1 启用进度监控 (`--show-progress`)
```
如果启用：
  1. 等待 2 秒（让处理开始）
  2. 调用 show_processing_status()
  
监控循环：
  while True:
    1. 获取 pipeline_status
    2. 获取各状态文档数量：
       - PENDING（待处理）
       - PROCESSING（处理中）
       - PROCESSED（已完成）
       - FAILED（失败）
    
    3. 显示状态信息：
       - 处理状态（进行中/空闲）
       - 任务名称
       - 批次进度（如果有）
       - 文档统计
       - 最新消息
    
    4. 检查完成条件：
       - pipeline 不忙
       - 没有 PENDING 文档
       - 没有 PROCESSING 文档
    
    5. 等待 5 秒后继续检查
```

#### 6.2 完成判断
```
完成条件：
  - pipeline_status["busy"] == False
  - len(pending_docs) == 0
  - len(processing_docs) == 0

完成时输出：
  - 成功处理: <数量> 个文档
  - 失败: <数量> 个文档（如果有）
```

## 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--news-file` | str | `./demo1202/F1QA/news/news_example.jsonl` | 新闻 JSONL 文件路径 |
| `--show-progress` | flag | False | 显示处理进度 |
| `--force-update` | flag | False | 强制重新插入所有文档（即使已处理） |
| `--skip-failed` | flag | False | 跳过失败的文档（不重新处理） |

## 使用示例

### 基本用法（增量更新）
```bash
python demo1202/f1qa_prepare.py --news-file ./demo1202/F1QA/news/f1_news.jsonl
```
- 只插入新文档
- 重新处理未完成的文档（pending, processing, failed）
- 跳过已处理的文档

### 显示处理进度
```bash
python demo1202/f1qa_prepare.py \
  --news-file ./demo1202/F1QA/news/f1_news.jsonl \
  --show-progress
```
- 实时显示处理进度
- 等待所有文档处理完成

### 跳过失败的文档
```bash
python demo1202/f1qa_prepare.py \
  --news-file ./demo1202/F1QA/news/f1_news.jsonl \
  --skip-failed
```
- 不重新处理失败的文档
- 只处理新文档和未完成的文档

### 强制更新所有文档
```bash
python demo1202/f1qa_prepare.py \
  --news-file ./demo1202/F1QA/news/f1_news.jsonl \
  --force-update
```
- 重新处理所有文档（包括已处理的）

### 组合使用
```bash
python demo1202/f1qa_prepare.py \
  --news-file ./demo1202/F1QA/news/f1_news.jsonl \
  --show-progress \
  --skip-failed
```
- 跳过失败的文档
- 显示处理进度

## 文档状态说明

LightRAG 中的文档有以下状态：

| 状态 | 说明 | 处理行为 |
|------|------|----------|
| `PENDING` | 等待处理 | 会被处理 |
| `PROCESSING` | 正在处理 | 如果异常中断，会被重置为 PENDING 并重新处理 |
| `PREPROCESSED` | 预处理完成 | 会被处理（完成最终处理） |
| `PROCESSED` | 处理完成 | 默认跳过（除非 `--force-update`） |
| `FAILED` | 处理失败 | 默认重新处理（除非 `--skip-failed`） |

## 关键设计决策

### 1. 为什么使用 URL 作为文件路径标识符？
- URL 是新闻的唯一标识符
- 即使内容相同，不同 URL 的新闻应该被视为不同文档
- 便于追踪文档来源

### 2. 为什么支持增量更新？
- 避免重复处理已完成的文档，节省计算资源
- 支持断点续传，处理中断后可以继续
- 提高处理效率

### 3. 为什么提供 `--skip-failed` 选项？
- 某些文档可能因为数据问题永久失败
- 避免反复尝试处理这些文档
- 提高处理效率

### 4. 为什么处理是异步的？
- 文档处理需要时间（LLM 调用、嵌入计算等）
- 异步处理可以并发处理多个文档
- 提高整体吞吐量

## 注意事项

1. **处理时间**：文档处理是异步的，可能需要较长时间才能完成
2. **内存占用**：加载嵌入模型会占用 GPU 显存或系统内存
3. **日志文件**：每次运行都会生成新的日志文件，注意磁盘空间
4. **工作目录**：确保 `./cache/f1qa_demo` 目录有足够的磁盘空间
5. **模型路径**：确保嵌入模型路径正确（`/home/infres/zzhu-24/large_files/bge-m3`）

## 故障排查

### 问题：文档没有被标记为 PROCESSED
**可能原因**：
1. 处理被中断（Ctrl+C、程序崩溃）
2. 处理过程中出现错误
3. 脚本在处理完成前退出

**解决方案**：
- 使用 `--show-progress` 等待处理完成
- 检查日志文件中的错误信息
- 重新运行脚本（会自动处理未完成的文档）

### 问题：大量文档被重新处理
**可能原因**：
1. 之前处理被中断，文档停留在 PROCESSING 状态
2. 处理失败，文档被标记为 FAILED

**解决方案**：
- 使用 `--skip-failed` 跳过失败的文档
- 检查文档状态（使用 `f1qa_status.py`）
- 手动清理或重置文档状态

### 问题：内存不足
**可能原因**：
1. 批量插入的文档太多
2. 嵌入模型占用过多显存

**解决方案**：
- 减少批量大小
- 使用 CPU 模式（如果 GPU 显存不足）
- 分批处理文档

## 相关脚本

- `f1qa_query.py`：查询脚本，用于问答
- `f1qa_lightrag.py`：完整流程脚本（插入+查询）
- `f1qa_status.py`：文档状态查看脚本

## 版本历史

- **v1.0**：初始版本，支持基本插入功能
- **v2.0**：添加增量更新支持
- **v2.1**：添加 `--skip-failed` 选项
- **v2.2**：改进日志输出和错误处理
