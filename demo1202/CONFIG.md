# 配置说明

## 模型路径配置

项目中的脚本包含硬编码的模型路径，需要根据你的实际环境进行修改。

### 需要配置的路径

1. **BGE-M3 嵌入模型路径**
   - 默认路径: `/home/infres/zzhu-24/large_files/bge-m3`
   - 需要修改的文件:
     - `metaqa_lightrag.py` (第 75 行)
     - `f1qa_prepare.py` (第 150, 153 行)
     - `f1qa_query.py` (第 101, 104 行)
     - `f1qa_status.py` (第 59, 62 行)
     - `f1qa_lightrag.py` (第 178, 181 行)

2. **调试日志路径**（可选）
   - 默认路径: `/home/infres/zzhu-24/kg-rag/LightRAG/.cursor/debug.log`
   - 需要修改的文件:
     - `metaqa_lightrag.py` (第 20 行)

3. **GraphML 文件路径**（可选，用于查看工具）
   - 默认路径: `/home/infres/zzhu-24/kg-rag/LightRAG/cache/metaqa_demo/graph_chunk_entity_relation.graphml`
   - 需要修改的文件:
     - `print_graphml_example.py` (第 108 行)
     - `view_graphml_format.py` (第 201 行)

4. **实体缓存文件路径**（可选，用于查看工具）
   - 默认路径: `/home/infres/zzhu-24/kg-rag/LightRAG/cache/metaqa_demo/vdb_entities.json`
   - 需要修改的文件:
     - `view_entity_cache.py` (第 149 行)

### 配置方法

#### 方法 1: 直接修改脚本文件

在对应的脚本文件中找到模型路径配置，修改为你的实际路径：

```python
# 修改前
embed_model_name = "/home/infres/zzhu-24/large_files/bge-m3"

# 修改后（示例）
embed_model_name = "/path/to/your/bge-m3"
```

#### 方法 2: 使用环境变量（推荐，需要修改代码）

可以修改代码使用环境变量：

```python
import os
embed_model_name = os.getenv("BGE_M3_PATH", "/default/path/to/bge-m3")
```

然后在运行前设置环境变量：

```bash
export BGE_M3_PATH="/path/to/your/bge-m3"
python demo1202/metaqa_lightrag.py
```

## 其他配置

### LLM 模型

默认使用的 LLM 模型：
- MetaQA: `Qwen/Qwen2.5-1.5B-Instruct`
- F1QA: `Qwen/Qwen2.5-3B-Instruct`

这些模型会从 HuggingFace 自动下载，无需手动配置路径。

### 工作目录

默认工作目录：
- MetaQA: `./cache/metaqa_demo`
- F1QA: `./cache/f1qa_demo`

可以在脚本中修改 `WORKING_DIR` 变量来更改。

### GPU 配置

脚本会自动检测 GPU 是否可用。如果使用 CPU，性能会显著下降。

确保已安装正确版本的 CUDA 和 PyTorch：

```bash
# 检查 CUDA 版本
nvcc --version

# 检查 PyTorch CUDA 支持
python -c "import torch; print(torch.cuda.is_available())"
```

## 快速配置检查清单

- [ ] 下载 BGE-M3 模型并配置路径
- [ ] 确保有足够的 GPU 显存（推荐 16GB+）
- [ ] 安装所有依赖（`pip install -r requirements.txt`）
- [ ] 检查 CUDA 和 PyTorch 版本兼容性
- [ ] 准备数据文件（MetaQA 或 F1QA 数据）
