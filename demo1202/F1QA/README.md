# F1 新闻抓取工具

`fetch_f1_news.py` 是一个用于从 Formula1.com 官网抓取 F1 新闻文章的 Python 脚本。该工具能够自动提取文章内容，并按周分组保存为 JSONL 格式文件。

## 功能特性

- **分页抓取**：支持指定页码范围，批量抓取新闻列表
- **智能解析**：优先使用 JSON-LD 结构化数据，回退到 HTML 解析
- **按周分组**：自动根据文章发布时间将新闻按周分组保存
- **去重处理**：基于 URL 自动去重，避免重复抓取
- **多格式支持**：支持多种日期时间格式的解析
- **容错机制**：单个页面或文章失败不影响整体抓取流程

## 依赖要求

```bash
pip install requests beautifulsoup4 tqdm
```

可选依赖（用于更灵活的日期解析）：
```bash
pip install python-dateutil
```

## 使用方法

### 基本用法

```bash
python fetch_f1_news.py
```

默认参数：
- 起始页码：1
- 结束页码：5
- 请求间隔：1.2 秒
- 输出目录：`news/`

### 自定义参数

```bash
python fetch_f1_news.py \
    --start 1 \
    --end 10 \
    --sleep 1.5 \
    --out-dir news
```

### 参数说明

- `--start`: 起始页码（1-100），默认为 1
- `--end`: 结束页码（1-100），默认为 5
- `--sleep`: 每次请求之间的休眠时间（秒），默认为 1.2，建议设置 1.0-2.0 秒以避免对服务器造成压力
- `--out-dir`: 输出目录路径，默认为 `news/`

## 输出格式

脚本会在指定的输出目录下创建按周分组的 JSONL 文件，文件名格式为：
```
f1_news_YYYY-MM-DD.jsonl
```
其中 `YYYY-MM-DD` 为该周周一的日期。

每个 JSONL 文件包含多行 JSON 对象，每行代表一篇新闻文章，格式如下：

```json
{
  "title": "文章标题",
  "summary": "文章摘要",
  "published_time": "2024-01-15T10:30:00+00:00",
  "url": "https://www.formula1.com/en/latest/article/...",
  "content": "文章正文内容..."
}
```

### 字段说明

- `title`: 文章标题
- `summary`: 文章摘要（可能为空）
- `published_time`: 发布时间（ISO 8601 格式）
- `url`: 文章原始 URL
- `content`: 文章正文内容

## 工作原理

1. **列表页抓取**：遍历指定范围内的列表页（`/en/latest?page=X`），提取所有文章链接
2. **URL 去重**：对收集到的 URL 进行去重处理
3. **文章解析**：
   - 优先从 JSON-LD 结构化数据中提取信息
   - 回退到 HTML meta 标签和 DOM 解析
   - 使用智能文本提取算法获取正文内容
4. **按周分组**：根据文章发布时间计算所属周（周一为周起始日），将文章分组
5. **文件保存**：按周将文章保存到对应的 JSONL 文件中，并按发布时间排序

## 注意事项

1. **请求频率**：请合理设置 `--sleep` 参数，避免对目标网站造成过大压力。建议至少 1.0 秒间隔
2. **网络稳定性**：确保网络连接稳定，脚本会跳过失败的页面或文章，但不会中断整体流程
3. **日期解析**：如果文章没有明确的发布时间，脚本会使用当前时间作为默认值
4. **内容质量**：脚本会过滤掉标题或内容为空的文章
5. **页码范围**：建议单次抓取不超过 100 页，如需更多数据可分多次运行

## 示例

### 抓取前 10 页新闻

```bash
python fetch_f1_news.py --start 1 --end 10 --out-dir news
```

### 抓取指定范围并设置较长延迟

```bash
python fetch_f1_news.py --start 5 --end 20 --sleep 2.0 --out-dir f1_news_data
```

## 输出示例

运行完成后，控制台会显示类似以下信息：

```
Collecting list pages: 100%|████████████| 10/10 [00:15<00:00,  1.52s/it]
Scraping articles: 100%|████████████| 150/150 [03:45<00:00,  1.50s/it]

[INFO] Writing 3 week files...
[INFO] Wrote 45 articles to news/f1_news_2024-01-08.jsonl (week starting 2024-01-08)
[INFO] Wrote 52 articles to news/f1_news_2024-01-15.jsonl (week starting 2024-01-15)
[INFO] Wrote 38 articles to news/f1_news_2024-01-22.jsonl (week starting 2024-01-22)
[INFO] Total: 135 unique articles across 3 weeks
```

## 故障排除

### 常见问题

1. **连接超时**：检查网络连接，或增加 `--sleep` 参数值
2. **解析失败**：某些文章可能因页面结构变化导致解析失败，这是正常现象，脚本会跳过并继续
3. **日期解析错误**：如果安装了 `python-dateutil`，日期解析会更加灵活

### 调试建议

- 先使用较小的页码范围（如 `--start 1 --end 2`）测试
- 检查输出目录中的文件，确认数据格式正确
- 查看控制台警告信息，了解哪些页面或文章处理失败

## 相关文件

- `news/`: 默认输出目录，包含按周分组的 JSONL 文件
- `sources.json`: F1 相关新闻源配置（供参考）
