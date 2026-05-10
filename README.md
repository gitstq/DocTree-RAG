<p align="center">
  <h1 align="center">🌳 DocTree-RAG</h1>
  <p align="center">
    <strong>Lightweight Document Tree Indexing & Reasoning-Based Retrieval Engine</strong><br>
    轻量级文档树索引与推理检索引擎
  </p>
  <p align="center">
    <a href="#-简体中文">简体中文</a> ·
    <a href="#-繁體中文">繁體中文</a> ·
    <a href="#-english">English</a>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python 3.8+">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License">
    <img src="https://img.shields.io/badge/Tests-82%20Passed-success.svg" alt="Tests">
    <img src="https://img.shields.io/badge/Zero_Dependencies-Core%20Features-orange.svg" alt="Zero Dependencies">
  </p>
</p>

---

<a id="简体中文"></a>

## 🎉 项目介绍

**DocTree-RAG** 是一款轻量级的文档树索引与推理检索引擎，专为长文档的智能检索而设计。传统基于向量数据库的 RAG 系统依赖语义**相似度**进行检索，但**相似 ≠ 相关**——真正的检索需要**推理**能力。

DocTree-RAG 的灵感来源于 GitHub Trending 热门项目 [PageIndex](https://github.com/VectifyAI/PageIndex) 的推理式 RAG 理念，但在此基础上进行了全面的差异化重构：

- 🌐 **多 LLM 后端**：不局限于单一提供商，支持 OpenAI、DeepSeek、Anthropic Claude、Ollama 本地模型
- 📄 **多格式文档**：支持 PDF、Markdown、TXT、HTML 四种主流格式
- 🔍 **混合检索引擎**：LLM 推理检索 + BM25 关键词搜索双引擎协作
- 📦 **零强制依赖**：核心功能（树模型、关键词搜索、文档解析）仅使用 Python 标准库
- 🖥️ **交互式 TUI**：可视化树结构浏览与实时搜索
- 🔄 **增量索引**：基于内容哈希检测文档变更，避免重复处理

### 解决的核心痛点

| 痛点 | DocTree-RAG 的解决方案 |
|------|----------------------|
| 向量 RAG 对专业长文档检索精度不足 | 基于推理的树状检索，理解文档逻辑结构 |
| 传统 RAG 的固定分块破坏文档语义 | 按文档自然结构构建层级树索引 |
| 依赖向量数据库，部署成本高 | 零外部依赖，纯本地运行 |
| 仅支持单一 LLM 提供商 | 多后端统一接口，自由切换 |
| 大型文档更新需全量重建索引 | 增量索引，仅处理变更部分 |

---

## ✨ 核心特性

### 🌳 层级树结构索引
- 自动识别文档的章节、标题层级结构
- 每个节点包含标题、摘要、页码范围、内容哈希
- 支持自定义最大树深度和节点大小

### 🧠 LLM 推理检索
- 利用大语言模型**推理**查询与文档节点的相关性
- 输出相关节点列表及推理过程说明
- 支持批量查询和温度参数调节

### 🔎 BM25 关键词搜索
- 基于 TF-IDF 的快速本地关键词匹配
- 支持 **AND**、**OR**、**NOT** 布尔运算符
- 支持短语查询（用引号包裹）
- 中英文分词支持

### 🤖 多 LLM 后端支持

| 提供商 | 支持模型 | 配置方式 |
|--------|---------|---------|
| **OpenAI** | GPT-4o, GPT-4o-mini | `OPENAI_API_KEY` |
| **DeepSeek** | deepseek-chat, deepseek-reasoner | `DEEPSEEK_API_KEY` |
| **Anthropic** | Claude 3.5 Sonnet, Claude 3 Haiku | `ANTHROPIC_API_KEY` |
| **Ollama** | llama3, qwen2, mistral 等 | `OLLAMA_BASE_URL`（默认 localhost） |

### 🖥️ 交互式 TUI 浏览器
- 可展开/折叠的树状视图
- 实时搜索过滤
- 节点详情面板（摘要、页码范围）
- 键盘导航（方向键、Enter 展开、/ 搜索、q 退出）
- rich 不可用时自动降级为纯文本模式

### 📤 多格式导出
- **JSON**：完整树结构数据
- **Markdown**：层级大纲与摘要
- **HTML**：可折叠的交互式树形页面

---

## 🚀 快速开始

### 环境要求

- **Python** 3.8 或更高版本
- （可选）用于 PDF 解析：`pip install PyPDF2`
- （可选）用于增强 TUI：`pip install rich`
- （可选）用于 LLM 功能：配置任一 LLM 提供商的 API Key

### 安装

```bash
# 克隆仓库
git clone https://github.com/gitstq/DocTree-RAG.git
cd DocTree-RAG

# 安装（核心功能零依赖）
pip install -e .

# 安装所有可选依赖
pip install -r requirements.txt
```

### 基本使用

```bash
# 1. 为文档构建树索引
doctree index document.pdf

# 2. 为整个目录构建索引（递归）
doctree index ./documents/

# 3. 跳过 LLM 摘要生成（纯关键词模式）
doctree index document.pdf --no-summary

# 4. 搜索已索引的文档
doctree search "机器学习基础概念"

# 5. 使用布尔运算符搜索
doctree search "深度学习 AND 卷积神经网络"

# 6. 交互式浏览文档树
doctree browse

# 7. 导出为 Markdown
doctree export --format markdown --output tree.md

# 8. 导出为 HTML
doctree export --format html --output tree.html

# 9. 列出所有已索引文档
doctree list

# 10. 查看文档树详细信息
doctree info <doc_id>
```

### LLM 配置

```bash
# 配置 OpenAI
doctree config --provider openai --model gpt-4o-mini

# 配置 DeepSeek
doctree config --provider deepseek --model deepseek-chat

# 配置 Anthropic Claude
doctree config --provider anthropic --model claude-3-5-sonnet-20241022

# 配置 Ollama 本地模型
doctree config --provider ollama --model qwen2

# 也可通过环境变量配置
export OPENAI_API_KEY="your-api-key"
export DEEPSEEK_API_KEY="your-api-key"
export ANTHROPIC_API_KEY="your-api-key"
```

---

## 📖 详细使用指南

### 索引命令详解

```bash
doctree index <file_or_dir> [选项]
```

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `--no-summary` | 跳过 LLM 摘要生成 | 关闭 |
| `--no-recursive` | 不递归处理子目录 | 关闭 |
| `--max-depth` | 最大树深度 | 无限制 |
| `--output` | 索引输出目录 | `.doctree/` |

### 搜索命令详解

```bash
doctree search <query> [选项]
```

搜索支持以下高级语法：
- `关键词1 AND 关键词2` — 同时包含两个词
- `关键词1 OR 关键词2` — 包含任一词
- `NOT 关键词` — 排除包含该词的结果
- `"精确短语"` — 精确短语匹配

### 导出格式说明

**JSON 格式**示例：
```json
{
  "doc_id": "abc123",
  "doc_name": "report.pdf",
  "created_at": "2026-05-10T12:00:00",
  "root_nodes": [
    {
      "node_id": "0001",
      "title": "第一章 概述",
      "summary": "本文档介绍了...",
      "start_page": 1,
      "end_page": 5,
      "children": [...]
    }
  ]
}
```

---

## 💡 设计思路与迭代规划

### 设计理念

DocTree-RAG 的核心理念是**让文档检索回归推理**。传统向量 RAG 将文档切碎后计算语义相似度，这种方式在处理专业长文档时往往表现不佳——因为"语义相似"不等于"逻辑相关"。

我们的方案是：**保持文档完整性**，按照文档的自然层级结构构建树状索引，然后让 LLM 通过推理来定位最相关的章节。这种方式更接近人类阅读文档时的思考过程。

### 技术选型原因

| 选择 | 原因 |
|------|------|
| Python | 生态丰富，AI/文档处理库最成熟 |
| 纯标准库核心 | 降低使用门槛，零依赖即可运行 |
| urllib 而非 SDK | 避免版本冲突，减少依赖体积 |
| BM25 而非向量 | 无需嵌入模型，纯本地计算，速度快 |
| rich（可选） | Python 生态最优秀的终端 UI 库 |

### 后续迭代计划

- [ ] **v0.2** — 支持 DOCX/EPUB 格式
- [ ] **v0.3** — Web UI 界面
- [ ] **v0.4** — 混合检索（推理 + 向量）
- [ ] **v0.5** — 多文档联合检索与交叉引用
- [ ] **v1.0** — 插件系统与 MCP 协议支持

---

## 📦 安装与部署

### 从源码安装

```bash
git clone https://github.com/gitstq/DocTree-RAG.git
cd DocTree-RAG
pip install -e .
```

### 作为库使用

```python
from doctree_rag.indexer import DocumentIndexer
from doctree_rag.retriever import ReasoningRetriever
from doctree_rag.keyword_search import KeywordSearchEngine

# 构建索引
indexer = DocumentIndexer(output_dir=".doctree")
tree = indexer.index_file("report.pdf", generate_summary=False)

# 关键词搜索
engine = KeywordSearchEngine()
results = engine.search("机器学习", tree)

# LLM 推理检索
retriever = ReasoningRetriever(provider="openai", model="gpt-4o-mini")
nodes = retriever.retrieve("文档中关于风险管理的论述", tree)
```

### 兼容环境

| 环境 | 最低版本 |
|------|---------|
| Python | 3.8+ |
| 操作系统 | Windows / macOS / Linux |
| 终端 | 任何支持 UTF-8 的终端 |

---

## 🤝 贡献指南

我们欢迎所有形式的贡献！请遵循以下规范：

### 提交 PR 流程

1. Fork 本仓库
2. 创建功能分支：`git checkout -b feat/my-feature`
3. 编写代码和测试
4. 确保测试通过：`python -m unittest discover tests/ -v`
5. 提交：`git commit -m "feat: 添加xxx功能"`
6. 推送：`git push origin feat/my-feature`
7. 发起 Pull Request

### 提交信息规范

遵循 Angular 提交规范：
- `feat:` 新功能
- `fix:` 修复问题
- `docs:` 文档更新
- `refactor:` 代码重构
- `test:` 测试相关
- `chore:` 构建/工具变更

### Issue 反馈

请使用 [GitHub Issues](https://github.com/gitstq/DocTree-RAG/issues) 提交 Bug 报告或功能建议，并附上：
- 问题描述
- 复现步骤
- 期望行为
- 实际行为
- 环境信息（Python 版本、操作系统等）

---

## 📄 开源协议

本项目基于 [MIT License](https://opensource.org/licenses/MIT) 开源。

```
MIT License

Copyright (c) 2026 DocTree-RAG Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

<a id="繁體中文"></a>

## 🎉 專案介紹

**DocTree-RAG** 是一款輕量級的文件樹索引與推理檢索引擎，專為長文件的智慧檢索而設計。傳統基於向量資料庫的 RAG 系統依賴語義**相似度**進行檢索，但**相似 ≠ 相關**——真正的檢索需要**推理**能力。

DocTree-RAG 的靈感來源於 GitHub Trending 熱門專案 [PageIndex](https://github.com/VectifyAI/PageIndex) 的推理式 RAG 理念，但在此基礎上進行了全面的差異化重構：

- 🌐 **多 LLM 後端**：不限於單一提供商，支援 OpenAI、DeepSeek、Anthropic Claude、Ollama 本地模型
- 📄 **多格式文件**：支援 PDF、Markdown、TXT、HTML 四種主流格式
- 🔍 **混合檢索引擎**：LLM 推理檢索 + BM25 關鍵字搜尋雙引擎協作
- 📦 **零強制依賴**：核心功能（樹模型、關鍵字搜尋、文件解析）僅使用 Python 標準庫
- 🖥️ **互動式 TUI**：視覺化樹結構瀏覽與即時搜尋
- 🔄 **增量索引**：基於內容雜湊偵測文件變更，避免重複處理

### 解決的核心痛點

| 痛點 | DocTree-RAG 的解決方案 |
|------|----------------------|
| 向量 RAG 對專業長文件檢索精度不足 | 基於推理的樹狀檢索，理解文件邏輯結構 |
| 傳統 RAG 的固定分塊破壞文件語義 | 按文件自然結構構建層級樹索引 |
| 依賴向量資料庫，部署成本高 | 零外部依賴，純本地運行 |
| 僅支援單一 LLM 提供商 | 多後端統一介面，自由切換 |
| 大型文件更新需全量重建索引 | 增量索引，僅處理變更部分 |

---

## ✨ 核心特性

### 🌳 層級樹結構索引
- 自動識別文件的章節、標題層級結構
- 每個節點包含標題、摘要、頁碼範圍、內容雜湊
- 支援自訂最大樹深度和節點大小

### 🧠 LLM 推理檢索
- 利用大語言模型**推理**查詢與文件節點的相關性
- 輸出相關節點列表及推理過程說明
- 支援批次查詢和溫度參數調節

### 🔎 BM25 關鍵字搜尋
- 基於 TF-IDF 的快速本地關鍵字匹配
- 支援 **AND**、**OR**、**NOT** 布林運算子
- 支援片語查詢（用引號包裹）
- 中英文分詞支援

### 🤖 多 LLM 後端支援

| 提供商 | 支援模型 | 設定方式 |
|--------|---------|---------|
| **OpenAI** | GPT-4o, GPT-4o-mini | `OPENAI_API_KEY` |
| **DeepSeek** | deepseek-chat, deepseek-reasoner | `DEEPSEEK_API_KEY` |
| **Anthropic** | Claude 3.5 Sonnet, Claude 3 Haiku | `ANTHROPIC_API_KEY` |
| **Ollama** | llama3, qwen2, mistral 等 | `OLLAMA_BASE_URL`（預設 localhost） |

### 🖥️ 互動式 TUI 瀏覽器
- 可展開/折疊的樹狀檢視
- 即時搜尋過濾
- 節點詳情面板（摘要、頁碼範圍）
- 鍵盤導航（方向鍵、Enter 展開、/ 搜尋、q 退出）
- rich 不可用時自動降級為純文字模式

### 📤 多格式匯出
- **JSON**：完整樹結構資料
- **Markdown**：層級大綱與摘要
- **HTML**：可折疊的互動式樹形頁面

---

## 🚀 快速開始

### 環境要求

- **Python** 3.8 或更高版本
- （可選）用於 PDF 解析：`pip install PyPDF2`
- （可選）用於增強 TUI：`pip install rich`
- （可選）用於 LLM 功能：設定任一 LLM 提供商的 API Key

### 安裝

```bash
# 克隆倉庫
git clone https://github.com/gitstq/DocTree-RAG.git
cd DocTree-RAG

# 安裝（核心功能零依賴）
pip install -e .

# 安裝所有可選依賴
pip install -r requirements.txt
```

### 基本使用

```bash
# 1. 為文件構建樹索引
doctree index document.pdf

# 2. 為整個目錄構建索引（遞迴）
doctree index ./documents/

# 3. 跳過 LLM 摘要生成（純關鍵字模式）
doctree index document.pdf --no-summary

# 4. 搜尋已索引的文件
doctree search "機器學習基礎概念"

# 5. 使用布林運算子搜尋
doctree search "深度學習 AND 卷積神經網路"

# 6. 互動式瀏覽文件樹
doctree browse

# 7. 匯出為 Markdown
doctree export --format markdown --output tree.md

# 8. 匯出為 HTML
doctree export --format html --output tree.html

# 9. 列出所有已索引文件
doctree list

# 10. 查看文件樹詳細資訊
doctree info <doc_id>
```

### LLM 設定

```bash
# 設定 OpenAI
doctree config --provider openai --model gpt-4o-mini

# 設定 DeepSeek
doctree config --provider deepseek --model deepseek-chat

# 設定 Anthropic Claude
doctree config --provider anthropic --model claude-3-5-sonnet-20241022

# 設定 Ollama 本地模型
doctree config --provider ollama --model qwen2

# 也可透過環境變數設定
export OPENAI_API_KEY="your-api-key"
export DEEPSEEK_API_KEY="your-api-key"
export ANTHROPIC_API_KEY="your-api-key"
```

---

## 📖 詳細使用指南

### 索引命令詳解

```bash
doctree index <file_or_dir> [選項]
```

| 選項 | 說明 | 預設值 |
|------|------|--------|
| `--no-summary` | 跳過 LLM 摘要生成 | 關閉 |
| `--no-recursive` | 不遞迴處理子目錄 | 關閉 |
| `--max-depth` | 最大樹深度 | 無限制 |
| `--output` | 索引輸出目錄 | `.doctree/` |

### 搜尋命令詳解

```bash
doctree search <query> [選項]
```

搜尋支援以下進階語法：
- `關鍵字1 AND 關鍵字2` — 同時包含兩個詞
- `關鍵字1 OR 關鍵字2` — 包含任一詞
- `NOT 關鍵字` — 排除包含該詞的結果
- `"精確片語"` — 精確片語匹配

---

## 💡 設計思路與迭代規劃

### 設計理念

DocTree-RAG 的核心理念是**讓文件檢索回歸推理**。傳統向量 RAG 將文件切碎後計算語義相似度，這種方式在處理專業長文件時往往表現不佳——因為「語義相似」不等於「邏輯相關」。

我們的方案是：**保持文件完整性**，按照文件的自然層級結構構建樹狀索引，然後讓 LLM 透過推理來定位最相關的章節。這種方式更接近人類閱讀文件時的思考過程。

### 後續迭代計劃

- [ ] **v0.2** — 支援 DOCX/EPUB 格式
- [ ] **v0.3** — Web UI 介面
- [ ] **v0.4** — 混合檢索（推理 + 向量）
- [ ] **v0.5** — 多文件聯合檢索與交叉引用
- [ ] **v1.0** — 外掛系統與 MCP 協議支援

---

## 📦 安裝與部署

### 從原始碼安裝

```bash
git clone https://github.com/gitstq/DocTree-RAG.git
cd DocTree-RAG
pip install -e .
```

### 作為庫使用

```python
from doctree_rag.indexer import DocumentIndexer
from doctree_rag.retriever import ReasoningRetriever
from doctree_rag.keyword_search import KeywordSearchEngine

# 構建索引
indexer = DocumentIndexer(output_dir=".doctree")
tree = indexer.index_file("report.pdf", generate_summary=False)

# 關鍵字搜尋
engine = KeywordSearchEngine()
results = engine.search("機器學習", tree)

# LLM 推理檢索
retriever = ReasoningRetriever(provider="openai", model="gpt-4o-mini")
nodes = retriever.retrieve("文件中關於風險管理的論述", tree)
```

### 相容環境

| 環境 | 最低版本 |
|------|---------|
| Python | 3.8+ |
| 作業系統 | Windows / macOS / Linux |
| 終端 | 任何支援 UTF-8 的終端 |

---

## 🤝 貢獻指南

我們歡迎所有形式的貢獻！請遵循以下規範：

### 提交 PR 流程

1. Fork 本倉庫
2. 建立功能分支：`git checkout -b feat/my-feature`
3. 編寫程式碼和測試
4. 確保測試通過：`python -m unittest discover tests/ -v`
5. 提交：`git commit -m "feat: 新增xxx功能"`
6. 推送：`git push origin feat/my-feature`
7. 發起 Pull Request

### 提交資訊規範

遵循 Angular 提交規範：
- `feat:` 新功能
- `fix:` 修復問題
- `docs:` 文件更新
- `refactor:` 程式碼重構
- `test:` 測試相關
- `chore:` 建構/工具變更

---

## 📄 開源協議

本專案基於 [MIT License](https://opensource.org/licenses/MIT) 開源。

---

<a id="english"></a>

## 🎉 Introduction

**DocTree-RAG** is a lightweight document tree indexing and reasoning-based retrieval engine designed for intelligent retrieval from long documents. Traditional vector-based RAG systems rely on semantic **similarity** for retrieval, but **similarity ≠ relevance** — true retrieval requires **reasoning** capabilities.

DocTree-RAG is inspired by the reasoning-based RAG concept from the GitHub Trending project [PageIndex](https://github.com/VectifyAI/PageIndex), but has been completely re-architected with significant differentiations:

- 🌐 **Multi-LLM Backend**: Not limited to a single provider — supports OpenAI, DeepSeek, Anthropic Claude, and Ollama local models
- 📄 **Multi-Format Documents**: Supports PDF, Markdown, TXT, and HTML
- 🔍 **Hybrid Retrieval Engine**: LLM reasoning retrieval + BM25 keyword search dual-engine collaboration
- 📦 **Zero Mandatory Dependencies**: Core features (tree model, keyword search, document parsing) use only Python standard library
- 🖥️ **Interactive TUI**: Visual tree structure browsing with real-time search
- 🔄 **Incremental Indexing**: Detects document changes via content hashing, avoiding reprocessing

### Core Pain Points Solved

| Pain Point | DocTree-RAG Solution |
|------------|---------------------|
| Vector RAG has low precision for professional long documents | Reasoning-based tree retrieval that understands document logic |
| Traditional RAG's fixed chunking destroys document semantics | Builds hierarchical tree index following natural document structure |
| Dependency on vector databases increases deployment costs | Zero external dependencies, runs entirely locally |
| Only supports a single LLM provider | Multi-backend unified interface, freely switchable |
| Large document updates require full index rebuild | Incremental indexing, only processes changed parts |

---

## ✨ Core Features

### 🌳 Hierarchical Tree Structure Indexing
- Automatically identifies document chapter and heading hierarchy
- Each node contains title, summary, page range, and content hash
- Configurable maximum tree depth and node size

### 🧠 LLM Reasoning Retrieval
- Leverages LLMs to **reason** about query-node relevance
- Outputs relevant node list with reasoning explanations
- Supports batch queries and temperature parameter tuning

### 🔎 BM25 Keyword Search
- Fast local keyword matching based on TF-IDF scoring
- Supports **AND**, **OR**, **NOT** boolean operators
- Supports phrase queries (wrap in quotes)
- Chinese and English tokenization support

### 🤖 Multi-LLM Backend Support

| Provider | Supported Models | Configuration |
|----------|-----------------|---------------|
| **OpenAI** | GPT-4o, GPT-4o-mini | `OPENAI_API_KEY` |
| **DeepSeek** | deepseek-chat, deepseek-reasoner | `DEEPSEEK_API_KEY` |
| **Anthropic** | Claude 3.5 Sonnet, Claude 3 Haiku | `ANTHROPIC_API_KEY` |
| **Ollama** | llama3, qwen2, mistral, etc. | `OLLAMA_BASE_URL` (default: localhost) |

### 🖥️ Interactive TUI Browser
- Expandable/collapsible tree view
- Real-time search filtering
- Node detail panel (summary, page range)
- Keyboard navigation (arrow keys, Enter to expand, / to search, q to quit)
- Automatic fallback to plain text mode when rich is unavailable

### 📤 Multi-Format Export
- **JSON**: Complete tree structure data
- **Markdown**: Hierarchical outline with summaries
- **HTML**: Interactive collapsible tree page

---

## 🚀 Quick Start

### Requirements

- **Python** 3.8 or higher
- (Optional) For PDF parsing: `pip install PyPDF2`
- (Optional) For enhanced TUI: `pip install rich`
- (Optional) For LLM features: Configure any LLM provider's API key

### Installation

```bash
# Clone the repository
git clone https://github.com/gitstq/DocTree-RAG.git
cd DocTree-RAG

# Install (zero dependencies for core features)
pip install -e .

# Install all optional dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# 1. Build a tree index for a document
doctree index document.pdf

# 2. Build index for an entire directory (recursive)
doctree index ./documents/

# 3. Skip LLM summary generation (keyword-only mode)
doctree index document.pdf --no-summary

# 4. Search across indexed documents
doctree search "machine learning fundamentals"

# 5. Search with boolean operators
doctree search "deep learning AND convolutional neural networks"

# 6. Interactive document tree browser
doctree browse

# 7. Export as Markdown
doctree export --format markdown --output tree.md

# 8. Export as HTML
doctree export --format html --output tree.html

# 9. List all indexed documents
doctree list

# 10. View document tree details
doctree info <doc_id>
```

### LLM Configuration

```bash
# Configure OpenAI
doctree config --provider openai --model gpt-4o-mini

# Configure DeepSeek
doctree config --provider deepseek --model deepseek-chat

# Configure Anthropic Claude
doctree config --provider anthropic --model claude-3-5-sonnet-20241022

# Configure Ollama local model
doctree config --provider ollama --model qwen2

# Or via environment variables
export OPENAI_API_KEY="your-api-key"
export DEEPSEEK_API_KEY="your-api-key"
export ANTHROPIC_API_KEY="your-api-key"
```

---

## 📖 Detailed Usage Guide

### Index Command Reference

```bash
doctree index <file_or_dir> [options]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--no-summary` | Skip LLM summary generation | Off |
| `--no-recursive` | Don't process subdirectories recursively | Off |
| `--max-depth` | Maximum tree depth | Unlimited |
| `--output` | Index output directory | `.doctree/` |

### Search Command Reference

```bash
doctree search <query> [options]
```

Advanced search syntax:
- `keyword1 AND keyword2` — Contains both keywords
- `keyword1 OR keyword2` — Contains either keyword
- `NOT keyword` — Excludes results containing the keyword
- `"exact phrase"` — Exact phrase matching

### Export Format Examples

**JSON format** example:
```json
{
  "doc_id": "abc123",
  "doc_name": "report.pdf",
  "created_at": "2026-05-10T12:00:00",
  "root_nodes": [
    {
      "node_id": "0001",
      "title": "Chapter 1 Overview",
      "summary": "This document describes...",
      "start_page": 1,
      "end_page": 5,
      "children": [...]
    }
  ]
}
```

---

## 💡 Design Philosophy & Roadmap

### Design Philosophy

DocTree-RAG's core philosophy is to **bring reasoning back to document retrieval**. Traditional vector RAG shreds documents into chunks and computes semantic similarity, which often performs poorly on professional long documents — because "semantically similar" doesn't mean "logically relevant".

Our approach: **preserve document integrity**, build hierarchical tree indexes following the document's natural structure, and let LLMs reason to locate the most relevant sections. This mirrors how humans actually read and navigate documents.

### Technology Choices

| Choice | Reason |
|--------|--------|
| Python | Richest ecosystem for AI/document processing |
| Pure stdlib core | Lower barrier to entry, zero dependencies to run |
| urllib over SDKs | Avoids version conflicts, reduces dependency size |
| BM25 over vectors | No embedding model needed, pure local computation, fast |
| rich (optional) | Best terminal UI library in Python ecosystem |

### Roadmap

- [ ] **v0.2** — DOCX/EPUB format support
- [ ] **v0.3** — Web UI interface
- [ ] **v0.4** — Hybrid retrieval (reasoning + vector)
- [ ] **v0.5** — Multi-document joint retrieval with cross-referencing
- [ ] **v1.0** — Plugin system & MCP protocol support

---

## 📦 Installation & Deployment

### Install from Source

```bash
git clone https://github.com/gitstq/DocTree-RAG.git
cd DocTree-RAG
pip install -e .
```

### Use as a Library

```python
from doctree_rag.indexer import DocumentIndexer
from doctree_rag.retriever import ReasoningRetriever
from doctree_rag.keyword_search import KeywordSearchEngine

# Build index
indexer = DocumentIndexer(output_dir=".doctree")
tree = indexer.index_file("report.pdf", generate_summary=False)

# Keyword search
engine = KeywordSearchEngine()
results = engine.search("machine learning", tree)

# LLM reasoning retrieval
retriever = ReasoningRetriever(provider="openai", model="gpt-4o-mini")
nodes = retriever.retrieve("risk management discussion in the document", tree)
```

### Compatible Environments

| Environment | Minimum Version |
|-------------|----------------|
| Python | 3.8+ |
| OS | Windows / macOS / Linux |
| Terminal | Any UTF-8 capable terminal |

---

## 🤝 Contributing

We welcome contributions of all forms! Please follow these guidelines:

### PR Submission Process

1. Fork this repository
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Write code and tests
4. Ensure tests pass: `python -m unittest discover tests/ -v`
5. Commit: `git commit -m "feat: add xxx feature"`
6. Push: `git push origin feat/my-feature`
7. Open a Pull Request

### Commit Message Convention

Follow the Angular commit convention:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation update
- `refactor:` Code refactoring
- `test:` Test-related changes
- `chore:` Build/tooling changes

### Issue Reporting

Please use [GitHub Issues](https://github.com/gitstq/DocTree-RAG/issues) to submit bug reports or feature requests, including:
- Problem description
- Steps to reproduce
- Expected behavior
- Actual behavior
- Environment info (Python version, OS, etc.)

---

## 📄 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

```
MIT License

Copyright (c) 2026 DocTree-RAG Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```
