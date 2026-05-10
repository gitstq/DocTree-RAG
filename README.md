# DocTree-RAG

轻量级文档树索引与推理检索引擎。

## 功能特性

- 多格式文档解析：PDF、Markdown、TXT、HTML
- 层级树结构索引：自动识别文档章节结构
- LLM 推理检索：利用大语言模型进行语义检索
- BM25 关键词搜索：快速本地关键词匹配
- 多 LLM 后端：OpenAI、DeepSeek、Anthropic Claude、Ollama
- 交互式 TUI 浏览器：基于 rich 的终端界面
- 多格式导出：JSON、Markdown、HTML
- 增量索引：基于内容哈希检测变更

## 安装

```bash
pip install -e .
# 或安装所有可选依赖
pip install -e ".[all]"
```

## 快速开始

```bash
# 索引文档
doctree index document.pdf

# 搜索
doctree search "查询内容"

# 浏览
doctree browse

# 导出
doctree export --format markdown --output tree.md

# 配置
doctree config --provider openai --model gpt-4o-mini
```

## 许可证

MIT License
