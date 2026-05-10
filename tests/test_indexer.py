"""
test_indexer.py - 文档索引器单元测试

测试文档解析、树构建和索引存储功能。
使用内联测试数据，不依赖外部文件。
"""

import json
import os
import sys
import tempfile
import unittest
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from doctree_rag.indexer import DocumentIndexer
from doctree_rag.tree_model import DocumentTree, TreeNode
from doctree_rag.utils import compute_text_hash, normalize_text


class TestDocumentParsing(unittest.TestCase):
    """文档解析功能的单元测试。"""

    def setUp(self):
        """创建临时目录和索引器。"""
        self.temp_dir = tempfile.mkdtemp()
        self.indexer = DocumentIndexer(
            index_dir=os.path.join(self.temp_dir, "indexes"),
            generate_summaries=False,
        )

    def tearDown(self):
        """清理临时目录。"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_parse_markdown(self):
        """测试 Markdown 文件解析。"""
        md_content = """# 文档标题

这是文档的引言部分。

## 第一章 背景介绍

研究背景的详细描述。

### 1.1 研究动机

具体的研究动机说明。

### 1.2 研究目标

研究的主要目标。

## 第二章 方法论

研究方法的详细描述。
"""
        md_path = os.path.join(self.temp_dir, "test.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        sections = self.indexer._parse_markdown(md_path)
        self.assertGreater(len(sections), 0)

        # 验证标题层级
        titles = [s["title"] for s in sections]
        self.assertIn("文档标题", titles)
        self.assertIn("第一章 背景介绍", titles)

        # 验证层级
        for s in sections:
            if s["title"] == "文档标题":
                self.assertEqual(s["level"], 1)
            elif s["title"] in ("第一章 背景介绍", "第二章 方法论"):
                self.assertEqual(s["level"], 2)

    def test_parse_txt(self):
        """测试纯文本文件解析。"""
        txt_content = """这是一个测试文档。

第一部分

这是第一部分的内容，包含一些描述性文字。
这里继续描述第一部分的内容。

第二部分

这是第二部分的内容，包含更多的描述性文字。
"""
        txt_path = os.path.join(self.temp_dir, "test.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(txt_content)

        sections = self.indexer._parse_txt(txt_path)
        self.assertGreater(len(sections), 0)

    def test_parse_html(self):
        """测试 HTML 文件解析。"""
        html_content = """<!DOCTYPE html>
<html>
<head><title>测试文档</title></head>
<body>
<h1>文档标题</h1>
<p>引言内容。</p>
<h2>第一章</h2>
<p>第一章的内容。</p>
<h3>1.1 小节</h3>
<p>小节内容。</p>
<h2>第二章</h2>
<p>第二章的内容。</p>
</body>
</html>
"""
        html_path = os.path.join(self.temp_dir, "test.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        sections = self.indexer._parse_html(html_path)
        self.assertGreater(len(sections), 0)

        titles = [s["title"] for s in sections]
        self.assertIn("文档标题", titles)
        self.assertIn("第一章", titles)

    def test_parse_txt_empty(self):
        """测试空文本文件。"""
        txt_path = os.path.join(self.temp_dir, "empty.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("")

        sections = self.indexer._parse_txt(txt_path)
        # 空文件应该返回空列表或单个空章节
        self.assertIsInstance(sections, list)

    def test_build_tree(self):
        """测试树构建。"""
        sections = [
            {"title": "第一章", "content": "第一章内容", "level": 1, "start_page": 1, "end_page": 5},
            {"title": "1.1 小节", "content": "小节内容", "level": 2, "start_page": 1, "end_page": 3},
            {"title": "1.2 小节", "content": "小节内容", "level": 2, "start_page": 3, "end_page": 5},
            {"title": "第二章", "content": "第二章内容", "level": 1, "start_page": 6, "end_page": 10},
        ]

        tree = self.indexer._build_tree(
            sections, "doc_test", "测试文档", "/path/to/doc"
        )

        self.assertEqual(tree.doc_id, "doc_test")
        self.assertEqual(tree.doc_name, "测试文档")
        self.assertEqual(len(tree.root_nodes), 2)
        self.assertEqual(tree.get_total_nodes(), 4)

        # 验证层级关系
        ch1 = tree.find_node_by_title("第一章") if hasattr(tree, 'find_node_by_title') else None
        all_nodes = tree.get_all_nodes()
        titles = [n.title for n in all_nodes]
        self.assertIn("第一章", titles)
        self.assertIn("1.1 小节", titles)
        self.assertIn("第二章", titles)

    def test_build_tree_flat(self):
        """测试扁平结构的树构建。"""
        sections = [
            {"title": "段落 1", "content": "内容1", "level": 1},
            {"title": "段落 2", "content": "内容2", "level": 1},
            {"title": "段落 3", "content": "内容3", "level": 1},
        ]

        tree = self.indexer._build_tree(
            sections, "doc_flat", "扁平文档", "/path/to/flat"
        )

        self.assertEqual(len(tree.root_nodes), 3)
        self.assertEqual(tree.get_total_nodes(), 3)
        self.assertEqual(tree.get_depth(), 1)

    def test_build_tree_nested(self):
        """测试深层嵌套的树构建。"""
        sections = [
            {"title": "L1", "content": "c1", "level": 1},
            {"title": "L2", "content": "c2", "level": 2},
            {"title": "L3", "content": "c3", "level": 3},
            {"title": "L4", "content": "c4", "level": 4},
            {"title": "L2b", "content": "c2b", "level": 2},
        ]

        tree = self.indexer._build_tree(
            sections, "doc_nested", "嵌套文档", "/path/to/nested"
        )

        self.assertEqual(len(tree.root_nodes), 1)
        self.assertEqual(tree.get_total_nodes(), 5)
        self.assertEqual(tree.get_depth(), 4)


class TestIndexStorage(unittest.TestCase):
    """索引存储功能的单元测试。"""

    def setUp(self):
        """创建临时目录和索引器。"""
        self.temp_dir = tempfile.mkdtemp()
        self.indexer = DocumentIndexer(
            index_dir=os.path.join(self.temp_dir, "indexes"),
        )

    def tearDown(self):
        """清理临时目录。"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_and_load_index(self):
        """测试索引保存和加载。"""
        tree = DocumentTree(
            doc_id="doc_save_test",
            doc_name="保存测试",
            doc_path="/path/to/save_test",
            root_nodes=[
                TreeNode(node_id="n1", title="根节点", content="内容"),
            ],
        )

        self.indexer._save_index(tree)
        loaded = self.indexer._load_index("doc_save_test")

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.doc_id, "doc_save_test")
        self.assertEqual(loaded.doc_name, "保存测试")
        self.assertEqual(loaded.get_total_nodes(), 1)

    def test_load_nonexistent_index(self):
        """测试加载不存在的索引。"""
        loaded = self.indexer._load_index("nonexistent")
        self.assertIsNone(loaded)

    def test_delete_index(self):
        """测试删除索引。"""
        tree = DocumentTree(
            doc_id="doc_delete_test",
            doc_name="删除测试",
        )
        self.indexer._save_index(tree)

        result = self.indexer.delete_index("doc_delete_test")
        self.assertTrue(result)

        loaded = self.indexer._load_index("doc_delete_test")
        self.assertIsNone(loaded)

    def test_delete_nonexistent_index(self):
        """测试删除不存在的索引。"""
        result = self.indexer.delete_index("nonexistent")
        self.assertFalse(result)

    def test_load_all_indexes(self):
        """测试加载所有索引。"""
        for i in range(3):
            tree = DocumentTree(
                doc_id=f"doc_multi_{i}",
                doc_name=f"文档 {i}",
            )
            self.indexer._save_index(tree)

        trees = self.indexer.load_all_indexes()
        self.assertEqual(len(trees), 3)

    def test_load_all_indexes_empty(self):
        """测试空目录加载。"""
        trees = self.indexer.load_all_indexes()
        self.assertEqual(len(trees), 0)


class TestFullIndexing(unittest.TestCase):
    """完整索引流程的集成测试。"""

    def setUp(self):
        """创建临时目录。"""
        self.temp_dir = tempfile.mkdtemp()
        index_dir = os.path.join(self.temp_dir, "indexes")
        self.indexer = DocumentIndexer(
            index_dir=index_dir,
            generate_summaries=False,
        )

    def tearDown(self):
        """清理临时目录。"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_index_markdown_file(self):
        """测试完整的 Markdown 文件索引流程。"""
        md_content = """# 项目文档

## 概述

本项目是一个文档管理工具。

## 架构设计

### 前端

使用 React 构建。

### 后端

使用 Python Flask 构建。

## 部署指南

使用 Docker 进行部署。
"""
        md_path = os.path.join(self.temp_dir, "project.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        tree = self.indexer.index_file(md_path)

        self.assertIsNotNone(tree)
        self.assertGreater(tree.get_total_nodes(), 0)
        self.assertGreater(tree.get_depth(), 0)
        self.assertIsNotNone(tree.doc_id)
        self.assertEqual(tree.doc_name, "project.md")

    def test_index_txt_file(self):
        """测试纯文本文件索引。"""
        txt_content = """测试文档

这是第一部分的内容。
包含多行文字。

这是第二部分的内容。
"""
        txt_path = os.path.join(self.temp_dir, "test.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(txt_content)

        tree = self.indexer.index_file(txt_path)
        self.assertIsNotNone(tree)
        self.assertGreater(tree.get_total_nodes(), 0)

    def test_index_nonexistent_file(self):
        """测试索引不存在的文件。"""
        from doctree_rag.utils import DocumentParseError
        with self.assertRaises(DocumentParseError):
            self.indexer.index_file("/nonexistent/file.txt")

    def test_index_unsupported_format(self):
        """测试索引不支持的文件格式。"""
        from doctree_rag.utils import DocumentParseError
        path = os.path.join(self.temp_dir, "test.xyz")
        with open(path, "w") as f:
            f.write("test")
        with self.assertRaises(DocumentParseError):
            self.indexer.index_file(path)

    def test_incremental_indexing(self):
        """测试增量索引（内容未变更时使用缓存）。"""
        md_content = "# 测试文档\n\n内容不变。"
        md_path = os.path.join(self.temp_dir, "incremental.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        # 第一次索引
        tree1 = self.indexer.index_file(md_path)
        self.assertIsNotNone(tree1)

        # 第二次索引（内容未变，应使用缓存）
        tree2 = self.indexer.index_file(md_path)
        self.assertIsNotNone(tree2)
        self.assertEqual(tree1.doc_id, tree2.doc_id)


if __name__ == "__main__":
    unittest.main()
