"""
exporter.py - 文档树导出工具

支持将文档树导出为 JSON、Markdown 和 HTML 格式。
"""

import json
import os
from typing import Any, Dict, List, Optional

from .tree_model import DocumentTree, TreeNode


class TreeExporter:
    """文档树导出器。

    支持 JSON、Markdown 和 HTML 三种导出格式。
    """

    def __init__(self, indent: int = 2) -> None:
        self.indent = indent

    def export(
        self,
        tree: DocumentTree,
        format: str = "json",
        output_path: str = "",
    ) -> str:
        """导出文档树。

        Args:
            tree: 文档树
            format: 输出格式 (json/markdown/html)
            output_path: 输出文件路径（可选，不指定则返回字符串）

        Returns:
            导出的内容字符串

        Raises:
            ValueError: 不支持的格式
        """
        format = format.lower()

        if format == "json":
            content = self._export_json(tree)
        elif format in ("markdown", "md"):
            content = self._export_markdown(tree)
        elif format == "html":
            content = self._export_html(tree)
        else:
            raise ValueError(f"不支持的导出格式: {format}")

        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)

        return content

    def _export_json(self, tree: DocumentTree) -> str:
        """导出为 JSON 格式。"""
        return tree.to_json(indent=self.indent)

    def _export_markdown(self, tree: DocumentTree) -> str:
        """导出为 Markdown 格式。"""
        return tree.to_markdown()

    def _export_html(self, tree: DocumentTree) -> str:
        """导出为 HTML 格式，包含可折叠的树形结构。"""
        html_parts = [
            "<!DOCTYPE html>",
            '<html lang="zh-CN">',
            "<head>",
            '  <meta charset="UTF-8">',
            '  <meta name="viewport" content="width=device-width, initial-scale=1.0">',
            f"  <title>{self._escape_html(tree.doc_name)} - DocTree</title>",
            "  <style>",
            self._get_html_css(),
            "  </style>",
            "</head>",
            "<body>",
            '  <div class="container">',
            f'    <h1 class="doc-title">{self._escape_html(tree.doc_name)}</h1>',
            '    <div class="doc-meta">',
            f'      <span>文档 ID: {self._escape_html(tree.doc_id)}</span>',
            f'      <span>路径: {self._escape_html(tree.doc_path)}</span>',
            f'      <span>节点数: {tree.get_total_nodes()}</span>',
            f'      <span>深度: {tree.get_depth()}</span>',
            "    </div>",
            '    <div class="tree">',
        ]

        for root in tree.root_nodes:
            html_parts.append(self._node_to_html(root))

        html_parts.extend([
            "    </div>",
            "  </div>",
            "  <script>",
            self._get_html_js(),
            "  </script>",
            "</body>",
            "</html>",
        ])

        return "\n".join(html_parts)

    def _node_to_html(self, node: TreeNode) -> str:
        """将单个节点转换为 HTML。"""
        page_info = ""
        if node.start_page is not None and node.end_page is not None:
            page_info = f' <span class="page-range">[页 {node.start_page}-{node.end_page}]</span>'

        summary_html = ""
        if node.summary:
            summary_html = f'<div class="node-summary">{self._escape_html(node.summary)}</div>'

        children_html = ""
        if node.children:
            children_parts = []
            for child in node.children:
                children_parts.append(self._node_to_html(child))
            children_html = '<div class="node-children">' + "\n".join(children_parts) + "</div>"

        toggle_icon = ""
        if node.children:
            toggle_icon = '<span class="toggle-icon">&#9654;</span> '

        return (
            f'<div class="tree-node" data-id="{self._escape_html(node.node_id)}">'
            f'  <div class="node-header" onclick="toggleNode(this)">'
            f'    {toggle_icon}'
            f'    <span class="node-title">{self._escape_html(node.title)}</span>'
            f'    {page_info}'
            f"  </div>"
            f"  {summary_html}"
            f"  {children_html}"
            f"</div>"
        )

    def _escape_html(self, text: str) -> str:
        """转义 HTML 特殊字符。"""
        text = text.replace("&", "&amp;")
        text = text.replace("<", "&lt;")
        text = text.replace(">", "&gt;")
        text = text.replace('"', "&quot;")
        text = text.replace("'", "&#39;")
        return text

    def _get_html_css(self) -> str:
        """获取 HTML 样式。"""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f8f9fa;
            padding: 20px;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            padding: 30px;
        }
        .doc-title {
            font-size: 24px;
            color: #1a1a2e;
            margin-bottom: 10px;
            border-bottom: 2px solid #e9ecef;
            padding-bottom: 10px;
        }
        .doc-meta {
            display: flex;
            gap: 20px;
            color: #6c757d;
            font-size: 14px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        .tree {
            margin-left: 0;
        }
        .tree-node {
            margin-left: 0;
        }
        .node-children {
            margin-left: 24px;
            border-left: 2px solid #e9ecef;
            padding-left: 12px;
        }
        .node-header {
            display: flex;
            align-items: center;
            padding: 6px 10px;
            cursor: pointer;
            border-radius: 4px;
            transition: background-color 0.2s;
            gap: 6px;
        }
        .node-header:hover {
            background-color: #f1f3f5;
        }
        .toggle-icon {
            font-size: 10px;
            color: #adb5bd;
            transition: transform 0.2s;
            display: inline-block;
            width: 12px;
            text-align: center;
        }
        .toggle-icon.expanded {
            transform: rotate(90deg);
        }
        .node-title {
            font-weight: 500;
            color: #2d3436;
        }
        .page-range {
            font-size: 12px;
            color: #adb5bd;
        }
        .node-summary {
            font-size: 14px;
            color: #636e72;
            margin: 4px 0 4px 28px;
            padding: 8px 12px;
            background: #f8f9fa;
            border-radius: 4px;
            border-left: 3px solid #74b9ff;
        }
        """

    def _get_html_js(self) -> str:
        """获取 HTML 交互脚本。"""
        return """
        function toggleNode(header) {
            var node = header.parentElement;
            var children = node.querySelector('.node-children');
            var icon = header.querySelector('.toggle-icon');

            if (children) {
                if (children.style.display === 'none') {
                    children.style.display = 'block';
                    if (icon) icon.classList.add('expanded');
                } else {
                    children.style.display = 'none';
                    if (icon) icon.classList.remove('expanded');
                }
            }
        }

        // 初始化：展开第一层
        document.addEventListener('DOMContentLoaded', function() {
            var topLevelNodes = document.querySelectorAll('.tree > .tree-node > .node-children');
            topLevelNodes.forEach(function(el) {
                el.style.display = 'block';
            });
            var topLevelIcons = document.querySelectorAll('.tree > .tree-node > .node-header .toggle-icon');
            topLevelIcons.forEach(function(el) {
                el.classList.add('expanded');
            });
        });
        """


def export_tree(
    tree: DocumentTree,
    format: str = "json",
    output_path: str = "",
) -> str:
    """便捷函数：导出文档树。

    Args:
        tree: 文档树
        format: 输出格式 (json/markdown/html)
        output_path: 输出文件路径

    Returns:
        导出的内容字符串
    """
    exporter = TreeExporter()
    return exporter.export(tree, format=format, output_path=output_path)
