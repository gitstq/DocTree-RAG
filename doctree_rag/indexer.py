"""
indexer.py - 文档解析与树索引构建

支持 PDF、Markdown、TXT、HTML 等多种文档格式的解析，
自动识别文档层级结构并构建树形索引。
支持增量索引（通过内容哈希检测变更）。
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .tree_model import DocumentTree, TreeNode
from .utils import (
    DocTreeError,
    DocumentParseError,
    ProgressCallback,
    compute_file_hash,
    compute_text_hash,
    count_tokens_approx,
    ensure_directory,
    extract_headings,
    generate_doc_id,
    generate_id,
    get_file_extension,
    is_supported_file,
    normalize_text,
    now_iso,
    split_text_by_length,
    truncate_text,
)


class DocumentIndexer:
    """文档索引器，负责解析文档并构建树形索引。

    Attributes:
        llm_client: LLM 客户端实例（可选，用于生成摘要）
        max_pages_per_node: 每个节点最大页数（PDF）
        max_tokens_per_node: 每个节点最大 token 数
        index_dir: 索引文件存储目录
        generate_summaries: 是否使用 LLM 生成摘要
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        max_pages_per_node: int = 5,
        max_tokens_per_node: int = 2000,
        index_dir: str = "",
        generate_summaries: bool = True,
    ) -> None:
        self.llm_client = llm_client
        self.max_pages_per_node = max_pages_per_node
        self.max_tokens_per_node = max_tokens_per_node
        self.index_dir = index_dir or os.path.join(
            os.path.expanduser("~"), ".doctree", "indexes"
        )
        self.generate_summaries = generate_summaries and llm_client is not None

    def index_file(
        self,
        file_path: str,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> DocumentTree:
        """索引单个文件，构建文档树。

        Args:
            file_path: 文件路径
            progress_callback: 进度回调

        Returns:
            构建好的 DocumentTree

        Raises:
            DocumentParseError: 文档解析失败
        """
        file_path = os.path.abspath(file_path)

        if not os.path.exists(file_path):
            raise DocumentParseError(f"文件不存在: {file_path}")

        if not is_supported_file(file_path):
            raise DocumentParseError(
                f"不支持的文件格式: {get_file_extension(file_path)}"
            )

        if progress_callback:
            progress_callback.update(0, f"正在解析: {os.path.basename(file_path)}")

        # 读取文件内容
        ext = get_file_extension(file_path)
        doc_name = os.path.basename(file_path)
        doc_id = generate_doc_id(doc_name)

        # 检查是否已有索引且内容未变更（增量索引）
        existing_tree = self._load_index(doc_id)
        if existing_tree:
            current_hash = compute_file_hash(file_path)
            if existing_tree.metadata.get("file_hash") == current_hash:
                if progress_callback:
                    progress_callback.finish()
                return existing_tree

        # 根据文件类型解析文档
        if ext == "pdf":
            sections = self._parse_pdf(file_path, progress_callback)
        elif ext in ("md", "markdown"):
            sections = self._parse_markdown(file_path)
        elif ext == "txt":
            sections = self._parse_txt(file_path)
        elif ext in ("html", "htm"):
            sections = self._parse_html(file_path)
        else:
            raise DocumentParseError(f"不支持的文件格式: {ext}")

        if progress_callback:
            progress_callback.update(30, "正在构建树结构")

        # 构建树结构
        tree = self._build_tree(sections, doc_id, doc_name, file_path)

        # 生成摘要
        if self.generate_summaries and self.llm_client:
            if progress_callback:
                progress_callback.update(50, "正在生成摘要")
            self._generate_all_summaries(tree, progress_callback)

        # 更新元数据
        tree.metadata["file_hash"] = compute_file_hash(file_path)
        tree.metadata["file_extension"] = ext
        tree.metadata["file_size"] = os.path.getsize(file_path)
        tree.updated_at = now_iso()

        # 保存索引
        if progress_callback:
            progress_callback.update(90, "正在保存索引")
        self._save_index(tree)

        if progress_callback:
            progress_callback.finish()

        return tree

    def index_directory(
        self,
        dir_path: str,
        recursive: bool = True,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> List[DocumentTree]:
        """索引目录中的所有文档。

        Args:
            dir_path: 目录路径
            recursive: 是否递归处理子目录
            progress_callback: 进度回调

        Returns:
            DocumentTree 列表
        """
        dir_path = os.path.abspath(dir_path)
        if not os.path.isdir(dir_path):
            raise DocumentParseError(f"目录不存在: {dir_path}")

        # 收集支持的文件
        files = []
        if recursive:
            for root, _, filenames in os.walk(dir_path):
                for fn in filenames:
                    fp = os.path.join(root, fn)
                    if is_supported_file(fp):
                        files.append(fp)
        else:
            for fn in os.listdir(dir_path):
                fp = os.path.join(dir_path, fn)
                if os.path.isfile(fp) and is_supported_file(fp):
                    files.append(fp)

        files.sort()

        if not files:
            return []

        if progress_callback:
            progress_callback.total = len(files)

        trees = []
        for i, fp in enumerate(files):
            if progress_callback:
                progress_callback.update(i, f"正在处理 ({i+1}/{len(files)}): {os.path.basename(fp)}")
            try:
                tree = self.index_file(fp)
                trees.append(tree)
            except DocumentParseError as e:
                # 跳过无法解析的文件，继续处理其他文件
                print(f"警告: 跳过文件 {fp}: {e}", file=__import__("sys").stderr)
                continue

        if progress_callback:
            progress_callback.finish()

        return trees

    # ========================================================
    # 文档解析方法
    # ========================================================

    def _parse_pdf(
        self,
        file_path: str,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> List[Dict[str, Any]]:
        """解析 PDF 文件。"""
        try:
            import PyPDF2
        except ImportError:
            try:
                import pdfplumber
                return self._parse_pdf_with_pdfplumber(file_path, progress_callback)
            except ImportError:
                raise DocumentParseError(
                    "解析 PDF 需要 PyPDF2 或 pdfplumber 库。"
                    "请安装: pip install PyPDF2"
                )

        return self._parse_pdf_with_pypdf2(file_path, progress_callback)

    def _parse_pdf_with_pypdf2(
        self,
        file_path: str,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> List[Dict[str, Any]]:
        """使用 PyPDF2 解析 PDF。"""
        import PyPDF2

        sections = []
        try:
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                total_pages = len(reader.pages)

                # 提取所有页面的文本
                pages_text = []
                for i, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    text = normalize_text(text)
                    pages_text.append((i + 1, text))

                    if progress_callback:
                        progress_callback.update(
                            int(10 * (i + 1) / total_pages),
                            f"正在提取文本: 第 {i+1}/{total_pages} 页"
                        )

                # 按页码范围分组为章节
                sections = self._group_pages_into_sections(pages_text)

        except Exception as e:
            raise DocumentParseError(f"PDF 解析失败: {e}")

        return sections

    def _parse_pdf_with_pdfplumber(
        self,
        file_path: str,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> List[Dict[str, Any]]:
        """使用 pdfplumber 解析 PDF。"""
        try:
            import pdfplumber
        except ImportError:
            raise DocumentParseError("需要 pdfplumber 库: pip install pdfplumber")

        sections = []
        try:
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                pages_text = []

                for i, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    text = normalize_text(text)
                    pages_text.append((i + 1, text))

                    if progress_callback:
                        progress_callback.update(
                            int(10 * (i + 1) / total_pages),
                            f"正在提取文本: 第 {i+1}/{total_pages} 页"
                        )

                sections = self._group_pages_into_sections(pages_text)

        except Exception as e:
            raise DocumentParseError(f"PDF 解析失败: {e}")

        return sections

    def _group_pages_into_sections(
        self, pages_text: List[Tuple[int, str]]
    ) -> List[Dict[str, Any]]:
        """将 PDF 页面按内容分组为章节。

        策略：
        1. 尝试根据文本特征（如全大写行、短行后跟长文本）检测标题
        2. 按 max_pages_per_node 分组
        3. 如果单节超过 max_tokens_per_node，进一步分割
        """
        sections = []

        # 尝试检测章节标题
        current_title = "文档开头"
        current_content = ""
        current_start_page = pages_text[0][0] if pages_text else 1
        current_end_page = current_start_page

        for page_num, text in pages_text:
            if not text.strip():
                current_end_page = page_num
                continue

            lines = text.split("\n")
            # 检测可能的标题行（短行、可能全大写或在文本开头）
            detected_title = None
            for j, line in enumerate(lines):
                stripped = line.strip()
                if (
                    j < 3  # 前几行
                    and stripped
                    and len(stripped) < 80
                    and not stripped.endswith("。")
                    and not stripped.endswith(".")
                    and not stripped.endswith("，")
                ):
                    # 可能是标题
                    if (
                        stripped.isupper()
                        or (len(stripped) < 40 and j == 0)
                        or re.match(r"^(第[一二三四五六七八九十百]+[章节编部篇]|Chapter|Section)\s", stripped, re.IGNORECASE)
                    ):
                        detected_title = stripped
                        # 保存之前的章节
                        if current_content.strip():
                            sections.append({
                                "title": current_title,
                                "content": current_content.strip(),
                                "start_page": current_start_page,
                                "end_page": current_end_page,
                                "level": 1,
                            })
                        current_title = detected_title
                        current_content = "\n".join(lines[j + 1:])
                        current_start_page = page_num
                        current_end_page = page_num
                        break

            if detected_title is None:
                current_content += "\n" + text
                current_end_page = page_num

            # 检查是否超过页数限制
            page_count = current_end_page - current_start_page + 1
            if page_count >= self.max_pages_per_node and current_content.strip():
                sections.append({
                    "title": current_title,
                    "content": current_content.strip(),
                    "start_page": current_start_page,
                    "end_page": current_end_page,
                    "level": 1,
                })
                current_title = f"续（第 {current_end_page + 1} 页起）"
                current_content = ""
                current_start_page = current_end_page + 1

        # 添加最后一个章节
        if current_content.strip():
            sections.append({
                "title": current_title,
                "content": current_content.strip(),
                "start_page": current_start_page,
                "end_page": current_end_page,
                "level": 1,
            })

        # 如果没有检测到任何章节，将整个文档作为一个章节
        if not sections and pages_text:
            all_text = "\n".join(text for _, text in pages_text)
            sections.append({
                "title": os.path.basename(""),
                "content": all_text.strip(),
                "start_page": pages_text[0][0],
                "end_page": pages_text[-1][0],
                "level": 1,
            })

        # 按最大 token 数进一步分割过长的章节
        final_sections = []
        for section in sections:
            tokens = count_tokens_approx(section["content"])
            if tokens <= self.max_tokens_per_node:
                final_sections.append(section)
            else:
                chunks = split_text_by_length(
                    section["content"],
                    max_length=self.max_tokens_per_node * 3,
                    overlap=100,
                )
                for i, chunk in enumerate(chunks):
                    sub_section = section.copy()
                    sub_section["content"] = chunk
                    if i > 0:
                        sub_section["title"] = f"{section['title']} (续{i})"
                    final_sections.append(sub_section)

        return final_sections

    def _parse_markdown(self, file_path: str) -> List[Dict[str, Any]]:
        """解析 Markdown 文件，根据标题层级构建章节结构。"""
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        text = normalize_text(text)
        lines = text.split("\n")

        sections = []
        current_section: Dict[str, Any] = {
            "title": "文档开头",
            "content": "",
            "level": 0,
            "start_page": None,
            "end_page": None,
        }

        for line in lines:
            heading_match = re.match(r"^(#{1,6})\s+(.+)$", line)
            if heading_match:
                # 保存之前的章节
                if current_section["content"].strip():
                    sections.append(current_section.copy())

                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                current_section = {
                    "title": title,
                    "content": "",
                    "level": level,
                    "start_page": None,
                    "end_page": None,
                }
            else:
                current_section["content"] += "\n" + line

        # 添加最后一个章节
        if current_section["content"].strip():
            sections.append(current_section)

        # 如果没有检测到标题，将整个文档作为一个章节
        if not sections and text.strip():
            sections.append({
                "title": os.path.basename(file_path),
                "content": text,
                "level": 1,
                "start_page": None,
                "end_page": None,
            })

        return sections

    def _parse_txt(self, file_path: str) -> List[Dict[str, Any]]:
        """解析纯文本文件。"""
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        text = normalize_text(text)
        lines = text.split("\n")

        sections = []
        current_title = "文档开头"
        current_content_lines: List[str] = []

        for line in lines:
            stripped = line.strip()
            # 检测可能的章节分隔（空行 + 短行 + 空行）
            if (
                stripped
                and len(stripped) < 60
                and not stripped.endswith("。")
                and not stripped.endswith(".")
                and not stripped.endswith("，")
                and not stripped.endswith(",")
                and current_content_lines
            ):
                # 保存之前的章节
                content = "\n".join(current_content_lines).strip()
                if content:
                    sections.append({
                        "title": current_title,
                        "content": content,
                        "level": 1,
                        "start_page": None,
                        "end_page": None,
                    })
                current_title = stripped
                current_content_lines = []
            else:
                current_content_lines.append(line)

        # 添加最后一个章节
        content = "\n".join(current_content_lines).strip()
        if content:
            sections.append({
                "title": current_title,
                "content": content,
                "level": 1,
                "start_page": None,
                "end_page": None,
            })

        # 如果没有检测到章节结构，按长度分割
        if not sections and text.strip():
            chunks = split_text_by_length(
                text, max_length=self.max_tokens_per_node * 3, overlap=100
            )
            for i, chunk in enumerate(chunks):
                sections.append({
                    "title": f"段落 {i + 1}" if len(chunks) > 1 else os.path.basename(file_path),
                    "content": chunk,
                    "level": 1,
                    "start_page": None,
                    "end_page": None,
                })

        return sections

    def _parse_html(self, file_path: str) -> List[Dict[str, Any]]:
        """解析 HTML 文件，提取文本内容并根据标题标签构建章节。"""
        with open(file_path, "r", encoding="utf-8") as f:
            html = f.read()

        # 简单的 HTML 标签提取（不依赖 BeautifulSoup）
        text = self._strip_html(html)
        text = normalize_text(text)

        # 尝试根据标题标签提取结构
        sections = []
        heading_pattern = re.compile(
            r"<(h[1-6])[^>]*>(.*?)</\1>", re.IGNORECASE | re.DOTALL
        )

        # 找到所有标题及其位置
        headings = []
        for match in heading_pattern.finditer(html):
            tag = match.group(1).lower()
            level = int(tag[1])
            title_text = self._strip_html(match.group(2)).strip()
            pos = match.end()
            headings.append((level, title_text, pos))

        if headings:
            for i, (level, title, start_pos) in enumerate(headings):
                end_pos = headings[i + 1][2] if i + 1 < len(headings) else len(html)
                content_html = html[start_pos:end_pos]
                content = self._strip_html(content_html)
                content = normalize_text(content)

                sections.append({
                    "title": title,
                    "content": content.strip(),
                    "level": level,
                    "start_page": None,
                    "end_page": None,
                })

        # 如果没有找到标题标签，将整个文档作为一个章节
        if not sections and text.strip():
            sections.append({
                "title": os.path.basename(file_path),
                "content": text,
                "level": 1,
                "start_page": None,
                "end_page": None,
            })

        return sections

    def _strip_html(self, html: str) -> str:
        """去除 HTML 标签，提取纯文本。"""
        # 替换常见的块级标签为换行
        html = re.sub(r"<(br|p|div|h[1-6]|li|tr)[^>]*>", "\n", html, flags=re.IGNORECASE)
        # 去除所有标签
        text = re.sub(r"<[^>]+>", "", html)
        # 解码 HTML 实体
        text = text.replace("&amp;", "&")
        text = text.replace("&lt;", "<")
        text = text.replace("&gt;", ">")
        text = text.replace("&quot;", '"')
        text = text.replace("&#39;", "'")
        text = text.replace("&nbsp;", " ")
        text = text.replace("&mdash;", "—")
        text = text.replace("&ndash;", "-")
        return text

    # ========================================================
    # 树构建
    # ========================================================

    def _build_tree(
        self,
        sections: List[Dict[str, Any]],
        doc_id: str,
        doc_name: str,
        doc_path: str,
    ) -> DocumentTree:
        """从解析的章节列表构建文档树。"""
        tree = DocumentTree(
            doc_id=doc_id,
            doc_name=doc_name,
            doc_path=doc_path,
        )

        if not sections:
            return tree

        # 根据 level 构建层级关系
        root_nodes: List[TreeNode] = []
        node_stack: List[Tuple[int, TreeNode]] = []  # (level, node)

        for i, section in enumerate(sections):
            node = TreeNode(
                node_id=generate_id("node"),
                title=section.get("title", f"章节 {i + 1}"),
                content=section.get("content", ""),
                start_page=section.get("start_page"),
                end_page=section.get("end_page"),
                content_hash=compute_text_hash(section.get("content", "")),
                metadata={"level": section.get("level", 1)},
            )

            level = section.get("level", 1)

            # 找到合适的父节点
            while node_stack and node_stack[-1][0] >= level:
                node_stack.pop()

            if node_stack:
                parent = node_stack[-1][1]
                parent.add_child(node)
            else:
                root_nodes.append(node)

            node_stack.append((level, node))

        tree.root_nodes = root_nodes
        return tree

    # ========================================================
    # 摘要生成
    # ========================================================

    def _generate_all_summaries(
        self,
        tree: DocumentTree,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> None:
        """为所有节点生成摘要。"""
        all_nodes = tree.get_all_nodes()
        total = len(all_nodes)

        for i, node in enumerate(all_nodes):
            if progress_callback:
                progress = 50 + int(40 * (i + 1) / total)
                progress_callback.update(progress, f"正在生成摘要: {node.title}")

            summary = self._generate_summary(node)
            if summary:
                node.summary = summary

    def _generate_summary(self, node: TreeNode) -> str:
        """为单个节点生成摘要。"""
        if not self.llm_client or not node.content:
            return ""

        content = truncate_text(node.content, max_length=3000)
        prompt = (
            f"请用简洁的中文总结以下文档章节的内容（不超过 200 字）：\n\n"
            f"标题: {node.title}\n\n"
            f"内容:\n{content}"
        )

        try:
            summary = self.llm_client.generate(
                prompt=prompt,
                system_prompt="你是一个文档摘要助手。请简洁准确地总结文档内容。",
                temperature=0.3,
                max_tokens=300,
            )
            return summary.strip()
        except Exception:
            return ""

    # ========================================================
    # 索引存储
    # ========================================================

    def _get_index_path(self, doc_id: str) -> str:
        """获取索引文件的存储路径。"""
        return os.path.join(self.index_dir, f"{doc_id}.json")

    def _save_index(self, tree: DocumentTree) -> None:
        """保存文档树索引到磁盘。"""
        ensure_directory(self.index_dir)
        index_path = self._get_index_path(tree.doc_id)
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(tree.to_json())

    def _load_index(self, doc_id: str) -> Optional[DocumentTree]:
        """从磁盘加载文档树索引。"""
        index_path = self._get_index_path(doc_id)
        if not os.path.exists(index_path):
            return None
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                return DocumentTree.from_json(f.read())
        except Exception:
            return None

    def load_all_indexes(self) -> List[DocumentTree]:
        """加载所有已保存的索引。"""
        trees = []
        if not os.path.exists(self.index_dir):
            return trees

        for filename in os.listdir(self.index_dir):
            if filename.endswith(".json"):
                try:
                    filepath = os.path.join(self.index_dir, filename)
                    with open(filepath, "r", encoding="utf-8") as f:
                        tree = DocumentTree.from_json(f.read())
                        trees.append(tree)
                except Exception:
                    continue

        return trees

    def delete_index(self, doc_id: str) -> bool:
        """删除指定文档的索引。"""
        index_path = self._get_index_path(doc_id)
        if os.path.exists(index_path):
            os.remove(index_path)
            return True
        return False
