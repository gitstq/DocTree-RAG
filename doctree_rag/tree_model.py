"""
tree_model.py - 树形数据结构模型

定义文档树的核心数据结构，包括 TreeNode 和 DocumentTree 类。
支持序列化/反序列化、节点查找、深度计算和 Markdown 导出等功能。
"""

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional


class TreeNode:
    """文档树节点，表示文档中的一个章节或段落。

    Attributes:
        node_id: 节点唯一标识符（自动生成的 UUID 或自定义 ID）
        title: 节点标题（通常是章节标题）
        summary: 节点内容摘要（可由 LLM 生成）
        start_page: 起始页码（仅适用于 PDF 等分页文档）
        end_page: 结束页码
        content_hash: 节点内容的哈希值，用于增量索引时检测变更
        children: 子节点列表
        metadata: 附加元数据字典
        content: 节点原始文本内容（可选存储）
        parent: 父节点引用（运行时使用，不参与序列化）
    """

    def __init__(
        self,
        node_id: str = "",
        title: str = "",
        summary: str = "",
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
        content_hash: str = "",
        content: str = "",
        children: Optional[List["TreeNode"]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.node_id = node_id
        self.title = title
        self.summary = summary
        self.start_page = start_page
        self.end_page = end_page
        self.content_hash = content_hash
        self.content = content
        self.children = children if children is not None else []
        self.metadata = metadata if metadata is not None else {}
        self.parent: Optional["TreeNode"] = None

        # 为子节点设置父引用
        for child in self.children:
            child.parent = self

    def add_child(self, child: "TreeNode") -> None:
        """添加子节点并建立双向引用。"""
        child.parent = self
        self.children.append(child)

    def remove_child(self, node_id: str) -> bool:
        """移除指定 ID 的子节点。返回是否成功移除。"""
        for i, child in enumerate(self.children):
            if child.node_id == node_id:
                self.children.pop(i)
                child.parent = None
                return True
        return False

    def find_node(self, node_id: str) -> Optional["TreeNode"]:
        """在当前节点及其所有后代中查找指定 ID 的节点。"""
        if self.node_id == node_id:
            return self
        for child in self.children:
            result = child.find_node(node_id)
            if result is not None:
                return result
        return None

    def get_depth(self) -> int:
        """获取当前节点的深度（根节点深度为 0）。"""
        depth = 0
        node = self.parent
        while node is not None:
            depth += 1
            node = node.parent
        return depth

    def get_level(self) -> int:
        """获取当前节点的层级（与 get_depth 相同，语义更明确）。"""
        return self.get_depth()

    def get_all_nodes(self) -> List["TreeNode"]:
        """获取当前节点及其所有后代节点的扁平列表。"""
        nodes = [self]
        for child in self.children:
            nodes.extend(child.get_all_nodes())
        return nodes

    def get_leaf_nodes(self) -> List["TreeNode"]:
        """获取所有叶子节点（没有子节点的节点）。"""
        if not self.children:
            return [self]
        leaves = []
        for child in self.children:
            leaves.extend(child.get_leaf_nodes())
        return leaves

    def count_descendants(self) -> int:
        """统计所有后代节点的数量。"""
        count = 0
        for child in self.children:
            count += 1 + child.count_descendants()
        return count

    def compute_content_hash(self, content: str) -> str:
        """计算文本内容的 SHA256 哈希值。"""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """将节点序列化为字典（不包含 parent 引用）。"""
        return {
            "node_id": self.node_id,
            "title": self.title,
            "summary": self.summary,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "content_hash": self.content_hash,
            "content": self.content,
            "children": [child.to_dict() for child in self.children],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TreeNode":
        """从字典反序列化创建节点。"""
        children = [cls.from_dict(c) for c in data.get("children", [])]
        node = cls(
            node_id=data.get("node_id", ""),
            title=data.get("title", ""),
            summary=data.get("summary", ""),
            start_page=data.get("start_page"),
            end_page=data.get("end_page"),
            content_hash=data.get("content_hash", ""),
            content=data.get("content", ""),
            children=children,
            metadata=data.get("metadata", {}),
        )
        # 建立父子引用
        for child in node.children:
            child.parent = node
        return node

    def to_markdown(self, level: int = 0) -> str:
        """将节点及其子树转换为 Markdown 格式。"""
        lines = []
        indent = "#" * min(level + 1, 6)
        if self.title:
            lines.append(f"{indent} {self.title}")
        if self.start_page is not None and self.end_page is not None:
            lines.append(f"*页码范围: {self.start_page}-{self.end_page}*")
        if self.summary:
            lines.append("")
            lines.append(self.summary)
        if self.content:
            lines.append("")
            lines.append(self.content)
        for child in self.children:
            lines.append("")
            lines.append(child.to_markdown(level + 1))
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"TreeNode(id={self.node_id!r}, title={self.title!r}, "
            f"children={len(self.children)})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TreeNode):
            return NotImplemented
        return self.node_id == other.node_id

    def __hash__(self) -> int:
        return hash(self.node_id)


class DocumentTree:
    """文档树，表示一个完整文档的层级结构。

    Attributes:
        doc_id: 文档唯一标识符
        doc_name: 文档名称
        doc_path: 文档原始路径
        root_nodes: 根节点列表（支持多根节点）
        created_at: 创建时间
        updated_at: 最后更新时间
        metadata: 附加元数据字典
    """

    def __init__(
        self,
        doc_id: str = "",
        doc_name: str = "",
        doc_path: str = "",
        root_nodes: Optional[List[TreeNode]] = None,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.doc_id = doc_id
        self.doc_name = doc_name
        self.doc_path = doc_path
        self.root_nodes = root_nodes if root_nodes is not None else []
        self.created_at = created_at or datetime.now().isoformat()
        self.updated_at = updated_at or datetime.now().isoformat()
        self.metadata = metadata if metadata is not None else {}

    def add_root(self, node: TreeNode) -> None:
        """添加根节点。"""
        self.root_nodes.append(node)

    def find_node(self, node_id: str) -> Optional[TreeNode]:
        """在整棵文档树中查找指定 ID 的节点。"""
        for root in self.root_nodes:
            result = root.find_node(node_id)
            if result is not None:
                return result
        return None

    def get_all_nodes(self) -> List[TreeNode]:
        """获取文档树中所有节点的扁平列表。"""
        nodes = []
        for root in self.root_nodes:
            nodes.extend(root.get_all_nodes())
        return nodes

    def get_leaf_nodes(self) -> List[TreeNode]:
        """获取文档树中所有叶子节点。"""
        leaves = []
        for root in self.root_nodes:
            leaves.extend(root.get_leaf_nodes())
        return leaves

    def get_depth(self) -> int:
        """获取文档树的最大深度。"""
        if not self.root_nodes:
            return 0

        def _max_depth(node: TreeNode) -> int:
            if not node.children:
                return 1
            return 1 + max(_max_depth(c) for c in node.children)

        return max(_max_depth(root) for root in self.root_nodes)

    def get_total_nodes(self) -> int:
        """获取文档树中节点的总数。"""
        return len(self.get_all_nodes())

    def to_dict(self) -> Dict[str, Any]:
        """将文档树序列化为字典。"""
        return {
            "doc_id": self.doc_id,
            "doc_name": self.doc_name,
            "doc_path": self.doc_path,
            "root_nodes": [root.to_dict() for root in self.root_nodes],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentTree":
        """从字典反序列化创建文档树。"""
        root_nodes = [TreeNode.from_dict(r) for r in data.get("root_nodes", [])]
        tree = cls(
            doc_id=data.get("doc_id", ""),
            doc_name=data.get("doc_name", ""),
            doc_path=data.get("doc_path", ""),
            root_nodes=root_nodes,
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            metadata=data.get("metadata", {}),
        )
        return tree

    def to_json(self, indent: int = 2) -> str:
        """将文档树序列化为 JSON 字符串。"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "DocumentTree":
        """从 JSON 字符串反序列化创建文档树。"""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def to_markdown(self) -> str:
        """将文档树转换为 Markdown 格式。"""
        lines = [f"# {self.doc_name}", ""]
        lines.append(f"- **文档 ID**: {self.doc_id}")
        lines.append(f"- **路径**: {self.doc_path}")
        lines.append(f"- **创建时间**: {self.created_at}")
        lines.append(f"- **节点数量**: {self.get_total_nodes()}")
        lines.append(f"- **树深度**: {self.get_depth()}")
        lines.append("")
        lines.append("---")
        lines.append("")
        for root in self.root_nodes:
            lines.append(root.to_markdown(level=1))
            lines.append("")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"DocumentTree(id={self.doc_id!r}, name={self.doc_name!r}, "
            f"roots={len(self.root_nodes)}, nodes={self.get_total_nodes()})"
        )
