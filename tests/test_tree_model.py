"""
test_tree_model.py - 树模型单元测试

测试 TreeNode 和 DocumentTree 的创建、序列化、查找等功能。
"""

import json
import unittest

import sys
import os

# 确保可以导入包
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from doctree_rag.tree_model import TreeNode, DocumentTree


class TestTreeNode(unittest.TestCase):
    """TreeNode 类的单元测试。"""

    def setUp(self):
        """创建测试用的树结构。"""
        self.leaf1 = TreeNode(
            node_id="node_1",
            title="第一章 引言",
            summary="本章介绍研究背景",
            start_page=1,
            end_page=5,
            content_hash="abc123",
            content="引言内容...",
        )
        self.leaf2 = TreeNode(
            node_id="node_2",
            title="第二章 方法",
            summary="本章描述研究方法",
            start_page=6,
            end_page=10,
            content_hash="def456",
            content="方法内容...",
        )
        self.leaf3 = TreeNode(
            node_id="node_3",
            title="2.1 数据收集",
            summary="数据收集的具体步骤",
            start_page=6,
            end_page=8,
            content_hash="ghi789",
        )
        self.root = TreeNode(
            node_id="node_root",
            title="文档根节点",
            summary="根节点摘要",
            children=[self.leaf1, self.leaf2],
        )
        self.leaf2.add_child(self.leaf3)

    def test_node_creation(self):
        """测试节点创建。"""
        node = TreeNode(
            node_id="test_1",
            title="测试标题",
            summary="测试摘要",
        )
        self.assertEqual(node.node_id, "test_1")
        self.assertEqual(node.title, "测试标题")
        self.assertEqual(node.summary, "测试摘要")
        self.assertIsNone(node.start_page)
        self.assertEqual(node.children, [])
        self.assertEqual(node.metadata, {})

    def test_add_child(self):
        """测试添加子节点。"""
        parent = TreeNode(node_id="p", title="父节点")
        child = TreeNode(node_id="c", title="子节点")
        parent.add_child(child)
        self.assertEqual(len(parent.children), 1)
        self.assertIs(child.parent, parent)

    def test_remove_child(self):
        """测试移除子节点。"""
        self.root.remove_child("node_1")
        self.assertEqual(len(self.root.children), 1)
        self.assertEqual(self.root.children[0].node_id, "node_2")

    def test_remove_child_not_found(self):
        """测试移除不存在的子节点。"""
        result = self.root.remove_child("nonexistent")
        self.assertFalse(result)
        self.assertEqual(len(self.root.children), 2)

    def test_find_node(self):
        """测试节点查找。"""
        found = self.root.find_node("node_3")
        self.assertIsNotNone(found)
        self.assertEqual(found.node_id, "node_3")
        self.assertEqual(found.title, "2.1 数据收集")

    def test_find_node_not_found(self):
        """测试查找不存在的节点。"""
        found = self.root.find_node("nonexistent")
        self.assertIsNone(found)

    def test_find_node_self(self):
        """测试查找自身。"""
        found = self.root.find_node("node_root")
        self.assertIsNotNone(found)
        self.assertIs(found, self.root)

    def test_get_depth(self):
        """测试节点深度计算。"""
        self.assertEqual(self.root.get_depth(), 0)
        self.assertEqual(self.leaf1.get_depth(), 1)
        self.assertEqual(self.leaf2.get_depth(), 1)
        self.assertEqual(self.leaf3.get_depth(), 2)

    def test_get_all_nodes(self):
        """测试获取所有节点。"""
        nodes = self.root.get_all_nodes()
        self.assertEqual(len(nodes), 4)
        ids = [n.node_id for n in nodes]
        self.assertIn("node_root", ids)
        self.assertIn("node_1", ids)
        self.assertIn("node_2", ids)
        self.assertIn("node_3", ids)

    def test_get_leaf_nodes(self):
        """测试获取叶子节点。"""
        leaves = self.root.get_leaf_nodes()
        self.assertEqual(len(leaves), 2)
        ids = [n.node_id for n in leaves]
        self.assertIn("node_1", ids)
        self.assertIn("node_3", ids)

    def test_count_descendants(self):
        """测试后代节点计数。"""
        self.assertEqual(self.root.count_descendants(), 3)
        self.assertEqual(self.leaf1.count_descendants(), 0)
        self.assertEqual(self.leaf2.count_descendants(), 1)

    def test_compute_content_hash(self):
        """测试内容哈希计算。"""
        hash1 = self.leaf1.compute_content_hash("测试内容")
        hash2 = self.leaf1.compute_content_hash("测试内容")
        hash3 = self.leaf1.compute_content_hash("不同内容")
        self.assertEqual(hash1, hash2)
        self.assertNotEqual(hash1, hash3)
        self.assertEqual(len(hash1), 16)

    def test_to_dict(self):
        """测试序列化为字典。"""
        d = self.leaf1.to_dict()
        self.assertEqual(d["node_id"], "node_1")
        self.assertEqual(d["title"], "第一章 引言")
        self.assertEqual(d["summary"], "本章介绍研究背景")
        self.assertEqual(d["start_page"], 1)
        self.assertEqual(d["end_page"], 5)
        self.assertEqual(d["content_hash"], "abc123")
        self.assertEqual(d["children"], [])
        self.assertNotIn("parent", d)

    def test_from_dict(self):
        """测试从字典反序列化。"""
        d = self.leaf1.to_dict()
        node = TreeNode.from_dict(d)
        self.assertEqual(node.node_id, "node_1")
        self.assertEqual(node.title, "第一章 引言")
        self.assertEqual(node.summary, "本章介绍研究背景")
        self.assertEqual(node.start_page, 1)

    def test_from_dict_with_children(self):
        """测试反序列化带子节点的字典。"""
        d = self.root.to_dict()
        node = TreeNode.from_dict(d)
        self.assertEqual(node.node_id, "node_root")
        self.assertEqual(len(node.children), 2)
        self.assertEqual(node.children[0].node_id, "node_1")
        # 验证父子引用
        self.assertIs(node.children[0].parent, node)

    def test_serialization_roundtrip(self):
        """测试序列化-反序列化的往返一致性。"""
        d = self.root.to_dict()
        json_str = json.dumps(d, ensure_ascii=False)
        d2 = json.loads(json_str)
        node = TreeNode.from_dict(d2)

        self.assertEqual(node.node_id, self.root.node_id)
        self.assertEqual(node.title, self.root.title)
        self.assertEqual(len(node.children), len(self.root.children))
        self.assertEqual(node.children[1].children[0].node_id, "node_3")

    def test_to_markdown(self):
        """测试 Markdown 转换。"""
        md = self.leaf1.to_markdown(level=0)
        self.assertIn("# 第一章 引言", md)
        self.assertIn("页码范围: 1-5", md)
        self.assertIn("本章介绍研究背景", md)

    def test_to_markdown_with_content(self):
        """测试带内容的 Markdown 转换。"""
        md = self.leaf1.to_markdown(level=1)
        self.assertIn("## 第一章 引言", md)
        self.assertIn("引言内容...", md)

    def test_equality(self):
        """测试节点相等性比较。"""
        node1 = TreeNode(node_id="same_id", title="标题A")
        node2 = TreeNode(node_id="same_id", title="标题B")
        node3 = TreeNode(node_id="diff_id", title="标题A")
        self.assertEqual(node1, node2)
        self.assertNotEqual(node1, node3)

    def test_hash(self):
        """测试节点哈希。"""
        node1 = TreeNode(node_id="same_id")
        node2 = TreeNode(node_id="same_id")
        self.assertEqual(hash(node1), hash(node2))

    def test_repr(self):
        """测试字符串表示。"""
        r = repr(self.leaf1)
        self.assertIn("node_1", r)
        self.assertIn("第一章 引言", r)

    def test_empty_node(self):
        """测试空节点。"""
        node = TreeNode()
        self.assertEqual(node.node_id, "")
        self.assertEqual(node.title, "")
        self.assertEqual(node.children, [])
        self.assertEqual(node.get_all_nodes(), [node])
        self.assertEqual(node.get_leaf_nodes(), [node])
        self.assertEqual(node.get_depth(), 0)


class TestDocumentTree(unittest.TestCase):
    """DocumentTree 类的单元测试。"""

    def setUp(self):
        """创建测试用的文档树。"""
        self.root1 = TreeNode(node_id="n1", title="第一章", summary="摘要1")
        self.root2 = TreeNode(node_id="n2", title="第二章", summary="摘要2")
        self.child = TreeNode(node_id="n3", title="2.1 小节", summary="子摘要")
        self.root2.add_child(self.child)

        self.tree = DocumentTree(
            doc_id="doc_test",
            doc_name="测试文档.pdf",
            doc_path="/path/to/test.pdf",
            root_nodes=[self.root1, self.root2],
        )

    def test_tree_creation(self):
        """测试文档树创建。"""
        self.assertEqual(self.tree.doc_id, "doc_test")
        self.assertEqual(self.tree.doc_name, "测试文档.pdf")
        self.assertEqual(self.tree.doc_path, "/path/to/test.pdf")
        self.assertEqual(len(self.tree.root_nodes), 2)
        self.assertIsNotNone(self.tree.created_at)
        self.assertIsNotNone(self.tree.updated_at)

    def test_add_root(self):
        """测试添加根节点。"""
        new_root = TreeNode(node_id="n4", title="附录")
        self.tree.add_root(new_root)
        self.assertEqual(len(self.tree.root_nodes), 3)

    def test_find_node(self):
        """测试在文档树中查找节点。"""
        found = self.tree.find_node("n3")
        self.assertIsNotNone(found)
        self.assertEqual(found.title, "2.1 小节")

    def test_find_node_not_found(self):
        """测试查找不存在的节点。"""
        found = self.tree.find_node("nonexistent")
        self.assertIsNone(found)

    def test_get_all_nodes(self):
        """测试获取所有节点。"""
        nodes = self.tree.get_all_nodes()
        self.assertEqual(len(nodes), 3)

    def test_get_leaf_nodes(self):
        """测试获取叶子节点。"""
        leaves = self.tree.get_leaf_nodes()
        self.assertEqual(len(leaves), 2)
        ids = [n.node_id for n in leaves]
        self.assertIn("n1", ids)
        self.assertIn("n3", ids)

    def test_get_depth(self):
        """测试树深度计算。"""
        self.assertEqual(self.tree.get_depth(), 2)

    def test_get_depth_empty(self):
        """测试空树深度。"""
        empty_tree = DocumentTree()
        self.assertEqual(empty_tree.get_depth(), 0)

    def test_get_total_nodes(self):
        """测试节点总数。"""
        self.assertEqual(self.tree.get_total_nodes(), 3)

    def test_to_dict(self):
        """测试序列化为字典。"""
        d = self.tree.to_dict()
        self.assertEqual(d["doc_id"], "doc_test")
        self.assertEqual(d["doc_name"], "测试文档.pdf")
        self.assertEqual(len(d["root_nodes"]), 2)
        self.assertIn("created_at", d)
        self.assertIn("updated_at", d)

    def test_from_dict(self):
        """测试从字典反序列化。"""
        d = self.tree.to_dict()
        tree = DocumentTree.from_dict(d)
        self.assertEqual(tree.doc_id, "doc_test")
        self.assertEqual(tree.doc_name, "测试文档.pdf")
        self.assertEqual(len(tree.root_nodes), 2)
        self.assertEqual(tree.root_nodes[1].children[0].node_id, "n3")

    def test_to_json(self):
        """测试 JSON 序列化。"""
        json_str = self.tree.to_json()
        self.assertIsInstance(json_str, str)
        data = json.loads(json_str)
        self.assertEqual(data["doc_id"], "doc_test")

    def test_from_json(self):
        """测试 JSON 反序列化。"""
        json_str = self.tree.to_json()
        tree = DocumentTree.from_json(json_str)
        self.assertEqual(tree.doc_id, "doc_test")
        self.assertEqual(tree.get_total_nodes(), 3)

    def test_to_markdown(self):
        """测试 Markdown 导出。"""
        md = self.tree.to_markdown()
        self.assertIn("# 测试文档.pdf", md)
        self.assertIn("doc_test", md)
        self.assertIn("/path/to/test.pdf", md)
        self.assertIn("节点数量", md)
        self.assertIn("树深度", md)

    def test_serialization_roundtrip(self):
        """测试完整的序列化-反序列化往返。"""
        json_str = self.tree.to_json()
        tree2 = DocumentTree.from_json(json_str)

        self.assertEqual(tree2.doc_id, self.tree.doc_id)
        self.assertEqual(tree2.doc_name, self.tree.doc_name)
        self.assertEqual(tree2.get_total_nodes(), self.tree.get_total_nodes())
        self.assertEqual(tree2.get_depth(), self.tree.get_depth())

        # 验证深层节点
        original_child = self.tree.find_node("n3")
        restored_child = tree2.find_node("n3")
        self.assertEqual(original_child.title, restored_child.title)
        self.assertEqual(original_child.summary, restored_child.summary)

    def test_repr(self):
        """测试字符串表示。"""
        r = repr(self.tree)
        self.assertIn("doc_test", r)
        self.assertIn("测试文档.pdf", r)


if __name__ == "__main__":
    unittest.main()
