"""
test_retriever.py - 检索引擎单元测试

测试推理检索和关键词搜索功能。
使用模拟 LLM 客户端进行测试。
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from doctree_rag.tree_model import TreeNode, DocumentTree
from doctree_rag.retriever import TreeRetriever, RetrievalResult
from doctree_rag.keyword_search import BM25Search


class MockLLMClient:
    """模拟 LLM 客户端，用于测试。"""

    def __init__(self, response: str = ""):
        self._response = response
        self.call_count = 0
        self.last_prompt = ""

    def generate(self, prompt: str, system_prompt: str = "", **kwargs) -> str:
        self.call_count += 1
        self.last_prompt = prompt
        return self._response

    def generate_stream(self, prompt: str, **kwargs):
        yield self._response

    def count_tokens(self, text: str) -> int:
        return len(text) // 4

    def is_available(self) -> bool:
        return bool(self._response)


class TestRetrievalResult(unittest.TestCase):
    """RetrievalResult 类的单元测试。"""

    def test_creation(self):
        """测试结果创建。"""
        result = RetrievalResult(
            node_id="n1",
            title="测试标题",
            summary="测试摘要",
            score=0.85,
            reasoning="相关度高",
        )
        self.assertEqual(result.node_id, "n1")
        self.assertEqual(result.title, "测试标题")
        self.assertEqual(result.score, 0.85)

    def test_to_dict(self):
        """测试字典转换。"""
        result = RetrievalResult(
            node_id="n1",
            title="标题",
            score=0.5,
        )
        d = result.to_dict()
        self.assertEqual(d["node_id"], "n1")
        self.assertEqual(d["title"], "标题")
        self.assertEqual(d["score"], 0.5)

    def test_repr(self):
        """测试字符串表示。"""
        result = RetrievalResult(node_id="n1", title="标题", score=0.75)
        r = repr(result)
        self.assertIn("n1", r)
        self.assertIn("0.750", r)


class TestBM25Search(unittest.TestCase):
    """BM25Search 类的单元测试。"""

    def setUp(self):
        """创建测试用的搜索引擎和文档树。"""
        self.search = BM25Search()

        # 创建测试文档树
        self.tree = DocumentTree(
            doc_id="doc_search",
            doc_name="搜索测试文档",
            root_nodes=[
                TreeNode(
                    node_id="n1",
                    title="机器学习基础",
                    summary="介绍机器学习的基本概念和算法",
                    content="机器学习是人工智能的一个分支，包括监督学习和无监督学习。",
                ),
                TreeNode(
                    node_id="n2",
                    title="深度学习",
                    summary="深度神经网络和深度学习框架",
                    content="深度学习使用多层神经网络来学习数据的层次表示。",
                ),
                TreeNode(
                    node_id="n3",
                    title="自然语言处理",
                    summary="NLP 技术在文本分析中的应用",
                    content="自然语言处理涉及文本分类、情感分析和机器翻译等任务。",
                ),
                TreeNode(
                    node_id="n4",
                    title="计算机视觉",
                    summary="图像识别和目标检测技术",
                    content="计算机视觉包括图像分类、目标检测和图像分割。",
                ),
            ],
        )

        self.search.index_tree(self.tree)

    def test_basic_search(self):
        """测试基本搜索。"""
        results = self.search.search("机器学习", top_k=2)
        self.assertGreater(len(results), 0)
        # 第一个结果应该与机器学习相关
        node_ids = [r[0] for r in results]
        self.assertIn("n1", node_ids)

    def test_search_empty_query(self):
        """测试空查询。"""
        results = self.search.search("", top_k=5)
        self.assertEqual(len(results), 0)

    def test_search_no_results(self):
        """测试无结果的查询。"""
        results = self.search.search("量子计算高能物理", top_k=5)
        # 可能返回空结果或低分结果
        self.assertIsInstance(results, list)

    def test_search_top_k(self):
        """测试结果数量限制。"""
        results = self.search.search("学习", top_k=2)
        self.assertLessEqual(len(results), 2)

    def test_search_with_doc_filter(self):
        """测试文档过滤。"""
        results = self.search.search("机器学习", top_k=5, doc_id="doc_search")
        self.assertGreater(len(results), 0)

    def test_search_with_doc_filter_no_match(self):
        """测试文档过滤（不匹配的文档 ID）。"""
        results = self.search.search("机器学习", top_k=5, doc_id="nonexistent")
        self.assertEqual(len(results), 0)

    def test_boolean_and_search(self):
        """测试 AND 布尔搜索。"""
        results = self.search.search("机器 AND 学习", top_k=5)
        self.assertIsInstance(results, list)

    def test_boolean_or_search(self):
        """测试 OR 布尔搜索。"""
        results = self.search.search("视觉 OR 语言", top_k=5)
        self.assertIsInstance(results, list)

    def test_boolean_not_search(self):
        """测试 NOT 布尔搜索。"""
        results = self.search.search("学习 NOT 深度", top_k=5)
        self.assertIsInstance(results, list)

    def test_phrase_search(self):
        """测试短语搜索。"""
        results = self.search.search('"机器学习"', top_k=5)
        self.assertIsInstance(results, list)

    def test_index_multiple_trees(self):
        """测试索引多棵文档树。"""
        tree2 = DocumentTree(
            doc_id="doc_search2",
            doc_name="第二篇文档",
            root_nodes=[
                TreeNode(
                    node_id="n5",
                    title="数据库系统",
                    summary="关系型数据库和 NoSQL 数据库",
                    content="数据库系统用于存储和管理结构化数据。",
                ),
            ],
        )
        self.search.index_tree(tree2)

        results = self.search.search("数据库", top_k=5)
        node_ids = [r[0] for r in results]
        self.assertIn("n5", node_ids)

    def test_get_stats(self):
        """测试统计信息。"""
        stats = self.search.get_stats()
        self.assertEqual(stats["total_docs"], 4)
        self.assertGreater(stats["vocabulary_size"], 0)
        self.assertEqual(stats["total_nodes_indexed"], 4)

    def test_chinese_tokenization(self):
        """测试中文分词。"""
        tokens = self.search._tokenize("机器学习算法")
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)

    def test_english_tokenization(self):
        """测试英文分词。"""
        tokens = self.search._tokenize("machine learning algorithms")
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)

    def test_mixed_tokenization(self):
        """测试中英文混合分词。"""
        tokens = self.search._tokenize("使用 Python 进行数据分析")
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)

    def test_stop_words_filtered(self):
        """测试停用词过滤。"""
        tokens = self.search._tokenize("the a an 的 了 在")
        # 停用词应该被过滤
        for token in tokens:
            self.assertNotIn(token, ["the", "a", "an", "的", "了", "在"])


class TestTreeRetriever(unittest.TestCase):
    """TreeRetriever 类的单元测试。"""

    def setUp(self):
        """创建测试用的检索器和文档树。"""
        self.tree = DocumentTree(
            doc_id="doc_retriever",
            doc_name="检索测试文档",
            root_nodes=[
                TreeNode(
                    node_id="r1",
                    title="Python 编程",
                    summary="Python 语言基础和高级特性",
                    content="Python 是一种广泛使用的高级编程语言。",
                ),
                TreeNode(
                    node_id="r2",
                    title="数据分析",
                    summary="使用 Pandas 和 NumPy 进行数据分析",
                    content="数据分析是数据科学的核心环节。",
                ),
            ],
        )

    def test_keyword_retrieve(self):
        """测试关键词检索。"""
        retriever = TreeRetriever()
        results = retriever.retrieve("Python 编程", self.tree, top_k=2)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        # 验证结果类型
        for r in results:
            self.assertIsInstance(r, RetrievalResult)
            self.assertIsNotNone(r.node_id)
            self.assertGreaterEqual(r.score, 0)
            self.assertLessEqual(r.score, 1)

    def test_keyword_retrieve_empty_query(self):
        """测试空查询的关键词检索。"""
        retriever = TreeRetriever()
        results = retriever.retrieve("", self.tree, top_k=5)
        self.assertEqual(len(results), 0)

    def test_retrieve_with_mock_llm(self):
        """测试使用模拟 LLM 的检索。"""
        mock_response = '''[
            {"node_id": "r1", "relevance": 0.9, "reasoning": "与 Python 编程直接相关"}
        ]'''
        mock_llm = MockLLMClient(response=mock_response)
        retriever = TreeRetriever(llm_client=mock_llm)

        results = retriever.retrieve("Python", self.tree, top_k=5)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].node_id, "r1")
        self.assertAlmostEqual(results[0].score, 0.9)
        self.assertEqual(mock_llm.call_count, 1)

    def test_retrieve_llm_fallback(self):
        """测试 LLM 失败时回退到关键词搜索。"""
        class FailingLLM:
            def generate(self, *args, **kwargs):
                from doctree_rag.utils import LLMError
                raise LLMError("模拟 LLM 失败")

        retriever = TreeRetriever(llm_client=FailingLLM())
        results = retriever.retrieve("Python", self.tree, top_k=5)
        # 应该回退到关键词搜索
        self.assertIsInstance(results, list)

    def test_retrieve_batch(self):
        """测试批量检索。"""
        retriever = TreeRetriever()
        queries = ["Python", "数据分析"]
        results = retriever.retrieve_batch(queries, [self.tree], top_k=2)
        self.assertIn("Python", results)
        self.assertIn("数据分析", results)
        self.assertIsInstance(results["Python"], list)
        self.assertIsInstance(results["数据分析"], list)

    def test_retrieve_score_normalization(self):
        """测试分数归一化。"""
        retriever = TreeRetriever()
        results = retriever.retrieve("Python", self.tree, top_k=10)
        for r in results:
            self.assertGreaterEqual(r.score, 0.0)
            self.assertLessEqual(r.score, 1.0)

    def test_retrieve_result_order(self):
        """测试结果按分数降序排列。"""
        retriever = TreeRetriever()
        results = retriever.retrieve("Python 数据", self.tree, top_k=10)
        if len(results) > 1:
            for i in range(len(results) - 1):
                self.assertGreaterEqual(results[i].score, results[i + 1].score)


if __name__ == "__main__":
    unittest.main()
