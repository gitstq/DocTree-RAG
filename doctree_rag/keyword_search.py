"""
keyword_search.py - 基于 BM25 的关键词搜索引擎

提供快速本地关键词搜索功能，支持 TF-IDF 评分、短语查询和布尔运算符。
不依赖外部库，纯 Python 实现。
"""

import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

from .tree_model import DocumentTree, TreeNode


class BM25Search:
    """基于 BM25 算法的关键词搜索引擎。

    对文档树的节点标题和摘要进行索引，支持快速关键词检索。

    Attributes:
        k1: BM25 词频饱和参数（默认 1.5）
        b: BM25 文档长度归一化参数（默认 0.75）
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        # 文档频率：词 -> 包含该词的文档数量
        self.doc_freq: Dict[str, int] = defaultdict(int)
        # 文档总数
        self.total_docs: int = 0
        # 每个文档的词频：doc_id -> {词: 频次}
        self.term_freqs: Dict[str, Counter] = {}
        # 每个文档的长度（词数）
        self.doc_lengths: Dict[str, int] = {}
        # 平均文档长度
        self.avg_doc_length: float = 0.0
        # 节点 ID 到 TreeNode 的映射
        self.node_map: Dict[str, TreeNode] = {}
        # 文档 ID 到节点 ID 列表的映射
        self.doc_nodes: Dict[str, List[str]] = defaultdict(list)

    def index_tree(self, tree: DocumentTree) -> None:
        """索引一棵文档树。"""
        for node in tree.get_all_nodes():
            self._index_node(node, tree.doc_id)

    def index_multiple_trees(self, trees: List[DocumentTree]) -> None:
        """索引多棵文档树。"""
        for tree in trees:
            self.index_tree(tree)

    def _index_node(self, node: TreeNode, doc_id: str) -> None:
        """索引单个节点。"""
        # 组合标题和摘要作为索引文本
        text_parts = [node.title, node.summary, node.content[:500] if node.content else ""]
        text = " ".join(part for part in text_parts if part)

        if not text.strip():
            return

        tokens = self._tokenize(text)
        if not tokens:
            return

        node_id = node.node_id
        self.node_map[node_id] = node
        self.doc_nodes[doc_id].append(node_id)

        # 更新词频
        tf = Counter(tokens)
        self.term_freqs[node_id] = tf
        self.doc_lengths[node_id] = len(tokens)

        # 更新文档频率
        seen_terms: Set[str] = set()
        for term in tokens:
            if term not in seen_terms:
                self.doc_freq[term] += 1
                seen_terms.add(term)

        self.total_docs += 1

    def _finalize(self) -> None:
        """计算平均文档长度（在搜索前调用）。"""
        if self.total_docs > 0:
            self.avg_doc_length = sum(self.doc_lengths.values()) / self.total_docs

    def search(
        self,
        query: str,
        top_k: int = 10,
        doc_id: Optional[str] = None,
    ) -> List[Tuple[str, float, str]]:
        """执行关键词搜索。

        Args:
            query: 搜索查询（支持 AND, OR, NOT 布尔运算符）
            top_k: 返回前 k 个结果
            doc_id: 限定在特定文档中搜索

        Returns:
            列表，每个元素为 (node_id, score, title) 元组，按分数降序排列
        """
        self._finalize()

        # 解析布尔查询
        bool_result = self._parse_boolean_query(query)

        # 如果是布尔查询结果，直接使用
        if bool_result is not None:
            matched_ids = bool_result
            # 为匹配的节点计算基础 BM25 分数
            results = []
            for node_id in matched_ids:
                node = self.node_map.get(node_id)
                if node is None:
                    continue
                if doc_id and node_id not in self.doc_nodes.get(doc_id, []):
                    continue
                score = self._compute_score(node_id, self._tokenize(query))
                results.append((node_id, score, node.title))
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]

        # 普通查询：提取关键词并计算 BM25 分数
        tokens = self._tokenize(query)
        if not tokens:
            return []

        scores: Dict[str, float] = defaultdict(float)
        candidate_ids = set(self.term_freqs.keys())

        # 如果限定了文档范围
        if doc_id:
            candidate_ids = set(self.doc_nodes.get(doc_id, []))

        for node_id in candidate_ids:
            score = self._compute_score(node_id, tokens)
            if score > 0:
                scores[node_id] = score

        # 排序并返回结果
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for node_id, score in sorted_results[:top_k]:
            node = self.node_map.get(node_id)
            title = node.title if node else ""
            results.append((node_id, score, title))

        return results

    def _compute_score(self, doc_id: str, query_tokens: List[str]) -> float:
        """计算单个文档的 BM25 分数。"""
        score = 0.0
        tf_map = self.term_freqs.get(doc_id, Counter())
        doc_len = self.doc_lengths.get(doc_id, 0)

        for term in query_tokens:
            if term not in tf_map:
                continue

            tf = tf_map[term]
            df = self.doc_freq.get(term, 0)
            idf = math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1.0)

            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (
                1 - self.b + self.b * (doc_len / max(self.avg_doc_length, 1e-10))
            )

            score += idf * (numerator / denominator)

        return score

    def _parse_boolean_query(
        self, query: str
    ) -> Optional[Set[str]]:
        """解析布尔查询表达式。

        支持:
        - AND: term1 AND term2（同时包含）
        - OR: term1 OR term2（包含任意一个）
        - NOT: term1 NOT term2（包含 term1 但不包含 term2）
        - 短语: "exact phrase"

        Returns:
            匹配的节点 ID 集合，如果查询不是布尔查询则返回 None
        """
        query = query.strip()

        # 检测布尔运算符
        has_and = bool(re.search(r"\bAND\b", query))
        has_or = bool(re.search(r"\bOR\b", query))
        has_not = bool(re.search(r"\bNOT\b", query))

        if not (has_and or has_or or has_not):
            # 检查短语查询
            phrase_match = re.search(r'"([^"]+)"', query)
            if phrase_match:
                phrase = phrase_match.group(1)
                phrase_tokens = self._tokenize(phrase)
                return self._phrase_search(phrase_tokens)
            return None

        # 解析 AND 查询
        if has_and:
            parts = re.split(r"\bAND\b", query)
            terms = [self._tokenize(p.strip()) for p in parts]
            return self._and_search(terms)

        # 解析 OR 查询
        if has_or:
            parts = re.split(r"\bOR\b", query)
            terms = [self._tokenize(p.strip()) for p in parts]
            return self._or_search(terms)

        # 解析 NOT 查询
        if has_not:
            parts = re.split(r"\bNOT\b", query)
            if len(parts) == 2:
                include_terms = self._tokenize(parts[0].strip())
                exclude_terms = self._tokenize(parts[1].strip())
                return self._not_search(include_terms, exclude_terms)

        return None

    def _and_search(self, term_groups: List[List[str]]) -> Set[str]:
        """AND 搜索：文档必须包含所有词组中的至少一个词。"""
        if not term_groups:
            return set()

        result = None
        for terms in term_groups:
            ids = self._find_docs_with_any_term(terms)
            if result is None:
                result = ids
            else:
                result = result & ids
            if not result:
                break

        return result or set()

    def _or_search(self, term_groups: List[List[str]]) -> Set[str]:
        """OR 搜索：文档包含任意词组中的任意词即可。"""
        result = set()
        for terms in term_groups:
            result |= self._find_docs_with_any_term(terms)
        return result

    def _not_search(
        self, include_terms: List[str], exclude_terms: List[str]
    ) -> Set[str]:
        """NOT 搜索：包含 include_terms 但不包含 exclude_terms。"""
        include_ids = self._find_docs_with_any_term(include_terms)
        exclude_ids = self._find_docs_with_any_term(exclude_terms)
        return include_ids - exclude_ids

    def _phrase_search(self, phrase_tokens: List[str]) -> Set[str]:
        """短语搜索：文档必须包含精确的短语。"""
        if not phrase_tokens:
            return set()

        matched = set()
        for node_id, tf in self.term_freqs.items():
            if all(term in tf for term in phrase_tokens):
                matched.add(node_id)

        return matched

    def _find_docs_with_any_term(self, terms: List[str]) -> Set[str]:
        """查找包含任意指定词的文档。"""
        result = set()
        for term in terms:
            for node_id, tf in self.term_freqs.items():
                if term in tf:
                    result.add(node_id)
        return result

    def _tokenize(self, text: str) -> List[str]:
        """文本分词。

        对中文进行单字切分，对英文进行词级切分，统一转小写。
        """
        tokens = []
        # 按空白字符和标点分割
        segments = re.split(r"[\s,.\-;:!?\(\)\[\]{}\"'/\\<>]+", text)

        for segment in segments:
            if not segment:
                continue
            # 检查是否包含 CJK 字符
            has_cjk = any(
                "\u4e00" <= c <= "\u9fff" or "\u3400" <= c <= "\u4dbf"
                for c in segment
            )
            if has_cjk:
                # CJK 字符单字切分
                for char in segment:
                    if (
                        "\u4e00" <= char <= "\u9fff"
                        or "\u3400" <= char <= "\u4dbf"
                        or char.isalnum()
                    ):
                        tokens.append(char.lower())
            else:
                # 英文按词切分
                word = segment.lower().strip()
                if word and len(word) > 0:
                    tokens.append(word)

        # 过滤停用词和过短的词
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "can", "shall",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after", "above",
            "below", "between", "out", "off", "over", "under", "again",
            "further", "then", "once", "here", "there", "when", "where",
            "why", "how", "all", "both", "each", "few", "more", "most",
            "other", "some", "such", "no", "nor", "not", "only", "own",
            "same", "so", "than", "too", "very", "just", "because", "but",
            "and", "or", "if", "while", "about", "up", "it", "its",
            "this", "that", "these", "those", "i", "me", "my", "we",
            "our", "you", "your", "he", "him", "his", "she", "her",
            "they", "them", "their", "what", "which", "who", "whom",
            "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
            "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
            "你", "会", "着", "没有", "看", "好", "自己", "这",
        }
        tokens = [t for t in tokens if t not in stop_words and len(t) > 0]

        return tokens

    def get_stats(self) -> Dict[str, any]:
        """获取搜索引擎的统计信息。"""
        self._finalize()
        return {
            "total_docs": self.total_docs,
            "vocabulary_size": len(self.doc_freq),
            "avg_doc_length": round(self.avg_doc_length, 2),
            "total_nodes_indexed": len(self.node_map),
        }
