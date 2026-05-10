"""
retriever.py - 基于推理的检索引擎

利用 LLM 对文档树进行语义推理，找出与查询最相关的节点。
当 LLM 不可用时，自动回退到关键词搜索。
"""

import json
from typing import Any, Dict, List, Optional, Tuple

from .tree_model import DocumentTree, TreeNode
from .keyword_search import BM25Search
from .utils import LLMError, truncate_text


class RetrievalResult:
    """检索结果，包含匹配的节点和推理说明。

    Attributes:
        node_id: 匹配节点的 ID
        title: 节点标题
        summary: 节点摘要
        score: 相关性分数（0-1）
        reasoning: 推理说明（LLM 生成）
        start_page: 起始页码
        end_page: 结束页码
        content: 节点内容（可选）
    """

    def __init__(
        self,
        node_id: str = "",
        title: str = "",
        summary: str = "",
        score: float = 0.0,
        reasoning: str = "",
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
        content: str = "",
    ) -> None:
        self.node_id = node_id
        self.title = title
        self.summary = summary
        self.score = score
        self.reasoning = reasoning
        self.start_page = start_page
        self.end_page = end_page
        self.content = content

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "node_id": self.node_id,
            "title": self.title,
            "summary": self.summary,
            "score": self.score,
            "reasoning": self.reasoning,
            "start_page": self.start_page,
            "end_page": self.end_page,
        }

    def __repr__(self) -> str:
        return (
            f"RetrievalResult(id={self.node_id!r}, title={self.title!r}, "
            f"score={self.score:.3f})"
        )


class TreeRetriever:
    """基于推理的文档树检索引擎。

    使用 LLM 对查询和文档树结构进行推理，找出最相关的节点。
    支持批量查询和关键词搜索回退。

    Attributes:
        llm_client: LLM 客户端实例（可选）
        keyword_search: 关键词搜索引擎实例
        temperature: LLM 生成温度
        max_tokens: LLM 最大生成 token 数
        top_k: 默认返回结果数量
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        keyword_search: Optional[BM25Search] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        top_k: int = 5,
    ) -> None:
        self.llm_client = llm_client
        self.keyword_search = keyword_search or BM25Search()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_k = top_k

    def retrieve(
        self,
        query: str,
        tree: DocumentTree,
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """对单棵文档树执行检索。

        Args:
            query: 查询文本
            tree: 文档树
            top_k: 返回前 k 个结果

        Returns:
            检索结果列表，按相关性分数降序排列
        """
        k = top_k or self.top_k

        # 如果有 LLM 客户端，使用推理检索
        if self.llm_client:
            try:
                return self._reasoning_retrieve(query, tree, k)
            except LLMError:
                # LLM 失败，回退到关键词搜索
                pass

        # 回退到关键词搜索
        return self._keyword_retrieve(query, tree, k)

    def retrieve_batch(
        self,
        queries: List[str],
        trees: List[DocumentTree],
        top_k: Optional[int] = None,
    ) -> Dict[str, List[RetrievalResult]]:
        """批量检索。

        Args:
            queries: 查询文本列表
            trees: 文档树列表
            top_k: 返回前 k 个结果

        Returns:
            字典，键为查询文本，值为对应的检索结果列表
        """
        results = {}
        for query in queries:
            all_results = []
            for tree in trees:
                all_results.extend(self.retrieve(query, tree, top_k))
            # 按分数排序并截取前 k 个
            all_results.sort(key=lambda r: r.score, reverse=True)
            k = top_k or self.top_k
            results[query] = all_results[:k]
        return results

    def _reasoning_retrieve(
        self,
        query: str,
        tree: DocumentTree,
        top_k: int,
    ) -> List[RetrievalResult]:
        """使用 LLM 推理进行检索。"""
        # 构建文档树的摘要视图
        tree_summary = self._build_tree_summary(tree)

        # 构建推理提示
        prompt = self._build_retrieval_prompt(query, tree_summary, top_k)

        # 调用 LLM
        response = self.llm_client.generate(
            prompt=prompt,
            system_prompt=(
                "你是一个文档检索助手。根据用户的查询和文档结构，"
                "找出最可能包含答案的文档章节。"
                "你必须严格按照指定的 JSON 格式输出结果。"
            ),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # 解析 LLM 响应
        return self._parse_retrieval_response(response, tree)

    def _build_tree_summary(self, tree: DocumentTree) -> str:
        """构建文档树的摘要视图，用于 LLM 推理。"""
        lines = [f"文档: {tree.doc_name}", ""]

        def _walk_node(node: TreeNode, indent: int = 0) -> None:
            prefix = "  " * indent
            page_info = ""
            if node.start_page is not None and node.end_page is not None:
                page_info = f" [页 {node.start_page}-{node.end_page}]"

            summary_info = ""
            if node.summary:
                summary_info = f" - {truncate_text(node.summary, 100)}"

            lines.append(
                f"{prefix}- [{node.node_id}] {node.title}{page_info}{summary_info}"
            )

            for child in node.children:
                _walk_node(child, indent + 1)

        for root in tree.root_nodes:
            _walk_node(root)

        return "\n".join(lines)

    def _build_retrieval_prompt(
        self, query: str, tree_summary: str, top_k: int
    ) -> str:
        """构建检索提示。"""
        prompt = f"""用户查询: {query}

文档结构:
{tree_summary}

请分析用户查询，从上面的文档结构中找出最相关的 {top_k} 个章节。

输出格式（严格 JSON 数组）:
[
  {{
    "node_id": "节点ID",
    "relevance": 0.0到1.0之间的相关性分数,
    "reasoning": "选择该章节的理由"
  }}
]

只输出 JSON 数组，不要输出其他内容。"""

        return prompt

    def _parse_retrieval_response(
        self, response: str, tree: DocumentTree
    ) -> List[RetrievalResult]:
        """解析 LLM 的检索响应。"""
        # 尝试从响应中提取 JSON
        json_str = response.strip()

        # 去除可能的 markdown 代码块标记
        if json_str.startswith("```"):
            lines = json_str.split("\n")
            # 去除首尾的 ``` 行
            start = 1 if lines[0].startswith("```") else 0
            end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
            json_str = "\n".join(lines[start:end]).strip()

        try:
            items = json.loads(json_str)
        except json.JSONDecodeError:
            # 如果 JSON 解析失败，回退到关键词搜索
            return []

        if not isinstance(items, list):
            return []

        results = []
        for item in items:
            if not isinstance(item, dict):
                continue

            node_id = item.get("node_id", "")
            relevance = float(item.get("relevance", 0))
            reasoning = item.get("reasoning", "")

            node = tree.find_node(node_id)
            if node is None:
                continue

            result = RetrievalResult(
                node_id=node.node_id,
                title=node.title,
                summary=node.summary,
                score=max(0.0, min(1.0, relevance)),
                reasoning=reasoning,
                start_page=node.start_page,
                end_page=node.end_page,
                content=node.content,
            )
            results.append(result)

        # 按分数降序排列
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def _keyword_retrieve(
        self,
        query: str,
        tree: DocumentTree,
        top_k: int,
    ) -> List[RetrievalResult]:
        """使用关键词搜索进行检索。"""
        # 确保文档树已被索引
        self.keyword_search.index_tree(tree)

        # 执行搜索
        search_results = self.keyword_search.search(query, top_k=top_k)

        # 转换为 RetrievalResult
        results = []
        max_score = search_results[0][1] if search_results else 1.0

        for node_id, score, title in search_results:
            node = tree.find_node(node_id)
            if node is None:
                continue

            # 归一化分数到 0-1
            normalized_score = score / max_score if max_score > 0 else 0.0

            result = RetrievalResult(
                node_id=node.node_id,
                title=node.title,
                summary=node.summary,
                score=normalized_score,
                reasoning="关键词匹配",
                start_page=node.start_page,
                end_page=node.end_page,
                content=node.content,
            )
            results.append(result)

        return results
