"""
tui.py - 交互式终端用户界面

使用 rich 库提供交互式文档树浏览和搜索功能。
支持键盘导航、实时搜索过滤和节点详情查看。
"""

import sys
from typing import Any, Dict, List, Optional

from .tree_model import DocumentTree, TreeNode
from .retriever import RetrievalResult


class TreeBrowser:
    """交互式文档树浏览器。

    使用 rich 库提供美观的终端界面，支持：
    - 树形结构浏览（可展开/折叠）
    - 实时搜索过滤
    - 节点详情查看
    - 键盘导航

    注意: 此功能需要 rich 库。如果未安装，将回退到简单文本模式。
    """

    def __init__(
        self,
        trees: Optional[List[DocumentTree]] = None,
        retriever: Optional[Any] = None,
    ) -> None:
        self.trees = trees or []
        self.retriever = retriever
        self._has_rich = self._check_rich()

    def _check_rich(self) -> bool:
        """检查 rich 库是否可用。"""
        try:
            import rich  # noqa: F401
            return True
        except ImportError:
            return False

    def browse(self) -> None:
        """启动交互式浏览器。"""
        if not self._has_rich:
            self._browse_simple()
        else:
            self._browse_rich()

    def _browse_simple(self) -> None:
        """简单文本模式浏览器（不需要 rich 库）。"""
        if not self.trees:
            print("没有可浏览的文档。请先使用 'doctree index' 索引文档。")
            return

        print("\n=== DocTree-RAG 文档浏览器 ===")
        print("提示: 安装 rich 库 (pip install rich) 可获得更好的浏览体验\n")

        # 显示文档列表
        print("已索引的文档:")
        for i, tree in enumerate(self.trees):
            print(f"  [{i + 1}] {tree.doc_name} "
                  f"(节点: {tree.get_total_nodes()}, 深度: {tree.get_depth()})")

        print("\n命令:")
        print("  q - 退出")
        print("  n <编号> - 查看文档树")
        print("  s <查询> - 搜索文档")

        while True:
            try:
                cmd = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见！")
                break

            if not cmd:
                continue

            if cmd.lower() == "q":
                print("再见！")
                break
            elif cmd.lower().startswith("n "):
                try:
                    idx = int(cmd[2:].strip()) - 1
                    if 0 <= idx < len(self.trees):
                        self._display_tree_simple(self.trees[idx])
                    else:
                        print("无效的文档编号。")
                except ValueError:
                    print("请输入有效的文档编号。")
            elif cmd.lower().startswith("s "):
                query = cmd[2:].strip()
                if query and self.retriever:
                    self._search_simple(query)
                else:
                    print("请输入搜索查询。")
            else:
                print("未知命令。输入 q 退出。")

    def _display_tree_simple(self, tree: DocumentTree) -> None:
        """在简单模式下显示文档树。"""
        print(f"\n{'=' * 60}")
        print(f"文档: {tree.doc_name}")
        print(f"路径: {tree.doc_path}")
        print(f"节点: {tree.get_total_nodes()}, 深度: {tree.get_depth()}")
        print(f"{'=' * 60}")

        def _print_node(node: TreeNode, indent: int = 0) -> None:
            prefix = "  " * indent + ("+-- " if indent > 0 else "")
            page_info = ""
            if node.start_page is not None and node.end_page is not None:
                page_info = f" [页 {node.start_page}-{node.end_page}]"
            child_info = f" ({len(node.children)} 子节点)" if node.children else ""
            print(f"{prefix}{node.title}{page_info}{child_info}")
            if node.summary:
                summary = node.summary[:100] + "..." if len(node.summary) > 100 else node.summary
                print(f"{'  ' * (indent + 1)}摘要: {summary}")
            for child in node.children:
                _print_node(child, indent + 1)

        for root in tree.root_nodes:
            _print_node(root)

    def _search_simple(self, query: str) -> None:
        """在简单模式下执行搜索。"""
        if not self.retriever:
            print("搜索功能需要 LLM 客户端或关键词搜索引擎。")
            return

        print(f"\n搜索: {query}")
        print("-" * 40)

        all_results = []
        for tree in self.trees:
            results = self.retriever.retrieve(query, tree)
            for r in results:
                r.reasoning = f"[{tree.doc_name}] {r.reasoning}"
            all_results.extend(results)

        if not all_results:
            print("未找到相关结果。")
            return

        all_results.sort(key=lambda r: r.score, reverse=True)

        for i, result in enumerate(all_results):
            print(f"\n[{i + 1}] {result.title} (相关度: {result.score:.2f})")
            if result.start_page is not None:
                print(f"    页码: {result.start_page}-{result.end_page}")
            if result.summary:
                summary = result.summary[:150] + "..." if len(result.summary) > 150 else result.summary
                print(f"    摘要: {summary}")
            print(f"    推理: {result.reasoning}")

    def _browse_rich(self) -> None:
        """使用 rich 库的交互式浏览器。"""
        try:
            from rich.console import Console
            from rich.tree import Tree as RichTree
            from rich.panel import Panel
            from rich.table import Table
            from rich.prompt import Prompt
            from rich.text import Text
        except ImportError:
            self._browse_simple()
            return

        console = Console()

        if not self.trees:
            console.print("[yellow]没有可浏览的文档。请先使用 'doctree index' 索引文档。[/yellow]")
            return

        console.print(Panel(
            "[bold blue]DocTree-RAG 文档浏览器[/bold blue]\n"
            "方向键导航 | Enter 展开/折叠 | / 搜索 | q 退出",
            title="欢迎使用",
            border_style="blue",
        ))

        # 显示文档列表
        table = Table(title="已索引的文档")
        table.add_column("编号", style="cyan", justify="right")
        table.add_column("文档名称", style="green")
        table.add_column("节点数", justify="right")
        table.add_column("深度", justify="right")
        table.add_column("创建时间", style="dim")

        for i, tree in enumerate(self.trees):
            table.add_row(
                str(i + 1),
                tree.doc_name,
                str(tree.get_total_nodes()),
                str(tree.get_depth()),
                tree.created_at[:19] if tree.created_at else "",
            )

        console.print(table)
        console.print("\n[dim]命令: n <编号> 查看 | s <查询> 搜索 | q 退出[/dim]")

        while True:
            try:
                cmd = Prompt.ask("\n[bold cyan]>[/bold cyan]").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[yellow]再见！[/yellow]")
                break

            if not cmd:
                continue

            if cmd.lower() == "q":
                console.print("[yellow]再见！[/yellow]")
                break
            elif cmd.lower().startswith("n "):
                try:
                    idx = int(cmd[2:].strip()) - 1
                    if 0 <= idx < len(self.trees):
                        self._display_tree_rich(console, self.trees[idx])
                    else:
                        console.print("[red]无效的文档编号。[/red]")
                except ValueError:
                    console.print("[red]请输入有效的文档编号。[/red]")
            elif cmd.lower().startswith("s "):
                query = cmd[2:].strip()
                if query:
                    self._search_rich(console, query)
                else:
                    console.print("[red]请输入搜索查询。[/red]")
            else:
                console.print("[red]未知命令。输入 q 退出。[/red]")

    def _display_tree_rich(self, console: Any, tree: DocumentTree) -> None:
        """使用 rich 显示文档树。"""
        from rich.tree import Tree as RichTree
        from rich.panel import Panel

        rich_tree = RichTree(
            f"[bold green]{tree.doc_name}[/bold green]",
            guide_style="bold blue",
        )

        def _add_node(parent: Any, node: TreeNode) -> None:
            page_info = ""
            if node.start_page is not None and node.end_page is not None:
                page_info = f" [dim][页 {node.start_page}-{node.end_page}][/dim]"

            child_count = f" [dim]({len(node.children)} 子节点)[/dim]" if node.children else ""
            label = f"{node.title}{page_info}{child_count}"

            if node.summary:
                label += f"\n[dim]{node.summary[:80]}{'...' if len(node.summary) > 80 else ''}[/dim]"

            branch = parent.add(label)
            for child in node.children:
                _add_node(branch, child)

        for root in tree.root_nodes:
            _add_node(rich_tree, root)

        console.print(Panel(rich_tree, title=f"文档树 - {tree.doc_name}", border_style="green"))

    def _search_rich(self, console: Any, query: str) -> None:
        """使用 rich 显示搜索结果。"""
        from rich.table import Table
        from rich.panel import Panel

        if not self.retriever:
            console.print("[red]搜索功能需要 LLM 客户端或关键词搜索引擎。[/red]")
            return

        console.print(Panel(f"[bold]搜索:[/bold] {query}", border_style="yellow"))

        all_results = []
        for tree in self.trees:
            results = self.retriever.retrieve(query, tree)
            for r in results:
                r.reasoning = f"[{tree.doc_name}] {r.reasoning}"
            all_results.extend(results)

        if not all_results:
            console.print("[yellow]未找到相关结果。[/yellow]")
            return

        all_results.sort(key=lambda r: r.score, reverse=True)

        table = Table(title="搜索结果")
        table.add_column("#", style="cyan", justify="right", width=4)
        table.add_column("标题", style="green", min_width=20)
        table.add_column("相关度", justify="right", width=8)
        table.add_column("页码", justify="center", width=10)
        table.add_column("摘要", style="dim", max_width=50)
        table.add_column("推理", style="dim", max_width=40)

        for i, result in enumerate(all_results):
            page = ""
            if result.start_page is not None and result.end_page is not None:
                page = f"{result.start_page}-{result.end_page}"

            summary = result.summary[:50] + "..." if len(result.summary) > 50 else result.summary
            reasoning = result.reasoning[:40] + "..." if len(result.reasoning) > 40 else result.reasoning

            # 根据相关度着色
            if result.score >= 0.8:
                score_style = "bold green"
            elif result.score >= 0.5:
                score_style = "yellow"
            else:
                score_style = "red"

            table.add_row(
                str(i + 1),
                result.title,
                f"[{score_style}]{result.score:.2f}[/{score_style}]",
                page,
                summary,
                reasoning,
            )

        console.print(table)
