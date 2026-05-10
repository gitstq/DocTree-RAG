"""
cli.py - 命令行接口入口

使用 argparse 提供完整的 CLI 命令集，支持文档索引、搜索、浏览、导出等功能。
"""

import argparse
import json
import os
import sys
from typing import List, Optional

from . import __version__
from .indexer import DocumentIndexer
from .retriever import TreeRetriever
from .keyword_search import BM25Search
from .llm_client import LLMClient
from .tui import TreeBrowser
from .exporter import TreeExporter
from .utils import (
    ConfigError,
    DocTreeError,
    IndexNotFoundError,
    ProgressCallback,
    create_console_progress_callback,
    ensure_directory,
    get_default_config_path,
    get_default_index_dir,
    load_config,
    save_config,
)


def create_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="doctree",
        description="DocTree-RAG: 轻量级文档树索引与推理检索引擎",
        epilog="示例:\n"
               "  doctree index document.pdf\n"
               "  doctree search \"机器学习基础概念\"\n"
               "  doctree browse\n"
               "  doctree export --format markdown --output tree.md\n"
               "  doctree config --provider openai --model gpt-4o-mini",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    # 全局选项
    parser.add_argument(
        "--provider",
        choices=["openai", "deepseek", "anthropic", "ollama"],
        default=None,
        help="LLM 提供商",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="模型名称",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="输出目录",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="详细输出",
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # --------------------------------------------------
    # index 命令
    # --------------------------------------------------
    index_parser = subparsers.add_parser(
        "index",
        help="为文档构建树索引",
        description="解析文档并构建层级树索引结构",
    )
    index_parser.add_argument(
        "file_or_dir",
        help="文件或目录路径",
    )
    index_parser.add_argument(
        "--no-summary",
        action="store_true",
        default=False,
        help="跳过 LLM 摘要生成",
    )
    index_parser.add_argument(
        "--no-recursive",
        action="store_true",
        default=False,
        help="不递归处理子目录",
    )
    index_parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="最大树深度",
    )

    # --------------------------------------------------
    # search 命令
    # --------------------------------------------------
    search_parser = subparsers.add_parser(
        "search",
        help="搜索已索引的文档",
        description="在已索引的文档中搜索相关内容",
    )
    search_parser.add_argument(
        "query",
        help="搜索查询",
    )
    search_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="返回前 k 个结果（默认: 5）",
    )
    search_parser.add_argument(
        "--doc-id",
        default=None,
        help="限定在特定文档中搜索",
    )
    search_parser.add_argument(
        "--keyword-only",
        action="store_true",
        default=False,
        help="仅使用关键词搜索（不使用 LLM）",
    )

    # --------------------------------------------------
    # browse 命令
    # --------------------------------------------------
    browse_parser = subparsers.add_parser(
        "browse",
        help="交互式浏览文档树",
        description="启动交互式终端界面浏览已索引的文档树",
    )

    # --------------------------------------------------
    # export 命令
    # --------------------------------------------------
    export_parser = subparsers.add_parser(
        "export",
        help="导出文档树",
        description="将文档树导出为 JSON、Markdown 或 HTML 格式",
    )
    export_parser.add_argument(
        "--format",
        choices=["json", "markdown", "html"],
        default="json",
        help="输出格式（默认: json）",
    )
    export_parser.add_argument(
        "--output",
        default=None,
        help="输出文件路径",
    )
    export_parser.add_argument(
        "--doc-id",
        default=None,
        help="指定要导出的文档 ID（不指定则导出所有）",
    )

    # --------------------------------------------------
    # list 命令
    # --------------------------------------------------
    list_parser = subparsers.add_parser(
        "list",
        help="列出所有已索引的文档",
        description="显示所有已索引文档的基本信息",
    )
    list_parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        dest="json_output",
        help="以 JSON 格式输出",
    )

    # --------------------------------------------------
    # info 命令
    # --------------------------------------------------
    info_parser = subparsers.add_parser(
        "info",
        help="显示文档树详细信息",
        description="显示指定文档的树结构和统计信息",
    )
    info_parser.add_argument(
        "doc_id",
        help="文档 ID",
    )
    info_parser.add_argument(
        "--format",
        choices=["text", "json", "markdown"],
        default="text",
        help="输出格式（默认: text）",
    )

    # --------------------------------------------------
    # config 命令
    # --------------------------------------------------
    config_parser = subparsers.add_parser(
        "config",
        help="配置 LLM 设置",
        description="查看或修改 LLM 配置",
    )
    config_parser.add_argument(
        "--provider",
        choices=["openai", "deepseek", "anthropic", "ollama"],
        default=None,
        help="设置 LLM 提供商",
    )
    config_parser.add_argument(
        "--model",
        default=None,
        help="设置模型名称",
    )
    config_parser.add_argument(
        "--api-key",
        default=None,
        help="设置 API 密钥",
    )
    config_parser.add_argument(
        "--base-url",
        default=None,
        help="设置 API 基础 URL",
    )
    config_parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="设置生成温度",
    )
    config_parser.add_argument(
        "--index-dir",
        default=None,
        help="设置索引存储目录",
    )
    config_parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="显示当前配置",
    )
    config_parser.add_argument(
        "--reset",
        action="store_true",
        default=False,
        help="重置为默认配置",
    )

    return parser


def _create_llm_client(
    args: argparse.Namespace,
    config: dict,
) -> Optional[LLMClient]:
    """根据参数和配置创建 LLM 客户端。"""
    provider = args.provider or config.get("provider", "openai")
    model = args.model or config.get("model", "gpt-4o-mini")
    api_key = config.get("api_key", "")
    base_url = config.get("base_url", "")
    temperature = config.get("temperature", 0.3)
    max_tokens = config.get("max_tokens", 2048)

    try:
        client = LLMClient(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return client
    except ConfigError as e:
        print(f"警告: 无法创建 LLM 客户端: {e}", file=sys.stderr)
        return None


def _create_indexer(
    args: argparse.Namespace,
    config: dict,
    llm_client: Optional[LLMClient],
) -> DocumentIndexer:
    """创建文档索引器。"""
    index_dir = args.output or config.get("index_dir", "") or get_default_index_dir()
    no_summary = getattr(args, "no_summary", False)

    return DocumentIndexer(
        llm_client=llm_client if not no_summary else None,
        max_pages_per_node=config.get("max_pages_per_node", 5),
        max_tokens_per_node=config.get("max_tokens_per_node", 2000),
        index_dir=index_dir,
        generate_summaries=not no_summary,
    )


def cmd_index(args: argparse.Namespace, config: dict) -> int:
    """执行 index 命令。"""
    llm_client = _create_llm_client(args, config)
    indexer = _create_indexer(args, config, llm_client)

    progress = ProgressCallback(description="开始索引")
    if args.verbose:
        progress.add_callback(create_console_progress_callback(args.verbose))

    file_or_dir = args.file_or_dir

    if os.path.isfile(file_or_dir):
        try:
            tree = indexer.index_file(file_or_dir, progress_callback=progress)
            print(f"索引完成: {tree.doc_name}")
            print(f"  文档 ID: {tree.doc_id}")
            print(f"  节点数: {tree.get_total_nodes()}")
            print(f"  树深度: {tree.get_depth()}")
            return 0
        except DocTreeError as e:
            print(f"错误: {e}", file=sys.stderr)
            return 1

    elif os.path.isdir(file_or_dir):
        try:
            trees = indexer.index_directory(
                file_or_dir,
                recursive=not getattr(args, "no_recursive", False),
                progress_callback=progress,
            )
            print(f"索引完成: 共 {len(trees)} 个文档")
            for tree in trees:
                print(f"  - {tree.doc_name} ({tree.get_total_nodes()} 节点)")
            return 0
        except DocTreeError as e:
            print(f"错误: {e}", file=sys.stderr)
            return 1
    else:
        print(f"错误: 路径不存在: {file_or_dir}", file=sys.stderr)
        return 1


def cmd_search(args: argparse.Namespace, config: dict) -> int:
    """执行 search 命令。"""
    index_dir = args.output or config.get("index_dir", "") or get_default_index_dir()
    indexer = DocumentIndexer(index_dir=index_dir)

    # 加载所有索引
    trees = indexer.load_all_indexes()
    if not trees:
        print("错误: 没有已索引的文档。请先使用 'doctree index' 索引文档。", file=sys.stderr)
        return 1

    # 如果指定了文档 ID，过滤
    if args.doc_id:
        trees = [t for t in trees if t.doc_id == args.doc_id]
        if not trees:
            print(f"错误: 未找到文档 ID: {args.doc_id}", file=sys.stderr)
            return 1

    # 创建检索器
    llm_client = None
    if not args.keyword_only:
        llm_client = _create_llm_client(args, config)

    keyword_search = BM25Search()
    for tree in trees:
        keyword_search.index_tree(tree)

    retriever = TreeRetriever(
        llm_client=llm_client,
        keyword_search=keyword_search,
        top_k=args.top_k,
    )

    # 执行检索
    all_results = []
    for tree in trees:
        results = retriever.retrieve(args.query, tree, top_k=args.top_k)
        for r in results:
            r.reasoning = f"[{tree.doc_name}] {r.reasoning}"
        all_results.extend(results)

    all_results.sort(key=lambda r: r.score, reverse=True)
    all_results = all_results[: args.top_k]

    if not all_results:
        print("未找到相关结果。")
        return 0

    # 输出结果
    print(f"搜索: {args.query}")
    print(f"找到 {len(all_results)} 个相关结果:\n")

    for i, result in enumerate(all_results):
        print(f"[{i + 1}] {result.title}")
        print(f"    相关度: {result.score:.3f}")
        if result.start_page is not None and result.end_page is not None:
            print(f"    页码: {result.start_page}-{result.end_page}")
        if result.summary:
            print(f"    摘要: {result.summary[:200]}")
        print(f"    推理: {result.reasoning}")
        print()

    return 0


def cmd_browse(args: argparse.Namespace, config: dict) -> int:
    """执行 browse 命令。"""
    index_dir = args.output or config.get("index_dir", "") or get_default_index_dir()
    indexer = DocumentIndexer(index_dir=index_dir)

    trees = indexer.load_all_indexes()
    if not trees:
        print("没有可浏览的文档。请先使用 'doctree index' 索引文档。")
        return 1

    # 创建检索器（可选）
    llm_client = _create_llm_client(args, config)
    keyword_search = BM25Search()
    for tree in trees:
        keyword_search.index_tree(tree)

    retriever = TreeRetriever(
        llm_client=llm_client,
        keyword_search=keyword_search,
    )

    browser = TreeBrowser(trees=trees, retriever=retriever)
    browser.browse()
    return 0


def cmd_export(args: argparse.Namespace, config: dict) -> int:
    """执行 export 命令。"""
    index_dir = args.output or config.get("index_dir", "") or get_default_index_dir()
    indexer = DocumentIndexer(index_dir=index_dir)

    trees = indexer.load_all_indexes()
    if not trees:
        print("错误: 没有已索引的文档。", file=sys.stderr)
        return 1

    if args.doc_id:
        trees = [t for t in trees if t.doc_id == args.doc_id]
        if not trees:
            print(f"错误: 未找到文档 ID: {args.doc_id}", file=sys.stderr)
            return 1

    exporter = TreeExporter()

    for tree in trees:
        output_path = args.output or ""
        if output_path and len(trees) > 1:
            # 多个文档时，自动添加文件名
            base, ext = os.path.splitext(output_path)
            output_path = f"{base}_{tree.doc_id}{ext}"

        content = exporter.export(
            tree,
            format=args.format,
            output_path=output_path,
        )

        if not output_path:
            print(content)

    return 0


def cmd_list(args: argparse.Namespace, config: dict) -> int:
    """执行 list 命令。"""
    index_dir = args.output or config.get("index_dir", "") or get_default_index_dir()
    indexer = DocumentIndexer(index_dir=index_dir)

    trees = indexer.load_all_indexes()
    if not trees:
        print("没有已索引的文档。")
        return 0

    if args.json_output:
        data = []
        for tree in trees:
            data.append({
                "doc_id": tree.doc_id,
                "doc_name": tree.doc_name,
                "doc_path": tree.doc_path,
                "total_nodes": tree.get_total_nodes(),
                "depth": tree.get_depth(),
                "created_at": tree.created_at,
                "updated_at": tree.updated_at,
            })
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print(f"已索引 {len(trees)} 个文档:\n")
        print(f"{'编号':<6} {'文档名称':<30} {'节点数':<8} {'深度':<6} {'创建时间'}")
        print("-" * 70)
        for i, tree in enumerate(trees):
            created = tree.created_at[:19] if tree.created_at else "N/A"
            print(f"{i + 1:<6} {tree.doc_name:<30} {tree.get_total_nodes():<8} {tree.get_depth():<6} {created}")

    return 0


def cmd_info(args: argparse.Namespace, config: dict) -> int:
    """执行 info 命令。"""
    index_dir = args.output or config.get("index_dir", "") or get_default_index_dir()
    indexer = DocumentIndexer(index_dir=index_dir)

    tree = indexer._load_index(args.doc_id)
    if not tree:
        print(f"错误: 未找到文档 ID: {args.doc_id}", file=sys.stderr)
        return 1

    if args.format == "json":
        print(tree.to_json())
    elif args.format == "markdown":
        print(tree.to_markdown())
    else:
        print(f"文档: {tree.doc_name}")
        print(f"ID: {tree.doc_id}")
        print(f"路径: {tree.doc_path}")
        print(f"创建时间: {tree.created_at}")
        print(f"更新时间: {tree.updated_at}")
        print(f"节点总数: {tree.get_total_nodes()}")
        print(f"树深度: {tree.get_depth()}")
        print(f"叶子节点数: {len(tree.get_leaf_nodes())}")
        print()

        # 显示树结构
        def _print_node(node, indent=0):
            prefix = "  " * indent + ("+-- " if indent > 0 else "")
            page = ""
            if node.start_page is not None and node.end_page is not None:
                page = f" [页 {node.start_page}-{node.end_page}]"
            children = f" ({len(node.children)} 子节点)" if node.children else ""
            print(f"{prefix}{node.title}{page}{children}")
            if node.summary:
                s = node.summary[:100] + "..." if len(node.summary) > 100 else node.summary
                print(f"{'  ' * (indent + 1)}摘要: {s}")
            for child in node.children:
                _print_node(child, indent + 1)

        for root in tree.root_nodes:
            _print_node(root)

    return 0


def cmd_config(args: argparse.Namespace, config: dict) -> int:
    """执行 config 命令。"""
    config_path = get_default_config_path()

    if args.reset:
        from .utils import DEFAULT_CONFIG
        config = DEFAULT_CONFIG.copy()
        save_config(config, config_path)
        print("配置已重置为默认值。")
        return 0

    # 更新配置
    changed = False
    if args.provider is not None:
        config["provider"] = args.provider
        changed = True
    if args.model is not None:
        config["model"] = args.model
        changed = True
    if args.api_key is not None:
        config["api_key"] = args.api_key
        changed = True
    if args.base_url is not None:
        config["base_url"] = args.base_url
        changed = True
    if args.temperature is not None:
        config["temperature"] = args.temperature
        changed = True
    if args.index_dir is not None:
        config["index_dir"] = args.index_dir
        changed = True

    if changed:
        save_config(config, config_path)
        print(f"配置已保存到: {config_path}")

    # 显示配置
    if args.show or not changed:
        print("当前配置:")
        print(f"  提供商: {config.get('provider', 'N/A')}")
        print(f"  模型: {config.get('model', 'N/A')}")
        print(f"  API 密钥: {'***' + config.get('api_key', '')[-4:] if config.get('api_key') else '(未设置)'}")
        print(f"  基础 URL: {config.get('base_url', '(默认)') or '(默认)'}")
        print(f"  温度: {config.get('temperature', 0.3)}")
        print(f"  最大 Token: {config.get('max_tokens', 2048)}")
        print(f"  索引目录: {config.get('index_dir', '') or get_default_index_dir()}")
        print(f"  配置文件: {config_path}")

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """CLI 主入口函数。"""
    parser = create_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    # 加载配置
    try:
        config = load_config()
    except ConfigError as e:
        print(f"配置加载失败: {e}", file=sys.stderr)
        config = {}

    # 分发命令
    command_map = {
        "index": cmd_index,
        "search": cmd_search,
        "browse": cmd_browse,
        "export": cmd_export,
        "list": cmd_list,
        "info": cmd_info,
        "config": cmd_config,
    }

    handler = command_map.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    try:
        return handler(args, config)
    except KeyboardInterrupt:
        print("\n操作已取消。")
        return 130
    except DocTreeError as e:
        print(f"错误: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        if args.verbose:
            import traceback
            traceback.print_exc()
        else:
            print(f"错误: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
