"""
utils.py - 工具函数集合

提供文档处理、文本操作、配置管理等通用工具函数。
"""

import hashlib
import json
import math
import os
import re
import sys
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


# ============================================================
# 自定义异常
# ============================================================

class DocTreeError(Exception):
    """DocTree-RAG 基础异常类。"""
    pass


class DocumentParseError(DocTreeError):
    """文档解析错误。"""
    pass


class LLMError(DocTreeError):
    """LLM 调用错误。"""
    pass


class ConfigError(DocTreeError):
    """配置错误。"""
    pass


class IndexNotFoundError(DocTreeError):
    """索引未找到错误。"""
    pass


# ============================================================
# 文本处理工具
# ============================================================

def normalize_text(text: str) -> str:
    """规范化文本：去除多余空白、统一换行符、去除零宽字符。"""
    # 去除零宽字符
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    # 统一换行符
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # 去除行首行尾空白
    lines = [line.strip() for line in text.split("\n")]
    # 合并连续空行为单个空行
    result_lines = []
    prev_empty = False
    for line in lines:
        if line == "":
            if not prev_empty:
                result_lines.append(line)
            prev_empty = True
        else:
            result_lines.append(line)
            prev_empty = False
    return "\n".join(result_lines)


def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
    """截断文本到指定长度。"""
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def count_tokens_approx(text: str) -> int:
    """粗略估算文本的 token 数量。

    对于英文，大约 1 token = 4 个字符。
    对于中文，大约 1 token = 1.5 个字符。
    这里使用混合估算策略。
    """
    if not text:
        return 0

    cjk_count = 0
    other_count = 0
    for char in text:
        cp = ord(char)
        # CJK 统一汉字范围
        if (
            (0x4E00 <= cp <= 0x9FFF)
            or (0x3400 <= cp <= 0x4DBF)
            or (0x20000 <= cp <= 0x2A6DF)
            or (0xF900 <= cp <= 0xFAFF)
        ):
            cjk_count += 1
        else:
            other_count += 1

    # 中文约 1.5 字符/token，其他约 4 字符/token
    cjk_tokens = math.ceil(cjk_count / 1.5)
    other_tokens = math.ceil(other_count / 4.0)
    return cjk_tokens + other_tokens


def split_text_by_length(
    text: str, max_length: int, overlap: int = 0
) -> List[str]:
    """按长度分割文本，支持重叠区域。"""
    if max_length <= 0:
        raise ValueError("max_length 必须大于 0")
    if overlap >= max_length:
        raise ValueError("overlap 必须小于 max_length")

    chunks = []
    start = 0
    while start < len(text):
        end = start + max_length
        # 尝试在句子边界处分割
        if end < len(text):
            # 查找最后一个句号、问号、感叹号或换行符
            last_break = max(
                text.rfind("。", start, end),
                text.rfind("！", start, end),
                text.rfind("？", start, end),
                text.rfind(".", start, end),
                text.rfind("!", start, end),
                text.rfind("?", start, end),
                text.rfind("\n", start, end),
            )
            if last_break > start:
                end = last_break + 1
        chunks.append(text[start:end])
        start = end - overlap if end < len(text) else end

    return chunks


def extract_headings(text: str) -> List[Tuple[int, str]]:
    """从 Markdown 文本中提取标题层级和标题文本。

    Returns:
        列表，每个元素为 (层级, 标题文本) 元组。
    """
    headings = []
    for line in text.split("\n"):
        line = line.strip()
        match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()
            headings.append((level, title))
    return headings


def clean_html_tags(text: str) -> str:
    """去除文本中的 HTML 标签。"""
    clean = re.sub(r"<[^>]+>", "", text)
    # 解码常见的 HTML 实体
    clean = clean.replace("&amp;", "&")
    clean = clean.replace("&lt;", "<")
    clean = clean.replace("&gt;", ">")
    clean = clean.replace("&quot;", '"')
    clean = clean.replace("&#39;", "'")
    clean = clean.replace("&nbsp;", " ")
    return clean


# ============================================================
# 文件和路径工具
# ============================================================

def compute_file_hash(file_path: str) -> str:
    """计算文件的 SHA256 哈希值。"""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            sha256.update(block)
    return sha256.hexdigest()[:16]


def compute_text_hash(text: str) -> str:
    """计算文本的 SHA256 哈希值。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def get_file_extension(file_path: str) -> str:
    """获取文件扩展名（小写，不含点号）。"""
    return Path(file_path).suffix.lower().lstrip(".")


def is_supported_file(file_path: str) -> bool:
    """检查文件是否为支持的文档格式。"""
    ext = get_file_extension(file_path)
    return ext in ("pdf", "md", "markdown", "txt", "html", "htm")


def ensure_directory(path: str) -> None:
    """确保目录存在，不存在则创建。"""
    Path(path).mkdir(parents=True, exist_ok=True)


def get_default_index_dir() -> str:
    """获取默认的索引存储目录。"""
    home = Path.home()
    return str(home / ".doctree" / "indexes")


def get_default_config_path() -> str:
    """获取默认的配置文件路径。"""
    home = Path.home()
    return str(home / ".doctree" / "config.json")


# ============================================================
# 配置管理
# ============================================================

DEFAULT_CONFIG = {
    "provider": "openai",
    "model": "gpt-4o-mini",
    "api_key": "",
    "base_url": "",
    "temperature": 0.3,
    "max_tokens": 2048,
    "index_dir": "",
    "max_pages_per_node": 5,
    "max_tokens_per_node": 2000,
}


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """加载配置文件。如果文件不存在，返回默认配置。"""
    path = config_path or get_default_config_path()
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                config = json.load(f)
            # 合并默认配置，确保所有键都存在
            merged = DEFAULT_CONFIG.copy()
            merged.update(config)
            return merged
        except (json.JSONDecodeError, IOError) as e:
            raise ConfigError(f"配置文件加载失败: {e}")
    return DEFAULT_CONFIG.copy()


def save_config(config: Dict[str, Any], config_path: Optional[str] = None) -> None:
    """保存配置到文件。"""
    path = config_path or get_default_config_path()
    ensure_directory(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


# ============================================================
# 进度回调
# ============================================================

class ProgressCallback:
    """进度回调工具类，用于报告长时间操作的处理进度。"""

    def __init__(self, total: int = 0, description: str = ""):
        self.total = total
        self.current = 0
        self.description = description
        self._callbacks: List[Callable[[int, int, str], None]] = []

    def add_callback(self, callback: Callable[[int, int, str], None]) -> None:
        """添加进度回调函数。回调签名: callback(current, total, description)"""
        self._callbacks.append(callback)

    def update(self, current: int, description: str = "") -> None:
        """更新进度。"""
        self.current = current
        if description:
            self.description = description
        for cb in self._callbacks:
            cb(self.current, self.total, self.description)

    def increment(self, step: int = 1, description: str = "") -> None:
        """递增进度。"""
        self.update(self.current + step, description)

    def finish(self) -> None:
        """标记任务完成。"""
        self.update(self.total, "完成")

    @property
    def progress_ratio(self) -> float:
        """获取进度比例 (0.0 ~ 1.0)。"""
        if self.total <= 0:
            return 0.0
        return min(self.current / self.total, 1.0)


def create_console_progress_callback(verbose: bool = False) -> Callable[[int, int, str], None]:
    """创建控制台进度回调函数。"""
    def callback(current: int, total: int, description: str) -> None:
        if not verbose:
            return
        if total > 0:
            ratio = current / total
            bar_len = 30
            filled = int(bar_len * ratio)
            bar = "=" * filled + "-" * (bar_len - filled)
            sys.stderr.write(
                f"\r[{bar}] {current}/{total} {description}"
            )
            if current >= total:
                sys.stderr.write("\n")
            sys.stderr.flush()
    return callback


# ============================================================
# 生成唯一 ID
# ============================================================

def generate_id(prefix: str = "node") -> str:
    """生成简短的唯一 ID。"""
    import uuid
    uid = uuid.uuid4().hex[:8]
    return f"{prefix}_{uid}"


def generate_doc_id(doc_name: str) -> str:
    """基于文档名称生成文档 ID。"""
    name_hash = hashlib.md5(doc_name.encode("utf-8")).hexdigest()[:8]
    return f"doc_{name_hash}"


# ============================================================
# 时间工具
# ============================================================

def now_iso() -> str:
    """获取当前时间的 ISO 格式字符串。"""
    return datetime.now().isoformat()


def format_timestamp(ts: str) -> str:
    """将 ISO 时间戳格式化为可读字符串。"""
    try:
        dt = datetime.fromisoformat(ts)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return ts
