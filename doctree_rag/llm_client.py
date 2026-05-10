"""
llm_client.py - 多后端 LLM 客户端

支持 OpenAI、DeepSeek、Anthropic Claude 和 Ollama 本地模型。
提供统一的调用接口，支持流式输出、重试机制和成本估算。
"""

import json
import os
import time
from typing import Any, Dict, Generator, List, Optional

from .utils import LLMError, ConfigError, count_tokens_approx


# ============================================================
# 模型配置信息
# ============================================================

MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    # OpenAI 模型
    "gpt-4o": {"provider": "openai", "max_input": 128000, "cost_per_1k_input": 0.0025, "cost_per_1k_output": 0.01},
    "gpt-4o-mini": {"provider": "openai", "max_input": 128000, "cost_per_1k_input": 0.00015, "cost_per_1k_output": 0.0006},
    "gpt-4-turbo": {"provider": "openai", "max_input": 128000, "cost_per_1k_input": 0.01, "cost_per_1k_output": 0.03},
    "gpt-3.5-turbo": {"provider": "openai", "max_input": 16385, "cost_per_1k_input": 0.0005, "cost_per_1k_output": 0.0015},
    # DeepSeek 模型
    "deepseek-chat": {"provider": "deepseek", "max_input": 64000, "cost_per_1k_input": 0.00014, "cost_per_1k_output": 0.00028},
    "deepseek-reasoner": {"provider": "deepseek", "max_input": 64000, "cost_per_1k_input": 0.0004, "cost_per_1k_output": 0.0016},
    # Anthropic 模型
    "claude-3-5-sonnet-20241022": {"provider": "anthropic", "max_input": 200000, "cost_per_1k_input": 0.003, "cost_per_1k_output": 0.015},
    "claude-3-haiku-20240307": {"provider": "anthropic", "max_input": 200000, "cost_per_1k_input": 0.00025, "cost_per_1k_output": 0.00125},
    "claude-3-sonnet-20240229": {"provider": "anthropic", "max_input": 200000, "cost_per_1k_input": 0.003, "cost_per_1k_output": 0.015},
    # Ollama 本地模型（成本为 0）
    "llama3": {"provider": "ollama", "max_input": 8192, "cost_per_1k_input": 0, "cost_per_1k_output": 0},
    "qwen2": {"provider": "ollama", "max_input": 8192, "cost_per_1k_input": 0, "cost_per_1k_output": 0},
    "mistral": {"provider": "ollama", "max_input": 8192, "cost_per_1k_input": 0, "cost_per_1k_output": 0},
    "gemma2": {"provider": "ollama", "max_input": 8192, "cost_per_1k_input": 0, "cost_per_1k_output": 0},
}


def get_model_info(model_name: str) -> Dict[str, Any]:
    """获取模型配置信息。如果模型不在预设列表中，返回默认值。"""
    if model_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_name].copy()
    return {
        "provider": "unknown",
        "max_input": 4096,
        "cost_per_1k_input": 0,
        "cost_per_1k_output": 0,
    }


# ============================================================
# 基础 LLM 客户端
# ============================================================

class LLMClient:
    """统一的 LLM 客户端接口。

    支持多后端切换，提供 generate()、generate_stream() 和 count_tokens() 方法。
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        api_key: str = "",
        base_url: str = "",
        temperature: float = 0.3,
        max_tokens: int = 2048,
        timeout: int = 120,
        max_retries: int = 3,
    ) -> None:
        """初始化 LLM 客户端。

        Args:
            provider: LLM 提供商 (openai/deepseek/anthropic/ollama)
            model: 模型名称
            api_key: API 密钥（Ollama 不需要）
            base_url: 自定义 API 基础 URL
            temperature: 生成温度
            max_tokens: 最大生成 token 数
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
        """
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key or self._get_env_api_key()
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.model_info = get_model_info(model)

        # 验证配置
        self._validate_config()

    def _get_env_api_key(self) -> str:
        """从环境变量获取 API 密钥。"""
        env_map = {
            "openai": ["OPENAI_API_KEY"],
            "deepseek": ["DEEPSEEK_API_KEY", "OPENAI_API_KEY"],
            "anthropic": ["ANTHROPIC_API_KEY"],
            "ollama": [],
        }
        env_vars = env_map.get(self.provider, [])
        for var in env_vars:
            value = os.environ.get(var, "")
            if value:
                return value
        return ""

    def _validate_config(self) -> None:
        """验证客户端配置是否正确。"""
        if self.provider not in ("openai", "deepseek", "anthropic", "ollama"):
            raise ConfigError(f"不支持的 LLM 提供商: {self.provider}")

        if self.provider != "ollama" and not self.api_key:
            raise ConfigError(
                f"{self.provider} 需要 API 密钥。请设置环境变量或通过配置文件提供。"
            )

    def _get_base_url(self) -> str:
        """获取 API 基础 URL。"""
        if self.base_url:
            return self.base_url.rstrip("/")

        default_urls = {
            "openai": "https://api.openai.com/v1",
            "deepseek": "https://api.deepseek.com/v1",
            "anthropic": "https://api.anthropic.com",
            "ollama": "http://localhost:11434",
        }
        return default_urls.get(self.provider, "")

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """生成文本回复。

        Args:
            prompt: 用户提示
            system_prompt: 系统提示
            temperature: 生成温度（覆盖默认值）
            max_tokens: 最大 token 数（覆盖默认值）

        Returns:
            生成的文本内容
        """
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        if self.provider in ("openai", "deepseek"):
            return self._generate_openai_compatible(prompt, system_prompt, temp, max_tok)
        elif self.provider == "anthropic":
            return self._generate_anthropic(prompt, system_prompt, temp, max_tok)
        elif self.provider == "ollama":
            return self._generate_ollama(prompt, system_prompt, temp, max_tok)
        else:
            raise LLMError(f"不支持的提供商: {self.provider}")

    def generate_stream(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Generator[str, None, None]:
        """流式生成文本回复。

        Yields:
            每次生成的文本片段
        """
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        if self.provider in ("openai", "deepseek"):
            yield from self._stream_openai_compatible(prompt, system_prompt, temp, max_tok)
        elif self.provider == "anthropic":
            yield from self._stream_anthropic(prompt, system_prompt, temp, max_tok)
        elif self.provider == "ollama":
            yield from self._stream_ollama(prompt, system_prompt, temp, max_tok)
        else:
            raise LLMError(f"不支持的提供商: {self.provider}")

    def count_tokens(self, text: str) -> int:
        """估算文本的 token 数量。"""
        return count_tokens_approx(text)

    def estimate_cost(self, input_text: str, output_text: str = "") -> float:
        """估算 API 调用成本（美元）。"""
        input_tokens = self.count_tokens(input_text)
        output_tokens = self.count_tokens(output_text) if output_text else 0
        input_cost = (input_tokens / 1000) * self.model_info["cost_per_1k_input"]
        output_cost = (output_tokens / 1000) * self.model_info["cost_per_1k_output"]
        return input_cost + output_cost

    def _retry_request(self, request_fn) -> Any:
        """带指数退避的重试机制。"""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return request_fn()
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = (2 ** attempt) * 1.0
                    time.sleep(wait_time)
        raise LLMError(f"LLM 请求失败（已重试 {self.max_retries} 次）: {last_error}")

    # --------------------------------------------------------
    # OpenAI 兼容接口（OpenAI / DeepSeek）
    # --------------------------------------------------------

    def _generate_openai_compatible(
        self, prompt: str, system_prompt: str, temperature: float, max_tokens: int
    ) -> str:
        """通过 OpenAI 兼容 API 生成文本。"""
        try:
            import urllib.request
            import urllib.error
        except ImportError:
            raise LLMError("需要 urllib 模块")

        base_url = self._get_base_url()
        url = f"{base_url}/chat/completions"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        def _request():
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            req.add_header("Content-Type", "application/json")

            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
                    return result["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace")
                raise LLMError(f"HTTP {e.code}: {body}")

        return self._retry_request(_request)

    def _stream_openai_compatible(
        self, prompt: str, system_prompt: str, temperature: float, max_tokens: int
    ) -> Generator[str, None, None]:
        """通过 OpenAI 兼容 API 流式生成文本。"""
        try:
            import urllib.request
        except ImportError:
            raise LLMError("需要 urllib 模块")

        base_url = self._get_base_url()
        url = f"{base_url}/chat/completions"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            for line in resp:
                line_str = line.decode("utf-8").strip()
                if not line_str or not line_str.startswith("data: "):
                    continue
                data_str = line_str[6:]
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield content
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

    # --------------------------------------------------------
    # Anthropic 接口
    # --------------------------------------------------------

    def _generate_anthropic(
        self, prompt: str, system_prompt: str, temperature: float, max_tokens: int
    ) -> str:
        """通过 Anthropic API 生成文本。"""
        try:
            import urllib.request
            import urllib.error
        except ImportError:
            raise LLMError("需要 urllib 模块")

        base_url = self._get_base_url()
        url = f"{base_url}/v1/messages"

        payload: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            payload["system"] = system_prompt

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }

        def _request():
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
                    return result["content"][0]["text"]
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace")
                raise LLMError(f"HTTP {e.code}: {body}")

        return self._retry_request(_request)

    def _stream_anthropic(
        self, prompt: str, system_prompt: str, temperature: float, max_tokens: int
    ) -> Generator[str, None, None]:
        """通过 Anthropic API 流式生成文本。"""
        try:
            import urllib.request
        except ImportError:
            raise LLMError("需要 urllib 模块")

        base_url = self._get_base_url()
        url = f"{base_url}/v1/messages"

        payload: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }
        if system_prompt:
            payload["system"] = system_prompt

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            for line in resp:
                line_str = line.decode("utf-8").strip()
                if not line_str or not line_str.startswith("data: "):
                    continue
                data_str = line_str[6:]
                try:
                    chunk = json.loads(data_str)
                    if chunk.get("type") == "content_block_delta":
                        delta = chunk.get("delta", {})
                        text = delta.get("text", "")
                        if text:
                            yield text
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

    # --------------------------------------------------------
    # Ollama 本地模型接口
    # --------------------------------------------------------

    def _generate_ollama(
        self, prompt: str, system_prompt: str, temperature: float, max_tokens: int
    ) -> str:
        """通过 Ollama 本地 API 生成文本。"""
        try:
            import urllib.request
            import urllib.error
        except ImportError:
            raise LLMError("需要 urllib 模块")

        base_url = self._get_base_url()
        url = f"{base_url}/api/generate"

        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "temperature": temperature,
            "options": {"num_predict": max_tokens},
            "stream": False,
        }

        def _request():
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data, method="POST")
            req.add_header("Content-Type", "application/json")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
                    return result.get("response", "")
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace")
                raise LLMError(f"HTTP {e.code}: {body}")
            except urllib.error.URLError as e:
                raise LLMError(f"无法连接到 Ollama 服务 ({base_url}): {e}")

        return self._retry_request(_request)

    def _stream_ollama(
        self, prompt: str, system_prompt: str, temperature: float, max_tokens: int
    ) -> Generator[str, None, None]:
        """通过 Ollama 本地 API 流式生成文本。"""
        try:
            import urllib.request
        except ImportError:
            raise LLMError("需要 urllib 模块")

        base_url = self._get_base_url()
        url = f"{base_url}/api/generate"

        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "temperature": temperature,
            "options": {"num_predict": max_tokens},
            "stream": True,
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")

        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            for line in resp:
                line_str = line.decode("utf-8").strip()
                if not line_str:
                    continue
                try:
                    chunk = json.loads(line_str)
                    text = chunk.get("response", "")
                    if text:
                        yield text
                    if chunk.get("done", False):
                        break
                except (json.JSONDecodeError, KeyError):
                    continue

    def is_available(self) -> bool:
        """检查 LLM 服务是否可用。"""
        try:
            result = self.generate("Hello", max_tokens=10)
            return bool(result)
        except Exception:
            return False

    def __repr__(self) -> str:
        return f"LLMClient(provider={self.provider!r}, model={self.model!r})"
