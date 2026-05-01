"""LLM API 统一客户端 — 支持 DeepSeek V4 / OpenAI / Anthropic 多后端"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Provider → (env_var, base_url)
_PROVIDER_CONFIG = {
    "deepseek": ("DEEPSEEK_API_KEY", "https://api.deepseek.com"),
    "openai": ("OPENAI_API_KEY", None),       # None = SDK 默认
    "anthropic": ("ANTHROPIC_API_KEY", None),
}

# VLM Provider → (env_var, base_url)
_VLM_PROVIDER_CONFIG = {
    "qwen": ("DASHSCOPE_API_KEY", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    "openai": ("OPENAI_API_KEY", None),
    "anthropic": ("ANTHROPIC_API_KEY", None),
}


class LLMClient:
    """统一的文本 LLM 调用接口 (DeepSeek V4 / OpenAI / Anthropic)"""

    def __init__(
        self,
        provider: str = "deepseek",
        model: str = "deepseek-v4-pro",
        max_tokens: int = 4096,
        temperature: float = 1.0,
    ):
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client_cache: dict[str, object] = {}

    def chat(
        self,
        system_prompt: str,
        user_message: str,
        response_format: Optional[dict] = None,
        cache_system: bool = False,
    ) -> str:
        """发送聊天请求，返回文本响应"""
        if self.provider in ("deepseek", "openai"):
            return self._chat_openai_compat(system_prompt, user_message, response_format)
        elif self.provider == "anthropic":
            return self._chat_anthropic(system_prompt, user_message, cache_system)
        else:
            raise ValueError(f"不支持的 LLM provider: {self.provider}")

    def chat_json(
        self,
        system_prompt: str,
        user_message: str,
        cache_system: bool = False,
    ) -> dict:
        """发送聊天请求，返回解析后的 JSON"""
        text = self.chat(
            system_prompt=system_prompt,
            user_message=user_message,
            response_format={"type": "json_object"},
            cache_system=cache_system,
        )
        return _extract_json(text)

    def _get_openai_client(self, provider: str):
        """获取 OpenAI 兼容客户端（支持 DeepSeek 等自定义 base_url）"""
        import openai

        if provider not in self._client_cache:
            env_var, base_url = _PROVIDER_CONFIG.get(provider, (None, None))
            api_key = os.getenv(env_var, "")
            if not api_key:
                raise RuntimeError(f"{env_var} 未设置")

            kwargs = {"api_key": api_key}
            if base_url is not None:
                kwargs["base_url"] = base_url
            self._client_cache[provider] = openai.OpenAI(**kwargs)

        return self._client_cache[provider]

    def _chat_openai_compat(
        self, system_prompt: str, user_message: str, response_format: Optional[dict]
    ) -> str:
        """OpenAI 兼容格式（DeepSeek / OpenAI 通用）"""
        client = self._get_openai_client(self.provider)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": messages,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format

        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    def _chat_anthropic(self, system_prompt: str, user_message: str, cache_system: bool) -> str:
        import anthropic

        if "anthropic" not in self._client_cache:
            api_key = os.getenv("ANTHROPIC_API_KEY", "")
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY 未设置")
            self._client_cache["anthropic"] = anthropic.Anthropic(api_key=api_key)

        client = self._client_cache["anthropic"]

        system_kwargs: dict = {"text": system_prompt}
        if cache_system:
            system_kwargs["cache_control"] = {"type": "ephemeral"}

        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=[system_kwargs],
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text


class VLMClient:
    """统一的多模态 VLM 调用接口 (Qwen-VL / GPT-4o / Claude)

    Qwen-VL 通过阿里云 DashScope 的 OpenAI 兼容接口调用。
    官方文档: https://qwen.ai/apiplatform
    """

    def __init__(
        self,
        provider: str = "qwen",
        model: str = "qwen-vl-max",
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ):
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client_cache: dict[str, object] = {}

    def chat(
        self,
        system_prompt: str,
        image_base64_list: list[str],
        text_prompt: str,
    ) -> str:
        """多模态对话：图片 + 文本 → 文本响应

        Args:
            system_prompt: 系统提示
            image_base64_list: 图片的 base64 data URL 列表
            text_prompt: 附带的文本提示
        """
        if self.provider == "qwen":
            return self._chat_qwen(system_prompt, image_base64_list, text_prompt)
        elif self.provider == "openai":
            return self._chat_openai(system_prompt, image_base64_list, text_prompt)
        elif self.provider == "anthropic":
            return self._chat_anthropic(system_prompt, image_base64_list, text_prompt)
        else:
            raise ValueError(f"不支持的 VLM provider: {self.provider}")

    def _get_qwen_client(self):
        """获取 Qwen-VL / DashScope 客户端（OpenAI 兼容）"""
        import openai

        if "qwen" not in self._client_cache:
            api_key = os.getenv("DASHSCOPE_API_KEY", "")
            if not api_key:
                raise RuntimeError("DASHSCOPE_API_KEY 未设置，请从阿里云百炼控制台获取")
            self._client_cache["qwen"] = openai.OpenAI(
                api_key=api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
        return self._client_cache["qwen"]

    def _chat_qwen(
        self, system_prompt: str, image_base64_list: list[str], text_prompt: str
    ) -> str:
        """Qwen-VL 多模态调用"""
        client = self._get_qwen_client()

        # 构建多模态 content 数组
        content: list[dict] = []

        # 图片
        for b64_url in image_base64_list:
            content.append({
                "type": "image_url",
                "image_url": {"url": b64_url},
            })

        # 文本
        content.append({"type": "text", "text": text_prompt})

        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content or ""

    def _chat_openai(
        self, system_prompt: str, image_base64_list: list[str], text_prompt: str
    ) -> str:
        """GPT-4o 多模态调用"""
        import openai

        if "openai" not in self._client_cache:
            api_key = os.getenv("OPENAI_API_KEY", "")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY 未设置")
            self._client_cache["openai"] = openai.OpenAI(api_key=api_key)

        client = self._client_cache["openai"]

        content: list[dict] = []
        for b64_url in image_base64_list:
            content.append({
                "type": "image_url",
                "image_url": {"url": b64_url},
            })
        content.append({"type": "text", "text": text_prompt})

        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content or ""

    def _chat_anthropic(
        self, system_prompt: str, image_base64_list: list[str], text_prompt: str
    ) -> str:
        """Claude 多模态调用"""
        import anthropic

        if "anthropic" not in self._client_cache:
            api_key = os.getenv("ANTHROPIC_API_KEY", "")
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY 未设置")
            self._client_cache["anthropic"] = anthropic.Anthropic(api_key=api_key)

        client = self._client_cache["anthropic"]

        content: list[dict] = []
        for b64_url in image_base64_list:
            # 解析 data URL: "data:image/jpeg;base64,XXXX"
            header, b64_data = b64_url.split(",", 1) if "," in b64_url else ("image/jpeg", b64_url)
            media_type = header.split(":")[1].split(";")[0] if ":" in header else "image/jpeg"

            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": b64_data,
                },
            })
        content.append({"type": "text", "text": text_prompt})

        system_kwargs: list = []
        if system_prompt:
            system_kwargs = [{"type": "text", "text": system_prompt}]

        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_kwargs,
            messages=[{"role": "user", "content": content}],
        )
        return response.content[0].text


def _extract_json(text: str) -> dict:
    """从 LLM 响应中提取 JSON（处理 markdown 代码块包裹）"""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:] if lines[0].startswith("```") else lines
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return json.loads(text)
