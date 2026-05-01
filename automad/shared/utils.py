"""共享工具函数"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
from pathlib import Path
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

# =============================================================================
# Embedding
# =============================================================================

_EMBEDDING_CLIENT = None
_EMBEDDING_MODEL = "text-embedding-3-small"
_EMBEDDING_DIM = 1536


def init_embedding(model: str = "text-embedding-3-small") -> None:
    global _EMBEDDING_MODEL, _EMBEDDING_CLIENT, _EMBEDDING_DIM
    _EMBEDDING_MODEL = model

    if model.startswith("text-embedding"):
        import openai

        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY 未设置")
        _EMBEDDING_CLIENT = openai.OpenAI(api_key=api_key)
        _EMBEDDING_DIM = 1536 if "small" in model else 3072
    else:
        # BGE-M3 等本地模型暂用 openai 兼容接口
        import openai

        base_url = os.getenv("EMBEDDING_BASE_URL", "http://localhost:11434/v1")
        api_key = os.getenv("EMBEDDING_API_KEY", "ollama")
        _EMBEDDING_CLIENT = openai.OpenAI(api_key=api_key, base_url=base_url)
        _EMBEDDING_DIM = 1024


def get_embedding(text: str | list[str]) -> np.ndarray:
    """获取文本的向量表示，返回 (dim,) 或 (n, dim) 的 numpy 数组"""
    global _EMBEDDING_CLIENT, _EMBEDDING_MODEL

    if _EMBEDDING_CLIENT is None:
        init_embedding()

    is_single = isinstance(text, str)
    texts = [text] if is_single else list(text)

    response = _EMBEDDING_CLIENT.embeddings.create(model=_EMBEDDING_MODEL, input=texts)
    embeddings = np.array([e.embedding for e in response.data], dtype=np.float32)

    return embeddings[0] if is_single else embeddings


# =============================================================================
# 检索工具
# =============================================================================


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个向量的 Cosine 相似度"""
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norm = b / (np.linalg.norm(b) + 1e-10)
    return float(np.dot(a_norm, b_norm))


def mmr_rerank(
    query_embedding: np.ndarray,
    candidates: list[tuple[float, object]],
    lambda_: float = 0.7,
    top_k: int | None = None,
) -> list[tuple[float, object]]:
    """Maximal Marginal Relevance 多样性重排

    Args:
        query_embedding: 查询向量
        candidates: [(相似度, 对象), ...]，按相似度降序排列
        lambda_: 相关性权重 (0-1)，越高越偏重相关性，越低越偏重多样性
        top_k: 返回数量，默认全部
    """
    if len(candidates) <= 1:
        return candidates[:top_k] if top_k else candidates

    selected: list[tuple[float, object]] = [candidates[0]]
    remaining = list(candidates[1:])

    # 需要候选对象的 embedding 来计算多样性，这里简化：使用 source_video 作为多样性指标
    # 实际实现中应由调用方传入 embeddings
    while remaining and (top_k is None or len(selected) < top_k):
        best_score = -float("inf")
        best_idx = 0

        for i, (sim, obj) in enumerate(remaining):
            # 简化 MMR：source_video 相同时施加惩罚
            diversity_penalty = 0.0
            for _, sel_obj in selected:
                if hasattr(obj, "source_video") and hasattr(sel_obj, "source_video"):
                    if obj.source_video == sel_obj.source_video:
                        diversity_penalty += 0.3
            score = lambda_ * sim - (1 - lambda_) * diversity_penalty
            if score > best_score:
                best_score = score
                best_idx = i

        selected.append(remaining.pop(best_idx))

    return selected


# =============================================================================
# 数值工具
# =============================================================================


def find_nearest(sorted_list: Sequence[float], target: float) -> tuple[float, int]:
    """在有序列表中找到最接近 target 的值和索引"""
    if not sorted_list:
        raise ValueError("列表不能为空")

    idx = _binary_search_nearest(sorted_list, target)
    return sorted_list[idx], idx


def _binary_search_nearest(arr: Sequence[float], target: float) -> int:
    lo, hi = 0, len(arr) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    if lo > 0 and abs(arr[lo - 1] - target) < abs(arr[lo] - target):
        return lo - 1
    return lo


def clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))


# =============================================================================
# 文件工具
# =============================================================================


def image_to_base64(image_path: str | Path) -> str:
    """将图片编码为 base64 data URL"""
    path = Path(image_path)
    ext = path.suffix.lower().lstrip(".")
    mime_map = {".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png", ".webp": "webp"}
    mime = mime_map.get(f".{ext}", "jpeg")

    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/{mime};base64,{data}"


def file_md5(path: str | Path, chunk_size: int = 8192) -> str:
    """计算文件 MD5"""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def video_short_name(path: str | Path) -> str:
    """从视频文件路径提取简短标识名"""
    return Path(path).stem
