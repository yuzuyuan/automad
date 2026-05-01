"""全局配置管理 — 支持环境变量和 .env 文件覆盖"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from shared.types import MADConfig

_DEFAULT_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def load_config(env_path: str | Path | None = None) -> MADConfig:
    """加载配置，优先级: 代码默认 → .env 文件 → 环境变量"""
    if env_path is None:
        env_path = _DEFAULT_ENV_PATH

    if Path(env_path).exists():
        load_dotenv(env_path, override=False)

    # 环境变量覆盖
    overrides: dict = {}

    def _env(key: str, default):
        val = os.getenv(key, "")
        return val if val else default

    overrides["video_fps"] = int(_env("MAD_VIDEO_FPS", 24))
    overrides["video_width"] = int(_env("MAD_VIDEO_WIDTH", 1920))
    overrides["video_height"] = int(_env("MAD_VIDEO_HEIGHT", 1080))
    overrides["video_codec"] = _env("MAD_VIDEO_CODEC", "libx264")
    overrides["audio_codec"] = _env("MAD_AUDIO_CODEC", "aac")

    overrides["data_dir"] = _env("MAD_DATA_DIR", "data")
    overrides["video_dir"] = _env("MAD_VIDEO_DIR", "data/videos")
    overrides["character_ref_dir"] = _env("MAD_CHAR_REF_DIR", "data/characters")
    overrides["chroma_path"] = _env("MAD_CHROMA_PATH", "data/chroma_db")
    overrides["cache_dir"] = _env("MAD_CACHE_DIR", "data/cache")
    overrides["output_dir"] = _env("MAD_OUTPUT_DIR", "output")

    overrides["embedding_model"] = _env("MAD_EMBEDDING_MODEL", "text-embedding-3-small")
    overrides["llm_provider"] = _env("MAD_LLM_PROVIDER", "deepseek")
    overrides["llm_model"] = _env("MAD_LLM_MODEL", "deepseek-v4-pro")
    overrides["vlm_provider"] = _env("MAD_VLM_PROVIDER", "qwen")
    overrides["vlm_model"] = _env("MAD_VLM_MODEL", "qwen-vl-max")

    overrides["llm_temperature"] = float(_env("MAD_LLM_TEMPERATURE", "1.0"))
    overrides["snap_tolerance_ms"] = float(_env("MAD_SNAP_TOLERANCE_MS", "50.0"))

    config = MADConfig(**overrides)

    _ensure_dirs(config)
    return config


def _ensure_dirs(config: MADConfig) -> None:
    """确保必要的目录存在"""
    for dir_path in [
        config.data_dir,
        config.video_dir,
        config.character_ref_dir,
        config.chroma_path,
        config.cache_dir,
        config.output_dir,
    ]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
