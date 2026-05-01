"""最终导出器 —— 高质量渲染与格式转换"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class Exporter:
    """视频导出器

    支持：
    - 多种分辨率输出（1080p, 720p, 4K）
    - H.264 / H.265 编码
    - 硬字幕烧录（预留）
    """

    PRESETS = {
        "1080p": {"width": 1920, "height": 1080, "bitrate": "8M"},
        "720p": {"width": 1280, "height": 720, "bitrate": "4M"},
        "4K": {"width": 3840, "height": 2160, "bitrate": "25M"},
    }

    def __init__(
        self,
        preset: str = "1080p",
        codec: str = "libx264",
        audio_codec: str = "aac",
        fps: int = 24,
    ):
        self.preset = preset
        self.codec = codec
        self.audio_codec = audio_codec
        self.fps = fps

    def export(
        self,
        input_path: str | Path,
        output_path: str | Path | None = None,
        preset: str | None = None,
    ) -> Path:
        """导出最终视频"""
        import subprocess

        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"输入文件不存在: {input_path}")

        preset_name = preset or self.preset
        preset_config = self.PRESETS.get(preset_name, self.PRESETS["1080p"])

        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_final{input_path.suffix}"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-c:v", self.codec,
            "-c:a", self.audio_codec,
            "-b:v", preset_config["bitrate"],
            "-vf", f"scale={preset_config['width']}:{preset_config['height']}",
            "-r", str(self.fps),
            "-preset", "medium",
            str(output_path),
        ]

        logger.info(f"导出: {preset_name} → {output_path}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"导出失败: {result.stderr}")

        logger.info(f"导出完成: {output_path}")
        return output_path
