"""视频渲染引擎 — MoviePy 首选 + OpenShot 备选"""

from __future__ import annotations

import logging
from pathlib import Path

from shared.types import SnapResult, MADConfig

logger = logging.getLogger(__name__)


class VideoRenderer:
    """统一的视频渲染接口

    首选 MoviePy（API 成熟、Python 生态好），
    备选方案为 OpenShot/libopenshot。
    """

    def __init__(self, config: MADConfig | None = None, backend: str = "moviepy"):
        self.config = config or MADConfig()
        self.backend = backend

    def render(
        self,
        snap_results: list[SnapResult],
        audio_path: str | Path,
        output_path: str | Path,
    ) -> Path:
        """渲染最终视频"""
        if self.backend == "moviepy":
            return self._render_moviepy(snap_results, audio_path, output_path)
        elif self.backend == "openshot":
            return self._render_openshot(snap_results, audio_path, output_path)
        else:
            return self._render_ffmpeg_concat(snap_results, audio_path, output_path)

    def _render_moviepy(
        self,
        snap_results: list[SnapResult],
        audio_path: str | Path,
        output_path: str | Path,
    ) -> Path:
        """使用 MoviePy 渲染"""
        from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips, vfx

        clips = []
        for sr in snap_results:
            try:
                clip = VideoFileClip(sr.source_video)
                clip = clip.subclip(sr.clip_start, sr.clip_end)

                # 变速
                if sr.speed_factor != 1.0:
                    clip = clip.with_effects([vfx.MultiplySpeed(sr.speed_factor)])

                clips.append(clip)
            except Exception as e:
                logger.error(f"加载片段失败 {sr.shot_id}: {e}")

        if not clips:
            raise RuntimeError("没有可用的视频片段")

        final = concatenate_videoclips(clips)

        # 叠加音频
        if Path(audio_path).exists():
            audio = AudioFileClip(str(audio_path))
            final = final.with_audio(audio.subclipped(0, final.duration))

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"渲染输出: {output} ({final.duration:.1f}s)")
        final.write_videofile(
            str(output),
            fps=self.config.video_fps,
            codec=self.config.video_codec,
            audio_codec=self.config.audio_codec,
            preset="medium",
        )

        # 清理
        for clip in clips:
            clip.close()
        final.close()

        logger.info(f"渲染完成: {output}")
        return output

    def _render_openshot(
        self,
        snap_results: list[SnapResult],
        audio_path: str | Path,
        output_path: str | Path,
    ) -> Path:
        """使用 OpenShot/libopenshot 渲染"""
        raise NotImplementedError(
            "OpenShot 渲染后端尚未实现。"
            "请先安装 libopenshot 或将 backend 设为 'moviepy'。"
            "OpenShot GitHub: https://github.com/OpenShot/libopenshot"
        )

    def _render_ffmpeg_concat(
        self,
        snap_results: list[SnapResult],
        audio_path: str | Path,
        output_path: str | Path,
    ) -> Path:
        """使用 FFmpeg concat 直接渲染（高性能方案）"""
        import subprocess
        import tempfile

        # 构建 concat 文件列表
        concat_list = []
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            for sr in snap_results:
                f.write(f"file '{Path(sr.source_video).absolute()}'\n")
                f.write(f"inpoint {sr.clip_start}\n")
                f.write(f"outpoint {sr.clip_end}\n")
            concat_file = f.name

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file,
            "-i", str(audio_path),
            "-c:v", self.config.video_codec,
            "-c:a", self.config.audio_codec,
            "-shortest",
            "-preset", "medium",
            str(output),
        ]

        logger.info(f"执行: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, capture_output=True)

        Path(concat_file).unlink(missing_ok=True)
        return output
