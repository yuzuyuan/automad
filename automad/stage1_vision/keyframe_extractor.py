"""关键帧提取器"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2

from shared.types import ShotAnnotation

logger = logging.getLogger(__name__)


def extract_keyframes(
    shot: ShotAnnotation,
    source_video: str | Path,
    n_frames: int = 3,
) -> list[tuple[float, str]]:
    """便捷函数：从一个 Shot 中提取关键帧

    Returns:
        [(timestamp_sec, keyframe_path), ...]
    """
    extractor = KeyframeExtractor(cache_dir="data/cache/keyframes")
    return extractor.extract(shot, source_video, n_frames)


class KeyframeExtractor:
    """从镜头中提取关键帧并保存

    策略：取 shot 的首帧、中间帧、尾帧（共 3 帧）。
    对于长镜头（>10s），额外在 1/4 和 3/4 位置采样。
    """

    def __init__(self, cache_dir: str = "data/cache/keyframes"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def extract(
        self,
        shot: ShotAnnotation,
        source_video: str | Path,
        n_frames: int = 3,
    ) -> list[tuple[float, str]]:
        cap = cv2.VideoCapture(str(source_video))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {source_video}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 24.0

        # 采样时间点
        start, end = shot.time_range
        if n_frames == 1:
            sample_ts = [(start + end) / 2]
        elif n_frames == 3:
            sample_ts = [start, (start + end) / 2, end - 0.1]
        else:
            sample_ts = [
                start + (end - start) * i / (n_frames - 1)
                for i in range(n_frames)
            ]
        sample_ts = [max(start, t) for t in sample_ts]

        results: list[tuple[float, str]] = []
        for i, ts in enumerate(sample_ts):
            frame_no = int(ts * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            if not ret:
                # 回退到最近的可用帧
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_no - 5))
                ret, frame = cap.read()
            if not ret:
                continue

            out_path = self.cache_dir / f"{shot.shot_id}_{i:02d}.jpg"
            cv2.imwrite(str(out_path), frame)
            results.append((ts, str(out_path)))

        cap.release()
        return results
