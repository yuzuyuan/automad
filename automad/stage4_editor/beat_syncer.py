"""卡点吸附引擎 —— 核心算法

让镜头切换严格吸附到 Beat/Onset 时间戳上。
不仅用空镜卡点——通过光流法检测镜头内 motion peak，
让动作爆发点精确对齐节拍。
"""

from __future__ import annotations

import logging

import numpy as np

from shared.types import (
    AudioAnalysis,
    SnapResult,
    SnapPolicy,
)
from shared.utils import find_nearest, clamp

logger = logging.getLogger(__name__)


def detect_motion_peaks(
    clip_frames: list[np.ndarray],
    fps: float = 24.0,
    sample_rate: int = 4,
    top_n: int = 5,
) -> list[float]:
    """检测镜头的运动峰值时刻

    使用 Farneback 光流法，逐帧计算运动幅值，
    返回运动能量最大的帧时间戳。

    Args:
        clip_frames: 镜头帧列表
        fps: 帧率
        sample_rate: 采样间隔（每 N 帧计算一次光流）
        top_n: 返回前 N 个峰值

    Returns:
        运动峰值时间戳列表（秒）
    """
    import cv2

    if len(clip_frames) < 2:
        return [0.0]

    motion_scores: list[tuple[int, float]] = []
    prev_gray = None

    for i in range(0, len(clip_frames), sample_rate):
        frame = clip_frames[i]
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).mean()
            motion_scores.append((i, mag))

        prev_gray = gray

    if not motion_scores:
        return [0.0]

    motion_scores.sort(key=lambda x: x[1], reverse=True)
    top_moments = motion_scores[:top_n]
    top_moments.sort(key=lambda x: x[0])  # 按时间排序

    return [frame_idx / fps for frame_idx, _ in top_moments]


class BeatSyncer:
    """卡点吸附同步器

    核心策略：
    1. 检测镜头内 motion peak
    2. 优先将 action_peak 对齐到最近的 beat
    3. 若无法对齐则通过裁剪或微调变速实现
    """

    def __init__(
        self,
        tolerance_ms: float = 50.0,
        speed_range_subtle: tuple[float, float] = (0.92, 1.08),
        speed_range_moderate: tuple[float, float] = (0.80, 1.25),
    ):
        self.tolerance_ms = tolerance_ms / 1000.0
        self.speed_range_subtle = speed_range_subtle
        self.speed_range_moderate = speed_range_moderate

    def snap(
        self,
        snap_result: SnapResult,
        audio: AudioAnalysis,
        policy: SnapPolicy = SnapPolicy.FLEXIBLE,
    ) -> SnapResult:
        """将单个 SnapResult 吸附到最近的 beat/onset

        Args:
            snap_result: 待吸附的片段
            audio: 音频分析结果
            policy: STRICT 严格对齐 / FLEXIBLE 允许 ±speed_range

        Returns:
            调整后的 SnapResult（clip_start/end 和 speed_factor 已被修改）
        """
        result = SnapResult(
            shot_id=snap_result.shot_id,
            source_video=snap_result.source_video,
            clip_start=snap_result.clip_start,
            clip_end=snap_result.clip_end,
            position=snap_result.position,
            speed_factor=snap_result.speed_factor,
            snap_anchor=snap_result.snap_anchor,
        )

        target_beat = self._find_snap_target(
            result.position, audio, policy
        )

        offset = target_beat - result.position

        if abs(offset) < self.tolerance_ms:
            # 偏移不可感知，无需处理
            result.position = target_beat
            result.snap_anchor = "start"
            result.speed_factor = 1.0
            return result

        clip_dur = result.clip_end - result.clip_start

        if offset > 0:
            # 镜头来早了 → 推迟起始，从开头裁掉 offset
            if offset < clip_dur * 0.3:
                result.clip_start += offset
                result.position = target_beat
                result.snap_anchor = "start"
            else:
                # 裁剪太多会丢失内容 → 使用微调慢放
                result.position = target_beat
                result.speed_factor = clip_dur / (clip_dur + offset)
                result.speed_factor = clamp(
                    result.speed_factor,
                    self.speed_range_subtle[0],
                    self.speed_range_subtle[1],
                )
                result.snap_anchor = "start"
        else:
            # 镜头来晚了 → 加速或从尾部裁剪
            speed_needed = clip_dur / (clip_dur - abs(offset))

            if self.speed_range_subtle[0] <= speed_needed <= self.speed_range_subtle[1]:
                # 微调加速
                result.position = target_beat
                result.speed_factor = speed_needed
                result.snap_anchor = "start"
            elif abs(offset) < clip_dur * 0.3:
                # 从尾部裁剪
                result.clip_end -= abs(offset)
                result.position = target_beat
                result.snap_anchor = "end"
            else:
                # 适度加速范围内
                speed_needed = clamp(speed_needed, *self.speed_range_moderate)
                result.position = target_beat
                result.speed_factor = speed_needed
                result.snap_anchor = "start"

        return result

    def snap_all(
        self,
        snap_results: list[SnapResult],
        audio: AudioAnalysis,
        policy: SnapPolicy = SnapPolicy.FLEXIBLE,
    ) -> list[SnapResult]:
        """批量卡点吸附"""
        synced = [self.snap(sr, audio, policy) for sr in snap_results]
        logger.info(
            f"卡点吸附完成: {len(synced)} 个片段, "
            f"{sum(1 for s in synced if s.speed_factor != 1.0)} 个调整了变速"
        )
        return synced

    @staticmethod
    def _find_snap_target(
        position: float,
        audio: AudioAnalysis,
        policy: SnapPolicy,
    ) -> float:
        """找到 position 附近最佳的吸附目标 beat/onset"""
        all_points = list(audio.beats) + list(audio.onsets) + list(audio.downbeats)
        if not all_points:
            return position

        nearest, _ = find_nearest(sorted(all_points), position)

        if policy == SnapPolicy.STRICT:
            return nearest

        # FLEXIBLE 模式：在 ±0.15s 范围内优先匹配 downbeat
        window = 0.15
        candidates_in_window = [
            p for p in all_points if abs(p - position) < window
        ]

        # 优先 downbeat → beat → onset
        for pool in [audio.downbeats, audio.beats, audio.onsets]:
            pool_in_window = [p for p in pool if abs(p - position) < window]
            if pool_in_window:
                return min(pool_in_window, key=lambda p: abs(p - position))

        return nearest
