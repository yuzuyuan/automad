"""时间线构建与动态时长分配"""

from __future__ import annotations

import logging
from copy import deepcopy

import numpy as np

from shared.types import (
    AudioAnalysis,
    LyricLine,
    ShotPlan,
    Segment,
    SnapResult,
    RhythmHint,
)
from shared.utils import find_nearest

logger = logging.getLogger(__name__)


class TimelineBuilder:
    """根据 LLM 的 ShotPlan 和音频分析结果，构建带时间轴的编辑片段列表

    核心职责：
    1. 动态时长分配——不是每句歌词一个镜头
    2. 切点吸附——面转场点严格 Snap 到最近的 beat/onset
    3. 合并 beat + onset 为候选切点
    """

    def __init__(
        self,
        beat_weight: float = 1.0,
        onset_weight: float = 0.7,
        default_clip_duration: float = 2.0,
    ):
        self.beat_weight = beat_weight
        self.onset_weight = onset_weight
        self.default_clip_duration = default_clip_duration

    def build(self, plans: list[ShotPlan], audio: AudioAnalysis) -> list[Segment]:
        """从 ShotPlan 列表构建时间线片段"""
        segments: list[Segment] = []
        cut_candidates = self._merge_cut_points(audio)

        for plan in plans:
            seg = self._build_segment(plan, audio, cut_candidates)
            segments.append(seg)

        logger.info(f"时间线构建完成: {len(segments)} 个片段, "
                     f"总时长={sum(s.end_ts - s.start_ts for s in segments):.1f}s")
        return segments

    def _build_segment(
        self,
        plan: ShotPlan,
        audio: AudioAnalysis,
        cut_candidates: list[float],
    ) -> Segment:
        line = plan.lyric_line
        seg_start = line.start_ts
        seg_end = line.end_ts
        n_shots = plan.target.shot_count
        rhythm = plan.target.rhythm_hint

        segment = Segment(lyric_line=line, start_ts=seg_start, end_ts=seg_end)

        if plan.selected_shot is None:
            logger.warning(f"歌词 '{line.text_original[:20]}...' 无匹配镜头，跳过")
            return segment

        if n_shots == 1:
            # 单镜头：整个歌词时长分配给一个镜头
            snap = SnapResult(
                shot_id=plan.selected_shot.shot_id,
                source_video=plan.selected_shot.source_video,
                clip_start=plan.selected_shot.time_range[0],
                clip_end=plan.selected_shot.time_range[1],
                position=seg_start,
            )
            segment.snap_results = [snap]

        else:
            # 多镜头快剪：在这句歌词内部划分切点
            cut_points = self._distribute_cuts(
                seg_start, seg_end, n_shots, cut_candidates, rhythm
            )
            segment.snap_results = self._assign_shots_to_cuts(
                plan, cut_points, seg_start, seg_end
            )

        return segment

    def _distribute_cuts(
        self,
        seg_start: float,
        seg_end: float,
        n_shots: int,
        cut_candidates: list[float],
        rhythm: RhythmHint,
    ) -> list[float]:
        """在歌词区间内选择 n_shots-1 个切点"""
        duration = seg_end - seg_start
        if n_shots <= 1 or duration <= 0:
            return []

        # 收集区间内的候选切点
        interval_candidates = [
            c for c in cut_candidates if seg_start + 0.3 < c < seg_end - 0.3
        ]

        # 理想切点：均匀分布
        ideal_cuts = [
            seg_start + duration * i / n_shots
            for i in range(1, n_shots)
        ]

        if not interval_candidates:
            return ideal_cuts

        # 贪心：每个理想切点吸附到最近候选点
        used: set[int] = set()
        final_cuts: list[float] = []

        for ideal in ideal_cuts:
            nearest, idx = find_nearest(interval_candidates, ideal)
            if idx not in used and abs(nearest - ideal) < duration / n_shots:
                final_cuts.append(nearest)
                used.add(idx)
            else:
                final_cuts.append(ideal)

        return sorted(final_cuts)

    def _assign_shots_to_cuts(
        self,
        plan: ShotPlan,
        cut_points: list[float],
        seg_start: float,
        seg_end: float,
    ) -> list[SnapResult]:
        """将镜头分配到各个切点区间"""
        intervals = [seg_start] + sorted(cut_points) + [seg_end]
        snaps: list[SnapResult] = []

        for i in range(len(intervals) - 1):
            t_start = intervals[i]
            t_end = intervals[i + 1]

            # 从候选镜头中选择（这里简化：使用最佳候选，避免重复）
            # 实际可扩展为：不同的子镜头使用不同的候选
            shot = plan.selected_shot
            if shot is None:
                break

            # 计算从镜头的哪一段取用
            clip_dur = t_end - t_start
            clip_mid = (shot.time_range[0] + shot.time_range[1]) / 2
            half = min(clip_dur / 2, shot.duration_sec / 2)

            snaps.append(SnapResult(
                shot_id=shot.shot_id,
                source_video=shot.source_video,
                clip_start=max(0, clip_mid - half),
                clip_end=min(shot.time_range[1], clip_mid + half),
                position=t_start,
            ))

        return snaps

    @staticmethod
    def _merge_cut_points(audio: AudioAnalysis) -> list[float]:
        """合并 beat 和 onset 为候选切点列表"""
        candidates: dict[float, float] = {}
        for b in audio.beats:
            candidates[round(b, 2)] = max(candidates.get(round(b, 2), 0), 1.0)
        for o in audio.onsets:
            candidates[round(o, 2)] = max(candidates.get(round(o, 2), 0), 0.7)
        return sorted(candidates.keys())
