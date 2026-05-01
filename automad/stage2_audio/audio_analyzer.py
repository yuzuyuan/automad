"""音频特征综合分析器"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from shared.types import AudioAnalysis

logger = logging.getLogger(__name__)


class AudioAnalyzer:
    """在 BeatDetector 基础上提供更高层的音频分析

    包括：
    - 能量峰值检测
    - 节奏变化点
    - 段落边界
    - 情绪曲线（基于频谱特征）
    """

    def analyze_energy_peaks(
        self, analysis: AudioAnalysis, top_n: int = 10, min_gap_sec: float = 2.0
    ) -> list[tuple[float, float]]:
        """找出能量曲线中的峰值

        Returns:
            [(时间戳, 能量值), ...] 按能量降序排列
        """
        if not analysis.energy_curve:
            return []

        times = np.linspace(0, analysis.duration, len(analysis.energy_curve))
        energy = np.array(analysis.energy_curve)

        # 滑动窗口找局部最大值
        window_samples = int(min_gap_sec * len(energy) / analysis.duration)
        window_samples = max(1, window_samples)

        peaks: list[tuple[float, float]] = []
        for i in range(window_samples, len(energy) - window_samples):
            window = energy[i - window_samples : i + window_samples + 1]
            if energy[i] == window.max() and energy[i] > np.mean(energy):
                peaks.append((float(times[i]), float(energy[i])))

        peaks.sort(key=lambda x: x[1], reverse=True)
        return peaks[:top_n]

    def get_audio_features(self, audio_path: str | Path) -> dict:
        """获取用于 LLM 提示的音频特征摘要"""
        from stage2_audio.beat_detector import analyze_audio

        analysis = analyze_audio(audio_path)
        energy_peaks = self.analyze_energy_peaks(analysis)

        # 统计 beats 密度变化（检测节奏突变）
        if len(analysis.beats) >= 2:
            beat_intervals = np.diff(analysis.beats)
            mean_interval = float(np.mean(beat_intervals))
            std_interval = float(np.std(beat_intervals))
            tempo_changes = [
                float(analysis.beats[i])
                for i in range(1, len(analysis.beats))
                if abs(beat_intervals[i - 1] - mean_interval) > 2 * std_interval
            ]
        else:
            mean_interval = 0.0
            tempo_changes = []

        return {
            "bpm": analysis.bpm,
            "duration_sec": analysis.duration,
            "beat_count": len(analysis.beats),
            "onset_count": len(analysis.onsets),
            "mean_beat_interval": mean_interval,
            "energy_peaks": [
                {"time": round(t, 2), "energy": round(e, 4)} for t, e in energy_peaks[:5]
            ],
            "tempo_change_points": [round(t, 2) for t in tempo_changes],
            "sections": [
                {"start": round(s.start, 1), "end": round(s.end, 1), "label": s.label}
                for s in analysis.sections
            ],
            "structure_summary": self._summarize_structure(analysis.sections),
        }

    @staticmethod
    def _summarize_structure(sections) -> str:
        if not sections:
            return "未知"
        labels = [s.label for s in sections]
        # 压缩连续相同标签
        compressed = [labels[0]]
        for label in labels[1:]:
            if label != compressed[-1]:
                compressed.append(label)
        return " → ".join(compressed)
