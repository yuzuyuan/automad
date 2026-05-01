"""Beat / Onset 双轨检测器"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from shared.types import AudioAnalysis, AudioSection

logger = logging.getLogger(__name__)


def analyze_audio(audio_path: str | Path) -> AudioAnalysis:
    """便捷函数：完整分析音频文件"""
    detector = BeatDetector()
    return detector.analyze(audio_path)


class BeatDetector:
    """基于 librosa 的音频节奏分析器

    双轨策略：
    1. beat_track — 节奏骨架，检测重拍和强拍
    2. onset_detect — 瞬态检测，精确起音点（鼓点、吉他扫弦）
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        hop_length: int = 512,
        onset_threshold: float = 0.3,
    ):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.onset_threshold = onset_threshold

    def analyze(self, audio_path: str | Path) -> AudioAnalysis:
        import librosa

        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        logger.info(f"分析音频: {path.name}")
        y, sr = librosa.load(str(path), sr=self.sample_rate)
        duration = len(y) / sr

        # --- Beat 骨架 ---
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=self.hop_length)
        bpm = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
        beats = librosa.frames_to_time(beat_frames, sr=sr, hop_length=self.hop_length).tolist()

        # --- Onset 瞬态 ---
        onset_frames = librosa.onset.onset_detect(
            y=y, sr=sr, hop_length=self.hop_length,
            backtrack=True, delta=self.onset_threshold,
        )
        onsets = librosa.frames_to_time(onset_frames, sr=sr, hop_length=self.hop_length).tolist()

        # --- 能量曲线 ---
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        energy_frames = librosa.frames_to_time(
            np.arange(len(rms)), sr=sr, hop_length=self.hop_length
        )
        energy_curve = rms.tolist()

        # --- 强拍（小节第一拍） ---
        # 对于 4/4 拍，每 4 个 beat 为一个强拍
        downbeats = beats[::4] if len(beats) >= 4 else beats

        # --- 段落结构分析 ---
        sections = self._detect_sections(y, sr, bpm, duration)

        logger.info(
            f"分析完成: BPM={bpm:.1f}, beats={len(beats)}, "
            f"onsets={len(onsets)}, sections={len(sections)}"
        )

        return AudioAnalysis(
            beats=beats,
            downbeats=downbeats,
            onsets=onsets,
            energy_curve=energy_curve,
            sections=sections,
            bpm=bpm,
            duration=duration,
        )

    def _detect_sections(
        self, y: np.ndarray, sr: int, bpm: float, duration: float
    ) -> list[AudioSection]:
        """基于频谱质心和能量变化检测音乐段落"""
        import librosa

        try:
            # 频谱质心 — 音色亮度指标（副歌通常更亮）
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            centroid_times = librosa.frames_to_time(
                np.arange(len(spectral_centroid)), sr=sr, hop_length=self.hop_length
            )

            # RMS 能量
            rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]

            # 简化的段落检测：基于能量变化
            rms_smooth = np.convolve(rms, np.ones(32) / 32, mode="same")
            rms_mean = np.mean(rms_smooth)
            rms_std = np.std(rms_smooth)

            # 将能量分为高/中/低三段
            high_threshold = rms_mean + 0.5 * rms_std
            low_threshold = rms_mean - 0.5 * rms_std

            sections: list[AudioSection] = []
            current_label = self._classify_energy(rms_smooth[0], low_threshold, high_threshold)
            section_start = 0.0

            rms_times = librosa.frames_to_time(
                np.arange(len(rms_smooth)), sr=sr, hop_length=self.hop_length
            )

            for i, (t, energy) in enumerate(zip(rms_times, rms_smooth)):
                label = self._classify_energy(energy, low_threshold, high_threshold)
                if label != current_label and (t - section_start) > 8.0:
                    sections.append(AudioSection(
                        start=section_start, end=t,
                        label=self._label_name(current_label),
                    ))
                    section_start = t
                    current_label = label

            sections.append(AudioSection(
                start=section_start, end=duration,
                label=self._label_name(current_label),
            ))

            return sections

        except Exception:
            logger.warning("段落检测失败，使用默认段落划分")
            return [AudioSection(start=0, end=duration, label="verse")]

    @staticmethod
    def _classify_energy(energy: float, low: float, high: float) -> int:
        if energy > high:
            return 2  # chorus
        elif energy < low:
            return 0  # intro/bridge
        return 1  # verse

    @staticmethod
    def _label_name(level: int) -> str:
        return {0: "intro", 1: "verse", 2: "chorus"}.get(level, "verse")
