"""全局叙事弧光分析 —— Step 1"""

from __future__ import annotations

import logging

from shared.types import (
    AudioAnalysis,
    GlobalDirectorContext,
    ArcStage,
    LyricLine,
)
from shared.llm_client import LLMClient
from stage3_director.prompt_templates import GLOBAL_DIRECTOR_SYSTEM, GLOBAL_DIRECTOR_USER

logger = logging.getLogger(__name__)


class GlobalDirector:
    """Step 1: 分析全局叙事弧光

    将整首歌词 + 创作者意图 + 音频分析结果输入 LLM，
    输出 GlobalDirectorContext，作为后续逐句匹配的前置上下文。
    """

    def __init__(self, llm_client: LLMClient | None = None):
        self.llm = llm_client or LLMClient()

    def analyze(
        self,
        lyric_lines: list[LyricLine],
        audio_analysis: AudioAnalysis,
        creative_intent: str,
        song_info: dict | None = None,
    ) -> GlobalDirectorContext:
        """执行全局叙事分析

        Args:
            lyric_lines: 解析后的歌词行列表
            audio_analysis: 音频分析结果
            creative_intent: 创作者意图描述（如"用少女乐队番剪出悬疑犯罪氛围"）
            song_info: 歌曲元信息 {"title": "...", "artist": "..."}

        Returns:
            GlobalDirectorContext
        """
        logger.info("开始全局叙事弧光分析...")

        # 格式化歌词
        lyric_text = self._format_lyrics(lyric_lines)

        # 格式化音频特征
        energy_peaks_str = ", ".join(
            f"{t:.1f}s" for t in audio_analysis.beats[::8][:5]
        )
        tempo_points = ", ".join(
            f"{t:.1f}s" for t in audio_analysis.downbeats[:5]
        )
        structure = " → ".join(
            f"{s.label}({s.start:.0f}s-{s.end:.0f}s)"
            for s in audio_analysis.sections
        )

        user_prompt = GLOBAL_DIRECTOR_USER.format(
            title=song_info.get("title", "未知") if song_info else "未知",
            artist=song_info.get("artist", "未知") if song_info else "未知",
            bpm=f"{audio_analysis.bpm:.1f}",
            duration=audio_analysis.duration,
            lyric_text=lyric_text,
            creative_intent=creative_intent,
            structure_summary=structure,
            energy_peaks=energy_peaks_str,
            tempo_changes=tempo_points,
        )

        result = self.llm.chat_json(
            system_prompt=GLOBAL_DIRECTOR_SYSTEM,
            user_message=user_prompt,
            cache_system=True,  # Global Context 仅计算一次，适合缓存
        )

        context = self._parse_result(result, audio_analysis)
        logger.info(f"叙事弧光分析完成: {context.narrative_arc}")
        return context

    @staticmethod
    def _format_lyrics(lines: list[LyricLine]) -> str:
        parts: list[str] = []
        for line in lines:
            text = line.text_original
            if line.text_translated:
                text += f" / {line.text_translated}"
            parts.append(f"[{line.start_ts:.2f}-{line.end_ts:.2f}] {text}")
        return "\n".join(parts)

    @staticmethod
    def _parse_result(result: dict, audio: AudioAnalysis) -> GlobalDirectorContext:
        stages = []
        for s in result.get("arc_stages", []):
            stages.append(ArcStage(
                section=s.get("section", ""),
                time_range=tuple(s.get("time_range", [0.0, 0.0])),
                narrative_function=s.get("narrative_function", ""),
                visual_theme=s.get("visual_theme", ""),
            ))

        return GlobalDirectorContext(
            narrative_arc=result.get("narrative_arc", ""),
            arc_stages=stages,
            emotional_curve=result.get("emotional_curve", {}),
            pacing_strategy=result.get("pacing_strategy", ""),
            key_motifs=result.get("key_motifs", []),
            character_arc_suggestion=result.get("character_arc_suggestion", ""),
        )
