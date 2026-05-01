"""逐句画面描述生成 —— Step 2"""

from __future__ import annotations

import logging
from typing import Optional

from shared.types import (
    GlobalDirectorContext,
    LyricLine,
    TargetPrompt,
    AudioAnalysis,
    ShotPlan,
    RhythmHint,
)
from shared.llm_client import LLMClient
from shared.utils import find_nearest
from stage3_director.prompt_templates import SHOT_PLANNER_SYSTEM, SHOT_PLANNER_USER

logger = logging.getLogger(__name__)


class ShotPlanner:
    """Step 2: 为每一句歌词生成目标画面描述

    以 GlobalDirectorContext 为前置 Prompt（利用 Prompt Caching），
    逐句传入歌词和上下文，生成 TargetPrompt。
    """

    def __init__(self, llm_client: LLMClient | None = None):
        self.llm = llm_client or LLMClient()
        self._cached_context: Optional[GlobalDirectorContext] = None
        self._cached_audio: Optional[AudioAnalysis] = None

    def set_context(self, context: GlobalDirectorContext, audio: AudioAnalysis) -> None:
        """设置全局上下文（将作为后续所有 system prompt 的前缀）"""
        self._cached_context = context
        self._cached_audio = audio

    def generate(
        self,
        lyric_line: LyricLine,
        previous_plan: Optional[ShotPlan] = None,
    ) -> TargetPrompt:
        """为单句歌词生成目标画面描述"""
        if self._cached_context is None or self._cached_audio is None:
            raise RuntimeError("请先调用 set_context() 设置全局上下文")

        context = self._cached_context
        audio = self._cached_audio

        # 找到当前歌词所属的叙事阶段
        stage = self._find_stage(context, lyric_line.start_ts)

        # 上一句信息
        prev_desc = "无（这是第一句）"
        if previous_plan and previous_plan.target:
            prev_desc = previous_plan.target.target_prompt

        # 距最近 beat 的偏移
        nearest_beat, _ = find_nearest(audio.beats, lyric_line.start_ts)
        offset_to_beat = lyric_line.start_ts - nearest_beat

        # 构建翻译行
        translated_line = ""
        if lyric_line.text_translated:
            translated_line = f"翻译：{lyric_line.text_translated}"

        system_prompt = SHOT_PLANNER_SYSTEM.format(
            global_context=self._context_to_text(context),
        )

        user_message = SHOT_PLANNER_USER.format(
            index=lyric_line.index + 1,
            total=0,  # 运行时由 pipeline 动态确定
            start_ts=lyric_line.start_ts,
            end_ts=lyric_line.end_ts,
            duration=lyric_line.duration,
            text_original=lyric_line.text_original,
            translated_line=translated_line,
            section_label=stage.section if stage else "verse",
            narrative_function=stage.narrative_function if stage else "叙事推进",
            visual_theme=stage.visual_theme if stage else "由你判断",
            previous_shot=prev_desc,
            offset_to_beat=offset_to_beat,
        )

        try:
            result = self.llm.chat_json(
                system_prompt=system_prompt,
                user_message=user_message,
            )
        except Exception as e:
            logger.error(f"LLM 调用失败于歌词第 {lyric_line.index} 句: {e}")
            # 降级：使用歌词原文作为 target_prompt
            result = {
                "target_prompt": f"与歌词意境匹配的画面: {lyric_line.text_original}",
                "shot_count": 1,
                "preferred_shot_types": ["medium"],
                "must_have_elements": [],
                "rhythm_hint": "on_beat",
            }

        return TargetPrompt(
            target_prompt=result.get("target_prompt", ""),
            shot_count=min(max(result.get("shot_count", 1), 1), 4),
            preferred_shot_types=result.get("preferred_shot_types", []),
            must_have_elements=result.get("must_have_elements", []),
            rhythm_hint=RhythmHint(result.get("rhythm_hint", "on_beat")),
        )

    def generate_all(
        self,
        lyric_lines: list[LyricLine],
    ) -> list[ShotPlan]:
        """逐句生成画面计划"""
        plans: list[ShotPlan] = []
        for i, line in enumerate(lyric_lines):
            previous = plans[-1] if plans else None
            target = self.generate(line, previous)
            plans.append(ShotPlan(lyric_line=line, target=target))

        logger.info(f"画面计划生成完成: {len(plans)} 句")
        return plans

    @staticmethod
    def _find_stage(context: GlobalDirectorContext, ts: float) -> Optional[ArcStage]:
        import shared.types as t
        for stage in context.arc_stages:
            if stage.time_range[0] <= ts < stage.time_range[1]:
                return stage
        return None

    @staticmethod
    def _context_to_text(ctx: GlobalDirectorContext) -> str:
        parts = [
            f"## 整体叙事弧光\n{ctx.narrative_arc}\n",
            f"## 节奏策略\n{ctx.pacing_strategy}\n",
            f"## 核心意象\n{', '.join(ctx.key_motifs)}\n",
            f"## 角色叙事\n{ctx.character_arc_suggestion}\n",
            "## 各阶段方案",
        ]
        for stage in ctx.arc_stages:
            parts.append(
                f"- {stage.section} [{stage.time_range[0]:.0f}s-{stage.time_range[1]:.0f}s]: "
                f"{stage.narrative_function} | 视觉: {stage.visual_theme}"
            )
        return "\n".join(parts)
