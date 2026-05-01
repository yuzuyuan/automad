"""VLM 画面标注器 —— 调用 Qwen-VL（阿里云 DashScope）对关键帧进行结构化标注

官方文档: https://qwen.ai/apiplatform
API 入口: DashScope OpenAI 兼容接口
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from shared.types import ShotAnnotation, CharacterInfo
from shared.llm_client import VLMClient
from shared.utils import image_to_base64

logger = logging.getLogger(__name__)

_VLM_ANNOTATION_SYSTEM_PROMPT = """你是一位专业的动漫画面分析专家。你需要对给定的动画关键帧进行细致的结构化标注。

## 标注维度

1. **人物**：识别画面中的角色。如果提供了角色参考图，请对比参考图确认角色身份。标注每个角色的表情和情绪状态。
2. **动作**：描述画面中的核心动作，评估动作强度（static/subtle/moderate/intense）。
3. **场景**：描述背景环境、光源类型、主色调。
4. **镜头语言**：判断景别（特写/中景/全景/远景）、角度、运动方式。
5. **叙事标签**：判断画面情绪氛围、叙事功能、象征性元素（如"飘落的樱花"可能象征消逝）。

## 输出格式

严格输出 JSON，字段如下：
{
  "characters": [{"name": "角色名", "emotion": "情绪", "expression_detail": "表情细节"}],
  "character_count": 0,
  "is_solo": false,
  "is_group": false,
  "primary_action": "动作描述",
  "action_intensity": "static|subtle|moderate|intense",
  "action_direction": null,
  "background": "背景描述",
  "setting_type": "indoor|outdoor|abstract",
  "dominant_color_palette": ["主色1", "主色2"],
  "lighting": "golden_hour|dark|harsh|soft|cool|warm|neon",
  "shot_type": "extreme_close_up|close_up|medium|full|wide",
  "camera_angle": "eye_level|low_angle|high_angle|dutch",
  "camera_movement": "static|pan_right|pan_left|tracking|zoom_in|zoom_out",
  "mood": ["情绪词1", "情绪词2"],
  "narrative_function": "emotional_climax|exposition|transition|action_peak|reflection|foreshadow",
  "symbolic_elements": ["象征元素1"],
  "visual_complexity": 0.5,
  "has_text_overlay": false,
  "is_flashback": false
}

注意：
- mood 用中文情绪词，如"忧伤"、"希望"、"紧张"、"温暖"
- symbolic_elements 对 MAD 剪辑至关重要，请仔细识别画面中的隐喻性元素
- visual_complexity 为 0-1 浮点数，0 表示极简画面，1 表示信息密度极高
"""


class VLMAnnotator:
    """多模态 VLM 画面标注器（Qwen-VL / GPT-4o / Claude）

    默认使用 Qwen-VL 通过阿里云 DashScope API。
    支持角色参考图库：通过预加载角色参考图提高识别准确率。
    """

    def __init__(
        self,
        character_ref_dir: Optional[str] = None,
        vlm_client: Optional[VLMClient] = None,
    ):
        self.character_refs: dict[str, str] = {}
        if character_ref_dir:
            self._load_character_gallery(character_ref_dir)
        self.vlm = vlm_client or VLMClient()

    def _load_character_gallery(self, ref_dir: str) -> None:
        """加载角色参考图库"""
        ref_path = Path(ref_dir)
        if not ref_path.exists():
            logger.warning(f"角色参考图库不存在: {ref_dir}")
            return

        for img_path in ref_path.glob("*"):
            if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                name = img_path.stem
                try:
                    self.character_refs[name] = image_to_base64(img_path)
                    logger.debug(f"加载角色参考: {name}")
                except Exception as e:
                    logger.warning(f"加载角色图失败 {img_path}: {e}")

        logger.info(f"已加载 {len(self.character_refs)} 个角色参考图")

    def annotate_shot(
        self,
        shot: ShotAnnotation,
        keyframe_paths: list[str],
    ) -> ShotAnnotation:
        """对单个镜头进行标注

        传入该镜头的多个关键帧路径，VLM 综合分析后输出标注。
        """
        if not keyframe_paths:
            logger.warning(f"镜头 {shot.shot_id} 无关键帧，跳过 VLM 标注")
            return shot

        # 构建多模态图片列表
        image_b64_list: list[str] = []

        # 添加角色参考图（带标签）
        for char_name, char_b64 in self.character_refs.items():
            image_b64_list.append(char_b64)

        # 添加关键帧
        for kf_path in keyframe_paths:
            try:
                b64 = image_to_base64(kf_path)
                image_b64_list.append(b64)
            except Exception as e:
                logger.error(f"读取关键帧失败 {kf_path}: {e}")

        # 构建文本提示
        char_labels = "\n".join(
            f"- [角色参考] {name}" for name in self.character_refs
        ) if self.character_refs else "（无角色参考图）"

        text_prompt = (
            f"[角色参考图]\n{char_labels}\n\n"
            f"[待标注关键帧] 共 {len(keyframe_paths)} 张\n"
            f"镜头时长为 {shot.duration_sec:.1f} 秒。\n"
            f"请对以上关键帧进行标注，严格输出 JSON。"
        )

        try:
            response_text = self.vlm.chat(
                system_prompt=_VLM_ANNOTATION_SYSTEM_PROMPT,
                image_base64_list=image_b64_list,
                text_prompt=text_prompt,
            )

            json_text = _extract_json(response_text)
            annotation_data = json.loads(json_text)
            self._merge_annotation(shot, annotation_data)
            logger.info(f"标注完成: {shot.shot_id} (Qwen-VL)")

        except Exception as e:
            logger.error(f"VLM 标注失败 {shot.shot_id}: {e}")
            logger.warning("将使用空的默认标注继续")

        return shot

    def _merge_annotation(self, shot: ShotAnnotation, data: dict) -> None:
        """将 VLM 返回的 JSON 合并到 ShotAnnotation"""
        shot.characters = [
            CharacterInfo(
                name=c.get("name", "未知"),
                ref_match=c.get("name", ""),
                emotion=c.get("emotion", ""),
                expression_detail=c.get("expression_detail", ""),
            )
            for c in data.get("characters", [])
        ]
        shot.character_count = data.get("character_count", len(shot.characters))
        shot.is_solo = data.get("is_solo", len(shot.characters) == 1)
        shot.is_group = data.get("is_group", len(shot.characters) >= 3)
        shot.primary_action = data.get("primary_action", "")
        shot.action_intensity = data.get("action_intensity", "static")
        shot.action_direction = data.get("action_direction")
        shot.background = data.get("background", "")
        shot.setting_type = data.get("setting_type", "indoor")
        shot.dominant_color_palette = data.get("dominant_color_palette", [])
        shot.lighting = data.get("lighting", "soft")
        shot.shot_type = data.get("shot_type", "medium")
        shot.camera_angle = data.get("camera_angle", "eye_level")
        shot.camera_movement = data.get("camera_movement", "static")
        shot.mood = data.get("mood", [])
        shot.narrative_function = data.get("narrative_function", "exposition")
        shot.symbolic_elements = data.get("symbolic_elements", [])
        shot.visual_complexity = data.get("visual_complexity", 0.5)
        shot.has_text_overlay = data.get("has_text_overlay", False)
        shot.is_flashback = data.get("is_flashback", False)


def _extract_json(text: str) -> str:
    """从 VLM 响应中提取 JSON"""
    text = text.strip()
    if "```" in text:
        lines = text.split("\n")
        start = next(i for i, l in enumerate(lines) if "{" in l or l.strip().startswith("```"))
        end = next(
            i for i in range(len(lines) - 1, -1, -1)
            if "}" in lines[i] or lines[i].strip() == "```"
        )
        lines = lines[start:end + 1]
        if lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text
