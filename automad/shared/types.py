"""MAD 剪辑系统核心数据类型定义"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# =============================================================================
# 枚举类型
# =============================================================================


class ActionIntensity(str, Enum):
    STATIC = "static"
    SUBTLE = "subtle"
    MODERATE = "moderate"
    INTENSE = "intense"


class ShotType(str, Enum):
    EXTREME_CLOSE_UP = "extreme_close_up"
    CLOSE_UP = "close_up"
    MEDIUM = "medium"
    FULL = "full"
    WIDE = "wide"


class CameraAngle(str, Enum):
    EYE_LEVEL = "eye_level"
    LOW_ANGLE = "low_angle"
    HIGH_ANGLE = "high_angle"
    DUTCH = "dutch"


class CameraMovement(str, Enum):
    STATIC = "static"
    PAN_RIGHT = "pan_right"
    PAN_LEFT = "pan_left"
    TRACKING = "tracking"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"


class SettingType(str, Enum):
    INDOOR = "indoor"
    OUTDOOR = "outdoor"
    ABSTRACT = "abstract"


class Lighting(str, Enum):
    GOLDEN_HOUR = "golden_hour"
    DARK = "dark"
    HARSH = "harsh"
    SOFT = "soft"
    COOL = "cool"
    WARM = "warm"
    NEON = "neon"


class NarrativeFunction(str, Enum):
    EMOTIONAL_CLIMAX = "emotional_climax"
    EXPOSITION = "exposition"
    TRANSITION = "transition"
    ACTION_PEAK = "action_peak"
    REFLECTION = "reflection"
    FORESHADOW = "foreshadow"


class RhythmHint(str, Enum):
    ON_BEAT = "on_beat"
    SYNCOPATED = "syncopated"
    FREE = "free"


class SnapPolicy(str, Enum):
    STRICT = "strict"
    FLEXIBLE = "flexible"


# =============================================================================
# 阶段 1 — 视觉标注
# =============================================================================


@dataclass
class CharacterInfo:
    name: str
    ref_match: str  # 角色参考图库中的 key
    emotion: str
    expression_detail: str = ""


@dataclass
class ShotAnnotation:
    """VLM 对单个镜头的完整标注"""

    shot_id: str
    source_video: str
    time_range: tuple[float, float]  # (start_sec, end_sec)
    duration_sec: float

    # 人物
    characters: list[CharacterInfo] = field(default_factory=list)
    character_count: int = 0
    is_solo: bool = False
    is_group: bool = False

    # 动作
    primary_action: str = ""
    action_intensity: ActionIntensity = ActionIntensity.STATIC
    action_direction: Optional[str] = None

    # 场景
    background: str = ""
    setting_type: SettingType = SettingType.INDOOR
    dominant_color_palette: list[str] = field(default_factory=list)
    lighting: Lighting = Lighting.SOFT

    # 镜头语言
    shot_type: ShotType = ShotType.MEDIUM
    camera_angle: CameraAngle = CameraAngle.EYE_LEVEL
    camera_movement: CameraMovement = CameraMovement.STATIC

    # 叙事
    mood: list[str] = field(default_factory=list)
    narrative_function: NarrativeFunction = NarrativeFunction.EXPOSITION
    symbolic_elements: list[str] = field(default_factory=list)

    # 元数据
    visual_complexity: float = 0.5
    has_text_overlay: bool = False
    is_flashback: bool = False

    def to_searchable_text(self) -> str:
        """将标注转为 Embedding 检索用的自然语言文本"""
        parts = [
            f"人物: {', '.join(c.name for c in self.characters)}" if self.characters else "",
            f"情绪: {', '.join(self.mood)}" if self.mood else "",
            f"动作: {self.primary_action} (强度: {self.action_intensity.value})",
            f"背景: {self.background}",
            f"镜头: {self.shot_type.value} {self.camera_angle.value} {self.camera_movement.value}",
            f"象征元素: {', '.join(self.symbolic_elements)}" if self.symbolic_elements else "",
            f"叙事功能: {self.narrative_function.value}",
        ]
        return " | ".join(p for p in parts if p)


# =============================================================================
# 阶段 2 — 音频与歌词
# =============================================================================


@dataclass
class LyricLine:
    index: int
    start_ts: float
    end_ts: float
    text_original: str
    text_translated: str = ""

    @property
    def duration(self) -> float:
        return self.end_ts - self.start_ts


@dataclass
class AudioSection:
    start: float
    end: float
    label: str  # "intro" | "verse" | "chorus" | "bridge" | "outro"


@dataclass
class AudioAnalysis:
    beats: list[float] = field(default_factory=list)
    downbeats: list[float] = field(default_factory=list)
    onsets: list[float] = field(default_factory=list)
    energy_curve: list[float] = field(default_factory=list)
    sections: list[AudioSection] = field(default_factory=list)
    bpm: float = 0.0
    duration: float = 0.0


# =============================================================================
# 阶段 3 — LLM 导演
# =============================================================================


@dataclass
class ArcStage:
    section: str  # 段落名称
    time_range: tuple[float, float]
    narrative_function: str
    visual_theme: str


@dataclass
class GlobalDirectorContext:
    """Step 1 输出 — 全局叙事弧光分析"""

    narrative_arc: str
    arc_stages: list[ArcStage] = field(default_factory=list)
    emotional_curve: dict[str, str] = field(default_factory=dict)
    pacing_strategy: str = ""
    key_motifs: list[str] = field(default_factory=list)
    character_arc_suggestion: str = ""


@dataclass
class TargetPrompt:
    """Step 2 输出 — 逐句画面描述"""

    target_prompt: str
    shot_count: int = 1
    preferred_shot_types: list[str] = field(default_factory=list)
    must_have_elements: list[str] = field(default_factory=list)
    rhythm_hint: RhythmHint = RhythmHint.ON_BEAT


@dataclass
class ShotPlan:
    """单句歌词的完整匹配结果"""

    lyric_line: LyricLine
    target: TargetPrompt
    candidates: list[tuple[float, ShotAnnotation]] = field(default_factory=list)
    selected_shot: Optional[ShotAnnotation] = None


# =============================================================================
# 阶段 4 — 剪辑与渲染
# =============================================================================


@dataclass
class SnapResult:
    """卡点吸附后的镜头片段"""

    shot_id: str
    source_video: str
    clip_start: float  # 在源视频中的裁剪起始
    clip_end: float  # 在源视频中的裁剪结束
    position: float  # 在时间线上的位置
    speed_factor: float = 1.0
    snap_anchor: str = "start"  # "start" | "end" | "action_peak"


@dataclass
class Segment:
    """时间线上的一个片段区间"""

    lyric_line: LyricLine
    snap_results: list[SnapResult] = field(default_factory=list)
    start_ts: float = 0.0
    end_ts: float = 0.0


# =============================================================================
# 全局配置
# =============================================================================


@dataclass
class MADConfig:
    """全局管线配置"""

    # 视频
    video_fps: int = 24
    video_width: int = 1920
    video_height: int = 1080
    video_codec: str = "libx264"
    audio_codec: str = "aac"

    # 路径
    data_dir: str = "data"
    video_dir: str = "data/videos"
    character_ref_dir: str = "data/characters"
    chroma_path: str = "data/chroma_db"
    cache_dir: str = "data/cache"
    output_dir: str = "output"

    # 阶段 1
    keyframes_per_shot: int = 3
    shot_min_duration: float = 0.5  # 最小镜长（秒）

    # 阶段 2
    sample_rate: int = 22050
    onset_threshold: float = 0.3

    # 阶段 3
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    retrieval_top_k: int = 5
    mmr_lambda: float = 0.7  # MMR 多样性权重

    # 阶段 3 — LLM (DeepSeek V4)
    llm_provider: str = "deepseek"  # "deepseek" | "openai" | "anthropic"
    llm_model: str = "deepseek-v4-pro"  # DeepSeek V4 旗舰模型
    llm_max_tokens: int = 4096
    llm_temperature: float = 1.0

    # VLM — Qwen-VL
    vlm_provider: str = "qwen"  # "qwen" | "anthropic" | "openai"
    vlm_model: str = "qwen-vl-max"  # Qwen-VL 旗舰（DashScope）

    # 阶段 4
    snap_tolerance_ms: float = 50.0  # 人类不可感知的偏移量
    speed_range_subtle: tuple[float, float] = (0.92, 1.08)
    speed_range_moderate: tuple[float, float] = (0.80, 1.25)
    speed_range_extreme: tuple[float, float] = (0.50, 2.00)
    default_transition_duration: float = 0.25  # 交叉溶解时长
