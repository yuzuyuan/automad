# 动漫 MAD 自动化剪辑系统 —— 架构设计

## 一、项目整体目录结构

```
automad/
├── Netease_url/                  # [已有] 网易云音乐解析+下载
├── stage1_vision/                # 阶段1：画面预处理与向量库
│   ├── shot_detector.py          #   镜头切分 (PySceneDetect)
│   ├── keyframe_extractor.py     #   关键帧提取
│   ├── vlm_annotator.py          #   VLM 标注（GPT-4o / Claude）
│   ├── embedding_store.py        #   向量数据库读写 (ChromaDB)
│   └── schema.py                 #   JSON 标注 Schema
├── stage2_audio/                 # 阶段2：音频分析与歌词处理
│   ├── music_fetcher.py          #   封装 Netease_url → 下载音频+LRC
│   ├── beat_detector.py          #   librosa Beat/Onset 检测
│   ├── lrc_parser.py             #   LRC 歌词解析 → 时间戳数组
│   └── audio_analyzer.py         #   音频特征（能量、频谱、段落）
├── stage3_director/              # 阶段3：LLM 导演引擎
│   ├── global_director.py        #   Step 1 — 全局叙事弧光分析
│   ├── shot_planner.py           #   Step 2 — 逐句画面描述生成
│   ├── prompt_templates.py       #   Prompt 模板库
│   └── retriever.py              #   语义检索 → 匹配镜头
├── stage4_editor/                # 阶段4：自动化剪辑与渲染
│   ├── timeline_builder.py       #   时间线构建与动态时长分配
│   ├── beat_syncer.py            #   卡点吸附引擎（核心算法）
│   ├── speed_adapter.py          #   镜头变速适配
│   ├── openshot_engine.py        #   OpenShot / libopenshot 渲染封装
│   └── exporter.py               #   最终导出
├── shared/                       # 共享模块
│   ├── config.py                 #   全局配置
│   ├── types.py                  #   数据类型定义
│   ├── llm_client.py             #   LLM API 统一客户端
│   └── utils.py                  #   工具函数
├── pipeline.py                   # 主 Pipeline 编排
└── requirements.txt              # 完整依赖
```

---

## 二、阶段 1：画面预处理与向量数据库

### 2.1 技术栈

| 模块 | 推荐技术 | 理由 |
|------|---------|------|
| 镜头切分 | **PySceneDetect** + OpenCV | 内容感知检测（HSV 直方图差 + 自适应阈值），远比固定帧步长准确 |
| 关键帧提取 | 自定义：取每个 shot 的中间帧 + 首尾帧（共 3 帧，覆盖起承转合） | 实测 3 帧对 VLM 描述效果最佳 |
| VLM 标注 | **Qwen-VL**（阿里云 DashScope，`qwen-vl-max`） | 多模态视觉理解 + OpenAI 兼容接口 |
| Embedding | **OpenAI text-embedding-3-small**（1536d，性价比最高） | 也可用 `BAAI/bge-m3` 本地部署节省成本 |
| 向量数据库 | **ChromaDB**（开发阶段）→ **Milvus**（生产阶段） | ChromaDB 零配置启动，Milvus 支持百万级镜头 |

### 2.2 VLM 标注 JSON Schema

```python
# shared/types.py

from pydantic import BaseModel
from typing import Optional, List

class ShotAnnotation(BaseModel):
    """单个镜头的 VLM 标注结构"""
    shot_id: str                          # 唯一 ID，如 "S01E03_0042"
    source_video: str                     # 来源视频文件名
    time_range: tuple[float, float]       # (start_sec, end_sec)
    duration_sec: float

    # -- 人物维度 --
    characters: List[dict]                # [{"name": "高松燈", "ref_match": "char_01",
                                          #   "emotion": "悲伤/决绝", "expression_detail": "眼角含泪但紧咬嘴唇"}]
    character_count: int                  # 画面中角色数量（含路人）
    is_solo: bool                         # 是否为单人镜头
    is_group: bool                        # 是否为群像镜头

    # -- 动作维度 --
    primary_action: str                   # "缓慢抬头", "冲向对方", "按下钢琴键"
    action_intensity: str                 # "static" | "subtle" | "moderate" | "intense"
    action_direction: Optional[str]       # "left→right" | "up→down" | "toward_camera"

    # -- 场景维度 --
    background: str                       # "黄昏的教室", "下雨的街道"
    setting_type: str                     # "indoor" | "outdoor" | "abstract"
    dominant_color_palette: List[str]     # ["暖橙", "深蓝", "灰"]
    lighting: str                         # "golden_hour" | "dark" | "harsh" | "soft"

    # -- 镜头语言 --
    shot_type: str                        # "close_up" | "medium" | "full" | "wide" | "extreme_close_up"
    camera_angle: str                     # "eye_level" | "low_angle" | "high_angle" | "dutch"
    camera_movement: str                  # "static" | "pan_right" | "tracking" | "zoom_in"

    # -- 叙事标签 --
    mood: List[str]                       # ["忧伤", "希望", "紧张"]
    narrative_function: str               # "emotional_climax" | "exposition" | "transition" | "action_peak"
    symbolic_elements: List[str]          # ["飘落的樱花", "破碎的镜子"] — 对 MAD 叙事特别重要

    # -- 技术元数据 --
    visual_complexity: float              # 0-1，画面信息密度
    has_text_overlay: bool                # 是否有字幕/文字
    is_flashback: bool                    # 是否为闪回/回忆镜头
```

**关键设计考量**：

- `symbolic_elements` 是 MAD 剪辑的灵魂——LLM 导演可以通过隐喻物件的匹配来"讲故事"（如 "飘落的樱花 → 美好事物的消逝 → 匹配伤感的歌词"）
- `action_intensity` 和 `mood` 是卡点匹配的核心——高能量 beat 会自动匹配 `intense` 镜头
- `narrative_function` 让全局叙事弧光分析和具体镜头之间对齐

### 2.3 角色参考图库机制

```python
# stage1_vision/vlm_annotator.py 核心逻辑

class VLMAnnotator:
    def __init__(self, character_ref_dir: str):
        # 加载角色参考图库：{角色名: base64_image}
        self.char_refs = self._load_character_gallery(character_ref_dir)

    def annotate_shot(self, keyframe_path: str) -> ShotAnnotation:
        # 构建多图 Message：
        # [系统提示 + 角色参考图1 + 角色参考图2 + ... + 待标注关键帧]
        # VLM 会对照参考图识别画面中的角色身份
        ...
```

角色参考图建议每个角色准备 2-3 张不同角度/表情的截图。

---

## 三、阶段 2：音频分析与歌词处理

### 3.1 技术栈

| 模块 | 推荐技术 | 理由 |
|------|---------|------|
| 音乐下载 | **复用 `Netease_url/main.py`** + `download_lyrics.py` | 已有完整实现，调用 `lyric_v1()` 获取 LRC |
| LRC 解析 | 自定义 `LrcParser` | LRC 格式 `[mm:ss.xx]` 正则解析为 `[(start_ts, end_ts, text)]` |
| Beat 检测 | **librosa** `beat.beat_track()` + `onset.onset_detect()` | 双轨检测：beat 负责节奏骨架，onset 负责瞬态卡点 |
| 结构分析 | **librosa** `decompose.hpss()` + `feature.spectral_centroid()` | 分离和声/打击乐，识别段落边界（主歌/副歌/桥段） |
| 特征提取 | **librosa** `feature.rms()`（能量曲线）、`feature.tempogram()`（节奏谱） | 为 LLM 导演提供量化的音频 "情绪起伏" 数据 |

### 3.2 LRC 解析器的设计

```python
# stage2_audio/lrc_parser.py

@dataclass
class LyricLine:
    index: int
    start_ts: float       # 该句开始时间（秒）
    end_ts: float         # 该句结束时间（下一句开始，或估为 start+3s）
    duration: float       # 持续时长
    text_original: str    # 原文（日文/中文）
    text_translated: str  # 翻译（若有）

def parse_lrc(lrc_path: str) -> List[LyricLine]:
    """解析 LRC 文件为结构化的歌词列表"""
    # 1. 正则提取 [mm:ss.xx]text
    # 2. 计算每句 end_ts（默认：下一句 start，最后一句 +3s）
    # 3. 合并原文+翻译（根据 download_lyrics.py 的输出格式）
    pass
```

### 3.3 Beat/Onset 双轨检测

```python
# stage2_audio/beat_detector.py

@dataclass
class AudioAnalysis:
    beats: List[float]           # 重拍时间戳数组
    downbeats: List[float]       # 强拍（小节第一拍）
    onsets: List[float]          # 瞬态起音点（用于精确卡点）
    energy_curve: List[float]    # 逐帧 RMS 能量
    sections: List[dict]         # [{"start":0, "end":45.2, "label":"intro"}, ...]
    bpm: float

def analyze_audio(audio_path: str) -> AudioAnalysis:
    y, sr = librosa.load(audio_path)

    # Beat 骨架
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beats = librosa.frames_to_time(beat_frames, sr=sr)

    # 强拍（小节线），对于卡点精度至关重要
    downbeats = beats[::4]  # 4/4 拍音乐中每个第4拍是小节强拍

    # Onset 瞬态 — 鼓点、吉他扫弦的精确位置
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='time')

    # 能量曲线
    rms = librosa.feature.rms(y=y)[0]
    energy_curve = rms.tolist()

    return AudioAnalysis(beats=beats, downbeats=downbeats, onsets=onset_frames, ...)
```

---

## 四、阶段 3：LLM 剪辑导演（核心大脑）

### 4.1 技术栈

| 模块 | 推荐 | 理由 |
|------|------|------|
| LLM | **DeepSeek V4** (`deepseek-v4-pro`, 1M 上下文) | 超长上下文 + OpenAI 兼容接口 |
| 检索 | ChromaDB 的 `collection.query()` | Cosine 相似度 |
| 检索 | ChromaDB 的 `collection.query()` | Cosine 相似度 |

### 4.2 两步 Prompt 策略

**Step 1 —— 全局宏观：Global_Director_Context**

```
System: 你是一位 MAD 剪辑导演，擅长将音乐与动漫画面进行叙事性融合。

User:
## 歌曲元信息
- 歌名：{title}
- 歌手：{artist}
- BPM：{bpm}
- 风格：{genre}

## 完整歌词（带时间轴）
{逐句歌词，格式： [00:12.50 - 00:16.20] 原文 / 翻译}

## 创作者意图
整体氛围：{mood_target}（如"用少女乐队番剪出悬疑犯罪氛围"）
目标情绪曲线：{emotion_target}

## 音频分析结果
- 段落结构：{sections}（intro/verse/chorus/bridge/outro）
- 能量峰值时刻：{energy_peaks}
- 节奏变化点：{tempo_changes}

请输出一份 JSON 格式的 Global_Director_Context：
{
  "narrative_arc": "对整首歌的叙事弧光总结（50字内）",
  "arc_stages": [
    {"section": "intro", "time_range": [0, 45], "narrative_function": "建立悬疑氛围，埋下伏笔",
     "visual_theme": "空镜、碎片化回忆、人物背影"},
    ...
  ],
  "emotional_curve": {"0s":"冷峻","45s":"紧张升起","90s":"冲突爆发",...},
  "pacing_strategy": "快-慢-快-爆发",
  "key_motifs": ["破碎", "雨", "奔跑", "对视"],
  "character_arc_suggestion": "主角从旁观者→卷入事件→内心挣扎→决绝行动"
}
```

**Step 2 —— 局部微观：Target_Prompt_B[j] 生成**

```
System: {Global_Director_Context 全文}

User:
当前处理的是歌词第 {j} 句（共 {total} 句）：

  [{start_ts:.2f} - {end_ts:.2f}] ({duration:.1f}s)
  「{lyric_text}」
  翻译：{translated_text}

当前处于音乐段落：{current_section}，该段落的叙事功能：{narrative_function}
上一句选中的镜头：{previous_shot_summary}
距上一个 beat 的时间偏移：{offset_to_beat}s

请输出：
{
  "target_prompt": "一个自然语言画面描述，用于向量检索。必须具体描述人物、动作、情绪、构图",
  "shot_count": 1,         // 建议该句歌词用几个镜头（1~4）
  "preferred_shot_types": ["close_up"],
  "must_have_elements": ["角色眼神特写"],
  "rhythm_hint": "on_beat" // "on_beat"(卡正拍)|"syncopated"(切分)|"free"
}
```

**整个 Step 2 的优化策略**：利用 DeepSeek V4 的 1M 上下文窗口，`Global_Director_Context` 作为 System Prompt 在每轮对话中复用。由于 DeepSeek 采用 OpenAI 兼容接口，可通过构建长消息历史实现上下文持续注入，无需额外缓存机制。

### 4.3 语义检索器

```python
# stage3_director/retriever.py

def retrieve_shots(
    target_prompt: str,
    top_k: int = 5,
    diversity_threshold: float = 0.15,
    filters: dict = None     # 如 {"shot_type": "close_up", "action_intensity": "intense"}
) -> List[tuple[float, ShotAnnotation]]:
    """
    对 target_prompt 进行 embedding，在向量库中检索。
    返回 top_k 结果，同时做多样性重排（MMR）——避免连续使用同一来源的镜头。
    """
    embedding = get_embedding(target_prompt)

    # ChromaDB 查询，可叠加元数据过滤
    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k * 3,          # 先拉 3 倍候选
        where=filters
    )

    # MMR 多样性重排：防止同源视频的镜头堆积
    reranked = mmr_rerank(results, lambda_=0.7)
    return reranked[:top_k]
```

---

## 五、阶段 4：自动化对齐与渲染

### 5.1 渲染方案选择

| 场景 | 推荐工具 | 说明 |
|------|---------|------|
| **首选方案** | **MoviePy** + FFmpeg | Python 生态最成熟的视频编辑库，clip 拼接、变速、转场都有现成 API |
| **OpenShot 方案** | libopenshot Python 绑定 | 可作为备选，通过 `openshot.Timeline` + `openshot.Clip` 操作 |
| **性能方案** | FFmpeg **concat demuxer** + **filter_complex** | 最高性能、完全可控，但需要手动处理时间线计算 |

**建议先用 MoviePy 快速迭代，后期遇到性能瓶颈再用 FFmpeg concat 重写渲染层。** MoviePy 的 API 与需求高度匹配：

```python
# MoviePy 实现示例
from moviepy.editor import VideoFileClip, concatenate_videoclips

clip = VideoFileClip("S01E03.mp4").subclip(shot.start_ts, shot.end_ts)

# 变速适配
if clip.duration > target_duration:
    clip = clip.fx(vfx.speedx, final_duration=target_duration)
elif clip.duration < target_duration:
    clip = clip.fx(vfx.speedx, final_duration=target_duration)  # 慢放

# 卡点吸附 —— 通过 subclip 精细裁剪实现
clip = clip.subclip(trim_start, trim_end)
```

### 5.2 动态时长分配算法

```python
# stage4_editor/timeline_builder.py

def allocate_duration(
    lyric_line: LyricLine,
    shot_plan: dict,           # LLM 输出的 shot_count, rhythm_hint
    beats: List[float],
    onset_times: List[float],
) -> List[Segment]:
    """
    根据歌词时长、节奏和视觉表现力，动态分配该句内部的镜头数量与时长。

    规则：
    1. LLM 的 shot_count 是建议，实际根据可用 beat 数量调整
    2. 每个镜头的转场点必须 Snap 到最近的 beat/onset
    3. 高能量句（副歌）片段时间向 beat 严格对齐
    4. 低能量句（主歌）可以允许 ±80ms 的对齐容差
    """
    seg_start = lyric_line.start_ts
    seg_end = lyric_line.end_ts
    n_shots = shot_plan["shot_count"]

    # 收集该区间内的 beat 和 onset
    interval_beats = [b for b in beats if seg_start <= b < seg_end]
    interval_onsets = [o for o in onset_times if seg_start <= o < seg_end]

    # 合并 beat + onset 为候选切点（beat 权重 1.0，onset 权重 0.7）
    cut_candidates = merge_cut_points(interval_beats, interval_onsets)

    # 动态规划：在 cut_candidates 中选择 n_shots-1 个切点，
    # 使得片段时长方差最小（避免忽快忽慢）
    cut_points = dp_optimal_cuts(seg_start, seg_end, n_shots - 1, cut_candidates)

    return build_segments(seg_start, seg_end, cut_points)
```

### 5.3 高级卡点吸附引擎（核心）

这是整个系统最关键的模块——**不能只用空镜卡点，普通镜头也必须通过裁剪或变速精准卡拍。**

```python
# stage4_editor/beat_syncer.py

@dataclass
class SnapResult:
    clip: VideoFileClip
    snap_anchor: str       # "start" | "end" | "both" | "action_peak"
    speed_factor: float    # 变速系数

def snap_to_beat(
    clip: VideoFileClip,
    target_start_beat: float,
    target_end_beat: float,
    beat_info: AudioAnalysis,
    snap_policy: str,      # "strict" | "flexible"
) -> SnapResult:
    """
    让镜头精准吸附到音乐节拍上。核心策略：

    1. 优先分析镜头内部的 motion 峰值时刻
       - 使用 OpenCV farneback 光流法检测画面运动
       - 找到运动能量最大的帧 → 作为 action_peak

    2. 对齐策略（按优先级）：
       a) action_peak → beat（最佳：情绪爆发点对齐重拍）
       b) clip.start → beat（次选：镜头起始对齐 beat）
       c) clip.end → beat（再次：镜头结束对齐 beat）
       d) speed_change → beat（备选：通过 0.9x~1.1x 微调变速）

    3. 变速范围限制：
       - subtle: 0.92x ~ 1.08x（几乎无感知）
       - moderate: 0.80x ~ 1.25x（可察觉但自然）
       - extreme: 0.50x ~ 2.00x（明显慢放/快进，用于特效段落）
    """
    # Step 1: 检测镜头内运动峰值
    motion_peaks = detect_motion_peaks(clip)

    # Step 2: 寻找最近的 beat 候选
    nearest_beat = find_nearest(beat_info.beats, target_start_beat)

    # Step 3: 计算需要的时间偏移
    offset = nearest_beat - target_start_beat

    # Step 4: 裁剪策略
    if abs(offset) < 0.05:  # <50ms，人类视觉不可感知
        return SnapResult(clip=clip, snap_anchor="start", speed_factor=1.0)
    elif offset > 0:  # 镜头早到了，需要推迟 → 从开头裁掉 offset 秒
        return SnapResult(clip=clip.subclip(offset, None), snap_anchor="start", speed_factor=1.0)
    else:  # 镜头晚到了 → 两种策略
        if abs(offset) < clip.duration * 0.15:  # 可以微调加速
            return SnapResult(
                clip=clip.fx(vfx.speedx, clip.duration / (clip.duration - abs(offset))),
                snap_anchor="start", speed_factor=clip.duration / (clip.duration - abs(offset))
            )
        else:  # 需要从尾部裁剪
            return SnapResult(clip=clip.subclip(0, clip.duration - abs(offset)),
                              snap_anchor="end", speed_factor=1.0)
```

**Motion Peak 检测（光流法）的实现思路**：

```python
def detect_motion_peaks(clip: VideoFileClip, sample_rate: int = 4) -> List[float]:
    """
    每隔 sample_rate 帧计算一次光流幅值，
    返回运动能量最大的帧的时间戳列表。
    这些时间戳是"动作爆发点"，用来对齐 beat 效果最好。
    """
    import cv2
    import numpy as np

    motion_scores = []
    prev_frame = None

    for t, frame in clip.iter_frames(with_times=True, fps=clip.fps / sample_rate):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if prev_frame is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2).mean()
            motion_scores.append((t, mag))
        prev_frame = gray

    motion_scores.sort(key=lambda x: x[1], reverse=True)
    return [t for t, _ in motion_scores[:5]]
```

### 5.4 渲染层封装

```python
# stage4_editor/openshot_engine.py

import openshot

class OpenShotRenderer:
    """基于 libopenshot 的渲染引擎"""

    def __init__(self, fps: int = 24, width: int = 1920, height: int = 1080):
        self.timeline = openshot.Timeline(width, height, openshot.Fraction(fps, 1))

    def add_clip(self, video_path: str, start_frame: int, end_frame: int,
                 position: float, speed: float = 1.0) -> None:
        clip = openshot.Clip(video_path)
        clip.Position(position)                    # 在时间线上的位置（秒）
        clip.Start(start_frame / self.timeline.info.fps.to_double())
        clip.End(end_frame / self.timeline.info.fps.to_double())
        if speed != 1.0:
            # libopenshot 变速
            clip.time = openshot.Time(1.0 / speed, 1.0)
        self.timeline.AddClip(clip)

    def add_transition(self, position: float, duration: float = 0.25) -> None:
        """在 position 处添加交叉溶解转场"""
        # libopenshot 的转场通过重叠 clip 实现
        pass

    def render(self, output_path: str) -> None:
        writer = openshot.FFmpegWriter(output_path)
        writer.SetVideoOptions("libx264", ...)
        self.timeline.Open()
        writer.WriteFrame(self.timeline, 1, self.timeline.GetMaxFrame())
        writer.Close()
```

**如果选择 MoviePy 方案，渲染层会更简洁**：

```python
def render_timeline(segments: List[SnapResult], audio_path: str, output_path: str):
    # 拼接所有片段
    final = concatenate_videoclips([s.clip for s in segments], method="compose")

    # 叠加音频
    audio = AudioFileClip(audio_path)
    final = final.set_audio(audio)

    # 渲染
    final.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=24)
```

---

## 六、主 Pipeline 编排

```python
# pipeline.py

class MADPipeline:
    def __init__(self, config: MADConfig):
        self.shot_detector = ShotDetector()
        self.vlm_annotator = VLMAnnotator(config.char_ref_dir)
        self.embedding_store = EmbeddingStore(config.chroma_path)
        self.music_fetcher = MusicFetcher()          # 封装 Netease_url
        self.lrc_parser = LrcParser()
        self.beat_detector = BeatDetector()
        self.global_director = GlobalDirector()
        self.shot_planner = ShotPlanner()
        self.retriever = ShotRetriever()
        self.timeline_builder = TimelineBuilder()
        self.beat_syncer = BeatSyncer()
        self.renderer = OpenShotRenderer()

    def run(self, video_dir: str, music_id: str, creative_intent: str):
        # ===== 阶段 1：画面预处理 =====
        shots = self.shot_detector.detect(video_dir)
        for shot in shots:
            keyframes = extract_keyframes(shot, n=3)
            annotation = self.vlm_annotator.annotate_shot(keyframes)
            embedding = get_embedding(annotation.to_searchable_text())
            self.embedding_store.insert(embedding, annotation)

        # ===== 阶段 2：音频分析 =====
        audio_path = self.music_fetcher.download(music_id)
        lrc_path = self.music_fetcher.download_lyrics(music_id)
        lyric_lines = self.lrc_parser.parse(lrc_path)
        audio_analysis = self.beat_detector.analyze(audio_path)

        # ===== 阶段 3：LLM 导演 =====
        global_context = self.global_director.analyze(
            lyric_lines, audio_analysis, creative_intent
        )
        plan = []
        for j, line in enumerate(lyric_lines):
            target = self.shot_planner.generate_target(
                global_context, line, j, plan[-1] if plan else None
            )
            candidates = self.retriever.retrieve(target["target_prompt"])
            plan.append({"line": line, "target": target, "candidates": candidates})

        # ===== 阶段 4：渲染 =====
        segments = self.timeline_builder.allocate(plan, audio_analysis)
        synced = [self.beat_syncer.snap(s, audio_analysis) for s in segments]
        self.renderer.render(synced, audio_path, "output.mp4")
```

---

## 七、完整依赖清单

```txt
# requirements.txt（全量）

# === 阶段 1：画面处理 ===
opencv-python>=4.9.0
scenedetect[opencv]>=0.6.3       # PySceneDetect
chromadb>=0.5.0
openai>=1.30.0                    # Embedding + GPT-4o API
anthropic>=0.30.0                 # Claude API
Pillow>=10.0.0

# === 阶段 2：音频 ===
librosa>=0.10.0
soundfile>=0.12.0
mutagen>=1.47.0                   # [已有] 音频标签

# === 阶段 3：LLM 导演 ===
# (openai / anthropic 已包含)

# === 阶段 4：渲染 ===
moviepy>=2.0.0                    # 首选方案
# libopenshot                      # 备选方案，需从源码编译

# === 共享 ===
pydantic>=2.0.0
numpy>=1.26.0
tqdm>=4.65.0
python-dotenv>=1.0.0

# === 音频来源（已有 Netease_url）===
# 继承 Netease_url/requirements.txt 中的依赖
requests>=2.28.0
flask>=3.0.0
cryptography>=40.0.0
```

---

## 八、关键架构决策总结

| 决策点 | 选择 | 依据 |
|--------|------|------|
| 镜头切分 | PySceneDetect 内容感知 | 比固定帧步长准确得多 |
| VLM | Qwen-VL / GPT-4o | 多模态 + 动漫理解力 |
| Embedding | text-embedding-3-small | 性价比最优，后续可换 bge-m3 本地化 |
| 向量库 | ChromaDB → Milvus 渐进升级 | 快速启动 vs 规模化 |
| 音频 | librosa beat + onset 双轨 | beat 给节奏骨架，onset 给精确瞬态 |
| LLM 导演 | DeepSeek V4 (`deepseek-v4-pro`) | 1M 上下文 + OpenAI 兼容接口 |
| 渲染 | **MoviePy（首选）** → FFmpeg（终极） | 工程可行性优先，OpenShot 作为可选项 |
| 架构模式 | Pipeline 编排 + 每阶段可独立运行 | 方便调试、A/B 测试 |

---

## 九、核心创新点

1. **VLM 标注的 `symbolic_elements` 维度**——让机器理解"樱花 = 消逝"这种隐喻，支撑叙事性匹配
2. **光流法 Motion Peak 检测**——让每个普通镜头都能找到内在的"情绪爆发帧"，用来卡 beat，而不只是依赖空镜
3. **两步 Prompt + Cache**——全局叙事弧光只算一次，逐句匹配只需传增量信息，把 API 成本压到可量产水平
