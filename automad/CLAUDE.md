# CLAUDE.md — 动漫 MAD 自动化剪辑系统

## 项目概述

本项目是一个面向动漫 MAD（Music Anime Douga）的自动化剪辑系统。核心理念是**歌词语义/情绪与画面元素的精准对应** + **音视频在叙事和节奏（卡点）上的融合**。

## 技术栈

- **语言**: Python 3.11+
- **视觉处理**: PySceneDetect, OpenCV, Pillow
- **VLM 标注**: **Qwen-VL** (阿里云 DashScope，OpenAI 兼容接口) — 多模态视觉标注
- **文本 LLM**: **DeepSeek V4** (`deepseek-v4-pro`，1M 上下文，OpenAI 兼容接口)
- **向量库**: ChromaDB (开发) / Milvus (生产)
- **Embedding**: OpenAI text-embedding-3-small 或 BAAI/bge-m3
- **音频**: librosa (beat/onset 检测, 频谱分析)
- **渲染**: MoviePy (首选) / libopenshot (备选) / FFmpeg
- **数据模型**: Pydantic v2 / dataclasses
- **音频源**: 已有 Netease_url 模块 (Flask API + 网易云音乐解析)

## API 密钥配置

在项目根目录创建 `.env` 文件：

```bash
# DeepSeek V4（文本 LLM）
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx

# Qwen-VL（多模态 VLM，阿里云 DashScope）
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxx

# OpenAI（Embedding 用）
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx

# 可选：其他 provider
# ANTHROPIC_API_KEY=sk-ant-xxxxxxxx
```

### API 端点一览

| 服务 | Provider | Base URL | 用途 |
|------|----------|----------|------|
| DeepSeek V4 | `deepseek` | `https://api.deepseek.com` | 文本 LLM 导演 |
| Qwen-VL | `qwen` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | 多模态画面标注 |
| OpenAI | `openai` | 默认 | Embedding + 备选 LLM |

## 项目结构

```
automad/
├── Netease_url/              # [已有] 网易云音乐解析+下载服务
│   ├── main.py               #   Flask API 入口 (端口 5000)
│   ├── music_api.py          #   核心 API：url, search, lyric, playlist, album
│   ├── music_downloader.py   #   下载器：同步/异步/批量
│   ├── cookie_manager.py     #   Cookie 管理
│   └── download_lyrics.py    #   独立 LRC 歌词下载脚本
├── stage1_vision/            # 画面预处理 + 向量库
├── stage2_audio/             # 音频分析 + LRC 解析
├── stage3_director/          # LLM 导演引擎 + 语义检索
├── stage4_editor/            # 时间线构建 + 卡点吸附 + 渲染
├── shared/                   # 共享工具/类型/配置
│   ├── types.py              #   全部 dataclass 类型定义
│   ├── config.py             #   .env 驱动配置
│   ├── llm_client.py         #   LLMClient (文本) + VLMClient (多模态)
│   └── utils.py              #   工具函数 (embedding, MMR, 光流)
├── ARCHITECTURE.md           # 完整架构设计文档（权威参考）
├── pipeline.py               # 主 Pipeline 编排
└── requirements.txt          # 完整依赖
```

## 关键架构决策

1. **镜头切分**：PySceneDetect 内容感知，不用固定帧步长
2. **关键帧**：每个 shot 取 3 帧（首/中/尾），覆盖起承转合
3. **VLM 标注**：Qwen-VL 通过 DashScope OpenAI 兼容接口，多图输入含角色参考库
4. **文本 LLM**：DeepSeek V4 (`deepseek-v4-pro`)，1M 上下文，OpenAI 兼容接口
5. **Beat 检测**：双轨 —— librosa `beat_track()` 给节奏骨架，`onset_detect()` 给精确瞬态
6. **两步 Prompt**：Step 1 全局叙事弧光分析（缓存复用），Step 2 逐句画面描述生成
7. **卡点策略**：通过光流法检测镜头内 motion peak，让动作爆发点精确对齐 beat
8. **渲染首选 MoviePy**：API 友好，快速迭代。OpenShot 作为备选

## 现有 Netease_url 模块使用方式

```python
# 在 pipeline 中复用已有代码，不要直接启动 Flask 服务
from music_api import lyric_v1, url_v1, NeteaseAPI
from cookie_manager import CookieManager

# 获取歌词（返回 dict，其中 lrc.lyric 是原文，tlyric.lyric 是翻译）
lyric = lyric_v1(song_id, cookies)

# 获取下载 URL
url_info = url_v1(song_id, 'lossless', cookies)
```

`download_lyrics.py` 会将 LRC 保存到 `./OutLyric/{id}.lrc`，格式为原文+翻译合并。

## 开发规范

- **类型注解**：所有公开函数和方法使用完整类型注解
- **数据模型**：使用 dataclass，定义在 `shared/types.py`
- **配置**：通过 `shared/config.py` 统一管理，支持 .env 覆盖
- **API 密钥**：存于 `.env` 文件，不提交到 git
- **LLM 交互**：文本 LLM 用 `shared/llm_client.LLMClient`，多模态 VLM 用 `shared/llm_client.VLMClient`
- **错误处理**：每个阶段模块有自己的异常处理，Pipeline 层统一捕获
- **日志**：使用 `logging` 模块，每个模块有自己的 logger
- **视频源文件**：放在 `data/videos/` 下，按番剧分目录

## 开发顺序

按 Pipeline 依赖关系：
1. `shared/` — 类型、配置、工具函数
2. `stage2_audio/` — 音频分析
3. `stage1_vision/` — 画面预处理 + VLM 标注
4. `stage3_director/` — LLM 导演 + 检索
5. `stage4_editor/` — 渲染输出
6. `pipeline.py` — 端到端编排

## 跨平台注意事项

- 路径统一使用 `pathlib.Path`，不使用硬编码 `/` 或 `\`
- FFmpeg 需要系统安装并在 PATH 中
- 中文字符路径和文件名需在所有模块中正确处理（UTF-8 编码）
