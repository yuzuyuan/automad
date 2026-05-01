"""MAD 自动化剪辑系统 — 主 Pipeline 编排

用法:
    python pipeline.py --video-dir data/videos --music-id 185668 \
        --intent "用少女乐队番剪出悬疑犯罪氛围" --output output/mad.mp4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from shared.config import load_config
from shared.types import MADConfig
from shared.llm_client import LLMClient, VLMClient
from shared.utils import init_embedding

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pipeline")


class MADPipeline:
    """MAD 剪辑主管线"""

    def __init__(self, config: MADConfig | None = None):
        self.config = config or load_config()
        self._init_modules()

    def _init_modules(self):
        """延迟导入各阶段模块"""
        from stage1_vision.shot_detector import ShotDetector
        from stage1_vision.keyframe_extractor import KeyframeExtractor
        from stage1_vision.vlm_annotator import VLMAnnotator
        from stage1_vision.embedding_store import EmbeddingStore
        from stage2_audio.music_fetcher import MusicFetcher
        from stage2_audio.lrc_parser import LrcParser
        from stage2_audio.beat_detector import BeatDetector
        from stage2_audio.audio_analyzer import AudioAnalyzer
        from stage3_director.global_director import GlobalDirector
        from stage3_director.shot_planner import ShotPlanner
        from stage3_director.retriever import ShotRetriever
        from stage4_editor.timeline_builder import TimelineBuilder
        from stage4_editor.beat_syncer import BeatSyncer
        from stage4_editor.openshot_engine import VideoRenderer

        llm = LLMClient(
            provider=self.config.llm_provider,
            model=self.config.llm_model,
            max_tokens=self.config.llm_max_tokens,
            temperature=self.config.llm_temperature,
        )

        vlm = VLMClient(
            provider=self.config.vlm_provider,
            model=self.config.vlm_model,
        )

        # 阶段 1
        self.shot_detector = ShotDetector(min_shot_duration=self.config.shot_min_duration)
        self.keyframe_extractor = KeyframeExtractor()
        self.vlm_annotator = VLMAnnotator(
            character_ref_dir=self.config.character_ref_dir,
            vlm_client=vlm,
        )
        self.embedding_store = EmbeddingStore(
            persist_path=self.config.chroma_path,
            embedding_model=self.config.embedding_model,
        )

        # 阶段 2
        self.music_fetcher = MusicFetcher()
        self.lrc_parser = LrcParser()
        self.beat_detector = BeatDetector(sample_rate=self.config.sample_rate)
        self.audio_analyzer = AudioAnalyzer()

        # 阶段 3
        self.global_director = GlobalDirector(llm_client=llm)
        self.shot_planner = ShotPlanner(llm_client=llm)
        self.shot_retriever = ShotRetriever(store=self.embedding_store)

        # 阶段 4
        self.timeline_builder = TimelineBuilder()
        self.beat_syncer = BeatSyncer(
            tolerance_ms=self.config.snap_tolerance_ms,
        )
        self.renderer = VideoRenderer(config=self.config)

    def run_stage1(self, video_dir: str, skip_vlm: bool = False) -> None:
        """阶段 1：画面预处理与向量库构建"""
        logger.info("=" * 60)
        logger.info("阶段 1：画面预处理与向量库构建")
        logger.info("=" * 60)

        shots = self.shot_detector.detect_batch(video_dir)
        logger.info(f"镜头检测完成: {len(shots)} 个镜头")

        if skip_vlm:
            logger.warning("跳过 VLM 标注（--skip-vlm）")
            return

        for i, shot in enumerate(shots):
            logger.info(f"[{i+1}/{len(shots)}] 标注镜头 {shot.shot_id}")

            # 提取关键帧
            keyframes = self.keyframe_extractor.extract(shot, shot.source_video)
            kf_paths = [p for _, p in keyframes]

            # VLM 标注
            self.vlm_annotator.annotate_shot(shot, kf_paths)

            # 入库
            self.embedding_store.insert(shot)

        logger.info(f"阶段 1 完成: {self.embedding_store.count()} 条记录入库")

    def run_stage2(self, music_id: str, quality: str = "lossless"):
        """阶段 2：音频分析与歌词处理

        Returns:
            (audio_path, lyric_lines, audio_analysis, song_info)
        """
        logger.info("=" * 60)
        logger.info("阶段 2：音频分析与歌词处理")
        logger.info("=" * 60)

        audio_path, lrc_text, song_info = self.music_fetcher.fetch_all(music_id, quality)
        lyric_lines = self.lrc_parser.parse_text(lrc_text)
        audio_analysis = self.beat_detector.analyze(audio_path)

        logger.info(f"阶段 2 完成: {len(lyric_lines)} 句歌词, BPM={audio_analysis.bpm:.1f}")
        return audio_path, lyric_lines, audio_analysis, song_info

    def run_stage3(
        self,
        lyric_lines,
        audio_analysis,
        song_info: dict,
        creative_intent: str,
    ):
        """阶段 3：LLM 导演与镜头检索

        Returns:
            list[ShotPlan]
        """
        logger.info("=" * 60)
        logger.info("阶段 3：LLM 剪辑导演")
        logger.info("=" * 60)

        # Step 1：全局叙事弧光
        global_context = self.global_director.analyze(
            lyric_lines=lyric_lines,
            audio_analysis=audio_analysis,
            creative_intent=creative_intent,
            song_info=song_info,
        )

        # Step 2：逐句画面描述
        self.shot_planner.set_context(global_context, audio_analysis)
        shot_plans = self.shot_planner.generate_all(lyric_lines)

        # 检索匹配镜头
        shot_plans = self.shot_retriever.fill_all_plans(shot_plans)

        return shot_plans

    def run_stage4(
        self,
        shot_plans,
        audio_analysis,
        audio_path: Path,
        output_path: str,
    ):
        """阶段 4：剪辑与渲染"""
        logger.info("=" * 60)
        logger.info("阶段 4：自动化剪辑与渲染")
        logger.info("=" * 60)

        # 构建时间线
        segments = self.timeline_builder.build(shot_plans, audio_analysis)

        # 卡点吸附
        all_snaps = []
        for seg in segments:
            synced = self.beat_syncer.snap_all(seg.snap_results, audio_analysis)
            all_snaps.extend(synced)

        # 渲染
        output = self.renderer.render(all_snaps, audio_path, output_path)
        return output

    def run(
        self,
        video_dir: str,
        music_id: str,
        creative_intent: str,
        output_path: str = "output/mad.mp4",
        skip_vlm: bool = False,
        quality: str = "lossless",
    ) -> Path:
        """运行完整的 MAD 剪辑管线"""
        logger.info("=" * 60)
        logger.info("MAD 自动化剪辑系统启动")
        logger.info(f"  视频目录: {video_dir}")
        logger.info(f"  音乐ID: {music_id}")
        logger.info(f"  创作意图: {creative_intent}")
        logger.info(f"  输出路径: {output_path}")
        logger.info("=" * 60)

        # 阶段 1
        self.run_stage1(video_dir, skip_vlm=skip_vlm)

        # 阶段 2
        audio_path, lyric_lines, audio_analysis, song_info = self.run_stage2(
            music_id, quality
        )

        # 阶段 3
        shot_plans = self.run_stage3(
            lyric_lines, audio_analysis, song_info, creative_intent
        )

        # 阶段 4
        output = self.run_stage4(shot_plans, audio_analysis, audio_path, output_path)

        logger.info("=" * 60)
        logger.info(f"完成！输出: {output}")
        logger.info("=" * 60)

        return output


def main():
    parser = argparse.ArgumentParser(
        description="MAD 动漫自动化剪辑系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--video-dir", default="data/videos", help="动漫视频目录")
    parser.add_argument("--music-id", required=True, help="网易云音乐 ID")
    parser.add_argument("--intent", default="", help="创作者意图描述")
    parser.add_argument("--output", default="output/mad.mp4", help="输出路径")
    parser.add_argument("--skip-vlm", action="store_true", help="跳过 VLM 标注（使用已有嵌入）")
    parser.add_argument("--quality", default="lossless", help="音频下载音质")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4], help="只运行指定阶段")
    parser.add_argument("--config", default=None, help="配置文件路径")

    args = parser.parse_args()

    # 初始化
    config = load_config(args.config)
    init_embedding(config.embedding_model)

    pipeline = MADPipeline(config)

    if args.stage == 1:
        pipeline.run_stage1(args.video_dir, skip_vlm=args.skip_vlm)
    elif args.stage == 2:
        pipeline.run_stage2(args.music_id, args.quality)
    elif args.stage == 3:
        audio_path, lyric_lines, audio_analysis, song_info = pipeline.run_stage2(
            args.music_id, args.quality
        )
        pipeline.run_stage3(lyric_lines, audio_analysis, song_info, args.intent)
    elif args.stage == 4:
        audio_path, lyric_lines, audio_analysis, song_info = pipeline.run_stage2(
            args.music_id, args.quality
        )
        shot_plans = pipeline.run_stage3(
            lyric_lines, audio_analysis, song_info, args.intent
        )
        pipeline.run_stage4(shot_plans, audio_analysis, audio_path, args.output)
    else:
        pipeline.run(
            video_dir=args.video_dir,
            music_id=args.music_id,
            creative_intent=args.intent,
            output_path=args.output,
            skip_vlm=args.skip_vlm,
            quality=args.quality,
        )


if __name__ == "__main__":
    main()
