"""镜头边界检测（Shot Boundary Detection）——基于 PySceneDetect"""

from __future__ import annotations

import logging
from pathlib import Path

import scenedetect
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector, AdaptiveDetector

from shared.types import ShotAnnotation

logger = logging.getLogger(__name__)


class ShotDetector:
    """基于内容感知的镜头切分器

    使用 PySceneDetect 进行物理镜头边界检测，而非固定帧步长采样。
    默认使用 AdaptiveDetector 以自适应阈值方式检测。
    """

    def __init__(
        self,
        min_shot_duration: float = 0.5,
        detector_type: str = "adaptive",  # "adaptive" | "content"
    ):
        self.min_shot_duration = min_shot_duration
        self.detector_type = detector_type

    def detect(self, video_path: str | Path) -> list[ShotAnnotation]:
        """对单个视频进行镜头切分"""
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        video_name = path.stem
        logger.info(f"检测镜头: {video_name}")

        video_manager = VideoManager([str(path)])
        scene_manager = SceneManager()

        if self.detector_type == "adaptive":
            scene_manager.add_detector(AdaptiveDetector(
                adaptive_threshold=3.0,
                min_scene_len=int(self.min_shot_duration * video_manager.get_framerate()),
            ))
        else:
            scene_manager.add_detector(ContentDetector(
                threshold=27.0,
                min_scene_len=int(self.min_shot_duration * video_manager.get_framerate()),
            ))

        video_manager.set_downscale_factor()
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)

        scene_list = scene_manager.get_scene_list()
        fps = video_manager.get_framerate()

        shots: list[ShotAnnotation] = []
        for i, (start_frame, end_frame) in enumerate(scene_list):
            start_sec = start_frame.get_seconds()
            end_sec = end_frame.get_seconds()
            duration = end_sec - start_sec

            if duration < self.min_shot_duration:
                continue

            shot_id = f"{video_name}_{i:04d}"
            shots.append(ShotAnnotation(
                shot_id=shot_id,
                source_video=str(path),
                time_range=(start_sec, end_sec),
                duration_sec=round(duration, 3),
            ))

        video_manager.release()
        logger.info(f"检测到 {len(shots)} 个镜头 (共 {sum(s.duration_sec for s in shots):.1f}s)")

        return shots

    def detect_batch(self, video_dir: str | Path, extensions: tuple[str, ...] = (".mp4", ".mkv", ".avi", ".mov")) -> list[ShotAnnotation]:
        """批量检测目录下所有视频的镜头"""
        video_dir = Path(video_dir)
        all_shots: list[ShotAnnotation] = []

        for ext in extensions:
            for video_path in video_dir.glob(f"*{ext}"):
                try:
                    shots = self.detect(video_path)
                    all_shots.extend(shots)
                except Exception as e:
                    logger.error(f"处理 {video_path.name} 失败: {e}")

        logger.info(f"批量检测完成: {len(all_shots)} 个镜头 (来自 {video_dir})")
        return all_shots
