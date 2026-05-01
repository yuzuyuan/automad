"""向量数据库接口 —— 基于 ChromaDB"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from shared.types import ShotAnnotation
from shared.utils import get_embedding, mmr_rerank

logger = logging.getLogger(__name__)


class EmbeddingStore:
    """ChromaDB 向量存储封装

    存储结构：
    - collection: "shots"
    - embedding: VLM 标注文本 → text-embedding-3-small 向量
    - metadata: ShotAnnotation 的 JSON 序列化
    - id: shot_id
    """

    def __init__(
        self,
        persist_path: str = "data/chroma_db",
        collection_name: str = "shots",
        embedding_model: str = "text-embedding-3-small",
    ):
        import chromadb

        self.persist_path = Path(persist_path)
        self.persist_path.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(self.persist_path))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.embedding_model = embedding_model

        logger.info(f"向量库就绪: {self.persist_path} (共 {self.collection.count()} 条)")

    def insert(self, annotation: ShotAnnotation) -> str:
        """插入单个镜头标注"""
        searchable_text = annotation.to_searchable_text()
        embedding = get_embedding(searchable_text).tolist()
        metadata = self._annotation_to_metadata(annotation)

        self.collection.upsert(
            ids=[annotation.shot_id],
            embeddings=[embedding],
            metadatas=[metadata],
            documents=[searchable_text],
        )
        return annotation.shot_id

    def insert_batch(self, annotations: list[ShotAnnotation]) -> list[str]:
        """批量插入镜头标注"""
        if not annotations:
            return []

        texts = [a.to_searchable_text() for a in annotations]
        embeddings = get_embedding(texts)
        ids = [a.shot_id for a in annotations]
        metadatas = [self._annotation_to_metadata(a) for a in annotations]

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            documents=texts,
        )
        logger.info(f"批量插入 {len(annotations)} 条记录")
        return ids

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        filters: Optional[dict] = None,
        apply_mmr: bool = True,
    ) -> list[tuple[float, ShotAnnotation]]:
        """语义检索

        Args:
            query_text: 自然语言查询（如 Target_Prompt）
            top_k: 返回数量
            filters: ChromaDB where 条件
            apply_mmr: 是否应用 MMR 多样性重排

        Returns:
            [(similarity_score, ShotAnnotation), ...]
        """
        query_embedding = get_embedding(query_text).tolist()

        n_results = top_k * 3 if apply_mmr else top_k

        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filters,
            include=["metadatas", "distances", "documents"],
        )

        ids_list = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]

        if not ids_list:
            return []

        # ChromaDB 返回的是 Cosine 距离，转为相似度
        candidates: list[tuple[float, ShotAnnotation]] = []
        for shot_id, distance, meta in zip(ids_list, distances, metadatas):
            similarity = 1.0 - distance  # cosine 相似度 = 1 - distance
            annotation = self._metadata_to_annotation(shot_id, meta)
            candidates.append((similarity, annotation))

        if apply_mmr and len(candidates) > 1:
            q_emb = np.array(query_embedding, dtype=np.float32)
            candidates = mmr_rerank(q_emb, candidates, lambda_=0.7)

        return candidates[:top_k]

    def query_by_shot_type(
        self, query_text: str, shot_types: list[str], top_k: int = 5
    ) -> list[tuple[float, ShotAnnotation]]:
        """按镜头类型过滤检索"""
        filters = {"shot_type": {"$in": shot_types}} if shot_types else None
        return self.query(query_text, top_k=top_k, filters=filters)

    def delete_shot(self, shot_id: str) -> None:
        self.collection.delete(ids=[shot_id])

    def clear(self) -> None:
        """清空集合"""
        all_ids = self.collection.get()["ids"]
        if all_ids:
            self.collection.delete(ids=all_ids)

    def count(self) -> int:
        return self.collection.count()

    # ---- 私有方法 ----

    @staticmethod
    def _annotation_to_metadata(annotation: ShotAnnotation) -> dict:
        """将 ShotAnnotation 展平为 ChromaDB metadata"""
        return {
            "source_video": annotation.source_video,
            "time_start": annotation.time_range[0],
            "time_end": annotation.time_range[1],
            "duration_sec": annotation.duration_sec,
            "character_count": annotation.character_count,
            "is_solo": annotation.is_solo,
            "is_group": annotation.is_group,
            "primary_action": annotation.primary_action,
            "action_intensity": str(annotation.action_intensity) if annotation.action_intensity else "",
            "background": annotation.background,
            "setting_type": str(annotation.setting_type) if annotation.setting_type else "",
            "shot_type": str(annotation.shot_type) if annotation.shot_type else "",
            "camera_angle": str(annotation.camera_angle) if annotation.camera_angle else "",
            "camera_movement": str(annotation.camera_movement) if annotation.camera_movement else "",
            "mood": ", ".join(annotation.mood) if annotation.mood else "",
            "narrative_function": str(annotation.narrative_function) if annotation.narrative_function else "",
            "symbolic_elements": ", ".join(annotation.symbolic_elements) if annotation.symbolic_elements else "",
            "visual_complexity": annotation.visual_complexity,
            "is_flashback": annotation.is_flashback,
            # ChromaDB metadata 不能存嵌套结构，完整数据存为 JSON 字符串
            "_full_json": json.dumps(_shot_to_dict(annotation), ensure_ascii=False),
        }

    @staticmethod
    def _metadata_to_annotation(shot_id: str, meta: dict) -> ShotAnnotation:
        """从 ChromaDB metadata 恢复 ShotAnnotation"""
        full_json = meta.get("_full_json", "{}")
        data = json.loads(full_json) if isinstance(full_json, str) else full_json
        return _dict_to_shot(shot_id, data)


def _shot_to_dict(shot: ShotAnnotation) -> dict:
    """ShotAnnotation → 可 JSON 序列化的 dict"""
    return {
        "shot_id": shot.shot_id,
        "source_video": shot.source_video,
        "time_range": list(shot.time_range),
        "duration_sec": shot.duration_sec,
        "characters": [
            {"name": c.name, "ref_match": c.ref_match, "emotion": c.emotion, "expression_detail": c.expression_detail}
            for c in shot.characters
        ],
        "character_count": shot.character_count,
        "is_solo": shot.is_solo,
        "is_group": shot.is_group,
        "primary_action": shot.primary_action,
        "action_intensity": str(shot.action_intensity) if shot.action_intensity else "",
        "action_direction": shot.action_direction,
        "background": shot.background,
        "setting_type": str(shot.setting_type) if shot.setting_type else "",
        "dominant_color_palette": shot.dominant_color_palette,
        "lighting": str(shot.lighting) if shot.lighting else "",
        "shot_type": str(shot.shot_type) if shot.shot_type else "",
        "camera_angle": str(shot.camera_angle) if shot.camera_angle else "",
        "camera_movement": str(shot.camera_movement) if shot.camera_movement else "",
        "mood": shot.mood,
        "narrative_function": str(shot.narrative_function) if shot.narrative_function else "",
        "symbolic_elements": shot.symbolic_elements,
        "visual_complexity": shot.visual_complexity,
        "has_text_overlay": shot.has_text_overlay,
        "is_flashback": shot.is_flashback,
    }


def _dict_to_shot(shot_id: str, data: dict) -> ShotAnnotation:
    """dict → ShotAnnotation"""
    from shared.types import ActionIntensity, ShotType, CameraAngle, CameraMovement, SettingType, Lighting, NarrativeFunction

    def _enum(cls, val, default):
        try:
            return cls(val)
        except (ValueError, TypeError):
            return default

    return ShotAnnotation(
        shot_id=data.get("shot_id", shot_id),
        source_video=data.get("source_video", ""),
        time_range=tuple(data.get("time_range", [0.0, 0.0])),
        duration_sec=data.get("duration_sec", 0.0),
        characters=[CharacterInfo(**c) for c in data.get("characters", [])],
        character_count=data.get("character_count", 0),
        is_solo=data.get("is_solo", False),
        is_group=data.get("is_group", False),
        primary_action=data.get("primary_action", ""),
        action_intensity=_enum(ActionIntensity, data.get("action_intensity"), ActionIntensity.STATIC),
        action_direction=data.get("action_direction"),
        background=data.get("background", ""),
        setting_type=_enum(SettingType, data.get("setting_type"), SettingType.INDOOR),
        dominant_color_palette=data.get("dominant_color_palette", []),
        lighting=_enum(Lighting, data.get("lighting"), Lighting.SOFT),
        shot_type=_enum(ShotType, data.get("shot_type"), ShotType.MEDIUM),
        camera_angle=_enum(CameraAngle, data.get("camera_angle"), CameraAngle.EYE_LEVEL),
        camera_movement=_enum(CameraMovement, data.get("camera_movement"), CameraMovement.STATIC),
        mood=data.get("mood", []),
        narrative_function=_enum(NarrativeFunction, data.get("narrative_function"), NarrativeFunction.EXPOSITION),
        symbolic_elements=data.get("symbolic_elements", []),
        visual_complexity=data.get("visual_complexity", 0.5),
        has_text_overlay=data.get("has_text_overlay", False),
        is_flashback=data.get("is_flashback", False),
    )
