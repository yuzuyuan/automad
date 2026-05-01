"""语义检索器 —— 将 TargetPrompt 匹配到视频镜头"""

from __future__ import annotations

import logging
from typing import Optional

from shared.types import TargetPrompt, ShotAnnotation, ShotPlan
from stage1_vision.embedding_store import EmbeddingStore

logger = logging.getLogger(__name__)


class ShotRetriever:
    """语义检索器：TargetPrompt → 向量检索 → 最优镜头匹配"""

    def __init__(self, store: EmbeddingStore):
        self.store = store

    def retrieve(
        self,
        target: TargetPrompt,
        top_k: int = 5,
    ) -> list[tuple[float, ShotAnnotation]]:
        """根据 target_prompt 检索最匹配的镜头

        Returns:
            [(相似度, ShotAnnotation), ...]，按相似度降序
        """
        return self.store.query(
            query_text=target.target_prompt,
            top_k=top_k,
            apply_mmr=True,
        )

    def retrieve_by_type(
        self,
        target: TargetPrompt,
        shot_types: list[str],
        top_k: int = 5,
    ) -> list[tuple[float, ShotAnnotation]]:
        """按镜头类型过滤后检索"""
        return self.store.query_by_shot_type(
            query_text=target.target_prompt,
            shot_types=shot_types,
            top_k=top_k,
        )

    def select_best(self, candidates: list[tuple[float, ShotAnnotation]]) -> Optional[ShotAnnotation]:
        """从候选中选择最佳镜头（简单策略：取相似度最高的）"""
        if not candidates:
            return None
        return candidates[0][1]

    def fill_plan(self, plan: ShotPlan, top_k: int = 5) -> ShotPlan:
        """为 ShotPlan 填充检索结果"""
        # 若有 preferred_shot_types，先按类型过滤检索
        if plan.target.preferred_shot_types:
            candidates = self.retrieve_by_type(
                plan.target,
                plan.target.preferred_shot_types,
                top_k=top_k,
            )
            # 若类型过滤后结果不够，补充通用检索
            if len(candidates) < top_k:
                more = self.retrieve(plan.target, top_k=top_k)
                # 合并去重
                existing_ids = {c[1].shot_id for c in candidates}
                for c in more:
                    if c[1].shot_id not in existing_ids:
                        candidates.append(c)
                    if len(candidates) >= top_k:
                        break
                candidates.sort(key=lambda x: x[0], reverse=True)
        else:
            candidates = self.retrieve(plan.target, top_k=top_k)

        plan.candidates = candidates
        plan.selected_shot = self.select_best(candidates)
        return plan

    def fill_all_plans(self, plans: list[ShotPlan]) -> list[ShotPlan]:
        """为所有 ShotPlan 填充检索结果"""
        for plan in plans:
            self.fill_plan(plan)
        logger.info(f"检索完成: {sum(1 for p in plans if p.selected_shot is not None)}/{len(plans)} 句匹配成功")
        return plans
