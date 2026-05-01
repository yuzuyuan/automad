"""镜头变速适配器"""

from __future__ import annotations

import logging
from enum import Enum

logger = logging.getLogger(__name__)


class SpeedTier(Enum):
    """变速层级"""
    NORMAL = (1.00, 1.00)
    SUBTLE = (0.92, 1.08)
    MODERATE = (0.80, 1.25)
    EXTREME = (0.50, 2.00)


class SpeedAdapter:
    """根据目标时长和镜头原始时长，计算最优变速系数

    规则：
    - 优先在 SUBTLE 范围内变速（几乎无感知）
    - 其次 MODERATE 范围（可察觉但自然）
    - 仅特效段落使用 EXTREME
    """

    def __init__(self):
        self.subtle = SpeedTier.SUBTLE.value
        self.moderate = SpeedTier.MODERATE.value
        self.extreme = SpeedTier.EXTREME.value

    def fit(
        self,
        source_duration: float,
        target_duration: float,
        allow_extreme: bool = False,
    ) -> tuple[float, str]:
        """计算最佳变速系数

        Args:
            source_duration: 源镜头时长（秒）
            target_duration: 目标时长（秒）
            allow_extreme: 是否允许极端变速

        Returns:
            (speed_factor, tier_name)
        """
        if source_duration <= 0 or target_duration <= 0:
            return 1.0, "normal"

        ratio = source_duration / target_duration

        # 如果已经匹配（±3% 容差）
        if 0.97 <= ratio <= 1.03:
            return 1.0, "normal"

        # Subtle 范围
        if self.subtle[0] <= ratio <= self.subtle[1]:
            return ratio, "subtle"

        # Moderate 范围
        if self.moderate[0] <= ratio <= self.moderate[1]:
            return ratio, "moderate"

        # Extreme 范围
        if allow_extreme and self.extreme[0] <= ratio <= self.extreme[1]:
            return ratio, "extreme"

        # 无法匹配：裁剪
        if ratio < 1.0:
            # 源比目标短 → 优先慢放到 moderate 上限
            return self.moderate[1], "moderate"
        else:
            # 源比目标长 → 优先快放到 moderate 下限
            return self.moderate[0], "moderate"

    def multi_segment_fit(
        self,
        source_durations: list[float],
        target_durations: list[float],
        allow_extreme: bool = False,
    ) -> list[tuple[float, str]]:
        """批量变速适配"""
        if len(source_durations) != len(target_durations):
            raise ValueError(f"长度不匹配: {len(source_durations)} vs {len(target_durations)}")

        return [
            self.fit(s, t, allow_extreme)
            for s, t in zip(source_durations, target_durations)
        ]
