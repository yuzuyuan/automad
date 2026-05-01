"""LRC 歌词文件解析器"""

from __future__ import annotations

import re
import logging
from pathlib import Path

from shared.types import LyricLine

logger = logging.getLogger(__name__)

_LRC_LINE_RE = re.compile(r"\[(\d{1,3}):(\d{2})(?:[\.:](\d{2,3}))?](.*)")


def parse_lrc(lrc_path: str | Path) -> list[LyricLine]:
    """解析 LRC 文件为结构化的歌词列表"""
    parser = LrcParser()
    return parser.parse(lrc_path)


class LrcParser:
    """LRC 格式歌词解析器

    支持格式：
    - [mm:ss.xx]text
    - [mm:ss:xx]text
    - 多时间标签：[00:01.00][00:15.30]重复歌词

    处理逻辑：
    - 原文歌词和翻译合并为同一 LyricLine
    - 无时间标签的行视为上一句的翻译
    """

    def parse(self, lrc_path: str | Path) -> list[LyricLine]:
        path = Path(lrc_path)
        if not path.exists():
            raise FileNotFoundError(f"LRC 文件不存在: {lrc_path}")

        raw_text = path.read_text(encoding="utf-8")
        return self.parse_text(raw_text)

    def parse_text(self, lrc_text: str) -> list[LyricLine]:
        # 第一遍：提取所有带时间标签的行
        timed_lines: list[tuple[float, str]] = []

        for line in lrc_text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue

            matches = list(_LRC_LINE_RE.finditer(line))
            if not matches:
                continue

            for m in matches:
                minutes = int(m.group(1))
                seconds = int(m.group(2))
                fraction_str = m.group(3) or "0"
                fraction = int(fraction_str) / (100 if len(fraction_str) == 2 else 1000)
                ts = minutes * 60 + seconds + fraction
                text = m.group(4).strip()
                if text:
                    timed_lines.append((ts, text))

        if not timed_lines:
            logger.warning("LRC 文件中未找到有效的时间标签行")
            return []

        timed_lines.sort(key=lambda x: x[0])

        # 合并完全重复的时间戳（歌词循环标记常出现）
        merged: list[tuple[float, str]] = []
        seen: set[tuple[float, str]] = set()
        for ts, text in timed_lines:
            key = (round(ts, 2), text)
            if key not in seen:
                seen.add(key)
                merged.append((ts, text))

        # 构建 LyricLine 列表，计算每行的 end_ts
        lyrics: list[LyricLine] = []
        for i, (ts, text) in enumerate(merged):
            next_ts = merged[i + 1][0] if i + 1 < len(merged) else ts + 3.0
            # 最长一句不超过 8 秒
            end_ts = min(next_ts, ts + 8.0)

            lyrics.append(LyricLine(
                index=i,
                start_ts=ts,
                end_ts=end_ts,
                text_original=text,
                text_translated="",
            ))

        # 尝试分离翻译行（简单启发式：包含中文的行可能是翻译）
        self._heuristic_translate_merge(lyrics)

        logger.info(f"解析完成: {len(lyrics)} 句歌词")
        return lyrics

    def _heuristic_translate_merge(self, lyrics: list[LyricLine]) -> None:
        """启发式翻译行合并——将连续中日文行配对"""
        import unicodedata

        def _has_cjk(text: str) -> bool:
            return any("CJK" in unicodedata.name(c, "") for c in text)

        def _has_kana(text: str) -> bool:
            return any(
                "HIRAGANA" in unicodedata.name(c, "") or "KATAKANA" in unicodedata.name(c, "")
                for c in text
            )

        # 简单策略：将相邻时间戳相近(<0.3s)的两个 LyricLine 合并为原文+翻译
        merged: list[LyricLine] = []
        skip_idx: set[int] = set()

        for i, line in enumerate(lyrics):
            if i in skip_idx:
                continue
            if i + 1 < len(lyrics):
                next_line = lyrics[i + 1]
                if abs(line.start_ts - next_line.start_ts) < 0.3:
                    # 判断哪个是日文哪个是中文
                    if _has_kana(line.text_original) and _has_cjk(next_line.text_original):
                        line.text_translated = next_line.text_original
                        skip_idx.add(i + 1)
                    elif _has_cjk(line.text_original) and _has_kana(next_line.text_original):
                        line.text_translated = line.text_original
                        line.text_original = next_line.text_original
                        skip_idx.add(i + 1)
            merged.append(line)

        lyrics[:] = merged
