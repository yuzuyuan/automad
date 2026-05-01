"""音乐与歌词获取——封装已有的 Netease_url 模块"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_NETEASE_DIR = Path(__file__).resolve().parent.parent / "Netease_url"


class MusicFetcher:
    """统一的音乐获取接口，封装网易云 API

    复用 Netease_url/ 中的 music_api.py 和 download_lyrics.py，
    不启动 Flask 服务，直接调用底层函数。
    """

    def __init__(self, download_dir: str = "downloads"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)

        if str(_NETEASE_DIR) not in sys.path:
            sys.path.insert(0, str(_NETEASE_DIR))

    def get_lyrics(self, music_id: str) -> str:
        """获取 LRC 格式歌词文本

        Returns:
            LRC 格式的完整歌词字符串
        """
        from music_api import lyric_v1, APIException
        from cookie_manager import CookieManager

        cm = CookieManager(cookie_file=str(_NETEASE_DIR / "cookie.txt"))
        cookies = cm.parse_cookies()

        try:
            result = lyric_v1(int(music_id), cookies)
            lrc = result.get("lrc", {}).get("lyric", "")
            tlyric = result.get("tlyric", {}).get("lyric", "")

            if not lrc:
                raise RuntimeError(f"歌曲 {music_id} 无歌词")

            # 合并原文和翻译
            if tlyric:
                return f"{lrc}\n{tlyric}"
            return lrc

        except APIException as e:
            raise RuntimeError(f"获取歌词失败 (API): {e}") from e

    def get_song_info(self, music_id: str) -> dict:
        """获取歌曲基本信息"""
        from music_api import name_v1, APIException

        try:
            result = name_v1(int(music_id))
            songs = result.get("songs", [])
            if not songs:
                raise RuntimeError(f"未找到歌曲 {music_id}")
            song = songs[0]
            return {
                "id": music_id,
                "name": song.get("name", ""),
                "artist": ", ".join(a["name"] for a in song.get("ar", [])),
                "album": song.get("al", {}).get("name", ""),
                "pic_url": song.get("al", {}).get("picUrl", ""),
                "duration_ms": song.get("dt", 0),
            }
        except APIException as e:
            raise RuntimeError(f"获取歌曲信息失败: {e}") from e

    def download_audio(self, music_id: str, quality: str = "lossless") -> Path:
        """下载音频文件，返回本地路径"""
        from music_api import url_v1, name_v1, APIException
        from cookie_manager import CookieManager

        import requests

        cm = CookieManager(cookie_file=str(_NETEASE_DIR / "cookie.txt"))
        cookies = cm.parse_cookies()
        song_info = self.get_song_info(music_id)

        # 构建安全文件名
        safe_name = f"{song_info['artist']} - {song_info['name']}"
        safe_name = "".join(c for c in safe_name if c not in r'<>:"/\|?*')

        # 检查是否已下载
        for ext in [".flac", ".mp3", ".m4a"]:
            existing = self.download_dir / f"{safe_name}{ext}"
            if existing.exists():
                logger.info(f"音频已存在: {existing}")
                return existing

        try:
            url_result = url_v1(int(music_id), quality, cookies)
        except APIException as e:
            raise RuntimeError(f"获取下载链接失败: {e}") from e

        data_list = url_result.get("data", [])
        if not data_list or not data_list[0].get("url"):
            # 降级到标准音质
            logger.warning("无损链接获取失败，降级到标准音质")
            url_result = url_v1(int(music_id), "standard", cookies)
            data_list = url_result.get("data", [])

        if not data_list or not data_list[0].get("url"):
            raise RuntimeError(f"无法获取歌曲 {music_id} 的下载链接")

        url = data_list[0]["url"]
        file_type = data_list[0].get("type", "mp3")
        output_path = self.download_dir / f"{safe_name}.{file_type}"

        logger.info(f"下载音频: {output_path.name}")
        resp = requests.get(url, timeout=120, stream=True)
        resp.raise_for_status()

        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        logger.info(f"下载完成: {output_path}")
        return output_path

    def fetch_all(self, music_id: str, quality: str = "lossless") -> tuple[Path, str, dict]:
        """一站式获取：下载音频 + 歌词 + 歌曲信息

        Returns:
            (audio_path, lrc_text, song_info_dict)
        """
        audio_path = self.download_audio(music_id, quality)
        lrc_text = self.get_lyrics(music_id)
        song_info = self.get_song_info(music_id)
        return audio_path, lrc_text, song_info
