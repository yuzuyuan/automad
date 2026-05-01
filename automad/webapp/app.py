"""MAD 剪辑系统 Web 应用 — Flask 后端"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

from flask import (
    Flask, render_template, request, jsonify, Response,
    send_file, stream_with_context,
)
from werkzeug.utils import secure_filename

# 确保项目根目录在 sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("webapp")

app = Flask(__name__)
app.secret_key = os.urandom(24).hex()

# =============================================================================
# 会话状态存储（内存中，重启即清）
# =============================================================================
_sessions: dict[str, dict] = {}
_sessions_lock = threading.Lock()


def _get_state() -> dict:
    """获取当前会话的状态字典"""
    sid = request.cookies.get("mad_sid", "")
    if not sid or sid not in _sessions:
        sid = os.urandom(12).hex()
        _sessions[sid] = {
            "api_keys": {},
            "video_dir": "",
            "videos": [],
            "cookie": "",
        }
    return _sessions[sid]


def _set_state_cookie(response: Response) -> Response:
    sid = request.cookies.get("mad_sid", "")
    if sid and sid in _sessions:
        response.set_cookie("mad_sid", sid, max_age=86400, httponly=True)
    return response


# =============================================================================
# 进度追踪
# =============================================================================
_pipeline_progress: dict = {}
_progress_lock = threading.Lock()


class PipelineProgress:
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.stage = ""
        self.percent = 0
        self.message = ""
        self.logs: list[str] = []
        self.status = "idle"  # idle | running | done | error
        self.output_path = ""
        self.error = ""

    def update(self, stage: str, percent: int, message: str):
        with _progress_lock:
            self.stage = stage
            self.percent = percent
            self.message = message
            self.logs.append(f"[{stage}] {message}")

    def finish(self, output_path: str = ""):
        with _progress_lock:
            self.status = "done"
            self.percent = 100
            self.output_path = output_path

    def fail(self, error: str):
        with _progress_lock:
            self.status = "error"
            self.error = error
            self.logs.append(f"[ERROR] {error}")

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "stage": self.stage,
            "percent": self.percent,
            "message": self.message,
            "logs": self.logs[-20:],
            "status": self.status,
            "output_path": self.output_path,
            "error": self.error,
        }


# =============================================================================
# 页面路由
# =============================================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.after_request
def after_request(response: Response) -> Response:
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Cache-Control"] = "no-store"
    return _set_state_cookie(response)


# =============================================================================
# API — 配置
# =============================================================================

@app.route("/api/keys/set", methods=["POST"])
def api_set_keys():
    """设置 API 密钥"""
    data = request.get_json() or {}
    state = _get_state()
    state["api_keys"] = {
        "DEEPSEEK_API_KEY": data.get("deepseek_key", ""),
        "DASHSCOPE_API_KEY": data.get("dashscope_key", ""),
        "OPENAI_API_KEY": data.get("openai_key", ""),
    }
    for k, v in state["api_keys"].items():
        if v:
            os.environ[k] = v
    return jsonify({"ok": True, "message": "API 密钥已设置"})


@app.route("/api/cookie/set", methods=["POST"])
def api_set_cookie():
    """设置网易云 Cookie"""
    data = request.get_json() or {}
    state = _get_state()
    state["cookie"] = data.get("cookie", "").strip()

    cookie_path = _PROJECT_ROOT / "Netease_url" / "cookie.txt"
    if state["cookie"]:
        cookie_path.write_text(state["cookie"], encoding="utf-8")
        return jsonify({"ok": True, "message": "Cookie 已保存"})
    else:
        return jsonify({"ok": False, "message": "Cookie 不能为空"}), 400


# =============================================================================
# API — 视频管理
# =============================================================================

@app.route("/api/videos/set-dir", methods=["POST"])
def api_set_video_dir():
    """设置视频目录"""
    data = request.get_json() or {}
    dir_path = data.get("path", "").strip()
    state = _get_state()

    if not dir_path:
        return jsonify({"ok": False, "message": "目录路径不能为空"}), 400

    p = Path(dir_path)
    if not p.exists():
        return jsonify({"ok": False, "message": f"目录不存在: {dir_path}"}), 400
    if not p.is_dir():
        return jsonify({"ok": False, "message": f"不是有效目录: {dir_path}"}), 400

    state["video_dir"] = str(p.resolve())

    # 扫描视频文件
    video_exts = {".mp4", ".mkv", ".avi", ".mov", ".flv", ".webm", ".m4v"}
    videos = []
    for ext in video_exts:
        for vp in p.glob(f"*{ext}"):
            size_mb = vp.stat().st_size / (1024 * 1024)
            videos.append({
                "name": vp.name,
                "path": str(vp.resolve()),
                "size_mb": round(size_mb, 1),
            })

    state["videos"] = videos
    videos.sort(key=lambda v: v["name"])

    return jsonify({
        "ok": True,
        "dir": state["video_dir"],
        "count": len(videos),
        "videos": videos,
    })


@app.route("/api/videos/list", methods=["GET"])
def api_list_videos():
    """列出已扫描的视频"""
    state = _get_state()
    return jsonify({
        "dir": state["video_dir"],
        "count": len(state["videos"]),
        "videos": state["videos"],
    })


# =============================================================================
# API — 音乐搜索（复用 Netease_url）
# =============================================================================

@app.route("/api/music/search", methods=["POST"])
def api_music_search():
    """搜索网易云音乐"""
    data = request.get_json() or {}
    keyword = data.get("keyword", "").strip()
    if not keyword:
        return jsonify({"ok": False, "message": "搜索关键词不能为空"}), 400

    try:
        _netease_path = str(_PROJECT_ROOT / "Netease_url")
        if _netease_path not in sys.path:
            sys.path.insert(0, _netease_path)

        from music_api import search_music
        from cookie_manager import CookieManager

        cm = CookieManager(cookie_file=str(_PROJECT_ROOT / "Netease_url" / "cookie.txt"))
        cookies = cm.parse_cookies()

        results = search_music(keyword, cookies, limit=20)

        songs = []
        for r in results:
            songs.append({
                "id": str(r.get("id", "")),
                "name": r.get("name", ""),
                "artists": r.get("artists", ""),
                "album": r.get("album", ""),
                "pic_url": r.get("picUrl", ""),
            })

        return jsonify({"ok": True, "count": len(songs), "songs": songs})

    except Exception as e:
        logger.error(f"搜索失败: {e}")
        return jsonify({"ok": False, "message": f"搜索失败: {e}"}), 500


@app.route("/api/music/info", methods=["POST"])
def api_music_info():
    """获取歌曲详细信息和歌词"""
    data = request.get_json() or {}
    music_id = data.get("id", "").strip()
    if not music_id:
        return jsonify({"ok": False, "message": "歌曲 ID 不能为空"}), 400

    try:
        _netease_path = str(_PROJECT_ROOT / "Netease_url")
        if _netease_path not in sys.path:
            sys.path.insert(0, _netease_path)

        from music_api import name_v1, lyric_v1
        from cookie_manager import CookieManager

        cm = CookieManager(cookie_file=str(_PROJECT_ROOT / "Netease_url" / "cookie.txt"))
        cookies = cm.parse_cookies()

        song_info = name_v1(int(music_id))
        lyric_info = lyric_v1(int(music_id), cookies)

        songs = song_info.get("songs", [])
        if not songs:
            return jsonify({"ok": False, "message": "未找到歌曲信息"}), 404

        song = songs[0]

        lrc_text = ""
        tlyric_text = ""
        if lyric_info:
            lrc_text = lyric_info.get("lrc", {}).get("lyric", "")
            tlyric_text = lyric_info.get("tlyric", {}).get("lyric", "")

        return jsonify({
            "ok": True,
            "song": {
                "id": music_id,
                "name": song.get("name", ""),
                "artist": ", ".join(a["name"] for a in song.get("ar", [])),
                "album": song.get("al", {}).get("name", ""),
                "pic_url": song.get("al", {}).get("picUrl", ""),
                "duration_ms": song.get("dt", 0),
                "duration_str": f"{song.get('dt', 0) // 1000 // 60}:{song.get('dt', 0) // 1000 % 60:02d}",
                "lyrics": lrc_text,
                "lyrics_translated": tlyric_text,
            }
        })

    except Exception as e:
        logger.error(f"获取歌曲信息失败: {e}")
        return jsonify({"ok": False, "message": f"获取失败: {e}"}), 500


# =============================================================================
# API — Pipeline 执行
# =============================================================================

@app.route("/api/pipeline/run", methods=["POST"])
def api_pipeline_run():
    """启动 MAD Pipeline"""
    data = request.get_json() or {}
    state = _get_state()

    task_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + os.urandom(4).hex()
    progress = PipelineProgress(task_id)

    with _progress_lock:
        _pipeline_progress[task_id] = progress

    def _run():
        try:
            # 确保 API 密钥注入环境
            for k, v in state.get("api_keys", {}).items():
                if v:
                    os.environ[k] = v

            progress.update("init", 0, "正在初始化 Pipeline...")

            from shared.config import load_config
            from shared.llm_client import LLMClient, VLMClient
            from shared.utils import init_embedding

            config = load_config()

            # 覆盖配置
            video_dir = data.get("video_dir", state.get("video_dir", "data/videos"))
            music_id = data.get("music_id", "")
            creative_intent = data.get("intent", "")
            output_path = data.get("output", f"webapp_output/{task_id}/mad_output.mp4")
            skip_vlm = data.get("skip_vlm", False)
            quality = data.get("quality", "lossless")

            if not music_id:
                progress.fail("未选择歌曲")
                return
            if not video_dir:
                progress.fail("未设置视频目录")
                return

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            init_embedding(config.embedding_model)

            # ——— 阶段 1 ———
            progress.update("stage1", 5, "镜头检测中...")

            from stage1_vision.shot_detector import ShotDetector
            from stage1_vision.keyframe_extractor import KeyframeExtractor
            from stage1_vision.vlm_annotator import VLMAnnotator
            from stage1_vision.embedding_store import EmbeddingStore

            detector = ShotDetector(min_shot_duration=config.shot_min_duration)
            kf_extractor = KeyframeExtractor()
            vlm = VLMClient(
                provider=config.vlm_provider,
                model=config.vlm_model,
            )
            annotator = VLMAnnotator(
                character_ref_dir=config.character_ref_dir,
                vlm_client=vlm,
            )
            store = EmbeddingStore(
                persist_path=config.chroma_path,
                embedding_model=config.embedding_model,
            )

            shots = detector.detect_batch(video_dir)
            progress.update("stage1", 15, f"检测到 {len(shots)} 个镜头")

            if not skip_vlm:
                for i, shot in enumerate(shots):
                    pct = 15 + int((i / max(len(shots), 1)) * 25)
                    progress.update("stage1", pct, f"VLM 标注: {shot.shot_id} ({i+1}/{len(shots)})")

                    keyframes = kf_extractor.extract(shot, shot.source_video)
                    kf_paths = [p for _, p in keyframes]
                    annotator.annotate_shot(shot, kf_paths)
                    store.insert(shot)
            else:
                progress.update("stage1", 40, "跳过 VLM 标注")

            progress.update("stage1", 40, f"阶段 1 完成: {store.count()} 条记录")

            # ——— 阶段 2 ———
            progress.update("stage2", 45, "获取音频和歌词...")

            from stage2_audio.music_fetcher import MusicFetcher
            from stage2_audio.lrc_parser import LrcParser
            from stage2_audio.beat_detector import BeatDetector

            fetcher = MusicFetcher()
            lrc_parser = LrcParser()
            beat_detector = BeatDetector(sample_rate=config.sample_rate)

            audio_path, lrc_text, song_info = fetcher.fetch_all(music_id, quality)
            lyric_lines = lrc_parser.parse_text(lrc_text)
            audio_analysis = beat_detector.analyze(audio_path)

            progress.update("stage2", 55, f"阶段 2 完成: {len(lyric_lines)} 句歌词, BPM={audio_analysis.bpm:.1f}")

            # ——— 阶段 3 ———
            progress.update("stage3", 60, "LLM 全局叙事分析...")

            from stage3_director.global_director import GlobalDirector
            from stage3_director.shot_planner import ShotPlanner
            from stage3_director.retriever import ShotRetriever

            llm = LLMClient(
                provider=config.llm_provider,
                model=config.llm_model,
                max_tokens=config.llm_max_tokens,
                temperature=config.llm_temperature,
            )

            director = GlobalDirector(llm_client=llm)
            planner = ShotPlanner(llm_client=llm)
            retriever = ShotRetriever(store=store)

            global_context = director.analyze(
                lyric_lines, audio_analysis, creative_intent, song_info
            )

            progress.update("stage3", 70, f"全局叙事: {global_context.narrative_arc}")

            planner.set_context(global_context, audio_analysis)
            shot_plans = planner.generate_all(lyric_lines)
            shot_plans = retriever.fill_all_plans(shot_plans)

            matched = sum(1 for p in shot_plans if p.selected_shot is not None)
            progress.update("stage3", 80, f"阶段 3 完成: {matched}/{len(shot_plans)} 句匹配")

            # ——— 阶段 4 ———
            progress.update("stage4", 85, "构建时间线...")

            from stage4_editor.timeline_builder import TimelineBuilder
            from stage4_editor.beat_syncer import BeatSyncer
            from stage4_editor.openshot_engine import VideoRenderer

            timeline = TimelineBuilder()
            syncer = BeatSyncer(tolerance_ms=config.snap_tolerance_ms)
            renderer = VideoRenderer(config=config)

            segments = timeline.build(shot_plans, audio_analysis)

            all_snaps = []
            for seg in segments:
                synced = syncer.snap_all(seg.snap_results, audio_analysis)
                all_snaps.extend(synced)

            progress.update("stage4", 90, f"渲染输出: {output_path}")

            output = renderer.render(all_snaps, audio_path, output_path)

            progress.finish(str(output))

        except Exception as e:
            logger.error(f"Pipeline 失败: {traceback.format_exc()}")
            progress.fail(str(e))

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return jsonify({"ok": True, "task_id": task_id})


@app.route("/api/pipeline/progress/<task_id>", methods=["GET"])
def api_pipeline_progress(task_id: str):
    """SSE 流式推送 Pipeline 进度"""
    def _stream():
        last_log_count = 0
        while True:
            with _progress_lock:
                progress = _pipeline_progress.get(task_id)

            if progress is None:
                yield f"data: {json.dumps({'status': 'not_found'})}\n\n"
                break

            data = progress.to_dict()
            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

            if progress.status in ("done", "error"):
                break

            time.sleep(0.5)

    return Response(
        stream_with_context(_stream()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/pipeline/cancel/<task_id>", methods=["POST"])
def api_pipeline_cancel(task_id: str):
    """取消 Pipeline 任务"""
    with _progress_lock:
        progress = _pipeline_progress.get(task_id)
        if progress and progress.status == "running":
            progress.status = "error"
            progress.error = "用户取消"
    return jsonify({"ok": True})


# =============================================================================
# API — 输出下载
# =============================================================================

@app.route("/api/output/download", methods=["GET"])
def api_download_output():
    """下载输出视频"""
    path = request.args.get("path", "")
    if not path or not Path(path).exists():
        return jsonify({"ok": False, "message": "文件不存在"}), 404

    return send_file(
        path,
        as_attachment=True,
        download_name=Path(path).name,
        mimetype="video/mp4",
    )


@app.route("/api/output/open-folder", methods=["POST"])
def api_open_output_folder():
    """在文件管理器中打开输出目录"""
    data = request.get_json() or {}
    path = data.get("path", "")
    if not path:
        return jsonify({"ok": False})

    import subprocess
    folder = str(Path(path).parent)
    if sys.platform == "win32":
        subprocess.Popen(["explorer", folder])
    elif sys.platform == "darwin":
        subprocess.Popen(["open", folder])
    else:
        subprocess.Popen(["xdg-open", folder])

    return jsonify({"ok": True})


# =============================================================================
# 启动入口
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="MAD 剪辑系统 Web 应用")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=8080, help="端口")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  MAD 动漫自动化剪辑系统 — Web 界面")
    print(f"  访问地址: http://{args.host}:{args.port}")
    print("=" * 60 + "\n")

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
