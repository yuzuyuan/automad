// MAD Studio — Frontend Logic

let currentStep = 1;
const totalSteps = 5;

// =============================================================================
// Step Navigation
// =============================================================================

function goToStep(n) {
  if (n < 1 || n > totalSteps) return;
  currentStep = n;

  document.querySelectorAll(".step-panel").forEach(p => p.classList.remove("active"));
  document.getElementById("panel-step" + n).classList.add("active");

  document.querySelectorAll(".steps-nav .step").forEach((s, i) => {
    s.classList.remove("active", "done");
    if (i + 1 === n) s.classList.add("active");
    if (i + 1 < n) s.classList.add("done");
  });

  document.getElementById("btn-prev").disabled = (n === 1);
  document.getElementById("btn-next").textContent = (n === totalSteps) ? "完成" : "下一步";
  document.getElementById("step-indicator").textContent = n + " / " + totalSteps;
}

function nextStep() {
  if (currentStep < totalSteps) goToStep(currentStep + 1);
}
function prevStep() {
  if (currentStep > 1) goToStep(currentStep - 1);
}

// Click nav steps directly
document.querySelectorAll(".steps-nav .step").forEach(el => {
  el.addEventListener("click", () => {
    const step = parseInt(el.dataset.step);
    goToStep(step);
  });
});

// =============================================================================
// API Helpers
// =============================================================================

async function api(method, url, body) {
  const opts = { method, headers: { "Content-Type": "application/json" } };
  if (body) opts.body = JSON.stringify(body);
  const resp = await fetch(url, opts);
  return resp.json();
}

function statusEl(id, text, ok) {
  const el = document.getElementById(id);
  el.textContent = text || "";
  el.style.color = ok === false ? "var(--danger)" : "var(--success)";
}

// =============================================================================
// Step 1: API Keys
// =============================================================================

async function saveKeys() {
  const deepseekKey = document.getElementById("key-deepseek").value.trim();
  const dashscopeKey = document.getElementById("key-dashscope").value.trim();
  const openaiKey = document.getElementById("key-openai").value.trim();
  const cookie = document.getElementById("cookie-netease").value.trim();

  const r1 = await api("POST", "/api/keys/set", {
    deepseek_key: deepseekKey,
    dashscope_key: dashscopeKey,
    openai_key: openaiKey,
  });

  if (cookie) {
    const r2 = await api("POST", "/api/cookie/set", { cookie });
    statusEl("keys-status", r2.message, r2.ok);
  } else {
    statusEl("keys-status", r1.message, r1.ok);
  }
}

// =============================================================================
// Step 2: Videos
// =============================================================================

async function scanVideos() {
  const dir = document.getElementById("video-dir").value.trim();
  if (!dir) {
    statusEl("video-status", "请输入目录路径", false);
    return;
  }

  statusEl("video-status", "扫描中...");
  const r = await api("POST", "/api/videos/set-dir", { path: dir });

  if (!r.ok) {
    statusEl("video-status", r.message, false);
    return;
  }

  statusEl("video-status", `发现 ${r.count} 个视频文件`);
  document.getElementById("video-count").textContent = `（共 ${r.count} 个）`;

  const listEl = document.getElementById("video-list");
  listEl.innerHTML = "";
  r.videos.forEach(v => {
    const li = document.createElement("li");
    li.innerHTML = `<span class="file-name">${escHtml(v.name)}</span><span class="file-size">${v.size_mb} MB</span>`;
    listEl.appendChild(li);
  });

  document.getElementById("video-list-container").classList.remove("hidden");
}

// =============================================================================
// Step 3: Music Search
// =============================================================================

let selectedSongId = "";
let selectedSongData = null;

async function searchMusic() {
  const keyword = document.getElementById("search-keyword").value.trim();
  if (!keyword) {
    statusEl("search-status", "请输入搜索关键词", false);
    return;
  }

  statusEl("search-status", "搜索中...");
  const r = await api("POST", "/api/music/search", { keyword });

  if (!r.ok) {
    statusEl("search-status", r.message, false);
    return;
  }

  statusEl("search-status", `找到 ${r.count} 首歌曲`);
  document.getElementById("search-results-container").classList.remove("hidden");

  const listEl = document.getElementById("search-results");
  listEl.innerHTML = "";
  r.songs.forEach(song => {
    const li = document.createElement("li");
    li.innerHTML = `
      <img class="song-thumb" src="${escAttr(song.pic_url)}?param=60y60" alt="" loading="lazy"
           onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%2242%22 height=%2242%22><rect fill=%22%23333%22 width=%2242%22 height=%2242%22/></svg>'">
      <div class="song-detail">
        <div class="song-name">${escHtml(song.name)}</div>
        <div class="song-artist">${escHtml(song.artists)} · ${escHtml(song.album)}</div>
      </div>
    `;
    li.addEventListener("click", () => selectSong(song, li));
    listEl.appendChild(li);
  });
}

async function selectSong(song, liEl) {
  document.querySelectorAll("#search-results li").forEach(l => l.classList.remove("selected"));
  liEl.classList.add("selected");
  selectedSongId = song.id;

  statusEl("search-status", `加载 "${song.name}" 详情...`);
  const r = await api("POST", "/api/music/info", { id: song.id });

  if (!r.ok) {
    statusEl("search-status", r.message, false);
    return;
  }

  selectedSongData = r.song;
  statusEl("search-status", "");

  document.getElementById("song-detail-container").classList.remove("hidden");
  document.getElementById("song-pic").src = r.song.pic_url + "?param=240y240";
  document.getElementById("song-name").textContent = r.song.name;
  document.getElementById("song-artist").textContent = "歌手: " + r.song.artist;
  document.getElementById("song-album").textContent = "专辑: " + r.song.album;
  document.getElementById("song-duration").textContent = "时长: " + r.song.duration_str;
  document.getElementById("song-id-display").textContent = r.song.id;

  // lyrics
  const lyricEl = document.getElementById("lyric-text");
  if (r.song.lyrics) {
    lyricEl.textContent = r.song.lyrics;
  } else {
    lyricEl.textContent = "（无歌词）";
  }
}

// =============================================================================
// Step 5: Run Pipeline
// =============================================================================

let currentTaskId = "";
let eventSource = null;

async function runPipeline() {
  if (!selectedSongId) {
    alert("请先在「选歌」步骤中选择一首歌曲");
    goToStep(3);
    return;
  }

  const videoDir = document.getElementById("video-dir").value.trim();
  const intent = document.getElementById("creative-intent").value.trim();
  const outputName = document.getElementById("output-name").value.trim() || "mad_output.mp4";
  const quality = document.getElementById("audio-quality").value;
  const skipVlm = document.getElementById("skip-vlm").checked;

  goToStep(5);

  // Reset UI
  document.getElementById("progress-container").classList.remove("hidden");
  document.getElementById("log-container").classList.remove("hidden");
  document.getElementById("result-container").classList.add("hidden");
  document.getElementById("progress-bar").style.width = "0%";
  document.getElementById("progress-text").textContent = "";
  document.getElementById("progress-stage").textContent = "";
  document.getElementById("log-output").textContent = "";

  document.getElementById("btn-run").classList.add("hidden");
  document.getElementById("btn-cancel").classList.remove("hidden");
  document.getElementById("task-indicator").classList.remove("hidden");
  document.getElementById("task-status-text").textContent = "运行中...";

  const r = await api("POST", "/api/pipeline/run", {
    video_dir: videoDir,
    music_id: selectedSongId,
    intent: intent,
    output: "webapp_output/" + outputName.replace(/[<>:"/\\|?*]/g, "_"),
    skip_vlm: skipVlm,
    quality: quality,
  });

  if (!r.ok) {
    alert("启动失败: " + (r.message || "未知错误"));
    resetRunUI();
    return;
  }

  currentTaskId = r.task_id;
  startProgressStream(currentTaskId);
}

function startProgressStream(taskId) {
  if (eventSource) eventSource.close();

  eventSource = new EventSource("/api/pipeline/progress/" + taskId);

  eventSource.onmessage = function(e) {
    const data = JSON.parse(e.data);
    updateProgress(data);

    if (data.status === "done" || data.status === "error") {
      eventSource.close();
      onPipelineDone(data);
    }
  };

  eventSource.onerror = function() {
    // SSE will auto-reconnect; if stream ends, handle gracefully
  };
}

function updateProgress(data) {
  document.getElementById("progress-bar").style.width = data.percent + "%";
  document.getElementById("progress-text").textContent = data.message || "";
  document.getElementById("progress-stage").textContent = "阶段: " + (data.stage || "...");
  document.getElementById("log-output").textContent = (data.logs || []).join("\n");
  document.getElementById("task-status-text").textContent = data.percent + "%";
}

function onPipelineDone(data) {
  resetRunUI();

  if (data.status === "done") {
    document.getElementById("result-container").classList.remove("hidden");
    document.getElementById("result-path").textContent = "输出文件: " + data.output_path;

    const downloadBtn = document.getElementById("btn-download");
    downloadBtn.href = "/api/output/download?path=" + encodeURIComponent(data.output_path);
    downloadBtn.setAttribute("download", "");

    // Store path for "open folder"
    document.getElementById("result-container").dataset.outputPath = data.output_path;

    document.getElementById("task-indicator").classList.add("hidden");
  } else {
    alert("运行失败: " + (data.error || "未知错误"));
    document.getElementById("task-indicator").classList.add("hidden");
    document.getElementById("btn-run").classList.remove("hidden");
    document.getElementById("btn-cancel").classList.add("hidden");
  }
}

function resetRunUI() {
  document.getElementById("btn-run").classList.remove("hidden");
  document.getElementById("btn-cancel").classList.add("hidden");
}

async function cancelPipeline() {
  if (currentTaskId) {
    await api("POST", "/api/pipeline/cancel/" + currentTaskId);
  }
  if (eventSource) eventSource.close();
  resetRunUI();
  document.getElementById("task-indicator").classList.add("hidden");
}

async function openOutputFolder() {
  const path = document.getElementById("result-container").dataset.outputPath || "";
  await api("POST", "/api/output/open-folder", { path });
}

// =============================================================================
// Utilities
// =============================================================================

function escHtml(s) {
  if (!s) return "";
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function escAttr(s) {
  if (!s) return "";
  return s.replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

// =============================================================================
// Init
// =============================================================================

goToStep(1);
