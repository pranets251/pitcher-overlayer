from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

ROOT = Path(__file__).parent
JOBS = ROOT / "jobs"
CALIBRATIONS = ROOT / "calibrations"
DEMO = ROOT / "demo"
for folder in (JOBS, CALIBRATIONS, DEMO): folder.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024 * 1024
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()


@app.after_request
def no_cache(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


def valid_job_id(value: str) -> bool:
    return len(value) == 32 and all(c in "0123456789abcdef" for c in value)


def set_job(job_id: str, **values):
    with jobs_lock:
        jobs.setdefault(job_id, {}).update(values)
        state = dict(jobs[job_id])
    folder = JOBS / job_id
    folder.mkdir(exist_ok=True)
    (folder / "status.json").write_text(json.dumps(state, indent=2))


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/api/results")
def results():
    job_id = request.args.get("job", "")
    if job_id:
        if not valid_job_id(job_id): return jsonify({"error": "Invalid job"}), 400
        path = JOBS / job_id / "output" / "analysis.json"
        media_base = f"/jobs/{job_id}/"
    else:
        path = DEMO / "analysis.json"
        if not path.exists(): path = ROOT / "output" / "analysis.json"
        media_base = "/demo/" if path.parent == DEMO else "/output/"
    if not path.exists(): return jsonify({"status": "pending"}), 202
    payload = json.loads(path.read_text()); payload["media_base"] = media_base
    payload["is_demo"] = not bool(job_id)
    return jsonify(payload)


@app.get("/api/status/<job_id>")
def status(job_id):
    if not valid_job_id(job_id): return jsonify({"error": "Invalid job"}), 400
    with jobs_lock: state = jobs.get(job_id)
    if state is None:
        path = JOBS / job_id / "status.json"
        if not path.exists(): return jsonify({"error": "Job not found"}), 404
        state = json.loads(path.read_text())
    return jsonify(state)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def run_analysis(job_id: str, video: Path, content_hash: str):
    folder = JOBS / job_id; output = folder / "output"; output.mkdir(exist_ok=True)
    result = output / "analysis.json"
    try:
        set_job(job_id, status="running", message="Splitting video into individual pitches")
        subprocess.run([sys.executable, str(ROOT/"analyze.py"), str(video),
                        "--out", str(result)], check=True, cwd=ROOT)
        set_job(job_id, status="running", message="Tracking ballpaths and selecting the best pitch-type matchups")
        env = os.environ.copy()
        env.update(PITCHER_RESULTS=str(result), PITCHER_OUTPUT_DIR=str(output),
                   PITCHER_CALIBRATION=str(CALIBRATIONS / f"{content_hash}.json"))
        subprocess.run([sys.executable, str(ROOT/"export_pairs.py")],
                       check=True, cwd=ROOT, env=env)
        set_job(job_id, status="complete", message="Analysis complete", result_url=f"/?job={job_id}")
    except Exception as exc:
        set_job(job_id, status="error", message=f"Analysis failed: {exc}")


@app.post("/api/analyze")
def analyze_upload():
    upload = request.files.get("video")
    if not upload or not upload.filename: return jsonify({"error": "Choose a video file"}), 400
    suffix = Path(upload.filename).suffix.lower()
    if suffix not in {".mov", ".mp4", ".m4v", ".avi"}:
        return jsonify({"error": "Supported formats: MOV, MP4, M4V, AVI"}), 400
    job_id = uuid.uuid4().hex
    folder = JOBS / job_id; folder.mkdir()
    target = folder / (secure_filename(Path(upload.filename).stem)[:80] + suffix)
    upload.save(target)
    content_hash = sha256_file(target)
    set_job(job_id, status="queued", message="Upload complete; analysis queued", content_hash=content_hash)
    threading.Thread(target=run_analysis, args=(job_id, target, content_hash), daemon=True).start()
    return jsonify({"job_id": job_id, "status": "queued", "message": "Upload complete; analysis started"}), 202


@app.get("/demo/<path:name>")
def demo_file(name): return send_from_directory(DEMO, name)


@app.get("/jobs/<job_id>/<path:name>")
def job_file(job_id, name):
    if not valid_job_id(job_id): return "Invalid job", 400
    return send_from_directory(JOBS / job_id / "output", name)


@app.get("/output/<path:name>")
def legacy_output(name): return send_from_directory(ROOT / "output", name)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5177, debug=False)
