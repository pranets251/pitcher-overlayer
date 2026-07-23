#!/usr/bin/env python3
"""Front-facing bullpen pitch segmentation and chonyy YOLO ball tracking."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def detect_paths(video: Path, candidates: list[float], model_dir: Path) -> list[dict]:
    import tensorflow as tf

    model = tf.saved_model.load(str(model_dir))
    infer = model.signatures["serving_default"]
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    paths = []
    for pitch_no, peak in enumerate(candidates, 1):
        # The smoothed delivery peak drifts between arm deceleration and follow-through;
        # keep a wider pre-peak window so early releases are not clipped.
        start = max(0.0, peak - 4.65)
        end = max(start + .3, peak - 1.35)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
        raw = []
        count = int((end - start) * fps)
        for j in range(count):
            ok, frame = cap.read()
            if not ok:
                break
            image = cv2.cvtColor(cv2.resize(frame, (416, 416)), cv2.COLOR_BGR2RGB)
            image = image[None].astype(np.float32) / 255.0
            pred = next(iter(infer(input_1=tf.constant(image)).values()))
            boxes, conf = pred[:, :, :4], pred[:, :, 4:]
            b, s, _, valid = tf.image.combined_non_max_suppression(
                tf.reshape(boxes, (1, -1, 1, 4)),
                tf.reshape(conf, (1, -1, tf.shape(conf)[-1])),
                10, 10, .45, .32,
            )
            for k in range(int(valid[0])):
                y1, x1, y2, x2 = [float(v) for v in b[0, k]]
                x, y = (x1 + x2) / 2, (y1 + y2) / 2
                if .28 < x < .72 and .04 < y < .72 and (y2-y1) < .32:
                    raw.append({"frame": int(start * fps) + j, "t": start + j/fps,
                                "x": x, "y": y, "w": x2-x1, "h": y2-y1,
                                "confidence": float(s[0, k])})
        # Score short contiguous YOLO/SORT-like tracks; prefer forward (increasing y/size) motion.
        runs = []
        for d in raw:
            attached = False
            for run in reversed(runs[-8:]):
                last = run[-1]
                gap = d["frame"] - last["frame"]
                dist = float(np.hypot(d["x"]-last["x"], d["y"]-last["y"]))
                if 0 < gap <= 2 and dist < .14:
                    run.append(d); attached = True; break
            if not attached:
                runs.append([d])
        def quality(run):
            if len(run) < 2: return -1
            span = run[-1]["frame"] - run[0]["frame"] + 1
            advance = run[-1]["y"] - run[0]["y"]
            growth = run[-1]["h"] - run[0]["h"]
            return len(run) * 2 + span + 8 * max(0, advance) + 3 * max(0, growth) + sum(x["confidence"] for x in run)
        best = max(runs, key=quality, default=[])
        if len(best) >= 3 and best[-1]["y"] > best[0]["y"] - .02:
            paths.append({"pitch": pitch_no, "delivery_peak": peak,
                          "release_time": round(best[0]["t"], 3), "points": best,
                          "confidence": round(float(np.mean([p["confidence"] for p in best])), 3)})
        print(f"[{pitch_no:02d}/{len(candidates)}] {peak:7.2f}s -> {len(best)} tracked frames", flush=True)
    cap.release()
    return paths


def motion_candidates(video: Path) -> tuple[list[float], dict]:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    scores, times = [], []
    prev = None
    stride = 3
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            small = cv2.resize(frame, (256, 144))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            # Pitcher occupies this region; exclude the foreground catcher.
            roi = gray[8:136, 55:155]
            if prev is not None:
                scores.append(float(np.mean(cv2.absdiff(roi, prev))))
                times.append(i / fps)
            prev = roi
        i += 1
    cap.release()
    a = np.asarray(scores)
    # Smooth across 0.7 s. A delivery creates a broad, unmistakable motion burst.
    width = max(3, int(fps / stride * .7))
    smooth = np.convolve(a, np.ones(width) / width, mode="same")
    threshold = max(float(np.percentile(smooth, 76)), 2.15)
    peaks = []
    for idx in range(1, len(smooth) - 1):
        if smooth[idx] < threshold or smooth[idx] < smooth[idx - 1] or smooth[idx] < smooth[idx + 1]:
            continue
        t = times[idx]
        if not peaks or t - peaks[-1][0] >= 7.0:
            peaks.append((t, float(smooth[idx])))
        elif smooth[idx] > peaks[-1][1]:
            peaks[-1] = (t, float(smooth[idx]))
    meta = {"fps": fps, "frames": n, "duration": n / fps, "motion_threshold": threshold}
    return [round(t, 3) for t, _ in peaks], meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--motion-only", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("output/analysis.json"))
    args = ap.parse_args()
    peaks, meta = motion_candidates(args.video)
    # Every motion-detected delivery is retained. Ball detection is no longer a
    # prerequisite; release timing is resolved later by session-wide pose matching.
    result = {
        "source": str(args.video),
        "meta": meta,
        "delivery_candidates": peaks,
        "pitches": [
            {"pitch": number, "delivery_peak": peak}
            for number, peak in enumerate(peaks, 1)
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({"candidate_count": len(peaks), **meta, "times": peaks}, indent=2))


if __name__ == "__main__":
    main()
