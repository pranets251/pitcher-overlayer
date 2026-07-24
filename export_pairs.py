#!/usr/bin/env python3
"""Release-sync and export consecutive pitch pairs with a baked hold."""
import json
import os
import subprocess
from itertools import combinations as choose_subsets
from itertools import product
from pathlib import Path

import cv2
import numpy as np
from analyze import detect_paths
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).parent
RESULT = Path(os.environ.get("PITCHER_RESULTS", ROOT / "output/analysis.json"))
OUTPUT = Path(os.environ.get("PITCHER_OUTPUT_DIR", RESULT.parent))
OUTPUT.mkdir(parents=True, exist_ok=True)
data = json.loads(RESULT.read_text())
video = Path(data["source"])
if not video.is_absolute():
    video = ROOT / video
fps = float(data["meta"]["fps"])
# Always restart from the immutable motion-candidate list. This makes reruns
# idempotent instead of repeatedly filtering an already validated subset.
pitches = [
    {"pitch": number, "delivery_peak": peak}
    for number, peak in enumerate(data["delivery_candidates"], 1)
]


def frame_at(number):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(number))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {number}")
    return cv2.resize(frame, (1280, 720))


def pose_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[40:650, 250:850]
    white = (hsv[:, :, 2] > 145) & (hsv[:, :, 1] < 105)
    red = (((hsv[:, :, 0] < 15) | (hsv[:, :, 0] > 165)) &
           (hsv[:, :, 1] > 75) & (hsv[:, :, 2] > 85))
    return (white | red).astype(np.float32)


def reference_release():
    cache_path = Path(os.environ.get(
        "PITCHER_CALIBRATION", OUTPUT / "release-calibration.json"))
    size = video.stat().st_size
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
        if cache.get("source_size") == size:
            return int(cache["reference_frame"]), "verified cached release"
    # With no ball tracking, establish the session reference inside the constrained
    # pre-peak release window of a middle delivery. Exact release is less important
    # than matching every other delivery to this identical body pose.
    anchor = pitches[len(pitches) // 2]
    frame = max(0, round((anchor["delivery_peak"] - 3.45) * fps))
    cache_path.write_text(json.dumps({
        "source_size": size,
        "reference_frame": frame,
        "reference_pitch": anchor["pitch"],
    }, indent=2))
    return frame, "automatic common-pose anchor"


reference_frame, sync_method = reference_release()
reference_mask = pose_mask(frame_at(reference_frame))
pose_offsets = (-8, 0, 8)
reference_sequence = {
    offset: pose_mask(frame_at(max(0, reference_frame + offset)))
    for offset in pose_offsets
}


def synchronized_release(pitch):
    if pitch.get("verified_release_frame"):
        return int(pitch["verified_release_frame"])
    center = max(0, round((pitch["delivery_peak"] - 3.45) * fps))
    best = None
    # Motion peaks drift relative to release, so search a constrained 1.6-second
    # pre-peak window. This prevents a similar pose elsewhere in the delivery from
    # becoming the synchronization point.
    radius = round(.8 * fps)
    start = max(0, center - radius)
    stop = center + radius
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    masks = {}
    read_start = max(0, start + min(pose_offsets))
    cap.set(cv2.CAP_PROP_POS_FRAMES, read_start)
    for frame_number in range(read_start, stop + max(pose_offsets) + 1):
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (1280, 720))
        masks[frame_number] = pose_mask(frame)
    for candidate in range(start, stop + 1):
        distances = []
        for offset in pose_offsets:
            candidate_mask = masks.get(candidate + offset)
            if candidate_mask is None:
                continue
            reference = reference_sequence[offset]
            intersection = float(np.sum(reference * candidate_mask))
            mass = float(np.sum(reference) + np.sum(candidate_mask))
            distances.append(1.0 - (2.0 * intersection / max(mass, 1.0)))
        if not distances:
            continue
        difference = float(np.mean(distances))
        if best is None or difference < best[0]:
            best = (difference, candidate)
    cap.release()
    if best is None:
        return center, 1.0
    return best[1], best[0]


release_matches = {p["pitch"]: synchronized_release(p) for p in pitches}
# Release-sequence scores form a low-distance pitch group and a high-distance
# between-pitch-motion group. Split them at their largest observed gap.
ordered_scores = sorted(score for _, score in release_matches.values())
lower_split = max(1, int(len(ordered_scores) * .25))
upper_split = max(lower_split + 1, int(len(ordered_scores) * .9))
gaps = [(ordered_scores[i+1] - ordered_scores[i], i)
        for i in range(lower_split, min(upper_split, len(ordered_scores) - 1))]
_, split_index = max(gaps, default=(0, len(ordered_scores)-1))
pose_cutoff = (ordered_scores[split_index] + ordered_scores[split_index+1]) / 2 \
    if len(ordered_scores) > 1 else 1.0
pitches = [p for p in pitches if release_matches[p["pitch"]][1] <= pose_cutoff]
for sequence_number, pitch in enumerate(pitches, 1):
    pitch["motion_candidate"] = pitch["pitch"]
    pitch["pitch"] = sequence_number
release_frames = {
    p["pitch"]: release_matches[p["motion_candidate"]][0] for p in pitches
}


def read_clip(pitch, before=.7, after=1.05):
    release = release_frames[pitch["pitch"]]
    start = max(0, release - round(before * fps))
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(round((before + after) * fps)):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(frame, (1280, 720)))
    cap.release()
    return frames, round(before * fps), start


def registration_shift(reference, offset):
    a = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32)[:330]
    b = cv2.cvtColor(offset, cv2.COLOR_BGR2GRAY).astype(np.float32)[:330]
    window = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (xoff, yoff), response = cv2.phaseCorrelate(a, b, window)
    if response < .05 or abs(xoff) > 80 or abs(yoff) > 80:
        return 0., 0.
    return xoff, yoff


GROUP_COLORS = {
    "Four-seam-like": "#00D6A3",
    "Sinker-like": "#FF7A59",
    "Changeup-like": "#7B8CFF",
    "Slider-like": "#FFD23F",
    "Curveball-like": "#E85AAD",
    "Cutter-like": "#3DCAE3",
}
model_dir = Path(os.environ.get(
    "PITCHER_MODEL",
    ROOT / "model/yolov4-tiny-baseball-416"))
cached_ballpaths = data.get("ballpaths", [])
if (len(cached_ballpaths) == len(pitches)
        and [path.get("pitch") for path in cached_ballpaths]
        == list(range(1, len(pitches)+1))):
    ballpaths = cached_ballpaths
else:
    ballpaths = detect_paths(
        video, [pitch["delivery_peak"] for pitch in pitches], model_dir)

# Pose matching provides a reliable coarse synchronization point, but the ball can
# leave the hand a few frames earlier or later relative to the same body pose.
# Refine only within a tight five-frame window using the first frame of each
# already-validated ball track. The bound prevents a stray detector hit from
# replacing the pose anchor with an unrelated moment in the delivery.
pose_release_frames = dict(release_frames)
for path in ballpaths:
    pitch_number = path["pitch"]
    if not path.get("points") or pitch_number not in release_frames:
        continue
    pose_frame = release_frames[pitch_number]
    tracked_frame = int(path["points"][0]["frame"])
    correction = max(-5, min(5, tracked_frame - pose_frame))
    release_frames[pitch_number] = pose_frame + correction


def classify_ballpaths(paths):
    """Assign best-fit pitch names from front-view movement and relative flight time."""
    features = []
    for path in paths:
        points = np.array([[p["x"], p["y"]] for p in path["points"]], dtype=float)
        source_t = np.linspace(0, 1, len(points))
        sample_t = np.linspace(0, 1, 16)
        curve = np.column_stack([
            np.interp(sample_t, source_t, points[:, axis])
            for axis in (0, 1)])
        curve -= curve[0]
        early_x = np.polyfit(sample_t[:7], curve[:7, 0], 1)
        early_y = np.polyfit(sample_t[:7], curve[:7, 1], 1)
        predicted = np.column_stack([
            np.polyval(early_x, sample_t[-3:]),
            np.polyval(early_y, sample_t[-3:])])
        departure = curve[-3:].mean(0) - predicted.mean(0)
        duration = path["points"][-1]["t"] - path["points"][0]["t"]
        features.append([curve[-1, 0], curve[-1, 1],
                         departure[0], departure[1], duration])
    values = np.asarray(features)
    standardized = (values-values.mean(0)) / (values.std(0)+1e-6)
    standardized[:, 2:4] *= 1.5
    candidates = []
    for cluster_count in range(3, min(6, len(paths)-1)+1):
        candidate_labels = KMeans(
            cluster_count, random_state=42, n_init=50).fit_predict(standardized)
        candidates.append((
            silhouette_score(standardized, candidate_labels),
            cluster_count,
            candidate_labels))
    _, cluster_count, labels = max(candidates)
    centers = {
        label: values[labels == label].mean(0)
        for label in range(cluster_count)}
    remaining = set(centers)
    names = {}
    fastball = min(remaining, key=lambda label: centers[label][4])
    names[fastball] = "Four-seam-like"; remaining.remove(fastball)
    if remaining:
        slider = min(remaining, key=lambda label: centers[label][0])
        names[slider] = "Slider-like"; remaining.remove(slider)
    if remaining:
        changeup = max(remaining, key=lambda label: centers[label][4])
        names[changeup] = "Changeup-like"; remaining.remove(changeup)
    if remaining:
        curveball = max(remaining, key=lambda label: centers[label][1])
        names[curveball] = "Curveball-like"; remaining.remove(curveball)
    leftovers = sorted(remaining, key=lambda label: centers[label][0])
    leftover_names = ["Sinker-like"] if len(leftovers) == 1 \
        else ["Cutter-like", "Sinker-like"]
    for label, name in zip(leftovers, leftover_names):
        names[label] = name
    for feature_index, (path, label) in enumerate(zip(paths, labels)):
        path["pitch_type"] = names[int(label)]
        path["path"] = np.asarray([
            np.interp(np.linspace(0, 1, 16),
                      np.linspace(0, 1, len(path["points"])),
                      [point[axis] for point in path["points"]])
            for axis in ("x", "y")]).T.round(5).tolist()
        path["late_departure"] = features[feature_index][2:4]
    return {name: sum(path["pitch_type"] == name for path in paths)
            for name in sorted(set(names.values()))}


pitch_type_counts = classify_ballpaths(ballpaths)
def tunnel_score(first, second):
    """Compare complete tracked paths through 150 ms, then reward late separation."""
    release_a = release_frames[first["pitch"]] / fps
    release_b = release_frames[second["pitch"]] / fps
    points_a = np.array([
        [point["t"]-release_a, point["x"], point["y"]]
        for point in first["points"]], dtype=float)
    points_b = np.array([
        [point["t"]-release_b, point["x"], point["y"]]
        for point in second["points"]], dtype=float)
    start = max(0.0, points_a[:, 0].min(), points_b[:, 0].min())
    end = min(points_a[:, 0].max(), points_b[:, 0].max())
    if end <= start + .06:
        return None
    times = np.linspace(start, end, 18)
    curve_a = np.column_stack([
        np.interp(times, points_a[:, 0], points_a[:, axis])
        for axis in (1, 2)])
    curve_b = np.column_stack([
        np.interp(times, points_b[:, 0], points_b[:, axis])
        for axis in (1, 2)])
    distances = np.linalg.norm(curve_a-curve_b, axis=1)
    decision_time = min(.15, start + .65*(end-start))
    decision_mask = times <= decision_time
    if not np.any(decision_mask):
        return None
    early = float(np.mean(distances[decision_mask]))
    decision = float(np.interp(decision_time, times, distances))
    late = float(np.mean(distances[-max(3, len(distances)//4):]))
    separation = max(0.0, late-decision)
    # Decision-point proximity is primary; early path similarity is secondary.
    closeness = .75*decision + .25*early
    closeness_score = np.exp(-closeness/.055)
    separation_score = 1-np.exp(-separation/.035)
    # Decision-window closeness leads; late separation remains a substantial
    # requirement. The geometric combination requires both qualities.
    score = 100 * closeness_score**.60 * separation_score**.40
    return {
        "tunnel_score": round(float(score), 1),
        "decision_distance": round(decision, 4),
        "early_distance": round(early, 4),
        "finish_distance": round(late, 4),
    }


best_matchups = {}
for first_index, first_path in enumerate(ballpaths):
    for second_path in ballpaths[first_index+1:]:
        if first_path["pitch_type"] == second_path["pitch_type"]:
            continue
        estimate = tunnel_score(first_path, second_path)
        if not estimate:
            continue
        key = " | ".join(sorted(
            (first_path["pitch_type"], second_path["pitch_type"])))
        candidate = (
            estimate["tunnel_score"], first_path, second_path, estimate)
        if key not in best_matchups or candidate[0] > best_matchups[key][0]:
            best_matchups[key] = candidate
selected_pairs = sorted(
    best_matchups.values(), key=lambda item: item[0], reverse=True)


def color_bgr(group):
    value = GROUP_COLORS[group].lstrip("#")
    return np.array([int(value[4:6], 16), int(value[2:4], 16),
                     int(value[0:2], 16)], dtype=np.float32)


def tracked_box(path, frame_number):
    points = path["points"]
    if frame_number < points[0]["frame"] or frame_number > points[-1]["frame"]:
        return None
    for left, right in zip(points, points[1:]):
        if left["frame"] <= frame_number <= right["frame"]:
            span = max(1, right["frame"]-left["frame"])
            amount = (frame_number-left["frame"]) / span
            return tuple(left[key]+amount*(right[key]-left[key])
                         for key in ("x", "y", "w", "h"))
    point = points[-1]
    return tuple(point[key] for key in ("x", "y", "w", "h"))


def tint_ball(frame, path, frame_number):
    box = tracked_box(path, frame_number)
    if box is None:
        return frame
    x, y, w, h = box
    cx, cy = int(x*frame.shape[1]), int(y*frame.shape[0])
    rx = int(np.clip(w*frame.shape[1]*.27, 6, 24))
    ry = int(np.clip(h*frame.shape[0]*.27, 6, 24))
    x1, x2 = max(0, cx-rx), min(frame.shape[1], cx+rx+1)
    y1, y2 = max(0, cy-ry), min(frame.shape[0], cy+ry+1)
    roi = frame[y1:y2, x1:x2]
    if not roi.size:
        return frame
    yy, xx = np.ogrid[y1:y2, x1:x2]
    ellipse = ((xx-cx)/max(rx, 1))**2 + ((yy-cy)/max(ry, 1))**2 <= 1
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    ball = ellipse & (hsv[:, :, 2] > 145) & (hsv[:, :, 1] < 115)
    if np.any(ball):
        target = color_bgr(path["pitch_type"])
        luminance = (.68+.32*(hsv[:, :, 2]/255.0))[..., None]
        colored = np.clip(target*luminance, 0, 255).astype(np.uint8)
        roi[ball] = colored[ball]
    return frame


combinations = []
for rank, (_, first_path, second_path, estimate) in enumerate(selected_pairs, 1):
    first = pitches[first_path["pitch"]-1]
    second = pitches[second_path["pitch"]-1]
    first_frames, release_idx, first_start = read_clip(first)
    second_frames, _, second_start = read_clip(second)
    count = min(len(first_frames), len(second_frames))
    xoff, yoff = registration_shift(
        first_frames[max(0, release_idx-10)],
        second_frames[max(0, release_idx-10)])
    matrix = np.float32([[1, 0, -xoff], [0, 1, -yoff]])
    temp = OUTPUT / f"combination-{rank:03d}-temp.mp4"
    final = OUTPUT / f"combination-{rank:03d}.mp4"
    writer = cv2.VideoWriter(
        str(temp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 720))
    decision_index = release_idx + round(.15 * fps)
    realtime_hold_frames = round(fps * 4)
    for index in range(count):
        first_tinted = tint_ball(
            first_frames[index].copy(), first_path, first_start+index)
        second_tinted = tint_ball(
            second_frames[index].copy(), second_path, second_start+index)
        aligned = cv2.warpAffine(second_tinted, matrix, (1280, 720))
        frame = cv2.addWeighted(first_tinted, .55, aligned, .45, 0)
        for _ in range(4):
            writer.write(frame)
        # The encoded file is four-times slowed and defaults to 4x playback.
        # Four encoded seconds therefore become a one-second real-time hold.
        if index == decision_index:
            for _ in range(realtime_hold_frames):
                writer.write(frame)
    writer.release()
    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(temp), "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "25", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(final),
    ], check=True)
    temp.unlink()
    combination = {
        "pitch_a": first["pitch"],
        "pitch_b": second["pitch"],
        "type_a": first_path["pitch_type"],
        "type_b": second_path["pitch_type"],
        "color_a": GROUP_COLORS[first_path["pitch_type"]],
        "color_b": GROUP_COLORS[second_path["pitch_type"]],
        "video": final.name,
        # Frames are written four times. This marks roughly 150 ms after release
        # in source time, when a hitter must begin committing to a decision.
        "decision_time_seconds": round((release_idx / fps + .15) * 4, 3),
        "decision_hold_realtime_seconds": 1.0,
        "release_frames": [
            release_frames[first["pitch"]],
            release_frames[second["pitch"]],
        ],
    }
    combination.update(estimate)
    combinations.append(combination)
    print(f"[{rank}/{len(selected_pairs)}] {final.name}", flush=True)
data.pop("pitch_type_counts", None)
data.pop("top_pairs", None)
data.pop("type_matchups", None)
data.pop("movement_group_colors", None)
for pitch in pitches:
    pitch.pop("pitch_type", None)
    pitch.pop("path", None)
    pitch.pop("late_departure", None)
data["pitches"] = pitches
data["ballpaths"] = ballpaths
data["pitch_type_counts"] = pitch_type_counts
data["pitch_type_colors"] = GROUP_COLORS
data["combinations"] = combinations
data["synchronization"] = {
    "method": f"{sync_method} with bounded ball-release refinement",
    "reference_frame": reference_frame,
    "pose_release_frames": pose_release_frames,
    "release_frames": release_frames,
    "ball_release_corrections": {
        pitch: release_frames[pitch] - pose_release_frames[pitch]
        for pitch in release_frames
    },
    "pose_validation_cutoff": round(float(pose_cutoff), 4),
    "accepted_deliveries": len(pitches),
    "motion_candidates": len(release_matches),
}
RESULT.write_text(json.dumps(data, indent=2))
print(f"Exported {len(combinations)} best cross-type tunnel overlays")
