import json
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import librosa
import numpy as np


def compute_motion_scores(
    segments: List[Dict[str, Any]],
    target_positions: Dict[int, List[float]],
    fps: float,
) -> Dict[int, float]:
    scores: Dict[int, float] = {}

    for seg in segments:
        sid = int(seg["segment_id"])
        s = int(seg["start_frame"])
        e = int(seg["end_frame"])

        centers = []
        for frame_idx in range(s, e + 1):
            box = target_positions.get(frame_idx)
            if not box:
                continue
            x1, y1, x2, y2 = box
            centers.append(((x1 + x2) * 0.5, (y1 + y2) * 0.5))

        if len(centers) < 3:
            scores[sid] = 0.0
            continue

        xy = np.asarray(centers, dtype=np.float32)
        v = np.linalg.norm(np.diff(xy, axis=0), axis=1) * fps
        a = np.abs(np.diff(v)) * fps

        speed_score = float(np.percentile(v, 90))
        accel_score = float(np.percentile(a, 85)) if len(a) else 0.0
        scores[sid] = speed_score * 0.7 + accel_score * 0.3

    return scores


def _extract_audio_wav(video_path: str, wav_path: str) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-ac",
        "1",
        "-ar",
        "22050",
        wav_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def compute_audio_scores(video_path: str, segments: List[Dict[str, Any]]) -> Dict[int, float]:
    if not segments:
        return {}

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = str(Path(tmpdir) / "audio.wav")
        _extract_audio_wav(video_path, wav_path)
        y, sr = librosa.load(wav_path, sr=22050, mono=True)

    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)

    scores: Dict[int, float] = {}
    for seg in segments:
        sid = int(seg["segment_id"])
        s = float(seg["start_sec"])
        e = float(seg["end_sec"])
        mask = (times >= s) & (times <= e)
        if not np.any(mask):
            scores[sid] = 0.0
            continue
        local = rms[mask]
        mean = float(np.mean(local))
        peak = float(np.percentile(local, 95))
        scores[sid] = mean * 0.5 + peak * 0.5

    return scores


def _minmax(values: List[float]) -> List[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if math.isclose(lo, hi):
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def rank_segments(
    segments: List[Dict[str, Any]],
    motion_scores: Dict[int, float],
    audio_scores: Dict[int, float],
    mode: str = "motion+audio",
) -> List[Dict[str, Any]]:
    if not segments:
        return []

    mids = [int(s["segment_id"]) for s in segments]
    mvals = [motion_scores.get(i, 0.0) for i in mids]
    avals = [audio_scores.get(i, 0.0) for i in mids]

    mn = _minmax(mvals)
    an = _minmax(avals)

    ranked = []
    for i, seg in enumerate(segments):
        if mode == "motion":
            score = mn[i]
        elif mode == "audio":
            score = an[i]
        else:
            score = 0.7 * mn[i] + 0.3 * an[i]

        row = dict(seg)
        row["motion_score"] = float(mvals[i])
        row["audio_score"] = float(avals[i])
        row["rank_score"] = float(score)
        ranked.append(row)

    ranked.sort(key=lambda x: x["rank_score"], reverse=True)
    return ranked


def write_ranked_segments(path: str, ranked_segments: List[Dict[str, Any]]) -> None:
    payload = {"ranked_segments": ranked_segments}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f)
