import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from embed import compute_embedding, load_target_embeddings, weighted_similarity


@dataclass
class IdentifyConfig:
    similarity_threshold: float = 0.55
    min_frames: int = 5
    top_n: int = 1


def _load_json(path: str) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _crop(frame: np.ndarray, box: Sequence[float]) -> Optional[np.ndarray]:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def _track_observations(detections: Dict[str, Any]) -> Dict[int, List[Tuple[int, List[float]]]]:
    by_track: Dict[int, List[Tuple[int, List[float]]]] = {}
    for chunk in detections.get("chunks", []):
        for row in chunk.get("records", []):
            frame_idx = int(row.get("frame_idx", -1))
            if frame_idx < 0:
                continue
            for trk in row.get("tracks", []):
                tid = int(trk.get("track_id", -1))
                box = trk.get("bbox")
                if tid < 0 or not isinstance(box, list) or len(box) != 4:
                    continue
                by_track.setdefault(tid, []).append((frame_idx, [float(v) for v in box]))
    return by_track


def _sample_observations(
    observations: List[Tuple[int, List[float]]],
    max_samples: int = 10,
) -> List[Tuple[int, List[float]]]:
    if len(observations) <= max_samples:
        return observations
    idx = np.linspace(0, len(observations) - 1, num=max_samples, dtype=int)
    return [observations[int(i)] for i in idx]


def identify_target(
    tracklets_json_path: str,
    ref_images: List[str],
    video_path: str,
    config: Optional[IdentifyConfig] = None,
) -> List[int]:
    cfg = config or IdentifyConfig()
    detections = _load_json(tracklets_json_path)
    track_obs = _track_observations(detections)

    candidates = {
        tid: obs
        for tid, obs in track_obs.items()
        if len(obs) >= int(cfg.min_frames)
    }
    if not candidates:
        return []

    target_embeddings = load_target_embeddings(ref_current_game_list=ref_images)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    scored: List[Tuple[int, float]] = []
    for tid, obs in candidates.items():
        sample = _sample_observations(obs, max_samples=10)
        sim_values: List[float] = []

        for frame_idx, box in sample:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = cap.read()
            if not ok:
                continue
            crop = _crop(frame, box)
            if crop is None or crop.size == 0:
                continue
            try:
                emb = compute_embedding(crop)
            except Exception:
                continue
            sim_values.append(float(weighted_similarity(emb, target_embeddings)))

        if not sim_values:
            continue
        scored.append((tid, float(np.mean(sim_values))))

    cap.release()

    scored.sort(key=lambda x: x[1], reverse=True)
    passing = [tid for tid, score in scored if score >= float(cfg.similarity_threshold)]
    return passing[: max(1, int(cfg.top_n))]
