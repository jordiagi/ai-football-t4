import json
from pathlib import Path
from typing import Any, Dict, List, Sequence


def frames_to_segments(
    presence_frames: Sequence[int],
    fps: float,
    pre_seconds: float,
    post_seconds: float,
    min_clip_seconds: float,
    merge_gap_seconds: float,
    total_frames: int,
) -> List[Dict[str, Any]]:
    if not presence_frames:
        return []

    frames = sorted(set(int(f) for f in presence_frames))
    max_gap_frames = int(merge_gap_seconds * fps)

    raw = []
    s = frames[0]
    e = frames[0]
    for f in frames[1:]:
        if f - e <= max_gap_frames:
            e = f
        else:
            raw.append((s, e))
            s, e = f, f
    raw.append((s, e))

    pre = int(pre_seconds * fps)
    post = int(post_seconds * fps)
    min_len = int(min_clip_seconds * fps)

    out: List[Dict[str, Any]] = []
    for idx, (a, b) in enumerate(raw):
        start = max(0, a - pre)
        end = min(total_frames - 1, b + post)
        if end - start + 1 < min_len:
            continue
        out.append(
            {
                "segment_id": idx,
                "start_frame": int(start),
                "end_frame": int(end),
                "start_sec": float(start / fps),
                "end_sec": float(end / fps),
                "duration_sec": float((end - start + 1) / fps),
            }
        )

    return out


def write_segments_json(
    path: str,
    segments: List[Dict[str, Any]],
    extra_meta: Dict[str, Any],
) -> None:
    payload = {"segments": segments, **extra_meta}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f)
