import subprocess
from pathlib import Path
from typing import Any, Dict, List, Sequence

import cv2


def _run(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def extract_segment_clips(
    video_path: str,
    segments: List[dict],
    out_dir: str,
    prefix: str,
) -> List[str]:
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    clips: List[str] = []
    for i, seg in enumerate(segments):
        clip_path = outp / f"{prefix}_{i:03d}.mp4"
        start = float(seg["start_sec"])
        end = float(seg["end_sec"])
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-to",
            f"{end:.3f}",
            "-i",
            video_path,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(clip_path),
        ]
        _run(cmd)
        clips.append(str(clip_path))
    return clips


def concat_clips(clips: List[str], output_path: str) -> str:
    if not clips:
        return ""

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    list_file = out.parent / f"{out.stem}_concat.txt"
    with list_file.open("w", encoding="utf-8") as f:
        for clip in clips:
            f.write(f"file '{Path(clip).resolve()}'\n")

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        "-c",
        "copy",
        str(out),
    ]
    _run(cmd)
    return str(out)


def export_highlights(
    video_path: str,
    segments: List[dict],
    ranked_segments: List[dict],
    top_k_segments: int,
    output_dir: str,
) -> Dict[str, str]:
    out_root = Path(output_dir)
    clips_dir = out_root / "clips"
    out_root.mkdir(parents=True, exist_ok=True)

    all_clips = extract_segment_clips(video_path, segments, str(clips_dir / "all"), "all")
    all_path = concat_clips(all_clips, str(out_root / "highlight_all.mp4")) if all_clips else ""

    short_candidates = ranked_segments[: int(top_k_segments)]
    short_clips = extract_segment_clips(video_path, short_candidates, str(clips_dir / "short"), "short")
    short_path = concat_clips(short_clips, str(out_root / "highlight_short.mp4")) if short_clips else ""

    return {
        "highlight_all": all_path,
        "highlight_short": short_path,
    }


def draw_player_bbox(frame, track_id: int, box: Sequence[float], is_target: bool = False) -> None:
    x1, y1, x2, y2 = [int(v) for v in box]
    color = (0, 255, 0) if is_target else (255, 255, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"ID {track_id}" + (" TARGET" if is_target else "")
    cv2.putText(
        frame,
        label,
        (x1, max(18, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
        cv2.LINE_AA,
    )


def _frame_track_map(tracklets: Dict[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
    fmap: Dict[int, List[Dict[str, Any]]] = {}
    for chunk in tracklets.get("chunks", []):
        for row in chunk.get("records", []):
            frame_idx = int(row.get("frame_idx", -1))
            if frame_idx < 0:
                continue
            fmap[frame_idx] = row.get("tracks", [])
    return fmap


def export_annotated_video(
    video_path: str,
    tracklets: Dict[str, Any],
    target_ids: Sequence[int],
    output_path: str,
) -> str:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Could not determine video resolution")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    target_set = set(int(t) for t in target_ids)
    frame_tracks = _frame_track_map(tracklets)
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        for trk in frame_tracks.get(frame_idx, []):
            tid = int(trk.get("track_id", -1))
            box = trk.get("bbox")
            if tid < 0 or not isinstance(box, list) or len(box) != 4:
                continue
            draw_player_bbox(frame, tid, box, is_target=tid in target_set)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    return str(out_path)
