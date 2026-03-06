import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from detect import detect_tracklets
from embed import compute_embedding, load_target_embeddings
from export import export_highlights
from intensity import compute_audio_scores, compute_motion_scores, rank_segments, write_ranked_segments
from match import PrecisionMatchConfig, PrecisionTargetMatcher
from segment import frames_to_segments, write_segments_json


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, payload: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f)


def _crop(frame: np.ndarray, box: List[float]) -> Optional[np.ndarray]:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def _flatten_records(detections: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for chunk in detections.get("chunks", []):
        rows.extend(chunk.get("records", []))
    rows.sort(key=lambda r: r["frame_idx"])
    return rows


def _match_target(
    video_path: str,
    detection_rows: List[Dict[str, Any]],
    target_embeddings,
    min_target_conf: float,
    reid_refresh_seconds: float,
    max_lost_seconds: float,
    fps: float,
) -> Dict[str, Any]:
    refresh_frames = max(1, int(reid_refresh_seconds * fps))
    max_missing_steps = max(1, int(max_lost_seconds * fps / max(1, refresh_frames)))

    matcher = PrecisionTargetMatcher(
        target_embeddings=target_embeddings,
        config=PrecisionMatchConfig(
            similarity_threshold=float(min_target_conf),
            stable_steps=3,
            confusable_margin=0.05,
            max_missing_steps=max_missing_steps,
        ),
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    presence_frames: List[int] = []
    target_positions: Dict[int, List[float]] = {}
    decisions: List[Dict[str, Any]] = []

    for row in detection_rows:
        frame_idx = int(row["frame_idx"])
        if frame_idx % refresh_frames != 0:
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue

        track_embeddings: Dict[int, np.ndarray] = {}
        track_boxes: Dict[int, List[float]] = {}

        for trk in row.get("tracks", []):
            tid = int(trk["track_id"])
            box = trk["bbox"]
            crop = _crop(frame, box)
            if crop is None or crop.size == 0:
                continue

            try:
                emb = compute_embedding(crop)
            except Exception:
                continue

            track_embeddings[tid] = emb
            track_boxes[tid] = box

        target_tid = matcher.update(track_embeddings)
        decisions.append({"frame_idx": frame_idx, "target_track_id": target_tid})

        if target_tid is not None and target_tid in track_boxes:
            presence_frames.append(frame_idx)
            target_positions[frame_idx] = track_boxes[target_tid]

    cap.release()

    return {
        "presence_frames": presence_frames,
        "target_positions": target_positions,
        "decisions": decisions,
    }


def run_pipeline(config_path: str, resume: bool) -> Dict[str, Any]:
    cfg = _load_json(config_path)

    video_path = cfg["video_path"]
    processing = cfg.get("processing", {})
    clips_cfg = cfg.get("clips", {})
    short_cfg = cfg.get("short_reel", {})
    target_cfg = cfg.get("target", {})

    output_dir = Path(str(cfg.get("output_dir", "output")))
    output_dir.mkdir(parents=True, exist_ok=True)

    detections_path = output_dir / "detections.json"
    if resume and detections_path.exists():
        detections = _load_json(str(detections_path))
    else:
        detections = detect_tracklets(
            video_path=video_path,
            output_dir=str(output_dir),
            config={
                "detect_stride": processing.get("detect_stride", 6),
                "chunk_seconds": processing.get("chunk_seconds", 180),
                "person_class_id": 0,
                "yolo_model": processing.get("yolo_model", "yolov8n.pt"),
            },
            resume=resume,
        )

    fps = float(detections["fps"])
    total_frames = int(detections["total_frames"])
    records = _flatten_records(detections)

    target_embeddings = load_target_embeddings(
        ref_current_game=target_cfg.get("ref_current_game"),
        ref_gallery=target_cfg.get("ref_gallery"),
        ref_current_game_list=target_cfg.get("ref_current_game_list"),
    )

    match_result = _match_target(
        video_path=video_path,
        detection_rows=records,
        target_embeddings=target_embeddings,
        min_target_conf=float(processing.get("min_target_conf", 0.75)),
        reid_refresh_seconds=float(processing.get("reid_refresh_seconds", 2.0)),
        max_lost_seconds=float(processing.get("max_lost_seconds", 5.0)),
        fps=fps,
    )

    _save_json(str(output_dir / "target_match.json"), match_result)

    segments = frames_to_segments(
        presence_frames=match_result["presence_frames"],
        fps=fps,
        pre_seconds=float(clips_cfg.get("pre_seconds", 3)),
        post_seconds=float(clips_cfg.get("post_seconds", 3)),
        min_clip_seconds=float(clips_cfg.get("min_clip_seconds", 3)),
        merge_gap_seconds=float(clips_cfg.get("merge_gap_seconds", 1.0)),
        total_frames=total_frames,
    )

    write_segments_json(
        path=str(output_dir / "segments.json"),
        segments=segments,
        extra_meta={"video_path": video_path, "fps": fps, "total_frames": total_frames},
    )

    motion_scores = compute_motion_scores(segments, match_result["target_positions"], fps)
    audio_scores = compute_audio_scores(video_path, segments)
    ranked = rank_segments(
        segments=segments,
        motion_scores=motion_scores,
        audio_scores=audio_scores,
        mode=str(short_cfg.get("mode", "motion+audio")),
    )

    write_ranked_segments(str(output_dir / "ranked_segments.json"), ranked)

    export_paths = export_highlights(
        video_path=video_path,
        segments=segments,
        ranked_segments=ranked,
        top_k_segments=int(short_cfg.get("top_k_segments", 12)),
        output_dir=str(output_dir),
    )

    summary = {
        "detections": str(detections_path),
        "segments": str(output_dir / "segments.json"),
        "ranked_segments": str(output_dir / "ranked_segments.json"),
        "target_match": str(output_dir / "target_match.json"),
        "highlight_all": export_paths.get("highlight_all", ""),
        "highlight_short": export_paths.get("highlight_short", ""),
    }
    _save_json(str(output_dir / "summary.json"), summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-player soccer video analysis pipeline")
    parser.add_argument("--config", required=True, help="Path to config.json")
    parser.add_argument("--resume", action="store_true", help="Resume from saved checkpoints/artifacts")
    args = parser.parse_args()

    summary = run_pipeline(config_path=args.config, resume=args.resume)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
