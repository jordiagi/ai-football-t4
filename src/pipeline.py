import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from detect import detect_tracklets
from download import download_video
from export import export_annotated_video, export_highlights
from identify import IdentifyConfig, identify_target
from intensity import compute_audio_scores, compute_motion_scores, rank_segments, write_ranked_segments
from segment import frames_to_segments, write_segments_json


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, payload: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f)


def _flatten_records(detections: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for chunk in detections.get("chunks", []):
        rows.extend(chunk.get("records", []))
    rows.sort(key=lambda r: int(r.get("frame_idx", -1)))
    return rows


def _area(box: Sequence[float]) -> float:
    x1, y1, x2, y2 = [float(v) for v in box]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _collect_target_presence(
    detections: Dict[str, Any],
    target_ids: Sequence[int],
) -> Dict[str, Any]:
    target_set = set(int(t) for t in target_ids)
    presence_frames: List[int] = []
    target_positions: Dict[int, List[float]] = {}

    for row in _flatten_records(detections):
        frame_idx = int(row.get("frame_idx", -1))
        if frame_idx < 0:
            continue
        matches = [t for t in row.get("tracks", []) if int(t.get("track_id", -1)) in target_set]
        if not matches:
            continue

        presence_frames.append(frame_idx)
        best = max(matches, key=lambda t: _area(t.get("bbox", [0, 0, 0, 0])))
        target_positions[frame_idx] = [float(v) for v in best["bbox"]]

    return {
        "target_ids": sorted(target_set),
        "presence_frames": sorted(set(presence_frames)),
        "target_positions": target_positions,
    }


def _resolve_ref_images(target_cfg: Dict[str, Any], cli_ref_images: Optional[List[str]]) -> List[str]:
    if cli_ref_images:
        return [str(p) for p in cli_ref_images]

    refs = target_cfg.get("ref_current_game_list")
    if isinstance(refs, list) and refs:
        return [str(p) for p in refs]

    single = target_cfg.get("ref_current_game")
    if single:
        return [str(single)]

    return []


def run_pipeline(
    config_path: str = "config.json",
    resume: bool = False,
    url: Optional[str] = None,
    ref_images: Optional[List[str]] = None,
    output_dir_override: Optional[str] = None,
    video_path_override: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = _load_json(config_path) if Path(config_path).exists() else {}

    processing = cfg.get("processing", {})
    clips_cfg = cfg.get("clips", {})
    short_cfg = cfg.get("short_reel", {})
    target_cfg = cfg.get("target", {})
    identify_cfg = cfg.get("identify", {})

    output_dir = Path(output_dir_override or cfg.get("output_dir", "output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    if url:
        video_path = download_video(url=url, output_dir=str(output_dir / "downloads"))
    else:
        video_path = str(video_path_override or cfg.get("video_path", ""))
        if not video_path:
            raise ValueError("Missing video input. Provide --url or --video-path (or config video_path).")

    refs = _resolve_ref_images(target_cfg=target_cfg, cli_ref_images=ref_images)
    if not refs:
        raise ValueError("Missing reference images. Provide --ref-images or config target.ref_current_game.")

    detections_path = output_dir / "detections.json"
    if resume and detections_path.exists():
        detections = _load_json(str(detections_path))
    else:
        detections = detect_tracklets(
            video_path=video_path,
            output_dir=str(output_dir),
            config={
                "detect_stride": int(processing.get("detect_stride", 6)),
                "chunk_seconds": int(processing.get("chunk_seconds", 180)),
                "person_class_id": 0,
                "yolo_model": str(processing.get("yolo_model", "yolov8n.pt")),
            },
            resume=resume,
        )

    target_ids = identify_target(
        tracklets_json_path=str(detections_path),
        ref_images=refs,
        video_path=video_path,
        config=IdentifyConfig(
            similarity_threshold=float(identify_cfg.get("similarity_threshold", 0.55)),
            min_frames=int(identify_cfg.get("min_frames", 5)),
            top_n=int(identify_cfg.get("top_n", 1)),
        ),
    )

    match_result = _collect_target_presence(detections=detections, target_ids=target_ids)
    _save_json(str(output_dir / "target_match.json"), match_result)

    fps = float(detections["fps"])
    total_frames = int(detections["total_frames"])

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
    annotated_video_path = export_annotated_video(
        video_path=video_path,
        tracklets=detections,
        target_ids=target_ids,
        output_path=str(output_dir / "annotated_full.mp4"),
    )

    summary = {
        "video_path": video_path,
        "reference_images": refs,
        "target_ids": target_ids,
        "detections": str(detections_path),
        "target_match": str(output_dir / "target_match.json"),
        "segments": str(output_dir / "segments.json"),
        "ranked_segments": str(output_dir / "ranked_segments.json"),
        "highlight_all": export_paths.get("highlight_all", ""),
        "highlight_short": export_paths.get("highlight_short", ""),
        "annotated_video": annotated_video_path,
    }
    _save_json(str(output_dir / "summary.json"), summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Soccer player identification and highlights pipeline")
    parser.add_argument("--config", default="config.json", help="Path to config.json (optional)")
    parser.add_argument("--resume", action="store_true", help="Resume from saved checkpoints/artifacts")
    parser.add_argument("--url", help="YouTube URL to download and process")
    parser.add_argument("--video-path", help="Local video path (used when --url is not provided)")
    parser.add_argument("--ref-images", nargs="+", help="One or more reference image paths")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    args = parser.parse_args()

    summary = run_pipeline(
        config_path=args.config,
        resume=args.resume,
        url=args.url,
        ref_images=args.ref_images,
        output_dir_override=args.output_dir,
        video_path_override=args.video_path,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
