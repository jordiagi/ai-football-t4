import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class DetectConfig:
    detect_stride: int = 6
    chunk_seconds: int = 180
    person_class_id: int = 0
    yolo_model: str = "yolov8n.pt"


def _clamp_chunk_seconds(chunk_seconds: int) -> int:
    """Constrain chunking to 2-5 minutes to keep memory and checkpoints bounded."""
    return max(120, min(300, int(chunk_seconds)))


def _load_tracker() -> Any:
    """Load ByteTrack from boxmot, compatible with both old and new API."""
    # boxmot v16+: ByteTrack at new path with kwargs constructor
    try:
        from boxmot.trackers.bytetrack.bytetrack import ByteTrack
        return ByteTrack(det_thresh=0.25, max_age=30, min_hits=1, iou_threshold=0.8)
    except ImportError:
        pass

    # boxmot <v16 fallback: BYTETracker with Args object
    try:
        from boxmot.trackers.bytetrack.byte_tracker import BYTETracker  # type: ignore

        class Args:
            track_thresh = 0.25
            track_buffer = 30
            match_thresh = 0.8
            mot20 = False

        return BYTETracker(Args())
    except ImportError as exc:
        raise RuntimeError(
            "boxmot ByteTrack is not available. Install `boxmot`."
        ) from exc


def _save_chunk_checkpoint(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)


def _load_chunk_checkpoint(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _yolo_person_detections(model: YOLO, frame: np.ndarray, person_class_id: int) -> np.ndarray:
    """Return detections in [x1,y1,x2,y2,conf,cls] for person class only.
    6-column format required by boxmot v16+ trackers."""
    results = model.predict(frame, verbose=False)
    if not results:
        return np.empty((0, 6), dtype=np.float32)

    boxes = results[0].boxes
    if boxes is None or boxes.xyxy is None:
        return np.empty((0, 6), dtype=np.float32)

    cls = boxes.cls.detach().cpu().numpy() if boxes.cls is not None else np.array([])
    xyxy = boxes.xyxy.detach().cpu().numpy() if boxes.xyxy is not None else np.empty((0, 4))
    conf = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.zeros((xyxy.shape[0],))

    keep = cls.astype(int) == person_class_id
    if keep.size == 0 or not keep.any():
        return np.empty((0, 6), dtype=np.float32)

    dets = np.concatenate([xyxy[keep], conf[keep, None], cls[keep, None]], axis=1)
    return dets.astype(np.float32)


def _tracker_update(tracker: Any, dets: np.ndarray, frame: np.ndarray) -> List[Dict[str, Any]]:
    """Convert ByteTrack output into normalized tracked detections.

    boxmot v16+ update(dets, img) returns ndarray [x1,y1,x2,y2,id,conf,cls,det_ind].
    Older API returned objects with .tlbr / .track_id attributes.
    """
    tracks_out: List[Dict[str, Any]] = []

    if dets.shape[0] == 0:
        return tracks_out

    track_result = None
    for call in (
        lambda: tracker.update(dets, frame),
        lambda: tracker.update(dets, frame.shape[:2], frame.shape[:2]),
    ):
        try:
            track_result = call()
            break
        except TypeError:
            continue

    if track_result is None or (isinstance(track_result, np.ndarray) and track_result.size == 0):
        return tracks_out

    if isinstance(track_result, np.ndarray):
        # v16+: [x1, y1, x2, y2, track_id, conf, cls, det_ind]
        for row in track_result:
            if len(row) < 5:
                continue
            tracks_out.append({
                "track_id": int(row[4]),
                "bbox": [float(row[0]), float(row[1]), float(row[2]), float(row[3])],
                "score": float(row[5]) if len(row) > 5 else 1.0,
            })
        return tracks_out

    # Legacy object-based API
    for t in track_result:
        tlbr = getattr(t, "tlbr", None)
        track_id = getattr(t, "track_id", None)
        score = getattr(t, "score", 1.0)
        if tlbr is None or track_id is None:
            continue
        tracks_out.append({
            "track_id": int(track_id),
            "bbox": [float(tlbr[0]), float(tlbr[1]), float(tlbr[2]), float(tlbr[3])],
            "score": float(score),
        })

    return tracks_out


def detect_tracklets(
    video_path: str,
    output_dir: str,
    config: Optional[Dict[str, Any]] = None,
    resume: bool = False,
) -> Dict[str, Any]:
    """
    Run YOLOv8 person detection every detect_stride frames and ByteTrack association.
    Processing is chunked (2-5 minutes) with per-chunk checkpoints.
    """
    config = config or {}
    cfg = DetectConfig(
        detect_stride=int(config.get("detect_stride", 6)),
        chunk_seconds=_clamp_chunk_seconds(int(config.get("chunk_seconds", 180))),
        person_class_id=int(config.get("person_class_id", 0)),
        yolo_model=str(config.get("yolo_model", "yolov8n.pt")),
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        raise RuntimeError("Video frame count unavailable or zero.")

    chunk_frames = int(cfg.chunk_seconds * fps)
    total_chunks = int(math.ceil(total_frames / chunk_frames))

    model = YOLO(cfg.yolo_model)
    tracker = _load_tracker()

    out_root = Path(output_dir)
    ckpt_dir = out_root / "checkpoints"
    out_root.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    chunks: List[Dict[str, Any]] = []

    for chunk_idx in range(total_chunks):
        start_frame = chunk_idx * chunk_frames
        end_frame = min((chunk_idx + 1) * chunk_frames, total_frames)
        ckpt_path = ckpt_dir / f"chunk_{chunk_idx:04d}.json"

        if resume:
            checkpoint = _load_chunk_checkpoint(ckpt_path)
            if checkpoint is not None:
                chunks.append(checkpoint)
                continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_idx = start_frame
        records: List[Dict[str, Any]] = []

        while frame_idx < end_frame:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % cfg.detect_stride == 0:
                dets = _yolo_person_detections(model, frame, cfg.person_class_id)
                tracks = _tracker_update(tracker, dets, frame)
                records.append(
                    {
                        "frame_idx": int(frame_idx),
                        "time_sec": float(frame_idx / fps),
                        "tracks": tracks,
                    }
                )

            frame_idx += 1

        chunk_payload = {
            "chunk_index": chunk_idx,
            "start_frame": int(start_frame),
            "end_frame": int(end_frame),
            "records": records,
        }
        _save_chunk_checkpoint(ckpt_path, chunk_payload)
        chunks.append(chunk_payload)

    cap.release()

    result = {
        "video_path": video_path,
        "fps": fps,
        "total_frames": total_frames,
        "detect_stride": cfg.detect_stride,
        "chunk_seconds": cfg.chunk_seconds,
        "chunks": chunks,
    }

    merged_path = Path(output_dir) / "detections.json"
    with merged_path.open("w", encoding="utf-8") as f:
        json.dump(result, f)

    return result
