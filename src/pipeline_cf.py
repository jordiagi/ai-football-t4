import argparse
import copy
import json
import socket
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

from cloudflare_client import CloudflareClient
from pipeline import run_pipeline


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = BASE_DIR / "config.json"
DEFAULT_CF_CONFIG_PATH = BASE_DIR / "cloudflare_config.json"


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)


def _ensure_player_ids(raw: Any) -> List[str]:
    if isinstance(raw, list):
        return [str(v) for v in raw if v]
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(v) for v in parsed if v]
        except json.JSONDecodeError:
            return []
    return []


def _collect_short_clips(output_dir: Path, ranked: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    short_dir = output_dir / "clips" / "short"
    if not short_dir.exists():
        return []

    clip_files = sorted(short_dir.glob("short_*.mp4"))
    selected = ranked[:top_k]
    rows: List[Dict[str, Any]] = []
    for idx, clip_path in enumerate(clip_files):
        if idx >= len(selected):
            break
        seg = selected[idx]
        rows.append(
            {
                "path": clip_path,
                "start_sec": float(seg.get("start_sec", 0.0)),
                "end_sec": float(seg.get("end_sec", 0.0)),
                "intensity_score": float(seg.get("rank_score", 0.0)),
            }
        )
    return rows


def _build_player_config(
    base_config: Dict[str, Any],
    *,
    video_path: str,
    player_id: str,
    ref_photo_path: Path,
    gallery_dir: Optional[Path],
    output_dir: Path,
) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_config)
    cfg["video_path"] = video_path
    cfg["output_dir"] = str(output_dir)

    target = cfg.setdefault("target", {})
    target["name"] = target.get("name") or player_id
    target["ref_current_game"] = str(ref_photo_path)
    # Use gallery dir if it has photos, otherwise fall back to primary photo's parent
    if gallery_dir and any(gallery_dir.iterdir()):
        target["ref_gallery"] = str(gallery_dir)
    else:
        target["ref_gallery"] = str(ref_photo_path.parent)

    return cfg


def _resolve_video_path(video_local_path: str, job_id: str) -> str:
    """If the path looks like a URL, download it first and return the local path."""
    if video_local_path.startswith("http://") or video_local_path.startswith("https://"):
        from download import download_youtube
        dest = BASE_DIR / "output" / "cloudflare_jobs" / job_id / "video.mp4"
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"[yt-dlp] Downloading {video_local_path} → {dest}")
        return download_youtube(video_local_path, str(dest))
    return video_local_path


def process_job(
    *,
    client: CloudflareClient,
    base_config: Dict[str, Any],
    mac_client_id: str,
    job: Dict[str, Any],
) -> None:
    job_id = str(job["id"])
    raw_path = str(job.get("video_local_path") or "")
    if not raw_path:
        raise ValueError(f"Job {job_id} missing video_local_path")
    video_local_path = _resolve_video_path(raw_path, job_id)

    player_ids = _ensure_player_ids(job.get("player_ids"))
    if not player_ids:
        raise ValueError(f"Job {job_id} has no player_ids")

    all_clips_payload: List[Dict[str, Any]] = []

    for player_id in player_ids:
        player_dir = BASE_DIR / "data" / "players" / player_id
        photo_path = client.download_player_photos(player_id, player_dir)

        output_dir = BASE_DIR / "output" / "cloudflare_jobs" / job_id / player_id
        gallery_dir = player_dir / "gallery"
        player_cfg = _build_player_config(
            base_config,
            video_path=video_local_path,
            player_id=player_id,
            ref_photo_path=photo_path,
            gallery_dir=gallery_dir if gallery_dir.exists() else None,
            output_dir=output_dir,
        )

        player_cfg_path = output_dir / "pipeline_config.json"
        _save_json(player_cfg_path, player_cfg)

        run_pipeline(config_path=str(player_cfg_path), resume=False)

        ranked_path = output_dir / "ranked_segments.json"
        ranked_payload = _load_json(ranked_path) if ranked_path.exists() else {"ranked_segments": []}
        ranked_segments = ranked_payload.get("ranked_segments", [])
        top_k = int(player_cfg.get("short_reel", {}).get("top_k_segments", 12))

        player_clips = _collect_short_clips(output_dir, ranked_segments, top_k)

        for clip in player_clips:
            clip_id = str(uuid.uuid4())
            client.upload_highlight(clip_id, Path(clip["path"]))
            all_clips_payload.append(
                {
                    "id": clip_id,
                    "player_id": player_id,
                    "start_sec": float(clip["start_sec"]),
                    "end_sec": float(clip["end_sec"]),
                    "intensity_score": float(clip["intensity_score"]),
                }
            )

    client.complete_job(job_id, all_clips_payload)


def run_daemon(client: CloudflareClient, base_config: Dict[str, Any], mac_client_id: str, poll_interval_seconds: int) -> None:
    hostname = socket.gethostname()

    while True:
        client.register_heartbeat(mac_client_id, hostname)

        job = client.claim_job(mac_client_id)
        if job is not None:
            job_id = str(job.get("id"))
            try:
                process_job(client=client, base_config=base_config, mac_client_id=mac_client_id, job=job)
                print(f"[ok] completed job {job_id}")
            except Exception as exc:
                client.fail_job(job_id, str(exc))
                print(f"[fail] job {job_id}: {exc}")

        time.sleep(max(1, int(poll_interval_seconds)))


def run_single_job(client: CloudflareClient, base_config: Dict[str, Any], mac_client_id: str, job_id: str) -> None:
    client.register_heartbeat(mac_client_id, socket.gethostname())
    claim_payload = client.claim_specific_job(job_id, mac_client_id)
    if isinstance(claim_payload, dict) and claim_payload.get("id"):
        job = claim_payload
    else:
        job = client.get_job(job_id)

    process_job(client=client, base_config=base_config, mac_client_id=mac_client_id, job=job)
    print(f"[ok] completed job {job_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cloudflare-integrated AI football highlights pipeline")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to local config.json")
    parser.add_argument(
        "--cloudflare-config",
        default=str(DEFAULT_CF_CONFIG_PATH),
        help="Path to cloudflare_config.json",
    )
    parser.add_argument("--job", help="Specific job ID to process once")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon and poll continuously")
    args = parser.parse_args()

    base_config = _load_json(Path(args.config))
    cf_config = _load_json(Path(args.cloudflare_config))

    worker_url = str(cf_config["worker_url"])
    mac_client_id = str(cf_config["mac_client_id"])
    poll_interval_seconds = int(cf_config.get("poll_interval_seconds", 30))

    client = CloudflareClient(worker_url=worker_url)

    if args.job:
        run_single_job(client, base_config, mac_client_id, args.job)
        return

    if args.daemon:
        run_daemon(client, base_config, mac_client_id, poll_interval_seconds)
        return

    raise SystemExit("Provide either --job <job_id> or --daemon")


if __name__ == "__main__":
    main()
