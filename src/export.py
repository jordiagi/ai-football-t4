import subprocess
from pathlib import Path
from typing import Dict, List


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
