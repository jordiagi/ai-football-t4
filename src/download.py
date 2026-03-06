import argparse
from pathlib import Path
from typing import Callable, Optional


def download_youtube(
    url: str,
    output_path: str,
    quality: str = "best",
    progress_callback: Optional[Callable[[str], None]] = None,
) -> str:
    from yt_dlp import YoutubeDL

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    last_percent = {"value": -1}

    def _emit(message: str) -> None:
        print(message)
        if progress_callback is not None:
            progress_callback(message)

    def _progress_hook(d: dict) -> None:
        status = d.get("status")
        if status == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            downloaded = d.get("downloaded_bytes") or 0
            if total and total > 0:
                pct = int((downloaded / total) * 100)
                if pct != last_percent["value"]:
                    last_percent["value"] = pct
                    _emit(f"Downloading... {pct}%")
        elif status == "finished":
            _emit("Download complete. Processing file...")

    ydl_opts = {
        "format": f"bv*[ext=mp4]+ba[ext=m4a]/{quality}[ext=mp4]/{quality}",
        "outtmpl": str(output),
        "merge_output_format": "mp4",
        "progress_hooks": [_progress_hook],
        "quiet": True,
        "noprogress": True,
    }

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return str(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download YouTube video to local mp4")
    parser.add_argument("youtube_url", help="YouTube URL to download")
    parser.add_argument("--output", default="data/matches/match_001/video.mp4", help="Output video file path")
    args = parser.parse_args()

    final_path = download_youtube(args.youtube_url, args.output)
    print(f"Saved to: {final_path}")


if __name__ == "__main__":
    main()
