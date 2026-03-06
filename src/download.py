import argparse
from pathlib import Path

from yt_dlp import YoutubeDL


def _build_opts(output_dir: str) -> dict:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outtmpl = str(out_dir / "%(title).200B [%(id)s].%(ext)s")
    return {
        "format": "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4][height<=1080]/best[height<=1080]",
        "outtmpl": outtmpl,
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }


def download_video(url: str, output_dir: str) -> str:
    """
    Download a YouTube video as mp4 (max 1080p).
    Returns local file path and skips download when target file already exists.
    """
    opts = _build_opts(output_dir)

    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
        expected = Path(ydl.prepare_filename(info))
        expected_mp4 = expected.with_suffix(".mp4")
        if expected_mp4.exists():
            return str(expected_mp4)
        ydl.download([url])

    return str(expected_mp4)


def download_youtube(url: str, output_path: str, quality: str = "best", progress_callback=None) -> str:
    """
    Backward-compatible wrapper.
    output_path should be a file path; download output goes to its parent folder.
    """
    _ = quality
    _ = progress_callback
    return download_video(url=url, output_dir=str(Path(output_path).parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download YouTube video to local mp4")
    parser.add_argument("youtube_url", help="YouTube URL to download")
    parser.add_argument("--output-dir", default="data/matches/match_001", help="Output directory")
    args = parser.parse_args()

    final_path = download_video(args.youtube_url, args.output_dir)
    print(f"Saved to: {final_path}")


if __name__ == "__main__":
    main()
