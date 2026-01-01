"""Audio downloader for podcast episodes."""

import subprocess
import re
from pathlib import Path

from rich.console import Console

console = Console()


class DownloadError(Exception):
    """Raised when audio download fails."""

    pass


def validate_spotify_url(url: str) -> bool:
    """Validate that the URL is a Spotify episode URL."""
    pattern = r"https?://open\.spotify\.com/episode/[a-zA-Z0-9]+"
    return bool(re.match(pattern, url))


def download_audio(url: str, output_dir: Path) -> Path:
    """
    Download audio from a Spotify podcast episode URL.

    Args:
        url: Spotify episode URL
        output_dir: Directory to save the downloaded audio

    Returns:
        Path to the downloaded audio file

    Raises:
        DownloadError: If download fails
    """
    if not validate_spotify_url(url):
        raise DownloadError(
            f"Invalid Spotify episode URL: {url}\n"
            "Expected format: https://open.spotify.com/episode/<id>"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[blue]Downloading audio from Spotify...[/blue]")

    try:
        result = subprocess.run(
            [
                "spotdl",
                url,
                "--output",
                str(output_dir),
                "--format",
                "mp3",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        # Find the downloaded file
        mp3_files = list(output_dir.glob("*.mp3"))
        if not mp3_files:
            raise DownloadError(
                f"Download appeared to succeed but no MP3 file found in {output_dir}"
            )

        # Return the most recently modified file
        downloaded_file = max(mp3_files, key=lambda p: p.stat().st_mtime)
        console.print(f"[green]Downloaded:[/green] {downloaded_file.name}")
        return downloaded_file

    except subprocess.CalledProcessError as e:
        raise DownloadError(
            f"Failed to download audio: {e.stderr or e.stdout or str(e)}"
        )
    except FileNotFoundError:
        raise DownloadError(
            "spotdl is not installed. Install it with: pip install spotdl"
        )


def get_episode_metadata(url: str) -> dict:
    """
    Get metadata for a Spotify episode.

    Args:
        url: Spotify episode URL

    Returns:
        Dictionary with episode metadata (title, show, etc.)
    """
    try:
        result = subprocess.run(
            ["spotdl", "meta", url],
            capture_output=True,
            text=True,
            check=True,
        )
        # Parse the metadata output (basic parsing)
        return {"raw_metadata": result.stdout}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {}
