"""Audio transcription using OpenAI Whisper."""

from dataclasses import dataclass
from pathlib import Path

import whisper
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


@dataclass
class TranscriptSegment:
    """A segment of the transcript with timing information."""

    start: float
    end: float
    text: str


@dataclass
class Transcript:
    """Complete transcript with metadata."""

    text: str
    segments: list[TranscriptSegment]
    language: str


class TranscriptionError(Exception):
    """Raised when transcription fails."""

    pass


def transcribe_audio(
    audio_path: Path,
    model_name: str = "base",
    language: str | None = None,
) -> Transcript:
    """
    Transcribe an audio file using OpenAI Whisper.

    Args:
        audio_path: Path to the audio file
        model_name: Whisper model to use (tiny, base, small, medium, large)
        language: Language code (e.g., 'en', 'es'). Auto-detected if None.

    Returns:
        Transcript object with full text and segments

    Raises:
        TranscriptionError: If transcription fails
    """
    if not audio_path.exists():
        raise TranscriptionError(f"Audio file not found: {audio_path}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Load model
        progress.add_task(description=f"Loading Whisper model ({model_name})...", total=None)
        try:
            model = whisper.load_model(model_name)
        except Exception as e:
            raise TranscriptionError(f"Failed to load Whisper model: {e}")

    console.print(f"[blue]Transcribing audio (this may take a while)...[/blue]")

    try:
        options = {}
        if language:
            options["language"] = language

        result = model.transcribe(str(audio_path), **options)

        segments = [
            TranscriptSegment(
                start=seg["start"],
                end=seg["end"],
                text=seg["text"].strip(),
            )
            for seg in result["segments"]
        ]

        transcript = Transcript(
            text=result["text"].strip(),
            segments=segments,
            language=result.get("language", "unknown"),
        )

        console.print(
            f"[green]Transcription complete![/green] "
            f"({len(segments)} segments, language: {transcript.language})"
        )

        return transcript

    except Exception as e:
        raise TranscriptionError(f"Transcription failed: {e}")


def format_transcript_with_timestamps(transcript: Transcript) -> str:
    """Format transcript with timestamps for each segment."""
    lines = []
    for seg in transcript.segments:
        timestamp = f"[{_format_time(seg.start)} -> {_format_time(seg.end)}]"
        lines.append(f"{timestamp} {seg.text}")
    return "\n".join(lines)


def _format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"
