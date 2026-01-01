"""CLI interface for podcast-insight."""

import json
from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from .analyzer import (
    PodcastInsights,
    analyze_transcript,
    format_insights_markdown,
    format_insights_text,
)
from .config import load_config
from .downloader import DownloadError, download_audio
from .transcriber import (
    TranscriptionError,
    Transcript,
    format_transcript_with_timestamps,
    transcribe_audio,
)

app = typer.Typer(
    name="podcast-insight",
    help="Extract transcripts and key insights from podcast episodes.",
    add_completion=False,
)

console = Console()


class OutputFormat(str, Enum):
    """Output format options."""

    text = "text"
    markdown = "markdown"
    json = "json"


class WhisperModel(str, Enum):
    """Available Whisper model sizes."""

    tiny = "tiny"
    base = "base"
    small = "small"
    medium = "medium"
    large = "large"


@app.command()
def main(
    url: str = typer.Argument(..., help="Spotify episode URL"),
    output_format: OutputFormat = typer.Option(
        OutputFormat.markdown,
        "--output",
        "-o",
        help="Output format",
    ),
    whisper_model: WhisperModel = typer.Option(
        WhisperModel.base,
        "--whisper-model",
        "-m",
        help="Whisper model size (larger = more accurate but slower)",
    ),
    save_transcript: Optional[Path] = typer.Option(
        None,
        "--save-transcript",
        "-t",
        help="Save raw transcript to file",
    ),
    save_output: Optional[Path] = typer.Option(
        None,
        "--save",
        "-s",
        help="Save insights to file",
    ),
    language: Optional[str] = typer.Option(
        None,
        "--language",
        "-l",
        help="Audio language code (e.g., 'en', 'es'). Auto-detected if not specified.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed progress",
    ),
):
    """
    Extract key insights from a podcast episode.

    Downloads the audio from a Spotify episode URL, transcribes it using
    OpenAI Whisper, and extracts key insights using Claude.

    Example:
        podcast-insight "https://open.spotify.com/episode/xxx"
    """
    try:
        # Load configuration
        config = load_config(whisper_model=whisper_model.value)

        console.print(
            Panel.fit(
                "[bold blue]Podcast Insight[/bold blue]\n"
                "Extracting insights from your podcast...",
                border_style="blue",
            )
        )

        # Step 1: Download audio
        console.print("\n[bold]Step 1/3:[/bold] Downloading audio...")
        audio_path = download_audio(url, config.temp_dir)

        # Step 2: Transcribe
        console.print("\n[bold]Step 2/3:[/bold] Transcribing audio...")
        transcript = transcribe_audio(
            audio_path,
            model_name=whisper_model.value,
            language=language,
        )

        # Save transcript if requested
        if save_transcript:
            save_transcript.write_text(transcript.text)
            console.print(f"[dim]Transcript saved to: {save_transcript}[/dim]")

        # Step 3: Analyze
        console.print("\n[bold]Step 3/3:[/bold] Extracting insights...")
        insights = analyze_transcript(
            transcript.text,
            api_key=config.anthropic_api_key,
        )

        # Format and output
        console.print("\n")
        output = _format_output(insights, output_format)

        if save_output:
            save_output.write_text(output)
            console.print(f"[green]Insights saved to: {save_output}[/green]\n")

        console.print(output)

        # Cleanup
        if audio_path.exists():
            audio_path.unlink()
            if verbose:
                console.print(f"[dim]Cleaned up temporary audio file[/dim]")

    except (DownloadError, TranscriptionError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except ValueError as e:
        console.print(f"[red]Configuration error:[/red] {e}")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled by user[/yellow]")
        raise typer.Exit(130)


def _format_output(insights: PodcastInsights, fmt: OutputFormat) -> str:
    """Format insights based on output format."""
    if fmt == OutputFormat.markdown:
        return format_insights_markdown(insights)
    elif fmt == OutputFormat.text:
        return format_insights_text(insights)
    elif fmt == OutputFormat.json:
        return json.dumps(
            {
                "title": insights.title,
                "summary": insights.summary,
                "key_topics": insights.key_topics,
                "main_insights": insights.main_insights,
                "notable_quotes": insights.notable_quotes,
                "action_items": insights.action_items,
            },
            indent=2,
        )
    return format_insights_markdown(insights)


@app.command("transcribe")
def transcribe_only(
    url: str = typer.Argument(..., help="Spotify episode URL"),
    whisper_model: WhisperModel = typer.Option(
        WhisperModel.base,
        "--whisper-model",
        "-m",
        help="Whisper model size",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Save transcript to file",
    ),
    timestamps: bool = typer.Option(
        False,
        "--timestamps",
        help="Include timestamps in output",
    ),
    language: Optional[str] = typer.Option(
        None,
        "--language",
        "-l",
        help="Audio language code",
    ),
):
    """
    Transcribe a podcast episode without insight extraction.

    Useful when you only need the transcript or want to use a different
    analysis tool.
    """
    try:
        config = load_config(whisper_model=whisper_model.value)

        console.print("[bold]Downloading audio...[/bold]")
        audio_path = download_audio(url, config.temp_dir)

        console.print("[bold]Transcribing...[/bold]")
        transcript = transcribe_audio(
            audio_path,
            model_name=whisper_model.value,
            language=language,
        )

        if timestamps:
            text = format_transcript_with_timestamps(transcript)
        else:
            text = transcript.text

        if output:
            output.write_text(text)
            console.print(f"[green]Transcript saved to: {output}[/green]")
        else:
            console.print("\n" + text)

        # Cleanup
        if audio_path.exists():
            audio_path.unlink()

    except (DownloadError, TranscriptionError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command("analyze")
def analyze_file(
    transcript_file: Path = typer.Argument(..., help="Path to transcript file"),
    output_format: OutputFormat = typer.Option(
        OutputFormat.markdown,
        "--output",
        "-o",
        help="Output format",
    ),
    save_output: Optional[Path] = typer.Option(
        None,
        "--save",
        "-s",
        help="Save insights to file",
    ),
):
    """
    Analyze an existing transcript file.

    Useful when you already have a transcript and just want to extract insights.
    """
    try:
        config = load_config()

        if not transcript_file.exists():
            console.print(f"[red]File not found:[/red] {transcript_file}")
            raise typer.Exit(1)

        transcript_text = transcript_file.read_text()
        console.print("[bold]Analyzing transcript...[/bold]")

        insights = analyze_transcript(
            transcript_text,
            api_key=config.anthropic_api_key,
        )

        output = _format_output(insights, output_format)

        if save_output:
            save_output.write_text(output)
            console.print(f"[green]Insights saved to: {save_output}[/green]\n")

        console.print(output)

    except ValueError as e:
        console.print(f"[red]Configuration error:[/red] {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
