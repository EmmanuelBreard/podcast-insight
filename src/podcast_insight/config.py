"""Configuration management for podcast-insight."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class Config:
    """Application configuration."""

    anthropic_api_key: str
    whisper_model: str = "base"
    output_dir: Path = Path("./output")
    temp_dir: Path = Path("/tmp/podcast_insight")

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.temp_dir = Path(self.temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)


def load_config(
    whisper_model: str = "base",
    output_dir: str | None = None,
) -> Config:
    """Load configuration from environment variables."""
    load_dotenv()

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable is required. "
            "Set it in your .env file or export it in your shell."
        )

    return Config(
        anthropic_api_key=api_key,
        whisper_model=whisper_model,
        output_dir=Path(output_dir) if output_dir else Path("./output"),
    )
