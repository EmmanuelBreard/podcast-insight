"""FastAPI backend for Podcast Insight web app."""

import os
import tempfile
import asyncio
from pathlib import Path
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import httpx
import anthropic
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Podcast Insight")


# Models
class AnalyzeRequest(BaseModel):
    spotify_url: str


class InsightResponse(BaseModel):
    title: str
    summary: str
    key_topics: list[str]
    main_insights: list[str]
    notable_quotes: list[str]
    action_items: list[str]
    transcript_preview: str


class StatusResponse(BaseModel):
    status: str
    message: str
    progress: int


# In-memory job tracking (for MVP - would use Redis in production)
jobs: dict[str, dict] = {}


# Config
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY required")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY required")


def validate_spotify_url(url: str) -> bool:
    """Validate Spotify episode URL."""
    import re
    pattern = r"https?://open\.spotify\.com/episode/[a-zA-Z0-9]+"
    return bool(re.match(pattern, url))


async def download_audio(url: str, output_dir: Path) -> Path:
    """Download audio from Spotify or YouTube using yt-dlp."""
    output_template = str(output_dir / "%(title)s.%(ext)s")

    result = await asyncio.create_subprocess_exec(
        "yt-dlp",
        "-x",  # Extract audio
        "--audio-format", "mp3",
        "--audio-quality", "0",  # Best quality
        "-o", output_template,
        url,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await result.communicate()

    if result.returncode != 0:
        error_msg = stderr.decode() if stderr else "Unknown error"
        raise Exception(f"Failed to download audio: {error_msg}")

    # Find the downloaded file
    audio_files = list(output_dir.glob("*.mp3")) + list(output_dir.glob("*.m4a")) + list(output_dir.glob("*.opus"))
    if not audio_files:
        raise Exception("No audio file found after download")

    return max(audio_files, key=lambda p: p.stat().st_mtime)


async def transcribe_with_openai(audio_path: Path) -> str:
    """Transcribe audio using OpenAI Whisper API."""
    client = OpenAI(api_key=OPENAI_API_KEY)

    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )

    return transcript


def analyze_transcript(transcript: str) -> dict:
    """Extract insights using Claude."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    prompt = """You are an expert at analyzing podcast transcripts and extracting valuable insights.

Analyze the following podcast transcript and extract:
1. A concise title for this episode (if not obvious, create a descriptive one)
2. An executive summary (2-3 sentences capturing the essence)
3. Key topics discussed (3-7 bullet points)
4. Main insights and takeaways (5-10 numbered points with brief context)
5. Notable quotes (2-5 memorable or impactful quotes from the episode)
6. Action items (practical takeaways the listener can apply, if applicable)

Format your response as JSON with these exact keys:
{
    "title": "Episode title",
    "summary": "2-3 sentence summary",
    "key_topics": ["Topic 1", "Topic 2", ...],
    "main_insights": ["Insight 1", "Insight 2", ...],
    "notable_quotes": ["Quote 1", "Quote 2", ...],
    "action_items": ["Action 1", "Action 2", ...]
}

Here is the transcript:

""" + transcript

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    import json
    response_text = message.content[0].text

    # Extract JSON from response
    try:
        # Try to find JSON in the response
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(response_text[start:end])
    except json.JSONDecodeError:
        pass

    # Fallback if JSON parsing fails
    return {
        "title": "Podcast Episode",
        "summary": response_text[:500],
        "key_topics": [],
        "main_insights": [],
        "notable_quotes": [],
        "action_items": []
    }


async def process_podcast(job_id: str, url: str):
    """Process a podcast URL asynchronously."""
    try:
        jobs[job_id]["status"] = "downloading"
        jobs[job_id]["progress"] = 10
        jobs[job_id]["message"] = "Downloading audio from Spotify..."

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            audio_path = await download_audio(url, temp_path)

            jobs[job_id]["status"] = "transcribing"
            jobs[job_id]["progress"] = 40
            jobs[job_id]["message"] = "Transcribing audio..."

            transcript = await transcribe_with_openai(audio_path)

            jobs[job_id]["status"] = "analyzing"
            jobs[job_id]["progress"] = 70
            jobs[job_id]["message"] = "Extracting insights with Claude..."

            insights = analyze_transcript(transcript)

            jobs[job_id]["status"] = "complete"
            jobs[job_id]["progress"] = 100
            jobs[job_id]["message"] = "Done!"
            jobs[job_id]["result"] = {
                **insights,
                "transcript_preview": transcript[:1000] + "..." if len(transcript) > 1000 else transcript
            }

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["message"] = str(e)
        jobs[job_id]["progress"] = 0


@app.post("/api/analyze")
async def analyze_podcast(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    """Start podcast analysis job."""
    if not validate_spotify_url(request.spotify_url):
        raise HTTPException(
            status_code=400,
            detail="Invalid Spotify episode URL. Use format: https://open.spotify.com/episode/..."
        )

    import uuid
    job_id = str(uuid.uuid4())

    jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "message": "Starting...",
        "result": None
    }

    background_tasks.add_task(process_podcast, job_id, request.spotify_url)

    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    """Get job status."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    return {
        "status": job["status"],
        "progress": job["progress"],
        "message": job["message"],
        "result": job.get("result")
    }


@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the frontend."""
    return FileResponse(Path(__file__).parent / "static" / "index.html")


# Mount static files
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=static_path), name="static")
