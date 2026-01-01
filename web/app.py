"""FastAPI backend for Podcast Insight web app."""

import os
import tempfile
import asyncio
import uuid
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import anthropic
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Podcast Insight")


# Models
class AnalyzeRequest(BaseModel):
    url: str


class StatusResponse(BaseModel):
    status: str
    message: str
    progress: int


# In-memory job tracking (for MVP - would use Redis in production)
jobs: dict[str, dict] = {}


# Config
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Only needed for file uploads

if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY required")


def validate_url(url: str) -> bool:
    """Validate YouTube or other supported URLs."""
    import re
    youtube_patterns = [
        r"https?://(www\.)?youtube\.com/watch\?v=",
        r"https?://youtu\.be/",
        r"https?://(www\.)?youtube\.com/live/",
    ]
    for pattern in youtube_patterns:
        if re.match(pattern, url):
            return True
    return False


async def get_youtube_transcript(url: str, output_dir: Path) -> str:
    """Extract transcript/subtitles from YouTube using yt-dlp."""
    output_template = str(output_dir / "%(title)s")

    # Try to get auto-generated or manual subtitles
    result = await asyncio.create_subprocess_exec(
        "yt-dlp",
        "--write-auto-sub",  # Get auto-generated subtitles
        "--write-sub",  # Also try manual subtitles
        "--sub-lang", "en",  # English
        "--sub-format", "vtt",  # VTT format (easier to parse)
        "--skip-download",  # Don't download video
        "-o", output_template,
        url,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await result.communicate()

    # Find the subtitle file
    vtt_files = list(output_dir.glob("*.vtt"))
    if not vtt_files:
        # Try alternative: get info and check for subtitles
        raise Exception("No transcript available for this video. Try uploading the audio file instead.")

    # Parse VTT file to plain text
    vtt_path = vtt_files[0]
    transcript = parse_vtt_to_text(vtt_path)

    if not transcript.strip():
        raise Exception("Transcript is empty. Try uploading the audio file instead.")

    return transcript


def parse_vtt_to_text(vtt_path: Path) -> str:
    """Parse VTT subtitle file to plain text."""
    with open(vtt_path, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.split("\n")
    text_lines = []
    seen = set()  # Deduplicate repeated lines

    for line in lines:
        line = line.strip()
        # Skip metadata, timestamps, and empty lines
        if not line:
            continue
        if line.startswith("WEBVTT"):
            continue
        if line.startswith("Kind:") or line.startswith("Language:"):
            continue
        if "-->" in line:  # Timestamp line
            continue
        if line.startswith("NOTE"):
            continue
        # Remove HTML tags like <c> </c>
        import re
        line = re.sub(r"<[^>]+>", "", line)
        line = line.strip()

        if line and line not in seen:
            seen.add(line)
            text_lines.append(line)

    return " ".join(text_lines)


async def transcribe_with_openai(audio_path: Path) -> str:
    """Transcribe audio using OpenAI Whisper API (for file uploads only)."""
    from openai import OpenAI

    if not OPENAI_API_KEY:
        raise Exception("OpenAI API key required for file uploads")

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

    prompt = """You are an expert at analyzing podcast transcripts and extracting the essential information.

Analyze the following podcast transcript and extract:
1. A concise title for this episode (if not obvious, create a descriptive one)
2. An executive summary (2-3 sentences capturing the essence)
3. Key topics discussed - for each topic, provide:
   - The topic name
   - The main information and key points about that topic
   - Any notable quotes related to that topic (if relevant)
   - A process diagram (ONLY if the topic describes a sequential process with 3+ steps OR a decision flow with branching paths). Use ASCII art with arrows (→, ↓) and boxes. Keep it simple and readable.

Format your response as JSON with these exact keys:
{
    "title": "Episode title",
    "summary": "2-3 sentence summary",
    "topics": [
        {
            "name": "Topic name",
            "key_points": ["Main point 1", "Main point 2", ...],
            "quote": "Notable quote if relevant, or null",
            "diagram": "ASCII diagram if a process/flow is described, or null"
        }
    ]
}

Here is the transcript:

""" + transcript

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    response_text = message.content[0].text

    # Extract JSON from response
    try:
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
        "topics": []
    }


async def process_youtube_url(job_id: str, url: str):
    """Process a YouTube URL - extract transcript directly, no audio download."""
    try:
        jobs[job_id]["status"] = "fetching"
        jobs[job_id]["progress"] = 20
        jobs[job_id]["message"] = "Fetching transcript from YouTube..."

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            transcript = await get_youtube_transcript(url, temp_path)

            jobs[job_id]["status"] = "analyzing"
            jobs[job_id]["progress"] = 60
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


async def process_uploaded_file(job_id: str, file_path: Path):
    """Process an uploaded audio file - uses Whisper for transcription."""
    try:
        jobs[job_id]["status"] = "transcribing"
        jobs[job_id]["progress"] = 30
        jobs[job_id]["message"] = "Transcribing audio with Whisper..."

        transcript = await transcribe_with_openai(file_path)

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
    finally:
        # Clean up uploaded file
        if file_path.exists():
            file_path.unlink()


@app.post("/api/analyze")
async def analyze_podcast(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    """Start podcast analysis job from YouTube URL."""
    if not validate_url(request.url):
        raise HTTPException(
            status_code=400,
            detail="Invalid URL. Please use a YouTube link (youtube.com or youtu.be)"
        )

    job_id = str(uuid.uuid4())

    jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "message": "Starting...",
        "result": None
    }

    background_tasks.add_task(process_youtube_url, job_id, request.url)

    return {"job_id": job_id}


@app.post("/api/upload")
async def upload_audio(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload and analyze an audio file (uses Whisper for transcription)."""
    if not OPENAI_API_KEY:
        raise HTTPException(
            status_code=400,
            detail="File upload requires OpenAI API key. Use YouTube URL instead."
        )

    # Validate file type
    allowed_types = ["audio/mpeg", "audio/mp3", "audio/wav", "audio/m4a", "audio/x-m4a", "audio/mp4"]
    if file.content_type not in allowed_types and not file.filename.endswith(('.mp3', '.wav', '.m4a')):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload MP3, WAV, or M4A files."
        )

    # Check file size (max 25MB for Whisper API)
    contents = await file.read()
    if len(contents) > 25 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="File too large. Maximum size is 25MB."
        )

    job_id = str(uuid.uuid4())

    # Save file temporarily
    temp_dir = Path(tempfile.gettempdir()) / "podcast_insight"
    temp_dir.mkdir(exist_ok=True)
    file_path = temp_dir / f"{job_id}_{file.filename}"

    with open(file_path, "wb") as f:
        f.write(contents)

    jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "message": "Processing upload...",
        "result": None
    }

    background_tasks.add_task(process_uploaded_file, job_id, file_path)

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
