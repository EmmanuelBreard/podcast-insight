#!/usr/bin/env python3
"""MCP server for Podcast Insight - extract insights from podcasts and videos."""

import asyncio
import json
import os
import re
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

import anthropic

# Load environment variables
load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable required")

server = Server("podcast-insight")


def validate_url(url: str) -> bool:
    """Validate YouTube URL."""
    patterns = [
        r"https?://(www\.)?youtube\.com/watch\?v=",
        r"https?://youtu\.be/",
        r"https?://(www\.)?youtube\.com/live/",
    ]
    return any(re.match(p, url) for p in patterns)


async def get_youtube_transcript(url: str, output_dir: Path) -> str:
    """Extract transcript from YouTube using yt-dlp."""
    output_template = str(output_dir / "%(title)s")

    result = await asyncio.create_subprocess_exec(
        "yt-dlp",
        "--write-auto-sub",
        "--write-sub",
        "--sub-lang", "en",
        "--sub-format", "vtt",
        "--skip-download",
        "-o", output_template,
        url,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await result.communicate()

    vtt_files = list(output_dir.glob("*.vtt"))
    if not vtt_files:
        raise Exception("No transcript available for this video")

    return parse_vtt_to_text(vtt_files[0])


def parse_vtt_to_text(vtt_path: Path) -> str:
    """Parse VTT subtitle file to plain text."""
    with open(vtt_path, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.split("\n")
    text_lines = []
    seen = set()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("WEBVTT"):
            continue
        if line.startswith("Kind:") or line.startswith("Language:"):
            continue
        if "-->" in line:
            continue
        if line.startswith("NOTE"):
            continue
        line = re.sub(r"<[^>]+>", "", line).strip()
        if line and line not in seen:
            seen.add(line)
            text_lines.append(line)

    return " ".join(text_lines)


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
   - A process diagram (ONLY if the topic describes a sequential process with 3+ steps OR a decision flow with branching paths). Use ASCII art with arrows and boxes. Keep it simple and readable.

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

    try:
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(response_text[start:end])
    except json.JSONDecodeError:
        pass

    return {
        "title": "Podcast Episode",
        "summary": response_text[:500],
        "topics": []
    }


def format_insights_as_markdown(insights: dict) -> str:
    """Format insights as readable markdown."""
    md = f"# {insights['title']}\n\n"
    md += f"## Summary\n\n{insights['summary']}\n\n"

    if insights.get("topics"):
        for topic in insights["topics"]:
            md += f"## {topic['name']}\n\n"
            for point in topic.get("key_points", []):
                md += f"- {point}\n"
            if topic.get("quote"):
                md += f"\n> \"{topic['quote']}\"\n"
            if topic.get("diagram"):
                md += f"\n```\n{topic['diagram']}\n```\n"
            md += "\n"

    return md


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="analyze_podcast",
            description="Extract key insights from a YouTube video or podcast. Returns structured notes with title, summary, key topics, and notable quotes.",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "YouTube URL (youtube.com/watch?v=... or youtu.be/...)"
                    }
                },
                "required": ["url"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    if name != "analyze_podcast":
        raise ValueError(f"Unknown tool: {name}")

    url = arguments.get("url", "").strip()
    if not url:
        return [TextContent(type="text", text="Error: URL is required")]

    if not validate_url(url):
        return [TextContent(type="text", text="Error: Invalid URL. Please use a YouTube link (youtube.com or youtu.be)")]

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            transcript = await get_youtube_transcript(url, Path(temp_dir))

            if not transcript.strip():
                return [TextContent(type="text", text="Error: Transcript is empty")]

            insights = analyze_transcript(transcript)
            markdown = format_insights_as_markdown(insights)

            return [TextContent(type="text", text=markdown)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]


def main():
    """Run the MCP server."""
    asyncio.run(stdio_server(server))


if __name__ == "__main__":
    main()
