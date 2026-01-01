# Podcast Insight

Extract key insights from podcast episodes using AI. Paste a Spotify link, get structured notes.

**[Try the web app →](https://podcast-insight.up.railway.app)** *(update with your actual URL)*

## What it does

1. Downloads audio from a Spotify podcast episode
2. Transcribes it using OpenAI Whisper
3. Extracts insights using Claude:
   - Executive summary
   - Key topics
   - Main takeaways
   - Notable quotes
   - Action items

## Web App

The simplest way to use Podcast Insight. Just paste a Spotify episode URL and get insights.

### Self-hosting

```bash
cd web
cp .env.example .env
# Add your ANTHROPIC_API_KEY and OPENAI_API_KEY

pip install -r requirements.txt
uvicorn app:app --reload
```

Open http://localhost:8000

### Deploy to Railway

1. Fork this repo
2. Create a new project on [Railway](https://railway.app)
3. Add environment variables: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`
4. Deploy

## CLI Tool

For local use with more control.

### Prerequisites

- Python 3.10+
- FFmpeg

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

### Install

```bash
pip install -e .
cp .env.example .env
# Add your ANTHROPIC_API_KEY
```

### Usage

```bash
podcast-insight "https://open.spotify.com/episode/..."
```

Options:

| Option | Description |
|--------|-------------|
| `--output`, `-o` | Format: `text`, `markdown`, `json` |
| `--whisper-model`, `-m` | Model: `tiny`, `base`, `small`, `medium`, `large` |
| `--save`, `-s` | Save insights to file |
| `--save-transcript`, `-t` | Save transcript to file |

### Examples

```bash
# Save as markdown
podcast-insight "https://open.spotify.com/episode/..." --save insights.md

# Use larger model for accuracy
podcast-insight "https://open.spotify.com/episode/..." -m medium

# Transcribe only
podcast-insight transcribe "https://open.spotify.com/episode/..."

# Analyze existing transcript
podcast-insight analyze transcript.txt
```

## Output Example

```markdown
# The Future of AI in Healthcare

## Summary
This episode explores how AI is transforming healthcare delivery, from diagnostic
imaging to drug discovery.

## Key Topics
- AI-powered diagnostic tools
- Drug discovery acceleration
- Privacy concerns in health AI

## Main Insights
1. AI can detect certain cancers earlier than human radiologists
2. Drug discovery timelines could drop from 10 years to 2-3 years
3. Data privacy remains the biggest barrier to AI adoption

## Notable Quotes
> "We're not replacing doctors, we're giving them superpowers"

## Action Items
- Research FDA-approved AI diagnostic tools
- Review hospital data sharing policies
```

## How it works

**Web app:** Uses OpenAI Whisper API for transcription (fast, ~$0.006/min).

**CLI:** Uses local Whisper models (free, slower, requires more RAM).

Both use Claude for insight extraction.

## License

MIT
