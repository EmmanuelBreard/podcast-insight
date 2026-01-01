"""Insight extraction using Claude API."""

from dataclasses import dataclass

import anthropic
from rich.console import Console

console = Console()


@dataclass
class PodcastInsights:
    """Extracted insights from a podcast episode."""

    title: str
    summary: str
    key_topics: list[str]
    main_insights: list[str]
    notable_quotes: list[str]
    action_items: list[str]


INSIGHT_PROMPT = """You are an expert at analyzing podcast transcripts and extracting valuable insights.

Analyze the following podcast transcript and extract:
1. A concise title for this episode (if not obvious, create a descriptive one)
2. An executive summary (2-3 sentences capturing the essence)
3. Key topics discussed (3-7 bullet points)
4. Main insights and takeaways (5-10 numbered points with brief context)
5. Notable quotes (2-5 memorable or impactful quotes from the episode)
6. Action items (practical takeaways the listener can apply, if applicable)

Format your response as follows (use these exact headers):

## Title
[Episode title]

## Summary
[2-3 sentence summary]

## Key Topics
- [Topic 1]
- [Topic 2]
...

## Main Insights
1. [Insight with context]
2. [Insight with context]
...

## Notable Quotes
> "[Quote 1]"
> "[Quote 2]"
...

## Action Items
- [Action 1]
- [Action 2]
...

Here is the transcript:

{transcript}"""


class AnalysisError(Exception):
    """Raised when insight analysis fails."""

    pass


def analyze_transcript(
    transcript: str,
    api_key: str,
    model: str = "claude-sonnet-4-20250514",
) -> PodcastInsights:
    """
    Analyze a podcast transcript and extract key insights using Claude.

    Args:
        transcript: The full transcript text
        api_key: Anthropic API key
        model: Claude model to use

    Returns:
        PodcastInsights object with extracted information

    Raises:
        AnalysisError: If analysis fails
    """
    console.print("[blue]Analyzing transcript with Claude...[/blue]")

    try:
        client = anthropic.Anthropic(api_key=api_key)

        message = client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": INSIGHT_PROMPT.format(transcript=transcript),
                }
            ],
        )

        response_text = message.content[0].text
        insights = _parse_insights(response_text)

        console.print("[green]Analysis complete![/green]")
        return insights

    except anthropic.APIError as e:
        raise AnalysisError(f"Claude API error: {e}")
    except Exception as e:
        raise AnalysisError(f"Analysis failed: {e}")


def _parse_insights(response: str) -> PodcastInsights:
    """Parse the structured response from Claude into a PodcastInsights object."""
    sections = {
        "title": "",
        "summary": "",
        "key_topics": [],
        "main_insights": [],
        "notable_quotes": [],
        "action_items": [],
    }

    current_section = None
    lines = response.strip().split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Detect section headers
        lower_line = line.lower()
        if "## title" in lower_line:
            current_section = "title"
        elif "## summary" in lower_line:
            current_section = "summary"
        elif "## key topics" in lower_line:
            current_section = "key_topics"
        elif "## main insights" in lower_line:
            current_section = "main_insights"
        elif "## notable quotes" in lower_line:
            current_section = "notable_quotes"
        elif "## action items" in lower_line:
            current_section = "action_items"
        elif current_section:
            # Parse content based on section type
            if current_section == "title":
                if not line.startswith("#"):
                    sections["title"] = line
            elif current_section == "summary":
                if not line.startswith("#"):
                    sections["summary"] += line + " "
            elif current_section in ["key_topics", "action_items"]:
                if line.startswith("- ") or line.startswith("* "):
                    sections[current_section].append(line[2:])
            elif current_section == "main_insights":
                # Handle numbered items
                if line and (line[0].isdigit() or line.startswith("- ")):
                    # Strip number and period/dash
                    content = line.lstrip("0123456789.-) ").strip()
                    if content:
                        sections[current_section].append(content)
            elif current_section == "notable_quotes":
                if line.startswith(">"):
                    quote = line[1:].strip().strip('"')
                    sections[current_section].append(quote)

    return PodcastInsights(
        title=sections["title"].strip(),
        summary=sections["summary"].strip(),
        key_topics=sections["key_topics"],
        main_insights=sections["main_insights"],
        notable_quotes=sections["notable_quotes"],
        action_items=sections["action_items"],
    )


def format_insights_markdown(insights: PodcastInsights) -> str:
    """Format insights as markdown."""
    lines = [
        f"# {insights.title}",
        "",
        "## Summary",
        insights.summary,
        "",
        "## Key Topics",
    ]

    for topic in insights.key_topics:
        lines.append(f"- {topic}")

    lines.extend(["", "## Main Insights"])
    for i, insight in enumerate(insights.main_insights, 1):
        lines.append(f"{i}. {insight}")

    lines.extend(["", "## Notable Quotes"])
    for quote in insights.notable_quotes:
        lines.append(f'> "{quote}"')

    if insights.action_items:
        lines.extend(["", "## Action Items"])
        for item in insights.action_items:
            lines.append(f"- {item}")

    return "\n".join(lines)


def format_insights_text(insights: PodcastInsights) -> str:
    """Format insights as plain text."""
    lines = [
        f"{'=' * 60}",
        f"  {insights.title}",
        f"{'=' * 60}",
        "",
        "SUMMARY",
        "-" * 40,
        insights.summary,
        "",
        "KEY TOPICS",
        "-" * 40,
    ]

    for topic in insights.key_topics:
        lines.append(f"  * {topic}")

    lines.extend(["", "MAIN INSIGHTS", "-" * 40])
    for i, insight in enumerate(insights.main_insights, 1):
        lines.append(f"  {i}. {insight}")

    lines.extend(["", "NOTABLE QUOTES", "-" * 40])
    for quote in insights.notable_quotes:
        lines.append(f'  "{quote}"')

    if insights.action_items:
        lines.extend(["", "ACTION ITEMS", "-" * 40])
        for item in insights.action_items:
            lines.append(f"  [ ] {item}")

    return "\n".join(lines)
