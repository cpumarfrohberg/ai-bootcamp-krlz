# GitHub data parsing utilities with bank-grade security validation
import re
from datetime import date, datetime
from typing import Any, Dict, List, NamedTuple

import frontmatter
import yaml
from config import FileProcessingConfig, GitHubConfig, PodcastConstants

from fetch_prep_data.reader import RawRepositoryFile

MAX_DEPTH = 10
MAX_FILES = GitHubConfig.MAX_FILES.value


class TranscriptEntry(NamedTuple):
    """Represents a single transcript entry with timestamp and text."""

    start: float
    text: str

    def __str__(self):
        return f"TranscriptEntry(start={self.start}s, text='{self.text[:PodcastConstants.MAX_TEXT_PREVIEW_LENGTH.value]}{'...' if len(self.text) > PodcastConstants.MAX_TEXT_PREVIEW_LENGTH.value else ''}')"

    def __repr__(self):
        return self.__str__()


def format_timestamp(seconds: float) -> str:
    """Convert seconds to H:MM:SS if > 1 hour, else M:SS"""
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, PodcastConstants.SECONDS_PER_HOUR.value)
    minutes, secs = divmod(remainder, PodcastConstants.SECONDS_PER_MINUTE.value)

    if hours > 0:
        return f"{hours}:{minutes:02}:{secs:02}"
    else:
        return f"{minutes}:{secs:02}"


def make_subtitles(transcript: List[TranscriptEntry]) -> str:
    """Convert transcript entries to subtitle format."""
    lines = []

    for entry in transcript:
        ts = format_timestamp(entry.start)
        text = entry.text.replace("\n", " ")
        lines.append(ts + " " + text)

    return "\n".join(lines)


def join_lines(transcript: List[TranscriptEntry]) -> str:
    """Join transcript entries into continuous text."""
    lines = []

    for entry in transcript:
        text = entry.text.replace("\n", " ")
        lines.append(text)

    return " ".join(lines)


def format_chunk(chunk: List[TranscriptEntry]) -> Dict[str, str]:
    """Format a chunk with start/end timestamps and text."""
    time_start = format_timestamp(chunk[0].start)
    time_end = format_timestamp(chunk[-1].start)
    text = join_lines(chunk)

    return {"start": time_start, "end": time_end, "text": text}


def parse_transcript_content(content: str) -> List[TranscriptEntry]:
    """
    Parse transcript content from markdown to extract timestamped entries.

    Args:
        content: Raw markdown content from podcast file

    Returns:
        List of TranscriptEntry objects with timestamps and text
    """
    transcript_entries = []
    timestamp_pattern = r"(\d{1,2}:\d{2}(?::\d{2})?)"  # Matches MM:SS or HH:MM:SS

    lines = content.split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Look for timestamp patterns
        match = re.search(timestamp_pattern, line)
        if match:
            timestamp_str = match.group(1)
            # Convert timestamp to seconds
            parts = timestamp_str.split(":")
            if len(parts) == 2:  # MM:SS
                seconds = int(
                    parts[0]
                ) * PodcastConstants.SECONDS_PER_MINUTE.value + int(parts[1])
            elif len(parts) == 3:  # HH:MM:SS
                seconds = (
                    int(parts[0]) * PodcastConstants.SECONDS_PER_HOUR.value
                    + int(parts[1]) * PodcastConstants.SECONDS_PER_MINUTE.value
                    + int(parts[2])
                )
            else:
                continue

            # Extract text after timestamp
            text = line[match.end() :].strip()
            if text:
                transcript_entries.append(TranscriptEntry(seconds, text))

    return transcript_entries


def parse_data(data_raw: List[RawRepositoryFile]) -> List[Dict[str, Any]]:
    """
    Parse raw GitHub repository files into structured data with security validation.
    Handles both regular markdown files and podcast transcript files.
    Converts datetime and date objects to ISO format strings for JSON compatibility.

    Args:
        data_raw: List of RawRepositoryFile objects from GitHub

    Returns:
        List of parsed document dictionaries with frontmatter metadata and transcript data

    Raises:
        ValueError: If input validation fails
    """
    if not isinstance(data_raw, list):
        raise ValueError("Input must be a list")

    if len(data_raw) > MAX_FILES:
        raise ValueError(f"Too many files: {len(data_raw)} (max: {MAX_FILES})")

    data_parsed = []
    for f in data_raw:
        if len(f.content) > FileProcessingConfig.MAX_CONTENT_SIZE.value:
            print(f"⚠️  Skipping oversized file {f.filename}: {len(f.content)} bytes")
            continue

        try:
            post = frontmatter.loads(f.content)
            data = post.to_dict()
            data["filename"] = f.filename

            # Convert datetime and date objects to ISO format strings for JSON compatibility
            data = _convert_datetime_to_string(data, depth=0)

            # Check if this is a podcast transcript file
            if (
                "_podcast" in f.filename
                or f.filename.startswith("s")
                or f.filename.startswith("_")
            ):
                # Parse transcript content
                transcript_entries = parse_transcript_content(f.content)
                if transcript_entries:
                    data["transcript_entries"] = transcript_entries
                    data["transcript_count"] = len(transcript_entries)
                    data["content_type"] = "podcast_transcript"

                    # Add subtitle format for easy reading
                    data["subtitles"] = make_subtitles(transcript_entries)

                    # Add joined text for search
                    data["content"] = join_lines(transcript_entries)

                    # Only add podcast files to results
                    data_parsed.append(data)
                else:
                    # Skip files without transcript entries silently
                    pass
            else:
                # For non-podcast files, use the content as-is
                if "content" not in data:
                    data["content"] = f.content
                data_parsed.append(data)

        except (yaml.YAMLError, UnicodeDecodeError):
            # Create document with just content and filename (silently skip frontmatter)
            data = {
                "content": f.content,
                "filename": f.filename,
                "title": "",
                "description": "",
            }
            data_parsed.append(data)

    return data_parsed


def _convert_datetime_to_string(data: Dict[str, Any], depth: int = 0) -> Dict[str, Any]:
    """
    Recursively convert datetime and date objects to ISO format strings with depth limit.

    Args:
        data: Dictionary that may contain datetime or date objects
        depth: Current recursion depth (for security)

    Returns:
        Dictionary with datetime and date objects converted to ISO strings

    Raises:
        ValueError: If recursion depth exceeds limit
    """
    if depth > MAX_DEPTH:
        raise ValueError(f"Data structure too deep: {depth} levels (max: {MAX_DEPTH})")

    converted = {}
    for key, value in data.items():
        if isinstance(value, (datetime, date)):
            converted[key] = value.isoformat()
        elif isinstance(value, dict):
            converted[key] = _convert_datetime_to_string(value, depth + 1)
        elif isinstance(value, list):
            converted[key] = [
                item.isoformat() if isinstance(item, (datetime, date)) else item
                for item in value
            ]
        else:
            converted[key] = value
    return converted
