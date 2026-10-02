"""Render text for GitHub step summaries."""

from __future__ import annotations


def summary_prose(value: str) -> str:
    """Render untrusted text as inert single-line Markdown."""

    text = " ".join(value.split())
    for character in ("\\", "`", "*", "_", "[", "]", "<", ">", "|", "#"):
        text = text.replace(character, f"\\{character}")
    return text
