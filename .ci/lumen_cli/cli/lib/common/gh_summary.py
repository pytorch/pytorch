from __future__ import annotations
import os
from pathlib import Path
from typing import Iterable, Mapping
import logging
logger = logging.getLogger(__name__)

def _summary_path() -> Path | None:
    """Return the Path to the GitHub step summary file, or None if not set."""
    p = os.environ.get("GITHUB_STEP_SUMMARY")
    return Path(p) if p else None

def write_gh_step_summary(md: str, *, append: bool = True) -> bool:
    """
    Write Markdown content to the GitHub Step Summary file.

    Returns:
        True if written successfully (in GitHub Actions environment),
        False if skipped (e.g., running locally where the variable is not set).
    """
    sp = _summary_path()
    if not sp:
        # When running locally, just log to console instead of failing.
        logger.info("[gh-summary] GITHUB_STEP_SUMMARY not set, skipping write.")
        return False
    sp.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with sp.open(mode, encoding="utf-8") as f:
        f.write(md.rstrip() + "\n")
    return True

def md_heading(text: str, level: int = 2) -> str:
    """Generate a Markdown heading string with the given level (1-6)."""
    return f"{'#' * max(1, min(level, 6))} {text}\n"

def md_kv_table(rows: Iterable[Mapping[str, str | int | float]]) -> str:
    """
    Render a list of dictionaries as a Markdown table.
    The first row (header) is derived from the union of all keys.
    """
    rows = list(rows)
    if not rows:
        return "_(no data)_\n"
    # Collect all columns across all rows
    cols = list({k for r in rows for k in r.keys()})
    header = "| " + " | ".join(cols) + " |\n"
    sep = "|" + "|".join([" --- " for _ in cols]) + "|\n"
    lines = []
    for r in rows:
        line = "| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n"
        lines.append(line)
    return header + sep + "".join(lines) + "\n"

def md_details(summary: str, content: str) -> str:
    """Generate a collapsible <details> block with a summary and inner content."""
    return f"<details>\n<summary>{summary}</summary>\n\n{content}\n\n</details>\n"
