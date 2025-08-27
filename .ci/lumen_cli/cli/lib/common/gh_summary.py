from __future__ import annotations
import os
from pathlib import Path
from typing import Iterable, Mapping, Optional
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, Tuple

logger = logging.getLogger(__name__)


def gh_summary_path() -> Path | None:
    """Return the Path to the GitHub step summary file, or None if not set."""
    p = os.environ.get("GITHUB_STEP_SUMMARY")
    return Path(p) if p else None


def write_gh_step_summary(md: str, *, append_content: bool = True) -> bool:
    """
    Write Markdown content to the GitHub Step Summary file if GITHUB_STEP_SUMMARY is set.
    append_content: default true, if True, append to the end of the file, else overwrite the whole file

    Returns:
        True if written successfully (in GitHub Actions environment),
        False if skipped (e.g., running locally where the variable is not set).
    """
    sp = gh_summary_path()
    if not sp:
        # When running locally, just log to console instead of failing.
        logger.info("[gh-summary] GITHUB_STEP_SUMMARY not set, skipping write.")
        return False
    sp.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append_content else "w"
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
        # Suppose you want to summarize benchmark results
        rows = [
            {"name": "transformer-small", "p50": 12.3, "p90(ms)": 18.4},
            {"name": "transformer-large", "p50": 45.1, "p90(ms)": 60.7},
        ]
        content = []
        content.append(md_heading("Benchmark Results", level=2))
        content.append(md_kv_table(rows))
        content.append(md_details("Raw logs", "```\n[INFO] benchmark log ...\n```"))
        # Join the pieces into one Markdown block
        markdown = '\n'.join(content)
        # Write to GitHub Actions summary (or log locally if not in CI)
        write_gh_step_summary(markdown, append=True)
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


# ---- helper test to generate a summary for list of pytest failures ------#


def summarize_failures_by_test_command(
    xml_and_labels: Iterable[Tuple[str | Path, str]],
    *,
    title: str = "Pytest Failures by Test Command",
    dedupe_within_command: bool = True,
):
    """
    Args:
      xml_and_labels: list of (xml_path, label) pairs.
                      Each XML corresponds to one pytest subprocess (one test command).
    Behavior:
      - Writes a section per test command if it has failures.
      - Each failed test is listed as 'path/to/test.py:test_name'.

    Example:
        xml = [
            ("reports/junit_cmd0.xml", "pytest -v -s tests/unit"),
            ("reports/junit_cmd1.xml", "pytest -v -s tests/integration"),
            ("reports/junit_cmd2.xml", "pytest -v -s tests/entrypoints"),
        ]
        summarize_failures_by_test_command(
            xmls,
            title="Consolidated Pytest Failures",
        )
    """
    write_gh_step_summary(md_heading(title, level=2))

    for xml_path, label in xml_and_labels:
        xmlp = Path(xml_path)
        failed = _parse_failed_simple(xmlp)
        if dedupe_within_command:
            failed = sorted(set(failed))
        if not failed:
            continue  # skip commands with no failures
        write_gh_step_summary(md_heading(f"Test Command: {label}", level=3))
        lines = "\n".join(f"- {item}" for item in failed)
        write_gh_step_summary(lines + "\n")


def _to_simple_name_from_testcase(tc: ET.Element) -> str:
    """
    Convert a <testcase> into 'path/to/test.py:test_name' format.
    Prefer the 'file' attribute if available, else fall back to classname.
    """
    name = tc.attrib.get("name", "")
    file_attr = tc.attrib.get("file")
    if file_attr:
        return f"{file_attr}:{name}"

    classname = tc.attrib.get("classname", "")
    parts = classname.split(".") if classname else []
    if len(parts) >= 1:
        # drop last part if it's a class, treat rest as module path
        mod_parts = parts[:-1] if len(parts) >= 2 else parts
        mod_path = "/".join(mod_parts) + ".py" if mod_parts else "unknown.py"
        return f"{mod_path}:{name}"
    return f"unknown.py:{name or 'unknown_test'}"


def _parse_failed_simple(xml_path: Path) -> list[str]:
    """
    Parse one XML, return failures as ['tests/a_test.py:test_x', ...].
    Only include <failure> and <error>.
    """
    if not xml_path.exists():
        return []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    failed = []
    for tc in root.iter("testcase"):
        if any(x.tag in {"failure", "error"} for x in tc):
            failed.append(_to_simple_name_from_testcase(tc))
    return failed


def summarize_content_from_file(
    output_dir: Path,
    freeze_file: str,
    title: str = "Wheels (pip freeze)",
    code_lang: str = "",  # e.g. "text" or "ini"
) -> bool:
    """
    Read a text file from output_dir/freeze_file and append it to
    the GitHub Step Summary as a Markdown code block.

    Returns True if something was written, False otherwise.
    """

    f = Path(output_dir) / freeze_file
    if not f.exists():
        return False

    content = f.read_text(encoding="utf-8").strip()
    if not content:
        return False
    md = []
    md.append(md_heading(title, 2))
    md.append(f"```{code_lang}".rstrip())
    md.append(content)
    md.append("```")

    return write_gh_step_summary("\n".join(md) + "\n")


def summarize_wheels(
    output_dir: Path,
    title: str = "Wheels",
    max_depth: Optional[int] = None,  # None = unlimited
):
    """
    Walk output_dir up to max_depth and list all *.whl files.
    Grouped as 'package: filename.whl'.

    Args:
        output_dir: base directory to search
        title: section title in GH summary
        max_depth: maximum folder depth relative to output_dir (0 = only top-level)
    """
    if not output_dir.exists():
        return False
    root = Path(output_dir)
    lines = [md_heading(title, 2)]

    for dirpath, _, filenames in os.walk(root):
        depth = Path(dirpath).relative_to(root).parts
        if max_depth is not None and len(depth) > max_depth:
            # skip going deeper
            continue

        for fname in sorted(filenames):
            if not fname.endswith(".whl"):
                continue
            pkg = fname.split("-")[0]
            relpath = str(Path(dirpath) / fname).replace(str(root) + os.sep, "")
            lines.append(f"- {pkg}: {relpath}")

    if len(lines) > 1:
        write_gh_step_summary("\n".join(lines) + "\n")
