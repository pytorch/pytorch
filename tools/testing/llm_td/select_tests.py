"""Ask an LLM which test files a change can break.

    python -m tools.testing.llm_td.select_tests --base origin/main --output llm_td_selection.json

The prompt holds the PR title and description, the diff, and every test file
CI can run annotated with its CI cost. Nothing from the change is executed:
in CI this script runs from the default branch and only reads the PR through git. The model answers through `claude -p`
with a JSON schema; on Bedrock set CLAUDE_CODE_USE_BEDROCK=1 and AWS
credentials. Any failure yields a run_all selection, so CI never skips tests
because the selector broke.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from tools.stats.import_test_stats import get_test_times
from tools.testing.discover_tests import TESTS
from tools.testing.llm_td.selection import needs_run_all, Selection, SWEEPS


sys.path.remove(str(REPO_ROOT))

DEFAULT_MODEL = (
    "us.anthropic.claude-opus-5-5"
    if os.environ.get("CLAUDE_CODE_USE_BEDROCK", "").lower() in ("1", "true", "yes")
    else "claude-opus-5-5"
)
SYSTEM_PROMPT = (Path(__file__).parent / "prompt.md").read_text()
DIFF_BUDGET = 120_000
BODY_BUDGET = 4_000
# Well under the CI job's 15 minute timeout, so a hung call still yields run_all.
QUERY_TIMEOUT = 600
# Room for the note that replaces a truncated file's tail.
TRUNCATION_MARKER = 40
# The PR text is wrapped in these tags and the prompt marks it as untrusted, so
# the PR must not be able to close them.
CLOSING_TAG_RE = re.compile(
    r"</(pr_title|pr_description|changed_files|diff)\s*>", re.IGNORECASE
)

SCHEMA = {
    "type": "object",
    "properties": {
        "run_all": {
            "type": "boolean",
            "description": "true only if the change is so broad that no subset is safe",
        },
        "reason": {"type": "string", "description": "one or two sentences"},
        "sweeps": {
            "type": "array",
            "items": {"type": "string", "enum": list(SWEEPS)},
            "description": "wrapper configs that must run in full",
        },
        "dirs": {
            "type": "array",
            "items": {"type": "string"},
            "description": "test directory prefixes to run in full, e.g. dynamo or inductor",
        },
        "tests": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "file": {"type": "string"},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                },
                "required": ["file", "confidence"],
            },
        },
    },
    "required": ["run_all", "reason", "sweeps", "dirs", "tests"],
}


def truncate_diff(diff: str, budget: int = DIFF_BUDGET) -> str:
    """Share the budget across files so that one huge generated file cannot crowd out the rest.

    Files under the fair share are kept whole and the rest of the budget is
    split evenly among the larger ones. A truncated file always keeps its
    `diff --git` line, so with hundreds of files the budget is soft."""
    if len(diff) <= budget:
        return diff
    parts = diff.split("\ndiff --git ")
    parts = [parts[0]] + ["diff --git " + p for p in parts[1:]]
    remaining = budget - (len(parts) - 1)
    cap = 0
    for i, size in enumerate(sorted(len(p) for p in parts)):
        cap = remaining // (len(parts) - i)
        if size > cap:
            break
        remaining -= size
    out = []
    for p in parts:
        if len(p) <= cap:
            out.append(p)
            continue
        keep = max(cap - TRUNCATION_MARKER, len(p.split("\n", 1)[0]))
        note = f"... [{len(p) - keep} chars truncated]"
        out.append(f"{p[:keep]}\n{note}")
    return "\n".join(out)


def test_minutes(test_times: dict) -> dict[str, float]:
    """Machine-minutes of one full CI pass of each test file, summed over every job and config."""
    minutes: dict[str, float] = {}
    for configs in test_times.values():
        for files in configs.values():
            for f, seconds in files.items():
                minutes[f] = minutes.get(f, 0.0) + seconds / 60
    return minutes


def build_prompt(
    title: str,
    body: str,
    diffstat: str,
    diff: str,
    candidates: list[str],
    minutes: dict[str, float],
) -> str:
    body = body.strip()
    if len(body) > BODY_BUDGET:
        body = body[:BODY_BUDGET] + "\n...[truncated]"
    title, body, diffstat, diff = (
        CLOSING_TAG_RE.sub(r"<\\/\1>", t) for t in (title, body, diffstat, diff)
    )
    return "\n".join(
        [
            f"# PR title\n<pr_title>\n{title}\n</pr_title>",
            f"# PR description\n<pr_description>\n{body or '(none)'}\n</pr_description>",
            f"# Changed files\n<changed_files>\n{diffstat}\n</changed_files>",
            f"# Diff\n<diff>\n{truncate_diff(diff)}\n</diff>",
            f"# Candidate test files ({len(candidates)})",
            "\n".join(f"{c} [{minutes.get(c, 0):.0f}m]" for c in candidates),
            "\nSelect the test files to run. Use the exact names from the candidate list.",
        ]
    )


def query_model(
    prompt: str,
    model: str,
    effort: str,
    checkout: str | None = None,
    timeout: int = QUERY_TIMEOUT,
) -> dict:
    """Run `claude -p` and return its JSON result.

    With `checkout`, the model may Read/Grep/Glob that tree; otherwise it has no tools."""
    tools = "Read,Grep,Glob" if checkout else ""
    cmd = [
        "claude", "-p", "--bare", "--no-session-persistence",
        "--model", model,
        "--effort", effort,
        "--system-prompt", SYSTEM_PROMPT,
        "--tools", tools,
        "--output-format", "json",
        "--json-schema", json.dumps(SCHEMA),
    ]  # fmt: skip
    if checkout:
        cmd += ["--allowed-tools", tools, "--add-dir", checkout]
        prompt = f"A checkout of the repository at this change is at {checkout}; use Read/Grep/Glob to find what exercises the changed code. Keep exploration focused.\n\n{prompt}"
    # An empty working directory keeps the CLI from picking up any project settings.
    with tempfile.TemporaryDirectory() as tmp:
        p = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            cwd=checkout or tmp,
            timeout=timeout,
        )
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError:
        return {"is_error": True, "result": (p.stdout + p.stderr)[-2000:]}


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True)


def pr_text(args: argparse.Namespace) -> tuple[str, str]:
    if args.title is not None:
        return args.title, args.body or ""
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if event_path and os.path.exists(event_path):
        with open(event_path) as f:
            pr = json.load(f).get("pull_request") or {}
        if pr:
            return pr.get("title") or "", pr.get("body") or ""
    return git("log", "-1", "--format=%s", args.head).strip(), git(
        "log", "-1", "--format=%b", args.head
    )


def select(args: argparse.Namespace, record: dict) -> Selection | None:
    """Make the selection, filling `record` with diagnostics; None after --print-prompt."""
    merge_base = git("merge-base", args.base, args.head).strip()
    record["merge_base"] = merge_base
    record["head"] = git("rev-parse", args.head).strip()
    # Without renames a moved file is listed at both paths, and its new path is
    # an added file below.
    changed = git(
        "diff", "--name-only", "--no-renames", merge_base, args.head
    ).splitlines()
    if needs_run_all(changed) and not args.print_prompt:
        return Selection(run_all=True, reason="change touches test infrastructure")
    test_times = get_test_times()
    if not test_times and not args.print_prompt:
        return Selection(run_all=True, reason="no test times")
    diffstat = git("diff", "--stat=200", "--no-renames", merge_base, args.head)
    diff = git("diff", merge_base, args.head)
    # TESTS comes from the checkout this script runs from, which in CI is the
    # trusted default branch rather than the PR; add the test files the PR adds.
    added = git(
        "diff", "--name-only", "--no-renames", "--diff-filter=A", merge_base, args.head
    ).splitlines()
    candidates = sorted(
        set(TESTS)
        | {
            f.removeprefix("test/").removesuffix(".py")
            for f in added
            if re.match(r"test/([\w/]+/)?test_\w+\.py$", f)
        }
    )
    title, body = pr_text(args)
    prompt = build_prompt(
        title, body, diffstat, diff, candidates, test_minutes(test_times)
    )
    if args.print_prompt:
        print(prompt)
        return None

    result = query_model(prompt, args.model, args.effort)
    out = None if result.get("is_error") else result.get("structured_output")
    record["cost_usd"] = result.get("total_cost_usd")
    record["model_output"] = out
    if out is None:
        record["error"] = str(result.get("result"))[:2000]
    return Selection.from_model_output(out, set(candidates), changed)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--base", default="origin/main", help="diff against merge-base(base, head)"
    )
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--title")
    parser.add_argument("--body")
    parser.add_argument(
        "--model", default=os.environ.get("LLM_TD_MODEL", DEFAULT_MODEL)
    )
    parser.add_argument("--effort", default=os.environ.get("LLM_TD_EFFORT", "low"))
    parser.add_argument(
        "--output", type=Path, default=REPO_ROOT / "llm_td_selection.json"
    )
    parser.add_argument(
        "--enforce",
        action="store_true",
        help="tell run_test.py to skip uncovered files",
    )
    parser.add_argument("--print-prompt", action="store_true")
    args = parser.parse_args(argv)

    record: dict = {
        "model": args.model,
        "effort": args.effort,
        "merge_base": None,
        "head": None,
        "cost_usd": None,
        "error": None,
        "model_output": None,
    }
    try:
        selection = select(args, record)
    except Exception as e:
        if args.print_prompt:
            raise
        record["error"] = repr(e)[:2000]
        selection = Selection(run_all=True, reason=f"selector failed: {repr(e)[:500]}")
    if selection is None:
        return
    selection.enforce = args.enforce
    args.output.write_text(json.dumps({**selection.to_json(), **record}, indent=1))
    print(
        f"LLM TD: run_all={selection.run_all} sweeps={sorted(selection.sweeps)} files={len(selection.files)}"
    )
    print(f"LLM TD: {selection.reason}")


if __name__ == "__main__":
    main()
