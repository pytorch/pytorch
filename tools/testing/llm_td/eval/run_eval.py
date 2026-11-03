"""Run the selector over eval samples, caching one result per sample.

    python -m tools.testing.llm_td.eval.run_eval --data DIR --tag sonnet5_medium \\
        --model us.anthropic.claude-sonnet-5 --effort medium --ids tools/testing/llm_td/eval/ids/eval.txt

Results go to DIR/results/<tag>/<id>.json and are never recomputed; rerun the
same command to fill in samples whose call failed transiently. With
--agentic, each worker checks the sample's head out into its own worktree
under DIR/worktrees and lets the model Read/Grep/Glob it.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import subprocess
import time
import traceback
from pathlib import Path

from tools.testing.llm_td.eval import REPO, write_json
from tools.testing.llm_td.select_tests import (
    build_prompt,
    query_model,
    QUERY_TIMEOUT,
    test_minutes,
)


TIMED_OUT = f"model call timed out after {QUERY_TIMEOUT}s"
TRANSIENT = re.compile(
    r"Throttl|rate limit|overloaded|\b(500|502|503|504|529)\b|timed out", re.IGNORECASE
)
ATTEMPTS = 4


def checkout(root: Path, slot: int, sha: str) -> str:
    wt = root / f"slot{slot}"
    if not (wt / ".git").exists():
        subprocess.run(
            ["git", "worktree", "add", "--detach", "-f", str(wt), sha],
            cwd=REPO,
            check=True,
            capture_output=True,
        )
    else:
        subprocess.run(
            ["git", "checkout", "-q", "--detach", "-f", sha],
            cwd=wt,
            check=True,
            capture_output=True,
        )
    return str(wt)


def run_one(
    sample: dict, minutes: dict, args: argparse.Namespace, wt: str | None
) -> dict:
    prompt = build_prompt(
        sample["title"],
        sample["body"] or "",
        sample["diffstat"],
        sample["diff"],
        sample["candidates"],
        minutes,
    )
    cost = 0.0
    for attempt in range(ATTEMPTS):
        if attempt:
            time.sleep(30 * attempt)
        t0 = time.time()
        try:
            d = query_model(prompt, args.model, args.effort, checkout=wt)
        except subprocess.TimeoutExpired:
            # Final rather than retried, as in CI: scored as a failed call, so as
            # run-all under final and TD's own split under final+tdN.
            d, out, transient = {"is_error": True, "result": TIMED_OUT}, None, False
            break
        cost += d.get("total_cost_usd") or 0.0
        # As in select_tests.select: an error result's output is discarded.
        out = None if d.get("is_error") else d.get("structured_output")
        # Only the CLI's own error fields: a model's prose can mention "500". A
        # result with no "type" is query_model's dump of output that was not the
        # CLI's JSON, i.e. a crash, which is final whatever its trace says.
        err = f"{d.get('subtype', '')} {d.get('result', '') if d.get('is_error') else ''}"  # fmt: skip
        transient = out is None and "type" in d and bool(TRANSIENT.search(err))
        if not transient:
            break
    return {
        "id": sample["id"],
        "model": args.model,
        "effort": args.effort,
        "agentic": wt is not None,
        "attempts": attempt + 1,
        # The last call's latency; cost_usd sums every attempt's.
        "wall_s": round(time.time() - t0, 1),
        "prompt_chars": len(prompt),
        "cost_usd": cost,
        "usage": d.get("usage"),
        "num_turns": d.get("num_turns"),
        "error": None if out is not None else str(d.get("result"))[:2000],
        "transient": transient,
        "out": out,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])  # fmt: skip
    p.add_argument("--ids", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--agentic", action="store_true")
    args = p.parse_args()

    out = args.data / "results" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    minutes = test_minutes(json.loads((args.data / "test-times.json").read_text()))
    ids = [i for i in args.ids.read_text().split() if not (out / f"{i}.json").exists()]
    print(f"{args.tag}: {len(ids)} to run", flush=True)

    def worker(slot: int) -> None:
        # Each worker owns one worktree slot, so the ids are dealt out round-robin.
        for sid in ids[slot :: args.jobs]:
            try:
                sample = json.loads((args.data / "samples" / f"{sid}.json").read_text())
                wt = (
                    checkout(args.data / "worktrees", slot, sample["sha"])
                    if args.agentic
                    else None
                )
                r = run_one(sample, minutes, args, wt)
            except Exception:
                print(
                    f"{sid} failed, left for a rerun:\n{traceback.format_exc()}",
                    flush=True,
                )
                continue
            if r.pop("transient"):
                print(
                    "transient failure, left for a rerun:",
                    sid,
                    r["error"][:200],
                    flush=True,
                )
                continue
            write_json(out / f"{sid}.json", r)

    with cf.ThreadPoolExecutor(args.jobs) as ex:
        list(ex.map(worker, range(args.jobs)))


if __name__ == "__main__":
    main()
