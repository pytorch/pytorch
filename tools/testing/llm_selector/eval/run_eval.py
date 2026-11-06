"""Run the selector over eval samples, caching one result per sample.

    python -m tools.testing.llm_selector.eval.run_eval --data DIR --tag sonnet5_medium \\
        --model us.anthropic.claude-sonnet-5 --effort medium --ids tools/testing/llm_selector/eval/ids/eval.txt

Results go to DIR/results/<tag>/<id>.json and are never recomputed; rerun the
same command to fill in samples on which run_eval itself raised (such as a
failed checkout), which get no file. A CLI crash is written as an error result,
as CI scores it, and so is not rerun. Throttling and 5xx errors, which
the eval's own concurrency provokes, are retried; a call that still fails after
ATTEMPTS is written as an error, so it is scored as the failed call it would be
in CI, which calls the model once. A timeout of the call is final, but an API
504 is a 5xx error and is retried. A tag's results must all come from one
--model, --effort and --agentic, and from what the model is sent for the sample
now (see score.prompt_sha), which score.py checks too. Samples that
select_tests answers without the model (see score.unasked) are skipped. With
--agentic, each worker checks the sample's head out into its own worktree
under DIR/worktrees/<tag> and lets the model Read/Grep/Glob it; query_model
limits those tools to the checkout. The prompt holds untrusted PR text, so
--agentic runs only with LLM_SELECTOR_AGENTIC_SANDBOX=1, in a sandbox whose only
secret is the model credential. Ctrl-C stops the workers after their calls in flight, whose
results are dropped, as the claude children die with the run.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import contextlib
import json
import os
import re
import subprocess
import threading
import time
import traceback
from pathlib import Path

from tools.testing.llm_selector.eval import REPO, write_json
from tools.testing.llm_selector.eval.score import prompt_sha, sample_prompt, unasked
from tools.testing.llm_selector.select_tests import (
    query_model,
    QUERY_TIMEOUT,
    test_minutes,
)


TIMED_OUT = f"model call timed out after {QUERY_TIMEOUT}s"
TRANSIENT = re.compile(
    r"Throttl|rate.?limit|overloaded|\b(429|500|502|503|504|529)\b", re.IGNORECASE
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
    sample: dict,
    minutes: dict,
    args: argparse.Namespace,
    wt: str | None,
    stop: threading.Event | None = None,
) -> dict:
    prompt = sample_prompt(sample, minutes)
    cost = 0.0
    for attempt in range(ATTEMPTS):
        t0 = time.time()
        try:
            d = query_model(prompt, args.model, args.effort, checkout=wt)
        except subprocess.TimeoutExpired:
            # Final, like a timeout the CLI reports: scored as a failed call, so
            # as run-all under final and TD's own split under final+tdN.
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
        if not transient or attempt == ATTEMPTS - 1:
            break
        if stop is None:
            time.sleep(30 * (attempt + 1))
        elif stop.wait(30 * (attempt + 1)):
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
        "prompt_sha": prompt_sha(prompt, wt is not None),
        "cost_usd": cost,
        "usage": d.get("usage"),
        "num_turns": d.get("num_turns"),
        "error": None if out is not None else str(d.get("result"))[:2000],
        # The last attempt also failed transiently.
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
    p.add_argument("--agentic", action="store_true", help="give the model Read, Grep and Glob over a checkout of the change, on untrusted PR text; needs LLM_SELECTOR_AGENTIC_SANDBOX=1")  # fmt: skip
    args = p.parse_args()
    if args.tag == "td":
        p.error("the tag td holds the TD replay, which score.py scores as td25")
    if args.agentic and os.environ.get("LLM_SELECTOR_AGENTIC_SANDBOX") != "1":
        p.error("--agentic gives the model Read, Grep and Glob on untrusted PR text; query_model limits them to the checkout, but run it only in a sandbox whose only secret is the model credential, with LLM_SELECTOR_AGENTIC_SANDBOX=1")  # fmt: skip
    # Agentic checkouts are handed to a CLI that runs in a temp dir.
    args.data = args.data.resolve()

    out = args.data / "results" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    minutes = test_minutes(json.loads((args.data / "test-times.json").read_text()))
    want = {"model": args.model, "effort": args.effort, "agentic": args.agentic}
    done = {x.stem: json.loads(x.read_text()) for x in out.glob("*.json")}
    if other := sorted(i for i, r in done.items() if {k: r.get(k) for k in want} != want):  # fmt: skip
        p.error(f"{len(other)} results in {out}, e.g. {other[0]}, were made with other settings than {want}; use another --tag")  # fmt: skip
    sent = {}
    for i in done:
        # score.py drops a sample whose prompt git cannot build in this clone.
        if (x := args.data / "samples" / f"{i}.json").exists():
            with contextlib.suppress(subprocess.CalledProcessError):
                sent[i] = prompt_sha(sample_prompt(json.loads(x.read_text()), minutes), args.agentic)  # fmt: skip
    if stale := sorted(i for i, h in sent.items() if done[i].get("prompt_sha") != h):
        p.error(f"{len(stale)} results in {out}, e.g. {stale[0]}, were made from another prompt, schema, sample or test-times.json; use another --tag")  # fmt: skip

    def is_unasked(sid: str) -> bool:
        x = args.data / "samples" / f"{sid}.json"
        return x.exists() and unasked(json.loads(x.read_text()))

    ids = [i for i in dict.fromkeys(args.ids.read_text().split()) if i not in done and not is_unasked(i)]  # fmt: skip
    print(f"{args.tag}: {len(ids)} without a result", flush=True)
    stop = threading.Event()

    def worker(slot: int) -> None:
        # Each worker owns one worktree slot, so the ids are dealt out round-robin.
        for sid in ids[slot :: args.jobs]:
            if stop.is_set():
                return
            try:
                sample = json.loads((args.data / "samples" / f"{sid}.json").read_text())
                wt = None
                if args.agentic:
                    try:
                        wt = checkout(args.data / "worktrees" / args.tag, slot, sample["sha"])  # fmt: skip
                    except subprocess.CalledProcessError as e:
                        print(f"{sid}: cannot check out {sample['sha']}, skipped: {e.stderr.decode(errors='replace').strip()}", flush=True)  # fmt: skip
                        continue
                r = run_one(sample, minutes, args, wt, stop)
                # A call Ctrl-C killed returns a crash; give the main thread a moment to set stop.
                if stop.is_set() or (r["out"] is None and stop.wait(1)):
                    return
                if r["transient"]:
                    print(f"{sid}: still failing after {ATTEMPTS} attempts, scored as an error:", r["error"][:200], flush=True)  # fmt: skip
                write_json(out / f"{sid}.json", r)
            except subprocess.CalledProcessError as e:
                # As score.py drops it: usually a main_base this clone lacks.
                print(f"{sid}: git cannot build its prompt in this clone, skipped: {e}", flush=True)  # fmt: skip
            except Exception:
                print(
                    f"{sid} failed, left for a rerun:\n{traceback.format_exc()}",
                    flush=True,
                )

    with cf.ThreadPoolExecutor(args.jobs) as ex:
        try:
            list(ex.map(worker, range(args.jobs)))
        except KeyboardInterrupt:
            stop.set()
            print("interrupted: waiting for the calls in flight, whose results are dropped", flush=True)  # fmt: skip
            raise


if __name__ == "__main__":
    main()
