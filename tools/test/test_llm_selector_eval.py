from __future__ import annotations

import argparse
import collections
import contextlib
import datetime as dt
import functools
import io
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import textwrap
import threading
import types
import unittest
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from tools.testing.llm_selector import select_tests, selection
from tools.testing.llm_selector.eval import (
    build_dataset as bd,
    clickhouse,
    score,
    td_replay,
)
from tools.testing.llm_selector.eval.build_dataset import strip_trailers
from tools.testing.llm_selector.eval.run_eval import run_one
from tools.testing.llm_selector.eval.score import (
    Choice,
    ci_jobs,
    config_of,
    cost_fraction,
    policy,
    truth_pairs,
)
from tools.testing.llm_selector.eval.td_replay import top25_per_config
from tools.testing.llm_selector.selection import apply_selection, Selection
from tools.testing.target_determination.heuristics.interface import TestPrioritizations
from tools.testing.test_run import TestRun


sys.path.remove(str(REPO_ROOT))

# config -> file -> [cpu, gpu] seconds
SECONDS = {
    "default": {"test_a": [100.0, 0.0], "test_b": [50.0, 50.0], "test_c": [10.0, 0.0], "test_d": [40.0, 0.0]},
    "slow": {"test_a": [0.0, 200.0], "test_c": [0.0, 0.0]},
}  # fmt: skip
SAMPLE = {
    "kind": "pr",
    "sha": "h",
    "title": "t",
    "body": None,
    "diffstat": "",
    "diff": "",
    "candidates": ["test_a", "test_b", "test_c", "test_d", "test_e"],
    "changed_files": ["torch/foo.py"],
    "truth": ["test_a", "test_b", "test_gone"],
    "truth_strong": ["test_a"],
    "truth_jobs": {"test_a": [["pull", "linux / test (slow, 1, 2, linux.2xlarge)"]]},
}


def answer(**kw: object) -> dict:
    """A well-formed model answer selecting nothing, with `kw` replacing its keys."""
    return {"run_all": False, "reason": "", "sweeps": [], "dirs": [], "tests": [], **kw}


NOTHING = {"out": answer()}


def pick(name: str, r: dict | None, td: dict | None, s: dict = SAMPLE, grep: Callable[[str], list[str]] | None = None) -> Choice:  # fmt: skip
    choice = policy(name, s, r, td, SECONDS, grep)
    if choice is None:
        raise AssertionError(f"{name} chose nothing")
    return choice


def write_data(d: str, files: Mapping[str, object]) -> Path:
    """Write `files` under `d` (str values as is, others as JSON), stamping each TD replay with its sample's current inputs and each result with no prompt_sha with its sample's."""
    data = Path(d)
    for name, obj in files.items():
        (data / name).parent.mkdir(parents=True, exist_ok=True)
        (data / name).write_text(obj if isinstance(obj, str) else json.dumps(obj))
    if not (data / "test-times.json").exists():
        return data
    times = (data / "test-times.json").read_bytes()
    for p in (data / "td").glob("*.json"):
        if (sample := data / "samples" / p.name).exists():
            inputs = score.replay_inputs("", times, sample.read_bytes())
            p.write_text(json.dumps({**json.loads(p.read_text()), "inputs": inputs}))
    minutes = select_tests.test_minutes(json.loads(times))
    for p in (data / "results").glob("*/*.json"):
        r = json.loads(p.read_text())
        if "prompt_sha" not in r and (sample := data / "samples" / p.name).exists():
            sent = score.prompt_sha(score.sample_prompt(json.loads(sample.read_text()), minutes))  # fmt: skip
            p.write_text(json.dumps({**r, "prompt_sha": sent}))
    return data


@contextlib.contextmanager
def patch_ci_jobs(jobs: dict[bool, set[tuple[str, str, str]]]) -> Iterator[None]:
    with mock.patch.object(score, "ci_jobs", lambda: jobs):
        score.job_runs_td.cache_clear()
        score.runs_td.cache_clear()
        try:
            yield
        finally:
            score.job_runs_td.cache_clear()
            score.runs_td.cache_clear()


START, END = dt.date(2026, 10, 1), dt.date(2026, 10, 26)
JOB = "pull / test (default, 1, 1)"
TD_JOBS = {
    True: {("pull", "pull", "default")},
    False: {("rocm-rx7900", "pull", "default")},
}


def fail(test: str, t: str, sha: str, br: str = "gh/x", wf: str = "pull", job: str = JOB, tf: str = "test_a", flaky: bool = False, cls: str = "TestA") -> dict:  # fmt: skip
    return {"tf": tf, "classname": cls, "test": test, "t": t, "sha": sha, "br": br, "wf": wf, "job": job, "flaky": flaky}  # fmt: skip


def day(d: int, month: int = 10) -> dt.datetime:
    return dt.datetime(2026, month, d, tzinfo=dt.timezone.utc)


def history(rows: list[dict], sha2pr: dict[str, int], authors: dict[int, str], landed: dict[int, dt.datetime] | None = None, disabled: tuple = ()) -> bd.History:  # fmt: skip
    """`disabled` holds (day, test) or (day, test, class) with class TestA by default."""
    with (
        mock.patch.object(bd, "pull_failures", return_value=rows),
        mock.patch.object(bd, "pr_of_shas", return_value=sha2pr),
        mock.patch.object(bd, "pr_authors", return_value=authors),
        mock.patch.object(bd, "landed_times", return_value=landed or {}),
        mock.patch.object(bd, "query", return_value=[{"d": d, "name": f"{t} (__main__.{c[0] if c else 'TestA'})"} for d, t, *c in disabled]),
    ):  # fmt: skip
        return bd.History.load(START, END)


class NoWaitEvent(threading.Event):
    def wait(self, timeout: float | None = None) -> bool:
        return self.is_set()


def no_waits() -> contextlib.AbstractContextManager:
    """main()'s stop event, whose waits are the retry backoff and the Ctrl-C grace, returns at once."""
    return mock.patch("tools.testing.llm_selector.eval.run_eval.threading", types.SimpleNamespace(Event=NoWaitEvent))  # fmt: skip


class TestRunEval(unittest.TestCase):
    def run_one(self, replies: list[object]) -> tuple[dict, mock.Mock, mock.Mock]:
        sample = {"id": "x", "title": "t", "body": None, "diffstat": "", "diff": "", "candidates": ["test_a"]}  # fmt: skip
        args = argparse.Namespace(model="m", effort="medium")
        with (
            mock.patch("tools.testing.llm_selector.eval.run_eval.query_model", side_effect=replies) as query,
            mock.patch("tools.testing.llm_selector.eval.run_eval.time.sleep") as sleep,
        ):  # fmt: skip
            return run_one(sample, {}, args, None), query, sleep

    def test_retry_then_success(self) -> None:
        out = answer()
        r, _, sleep = self.run_one([
            {"type": "result", "is_error": True, "result": "API Error: 502 Bad Gateway", "total_cost_usd": 0.1},
            {"type": "result", "structured_output": out, "total_cost_usd": 0.2},
        ])  # fmt: skip
        self.assertEqual(
            (r["out"], r["error"], r["transient"], r["attempts"]), (out, None, False, 2)
        )
        self.assertAlmostEqual(r["cost_usd"], 0.3)
        self.assertEqual(sleep.call_count, 1)

    def test_not_transient(self) -> None:
        # Only an error's text is matched, and an error drops the structured output as CI does.
        for reply in (
            {
                "type": "result",
                "subtype": "error_max_structured_output_retries",
                "result": "the 500 tests timed out",
            },
            {
                "type": "result",
                "is_error": True,
                "result": "bad schema",
                "structured_output": answer(),
            },
            # query_model's fallback for output that is not JSON: a crash, whatever its trace says.
            {"is_error": True, "result": "TypeError at cli.js:503:12"},
        ):
            r, query, _ = self.run_one([reply])
            self.assertEqual(
                (r["out"], r["transient"], query.call_count), (None, False, 1)
            )
            self.assertIsNotNone(r["error"])

    def test_timeout_is_final(self) -> None:
        from subprocess import TimeoutExpired

        from tools.testing.llm_selector.eval.run_eval import TIMED_OUT

        r, query, _ = self.run_one([TimeoutExpired("claude", 1)])
        self.assertEqual((r["out"], r["error"], r["transient"], query.call_count), (None, TIMED_OUT, False, 1))  # fmt: skip
        # A timeout the CLI reports is as final as one query_model raises.
        r, query, _ = self.run_one([{"type": "result", "is_error": True, "result": "Request timed out"}])  # fmt: skip
        self.assertEqual((r["transient"], r["attempts"], query.call_count), (False, 1, 1))  # fmt: skip

    def test_transient(self) -> None:
        for text in ("API Error: 429 Too Many Requests", '{"type":"rate_limit_error"}', "ThrottlingException", "503"):  # fmt: skip
            r, query, _ = self.run_one([{"type": "result", "is_error": True, "result": text}, {"type": "result", "structured_output": answer()}])  # fmt: skip
            self.assertEqual((r["error"], r["attempts"]), (None, 2), text)

    def test_gives_up(self) -> None:
        r, query, sleep = self.run_one(
            [{"type": "result", "is_error": True, "result": "529 overloaded"}] * 4
        )
        self.assertEqual((r["out"], r["error"], r["transient"], r["attempts"]), (None, "529 overloaded", True, 4))  # fmt: skip
        self.assertEqual((query.call_count, sleep.call_count), (4, 3))

    def test_stop_during_backoff_ends_the_retries(self) -> None:
        sample = {"id": "x", "title": "t", "body": None, "diffstat": "", "diff": "", "candidates": ["test_a"]}  # fmt: skip
        stop = threading.Event()
        reply = {"type": "result", "is_error": True, "result": "529 overloaded"}
        with (
            mock.patch("tools.testing.llm_selector.eval.run_eval.query_model", return_value=reply) as query,
            mock.patch.object(stop, "wait", return_value=True) as wait,
        ):  # fmt: skip
            r = run_one(sample, {}, argparse.Namespace(model="m", effort="medium"), None, stop)  # fmt: skip
        # Ctrl-C lands during the first backoff: no second call is started.
        self.assertEqual((query.call_count, r["transient"], r["attempts"]), (1, True, 1))  # fmt: skip
        wait.assert_called_once_with(30)

    def test_main_writes_exhausted_and_resolves_data(self) -> None:
        from tools.testing.llm_selector.eval import run_eval

        sample = {"id": "a", "sha": "f" * 40, "title": "t", "body": None, "diffstat": "", "diff": "", "candidates": ["test_a"], "changed_files": ["torch/foo.py"]}  # fmt: skip
        with tempfile.TemporaryDirectory() as d:
            data = write_data(d, {"samples/a.json": sample, "test-times.json": {}})
            (data / "ids.txt").write_text("a")
            argv = ["run_eval", "--data", data.name, "--tag", "t", "--model", "m", "--ids", str(data / "ids.txt"), "--jobs", "1", "--agentic"]  # fmt: skip
            with (
                contextlib.chdir(data.parent),  # --data is relative to this cwd.
                mock.patch.object(sys, "argv", argv),
                mock.patch.dict(os.environ, {"LLM_SELECTOR_AGENTIC_SANDBOX": "1"}),
                mock.patch.object(run_eval, "checkout", return_value="wt") as co,
                mock.patch.object(run_eval, "query_model", return_value={"type": "result", "is_error": True, "result": "529"}) as query,
                no_waits(),
                mock.patch("sys.stdout", io.StringIO()),
            ):  # fmt: skip
                run_eval.main()
            self.assertEqual(co.call_args.args, (data.resolve() / "worktrees/t", 0, "f" * 40))  # fmt: skip
            r = json.loads((data / "results/t/a.json").read_text())
        self.assertEqual(
            (r["error"], r["transient"], r["agentic"]), ("529", True, True)
        )
        prompt = run_eval.sample_prompt(sample, {})
        self.assertEqual(query.call_args, mock.call(prompt, "m", "medium", checkout="wt"))  # fmt: skip
        # The checkout preamble and the tools are sent too.
        self.assertEqual(r["prompt_sha"], run_eval.prompt_sha(prompt, True))
        self.assertNotEqual(r["prompt_sha"], run_eval.prompt_sha(prompt))
        for name in ("SYSTEM_PROMPT", "AGENTIC_TOOLS", "CHECKOUT_PREAMBLE"):
            with mock.patch.object(score, name, "x"):
                self.assertNotEqual(r["prompt_sha"], run_eval.prompt_sha(prompt, True))

    def test_main_survives_a_failed_write(self) -> None:
        from tools.testing.llm_selector.eval import run_eval

        r = {"type": "result", "structured_output": answer()}
        writes = [OSError("full"), None]
        with tempfile.TemporaryDirectory() as d:
            samples = {f"samples/{i}.json": {**SAMPLE, "id": i} for i in "ab"}
            # select_tests runs everything on c without asking the model.
            samples["samples/c.json"] = dict(SAMPLE, id="c", changed_files=[])
            data = write_data(d, samples)
            (data / "test-times.json").write_text("{}")
            (data / "ids.txt").write_text("a b c a b")
            ids = str(data / "ids.txt")
            argv = ["run_eval", "--data", d, "--tag", "t", "--model", "m", "--ids", ids]
            with (
                mock.patch.object(sys, "argv", [*argv, "--jobs", "1"]),
                mock.patch("sys.stdout", io.StringIO()) as out,
                mock.patch.object(run_eval, "query_model", return_value=r) as query,
                mock.patch.object(run_eval, "write_json", side_effect=writes) as write,
            ):
                run_eval.main()
        # Each id is paid once, c never, and a's failed write leaves the slot running b.
        self.assertEqual((query.call_count, write.call_count), (2, 2))
        self.assertEqual(write.call_args.args[0].name, "b.json")
        self.assertIn("a failed, left for a rerun", out.getvalue())
        # c needs no result, so it is not counted as missing one.
        self.assertTrue(out.getvalue().startswith("t: 2 without a result\n"))

    def test_main_stops_on_ctrl_c(self) -> None:
        import signal

        from tools.testing.llm_selector.eval import run_eval

        def killed(*args: object, **kwargs: object) -> dict:
            # Ctrl-C reaches the main thread, and the claude child dies with output that is not JSON.
            os.kill(os.getpid(), signal.SIGINT)
            return {"is_error": True, "result": "killed"}

        with tempfile.TemporaryDirectory() as d:
            data = write_data(d, {f"samples/{i}.json": {**SAMPLE, "id": i} for i in "abc"})  # fmt: skip
            (data / "test-times.json").write_text("{}")
            (data / "ids.txt").write_text("a b c")
            argv = ["run_eval", "--data", d, "--tag", "t", "--model", "m", "--ids", str(data / "ids.txt"), "--jobs", "1"]  # fmt: skip
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch("sys.stdout", io.StringIO()) as out,
                mock.patch.object(run_eval, "query_model", side_effect=killed) as query,
                self.assertRaises(KeyboardInterrupt),
            ):
                old = signal.signal(signal.SIGINT, signal.default_int_handler)
                try:
                    run_eval.main()
                finally:
                    signal.signal(signal.SIGINT, old)
            self.assertEqual(list((data / "results/t").iterdir()), [])
        # The killed call is not saved, and no further id is paid for.
        self.assertEqual(query.call_count, 1)
        self.assertIn("interrupted", out.getvalue())

    def test_main_refuses_agentic_outside_a_sandbox(self) -> None:
        from tools.testing.llm_selector.eval import run_eval

        argv = ["run_eval", "--data", "x", "--tag", "t", "--model", "m", "--ids", "x", "--agentic"]  # fmt: skip
        for env in ({}, {"LLM_SELECTOR_AGENTIC_SANDBOX": "0"}):
            env = {k: v for k, v in os.environ.items() if k != "LLM_SELECTOR_AGENTIC_SANDBOX"} | env  # fmt: skip
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.dict(os.environ, env, clear=True),
                mock.patch("sys.stderr", io.StringIO()) as err,
                mock.patch.object(run_eval, "query_model") as query,
                self.assertRaises(SystemExit),
            ):
                run_eval.main()
            self.assertIn("LLM_SELECTOR_AGENTIC_SANDBOX=1", err.getvalue())
            query.assert_not_called()

    def test_main_refuses_tag_td(self) -> None:
        from tools.testing.llm_selector.eval import run_eval

        argv = ["run_eval", "--data", "x", "--tag", "td", "--model", "m", "--ids", "x"]
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch("sys.stderr", io.StringIO()) as err,
            self.assertRaises(SystemExit),
        ):
            run_eval.main()
        self.assertIn("the tag td holds the TD replay", err.getvalue())

    def test_main_refuses_other_settings(self) -> None:
        from tools.testing.llm_selector.eval import run_eval

        with tempfile.TemporaryDirectory() as d:
            data = Path(d)
            (data / "results/t").mkdir(parents=True)
            (data / "results/t/a.json").write_text(json.dumps({"model": "m", "effort": "high", "agentic": False}))  # fmt: skip
            (data / "test-times.json").write_text("{}")
            (data / "ids.txt").write_text("a b")
            argv = ["run_eval", "--data", d, "--tag", "t", "--model", "m", "--ids", str(data / "ids.txt")]  # fmt: skip
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch("sys.stderr", io.StringIO()) as err,
                self.assertRaises(SystemExit),
            ):
                run_eval.main()
        self.assertIn("1 results in", err.getvalue())

    def test_main_refuses_another_prompt(self) -> None:
        from tools.testing.llm_selector.eval import run_eval

        sample = {"id": "a", "sha": "f" * 40, "title": "t", "body": None, "diffstat": "", "diff": "", "candidates": ["test_a"], "changed_files": ["torch/foo.py"]}  # fmt: skip
        sent = run_eval.prompt_sha(run_eval.sample_prompt(sample, {}))
        result = {"model": "m", "effort": "medium", "agentic": False}
        r = {"type": "result", "is_error": True, "result": "x"}
        for prompt_sha, schema, refused in ((sent, score.SCHEMA, False), (None, score.SCHEMA, True), (sent, {}, True)):  # fmt: skip
            with tempfile.TemporaryDirectory() as d:
                data = write_data(d, {"samples/a.json": sample, "samples/b.json": sample, "test-times.json": {}, "results/t/a.json": {**result, "prompt_sha": prompt_sha}})  # fmt: skip
                (data / "ids.txt").write_text("a b")
                argv = ["run_eval", "--data", d, "--tag", "t", "--model", "m", "--ids", str(data / "ids.txt")]  # fmt: skip
                with (
                    mock.patch.object(sys, "argv", argv),
                    mock.patch.object(score, "SCHEMA", schema),
                    mock.patch("sys.stderr", io.StringIO()) as err,
                    mock.patch("sys.stdout", io.StringIO()),
                    mock.patch.object(run_eval, "query_model", return_value=r) as query,
                    no_waits(),
                ):
                    try:
                        run_eval.main()
                    except SystemExit:
                        self.assertTrue(refused)
                        self.assertIn("another prompt", err.getvalue())
                    else:
                        self.assertFalse(refused)
                # Only the unfinished b is run, and only when a's result is accepted.
                self.assertEqual(query.call_count, 0 if refused else 1)

    def test_main_skips_a_prompt_git_cannot_build(self) -> None:
        from tools.testing.llm_selector.eval import run_eval

        # The diff adds a candidate, which sample_prompt looks up at a main_base this clone lacks.
        sample = {"id": "a", "sha": "f" * 40, "title": "t", "body": None, "diffstat": "", "diff": "diff --git a/test/test_new.py b/test/test_new.py\nnew file mode 100644\n", "candidates": ["test_new"], "changed_files": ["test/test_new.py"], "main_base": "0" * 40}  # fmt: skip
        result = {"model": "m", "effort": "medium", "agentic": False, "prompt_sha": "x"}
        with tempfile.TemporaryDirectory() as d:
            data = write_data(d, {"samples/a.json": sample, "test-times.json": {}, "results/t/a.json": result})  # fmt: skip
            (data / "ids.txt").write_text("a")
            argv = ["run_eval", "--data", d, "--tag", "t", "--model", "m", "--ids", str(data / "ids.txt")]  # fmt: skip
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch("sys.stdout", io.StringIO()) as out,
                mock.patch.object(run_eval, "query_model") as query,
            ):
                run_eval.main()
        self.assertEqual((out.getvalue(), query.call_count), ("t: 0 without a result\n", 0))  # fmt: skip
        # Without a result yet it is skipped with one line, not a traceback.
        with tempfile.TemporaryDirectory() as d:
            data = write_data(d, {"samples/a.json": sample, "test-times.json": {}})
            (data / "ids.txt").write_text("a")
            argv = ["run_eval", "--data", d, "--tag", "t", "--model", "m", "--ids", str(data / "ids.txt")]  # fmt: skip
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch("sys.stdout", io.StringIO()) as out,
                mock.patch.object(run_eval, "query_model") as query,
            ):
                run_eval.main()
        lines = out.getvalue().splitlines()
        self.assertEqual((len(lines), query.call_count), (2, 0))
        self.assertTrue(lines[1].startswith("a: git cannot build its prompt in this clone, skipped"))  # fmt: skip
        # An agentic checkout git cannot make is named as such, with git's own error.
        with tempfile.TemporaryDirectory() as d:
            data = write_data(d, {"samples/a.json": sample, "test-times.json": {}})
            (data / "ids.txt").write_text("a")
            err = b"fatal: invalid reference\n"
            failed = subprocess.CalledProcessError(128, ["git"], stderr=err)
            with (
                mock.patch.object(sys, "argv", ["run_eval", "--data", d, "--tag", "t", "--model", "m", "--ids", str(data / "ids.txt"), "--agentic"]),
                mock.patch.dict(os.environ, {"LLM_SELECTOR_AGENTIC_SANDBOX": "1"}),
                mock.patch("sys.stdout", io.StringIO()) as out,
                mock.patch.object(run_eval, "checkout", side_effect=failed),
                mock.patch.object(run_eval, "query_model") as query,
            ):  # fmt: skip
                run_eval.main()
        self.assertEqual(out.getvalue().splitlines()[1], f"a: cannot check out {'f' * 40}, skipped: fatal: invalid reference")  # fmt: skip
        query.assert_not_called()

    def test_main_tops_up_agentic_results(self) -> None:
        from tools.testing.llm_selector.eval import run_eval

        sample = {"id": "a", "sha": "f" * 40, "title": "t", "body": None, "diffstat": "", "diff": "", "candidates": ["test_a"], "changed_files": ["torch/foo.py"]}  # fmt: skip
        prompt = run_eval.sample_prompt(sample, {})
        result = {"model": "m", "effort": "medium", "agentic": True}
        r = {"type": "result", "is_error": True, "result": "x"}
        # An agentic result is accepted only with the agentic prompt's hash.
        for agentic_sha, refused in ((True, False), (False, True)):
            with tempfile.TemporaryDirectory() as d:
                data = write_data(d, {"samples/a.json": sample, "samples/b.json": sample, "test-times.json": {}, "results/t/a.json": {**result, "prompt_sha": run_eval.prompt_sha(prompt, agentic_sha)}})  # fmt: skip
                (data / "ids.txt").write_text("a b")
                argv = ["run_eval", "--data", d, "--tag", "t", "--model", "m", "--ids", str(data / "ids.txt"), "--agentic"]  # fmt: skip
                with (
                    mock.patch.object(sys, "argv", argv),
                    mock.patch.dict(os.environ, {"LLM_SELECTOR_AGENTIC_SANDBOX": "1"}),
                    mock.patch("sys.stderr", io.StringIO()) as err,
                    mock.patch("sys.stdout", io.StringIO()),
                    mock.patch.object(run_eval, "checkout", return_value="wt"),
                    mock.patch.object(run_eval, "query_model", return_value=r) as query,
                    no_waits(),
                ):
                    try:
                        run_eval.main()
                    except SystemExit:
                        self.assertTrue(refused)
                        self.assertIn("another prompt", err.getvalue())
                    else:
                        self.assertFalse(refused)
                        b = json.loads((data / "results/t/b.json").read_text())
                        self.assertEqual(b["prompt_sha"], run_eval.prompt_sha(prompt, True))  # fmt: skip
                self.assertEqual(query.call_count, 0 if refused else 1)

    def test_checkout(self) -> None:
        import subprocess

        from tools.testing.llm_selector.eval import run_eval

        env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull, GIT_CONFIG_NOSYSTEM="1", GIT_AUTHOR_NAME="a", GIT_AUTHOR_EMAIL="a@a", GIT_COMMITTER_NAME="a", GIT_COMMITTER_EMAIL="a@a")  # fmt: skip
        with (
            tempfile.TemporaryDirectory() as repo,
            tempfile.TemporaryDirectory() as root,
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch.object(run_eval, "REPO", Path(repo)),
        ):

            def git(*args: str, cwd: str = repo) -> str:
                return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()  # fmt: skip

            git("init", "-q")
            shas = []
            for text in ("1", "2"):
                (Path(repo) / "f").write_text(text)
                git("add", "-A")
                git("commit", "-qm", text)
                shas.append(git("rev-parse", "HEAD"))
            wt = run_eval.checkout(Path(root), 3, shas[0])
            self.assertEqual(wt, str(Path(root) / "slot3"))
            self.assertEqual(git("rev-parse", "HEAD", cwd=wt), shas[0])
            # The slot is reused, and a file the model's tools left behind is overwritten.
            (Path(wt) / "f").write_text("dirty")
            self.assertEqual(run_eval.checkout(Path(root), 3, shas[1]), wt)
            self.assertEqual(git("rev-parse", "HEAD", cwd=wt), shas[1])
            self.assertEqual((Path(wt) / "f").read_text(), "2")
            self.assertEqual(len(git("worktree", "list").splitlines()), 2)


class TestEvalHelpers(unittest.TestCase):
    def test_has_head(self) -> None:
        self.assertTrue(score.has_head(select_tests.git("rev-parse", "HEAD").strip()))
        self.assertFalse(score.has_head("0" * 40))

    def test_write_json(self) -> None:
        from tools.testing.llm_selector.eval import write_json

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "r.json"
            write_json(path, {"a": 1})
            self.assertEqual((json.loads(path.read_text()), path.stat().st_mode & 0o777), ({"a": 1}, 0o644))  # fmt: skip
            # A failed write keeps the old file and leaves no temporary one.
            with self.assertRaises(TypeError):
                write_json(path, {"a": object()})
            self.assertEqual((json.loads(path.read_text()), list(Path(d).iterdir())), ({"a": 1}, [path]))  # fmt: skip


class TestScore(unittest.TestCase):
    def setUp(self) -> None:
        self.enterContext(mock.patch.object(score, "has_head", return_value=True))

    def test_config_of(self) -> None:
        self.assertEqual(
            config_of(
                "linux-jammy-py3.10-clang21 / test (crossref, 1, 2, lf.linux.2xlarge)"
            ),
            "crossref",
        )
        self.assertEqual(config_of("win / test (default)"), "default")
        self.assertEqual(config_of("(default)"), "default")
        self.assertEqual(config_of(" weird job "), "weird job")

    def test_truth_pairs(self) -> None:
        with patch_ci_jobs({True: {("pull", "linux", "slow")}, False: set()}):
            self.assertEqual(truth_pairs(SAMPLE, strong=True), [("test_a", "slow")])
            # test_b has no jobs and falls back to the default config; test_gone is not a candidate.
            self.assertEqual(
                truth_pairs(SAMPLE, strong=False),
                [("test_a", "slow"), ("test_b", "default")],
            )
            self.assertEqual(
                truth_pairs({**SAMPLE, "kind": "revert"}, strong=True),
                [("test_a", "slow"), ("test_b", "default")],
            )

    def test_ci_jobs(self) -> None:
        workflows = {
            "pull.yml": """
                name: pull
                on: pull_request
                jobs:
                  build:
                    name: linux-jammy-py3.10-gcc11
                    uses: ./.github/workflows/_linux-build.yml
                    with:
                      test-matrix: |
                        { include: [{ config: "default", shard: 1 }] }
                  test:
                    name: linux-jammy-py3.10-gcc11
                    uses: ./.github/workflows/_linux-test.yml
                    with:
                      test-matrix: ${{ needs.build.outputs.test-matrix }}
                      tests-to-include: ${{ github.event.inputs.tests-to-include || '' }}
                  xpu-build:
                    name: linux-noble-xpu-n-py3.10
                    uses: ./.github/workflows/_linux-build.yml
                    with:
                      build-environment: linux-noble-xpu-n-py3.10
                      test-matrix: |
                        { include: [{ config: "default", shard: 1 }] }
                  xpu-test:
                    name: linux-noble-xpu-n-py3.10
                    uses: ./.github/workflows/_xpu-test.yml
                    with:
                      build-environment: ${{ needs.xpu-build.outputs.build-environment }}
                      test-matrix: ${{ needs.xpu-build.outputs.test-matrix }}
                  onnx-test:
                    name: linux-jammy-py3-clang12-onnx
                    uses: ./.github/workflows/_linux-test.yml
                    with:
                      build-environment: linux-jammy-py3-clang12-onnx
                      test-matrix: |
                        { include: [{ config: "default", shard: 1 }] }
                  dynamo-cpython-test:
                    name: dynamo-cpython-test
                    uses: ./.github/workflows/_linux-test.yml
                    with:
                      test-matrix: |
                        { include: [{ config: "dynamo_cpython", shard: 1 }, { config: "dynamo_cpython_313", shard: 1 }] }
            """,
            "inductor.yml": """
                name: inductor
                on: pull_request
                jobs:
                  unit-test:
                    name: unit-test
                    uses: ./.github/workflows/inductor-unittest.yml
            """,
            "inductor-unittest.yml": """
                name: inductor-unittest
                on:
                  workflow_call:
                jobs:
                  inductor-test:
                    uses: ./.github/workflows/_linux-test.yml
                    with:
                      test-matrix: |
                        { include: [{ config: "inductor_core", shard: 1 }] }
            """,
            "trunk.yml": """
                name: trunk
                on: push
                jobs:
                  dynamo-unittest:
                    name: dynamo-unittest
                    uses: ./.github/workflows/dynamo-unittest.yml
                  macos-test:
                    name: macos-py3-arm64
                    uses: ./.github/workflows/_mac-test.yml
                    with:
                      build-environment: macos-py3-arm64
                      test-matrix: |
                        { include: [{ config: "mps", shard: 1 }] }
                  sm100-test:
                    name: linux-jammy-aarch64-cuda13.2-py3.12-gcc13-sm100
                    uses: ./.github/workflows/_linux-test.yml
                    with:
                      build-environment: linux-jammy-aarch64-cuda13.2-py3.12-gcc13
                      force-no-td: true
                      test-matrix: |
                        { include: [{ config: "smoke_b200", shard: 1 }] }
            """,
            "dynamo-unittest.yml": """
                name: dynamo-unittest
                on:
                  workflow_call:
                  pull_request:
                jobs:
                  dynamo-test:
                    uses: ./.github/workflows/_linux-test.yml
                    with:
                      test-matrix: |
                        { include: [{ config: "dynamo_core", shard: 1 }] }
                  dynamo-no-td:
                    uses: ./.github/workflows/_linux-test.yml
                    with:
                      force-no-td: true
                      test-matrix: |
                        { include: [{ config: "dynamo_wrapped", shard: 1 }] }
            """,
            "rocm-rx7900.yml": """
                name: rocm-rx7900
                on: pull_request
                jobs:
                  test:
                    name: linux-noble-rocm-py3.12-rx7900
                    uses: ./.github/workflows/_rocm-test.yml
                    with:
                      tests-to-include: >-
                        test_nn
                        test_torch
                      test-matrix: |
                        { include: [{ config: "default", shard: 1 }] }
            """,
            "periodic.yml": """
                name: periodic
                on: schedule
                jobs:
                  test:
                    name: linux-jammy-cuda13.2-py3-gcc11-slow-gradcheck
                    uses: ./.github/workflows/_linux-test.yml
                    with:
                      test-matrix: |
                        { include: [{ config: "default", shard: 1 }] }
                  shared-test:
                    name: linux-jammy-py3.10-gcc11
                    uses: ./.github/workflows/_linux-test.yml
                    with:
                      test-matrix: |
                        { include: [{ config: "default", shard: 1 }] }
            """,
        }  # fmt: skip
        with tempfile.TemporaryDirectory() as d:
            for name, text in workflows.items():
                (Path(d) / name).write_text(textwrap.dedent(text))
            jobs = ci_jobs(Path(d))
        self.assertEqual(
            jobs[True],
            {
                ("pull", "linux-jammy-py3.10-gcc11", "default"),
                ("trunk", "dynamo-unittest / dynamo-test", "dynamo_core"),
            },
        )
        self.assertEqual(
            jobs[False],
            {
                ("inductor", "unit-test / inductor-test", "inductor_core"),
                ("dynamo-unittest", "dynamo-test", "dynamo_core"),
                (
                    "periodic",
                    "linux-jammy-cuda13.2-py3-gcc11-slow-gradcheck",
                    "default",
                ),
                ("periodic", "linux-jammy-py3.10-gcc11", "default"),
                # Not on macOS, in xpu or onnx builds, or under force-no-td, also inside a called workflow.
                ("trunk", "macos-py3-arm64", "mps"),
                ("pull", "linux-noble-xpu-n-py3.10", "default"),
                ("pull", "linux-jammy-py3-clang12-onnx", "default"),
                # test.sh exports NO_TD=1 for this config.
                ("pull", "dynamo-cpython-test", "dynamo_cpython"),
                ("pull", "dynamo-cpython-test", "dynamo_cpython_313"),
                (
                    "trunk",
                    "linux-jammy-aarch64-cuda13.2-py3.12-gcc13-sm100",
                    "smoke_b200",
                ),
                ("trunk", "dynamo-unittest / dynamo-no-td", "dynamo_wrapped"),
                ("dynamo-unittest", "dynamo-no-td", "dynamo_wrapped"),
                # TESTS_TO_INCLUDE turns TD off; pull's dispatch-input expression does not.
                ("rocm-rx7900", "linux-noble-rocm-py3.12-rx7900", "default"),
            },
        )
        truth_jobs = {
            "test_a": ["unit-test / inductor-test / test (inductor_core, 1, 2, linux.2xlarge)"],
            # The periodic job at an older CUDA version, and a pull job: only the pull failure counts.
            "test_b": [
                "linux-jammy-cuda12.8-py3-gcc11-slow-gradcheck / test (default, 1, 2, linux.2xlarge)",
                "linux-jammy-py3.10-gcc11 / test (default, 1, 2, linux.2xlarge)",
            ],
            # dynamo-unittest under trunk, with a matrix suffix, and under its own pull_request trigger.
            "test_c": [
                "dynamo-unittest / dynamo-test (3.11) / test (dynamo_core, 1, 2, linux.2xlarge)",
                "dynamo-test (3.12) / test (dynamo_core, 1, 2, linux.2xlarge)",
            ],
            # Unknown jobs fall back to their config.
            "test_d": ["linux-old / test (default, 1, 1, linux.2xlarge)", "linux-old / test (slow, 1, 1, linux.2xlarge)"],
            # A name periodic shares with pull counts by its recorded workflow, and as TD without one.
            "test_e": [["periodic", "linux-jammy-py3.10-gcc11 / test (default, 1, 1, linux.2xlarge)"]],
            "test_f": [["pull", "linux-jammy-py3.10-gcc11 / test (default, 1, 1, linux.2xlarge)"]],
            "test_g": ["linux-jammy-py3.10-gcc11 / test (default, 1, 1, linux.2xlarge)"],
            # A recorded TD workflow with an unknown job falls back to its config; a non-TD one never counts.
            "test_h": [["trunk", "linux-old / test (default, 1, 1, linux.2xlarge)"], ["inductor", "linux-old / test (slow, 1, 1, linux.2xlarge)"]],
            "test_i": [["rocm-rx7900", "linux-noble-rocm-py3.12-rx7900 / test (default, 1, 1, linux.rocm.gpu)"]],
            # An unknown macOS, xpu or onnx job does not fall back to its config.
            "test_j": [["trunk", "macos-old / test (default, 1, 1, macos-m1)"], "linux-xpu-old / test (default, 1, 1, linux.idc.xpu)"],
        }  # fmt: skip
        sample = {**SAMPLE, "kind": "revert", "truth": list(truth_jobs), "candidates": list(truth_jobs), "truth_jobs": truth_jobs}  # fmt: skip
        with patch_ci_jobs(jobs):
            got = truth_pairs(sample, strong=True)
            self.assertFalse(score.job_runs_td(truth_jobs["test_b"][0]))
            self.assertFalse(score.job_runs_td(truth_jobs["test_c"][1]))
            self.assertFalse(score.runs_td("mps"))
            self.assertFalse(score.runs_td("smoke_b200"))
        self.assertEqual(
            got,
            [
                ("test_b", "default"),
                ("test_c", "dynamo_core"),
                ("test_d", "default"),
                ("test_f", "default"),
                ("test_g", "default"),
                ("test_h", "default"),
            ],
        )

    def test_no_td_configs_match_test_sh(self) -> None:
        # The TEST_CONFIG patterns of test.sh's top-level dispatch that set NO_TD, inline or through a function they call.
        # A setter the scan cannot map to a pattern (a helper no branch calls, or a top-level NO_TD=) fails it.
        func_def = re.compile(r"(?:function\s+(\w+)|(\w+)\s*\(\s*\))")
        dispatch = re.compile(r'(?:el)?if \[\[ "?\$\{?TEST_CONFIG\}?"? ==? ["\']?([^ "\']+)')  # fmt: skip
        func, sets_no_td, branch, calls, compound = None, set(), None, {}, set()
        for line in (REPO_ROOT / ".ci/pytorch/test.sh").read_text().splitlines():
            if line.lstrip().startswith("#"):
                continue
            if m := func_def.match(line):
                func = m.group(1) or m.group(2)
            elif line.startswith("}"):
                func = None
            if m := dispatch.match(line):
                branch = m.group(1)
                if re.search(r"&&|\|\||-a |-o ", line):
                    compound.add(branch)
            elif re.match(r"\S", line):
                branch = None
            elif branch:
                calls.setdefault(branch, set()).update(re.findall(r"\w+", line))
            if re.search(r"\bNO_TD=", line):
                sets_no_td.add(func or branch)
        unmapped = {f for f in sets_no_td if f not in calls and not any(f in words for words in calls.values())}  # fmt: skip
        self.assertTrue(calls, "found no TEST_CONFIG dispatch in test.sh")
        self.assertFalse(unmapped, f"cannot map test.sh's NO_TD to TEST_CONFIG patterns (setters outside the dispatch: {unmapped}); update this scan or NO_TD_CONFIGS")  # fmt: skip
        dispatched = {p for p, words in calls.items() if p in sets_no_td or words & sets_no_td}  # fmt: skip
        # NO_TD_CONFIGS matches TEST_CONFIG alone, so it cannot express a branch's other conditions.
        self.assertFalse(dispatched & compound, f"test.sh sets NO_TD under compound conditions: {dispatched & compound}")  # fmt: skip
        self.assertEqual(dispatched, set(score.NO_TD_CONFIGS))

    def test_unasked(self) -> None:
        td = {"scored": ["test_a"], "top25": ["test_e"], "top25_cfg": {"default": ["test_a"]}}  # fmt: skip
        r = {"out": answer(tests=[{"file": "test_b"}])}
        huge = {**SAMPLE, "changed_files": [f"torch/f{i}.py" for i in range(select_tests.MAX_CHANGED_FILES + 1)]}  # fmt: skip
        empty = {**SAMPLE, "changed_files": []}
        infra = {**SAMPLE, "changed_files": ["test/run_test.py"]}
        for name in ("model", "final", "final-hm", "final+td45"):
            for s in (huge, empty, infra):
                self.assertTrue(pick(name, r, td, s).sel.run_all, name)
        # As apply_selection does not enforce a job with no test times, even a run-all keeps TD's cut there.
        zero = {**SECONDS, "zero": {"test_a": [0.0, 0.0]}}
        cut = {**td, "top25_cfg": {"zero": ["test_b"]}}
        for s, out in ((empty, None), (SAMPLE, answer(run_all=True))):
            choice = policy("final+td45", s, {"out": out}, cut, zero)
            if choice is None:
                raise AssertionError("final+td45 chose nothing")
            self.assertEqual([choice.covers(f, c) for f, c in (("test_a", "zero"), ("test_b", "zero"), ("test_c", "default"))], [False, True, True])  # fmt: skip
        files = {
            "test-times.json": {"linux": {"default": {"test_a": 10.0}}},
            "samples/a.json": huge,
            "td/a.json": td,
        }  # fmt: skip
        with (
            tempfile.TemporaryDirectory() as d,
            patch_ci_jobs({True: {("pull", "linux", "default")}, False: set()}),
        ):
            data = write_data(d, files)
            rows = score.score(data, ["a"], ["m"], ["model"], True, data / "samples", data / "td", "")  # fmt: skip
        self.assertEqual([(x["run_all"], x["cost_usd"], x["error"]) for x in rows], [(True, None, False)])  # fmt: skip

    def test_hedge(self) -> None:
        # test_e has no recorded time and test_d no positive score, so neither is ever added.
        td = {"scored": ["test_e", "test_a", "test_b", "test_c"], "top25": [], "top25_cfg": {}}  # fmt: skip
        # default: 5% of 250s fits only test_c; slow: test_a (200s) does not fit 5% of 200s.
        choice = pick("final+td5", NOTHING, td)
        self.assertEqual(choice.per_config, {"default": {"test_c"}, "slow": set()})
        # A config in which nothing runs keeps TD's cut, as Selection.split does.
        cut = {**td, "top25_cfg": {"slow": ["test_a"]}}
        self.assertEqual(pick("final+td5", NOTHING, cut).per_config["slow"], {"test_a"})
        r = {"out": answer(tests=[{"file": "test_c"}])}
        self.assertEqual(pick("final+td5", r, cut).per_config["slow"], set())
        # So does an untimed file, which is in every job's test list.
        r = {"out": answer(tests=[{"file": "test_e"}])}
        self.assertEqual(pick("final+td5", r, cut).per_config["slow"], set())
        # A 20% share: 50s of default fits test_d (40s) and test_c, not test_b after them.
        ranked = {"scored": ["test_d", "test_c", "test_b"], "top25": [], "top25_cfg": {}}  # fmt: skip
        with mock.patch.object(score, "LLM_SELECTOR_HEDGE", 0.2):
            self.assertEqual(pick("final+td", NOTHING, ranked).per_config, {"default": {"test_d", "test_c"}, "slow": set()})  # fmt: skip
        with mock.patch.object(score, "LLM_SELECTOR_HEDGE", 0.45):
            self.assertEqual(pick("final+td", NOTHING, td).per_config, pick("final+td45", NOTHING, td).per_config)  # fmt: skip
        self.assertEqual(pick("final+td5", NOTHING, ranked).per_config["default"], {"test_c"})  # fmt: skip
        # 45% of 250s: test_a fits, test_b then does not, test_c after it still does.
        choice = pick("final+td45", NOTHING, td)
        self.assertEqual(
            choice.per_config, {"default": {"test_a", "test_c"}, "slow": set()}
        )
        self.assertTrue(choice.covers("test_a", "default"))
        self.assertFalse(choice.covers("test_a", "slow"))
        # A timed file that is not a candidate adds nothing to the budget.
        extra = {**SECONDS, "default": {**SECONDS["default"], "test_x": [250.0, 0.0]}}
        choice = policy("final+td45", SAMPLE, NOTHING, td, extra)
        if choice is None:
            raise AssertionError("final+td45 chose nothing")
        self.assertEqual(choice.per_config["default"], {"test_a", "test_c"})
        # With no fallback job times, unknown configs hedge nothing and keep TD's cut.
        self.assertEqual(pick("final+td45", NOTHING, {**td, "top25": ["test_b"]}).other, {"test_b"})  # fmt: skip
        # As in run_test.py, an unknown config is hedged over the candidates in default/<config>, else default/default;
        # counting the non-candidate test_x would let test_b fit too.
        fallback = {"default": {"test_a": 30.0, "test_b": 40.0, "test_d": 30.0, "test_x": 100.0}, "rare": {"test_b": 1.0, "test_c": 9.0}}  # fmt: skip
        choice = policy("final+td45", SAMPLE, NOTHING, td, SECONDS, None, fallback)
        if choice is None:
            raise AssertionError("final+td45 chose nothing")
        self.assertEqual((choice.other, choice.per_config["rare"]), ({"test_a"}, {"test_b"}))  # fmt: skip
        self.assertEqual(choice.per_config["default"], {"test_a", "test_c"})
        self.assertTrue(choice.covers("test_a", "unknown"))
        # Files the selection already covers cost no budget.
        r = {"out": answer(tests=[{"file": "test_a"}])}
        self.assertEqual(
            pick("final+td45", r, td).per_config["default"],
            {"test_b", "test_c"},
        )
        self.assertIsNone(policy("final+td5", SAMPLE, NOTHING, None, SECONDS))
        for bad in ("final+tdx", "final+td5x", "nonsense"):
            with self.assertRaisesRegex(ValueError, "unknown policy"):
                policy(bad, SAMPLE, NOTHING, td, SECONDS)

    def test_policy_matches_apply_selection(self) -> None:
        # What final+td runs in a job is what apply_selection runs there, including a job with
        # no test times, a run-all, a failed or malformed answer, a job timed by a fallback job,
        # sweeps, dirs, and an inductor config whose glob always covers some files.
        rng = random.Random(0)
        files = [f"test_{i}" for i in range(5)] + ["dynamo/test_a", "dynamo/test_b", "test_torch", "test_ops"]  # fmt: skip
        diverged = 0
        for _ in range(2000):
            seconds = {f: float(rng.choice([0, 0, 5, 20, 60])) for f in files if rng.random() < 0.6}  # fmt: skip
            job = [*seconds, *(f for f in files if f not in seconds and rng.random() < 0.5)]  # fmt: skip
            rng.shuffle(job)
            n = rng.randint(0, len(job))
            include, exclude = [TestRun(f) for f in job[:n]], [TestRun(f) for f in job[n:]]  # fmt: skip
            scored = [f for f in job if rng.random() < 0.6]
            picked = rng.sample(job, min(len(job), rng.randint(0, 2)))
            prev = rng.sample(files, rng.randint(0, 2))
            changed = rng.choice([["torch/foo.py"]] * 3 + [[]])
            s = {**SAMPLE, "candidates": job, "truth": [], "previous_failures": prev, "changed_files": changed}  # fmt: skip
            sweeps = rng.sample(["dynamo_wrapped", "crossref"], rng.randint(0, 1))
            chosen = answer(tests=[{"file": f} for f in picked], sweeps=sweeps, dirs=rng.choice([[], [], ["dynamo"]]))  # fmt: skip
            out = rng.choice([chosen] * 3 + [answer(run_all=True), {"tests": "test_0"}, None])  # fmt: skip
            # The job's times are its own, the fallback job's for its config, or default/default's.
            config = rng.choice(["c", "inductor", "dynamo_wrapped", "crossref"])
            how = rng.choice(["own", "fallback", "default"])
            own = {config: {f: [t, 0.0] for f, t in seconds.items()}} if how == "own" else {}  # fmt: skip
            fallback = {} if how == "own" else {config if how == "fallback" else "default": seconds}  # fmt: skip
            td = {"scored": scored, "top25": job[:n], "top25_cfg": {config: job[:n]} if how == "own" else {}}  # fmt: skip
            choice = policy("final+td", s, {"out": out}, td, own, None, fallback)
            if choice is None:
                raise AssertionError("final+td chose nothing")
            # What select_tests writes: no model call without changed files, and an error is never enforced.
            sel = Selection(run_all=True) if not changed else Selection.from_model_output(out, set(job), changed)  # fmt: skip
            sel.enforce = sel.error is None
            applied, ran = {}, {}
            for c in (config, ""):
                with (
                    mock.patch.object(selection, "ENFORCEMENT_ENABLED", True),
                    mock.patch.object(selection, "get_pr_number", return_value=1),
                    mock.patch("sys.stderr", io.StringIO()),
                ):
                    applied[c] = apply_selection(sel, include, exclude, lambda x: float(x.test_file in scored), seconds, test_config=c, td_enabled=True, previous_failures=prev)  # fmt: skip
                ran[c] = {x.test_file for x in (applied[c].split.include if applied[c].enforced else include)}  # fmt: skip
            case = (config, how, seconds, job, n, scored, out, prev, changed)
            predicted = {f for f in job if choice.covers(f, config)}
            if how == "default":
                # Known divergence: a config that no job records is hedged as config "",
                # whose coverage lacks the config's sweep and ALWAYS_COVERED files.
                blank = applied[""]
                kept = blank.enforced or (blank.kept_td and sum(blank.file_time.values()) > 0)  # fmt: skip
                self.assertEqual(predicted, ran[""] | ({f for f in job if sel.covers(f, config)} if kept else set()), case)  # fmt: skip
                if predicted != ran[config]:
                    diverged += 1
                    self.assertTrue(any(sel.covers(f, config) != sel.covers(f, "") for f in job), case)  # fmt: skip
                continue
            self.assertEqual(predicted, ran[config], case)
        self.assertGreater(diverged, 0)

    def test_hedge_previous_failures(self) -> None:
        # Previous failures in TD's cut run outside the budget; test_b is not in the cut.
        td = {"scored": ["test_a", "test_c"], "top25": ["test_a"], "top25_cfg": {"default": ["test_a"], "slow": []}}  # fmt: skip
        s = {**SAMPLE, "previous_failures": ["test_a", "test_b"]}
        choice = pick("final+td5", NOTHING, td, s)
        self.assertEqual(choice.per_config, {"default": {"test_a", "test_c"}, "slow": set()})  # fmt: skip
        # Without fallback times other jobs are not enforced and run TD's cut.
        self.assertEqual(choice.other, {"test_a"})
        self.assertEqual(pick("final+td5", NOTHING, td).per_config["default"], {"test_c"})  # fmt: skip

    def test_hedge_failed_call_keeps_td(self) -> None:
        td = {"scored": ["test_a"], "top25": ["test_e"], "top25_cfg": {"default": ["test_a"], "slow": []}}  # fmt: skip
        for r in ({"out": None}, None, {"out": {"tests": "test_a"}}):
            choice = pick("final+td45", r, td)
            self.assertFalse(choice.sel.run_all)
            self.assertEqual(choice.per_config, {"default": {"test_a"}, "slow": set()})
            self.assertEqual(choice.other, {"test_e"})
            self.assertFalse(choice.covers("test_torch", "inductor"))
        # A deliberate run-all still runs everything.
        self.assertTrue(pick("final+td45", {"out": answer(run_all=True)}, td).sel.run_all)  # fmt: skip
        infra = {**SAMPLE, "changed_files": ["test/run_test.py"]}
        self.assertTrue(pick("final+td45", {"out": None}, td, infra).sel.run_all)

    def test_policy_passes_deleted(self) -> None:
        # A deleted data file under test/ needs no reader, as select_tests passes it.
        s = {**SAMPLE, "changed_files": ["test/data/gone.json"], "deleted": ["test/data/gone.json"]}  # fmt: skip
        td = {"scored": [], "top25": [], "top25_cfg": {"default": [], "slow": []}}
        out = {"out": answer(tests=[{"file": "test_a", "confidence": "low"}])}
        for name in ("final", "final-hm", "final+td45"):
            self.assertFalse(pick(name, out, td, s, lambda p: []).sel.run_all, name)
            self.assertTrue(pick(name, out, td, {**s, "deleted": []}, lambda p: []).sel.run_all, name)  # fmt: skip

    def test_final_hm_malformed(self) -> None:
        # Malformed answers run everything, as under final.
        tests = ["test_a", {"file": "test_c", "confidence": "low"}, {"file": "test_b", "confidence": "high"}]  # fmt: skip
        for out in ({"tests": None}, {"tests": tests}, ["test_a"], "test_a", None):
            self.assertTrue(pick("final-hm", {"out": out}, None).sel.run_all)
            self.assertTrue(pick("final", {"out": out}, None).sel.run_all)
        tests = [{"file": "test_c", "confidence": "low"}, {"file": "test_b", "confidence": "high"}, {"file": "test_a", "confidence": ["high"]}]  # fmt: skip
        self.assertEqual(
            pick("final-hm", {"out": answer(tests=tests)}, None).sel.files, {"test_b"}
        )
        # An unmatched name makes the answer malformed under both.
        tests = [{"file": "test_a", "confidence": "low"}, {"file": "bogus", "confidence": "high"}]  # fmt: skip
        for name in ("final", "final-hm"):
            self.assertTrue(pick(name, {"out": answer(tests=tests)}, None).sel.run_all)
        # With grep, a file a driver runs names the driver under every policy.
        jit = {**SAMPLE, "candidates": [*SAMPLE["candidates"], "test_jit"]}
        out = answer(tests=[{"file": "jit/test_freezing", "confidence": "high"}])
        for name in ("model", "final", "final-hm"):
            choice = policy(name, jit, {"out": out}, None, SECONDS, lambda p: ["test/jit/test_freezing.py"] if p == "^" else [])  # fmt: skip
            if choice is None:
                raise AssertionError(f"{name} chose nothing")
            self.assertEqual((choice.sel.files, choice.sel.run_all), ({"test_jit"}, False), name)  # fmt: skip
        # Filtering out every name leaves a well-formed empty answer.
        tests = [{"file": "test_a", "confidence": "low"}]
        self.assertEqual(pick("final", {"out": answer(tests=tests)}, None).sel.files, {"test_a"})  # fmt: skip
        empty = pick("final-hm", {"out": answer(tests=tests)}, None).sel
        self.assertEqual((empty.run_all, empty.files), (False, set()))
        hm = pick("final-hm", {"out": answer(tests=tests, dirs=["sub"])}, None, {**SAMPLE, "candidates": [*SAMPLE["candidates"], "sub/test_x"]})  # fmt: skip
        self.assertEqual((hm.sel.run_all, hm.sel.files), (False, {"sub/test_x"}))

    def test_score_same_ids(self) -> None:
        sample = {**SAMPLE, "truth_jobs": {"test_a": [["pull", "linux / test (default, 1, 2, linux.2xlarge)"]]}}  # fmt: skip
        td = {"scored": [], "top25": ["test_a"], "top25_cfg": {"default": ["test_a"]}}
        result = {
            "out": answer(tests=[{"file": "test_a"}]),
            "error": None,
            "cost_usd": 0.1,
        }
        failed = {"out": None, "error": "boom", "cost_usd": 0.0}
        # The top-level "default" is run_test.py's fallback for unknown jobs, not a job to price.
        times = {
            "linux": {"default": {"test_a": 10.0, "test_b": 5.0}},
            "linux-cuda": {"default": {"test_a": 20.0}},
            "default": {"default": {"test_a": 1000.0, "test_b": 1000.0}},
        }
        files = {
            "test-times.json": times,
            "samples/a.json": sample, "samples/b.json": sample, "samples/c.json": sample,
            "td/a.json": td, "td/b.json": td,
            "results/m1/a.json": result, "results/m1/b.json": result, "results/m1/c.json": result,
            "results/m2/a.json": failed, "results/m2/c.json": failed,
            # Not a model's: td is scored only on its replay.
            "results/td/a.json": {**failed, "cost_usd": 5.0},
        }  # fmt: skip
        with (
            tempfile.TemporaryDirectory() as d,
            patch_ci_jobs({True: {("pull", "linux", "default")}, False: set()}),
        ):
            data = write_data(d, files)
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                dirs = data / "samples", data / "td"
                rows = score.score(data, ["a", "b", "c", "a"], ["m1", "m2", "td"], ["model"], True, *dirs, "")  # fmt: skip
            b = data / "results/m1/b.json"
            b.write_text(json.dumps({**json.loads(b.read_text()), "effort": "high"}))
            with self.assertRaisesRegex(ValueError, r"\['m1'\] mix"):
                score.score(data, ["a", "b"], ["m1"], ["model"], True, *dirs, "")
        self.assertEqual(sorted((x["tag"], x["policy"], x["id"]) for x in rows), [("m1", "model", "a"), ("m2", "model", "a"), ("td", "td25", "a")])  # fmt: skip
        self.assertEqual(err.getvalue(), "dropped 1 of 3 ids with no m2 result\ndropped 1 of 3 ids with no td replay\n")  # fmt: skip
        fields = ("pairs", "hit", "missed", "cpu", "gpu", "run_all", "files", "error")
        by_tag = {x["tag"]: tuple(x[k] for k in fields) for x in rows}
        self.assertEqual(by_tag, {
            "m1": (1, 1, [], 10 / 15, 1.0, False, 1, False),
            "m2": (1, 1, [], 1.0, 1.0, True, 0, True),
            "td": (1, 1, [], 10 / 15, 1.0, False, 1, False),
        })  # fmt: skip

    def test_summarize(self) -> None:
        row = {"tag": "t", "policy": "p", "kind": "pr", "cost_usd": 0.1, "error": False, "cpu": 0.5, "gpu": 0.0}  # fmt: skip
        run_all = {"files": 0, "run_all": True, "cpu": 1.0, "gpu": 1.0, "error": True}
        rows = [
            {**row, "pairs": 2, "hit": 1, "files": 3, "run_all": False},
            {**row, "pairs": 1, "hit": 1, "files": 5, "run_all": False},
            {**row, "pairs": 0, "hit": 0, "files": 9, "run_all": False},
            {**row, "pairs": 1, "hit": 1, **run_all},
            {**row, "pairs": 0, "hit": 0, **run_all, "error": False, "cost_usd": None},
        ]
        (s,) = score.summarize(rows)
        self.assertEqual(s, {
            "tag": "t", "policy": "p", "kind": "pr", "n": 5, "n_failing": 3,
            "caught_any": 1.0, "caught_all": 2 / 3, "pair_recall": 3 / 4,
            "cpu": 0.7, "gpu": 0.4, "run_all": 2, "median_files": 5,
            "usd_per_call": 0.1, "errors": 1,
        })  # fmt: skip

    def test_score_grep(self) -> None:
        sample = {**SAMPLE, "sha": "abc", "changed_files": ["test/test_c.py"]}
        files = {
            "test-times.json": {"linux": {"default": {"test_a": 10.0}}},
            "samples/a.json": sample,
            "td/a.json": {"scored": [], "top25": [], "top25_cfg": {}},
            "results/m/a.json": {"out": answer(), "error": None, "cost_usd": 0.1},
        }  # fmt: skip
        for grep, n_files, msg in (
            (mock.Mock(return_value=["test/test_a.py"]), [2, 2], ""),
            (mock.Mock(side_effect=RuntimeError("bad revision")), [], "dropped 1 of 1 ids with no working git grep at their head (not in this clone?)\n"),
        ):  # fmt: skip
            with (
                tempfile.TemporaryDirectory() as d,
                patch_ci_jobs({True: {("pull", "linux", "default")}, False: set()}),
                mock.patch.object(score, "grep_at", grep),
            ):
                data = write_data(d, files)
                err = io.StringIO()
                with contextlib.redirect_stderr(err):
                    rows = score.score(data, ["a"], ["m"], ["final", "final-hm"], True, data / "samples", data / "td", "")  # fmt: skip
            self.assertEqual([x["files"] for x in rows], n_files)
            self.assertEqual(err.getvalue(), msg)
            self.assertEqual(grep.call_args.args[1], "abc")
        # A head missing from the clone drops the sample also under policies that never grep.
        msg = "dropped 1 of 1 ids with no working git grep at their head (not in this clone?)\n"
        with (
            tempfile.TemporaryDirectory() as d,
            patch_ci_jobs({True: {("pull", "linux", "default")}, False: set()}),
            mock.patch.object(score, "has_head", return_value=False) as has_head,
        ):
            data = write_data(d, files)
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                dirs = data / "samples", data / "td"
                rows = score.score(data, ["a"], ["m", "td"], ["model"], True, *dirs, "")
        self.assertEqual((rows, err.getvalue()), ([], msg))
        has_head.assert_called_once_with("abc")

    def test_score_drops_a_sample_without_main_base(self) -> None:
        # The diff adds a candidate, which sample_prompt looks up at main_base.
        diff = "diff --git a/test/test_e.py b/test/test_e.py\nnew file mode 100644\n"
        sample = {**SAMPLE, "diff": diff, "changed_files": ["test/test_e.py"], "main_base": "0" * 40}  # fmt: skip
        td = {"scored": [], "top25": [], "top25_cfg": {}}
        result = {"out": answer(), "error": None, "cost_usd": 0.1, "prompt_sha": "x"}  # fmt: skip
        files = {
            "test-times.json": {}, "samples/a.json": SAMPLE, "samples/b.json": sample,
            "td/a.json": td, "td/b.json": td, "results/m/a.json": {**result, "prompt_sha": score.prompt_sha(score.sample_prompt(SAMPLE, {}))}, "results/m/b.json": result,
        }  # fmt: skip
        with (
            tempfile.TemporaryDirectory() as d,
            patch_ci_jobs({True: {("pull", "linux", "default")}, False: set()}),
        ):
            data = write_data(d, files)
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                rows = score.score(data, ["a", "b"], ["m", "td"], ["model"], True, data / "samples", data / "td", "")  # fmt: skip
        self.assertEqual({x["id"] for x in rows}, {"a"})
        self.assertEqual(err.getvalue(), "dropped 1 of 2 ids with no main_base (default base) in this clone\n")  # fmt: skip

    def test_grep_head_copies(self) -> None:
        cached = functools.cache(lambda pattern, sha: ["test/test_a.py"])
        with mock.patch.object(score, "grep_at", cached):
            grep = score.grep_head({"sha": "abc"}, "a", set())
            grep("p").append("test/test_b.py")
            self.assertEqual(grep("p"), ["test/test_a.py"])

    def test_score_errors_and_stale_replays(self) -> None:
        td = {"scored": [], "top25": [], "top25_cfg": {}}
        files = {
            "test-times.json": {"linux": {"default": {"test_a": 10.0}}},
            "samples/a.json": SAMPLE, "samples/b.json": SAMPLE, "samples/c.json": SAMPLE,
            "td/a.json": td, "td/b.json": td, "td/c.json": td,
            "results/m/a.json": {"out": None, "error": "429 rate limited", "cost_usd": 0.1, "transient": True},
            "results/m/b.json": {"out": answer(), "error": None, "cost_usd": 0.1},
            "results/m/c.json": {"out": answer(tests=[{"file": "bogus"}]), "error": None, "cost_usd": 0.1},
        }  # fmt: skip
        with (
            tempfile.TemporaryDirectory() as d,
            patch_ci_jobs({True: {("pull", "linux", "default")}, False: set()}),
            # No file under test/ is named bogus.
            mock.patch.object(score, "grep_at", return_value=[]),
        ):
            data = write_data(d, files)
            (data / "samples/b.json").write_text(json.dumps({**SAMPLE, "sha": "rebuilt"}))  # fmt: skip
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                rows = score.score(data, ["a", "b", "c"], ["m"], ["model", "final"], True, data / "samples", data / "td", "")  # fmt: skip
                # Replays of another STATS snapshot or TD code are stale too, even ones that record it.
                (data / "td/a.json").write_text(json.dumps({**json.loads((data / "td/a.json").read_text()), "stats": "new"}))  # fmt: skip
                stale = score.score(data, ["a", "b"], ["m"], ["model"], True, data / "samples", data / "td", "new")  # fmt: skip
        # A failed model call and a malformed answer run everything and count as errors, as select_tests records them.
        self.assertEqual([(x["id"], x["run_all"], x["error"]) for x in rows], [("a", True, True)] * 2 + [("c", True, True)] * 2)  # fmt: skip
        self.assertEqual(stale, [])
        why = "td replay of the current test-times.json, sample, STATS and TD code"
        transient = "scored 1 m results as errors that still had transient (throttled or 5xx) errors after run_eval's retries"
        self.assertEqual(err.getvalue(), f"dropped 1 of 3 ids with no {why}\n{transient}\ndropped 2 of 2 ids with no {why}\n")  # fmt: skip

    def test_score_stale_replay_of_test_times(self) -> None:
        files = {
            "test-times.json": {"linux": {"default": {"test_a": 10.0}}},
            "samples/a.json": SAMPLE,
            "td/a.json": {"scored": [], "top25": [], "top25_cfg": {}},
        }
        with (
            tempfile.TemporaryDirectory() as d,
            patch_ci_jobs({True: {("pull", "linux", "default")}, False: set()}),
        ):
            data = write_data(d, files)
            (data / "test-times.json").write_text(json.dumps({"linux": {"default": {"test_a": 20.0}}}))  # fmt: skip
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                rows = score.score(data, ["a"], ["td"], [], True, data / "samples", data / "td", "")  # fmt: skip
        self.assertEqual(rows, [])
        self.assertEqual(err.getvalue(), "dropped 1 of 1 ids with no td replay of the current test-times.json, sample, STATS and TD code\n")  # fmt: skip

    def test_score_stale_results(self) -> None:
        td = {"scored": [], "top25": [], "top25_cfg": {}}
        result = {"out": answer(), "error": None, "cost_usd": 0.1}
        files = {
            "test-times.json": {"linux": {"default": {"test_a": 10.0}}},
            "samples/a.json": SAMPLE, "samples/b.json": SAMPLE, "samples/c.json": SAMPLE,
            "td/a.json": td, "td/b.json": td, "td/c.json": td,
            "results/m/a.json": result, "results/m/b.json": {**result, "prompt_sha": "old"}, "results/m/c.json": result,
        }  # fmt: skip
        with (
            tempfile.TemporaryDirectory() as d,
            patch_ci_jobs({True: {("pull", "linux", "default")}, False: set()}),
        ):
            data = write_data(d, files)
            # run_eval always records prompt_sha, so a result without one is stale too.
            (data / "results/m/c.json").write_text(json.dumps(result))
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                rows = score.score(data, ["a", "b", "c", "gone"], ["m", "td"], ["model"], True, data / "samples", data / "td", "")  # fmt: skip
        self.assertEqual({x["id"] for x in rows}, {"a"})
        self.assertEqual(err.getvalue(), "dropped 2 of 4 ids with no m result of the current sample, test-times.json, prompt and schema\ndropped 1 of 4 ids with no sample\n")  # fmt: skip

    def test_score_results_of_another_prompt_are_stale(self) -> None:
        # The prompt carries the sample's diff and each candidate's minutes from test-times.json.
        times = {"linux": {"default": {"test_a": 600.0}}}
        sent = score.prompt_sha(score.sample_prompt(SAMPLE, select_tests.test_minutes(times)))  # fmt: skip
        untimed = score.prompt_sha(score.sample_prompt(SAMPLE, {}))
        td = {"scored": [], "top25": [], "top25_cfg": {}}
        result = {"out": answer(), "error": None, "cost_usd": 0.1}
        files = {
            "test-times.json": times,
            "samples/a.json": SAMPLE, "samples/b.json": {**SAMPLE, "diff": "+x"}, "samples/c.json": SAMPLE,
            "td/a.json": td, "td/b.json": td, "td/c.json": td,
            "results/m/a.json": {**result, "prompt_sha": sent}, "results/m/b.json": {**result, "prompt_sha": sent}, "results/m/c.json": {**result, "prompt_sha": untimed},
        }  # fmt: skip
        with (
            tempfile.TemporaryDirectory() as d,
            patch_ci_jobs({True: {("pull", "linux", "default")}, False: set()}),
        ):
            data = write_data(d, files)
            with contextlib.redirect_stderr(io.StringIO()):
                rows = score.score(data, ["a", "b", "c"], ["m"], ["model"], True, data / "samples", data / "td", "")  # fmt: skip
        self.assertEqual({x["id"] for x in rows}, {"a"})

    def test_score_agentic_results(self) -> None:
        # An agentic result is current only with the agentic prompt's hash.
        prompt = score.sample_prompt(SAMPLE, {})
        td = {"scored": [], "top25": [], "top25_cfg": {}}
        result = {"out": answer(), "error": None, "cost_usd": 0.1, "agentic": True}
        files = {
            "test-times.json": {},
            "samples/a.json": SAMPLE, "samples/b.json": SAMPLE, "td/a.json": td, "td/b.json": td,
            "results/m/a.json": {**result, "prompt_sha": score.prompt_sha(prompt, True)}, "results/m/b.json": {**result, "prompt_sha": score.prompt_sha(prompt)},
        }  # fmt: skip
        with (
            tempfile.TemporaryDirectory() as d,
            patch_ci_jobs({True: {("pull", "linux", "default")}, False: set()}),
        ):
            data = write_data(d, files)
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                rows = score.score(data, ["a", "b"], ["m"], ["model"], True, data / "samples", data / "td", "")  # fmt: skip
        self.assertEqual({x["id"] for x in rows}, {"a"})
        self.assertIn("dropped 1 of 2 ids with no m result of the current sample", err.getvalue())  # fmt: skip

    def test_sample_prompt_is_the_ci_prompt(self) -> None:
        env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull, GIT_CONFIG_NOSYSTEM="1", GIT_AUTHOR_NAME="a", GIT_AUTHOR_EMAIL="a@a", GIT_COMMITTER_NAME="a", GIT_COMMITTER_EMAIL="a@a")  # fmt: skip
        times = {"linux": {"default": {"test_foo": 600.0}}}
        with (
            tempfile.TemporaryDirectory() as repo,
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch.object(select_tests, "REPO_ROOT", Path(repo)),
            mock.patch.object(select_tests, "TESTS", ["test_bar", "test_foo", "test_listed"]),
            mock.patch.object(select_tests, "get_test_times", return_value=times),
        ):  # fmt: skip
            git, test = select_tests.git, Path(repo) / "test"
            git("init", "-q")
            (test / "nn").mkdir(parents=True)
            (test / "test_foo.py").write_text("x = 1\n" * 50)
            (test / "test_bar.py").write_text("")
            git("add", "-A")
            git("commit", "-qm", "base")
            base = git("rev-parse", "HEAD").strip()
            # Main, whose TESTS CI reads, added test_listed after the PR's merge base.
            (test / "test_listed.py").write_text("")
            git("add", "-A")
            git("commit", "-qm", "main")
            main = git("rev-parse", "HEAD").strip()
            git("checkout", "-q", base)
            git("mv", "test/test_foo.py", "test/nn/test_foo.py")
            (test / "test_new.py").write_text("y = 2\n")
            (test / "test_listed.py").write_text("z = 3\n")
            (test / "helper.py").write_text("")
            (test / "test_bar.py").write_text("import helper\n")
            git("add", "-A")
            git("commit", "-qm", "head")
            diffstat, diff = select_tests.diffstat_and_diff(base, "HEAD")
            with contextlib.redirect_stdout(io.StringIO()) as out:
                select_tests.main(["--title", "t", "--body", "b", "--base", main, "--print-prompt"])  # fmt: skip
            candidates = ["nn/test_foo", "test_bar", "test_foo", "test_listed", "test_new"]  # fmt: skip
            sample = {"title": "t", "body": "b", "diffstat": diffstat, "diff": diff, "candidates": candidates, "base": base, "main_base": main}  # fmt: skip
            minutes = select_tests.test_minutes(times)
            prompt = score.sample_prompt(sample, minutes)
            unlisted = score.sample_prompt({**sample, "main_base": None}, minutes)
        self.assertEqual(prompt + "\n", out.getvalue())
        self.assertIn("rename to test/nn/test_foo.py", diff)
        self.assertIn("b/test/test_listed.py\nnew file mode ", diff)
        self.assertIn("<added_test_files>\nnn/test_foo [0m]\ntest_new [0m]\n</added_test_files>", prompt)  # fmt: skip
        # Diffed against main_base, which a sample without one defaults to, test_listed is new.
        self.assertIn("<added_test_files>\nnn/test_foo [0m]\ntest_listed [0m]\ntest_new [0m]\n</added_test_files>", unlisted)  # fmt: skip

    def test_test_seconds_skips_non_td_envs(self) -> None:
        times = {
            "linux": {"default": {"test_a": 10.0}},
            "macos-py3-arm64": {"default": {"test_a": 100.0}},
            "linux-jammy-xpu-n-py3.10": {"default": {"test_a": 1000.0}},
            "linux-jammy-py3-clang12-onnx": {"default": {"test_b": 5.0}},
        }
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "test-times.json"
            path.write_text(json.dumps(times))
            self.assertEqual(score.test_seconds(path), {"default": {"test_a": [10.0, 0.0]}})  # fmt: skip

    def test_replay_stats(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            stats, code = Path(d) / "stats", Path(d) / "td"
            for p in [*(stats / a for a in score.STATS_ARTIFACTS), code / "heuristics/h.py"]:  # fmt: skip
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("1")
            replay = Path(d) / "td_replay.py"
            replay.write_text("1")
            with (
                mock.patch.object(score, "TD_CODE", code),
                mock.patch.object(score, "REPLAY_CODE", [replay]),
            ):
                digests = [score.replay_stats(stats)]
                (stats / score.STATS_ARTIFACTS[0]).write_text("2")
                digests.append(score.replay_stats(stats))
                (code / "heuristics/h.py").write_text("2")
                digests += [score.replay_stats(stats), score.replay_stats(stats)]
                replay.write_text("2")
                digests.append(score.replay_stats(stats))
                with mock.patch.object(score, "NO_TD_ENV", re.compile("macos")):
                    digests.append(score.replay_stats(stats))
        self.assertEqual(len(set(digests)), 5)
        self.assertEqual(digests[2], digests[3])
        self.assertIn(Path(score.__file__).with_name("td_replay.py"), score.REPLAY_CODE)

    def test_td_replay_cuts_as_ci(self) -> None:
        # td_replay hard-codes split_tests' cut with TD on; replay_stats hashes td_replay but not test_selections.
        from tools.testing import test_selections

        class Scores:
            def amend_tests(self, files: list[str]) -> None:
                pass

            def get_top_per_tests(self, percent: int) -> tuple[list, list]:
                cuts.append(percent)
                return [], []

        cuts: list[int] = []
        top25_per_config(Scores(), {"default": set()})  # type: ignore[arg-type]
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stderr(io.StringIO()):  # fmt: skip
            test_selections.split_tests(Scores(), {}, selection_file=Path(d) / "none.json", test_config="default", td_enabled=True, previous_failures=None, record=False)  # type: ignore[arg-type]  # fmt: skip
        self.assertEqual(cuts, [25, 25])
        replay = Path(td_replay.__file__).read_text()
        self.assertEqual(replay.count("get_top_per_tests(25)"), 2)

    def test_td_replay_main(self) -> None:
        from tools.testing.target_determination import heuristics

        class Trial:
            name, trial_mode = "Trial", True

        seen = []

        def prioritize(cands: list[str], file: io.StringIO) -> mock.Mock:
            # What td_replay patched into the heuristics modules for this sample.
            mod = {m: sys.modules[td_replay.H + m] for m in ("historical_class_failure_correlation", "previously_failed_in_pr", "mentioned_in_pr", "llm", *td_replay.CHANGED_FILES_READERS)}  # fmt: skip
            roots = {mod[m].REPO_ROOT for m in ("utils", "historical_class_failure_correlation")}  # fmt: skip
            changed = [mod[m].query_changed_files() for m in td_replay.CHANGED_FILES_READERS]  # fmt: skip
            failed = mod["previously_failed_in_pr"]
            prev = failed.get_previous_failures(), failed.read_additional_test_failures_file()  # fmt: skip
            mentioned, info = mod["mentioned_in_pr"], mod["mentioned_in_pr"].get_git_commit_info()  # fmt: skip
            # As MentionedInPR does: the PR's body and those of the issues the commit text links.
            bodies = [mentioned.get_pr_number(), *(mentioned.get_issue_or_pr_body(int(n)) for n in re.findall(r"#(\d+)", info))]  # fmt: skip
            seen.append((roots, prev, info, bodies, mod["llm"].LLM().get_mappings(), changed))  # fmt: skip
            # A heuristic that raised has no results; get_aggregated_priorities leaves a trial_mode one out anyway.
            ran = [h.name for h in heuristics.HEURISTICS if not h.trial_mode]
            if "test_e" not in cands:
                print(f"Error in {ran.pop()}: x", file=file)
            tp = TestPrioritizations(cands, {TestRun("test_b"): 2.0, TestRun("test_a", included=["TestX"]): 1.0})  # fmt: skip
            return mock.Mock(get_aggregated_priorities=lambda: tp, to_json=lambda: dict.fromkeys(ran))  # fmt: skip

        with tempfile.TemporaryDirectory() as d:
            stats = Path(d) / ".additional_ci_files"
            stats.mkdir()
            for a in score.STATS_ARTIFACTS:
                (stats / a).write_text("{}")
            a = {**SAMPLE, "previous_failures": ["test_c"]}
            broken = {**SAMPLE, "candidates": ["test_a", "test_b"], "changed_files": ["torch/bar.py", "docs/b.md"], "title": "u", "body": "v #12"}  # fmt: skip
            data = write_data(d, {"test-times.json": {"linux": {"default": {"test_a": 100.0, "test_c": 10.0}}}, "samples/a.json": a, "samples/b.json": broken, "td/b.json": {}})  # fmt: skip

            def replay(*argv: str) -> None:
                with (
                    mock.patch.object(sys, "argv", ["td_replay", "--data", d, "--stats", str(stats), *argv]),
                    mock.patch("tools.testing.target_determination.determinator.get_test_prioritizations", prioritize),
                    mock.patch.object(heuristics, "HEURISTICS", [*heuristics.HEURISTICS, Trial()]),
                    mock.patch("sys.stdout", io.StringIO()),
                    # What the PR-local inputs would give without td_replay's patches.
                    mock.patch(td_replay.H + "previously_failed_in_pr.read_additional_test_failures_file", lambda: {"test_leak"}),
                    mock.patch(td_replay.H + "llm.LLM.get_mappings", lambda self: {"test_leak": 1.0}),
                    mock.patch(td_replay.H + "mentioned_in_pr.get_pr_number", lambda: 1),
                    mock.patch(td_replay.H + "mentioned_in_pr.get_issue_or_pr_body", lambda n: "test_leak"),
                ):  # fmt: skip
                    td_replay.main()

            with self.assertRaisesRegex(SystemExit, r"a heuristic raised on 1 samples, left for a rerun: \['b'\]"):  # fmt: skip
                replay()
            n, root = len(td_replay.CHANGED_FILES_READERS), stats.absolute().parent
            expected = [({root}, ({"test_c"}, set()), "t\n", [None], {}, [["torch/foo.py"]] * n), ({root}, (set(), set()), "u\nv #12", [None, ""], {}, [["torch/bar.py", "docs/b.md"]] * n)]  # fmt: skip
            self.assertEqual(seen, expected)
            self.assertFalse((data / "td/b.json").exists())
            td = json.loads((data / "td/a.json").read_text())
            self.assertEqual((td["scored"], td["top25"], td["top25_cfg"]), (["test_b", "test_a"], ["test_a", "test_b"], {"default": ["test_a", "test_b"]}))  # fmt: skip
            # What score checks a replay against is what td_replay records.
            with patch_ci_jobs({True: {("pull", "linux", "default")}, False: set()}):
                rows = score.score(data, ["a"], ["td"], [], True, data / "samples", data / "td", score.replay_stats(stats))  # fmt: skip
            self.assertEqual(len(rows), 1)
            # A current replay is not recomputed.
            (data / "ids.txt").write_text("a")
            replay("--ids", str(data / "ids.txt"))
            self.assertEqual(len(seen), 2)
            err = io.StringIO()
            with self.assertRaises(SystemExit), mock.patch("sys.stderr", err):
                replay("--shard", "2", "2")
            self.assertIn("--shard I N needs 0 <= I < N", err.getvalue())
            with self.assertRaisesRegex(SystemExit, "must be a directory named"):
                replay("--stats", d)
            empty = Path(d) / "empty" / stats.name
            empty.mkdir(parents=True)
            with self.assertRaisesRegex(SystemExit, "lacks"):
                replay("--stats", str(empty))

    def test_td_replay_patches_every_changed_files_reader(self) -> None:
        heuristics = REPO_ROOT / "tools/testing/target_determination/heuristics"
        readers = {p.stem for p in heuristics.glob("*.py") if "query_changed_files" in p.read_text()}  # fmt: skip
        self.assertEqual(readers, set(td_replay.CHANGED_FILES_READERS))

    def test_score_no_workflow_truth(self) -> None:
        sample = {**SAMPLE, "truth_jobs": {"test_a": ["linux / test (default, 1, 2, linux.2xlarge)"]}}  # fmt: skip
        files = {
            "test-times.json": {"linux": {"default": {"test_a": 10.0}}},
            "samples/a.json": sample,
            "td/a.json": {"scored": [], "top25": [], "top25_cfg": {}},
            "results/m/a.json": {"out": answer(), "error": None, "cost_usd": 0.1},
        }
        with (
            tempfile.TemporaryDirectory() as d,
            patch_ci_jobs({True: {("pull", "linux", "default")}, False: set()}),
        ):
            data = write_data(d, files)
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                rows = score.score(data, ["a"], ["m"], ["model"], True, data / "samples", data / "td", "")  # fmt: skip
        self.assertEqual([(x["id"], x["pairs"]) for x in rows], [("a", 1)])
        self.assertIn("scored 1 of 1 ids with truth jobs that record no workflow", err.getvalue())  # fmt: skip

    def test_score_hedge_over_the_fallback_job(self) -> None:
        # test_d failed in rare, which no job records, so only the fallback job's rare times let final+td hedge it.
        jobs = {"test_a": [["pull", "linux / test (default, 1, 2, linux.2xlarge)"]], "test_d": [["pull", "linux / test (rare, 1, 1, linux.2xlarge)"]]}  # fmt: skip
        sample = {**SAMPLE, "truth": ["test_a", "test_d"], "truth_strong": ["test_a", "test_d"], "truth_jobs": jobs}  # fmt: skip
        times = {
            "linux": {"default": {"test_a": 100.0, "test_b": 100.0, "test_c": 100.0}, "slow": {"test_a": 50.0, "test_b": 50.0}},
            "linux-cuda": {"default": {"test_a": 40.0}},
            "default": {"rare": {"test_c": 90.0, "test_d": 10.0}},
        }  # fmt: skip
        files = {
            "test-times.json": times,
            "samples/a.json": sample,
            "td/a.json": {"scored": ["test_d", "test_b"], "top25": [], "top25_cfg": {"default": ["test_b"]}},
            "results/m/a.json": {"out": answer(tests=[{"file": "test_a"}]), "error": None, "cost_usd": 0.1},
        }  # fmt: skip
        # slow runs no TD, so its seconds are not priced.
        ci = {True: {("pull", "linux", "default"), ("pull", "linux", "rare")}, False: {("pull", "linux", "slow")}}  # fmt: skip
        with (
            tempfile.TemporaryDirectory() as d,
            patch_ci_jobs(ci),
            mock.patch.object(score, "grep_at", return_value=[]),
        ):
            data = write_data(d, files)
            rows = score.score(data, ["a"], ["m"], ["final+td"], True, data / "samples", data / "td", "")  # fmt: skip
        fields = ("policy", "pairs", "hit", "missed", "cpu", "gpu")
        self.assertEqual([tuple(x[k] for k in fields) for x in rows], [("final+td", 2, 2, [], 100 / 300, 1.0)])  # fmt: skip

    def test_score_main(self) -> None:
        row = {"tag": "m", "policy": "final+td", "kind": "pr", "pairs": 1, "hit": 1, "cpu": 0.5, "gpu": 0.0, "run_all": False, "files": 3, "cost_usd": 0.1, "error": False}  # fmt: skip
        with tempfile.TemporaryDirectory() as d:
            data, stats = Path(d), Path(d) / "stats"
            stats.mkdir()
            (data / "ids.txt").write_text("a b")

            def main(*argv: str) -> tuple[mock.Mock, str]:
                with (
                    mock.patch.object(sys, "argv", ["score", "--data", d, "--ids", str(data / "ids.txt"), "--stats", str(stats), *argv]),
                    mock.patch.object(score, "score", return_value=[row]) as scored,
                    mock.patch.object(score, "replay_stats", return_value="s"),
                    mock.patch("sys.stdout", io.StringIO()) as out,
                    mock.patch("sys.stderr", io.StringIO()) as err,
                ):  # fmt: skip
                    try:
                        score.main()
                    except SystemExit:
                        return scored, err.getvalue()
                return scored, out.getvalue()

            self.assertRegex(main("m")[1], "lacks")
            for a in score.STATS_ARTIFACTS:
                (stats / a).write_text("{}")
            for bad in ("td25", "final+tdx"):
                scored, err = main("--policies", f"model,{bad}", "m")
                self.assertIn(f"unknown --policies ['{bad}']", err)
                scored.assert_not_called()
            for dup in ("final+td,final+td20", "final+td5,final+td5.0"):
                scored, err = main("--policies", dup, "m")
                self.assertIn("two hedge policies of the same share", err)
                scored.assert_not_called()
            scored, _ = main("--policies", "model,final+td5,model,final+td5", "m")
            self.assertEqual(scored.call_args.args[3], ["model", "final+td5"])
            scored, out = main("m", "td")
            self.assertEqual(scored.call_args.args, (data, ["a", "b"], ["m", "td"], ["model", "final", "final-hm", "final+td"], True, data / "samples", data / "td", "s"))  # fmt: skip
            self.assertRegex(out, r"^m +final\+td +pr +n=  1 fail=  1 caught=100.0%")
            scored, _ = main("--all-truth", "--td-dir", d, "--policies", "final+td5", "--json", str(data / "rows.json"), "m")  # fmt: skip
            self.assertEqual(scored.call_args.args[3:7], (["final+td5"], False, data / "samples", data))  # fmt: skip
            self.assertEqual(json.loads((data / "rows.json").read_text()), [row])

    def test_ci_jobs_of_the_real_workflows(self) -> None:
        score.runs_td.cache_clear()
        self.assertTrue(score.runs_td("default"))
        self.assertIn("pull", {w for w, _, c in ci_jobs()[True] if c == "default"})
        self.assertEqual(score.MATRIX_CONFIG.findall("""{ config: "a" }, { config: 'b' }, {config: c }\n# No `d` config: trunk.yml runs it"""), ["a", "b", "c"])  # fmt: skip

    def test_td25_per_config(self) -> None:
        cands = ["test_a", "test_b", "test_c", "test_d", "test_e"]
        scores = {
            TestRun("test_e"): 3.0,
            TestRun("test_a", included=["TestX"]): 2.0,
            TestRun("test_c"): 1.0,
        }
        tp = TestPrioritizations(cands, scores)
        self.assertEqual(
            {t.test_file for t in tp.get_top_per_tests(25)[0]}, {"test_e", "test_a"}
        )
        # The cut is 25% + 1 of each config's own runs plus the files with no
        # recorded time (test_e, as if newly added); test_a's partial run counts.
        s = {"candidates": cands, "truth": []}
        cut = top25_per_config(tp, score.job_files(s, SECONDS))
        self.assertEqual(cut, {"default": ["test_a", "test_e"], "slow": ["test_a", "test_e"]})  # fmt: skip
        timed = {**s, "candidates": ["test_a", "test_b", "test_c", "test_d"]}
        self.assertEqual(top25_per_config(tp, score.job_files(timed, SECONDS)), {"default": ["test_a", "test_c"], "slow": ["test_a"]})  # fmt: skip
        self.assertEqual(tp.get_top_per_tests(100)[0][0], TestRun("test_e"))
        # A file that failed in slow was in slow's job, though slow records no time of it.
        failed = {**timed, "truth": ["test_d"], "truth_jobs": {"test_d": [["pull", "linux / test (slow, 1, 1)"]]}}  # fmt: skip
        self.assertEqual(score.job_files(failed, SECONDS)["slow"], {"test_a", "test_c", "test_d"})  # fmt: skip
        tp_d = TestPrioritizations(timed["candidates"], {TestRun("test_d"): 1.0})
        self.assertNotIn("test_d", top25_per_config(tp_d, score.job_files(timed, SECONDS))["slow"])  # fmt: skip
        self.assertEqual(top25_per_config(tp_d, score.job_files(failed, SECONDS))["slow"], ["test_d"])  # fmt: skip
        td = {"top25": ["test_a", "test_e"], "top25_cfg": cut}
        choice = pick("td25", None, td)
        self.assertTrue(choice.covers("test_a", "slow"))
        self.assertTrue(choice.covers("test_e", "slow"))
        self.assertFalse(choice.covers("test_c", "slow"))
        self.assertTrue(choice.covers("test_e", "not_in_test_times"))
        self.assertEqual(choice.files(), {"test_a", "test_e"})
        # TD runs only its cut, even of the files a selection always covers.
        ind = pick("td25", None, {"top25": [], "top25_cfg": {"inductor": ["test_a"]}})
        self.assertEqual([ind.covers("test_a", "inductor"), ind.covers("test_torch", "inductor")], [True, False])  # fmt: skip

    def test_cost_fraction(self) -> None:
        cands = {"test_a", "test_b", "test_c"}
        choice = Choice(Selection({"test_b"}), {"slow": {"test_a"}})
        self.assertEqual(cost_fraction(choice, cands, SECONDS), (50 / 160, 250 / 250))
        only_b = Choice(Selection({"test_b"}))
        self.assertEqual(cost_fraction(only_b, cands, SECONDS), (50 / 160, 50 / 250))
        self.assertEqual(
            cost_fraction(only_b, cands, SECONDS, lambda c: c == "default"),
            (50 / 160, 50 / 50),
        )
        self.assertEqual(
            cost_fraction(Choice(Selection(run_all=True)), cands, SECONDS), (1.0, 1.0)
        )


@contextlib.contextmanager
def scratch_repo() -> Iterator[None]:
    """A new git repo as bd.REPO and select_tests.REPO_ROOT, isolated from the caller's git environment and config."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull, GIT_CONFIG_NOSYSTEM="1", GIT_AUTHOR_NAME="a", GIT_AUTHOR_EMAIL="a@a", GIT_COMMITTER_NAME="a", GIT_COMMITTER_EMAIL="a@a")  # fmt: skip
    with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, env, clear=True), mock.patch.object(bd, "REPO", Path(d)), mock.patch.object(select_tests, "REPO_ROOT", Path(d)):  # fmt: skip
        bd.git("init", "-q")
        caches = (bd.is_ancestor, bd.own_change, bd.below)
        for c in caches:
            c.cache_clear()
        try:
            yield
        finally:
            for c in caches:
                c.cache_clear()


def commit_files(files: dict[str, str], *parents: str, links: dict[str, str] | None = None) -> str:  # fmt: skip
    """In a scratch_repo, a commit of a tree holding `files` and submodule commits `links`, so commits of equal files have equal trees."""

    def piped(args: list[str], stdin: str) -> str:
        return subprocess.run(["git", *args], cwd=bd.REPO, input=stdin, capture_output=True, text=True, check=True).stdout.strip()  # fmt: skip

    entries = "".join(f"100644 blob {piped(['hash-object', '-w', '--stdin'], v)}\t{k}\n" for k, v in sorted(files.items()))  # fmt: skip
    entries += "".join(f"160000 commit {v}\t{k}\n" for k, v in sorted((links or {}).items()))  # fmt: skip
    return bd.git("commit-tree", piped(["mktree"], entries), *(a for p in parents for a in ("-p", p)), "-m", "c").strip()  # fmt: skip


class TestClickhouse(unittest.TestCase):
    def test_query(self) -> None:
        ok = subprocess.CompletedProcess([], 0, 'log\n{"columns": [{"name": "a"}], "rows": [[1]]}', "")  # fmt: skip
        busy = subprocess.CompletedProcess([], 1, "", "HTTP 503 Service Unavailable")
        crowded = subprocess.CompletedProcess([], 1, "", "Code: 202. DB::Exception: Too many simultaneous queries")  # fmt: skip
        slow = subprocess.CompletedProcess([], 1, "", "Code: 159. DB::Exception: Timeout exceeded: elapsed 500 seconds")  # fmt: skip
        gateway = subprocess.CompletedProcess([], 1, "", "502 Bad Gateway")
        token = subprocess.CompletedProcess([], 22, "", "curl: (22) The requested URL returned error: 503")  # fmt: skip
        bad = subprocess.CompletedProcess([], 2, "", "Code: 62. Syntax error")
        # gcx's Go network errors, and the wrapper's re-login when its token check hit one.
        dial = subprocess.CompletedProcess([], 1, "", 'Error: Post "/api/ds/query": dial tcp 10.0.0.1:443: i/o timeout')  # fmt: skip
        eof = subprocess.CompletedProcess([], 1, "", 'Error: Post "/api/ds/query": unexpected EOF')  # fmt: skip
        tls = subprocess.CompletedProcess([], 1, "", "Error: net/http: TLS handshake timeout")  # fmt: skip
        login = subprocess.CompletedProcess([], 1, "", "error: gcx login failed for pytorchci")  # fmt: skip
        # A result that is not one, whose rows would look transient if searched.
        null = subprocess.CompletedProcess([], 0, '{"columns": null, "rows": [["connection timed out"]]}', "")  # fmt: skip
        hung = subprocess.TimeoutExpired("gcx", clickhouse.TIMEOUT)
        # fmt: off
        cases = [
            ([ok], 1, [{"a": 1}]),
            ([busy, crowded, ok], 3, [{"a": 1}]),
            ([bad, ok], 1, "attempt 1 of 3 exited with code 2: Code: 62. Syntax error"),
            ([slow, ok], 1, "attempt 1 of 3 exited with code 1: Code: 159. DB::Exception: Timeout exceeded: elapsed 500 seconds"),
            ([gateway, ok], 2, [{"a": 1}]),
            ([token, ok], 2, [{"a": 1}]),
            ([dial, eof, ok], 3, [{"a": 1}]),
            ([tls, login, ok], 3, [{"a": 1}]),
            ([busy, hung, ok], 2, "attempt 2 of 3 timed out after 600s"),
            ([null, ok], 1, 'attempt 1 of 3 returned no result: {"columns": null, "rows": [["connection timed out"]]}'),
            ([busy, busy, busy], 3, "attempt 3 of 3 exited with code 1: HTTP 503 Service Unavailable"),
        ]
        # fmt: on
        for outcomes, calls, want in cases:
            with mock.patch.object(subprocess, "run", side_effect=outcomes) as run, mock.patch.object(clickhouse.time, "sleep"):  # fmt: skip
                if isinstance(want, list):
                    self.assertEqual(clickhouse.query("SELECT 1"), want)
                else:
                    with self.assertRaises(RuntimeError) as e:
                        clickhouse.query("SELECT 1")
                    self.assertEqual(str(e.exception), f"ClickHouse query failed, {want}\nSQL: SELECT 1")  # fmt: skip
                self.assertEqual(run.call_count, calls)
                self.assertEqual(run.call_args.kwargs["timeout"], clickhouse.TIMEOUT)
                self.assertEqual(run.call_args.args[0][1:], ["datasources", "clickhouse", "query", "-d", clickhouse.CLICKHOUSE_UID, "--limit", "0", "-o", "json", "SELECT 1"])  # fmt: skip


class TestBuildDataset(unittest.TestCase):
    def test_strip_trailers(self) -> None:
        msg = "Body line\n\nCo-authored-by: x\nPull Request resolved: https://x\nApproved by: https://y\nghstack-source-id: 1\nKeep: me\n"  # @lint-ignore
        self.assertEqual(strip_trailers(msg), "Body line\n\nKeep: me")

    def test_main_refuses_open_window(self) -> None:
        argv = ["build_dataset", "--data", "d", "--start", "2026-10-01", "--end", str(dt.date.today()), "--kinds", "head"]  # fmt: skip
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.object(sys, "argv", [*argv[:2], tmp, *argv[3:]]),
            mock.patch.object(bd, "git") as git,
            mock.patch.object(bd.History, "load") as load,
            contextlib.redirect_stderr(io.StringIO()) as err,
            self.assertRaises(SystemExit),
        ):  # fmt: skip
            bd.main()
        git.assert_not_called()
        load.assert_not_called()
        self.assertIn("rerun on or after", err.getvalue())
        # Head samples extend pr_ samples, so a DIR without any builds none, without loading CI history, and goes on to the later kinds.
        closed = [*argv[:6], "2026-10-02", *argv[7:], "revert"]
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.object(sys, "argv", [*closed[:2], tmp, *closed[3:]]),
            mock.patch.object(bd, "git") as git,
            mock.patch.object(bd.History, "load") as load,
            mock.patch.object(bd, "revert_samples") as revert,
            contextlib.redirect_stdout(io.StringIO()) as out,
        ):  # fmt: skip
            bd.main()
        load.assert_not_called()
        revert.assert_called_once()
        self.assertIn(f"no head samples: there are no pr_ samples in {tmp}/samples", out.getvalue())  # fmt: skip
        # The guard is for PR truth only.
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.object(sys, "argv", [*argv[:2], tmp, *argv[3:-1], "revert"]),
            mock.patch.object(bd, "git"),
            mock.patch.object(bd, "revert_samples") as revert,
        ):  # fmt: skip
            bd.main()
        revert.assert_called_once()

    def test_commit_text(self) -> None:
        # As trymerge.py's gen_commit_message lands a ghstack PR and native_stack.py a GitHub-native one.
        msg = "Fix (#12) the thing (#203)\n\nBody\n\nPull Request resolved: https://github.com/pytorch/pytorch/pull/203\nApproved by: https://github.com/y\nghstack dependencies: #201, #202\nStack dependencies: #200\n\nCo-authored-by: z <z@z>\n"  # @lint-ignore
        with scratch_repo():
            bd.git("commit", "-q", "--allow-empty", "-m", msg)
            self.assertEqual(bd.commit_text("HEAD"), ("Fix (#12) the thing", "Body"))
            bd.git("commit", "-q", "--allow-empty", "-m", "No PR\n\nKeep: me")
            self.assertEqual(bd.commit_text("HEAD"), ("No PR", "Keep: me"))

    def test_history_load(self) -> None:
        rows = [
            fail("test_x", "2026-10-05 00:00:00", "a"),
            fail("test_x", "2026-10-05 00:00:00", "m", br="main", flaky=True),
            fail("test_x", "2026-10-05 00:00:00", "t", br="trunk/t"),
            # A fork's main is a PR branch, not main; here PR 2's, whose author is unknown.
            fail("test_x", "2026-10-05 00:00:00", "g", br="f/pytorch:main"),
            fail("test_x", "2026-10-05 00:00:00", "u"),
        ]
        # A cache may hold the flag as a string.
        rows[0]["flaky"], rows[1]["flaky"] = "0", "1"
        landed = {1: day(7)}
        with (
            mock.patch.object(bd, "pull_failures", return_value=rows) as pull,
            mock.patch.object(bd, "pr_of_shas", return_value={"a": 1, "g": 2}) as pr_of_shas,
            mock.patch.object(bd, "pr_authors", return_value={1: "me"}),
            mock.patch.object(bd, "landed_times", return_value=landed) as landed_times,
            mock.patch.object(bd, "query", return_value=[]) as query,
        ):  # fmt: skip
            h = bd.History.load(START, END)
        pull.assert_called_once_with(dt.date(2026, 9, 1), dt.date(2026, 11, 9))
        landed_times.assert_called_once_with(dt.date(2026, 9, 1))
        pr_of_shas.assert_called_once_with(["a", "g", "u"])
        self.assertIn("day >= '2026-09-01' AND day < '2026-11-09'", query.call_args.args[0])  # fmt: skip
        self.assertEqual({sha: [r["pr"] for r in rs] for sha, rs in h.heads.items()}, {"a": [1], "g": [2]})  # fmt: skip
        k = ("test_a", "TestA", "test_x")
        self.assertEqual(len(h.main[k]), 2)
        self.assertEqual(sorted((who, fl) for _, who, _, _, fl in h.seen[k]), [("2", False), ("main", False), ("main", False), ("main", True), ("me", False)])  # fmt: skip
        self.assertIs(h.heads["a"][0]["flaky"], False)
        self.assertEqual(h.landed, landed)

    def test_why_dropped(self) -> None:
        t = "2026-10-05 00:00:00"
        mine = ["test_real", "test_main", "test_main_after", "test_other_in", "test_other_out", "test_unmapped", "test_fork_main",
                "test_same_author", "test_own_head", "test_disabled", "test_disabled_out", "test_base_in", "test_base_out",
                "test_main_flaky", "test_other_flaky", "test_base_flaky", "test_disabled_before", "test_disabled_cls", "test_other_cls", "test_disabled_after"]  # fmt: skip
        rows = [fail(name, t, "s") for name in mine] + [
            fail("test_green", t, "s", job="pull / test (default, 2, 2)"),
            fail("test_unstable", t, "s", job="pull / test (default, 1, 1, unstable)"),
            fail("test_holdout", t, "s", wf="td-holdout-unstable"),
            fail("test_periodic", t, "s", wf="periodic"),
            fail("test_rx", t, "s", wf="rocm-rx7900"),
            fail("test_late", "2026-10-26 00:00:00", "s"),
            fail("test_early", "2026-09-16 23:59:59", "s"),
            fail("test_early_in", "2026-09-17 00:00:00", "s"),
            fail("test_after_landing", "2026-10-08 00:00:00", "s"),
            # Passed again in its job; it still counts against the other PRs.
            fail("test_own_flaky", t, "s", flaky=True),
            fail("test_own_flaky", "2026-10-06 00:00:00", "o"),
            # Main's and another PR's failures count even when they passed again in their job.
            fail("test_main_flaky", "2026-10-06 00:00:00", "m", br="main", flaky=True),
            fail("test_other_flaky", "2026-10-06 00:00:00", "o", flaky=True),
            fail("test_base_flaky", "2026-09-05 00:00:00", "m", br="main", flaky=True),
            fail("test_main", "2026-10-07 00:00:00", "m", br="main"),
            # PR 1 lands 10-07 12:00; its own breakage on main afterwards is not held against it.
            fail("test_main_after", "2026-10-08 00:00:00", "m", br="main"),
            # WINDOW is 14 days, exclusive.
            fail("test_other_in", "2026-09-21 00:00:01", "o"),
            fail("test_other_out", "2026-09-21 00:00:00", "o"),
            fail("test_unmapped", "2026-10-06 00:00:00", "u"),
            fail("test_fork_main", "2026-10-06 00:00:00", "g", br="f/pytorch:main"),
            # A same-author sibling is only checked as a PR below this one (see test_stacked).
            fail("test_same_author", "2026-10-06 00:00:00", "c"),
            fail("test_own_head", "2026-10-06 00:00:00", "s2"),
            fail("test_base_in", "2026-09-05 00:00:00", "m", br="main"),
            fail("test_base_out", "2026-09-01 00:00:00", "m", br="main"),
            # Main broke it 17 days after the merge base and 15 after this failure, before PR 1 landed.
            fail("test_base_later", "2026-09-17 00:00:00", "s"),
            fail("test_base_later", "2026-10-02 00:00:00", "m", br="main"),
            # Main broke it 2 days after the merge base and 15 before this failure, before PR 1 landed.
            fail("test_base_after", "2026-10-02 00:00:00", "s"),
            fail("test_base_after", "2026-09-17 00:00:00", "m", br="main"),
            # Another author's failure 15 days after this one, still before PR 1 landed.
            fail("test_other_later", "2026-09-18 00:00:00", "s"),
            fail("test_other_later", "2026-10-03 00:00:00", "o"),
            # The same test name in another class is another test.
            fail("test_other_cls", "2026-10-06 00:00:00", "o", cls="TestB"),
        ]  # fmt: skip
        disabled = (
            ("2026-09-22", "test_disabled"),
            ("2026-10-19", "test_disabled_out"),
            ("2026-09-20", "test_disabled_before"),
            ("2026-10-05", "test_disabled_cls", "TestB"),
            ("2026-10-18", "test_disabled_after"),
        )
        h = history(rows, {"s": 1, "s2": 1, "o": 2, "c": 3}, {1: "me", 2: "x", 3: "me"}, {1: dt.datetime(2026, 10, 7, 12, tzinfo=dt.timezone.utc)}, disabled)  # fmt: skip
        with (
            patch_ci_jobs(TD_JOBS),
            mock.patch.object(bd, "green_jobs", return_value={("s", "pull", "pull / test (default, 2, 2)")}) as green,
        ):  # fmt: skip
            h.load_green({"s", "o"})
            h.load_green({"s"})
            got = {r["test"]: h.why_dropped(r) for r in h.heads["s"]}
            # The merge base with main at 09-15: main broke test_base_in 10 days before it.
            near_base = {r["test"]: h.why_dropped(r, base_t=day(15, 9)) for r in h.heads["s"] if r["test"] in ("test_base_in", "test_base_out", "test_base_flaky", "test_base_later", "test_base_after")}  # fmt: skip
            other = {r["test"]: h.why_dropped(r) for r in h.heads["o"]}
            # Main breaks test_main_after near this merge base, but only after PR 1 landed.
            after = h.why_dropped(next(r for r in h.heads["s"] if r["test"] == "test_main_after"), base_t=day(4))  # fmt: skip
        # Each head is queried once.
        self.assertEqual([c.args[0] for c in green.call_args_list], [["o", "s"], []])
        self.assertEqual(got, {
            "test_real": None, "test_main": "other_failure", "test_main_after": None, "test_other_in": "other_failure", "test_other_out": None,
            "test_unmapped": "other_failure", "test_fork_main": "other_failure", "test_same_author": None, "test_own_head": None,
            "test_disabled": "disabled", "test_disabled_out": None, "test_base_in": None, "test_base_out": None, "test_green": "job_green",
            "test_unstable": "job", "test_holdout": "job", "test_periodic": "job", "test_rx": "job", "test_late": "window",
            "test_early": "window", "test_early_in": None, "test_after_landing": "after_landing", "test_own_flaky": "passed_in_job",
            "test_main_flaky": "other_failure", "test_other_flaky": "other_failure", "test_base_flaky": None,
            "test_disabled_before": None, "test_disabled_cls": None, "test_other_cls": None, "test_other_later": None,
            "test_disabled_after": "disabled", "test_base_later": None, "test_base_after": None,
        })  # fmt: skip
        self.assertEqual(near_base, {"test_base_in": "main_near_base", "test_base_out": None, "test_base_flaky": "main_near_base", "test_base_later": None, "test_base_after": "main_near_base"})  # fmt: skip
        self.assertEqual(other["test_own_flaky"], "other_failure")
        self.assertIsNone(after)

    def test_stacked(self) -> None:
        # A ghstack stack, built as ghstack builds it: L (pr 1) below U (pr 2), both pushed twice. Each
        # PR's head has its previous head and a synthetic gh/USER/N/base commit with the tree of
        # the PR below as parents, so L's heads are neither U's CI base nor its ancestors.
        # N (pr 4) is a GitHub-native PR on top of L's second head, W (pr 3) another PR of the author's.
        with scratch_repo():

            def commit(content: str, *parents: str) -> str:
                return commit_files({"f": content}, *parents)

            m = commit("main")
            bd.git("update-ref", "refs/remotes/origin/main", m)
            l_base = commit("main", m)
            l1 = commit("L1", l_base)
            l2 = commit("L2", l1, l_base)
            u_base1 = commit("L1", m)
            u1 = commit("U1", u_base1)
            u_base2 = commit("L2", u_base1)
            u2 = commit("U2", u1, u_base2)
            n = commit("N", l2)
            w = commit("W", m)
            # L broke test_x in both versions and test_q only in its second; main broke test_v long ago on m.
            rows = [
                fail("test_x", "2026-10-05 00:00:00", l1),
                fail("test_x", "2026-10-06 00:00:00", l2),
                fail("test_q", "2026-10-06 00:00:00", l2),
                fail("test_z", "2026-10-05 00:00:00", w),
                fail("test_v", "2026-09-20 00:00:00", m, br="main"),
            ] + [fail(t, "2026-10-07 00:00:00", u) for u in (u1, u2, n) for t in ("test_x", "test_q", "test_z", "test_v", "test_y")]  # fmt: skip
            h = history(rows, {l1: 1, l2: 1, u1: 2, u2: 2, w: 3, n: 4}, {1: "me", 2: "me", 3: "me", 4: "me"})  # fmt: skip

            def truth(sha: str, base: str) -> tuple[list[str], dict]:
                dropped: collections.Counter = collections.Counter()
                mb = bd.git("merge-base", base, sha).strip()
                kept = h.truth(sha, {"base": base, "merge_base": mb, "main_base": m, "base_source": "ci"}, dropped)  # fmt: skip
                return sorted(r["test"] for r in kept or []), dict(dropped)

            with (
                patch_ci_jobs(TD_JOBS),
                mock.patch.object(bd, "green_jobs", return_value=set()),
                mock.patch.object(bd, "commit_time", return_value=day(20)),
            ):  # fmt: skip
                h.load_green(set(h.heads))
                # Each U head drops what the L version below it broke; its own failures, W's and main's stay.
                self.assertEqual(truth(u1, u_base1), (["test_q", "test_v", "test_y", "test_z"], {"stacked": 1}))  # fmt: skip
                self.assertEqual(truth(u2, u_base2), (["test_v", "test_y", "test_z"], {"stacked": 2}))  # fmt: skip
                # CI recorded only U's later base: diffed against the merge base, the first L version is below.
                self.assertEqual(truth(u1, u_base2), (["test_q", "test_v", "test_y", "test_z"], {"stacked": 1}))  # fmt: skip
                # Diffed against L's head, the native PR finds L below it by ancestry; diffed against
                # main, its diff contains L's change, so L's failures are its own too.
                self.assertEqual(truth(n, l2), (["test_v", "test_y", "test_z"], {"stacked": 2}))  # fmt: skip
                self.assertEqual(truth(n, m), (["test_q", "test_v", "test_x", "test_y", "test_z"], {}))  # fmt: skip
                self.assertEqual(truth(l2, l_base), (["test_q", "test_x"], {}))
            self.assertEqual(h.other_heads({u1}), {l1, l2, w, n})

    def test_stacked_down_the_stack(self) -> None:
        # A ghstack stack L0 (pr 1) < L (pr 2) < U (pr 3): U's base has L's tree, and L's change edits a
        # line two away from L0's, so L0's change reverse-applies to U's base with one line of context
        # but not with two.
        lines = [str(i) for i in range(1, 10)]

        def a(**edits: str) -> str:
            return "".join(edits.get(f"l{i}", x) + "\n" for i, x in enumerate(lines, 1))  # fmt: skip

        with scratch_repo():
            m = commit_files({"a": a(), "b": "b\n"})
            bd.git("update-ref", "refs/remotes/origin/main", m)
            main, k0_files, k1_files = ({"a": a(**e), "b": "b\n"} for e in ({}, {"l2": "x"}, {"l2": "x", "l4": "y"}))  # fmt: skip
            k0 = commit_files(k0_files, commit_files(main, m))
            k1 = commit_files(k1_files, commit_files(k0_files, m))
            k2_base = commit_files(k1_files, m)
            k2 = commit_files({**k1_files, "b": "u\n"}, k2_base)
            # P (pr 4) changes L0's line differently, so its change is not in U's base. Q (pr 5) sets
            # line 5 to what U's base has on line 4, so its change reverse-applies only with no context.
            p = commit_files({"a": a(l2="p"), "b": "b\n"}, m)
            q = commit_files({"a": a(l5="y"), "b": "b\n"}, m)
            rows = [fail("test_x", "2026-10-05 00:00:00", k0), fail("test_w", "2026-10-05 00:00:00", k1), fail("test_y", "2026-10-05 00:00:00", p), fail("test_v", "2026-10-05 00:00:00", q)]  # fmt: skip
            rows += [fail(t, "2026-10-07 00:00:00", k2) for t in ("test_x", "test_w", "test_y", "test_v", "test_z")]  # fmt: skip
            h = history(rows, {k0: 1, k1: 2, k2: 3, p: 4, q: 5}, {1: "me", 2: "me", 3: "me", 4: "me", 5: "me"})  # fmt: skip
            dropped: collections.Counter = collections.Counter()
            with (
                patch_ci_jobs(TD_JOBS),
                mock.patch.object(bd, "green_jobs", return_value=set()),
                mock.patch.object(bd, "commit_time", return_value=day(20)),
            ):  # fmt: skip
                h.load_green(set(h.heads))
                kept = h.truth(k2, {"base": k2_base, "merge_base": k2_base, "main_base": m, "base_source": "ci"}, dropped)  # fmt: skip
        self.assertEqual(sorted(r["test"] for r in kept or []), ["test_v", "test_y", "test_z"])  # fmt: skip
        self.assertEqual(dropped, {"stacked": 2})

    def test_below_ignores_diff_config(self) -> None:
        # B has O's change but is not its descendant, so only the reverse-applied patch finds it.
        # O's two hunks are five lines apart, and it bumps a submodule.
        lines = [f"{i}\n" if i != 9 else "  9\n" for i in range(20)]
        changed = [*lines[:3], "o\n", *lines[4:15], "o\n", *lines[16:]]
        with scratch_repo():
            m = commit_files({"a": "".join(lines)}, links={"s": "1" * 40})
            bd.git("update-ref", "refs/remotes/origin/main", m)
            o = commit_files({"a": "".join(changed)}, m, links={"s": "2" * 40})
            b = commit_files({"a": "".join(changed), "z": "z\n"}, m, links={"s": "2" * 40})  # fmt: skip
            # R only re-indents line 9, which m lacks; ignoring whitespace would find it in m.
            r = commit_files({"a": "".join(lines[:9] + ["    9\n"] + lines[10:])}, m, links={"s": "1" * 40})  # fmt: skip
            want = bd.own_change(o)
            self.assertEqual(want.count(b"\n@@ "), 3)
            self.assertIn(b"+Subproject commit " + b"2" * 40, want)
            plain = (("diff.noprefix", "true"), ("diff.external", "false"), ("color.diff", "always"), ("apply.whitespace", "error"))  # fmt: skip
            cases = [(plain, {}), ((("diff.context", "0"),), {}), ((("diff.context", "1"),), {}), ((("diff.context", "5"), ("diff.interHunkContext", "10")), {}),
                     ((("diff.submodule", "log"),), {}), ((("diff.submodule", "diff"),), {}), ((), {"GIT_DIFF_OPTS": "--unified=0"}),
                     ((("apply.ignoreWhitespace", "change"),), {})]  # fmt: skip
            for config, env in cases:
                with self.subTest(config=config, env=env), mock.patch.dict(os.environ, env):  # fmt: skip
                    for k, v in config:
                        bd.git("config", k, v)
                    for c in (bd.own_change, bd.below):
                        c.cache_clear()
                    try:
                        self.assertEqual(bd.own_change(o), want)
                        self.assertTrue(bd.below(o, b))
                        self.assertFalse(bd.below(o, m))
                        self.assertFalse(bd.below(r, m))
                    finally:
                        for k, _ in config:
                            bd.git("config", "--unset", k)

    def test_below_needs_the_lower_main_base(self) -> None:
        # Main added g after U's merge base m0 and D, based on m1, deletes it: D's change reverse-applies
        # to m0 and has no deletion hunk of a file that stays, but m0 predates g rather than having D's change.
        with scratch_repo():
            m0 = commit_files({"a": "a\n"})
            m1 = commit_files({"a": "a\n", "g": "g\n"}, m0)
            bd.git("update-ref", "refs/remotes/origin/main", m1)
            d = commit_files({"a": "a\n"}, m1)
            b = commit_files({"a": "a\n", "z": "z\n"}, m1)
            self.assertFalse(bd.below(d, m0))
            # Based on m1, which has g, U's base lacking it does have D's change.
            self.assertTrue(bd.below(d, b))

    def test_stacked_bounds(self) -> None:
        # U (pr 1, "me") is based on B, which has the changes of O (pr 2, another author's, landed),
        # S (pr 3, mine) and F (pr 4, mine). O and S broke their tests 20 days before U's failure,
        # outside WINDOW; F's failure passed again in its job. D (pr 5, mine) deletes a line whose
        # context is two blank lines, which B has elsewhere, so D's change reverse-applies to B
        # with one line of context though B does not have it. P (pr 6) and Q (pr 7), another author's,
        # failed within WINDOW but after U landed, which other_failure leaves out; B has P's change only.
        # R (pr 8), another author's in B, failed 15 days after U.
        main = {"a": "a\n\nX\n\nb\n\n\nc\n", "o": "o\n", "s": "s\n", "f": "f\n", "p": "p\n", "q": "q\n", "r": "r\n"}  # fmt: skip
        with scratch_repo():
            m = commit_files(main)
            bd.git("update-ref", "refs/remotes/origin/main", m)
            changed = dict.fromkeys("osfpr", "changed\n")
            o, s, f, p, r = (commit_files({**main, k: v}, m) for k, v in changed.items())  # fmt: skip
            q = commit_files({**main, "q": "changed\n"}, m)
            d = commit_files({**main, "a": "a\n\n\nb\n\n\nc\n"}, m)
            b = commit_files({**main, **changed}, m)
            u = commit_files({**main, **changed, "u": "u\n"}, b)
            rows = [
                fail("test_o", "2026-09-17 00:00:00", o),
                fail("test_s", "2026-09-17 00:00:00", s),
                fail("test_f", "2026-10-06 00:00:00", f, flaky=True),
                fail("test_d", "2026-10-06 00:00:00", d),
                fail("test_p", "2026-10-09 00:00:00", p),
                fail("test_q", "2026-10-09 00:00:00", q),
                fail("test_r", "2026-10-22 00:00:00", r),
            ] + [fail(t, "2026-10-07 00:00:00", u) for t in ("test_o", "test_s", "test_f", "test_d", "test_p", "test_q", "test_r")]  # fmt: skip
            authors = {1: "me", 2: "x", 3: "me", 4: "me", 5: "me", 6: "y", 7: "y", 8: "y"}  # fmt: skip
            h = history(rows, {u: 1, o: 2, s: 3, f: 4, d: 5, p: 6, q: 7, r: 8}, authors, {2: day(18, 9), 1: day(8)})  # fmt: skip
            dropped: collections.Counter = collections.Counter()
            with (
                patch_ci_jobs(TD_JOBS),
                mock.patch.object(bd, "green_jobs", return_value=set()),
                mock.patch.object(bd, "commit_time", return_value=day(20)),
            ):  # fmt: skip
                h.load_green(set(h.heads))
                kept = h.truth(u, {"base": b, "merge_base": b, "main_base": m, "base_source": "ci"}, dropped)  # fmt: skip
            self.assertTrue(all(bd.below(x, b) for x in (o, s, f, p, r)))
            self.assertFalse(any(bd.below(x, b) for x in (d, q)))
            self.assertEqual(len(bd.deletion_hunks(bd.own_change(d))), 1)
        self.assertEqual(sorted(r["test"] for r in kept or []), ["test_d", "test_f", "test_o", "test_q", "test_r"])  # fmt: skip
        self.assertEqual(dropped, {"stacked": 2})

    def test_stacked_lower_rows(self) -> None:
        # U (pr 1, "me") is based on B, which has the changes of the author's A, J, G and K (prs 2-5).
        # A landed on 10-01, before U's failure, so its earlier failure no longer counts. J's failure
        # is in a job that does not run TD, G's in a job that passed on G. K lands only after U failed.
        main: dict[str, str] = {k: k + "\n" for k in "ajgk"}
        green_job = "pull / test (default, 2, 2)"
        with scratch_repo():
            m = commit_files(main)
            bd.git("update-ref", "refs/remotes/origin/main", m)
            a, j, g, k = (commit_files({**main, x: "changed\n"}, m) for x in "ajgk")
            b = commit_files(dict.fromkeys(main, "changed\n"), m)
            u = commit_files({**dict.fromkeys(main, "changed\n"), "u": "u\n"}, b)
            rows = [
                fail("test_a", "2026-09-28 00:00:00", a),
                fail("test_j", "2026-10-06 00:00:00", j, wf="periodic"),
                fail("test_g", "2026-10-06 00:00:00", g, job=green_job),
                fail("test_k", "2026-10-06 00:00:00", k),
            ] + [fail(t, "2026-10-07 00:00:00", u) for t in ("test_a", "test_j", "test_g", "test_k")]  # fmt: skip
            h = history(rows, {u: 1, a: 2, j: 3, g: 4, k: 5}, dict.fromkeys(range(1, 6), "me"), {2: day(1), 5: day(10)})  # fmt: skip
            dropped: collections.Counter = collections.Counter()
            with (
                patch_ci_jobs(TD_JOBS),
                mock.patch.object(bd, "green_jobs", lambda shas: {(g, "pull", green_job)} & {(x, "pull", green_job) for x in shas}),
                mock.patch.object(bd, "commit_time", return_value=day(20)),
            ):  # fmt: skip
                bases = {"base": b, "merge_base": b, "main_base": m, "base_source": "ci"}  # fmt: skip
                h.load_green({u})
                with self.assertRaisesRegex(AssertionError, "load_green"):
                    h.truth(u, bases, dropped)
                h.load_green(set(h.heads))
                kept = h.truth(u, bases, dropped)
            self.assertTrue(all(bd.below(x, b) for x in (a, j, g, k)))
        self.assertEqual(sorted(r["test"] for r in kept or []), ["test_a", "test_g", "test_j"])  # fmt: skip
        self.assertEqual(dropped, {"stacked": 1})

    def test_truth_skips_a_main_base_before_the_loaded_failures(self) -> None:
        h = history([fail("test_x", "2026-10-05 00:00:00", "s")], {"s": 1}, {1: "me"})
        h.green["s"] = set()
        bases = {"base": "b", "merge_base": "b", "main_base": "b", "base_source": "ci"}
        # START - LOOKBACK + WINDOW is 2026-09-15.
        with patch_ci_jobs(TD_JOBS), contextlib.redirect_stdout(io.StringIO()) as out:
            for t, want in ((day(14, 9), None), (day(15, 9), ["test_x"])):
                with mock.patch.object(bd, "commit_time", return_value=t):
                    kept = h.truth("s", bases, collections.Counter())
                self.assertEqual(want, kept and [r["test"] for r in kept])
        self.assertEqual(out.getvalue(), "skipping s: merge base before the loaded failures\n")  # fmt: skip

    def test_pull_failures_marks_flaky_rows(self) -> None:
        rows = [{"job": JOB, "flaky": 1}, {"job": JOB, "flaky": 0}]
        with mock.patch.object(bd, "query", return_value=rows) as query, contextlib.redirect_stdout(io.StringIO()):  # fmt: skip
            got = bd.pull_failures(dt.date(2026, 10, 1), dt.date(2026, 10, 2))
        self.assertEqual(got, rows)
        sql = query.call_args.args[0]
        self.assertIn("(f.job_id, f.tf, f.classname, f.name) IN (SELECT * FROM passed) flaky", sql)  # fmt: skip
        # SKIP_JOBS are left out in ClickHouse, before the empty-chunk check.
        self.assertIn("repository_full_name = 'pytorch/pytorch'\n                AND name NOT LIKE '%rerun_disabled_tests%' AND name NOT LIKE '%mem_leak_check%')", sql)  # fmt: skip
        self.assertNotIn("NOT IN", sql)
        # The last two-day chunk stops at `end`.
        with mock.patch.object(bd, "query", return_value=rows[:1]) as query, contextlib.redirect_stdout(io.StringIO()):  # fmt: skip
            bd.pull_failures(dt.date(2026, 10, 1), dt.date(2026, 10, 4))
        self.assertEqual([re.findall(r"time_inserted >= '(\S+)' AND time_inserted < '(\S+)'", c.args[0]) for c in query.call_args_list], [[("2026-10-01", "2026-10-03")], [("2026-10-03", "2026-10-04")]])  # fmt: skip
        # Two days of CI with no failing test means the legacy store is no longer written.
        with mock.patch.object(bd, "query", side_effect=[rows[:1], []]), contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "from 2026-10-03 to 2026-10-04"):  # fmt: skip
            bd.pull_failures(dt.date(2026, 10, 1), dt.date(2026, 10, 4))

    def test_write_sample(self) -> None:
        with scratch_repo():
            bd.git("commit", "-q", "--allow-empty", "-m", "base")
            for f in ("gone.py", "moved.py"):
                (bd.REPO / f).write_text(f)
            bd.git("add", ".")
            bd.git("commit", "-q", "-m", "files")
            base = bd.git("rev-parse", "HEAD").strip()
            bd.git("rm", "-q", "gone.py")
            bd.git("mv", "moved.py", "new.py")
            bd.git("commit", "-q", "-m", "change")
            head = bd.git("rev-parse", "HEAD").strip()
            # deleted_files runs select_tests' git, which scratch_repo also points here.
            with (
                tempfile.TemporaryDirectory() as d,
                mock.patch.object(bd, "candidates_at", return_value=[]),
                mock.patch.object(bd, "changed_files", return_value=[]),
                mock.patch.object(bd, "diffstat_and_diff", return_value=("", "")),
            ):  # fmt: skip
                bd.write_sample(Path(d), "s", "neutral", head, base, "t", "b", window=(START, END), truth=[])  # fmt: skip
                s = json.loads((Path(d) / "s.json").read_text())
                # Written once: a rebuild keeps the sample.
                bd.write_sample(Path(d), "s", "neutral", head, base, "t2", "b", window=(START, END), truth=[])  # fmt: skip
                self.assertEqual(json.loads((Path(d) / "s.json").read_text()), s)
        self.assertEqual(s["deleted"], ["gone.py", "moved.py"])
        self.assertEqual((s["title"], s["builder"], s["window"], s["workflows"]), ("t", bd.BUILDER, [str(START), str(END)], bd.workflows()))  # fmt: skip
        # The mark follows the job classification.
        with patch_ci_jobs(TD_JOBS):
            self.assertNotEqual(bd.workflows(), s["workflows"])

    def test_main_refuses_another_build(self) -> None:
        neutral = [str(END - dt.timedelta(days=25)), str(END)]
        argv = ["build_dataset", "--data", "", "--start", str(START), "--end", str(END), "--kinds"]  # fmt: skip
        # Every kind refuses samples of its own of another builder or window, or of none, before any work, including the kinds before it.
        cases = [
            ("rev_x", {"window": [str(START), str(END)]}, "revert", "rev_ samples in {}, e.g. rev_x, predate the builder field; build into a fresh --data"),
            ("rev_x", {"builder": bd.BUILDER + 1, "window": [str(START), str(END)]}, "revert", f"rev_ samples in {{}}, e.g. rev_x, were written by builder {bd.BUILDER + 1}, not {bd.BUILDER}; build into a fresh --data"),
            ("neu_x", {"builder": bd.BUILDER, "window": [neutral[0], "2026-10-27"]}, "neutral", f"neu_ samples in {{}}, e.g. neu_x, were drawn from {[neutral[0], '2026-10-27']}, not {neutral[0]}/{END}"),
            ("head_1_x", {}, "head", "head_ samples in {}, e.g. head_1_x, predate the builder field; build into a fresh --data"),
            # Judged by another checkout's workflows, whose job classification can differ.
            ("rev_x", {"builder": bd.BUILDER, "window": [str(START), str(END)], "workflows": "0" * 12}, "revert", f"rev_ samples in {{}}, e.g. rev_x, judged jobs by workflows {'0' * 12}, not this checkout's {bd.workflows()}; extend it from the checkout that built it or build into a fresh --data"),
        ]  # fmt: skip
        for sid, mark, kind, msg in cases:
            with (
                tempfile.TemporaryDirectory() as tmp,
                mock.patch.object(sys, "argv", [*argv[:2], tmp, *argv[3:], "pr", kind]),
                mock.patch.object(bd, "WINDOW", dt.timedelta(0)),
                mock.patch.object(bd, "git") as git,
                mock.patch.object(bd.History, "load") as load,
                self.assertRaises(SystemExit) as e,
            ):  # fmt: skip
                (Path(tmp) / "samples").mkdir()
                (Path(tmp) / "samples" / f"{sid}.json").write_text(json.dumps(mark))
                bd.main()
            self.assertEqual(e.exception.code, f"refusing {kind} samples: " + msg.format(Path(tmp) / "samples"))  # fmt: skip
            git.assert_not_called()
            load.assert_not_called()
        # Samples of this builder and window are extended.
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.object(sys, "argv", [*argv[:2], tmp, *argv[3:], "revert", "neutral"]),
            mock.patch.object(bd, "git"),
            mock.patch.object(bd, "revert_samples") as revert,
            mock.patch.object(bd, "neutral_samples") as neutral_samples,
        ):  # fmt: skip
            (Path(tmp) / "samples").mkdir()
            (Path(tmp) / "samples" / "rev_x.json").write_text(json.dumps({"builder": bd.BUILDER, "window": [str(START), str(END)], "workflows": bd.workflows()}))  # fmt: skip
            (Path(tmp) / "samples" / "neu_x.json").write_text(json.dumps({"builder": bd.BUILDER, "window": neutral, "workflows": bd.workflows()}))  # fmt: skip
            bd.main()
        revert.assert_called_once_with(Path(tmp) / "samples", 8, START, END)
        neutral_samples.assert_called_once_with(Path(tmp) / "samples", END - dt.timedelta(days=25), END, 200)  # fmt: skip

    def test_fetch_retries_a_failed_batch_per_sha(self) -> None:
        def run(cmd: list[str], **kw: object) -> subprocess.CompletedProcess:
            shas = cmd[4:]
            return subprocess.CompletedProcess(
                cmd, int("gone" in shas), "", "not our ref"
            )

        shas = [f"s{i}" for i in range(150)] + ["gone"]
        with mock.patch.object(bd.subprocess, "run", side_effect=run) as m, contextlib.redirect_stdout(io.StringIO()) as out:  # fmt: skip
            bd.fetch(shas)
        # The first batch, then the second and each of its 51 shas.
        self.assertEqual([c.args[0][4:] for c in m.call_args_list], [shas[:100], shas[100:], *([s] for s in shas[100:])])  # fmt: skip
        self.assertEqual(out.getvalue(), "git fetch of gone failed: not our ref\n")

    def test_is_ancestor(self) -> None:
        with scratch_repo():
            for msg in ("a", "b"):
                bd.git("commit", "-q", "--allow-empty", "-m", msg)
            a, b = bd.git("rev-list", "HEAD").split()[::-1]
            self.assertTrue(bd.is_ancestor(a, b))
            self.assertFalse(bd.is_ancestor(b, a))
            self.assertFalse(bd.is_ancestor("0" * 40, b))

    def test_landed_times_takes_the_last_trailer(self) -> None:
        # PR 2 also landed earlier, as the older commit at 50 the log lists after it.
        log = "\n100 Reland\n\nPull Request resolved: https://github.com/pytorch/pytorch/pull/1\n\nPull Request resolved: https://github.com/pytorch/pytorch/pull/2\n\0\n200 x\n\0\n50 Land\n\nPull Request resolved: https://github.com/pytorch/pytorch/pull/2\n\0"  # fmt: skip
        with mock.patch.object(bd, "git", return_value=log):
            got = bd.landed_times(dt.date(2026, 10, 1))
        self.assertEqual(got, {2: dt.datetime.fromtimestamp(50, tz=dt.timezone.utc)})

    def test_revert_truth_jobs_commits(self) -> None:
        def run(*args: str, cwd: str, date: str = "2026-10-05T00:00:00+00:00") -> str:
            env = {**os.environ, "GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date}
            return subprocess.run(["git", *args], cwd=cwd, env=env, capture_output=True, text=True, check=True).stdout  # fmt: skip

        with scratch_repo():
            d = str(bd.REPO)
            commits = {}
            for name, hour in (("base", 0), ("change", 1), ("after", 2), ("revert", 3), ("late", 5)):  # fmt: skip
                msg = f"This reverts commit {commits['change']}." if name == "revert" else name  # fmt: skip
                run("commit", "-q", "--allow-empty", "-m", msg, cwd=d, date=f"2026-10-05T0{hour}:00:00+00:00")  # fmt: skip
                commits[name] = run("rev-parse", "HEAD", cwd=d).strip()
            run("update-ref", "refs/remotes/origin/main", "HEAD", cwd=d)
            queries: list[str] = []
            landed = dt.datetime(2026, 10, 5, 1, tzinfo=dt.timezone.utc)
            td, other = ("trunk", "linux / test (default, 1, 1)"), ("periodic", "linux / test (default, 1, 1)")  # fmt: skip
            # As for PR truth, an unstable job and one that passed on another attempt on the commit are not truth.
            unstable, rerun = ("trunk", "linux / test (default, 1, 1, unstable)"), ("trunk", "linux / test (default, 1, 2)")  # fmt: skip
            failed = [{"tf": "test_a", "classname": "TestA", "test": "test_x", "sha": commits["change"], "wf": wf, "job": j} for wf, j in (td, other, unstable, rerun)]  # fmt: skip
            failed += [{"tf": "test_b", "classname": "TestB", "test": "test_q'r", "sha": commits["change"], "wf": other[0], "job": other[1]}]  # fmt: skip
            # A key with a class matches only that class's test.
            failed += [{"tf": "test_c", "classname": c, "test": "test_y", "sha": commits["change"], "wf": td[0], "job": td[1]} for c in ("TestOther", "TestC")]  # fmt: skip
            signal = {("test_a", "", "test_x"), ("test_b", "", "test_q'r"), ("test_c", "TestC", "test_y")}  # fmt: skip
            with (
                patch_ci_jobs({True: {(*td, "default")}, False: {(*other, "default")}}),
                mock.patch.object(bd, "git", lambda *a, check=True: run(*a, cwd=d)),
                mock.patch.object(bd, "query", lambda sql: queries.append(sql) or failed),
                mock.patch.object(bd, "green_jobs", return_value={(commits["change"], *rerun), (commits["after"], *td)}),
            ):  # fmt: skip
                got = bd.revert_truth_jobs(signal, commits["change"], landed, "2026-10-05 04:00:00")  # fmt: skip
                # Decided before the change landed: no commit qualifies, so only the change itself.
                bd.revert_truth_jobs({("test_a", "", "test_x")}, commits["change"], landed, "2026-10-05 00:30:00")  # fmt: skip
                # Decided after the revert landed: the range still ends before the revert, so "late" is not counted.
                bd.revert_truth_jobs({("test_a", "", "test_x")}, commits["change"], landed, "2026-10-05 06:00:00")  # fmt: skip
                failed = failed[1:-1]
                # Without the TD job's rows, only periodic, unstable and rerun-green rows remain, so revert_samples skips it.
                none = bd.revert_truth_jobs(signal, commits["change"], landed, "2026-10-05 04:00:00")  # fmt: skip
        self.assertEqual(got, {"test_a": [td], "test_c": [td]})
        self.assertEqual(none, {})
        # The change and the commit after it, not the revert or what landed after the decision.
        self.assertIn(f"head_sha IN ('{commits['after']}','{commits['change']}')", queries[0])  # fmt: skip
        self.assertIn(f"head_sha IN ('{commits['change']}')", queries[1])
        self.assertIn(f"head_sha IN ('{commits['after']}','{commits['change']}')", queries[2])  # fmt: skip
        self.assertEqual(queries[0].count("created_at <= parseDateTimeBestEffort('2026-10-05 04:00:00') + INTERVAL 1 HOUR"), 2)  # fmt: skip
        # Only the signal tests count, not every test of their files.
        self.assertIn("AND (file, t.name) IN (('test_a.py', 'test_x'),('test_b.py', 'test_q\\'r'),('test_c.py', 'test_y'))", queries[0])  # fmt: skip
        # A test that also passed in its job is not a failure there.
        self.assertIn("HAVING countIf(failure_count > 0 OR error_count > 0) > 0 AND countIf(failure_count = 0 AND error_count = 0 AND skipped_count = 0) = 0", queries[0])  # fmt: skip

    def test_pr_text_retries_then_raises(self) -> None:
        ok = subprocess.CompletedProcess([], 0, '{"title": "t", "body": "b"}', "")
        bad = subprocess.CompletedProcess([], 1, "", "API rate limit exceeded")
        with mock.patch.object(bd.time, "sleep") as sleep:
            with mock.patch.object(bd.subprocess, "run", side_effect=[bad, bad, ok]):
                self.assertEqual(bd.pr_text(1), ("t", "b"))
            self.assertEqual(sleep.call_count, 2)
            with mock.patch.object(bd.subprocess, "run", return_value=bad) as run:
                with self.assertRaises(subprocess.CalledProcessError) as e:
                    bd.pr_text(1)
            self.assertEqual(e.exception.stderr, "API rate limit exceeded")
            self.assertEqual(run.call_count, 4)
            gone = subprocess.CompletedProcess([], 1, "", "gh: Not Found (HTTP 404)")
            with mock.patch.object(bd.subprocess, "run", return_value=gone) as run:
                with self.assertRaises(subprocess.CalledProcessError) as e:
                    bd.pr_text(1)
            self.assertEqual(e.exception.stderr, "gh: Not Found (HTTP 404)")
            self.assertEqual(run.call_count, 1)
            junk = subprocess.CompletedProcess([], 0, "null", "")
            with mock.patch.object(bd.subprocess, "run", return_value=junk) as run:
                with self.assertRaisesRegex(
                    ValueError, "gh api returned no PR 1: 'null'"
                ):
                    bd.pr_text(1)
            self.assertEqual(run.call_count, 1)

    def test_pr_text_failure_skips_sample(self) -> None:
        h = history([fail("test_x", "2026-10-05 00:00:00", "s")], {"s": 1}, {1: "me"})
        error = subprocess.CalledProcessError(1, "gh")
        with (
            tempfile.TemporaryDirectory() as out,
            patch_ci_jobs(TD_JOBS),
            mock.patch.object(bd, "green_jobs", return_value=set()),
            mock.patch.object(bd, "fetch"),
            mock.patch.object(bd, "has_commit", return_value=True),
            mock.patch.object(bd, "git", return_value="base\n"),
            mock.patch.object(bd, "commit_time", return_value=day(4)),
            mock.patch.object(bd, "ci_bases", return_value={}),
            mock.patch.object(bd, "pr_text", side_effect=error) as pr_text,
            mock.patch.object(bd, "write_sample") as write,
            contextlib.redirect_stdout(io.StringIO()) as stdout,
        ):  # fmt: skip
            bd.pr_samples(h, Path(out), 1)
            # A rebuild keeps an existing sample without asking gh again.
            (Path(out) / "pr_1.json").write_text("{}")
            bd.pr_samples(h, Path(out), 1)
        pr_text.assert_called_once_with(1)
        write.assert_not_called()
        # Skipped by pr_samples' own handler, not counted by run_each as raising.
        self.assertIn(f"skipping PR 1: {error}", stdout.getvalue())
        self.assertNotIn("raised", stdout.getvalue())

    def test_unexpected_error_skips_only_its_sample(self) -> None:
        h = history([fail("test_x", "2026-10-05 00:00:00", "s"), fail("test_x", "2026-10-05 00:00:00", "t", tf="test_b")], {"s": 1, "t": 2}, {1: "me", 2: "you"})  # fmt: skip
        bad = UnicodeDecodeError("utf-8", b"", 0, 1, "bad")
        with (
            tempfile.TemporaryDirectory() as out,
            patch_ci_jobs(TD_JOBS),
            mock.patch.object(bd, "green_jobs", return_value=set()),
            mock.patch.object(bd, "fetch"),
            mock.patch.object(bd, "has_commit", return_value=True),
            mock.patch.object(bd, "git", return_value="base\n"),
            mock.patch.object(bd, "commit_time", return_value=day(4)),
            mock.patch.object(bd, "ci_bases", return_value={}),
            mock.patch.object(bd, "pr_text", side_effect=[bad, ("t", "b")]),
            mock.patch.object(bd, "write_sample") as write,
            contextlib.redirect_stdout(io.StringIO()) as stdout,
        ):  # fmt: skip
            bd.pr_samples(h, Path(out), 1)
            # With no pr_ sample written, the head pass reports and returns, so later kinds still run.
            bd.head_samples(h, Path(out), 1)
        self.assertEqual([c.args[1] for c in write.call_args_list], ["pr_2"])
        self.assertIn("1 of 2 PR samples raised and were skipped", stdout.getvalue())
        self.assertIn("pr truth: dropped by rule over 2 heads", stdout.getvalue())
        self.assertIn("no head samples", stdout.getvalue())

    def test_pr_samples(self) -> None:
        t = "2026-10-05 00:00:00"
        dynamo = "dynamo-unittest / dynamo-test (3.1{}) / test (default, 1, 1)"
        rows = [
            # The later head's rows come first, the earlier head is still the sample's.
            fail("test_x", "2026-10-06 00:00:00", "s1", tf="test_c"),
            # The date is the earliest failure from --start on, not the first row's or one from before --start.
            fail("test_w", "2026-10-07 00:00:00", "s", tf="test_d"),
            fail("test_v", "2026-09-30 00:00:00", "s", tf="test_d"),
            fail("test_x", t, "s", tf="test_a"),
            fail("test_x", t, "s", tf="test_b"),
            fail("test_x", t, "s", tf="test_b", job="win / test (default, 1, 1)"),
            fail("test_x", t, "s", tf="test_c"),
            # A one-off: one environment, one head.
            fail("test_x", t, "s", tf="test_d"),
            # Main broke test_a 10 days before the merge base with main, and test_c's test_y, another test of the file.
            fail("test_x", "2026-09-05 00:00:00", "m", br="main", tf="test_a"),
            fail("test_y", "2026-09-05 00:00:00", "m", br="main", tf="test_c"),
            # The earliest failing head before --start is not the sample's.
            fail("test_x", "2026-09-30 00:00:00", "s0", tf="test_a"),
            # The later head above reproduces test_c, and a head before --start test_e.
            fail("test_x", t, "s", tf="test_e"),
            fail("test_x", "2026-09-28 00:00:00", "s0", tf="test_e"),
            # Two Python versions of one reusable workflow's job are two environments.
            fail("test_x", t, "s", tf="test_f", job=dynamo.format(1)),
            fail("test_x", t, "s", tf="test_f", job=dynamo.format(2)),
        ]
        h = history(rows, {"s": 1, "s0": 1, "s1": 1}, {1: "me"})
        merge_bases = {("origin/main", "s"): "main_mb", ("lower", "s"): "lower", ("origin/main", "s0"): "main_mb0", ("origin/main", "s1"): "main_mb1"}  # fmt: skip
        with (
            tempfile.TemporaryDirectory() as out,
            patch_ci_jobs({True: {("pull", "pull", "default"), ("pull", "win", "default"), *(("pull", dynamo.format(v).rsplit(" / ", 1)[0], "default") for v in (1, 2))}, False: set()}),
            mock.patch.object(bd, "green_jobs", return_value=set()),
            mock.patch.object(bd, "fetch") as fetch,
            mock.patch.object(bd, "has_commit", return_value=True),
            mock.patch.object(bd, "git", lambda *a, check=True: merge_bases[a[1:]] + "\n"),
            mock.patch.object(bd, "commit_time", return_value=day(15, 9)) as commit_time,
            mock.patch.object(bd, "ci_bases", return_value={"s": ["lower"]}) as ci_bases,
            mock.patch.object(bd, "pr_text", return_value=("t", "b")),
            mock.patch.object(bd, "write_sample") as write,
            contextlib.redirect_stdout(io.StringIO()) as stdout,
        ):  # fmt: skip
            bd.pr_samples(h, Path(out), 1)
        # Every head of the PR is judged, as each can reproduce a failure of the sample's head.
        ci_bases.assert_called_once_with({"s": 1, "s0": 1, "s1": 1})
        # A CI base recorded after the head is not in its history, so it is fetched too.
        self.assertEqual(fetch.call_args_list[-1].args, (["lower"],))
        # Main breakage is checked around the merge base with main, the diff is against the PR's CI base.
        self.assertEqual(sorted(c.args[0] for c in commit_time.call_args_list), ["main_mb", "main_mb0", "main_mb1"])  # fmt: skip
        kw = write.call_args.kwargs
        win = ("pull", "win / test (default, 1, 1)")
        self.assertEqual((write.call_args.args[3], kw["date"]), ("s", t))
        self.assertEqual((kw["truth"], kw["truth_strong"], kw["truth_jobs"]), (["test_b", "test_c", "test_d", "test_e", "test_f"], ["test_b", "test_c", "test_e", "test_f"], {"test_b": [("pull", JOB), win], "test_c": [("pull", JOB)], "test_d": [("pull", JOB)], "test_e": [("pull", JOB)], "test_f": [("pull", dynamo.format(v)) for v in (1, 2)]}))  # fmt: skip
        self.assertEqual({k: kw[k] for k in ("base", "merge_base", "main_base", "base_source")}, {"base": "lower", "merge_base": "lower", "main_base": "main_mb", "base_source": "ci"})  # fmt: skip
        self.assertIn("pr truth: dropped by rule over 1 heads: {'main_near_base': 1}", stdout.getvalue())  # fmt: skip

    def test_pr_samples_judge_every_head_by_all_rules(self) -> None:
        rows = [
            # Main broke test_z 15 days before the sample's head failed it, so only the rule on the head's merge base drops it.
            fail("test_z", "2026-10-05 00:00:00", "s", tf="test_z"),
            fail("test_z", "2026-09-20 00:00:00", "m", br="main", tf="test_z"),
            fail("test_x", "2026-10-07 00:00:00", "s", tf="test_b"),
            # The later head reproduces test_b, but main broke it 6 days before that head's merge base with main.
            fail("test_x", "2026-10-08 00:00:00", "s1", tf="test_b"),
            fail("test_x", "2026-09-10 00:00:00", "m", br="main", tf="test_b"),
        ]
        h = history(rows, {"s": 1, "s1": 1}, {1: "me"})
        merge_bases = {("origin/main", "s"): "mb", ("origin/main", "s1"): "mb1"}
        with (
            tempfile.TemporaryDirectory() as out,
            patch_ci_jobs(TD_JOBS),
            mock.patch.object(bd, "green_jobs", return_value=set()),
            mock.patch.object(bd, "fetch"),
            mock.patch.object(bd, "has_commit", return_value=True),
            mock.patch.object(bd, "git", lambda *a, check=True: merge_bases[a[1:]] + "\n"),
            mock.patch.object(bd, "commit_time", side_effect={"mb": day(25, 9), "mb1": day(16, 9)}.get),
            mock.patch.object(bd, "ci_bases", return_value={}),
            mock.patch.object(bd, "pr_text", return_value=("t", "b")),
            mock.patch.object(bd, "write_sample") as write,
            contextlib.redirect_stdout(io.StringIO()) as stdout,
        ):  # fmt: skip
            bd.pr_samples(h, Path(out), 1)
        kw = write.call_args.kwargs
        # The date is the sample's earliest failure in its truth, not test_z's, and s1's test_b is not a reproduction.
        self.assertEqual((write.call_args.args[3], kw["truth"], kw["truth_strong"], kw["date"]), ("s", ["test_b"], [], "2026-10-07 00:00:00"))  # fmt: skip
        self.assertIn("pr truth: dropped by rule over 1 heads: {'main_near_base': 1}", stdout.getvalue())  # fmt: skip

    def test_empty_truth_writes_no_sample(self) -> None:
        # main_near_base drops all of the earliest head's failures, so PR 1 gets no pr_ sample and the later head does not stand in.
        rows = [
            fail("test_x", "2026-10-05 00:00:00", "s"),
            fail("test_x", "2026-09-05 00:00:00", "m", br="main"),
            fail("test_x", "2026-10-06 00:00:00", "s1", tf="test_b"),
        ]
        h = history(rows, {"s": 1, "s1": 1}, {1: "me"})
        with (
            tempfile.TemporaryDirectory() as d,
            patch_ci_jobs(TD_JOBS),
            mock.patch.object(bd, "green_jobs", return_value=set()),
            mock.patch.object(bd, "fetch"),
            mock.patch.object(bd, "has_commit", return_value=True),
            mock.patch.object(bd, "git", return_value="main_mb\n"),
            mock.patch.object(bd, "commit_time", return_value=day(15, 9)),
            mock.patch.object(bd, "ci_bases", return_value={}),
            mock.patch.object(bd, "head_created", return_value={}),
            mock.patch.object(bd, "pr_text", return_value=("t", "b")),
            mock.patch.object(bd, "write_sample") as write,
            contextlib.redirect_stdout(io.StringIO()) as stdout,
        ):  # fmt: skip
            out = Path(d)
            bd.pr_samples(h, out, 1)
            write.assert_not_called()
            self.assertIn("pr truth: dropped by rule over 1 heads: {'main_near_base': 1}", stdout.getvalue())  # fmt: skip
            # Given a pr_ sample, the head pass writes the later head but not the one whose truth is empty.
            bd.write_json(out / "pr_1.json", {"pr": 1, "title": "t", "body": "b"})
            bd.head_samples(h, out, 1)
        self.assertEqual([c.args[1] for c in write.call_args_list], ["head_1_s1"])
        self.assertIn("head truth: dropped by rule over 2 heads: {'main_near_base': 1}", stdout.getvalue())  # fmt: skip

    def test_pr_and_head_truth_agree(self) -> None:
        green_job = "pull / test (default, 2, 2)"
        rows = [
            fail("test_real", "2026-10-05 00:00:00", "h1"),
            fail("test_green", "2026-10-05 00:00:00", "h1", job=green_job),
            fail("test_disabled", "2026-10-05 00:00:00", "h1"),
            fail("test_near_base", "2026-10-05 00:00:00", "h1"),
            fail("test_near_base", "2026-09-12 00:00:00", "m", br="main"),
            # 10 days before: once only head truth's window, now both kinds'.
            fail("test_other", "2026-10-05 00:00:00", "h1"),
            fail("test_other", "2026-09-25 00:00:00", "o"),
            fail("test_later", "2026-10-08 00:00:00", "h2"),
        ]
        h = history(rows, {"h1": 1, "h2": 1, "o": 2}, {1: "me", 2: "x"}, disabled=(("2026-10-01", "test_disabled"),))  # fmt: skip
        bases = {"base": "b", "merge_base": "b", "main_base": "m", "base_source": "ci"}
        with (
            tempfile.TemporaryDirectory() as d,
            patch_ci_jobs(TD_JOBS),
            mock.patch.object(bd, "green_jobs", return_value={("h1", "pull", green_job)}),
            mock.patch.object(bd, "fetch"),
            mock.patch.object(bd, "has_commit", return_value=True),
            mock.patch.object(bd, "ci_bases", return_value={}),
            mock.patch.object(bd, "pr_bases", return_value=bases),
            mock.patch.object(bd, "commit_time", return_value=day(20, 9)),
            mock.patch.object(bd, "pr_text", return_value=("t", "b")),
            mock.patch.object(bd, "diffstat_and_diff", return_value=("", "")),
            mock.patch.object(bd, "changed_files", return_value=[]),
            mock.patch.object(bd, "deleted_files", return_value=[]),
            mock.patch.object(bd, "candidates_at", return_value=[]),
            mock.patch.object(bd, "head_created", return_value={}),
            contextlib.redirect_stdout(io.StringIO()) as stdout,
        ):  # fmt: skip
            out = Path(d)
            bd.pr_samples(h, out, 1)
            bd.head_samples(h, out, 1)
            samples = {p.stem: json.loads(p.read_text()) for p in out.glob("*.json")}
            # A rebuild into the same DIR writes and counts no sample again.
            bd.pr_samples(h, out, 1)
            bd.head_samples(h, out, 1)
        self.assertEqual(sorted(samples), ["head_1_h1", "head_1_h2", "pr_1"])
        pr, head = samples["pr_1"], samples["head_1_h1"]
        self.assertEqual((pr["sha"], pr["truth"]), ("h1", ["test_a"]))
        self.assertEqual({(s["builder"], tuple(s["window"])) for s in samples.values()}, {(bd.BUILDER, (str(START), str(END)))})  # fmt: skip
        self.assertEqual((pr["truth"], pr["truth_jobs"]), (head["truth"], head["truth_jobs"]))  # fmt: skip
        dropped = (
            "{'disabled': 1, 'job_green': 1, 'main_near_base': 1, 'other_failure': 1}"
        )
        self.assertIn(f"pr truth: dropped by rule over 1 heads: {dropped}", stdout.getvalue())  # fmt: skip
        self.assertIn(f"head truth: dropped by rule over 2 heads: {dropped}", stdout.getvalue())  # fmt: skip
        for line in ("0 of 1 PR samples written", "0 of 2 head samples written", "0 of 0 PR samples written", "0 of 0 head samples written"):  # fmt: skip
            self.assertIn(line, stdout.getvalue())

    def test_pr_bases(self) -> None:
        # ghstack's base at the push is in the head's history; a later base and a fork's base merge-base older.
        merge_bases = {("origin/main", "h"): "m", ("b_push", "h"): "b_push", ("b_later", "h"): "b_old", ("b_gone", "h"): ""}  # fmt: skip

        def git(*a: str, check: bool = True) -> str:
            if a[:2] == ("merge-base", "--independent"):
                self.assertEqual(set(a[2:]), {"b_push", "b_old"})
                return "b_push\n"
            return merge_bases[(a[1], a[2])] + "\n"

        with mock.patch.object(bd, "git", git):
            got = bd.pr_bases("h", ["b_gone", "b_later", "b_push"])
            self.assertEqual(got, {"base": "b_push", "merge_base": "b_push", "main_base": "m", "base_source": "ci"})  # fmt: skip
            self.assertEqual(bd.pr_bases("h", ["b_later"])["merge_base"], "b_old")  # type: ignore[index]
            # b_child is a later ghstack base on top of b_push, so it merge-bases at b_push too; b_push wins in either order.
            merge_bases[("b_child", "h")] = "b_push"
            for order in (["b_child", "b_push"], ["b_push", "b_child"]):
                self.assertEqual(bd.pr_bases("h", order)["base"], "b_push")  # type: ignore[index]
            main = {"base": "m", "merge_base": "m", "main_base": "m", "base_source": "main"}  # fmt: skip
            self.assertEqual(bd.pr_bases("h", ["b_gone"]), main)
            self.assertEqual(bd.pr_bases("h", []), main)
            merge_bases[("origin/main", "h")] = ""
            self.assertIsNone(bd.pr_bases("h", ["b_push"]))

    def test_head_samples(self) -> None:
        job_a, job_b = "linux / test (default, 1, 2)", "linux / test (default, 2, 2)"
        t = "2026-10-05 00:00:00"
        # h1 is the pr_ sample's head: job_a failed, then passed on a rerun, so only job_b's failure is truth.
        rows = [
            fail("test_x", t, "h1", job=job_a, tf="test_a"),
            fail("test_x", t, "h1", job=job_b, tf="test_c"),
            fail("test_x", t, "h2", job=job_a, tf="test_b"),
            fail("test_x", t, "h3", job=job_a, tf="test_d"),
            # Passed again in its job, so it left the pytest cache; and a job that does not run TD.
            fail("test_y", t, "h3", job=job_a, tf="test_h", flaky=True),
            fail("test_x", t, "h3", wf="periodic", job=job_a, tf="test_i"),
            fail("test_x", t, "h4", job=job_a, tf="test_e"),
            # Every failure of h5 is in a job that passed, and h6 is another PR's.
            fail("test_x", t, "h5", job=job_a, tf="test_f"),
            fail("test_x", t, "h6", job=job_a, tf="test_g"),
        ]
        h = history(rows, {"h1": 1, "h2": 1, "h3": 1, "h4": 1, "h5": 1, "h6": 2}, {1: "me", 2: "x"})  # fmt: skip
        # h1 and h2 have a CI base each, h3 has no merge base with main, h4 has no CI base.
        merge_bases = {("origin/main", "h1"): "m1", ("b1", "h1"): "b1", ("origin/main", "h2"): "m2", ("b2", "h2"): "mb2", ("origin/main", "h3"): "", ("origin/main", "h4"): "m4"}  # fmt: skip
        with (
            tempfile.TemporaryDirectory() as out,
            patch_ci_jobs({True: {("pull", "linux", "default")}, False: set()}),
            mock.patch.object(bd, "green_jobs", return_value={("h1", "pull", job_a), ("h5", "pull", job_a)}),
            mock.patch.object(bd, "fetch") as fetch,
            mock.patch.object(bd, "has_commit", return_value=True),
            mock.patch.object(bd, "commit_time", return_value=day(4)),
            mock.patch.object(bd, "ci_bases", return_value={"h1": ["b1"], "h2": ["b2"], "h3": ["b3"]}) as ci_bases,
            # Every failure is at 10-05: h1's CI started before them, h2's after, h4's is unknown.
            mock.patch.object(bd, "head_created", return_value={"h1": "2026-10-04 00:00:00", "h2": "2026-10-06 00:00:00"}),
            mock.patch.object(bd, "git", lambda *a, check=True: merge_bases[a[1:]] + "\n"),
            mock.patch.object(bd, "write_sample") as write,
            contextlib.redirect_stdout(io.StringIO()),
        ):  # fmt: skip
            pr_1 = {"pr": 1, "sha": "h1", "base": "pr_base", "title": "t", "body": "b", "builder": bd.BUILDER, "window": [str(START), str(END)], "workflows": bd.workflows()}  # fmt: skip
            # pr_ samples of another builder or window, or of none, refuse both PR kinds before History.load.
            data = Path(out) / "data"
            (samples := data / "samples").mkdir(parents=True)
            argv = ["build_dataset", "--data", str(data), "--start", str(START), "--end", str(END), "--kinds"]  # fmt: skip
            want = [
                ({"window": [str(START), "2026-10-27"]}, f"pr_ samples in {samples}, e.g. pr_1, were drawn from {[str(START), '2026-10-27']}, not {START}/{END}"),
                ({"builder": bd.BUILDER - 1}, f"pr_ samples in {samples}, e.g. pr_1, were written by builder {bd.BUILDER - 1}, not {bd.BUILDER}"),
                ({"builder": None, "window": None}, f"pr_ samples in {samples}, e.g. pr_1, predate the builder field"),
            ]  # fmt: skip
            for mark, msg in want:
                (samples / "pr_1.json").write_text(json.dumps({**pr_1, **mark}))
                self.assertIn(msg, bd.other_build(samples, "pr_", (START, END)) or "")
                for kind in ("pr", "head"):
                    # A zero WINDOW lets the guard on --end + WINDOW pass.
                    with mock.patch.object(sys, "argv", [*argv, kind]), mock.patch.object(bd, "WINDOW", dt.timedelta(0)), mock.patch.object(bd.History, "load") as load, self.assertRaises(SystemExit) as e:  # fmt: skip
                        bd.main()
                    self.assertIn(msg, str(e.exception.code))
                    load.assert_not_called()
            fetch.assert_not_called()
            write.assert_not_called()
            (samples / "pr_1.json").write_text(json.dumps(pr_1))
            self.assertIsNone(bd.other_build(samples, "pr_", (START, END)))
            (Path(out) / "pr_1.json").write_text(json.dumps(pr_1))
            bd.head_samples(h, Path(out), 1)
        ci_bases.assert_called_once_with({"h1": 1, "h2": 1, "h3": 1, "h4": 1})
        self.assertEqual(fetch.call_args_list[-1].args, (["b1", "b2", "b3"],))
        got = {c.args[1]: (c.args[2:], c.kwargs) for c in write.call_args_list}
        self.assertEqual(sorted(got), ["head_1_h1", "head_1_h2", "head_1_h4"])
        # h2's previous failures are the other heads' (also of h3 and h5, which have no sample) but h3's test_h and test_i.
        previous = {"head_1_h1": [], "head_1_h2": ["test_a", "test_c", "test_d", "test_e", "test_f"], "head_1_h4": []}  # fmt: skip
        want = {
            "head_1_h1": ("h1", "b1", "b1", "m1", "ci", {"test_c": [("pull", job_b)]}),
            "head_1_h2": ("h2", "b2", "mb2", "m2", "ci", {"test_b": [("pull", job_a)]}),
            "head_1_h4": (
                "h4",
                "m4",
                "m4",
                "m4",
                "main",
                {"test_e": [("pull", job_a)]},
            ),
        }
        for sid, (sha, base, merge_base, main_base, source, truth_jobs) in want.items():
            args, kw = got[sid]
            self.assertEqual(args, ("head", sha))
            bases = {"base": base, "merge_base": merge_base, "main_base": main_base, "base_source": source}  # fmt: skip
            self.assertEqual(kw, {"title": "t", "body": "b", **bases, "pr": 1, "window": (START, END), "truth": sorted(truth_jobs), "truth_jobs": truth_jobs, "previous_failures": previous[sid]})  # fmt: skip

    def test_revert_truth_is_truth_jobs(self) -> None:
        rows = [{"reverted_at": t, "commit_sha": "c" * 40, "source_signal_keys": ["test_a.py::x", "test_b.py::TestB::y", "no_test_key"]} for t in ("2026-10-02 01:00:00", "2026-10-03 01:00:00")]  # fmt: skip
        jobs = {"test_a": [("trunk", "linux / test (default, 1, 1)")]}
        with (
            tempfile.TemporaryDirectory() as out,
            mock.patch.object(bd, "query", return_value=rows),
            mock.patch.object(bd, "check_written"),
            mock.patch.object(bd, "fetch"),
            mock.patch.object(bd, "has_commit", return_value=True),
            mock.patch.object(bd, "commit_time"),
            mock.patch.object(bd, "revert_truth_jobs", return_value=jobs) as truth_jobs,
            mock.patch.object(bd, "git", return_value="t\nb"),
            mock.patch.object(bd, "write_sample") as write,
        ):  # fmt: skip
            bd.revert_samples(Path(out), 1, dt.date(2026, 10, 1), dt.date(2026, 10, 26))
        truth_jobs.assert_called_once_with({("test_a", "", "x"), ("test_b", "TestB", "y")}, "c" * 40, mock.ANY, "2026-10-02 01:00:00")  # fmt: skip
        kw = write.call_args.kwargs
        self.assertEqual((kw["window"], kw["truth"], kw["truth_jobs"]), ((dt.date(2026, 10, 1), dt.date(2026, 10, 26)), ["test_a"], jobs))  # fmt: skip
        with (
            tempfile.TemporaryDirectory() as out,
            mock.patch.object(bd, "query", return_value=rows),
            mock.patch.object(bd, "check_written"),
            mock.patch.object(bd, "fetch"),
            mock.patch.object(bd, "has_commit", return_value=True),
            mock.patch.object(bd, "commit_time"),
            mock.patch.object(bd, "revert_truth_jobs", return_value={}),
            mock.patch.object(bd, "git", return_value="t\nb"),
            mock.patch.object(bd, "write_sample") as write,
            contextlib.redirect_stdout(io.StringIO()),
            # With no revert left, the store is suspect rather than every revert truthless.
            self.assertRaisesRegex(RuntimeError, "none of the 1 reverts has a failure of its signal tests"),
        ):  # fmt: skip
            bd.revert_samples(Path(out), 1, dt.date(2026, 10, 1), dt.date(2026, 10, 26))
        write.assert_not_called()
        # A ClickHouse failure is counted as raising, not as a revert with no truth.
        with (
            tempfile.TemporaryDirectory() as out,
            mock.patch.object(bd, "query", return_value=rows),
            mock.patch.object(bd, "check_written"),
            mock.patch.object(bd, "fetch"),
            mock.patch.object(bd, "has_commit", return_value=True),
            mock.patch.object(bd, "commit_time"),
            mock.patch.object(bd, "revert_truth_jobs", side_effect=RuntimeError("ClickHouse query failed")),
            contextlib.redirect_stdout(io.StringIO()) as stdout,
        ):  # fmt: skip
            bd.revert_samples(Path(out), 1, dt.date(2026, 10, 1), dt.date(2026, 10, 26))
        self.assertIn("1 of 1 revert samples raised and were skipped", stdout.getvalue())  # fmt: skip

    def test_revert_samples_check_the_store(self) -> None:
        rows = [{"reverted_at": f"2026-10-0{d} 01:00:00", "commit_sha": c * 40, "source_signal_keys": ["test_a.py::x"]} for d, c in ((2, "a"), (5, "b"), (9, "c"))]  # fmt: skip
        written = [{"d": "2026-10-02"}]

        def query(sql: str) -> list[dict]:
            return rows if "autorevert_events_v2" in sql else written

        def revert(truth: dict, raises: str | None) -> list[str]:
            with (
                mock.patch.object(bd, "query", side_effect=query) as q,
                mock.patch.object(bd, "fetch"),
                mock.patch.object(bd, "has_commit", return_value=True),
                mock.patch.object(bd, "commit_time"),
                mock.patch.object(bd, "revert_truth_jobs", side_effect=lambda s, sha, *a: truth.get(sha, {})),
                mock.patch.object(bd, "commit_text", return_value=("t", "b")),
                mock.patch.object(bd, "write_sample"),
                contextlib.redirect_stdout(io.StringIO()),
                self.assertRaisesRegex(RuntimeError, raises) if raises else contextlib.nullcontext(),
            ):  # fmt: skip
                bd.revert_samples(Path("unused"), 1, dt.date(2026, 10, 1), dt.date(2026, 10, 26))  # fmt: skip
            return [c.args[0] for c in q.call_args_list[1:]]

        # Only the first and last revert days are probed; the last has no failing test, as a store no longer written gives.
        (probe,) = revert({}, "tests.all_test_runs has no failing test on 2026-10-09")
        self.assertIn("'2026-10-02'", probe)
        self.assertIn("'2026-10-09'", probe)
        self.assertNotIn("'2026-10-05'", probe)
        written.append({"d": "2026-10-09"})
        revert({}, "none of the 3 reverts has a failure of its signal tests")
        revert({"b" * 40: {"test_a": [("trunk", JOB)]}}, None)

    def test_neutral_samples(self) -> None:
        start, end = dt.date(2026, 10, 1), dt.date(2026, 10, 26)
        titles: dict[str, str] = {"a" * 40: "Fix foo", "b" * 40: "[ghstack] Revert bar", "c" * 40: "Reland: baz", "d" * 40: "Add RevertibleOp"}  # fmt: skip
        # Reverted by a message and by the autorevert bot.
        titles |= {"e" * 40: "Add e", "f" * 40: "Add f"}
        calls = []

        def git(*args: str) -> str:
            calls.append(args)
            if args[:2] == ("log", "-1"):
                return titles[args[-1]] + "\n\nbody"
            if "--first-parent" in args:
                return "".join(f"{sha} {t}\n" for sha, t in titles.items())
            return f"Revert\n\nThis reverts commit {'e' * 40}.\n"

        with (
            tempfile.TemporaryDirectory() as out,
            mock.patch.object(bd, "git", git),
            mock.patch.object(bd, "query", return_value=[{"commit_sha": "f" * 40}]),
            mock.patch.object(bd, "write_sample") as write,
        ):
            bd.neutral_samples(Path(out), start, end, 10)
        self.assertEqual(sorted(c.args[1] for c in write.call_args_list), ["neu_" + "a" * 12, "neu_" + "d" * 12])  # fmt: skip
        self.assertEqual({c.kwargs["window"] for c in write.call_args_list}, {(start, end)})  # fmt: skip
        log = next(c for c in calls if "--first-parent" in c)
        self.assertIn("--since=2026-10-01 00:00 +0000", log)
        self.assertIn("--until=2026-10-26 00:00 +0000", log)

    def test_neutral_draw_survives_a_later_revert(self) -> None:
        shas = [f"{i:02d}" + "a" * 38 for i in range(30)]

        def draw(revert_body: str, bot: list[dict]) -> list[str]:
            def git(*args: str) -> str:
                if args[:2] == ("log", "-1"):
                    return "Add x\n\nbody"
                if "--first-parent" in args:
                    return "".join(f"{sha} Add x\n" for sha in shas)
                return revert_body

            with (
                tempfile.TemporaryDirectory() as out,
                mock.patch.object(bd, "git", git),
                mock.patch.object(bd, "query", return_value=bot) as query,
                mock.patch.object(bd, "write_sample") as write,
            ):
                bd.neutral_samples(Path(out), dt.date(2026, 10, 1), dt.date(2026, 10, 26), 10)  # fmt: skip
            self.assertIn("dry_run = 0", query.call_args.args[0])
            return [c.args[3] for c in write.call_args_list]

        before = draw("", [])
        by_msg, by_bot = before[:2]
        later = draw(f"This reverts commit {by_msg}.", [{"commit_sha": by_bot}])
        self.assertEqual(later, before[2:])

    def test_neutral_population_drops_revert_titles(self) -> None:
        plain = [f"{i:02d}" + "a" * 38 for i in range(20)]
        titles = dict.fromkeys(plain, "Add x") | {f"{i:02d}" + "b" * 38: t for i, t in enumerate(["Revert x"] * 10 + ["Reland x"] * 10)}  # fmt: skip

        def git(*args: str) -> str:
            if args[:2] == ("log", "-1"):
                return titles[args[-1]] + "\n\nbody"
            if "--first-parent" in args:
                return "".join(f"{sha} {t}\n" for sha, t in titles.items())
            return ""

        with (
            tempfile.TemporaryDirectory() as out,
            mock.patch.object(bd, "git", git),
            mock.patch.object(bd, "query", return_value=[]),
            mock.patch.object(bd, "write_sample") as write,
        ):
            bd.neutral_samples(Path(out), dt.date(2026, 10, 1), dt.date(2026, 10, 26), 10)  # fmt: skip
        # The draw is from the titles that do not start with Revert or Reland, so all 10 survive.
        self.assertEqual([c.args[3] for c in write.call_args_list], random.Random(0).sample(plain, 10))  # fmt: skip

    def test_checked_in_ids(self) -> None:
        ids = {n: set((REPO_ROOT / "tools/testing/llm_selector/eval/ids" / f"{n}.txt").read_text().split()) for n in ("eval", "dev", "agentic", "neutral", "head", "pilot")}  # fmt: skip
        self.assertLessEqual(ids["neutral"], ids["eval"])
        self.assertLessEqual(ids["agentic"], ids["eval"])
        self.assertEqual(ids["dev"] & ids["eval"], set())
        self.assertEqual(ids["pilot"] & ids["eval"], set())
        every = set().union(*ids.values())
        shas = {k: {i.removeprefix(f"{k}_") for i in every if i.startswith(f"{k}_")} for k in ("neu", "rev")}  # fmt: skip
        self.assertEqual(shas["neu"] & shas["rev"], set())
        prs = {i for i in ids["eval"] | ids["dev"] if i.startswith("pr_")}
        self.assertEqual({f"pr_{i.split('_')[1]}" for i in ids["head"]} - prs, set())

    def test_candidates_at(self) -> None:
        ls = "test/test_a.py\ntest/ao/test_b.py\ntest/fx/test_c.py\ntest/test_skip.py\ntest/helper.py\n"
        added = "test/test_new.py\ntest/ao/test_d.py\ntest/fx/test_e.py\ntest/helper_new.py\n"
        named = 'BLOCKLISTED_PATTERNS = ["ao"]\nTESTS = discover_tests(blocklisted_patterns=BLOCKLISTED_PATTERNS, blocklisted_tests=["test_skip"], extra_tests=["extra"])\n'  # fmt: skip
        inline = 'TESTS = discover_tests(blocklisted_patterns=["fx"], blocklisted_tests=["test_skip"])\n'  # fmt: skip

        def git(*a: str) -> str:
            if a[0] == "diff" and not {"--diff-filter=A", "--no-renames"} <= set(a):
                raise AssertionError(f"not only the added files: {a}")
            if (a[0], a[1].split(":")[0] if a[0] == "show" else a[3]) in (
                ("show", "lower"),
                ("ls-tree", "lower"),
            ):
                raise AssertionError(f"TESTS not read at main: {a}")
            if a[0] == "diff" and a[-2:] != (diff_base, "s"):
                raise AssertionError(f"not diffed against {diff_base}: {a}")
            return {"show": src, "ls-tree": ls, "diff": added}[a[0]]

        def unquoted(*a: str) -> str:
            if a[1] != "-z":
                raise AssertionError(f"paths may be quoted: {a}")
            return git(a[0], *a[2:]).replace("\n", "\0")

        for src, want, new in ((named, ["extra", "fx/test_c", "test_a"], ["fx/test_e", "test_new"]), (inline, ["ao/test_b", "test_a"], ["ao/test_d", "test_new"])):  # fmt: skip
            with mock.patch.object(bd, "git", git), mock.patch.object(select_tests, "git", unquoted):  # fmt: skip
                diff_base = "b"
                self.assertEqual(bd.candidates_at("b"), want)
                self.assertEqual(bd.candidates_at("b", "s"), sorted(want + new))
                diff_base = "lower"
                self.assertEqual(
                    bd.candidates_at("b", "s", "lower"), sorted(want + new)
                )

    def test_write_sample_matches_select_tests(self) -> None:
        # select_tests' git and build_dataset's both run in the scratch repo. git C-quotes a
        # non-ASCII path in a listing without -z, which select_tests passes.
        odd = "test/test_\u00e9.py"
        with scratch_repo():
            repo, git = bd.REPO, bd.git
            (Path(repo) / "test").mkdir()
            (Path(repo) / "test" / "run_test.py").write_text("x = 1\n" * 50)
            (Path(repo) / odd).write_text("")
            (Path(repo) / "tools/testing").mkdir(parents=True)
            (Path(repo) / "tools/testing/discover_tests.py").write_text("TESTS = discover_tests(blocklisted_patterns=[])\n")  # fmt: skip
            git("add", "-A")
            git("commit", "-qm", "base")
            git("mv", "test/run_test.py", "test/my file.py")
            git("rm", "-q", odd)
            git("commit", "-qm", "move")
            (Path(repo) / "test" / "test_x.py").write_text("")
            (Path(repo) / odd).write_text("")
            git("add", "-A")
            git("commit", "-qm", "add")
            with mock.patch.object(bd, "candidates_at", return_value=[]) as candidates:
                bd.write_sample(Path(repo), "s", "pr", "HEAD~1", "HEAD~2", "t", "b", window=(START, END))  # fmt: skip
                bd.write_sample(Path(repo), "s2", "pr", "HEAD", "lower", "t", "b", merge_base="HEAD~1", main_base="HEAD~2", window=(START, END))  # fmt: skip
            s, s2 = (
                json.loads((Path(repo) / f"{n}.json").read_text()) for n in ("s", "s2")
            )
            self.assertIn("test_\u00e9", bd.candidates_at("HEAD~2"))
            self.assertIn("test_\u00e9", bd.candidates_at("HEAD~1", "HEAD"))
        self.assertEqual(s["changed_files"], ["test/my file.py", "test/run_test.py", odd])  # fmt: skip
        self.assertEqual((s["deleted"], s2["deleted"]), (["test/run_test.py", odd], []))  # fmt: skip
        self.assertIn("test/run_test.py", s["diffstat"])
        self.assertIn("rename from test/run_test.py", s["diff"])
        self.assertEqual((s["base"], s["merge_base"], s["main_base"]), ("HEAD~2", "HEAD~2", "HEAD~2"))  # fmt: skip
        self.assertEqual((s2["base"], s2["merge_base"], s2["main_base"], s2["changed_files"]), ("lower", "HEAD~1", "HEAD~2", ["test/test_x.py", odd]))  # fmt: skip
        self.assertEqual(candidates.call_args_list, [mock.call("HEAD~2", "HEAD~1", "HEAD~2"), mock.call("HEAD~2", "HEAD", "HEAD~1")])  # fmt: skip


if __name__ == "__main__":
    unittest.main()
