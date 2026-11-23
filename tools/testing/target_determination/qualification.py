from __future__ import annotations

import hashlib
import importlib.util
import json
import marshal
import math
import os
import platform
import signal
import stat
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import CodeType, ModuleType
from typing import Any, Iterator

from tools.testing.test_run import TestRun


EXPERIMENT_BRANCH = "cicd-td-same-input-qualification-20261123"
EXPERIMENT_FORK = "scotts/pytorch"
REFERENCE_BLOB = "7198a193aa089995ac8a11bac4d2b9cc53440a46"
REFERENCE_SHA256 = "795c000b774d1c7bbbdabb3108cd1a2b608e10976a2fd5502f30f85162a389c1"
CANDIDATE_SHA256 = "c148a407877e32f6846f1f8e17ae0da184b4b16ad100aad734cd4cd4699fd5c0"
TEST_RUN_SHA256 = "5e4c204ee649b6ca01f0c7e7056e71ba3473897ecb72f72be86e67a134387c35"
MAX_TESTS = 2000
MAX_ENTRIES = 40000
MAX_BYTES = 8 * 1024 * 1024
MAX_SECONDS = 300
PAIRS = (("original", "candidate"), ("candidate", "original"), ("original", "candidate"))
COMPONENT = "fresh aggregate, ordered adds/revalidation, ONE default aggregation"


class QualificationInvalid(RuntimeError):
    def __init__(self, message: str, report: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.report = report


def _encoded(value: Any) -> bytes:
    # Inputs have bounded scalar fields. Stream the aggregate instead of allocating all JSON first.
    encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"), allow_nan=False)
    output = bytearray()
    for chunk in encoder.iterencode(value):
        raw = chunk.encode()
        if len(output) + len(raw) > MAX_BYTES:
            raise QualificationInvalid("Qualification JSON exceeds the byte bound")
        output.extend(raw)
    return bytes(output)


def _digest(value: Any) -> str:
    return hashlib.sha256(_encoded(value)).hexdigest()


def _read(path: Path, limit: int) -> bytes:
    fd = os.open(path.resolve(), os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise QualificationInvalid("A bounded regular source/event file is required")
        data = stream.read(limit + 1)
        if len(data) > limit:
            raise QualificationInvalid("Source/event file grew beyond its bound")
        return data


def _file_digest(path: Path) -> str:
    return hashlib.sha256(_read(path, 32 * 1024 * 1024)).hexdigest()


def experiment_context() -> dict[str, Any] | None:
    if (
        os.getenv("CI") != "true"
        or os.getenv("GITHUB_ACTIONS") != "true"
        or os.getenv("GITHUB_REPOSITORY") != "pytorch/pytorch"
        or os.getenv("GITHUB_HEAD_REF") != EXPERIMENT_BRANCH
        or os.getenv("GITHUB_EVENT_NAME") != "pull_request"
    ):
        return None
    event = json.loads(_read(Path(os.environ["GITHUB_EVENT_PATH"]), 2 * 1024 * 1024))
    head = event["pull_request"]["head"]
    if (
        event["repository"]["full_name"] != "pytorch/pytorch"
        or (head.get("repo") or {}).get("full_name") != EXPERIMENT_FORK
        or head["ref"] != EXPERIMENT_BRANCH
    ):
        return None
    sha = head["sha"]
    if type(sha) is not str or len(sha) != 40 or any(c not in "0123456789abcdef" for c in sha):
        raise QualificationInvalid("Invalid event head SHA")
    return {
        "repository": "pytorch/pytorch",
        "head_repository": EXPERIMENT_FORK,
        "head_ref": EXPERIMENT_BRANCH,
        "head_sha": sha,
        "pull_request": event["pull_request"]["number"],
        "run_id": os.getenv("GITHUB_RUN_ID"),
        "run_attempt": os.getenv("GITHUB_RUN_ATTEMPT"),
        "event_sha": os.getenv("GITHUB_SHA"),
    }


def _runtime_context(context: dict[str, Any], repo: Path) -> str:
    if (
        context["repository"] != "pytorch/pytorch"
        or context["head_repository"] != EXPERIMENT_FORK
        or context["head_ref"] != EXPERIMENT_BRANCH
        or platform.python_implementation() != "CPython"
        or sys.version_info[:2] != (3, 12)
        or "+meta" in sys.version
    ):
        raise QualificationInvalid("Owned ordinary CI stock CPython 3.12 is required")
    checkout = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True, timeout=10
    ).strip()
    if checkout != context["head_sha"]:
        raise QualificationInvalid("Checkout does not match the owned event head")
    return checkout


def _scores(tp: Any) -> tuple[tuple[str, frozenset[str], frozenset[str], int | float], ...]:
    if (
        vars(tp).keys() != {"_original_tests", "_test_scores"}
        or type(tp._original_tests) is not frozenset
        or type(tp._test_scores) is not dict
    ):
        raise QualificationInvalid("Unsupported prioritization shape or instance overrides")
    if len(tp._original_tests) > MAX_TESTS or any(
        type(name) is not str or len(name) > 1024 for name in tp._original_tests
    ):
        raise QualificationInvalid("Unsupported prioritization universe")
    if len(tp._test_scores) > MAX_ENTRIES:
        raise QualificationInvalid("Too many scores")
    rows = []
    members = 0
    for run, score in tp._test_scores.items():
        if type(run) is not TestRun or vars(run).keys() != {"test_file", "_included", "_excluded"}:
            raise QualificationInvalid("Unsupported TestRun type or instance overrides")
        if type(run.test_file) is not str or "::" in run.test_file or len(run.test_file) > 1024:
            raise QualificationInvalid("Unsupported stored test name")
        if type(run._included) is not frozenset or type(run._excluded) is not frozenset:
            raise QualificationInvalid("Unsupported class containers")
        if run._included and run._excluded:
            raise QualificationInvalid("Mixed include/exclude state")
        members += len(run._included) + len(run._excluded)
        if members > MAX_ENTRIES:
            raise QualificationInvalid("Class capture exceeds its cumulative bound")
        if any(type(s) is not str or len(s) > 1024 for s in run._included | run._excluded):
            raise QualificationInvalid("Unsupported class member")
        if type(score) not in (int, float) or not math.isfinite(score):
            raise QualificationInvalid("Unsupported score")
        rows.append((run.test_file, run._included, run._excluded, score))
    return tuple(rows)


def _state(tp: Any) -> dict[str, Any]:
    return {
        "universe": list(tp._original_tests),
        "scores": [[name, list(ins), list(exs), score] for name, ins, exs, score in _scores(tp)],
    }


@dataclass(frozen=True)
class CapturedResult:
    name: str
    trial_mode: bool
    universe: frozenset[str]
    scores: tuple[tuple[str, frozenset[str], frozenset[str], int | float], ...]

    def as_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "trial_mode": self.trial_mode,
            "universe": list(self.universe),
            "scores": [[name, list(ins), list(exs), score] for name, ins, exs, score in self.scores],
        }


@dataclass(frozen=True)
class HeuristicKey:
    name: str
    trial_mode: bool


class SnapshotRecorder:
    def __init__(self, tests: list[str], heuristics: list[Any]) -> None:
        if not tests or len(tests) > MAX_TESTS or len(set(tests)) != len(tests):
            raise QualificationInvalid("Empty, duplicate or oversized universe")
        if any(type(test) is not str or not test or len(test) > 1024 for test in tests):
            raise QualificationInvalid("Unsupported universe member")
        self.tests = tuple(tests)
        self.live_tests = tests
        self.expected = tuple((h.name, h.trial_mode) for h in heuristics)
        if (
            not self.expected
            or len(self.expected) > 16
            or not any(trial for _, trial in self.expected)
            or len({name for name, _ in self.expected}) != len(self.expected)
        ):
            raise QualificationInvalid("A bounded unique registry including trial results is required")
        self.results: list[CapturedResult] = []
        self.live: list[tuple[Any, Any]] = []
        self.errors: list[str] = []
        self.capture_bytes = len(_encoded({"tests": list(tests), "registry": self.expected}))
        self.capture_entries = 0
        self.capture_members = 0

    def capture(self, heuristic: Any, tp: Any) -> None:
        # This observer never drops or alters the normal ranking; publication rejects errors later.
        try:
            if type(heuristic.name) is not str or type(heuristic.trial_mode) is not bool:
                raise QualificationInvalid("Unsupported heuristic metadata")
            result = CapturedResult(heuristic.name, heuristic.trial_mode, tp._original_tests, _scores(tp))
            if result.universe != frozenset(self.tests):
                raise QualificationInvalid("Heuristic universe mismatch")
            members = sum(len(ins) + len(exs) for _, ins, exs, _ in result.scores)
            size = len(_encoded(result.as_json())) + 32
            if (
                self.capture_entries + len(result.scores) > MAX_ENTRIES
                or self.capture_members + members > MAX_ENTRIES
                or self.capture_bytes + size > MAX_BYTES
            ):
                raise QualificationInvalid("Combined capture exceeds its cumulative bound")
            self.results.append(result)
            self.live.append((heuristic, tp))
            self.capture_bytes += size
            self.capture_entries += len(result.scores)
            self.capture_members += members
        except Exception as error:
            self.errors.append(f"{type(error).__name__}: {str(error)[:1000]}")

    def inputs(self) -> dict[str, Any]:
        return {"tests": list(self.tests), "heuristics": [r.as_json() for r in self.results]}

    def check_live(self) -> None:
        if tuple(self.live_tests) != self.tests:
            raise QualificationInvalid("Live universe/order changed")
        if self.errors or tuple((r.name, r.trial_mode) for r in self.results) != self.expected:
            raise QualificationInvalid(f"Incomplete or invalid capture: {self.errors}")
        for captured, (heuristic, tp) in zip(self.results, self.live):
            current = CapturedResult(heuristic.name, heuristic.trial_mode, tp._original_tests, _scores(tp))
            if _encoded(current.as_json()) != _encoded(captured.as_json()):
                raise QualificationInvalid("Live heuristic state changed")


def _restore(result: CapturedResult, module: ModuleType) -> Any:
    # Restore exact received state, not unknown constructor/assignment history. This is untimed.
    tp = object.__new__(module.TestPrioritizations)
    tp._original_tests = result.universe
    tp._test_scores = {}
    for name, ins, exs, score in result.scores:
        run = object.__new__(TestRun)
        run.test_file, run._included, run._excluded = name, ins, exs
        tp._test_scores[run] = score
    if _state(tp) != {"universe": list(result.universe), "scores": result.as_json()["scores"]}:
        raise QualificationInvalid("Restoration changed zero scores or ordered state")
    tp.validate()
    return tp


def _output(tp: Any) -> dict[str, Any]:
    return {"state": _state(tp), "export": tp.to_json(), "order": [run.to_json() for run in tp.get_all_tests()]}


def _prepare(recorder: SnapshotRecorder, module: ModuleType) -> list[tuple[Any, Any]]:
    return [(h, _restore(r, module)) for r, (h, _) in zip(recorder.results, recorder.live)]


def _aggregate(recorder: SnapshotRecorder, module: ModuleType, prepared: list[tuple[Any, Any]]) -> tuple[Any, Any]:
    aggregate = module.AggregatedHeuristics(list(recorder.tests))
    for heuristic, tp in prepared:
        aggregate.add_heuristic_results(heuristic, tp)
    return aggregate, aggregate.get_aggregated_priorities()


def _outputs(aggregate: Any, default: Any) -> dict[str, Any]:
    return {"default": _output(default), "trial": _output(aggregate.get_aggregated_priorities(include_trial=True))}


def measure_component(recorder: SnapshotRecorder, module: ModuleType) -> tuple[dict[str, int], dict[str, Any]]:
    prepared = _prepare(recorder, module)
    before = _digest([_state(tp) for _, tp in prepared])
    cpu_start, wall_start = time.process_time_ns(), time.perf_counter_ns()
    aggregate, default = _aggregate(recorder, module, prepared)
    wall_end, cpu_end = time.perf_counter_ns(), time.process_time_ns()
    outputs = _outputs(aggregate, default)
    if before != _digest([_state(tp) for _, tp in prepared]):
        raise QualificationInvalid("Aggregation mutated a restored input")
    return {"cpu_ns": cpu_end - cpu_start, "wall_ns": wall_end - wall_start}, outputs


@contextmanager
def _module(source: bytes, name: str) -> Iterator[ModuleType]:
    with tempfile.TemporaryDirectory(prefix="td-source-") as directory:
        path = Path(directory) / "source.py"
        path.write_bytes(source)
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise QualificationInvalid("Verified source loader unavailable")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module


@contextmanager
def load_reference(repo: Path) -> Iterator[ModuleType]:
    path = repo / "tools/testing/target_determination/qualification_data/interface_reference.py.txt"
    source = _read(path, MAX_BYTES)
    if hashlib.sha256(source).hexdigest() != REFERENCE_SHA256:
        raise QualificationInvalid("Reference SHA-256 mismatch")
    with _module(source, "_td_reference") as module:
        if module.TestRun is not TestRun:
            raise QualificationInvalid("Reference must share verified stock TestRun")
        yield module


def _code_fingerprint(code: CodeType) -> bytes:
    constants = tuple(_code_fingerprint(c) if isinstance(c, CodeType) else c for c in code.co_consts)
    return marshal.dumps(code.replace(co_filename="<verified-source>", co_consts=constants))


def _verify_test_run(verified: ModuleType) -> str:
    codes = []
    for name, value in vars(verified.TestRun).items():
        expected = value.__func__ if isinstance(value, staticmethod) else value
        if not hasattr(expected, "__code__"):
            continue
        actual = getattr(TestRun, name)
        if not hasattr(actual, "__code__") or _code_fingerprint(actual.__code__) != _code_fingerprint(expected.__code__):
            raise QualificationInvalid("Loaded TestRun code differs from verified source")
        if actual.__globals__.get("TestRun") is not TestRun:
            raise QualificationInvalid("Loaded TestRun global class binding changed")
        codes.append(_code_fingerprint(actual.__code__))
    return hashlib.sha256(b"".join(codes)).hexdigest()


def _live_bindings(recorder: SnapshotRecorder, aggregate: Any, default: Any) -> ModuleType:
    candidate = sys.modules[type(aggregate).__module__]
    if (
        type(default) is not candidate.TestPrioritizations
        or type(aggregate) is not candidate.AggregatedHeuristics
        or any(type(tp) is not candidate.TestPrioritizations for _, tp in recorder.live)
    ):
        raise QualificationInvalid("Unexpected live module/class bindings")
    if [(id(h), id(tp)) for h, tp in aggregate._heuristic_results.items()] != [(id(h), id(tp)) for h, tp in recorder.live]:
        raise QualificationInvalid("Live aggregate did not accept captured results in order")
    return candidate


@contextmanager
def _deadline() -> Iterator[None]:
    timers = ((signal.ITIMER_REAL, signal.SIGALRM), (signal.ITIMER_PROF, signal.SIGPROF))
    if any(signal.getitimer(timer) != (0.0, 0.0) for timer, _ in timers):
        raise QualificationInvalid("Existing timer prevents bounded qualification")
    previous = [(sig, signal.getsignal(sig)) for _, sig in timers]

    def expired(signum: int, frame: Any) -> None:
        raise QualificationInvalid("Post-prediction qualification exceeded five minutes")

    try:
        for timer, sig in timers:
            signal.signal(sig, expired)
            signal.setitimer(timer, MAX_SECONDS)
        yield
    finally:
        for timer, _ in timers:
            signal.setitimer(timer, 0)
        for sig, handler in previous:
            signal.signal(sig, handler)


def run_ci_qualification(recorder: SnapshotRecorder, aggregate: Any, default: Any, context: dict[str, Any]) -> dict[str, Any]:
    report: dict[str, Any] = {
        "status": "invalid", "component": COMPONENT, "context": context,
        "warmup": "one untimed full aggregate/add/default per arm, with trial fidelity checks",
        "excluded": ["heuristic generation/initial constructors", "clone/restore", "trial aggregation", "source/import/IO", "state/output checks"],
        "capture_dimension_bounded_not_wall_bounded": True,
        "whole_TD_or_CI_savings_claim": False, "pairs": [],
    }
    wall_start, cpu_start = time.perf_counter_ns(), time.process_time_ns()
    try:
        with _deadline():
            repo = Path(__file__).resolve().parents[3]
            checkout = _runtime_context(context, repo)
            live_module = _live_bindings(recorder, aggregate, default)
            candidate_source = _read(Path(live_module.__file__), MAX_BYTES)
            stock_source = _read(repo / "tools/testing/test_run.py", MAX_BYTES)
            if hashlib.sha256(candidate_source).hexdigest() != CANDIDATE_SHA256 or hashlib.sha256(stock_source).hexdigest() != TEST_RUN_SHA256:
                raise QualificationInvalid("Candidate/stock source contract changed")
            recorder.check_live()
            inputs = recorder.inputs()
            input_digest = _digest(inputs)
            expected_outputs = _outputs(aggregate, default)
            expected = _encoded(expected_outputs)
            report.update({"inputs": inputs, "input_sha256": input_digest, "checkout_sha": checkout, "python": sys.version, "python_executable_sha256": _file_digest(Path(sys.executable)), "reference_blob_provenance": REFERENCE_BLOB, "reference_sha256": REFERENCE_SHA256, "measured_candidate_sha256": CANDIDATE_SHA256, "test_run_sha256": TEST_RUN_SHA256})
            with _module(stock_source, "_td_verified_stock") as stock, load_reference(repo) as original, _module(candidate_source, "_td_verified_candidate") as candidate:
                runtime_code = _verify_test_run(stock)
                if original.TestRun is not TestRun or candidate.TestRun is not TestRun:
                    raise QualificationInvalid("Measured modules must share verified TestRun")
                modules = {"original": original, "candidate": candidate}
                for module in modules.values():
                    prepared = _prepare(recorder, module)
                    before = _digest([_state(tp) for _, tp in prepared])
                    warm_aggregate, warm_default = _aggregate(recorder, module, prepared)
                    if _encoded(_outputs(warm_aggregate, warm_default)) != expected or before != _digest([_state(tp) for _, tp in prepared]):
                        raise QualificationInvalid("Warmup changed input or live default/trial fidelity")
                for order in PAIRS:
                    pair: dict[str, Any] = {"order": list(order), "arms": []}
                    report["pairs"].append(pair)
                    for arm in order:
                        timings, outputs = measure_component(recorder, modules[arm])
                        pair["arms"].append({"arm": arm, **timings, "outputs_sha256": _digest(outputs)})
                        if _encoded(outputs) != expected:
                            raise QualificationInvalid("Measured output/order differs from live default/trial")
                    recorder.check_live()
                    if input_digest != _digest(recorder.inputs()):
                        raise QualificationInvalid("Captured inputs changed")
                if runtime_code != _verify_test_run(stock):
                    raise QualificationInvalid("Loaded stock code changed during qualification")
            report.update({"outputs": expected_outputs, "runtime_test_run_code_sha256": runtime_code, "pre_post_input_sha256": [input_digest, _digest(recorder.inputs())], "status": "passed"})
            _encoded(report)
        # Return only after both timers/handlers are restored. The entry emits exactly one record.
        report["overall_wall_ns"] = time.perf_counter_ns() - wall_start
        report["overall_cpu_ns"] = time.process_time_ns() - cpu_start
        return report
    except Exception as error:
        report.update({"status": "invalid", "error": f"{type(error).__name__}: {str(error)[:1000]}", "overall_wall_ns": time.perf_counter_ns() - wall_start})
        raise QualificationInvalid(report["error"], report) from error


def emit_result(report: dict[str, Any]) -> None:
    try:
        encoded = _encoded(report)
    except Exception as error:
        fallback = {"status": "invalid", "error": f"Terminal report bound/encoding failure: {type(error).__name__}"}
        print("TD_QUALIFICATION_RESULT_JSON " + json.dumps(fallback))
        raise QualificationInvalid(fallback["error"]) from error
    print("TD_QUALIFICATION_RESULT_JSON " + encoded.decode())
