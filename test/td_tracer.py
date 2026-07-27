from __future__ import annotations

import collections
import importlib.util
import json
import logging
import os
import platform
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, TYPE_CHECKING

import pytest
from filelock import FileLock


if TYPE_CHECKING:
    from types import CodeType

    from _pytest.config import Config
    from _pytest.config.argparsing import Parser


TD_TRACER_OPTION_NAME = "td_tracer"
TD_TRACER_RUN_ID_ENV = "PYTORCH_TD_TRACER_RUN_ID"
TD_TRACER_CURRENT_TEST_ENV = "PYTORCH_TD_TRACER_CURRENT_TEST"
TD_TRACER_DEFER_MERGE_ENV = "PYTORCH_TD_TRACER_DEFER_MERGE"

_SCHEMA_VERSION = 4
_TOOL_ID = 4
_WORKER_OUTPUT_KEY = "td_tracer_fragment"
_WORKER_RUN_ID_KEY = "td_tracer_run_id"
_WORKER_PARTICIPANT_ID_KEY = "td_tracer_participant_id"
_OUTER_FAILURE_MARKER = "outer-failure"
_SUCCESSFUL_EXIT_CODES = {
    int(pytest.ExitCode.OK),
    int(pytest.ExitCode.NO_TESTS_COLLECTED),
}


def td_tracer_arguments(parser: Parser) -> None:
    group = parser.getgroup(TD_TRACER_OPTION_NAME)
    group.addoption(
        "--td-tracer",
        action="store",
        dest=TD_TRACER_OPTION_NAME,
        default=None,
        metavar="path",
        help="Record per-test executed torch files to the given JSON file.",
    )


def ensure_td_tracer_run_id() -> str:
    run_id = os.environ.get(TD_TRACER_RUN_ID_ENV)
    if run_id is None:
        run_id = uuid.uuid4().hex
        os.environ[TD_TRACER_RUN_ID_ENV] = run_id
    return run_id


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(data, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _run_dir(output_path: Path, run_id: str) -> Path:
    return Path(f"{output_path}.td-tracer.d") / run_id


def _load_fragment(path: Path, run_id: str) -> dict[str, Any]:
    with path.open(encoding="utf-8") as input_file:
        fragment = json.load(input_file)
    if fragment.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError(f"Unsupported TD tracer schema in {path}")
    if fragment.get("run_id") != run_id:
        raise ValueError(f"Unexpected TD tracer run ID in {path}")
    if fragment.get("state") != "finished":
        raise ValueError(f"Unfinished TD tracer fragment in {path}")
    return fragment


def _merge_td_tracer_fragments_locked(output_path: Path, run_id: str) -> dict[str, Any]:
    fragment_dir = _run_dir(output_path, run_id)
    fragments = [
        _load_fragment(path, run_id) for path in sorted(fragment_dir.glob("*.json"))
    ]
    running = sorted(path.stem for path in fragment_dir.glob("*.running"))
    outer_failed = (fragment_dir / _OUTER_FAILURE_MARKER).exists()

    coverage_by_test: dict[str, set[str]] = collections.defaultdict(set)
    environments: dict[str, dict[str, Any]] = {}
    revisions: set[str] = set()
    participants = []

    for fragment in fragments:
        for test, paths in fragment["coverage_by_test"].items():
            coverage_by_test[test].update(paths)
        environment = fragment["environment"]
        environments[json.dumps(environment, sort_keys=True)] = environment
        if environment["revision"] is not None:
            revisions.add(environment["revision"])
        participants.append(
            {
                "complete": fragment["complete"],
                "exit_status": fragment["exit_status"],
                "participant_id": fragment["participant_id"],
                "pid": fragment["pid"],
                "session_id": fragment["session_id"],
                "worker_id": fragment["worker_id"],
            }
        )

    complete = (
        bool(fragments)
        and not running
        and not outer_failed
        and len(revisions) <= 1
        and all(fragment["complete"] for fragment in fragments)
    )
    successful = (
        bool(fragments)
        and not outer_failed
        and all(
            fragment["exit_status"] in _SUCCESSFUL_EXIT_CODES for fragment in fragments
        )
    )
    merged = {
        "schema_version": _SCHEMA_VERSION,
        "run_id": run_id,
        "complete": complete,
        "successful": successful,
        "usable": complete and successful,
        "running_participants": running,
        "participants": sorted(
            participants, key=lambda participant: participant["participant_id"]
        ),
        "environments": [environments[key] for key in sorted(environments)],
        "coverage_by_test": {
            test: sorted(paths) for test, paths in sorted(coverage_by_test.items())
        },
    }
    _atomic_write_json(output_path, merged)
    return merged


def merge_td_tracer_fragments(
    output_path: str | os.PathLike[str], run_id: str
) -> dict[str, Any]:
    output_path = Path(output_path).resolve()
    with FileLock(f"{output_path}.lock"):
        return _merge_td_tracer_fragments_locked(output_path, run_id)


def finalize_td_tracer(
    output_path: str | os.PathLike[str], run_id: str, test_successful: bool
) -> dict[str, Any]:
    output_path = Path(output_path).resolve()
    with FileLock(f"{output_path}.lock"):
        if not test_successful:
            _atomic_write_json(
                _run_dir(output_path, run_id) / _OUTER_FAILURE_MARKER,
                {"schema_version": _SCHEMA_VERSION, "run_id": run_id},
            )
        return _merge_td_tracer_fragments_locked(output_path, run_id)


class TDTracer:
    def __init__(self, config: Config) -> None:
        if not hasattr(sys, "monitoring"):
            raise pytest.UsageError("--td-tracer requires Python 3.12 or newer")

        torch_spec = importlib.util.find_spec("torch")
        if torch_spec is None or torch_spec.origin is None:
            raise pytest.UsageError("--td-tracer could not locate the torch package")

        self._config = config
        self._out_path = Path(config.getoption(TD_TRACER_OPTION_NAME)).resolve()
        self._defer_merge = os.environ.get(TD_TRACER_DEFER_MERGE_ENV) == "1"
        self._torch_dir = Path(torch_spec.origin).resolve().parent
        self._repo_root = self._torch_dir.parent
        self._python_roots = (self._torch_dir,)
        self._code_paths: dict[CodeType, str | None] = {}
        self._current_test: str | None = None
        self._parent_test = os.environ.get(TD_TRACER_CURRENT_TEST_ENV)
        self._collection_target: Path | None = None
        self._collecting = False
        self._shared_collection_coverage: set[str] = set()
        self._collection_coverage: dict[Path, set[str]] = collections.defaultdict(set)
        self._coverage_by_test: dict[str, set[str]] = collections.defaultdict(set)
        if self._parent_test is not None:
            self._coverage_by_test.setdefault(self._parent_test, set())
        self._fixture_coverage: dict[object, set[str]] = collections.defaultdict(set)
        self._fixture_consumers: dict[object, set[str]] = collections.defaultdict(set)
        self._active_fixture_setups: list[object] = []
        self._active_fixture_teardowns: set[object] = set()
        self._remaining_tests: set[str] = set()

        worker_input = getattr(config, "workerinput", None)
        self._is_xdist_worker = worker_input is not None
        self._run_id = (
            worker_input[_WORKER_RUN_ID_KEY]
            if self._is_xdist_worker
            else ensure_td_tracer_run_id()
        )
        self._session_id = os.environ.get("PYTEST_XDIST_TESTRUNUID", uuid.uuid4().hex)
        self._participant_id = (
            worker_input[_WORKER_PARTICIPANT_ID_KEY]
            if self._is_xdist_worker
            else uuid.uuid4().hex
        )
        self._worker_id = (
            worker_input.get("workerid", "worker")
            if self._is_xdist_worker
            else "controller"
        )
        self._fragment_dir = _run_dir(self._out_path, self._run_id)
        self._fragment_path = self._fragment_dir / f"{self._participant_id}.json"
        self._running_path = self._fragment_dir / f"{self._participant_id}.running"

        owner = sys.monitoring.get_tool(_TOOL_ID)
        if owner is not None:
            raise pytest.UsageError(
                f"--td-tracer requires monitoring tool ID {_TOOL_ID}, owned by {owner!r}"
            )
        sys.monitoring.use_tool_id(_TOOL_ID, TD_TRACER_OPTION_NAME)
        self._tid = _TOOL_ID
        try:
            sys.monitoring.register_callback(
                self._tid,
                sys.monitoring.events.PY_START,
                self._on_py_start,
            )
            sys.monitoring.set_events(self._tid, sys.monitoring.events.PY_START)
            if not self._is_xdist_worker:
                self._write_running_marker()
        except BaseException:
            self._release_tool()
            raise

    def _write_running_marker(self) -> None:
        _atomic_write_json(
            self._running_path,
            {
                "schema_version": _SCHEMA_VERSION,
                "run_id": self._run_id,
                "participant_id": self._participant_id,
                "pid": os.getpid(),
                "state": "running",
            },
        )
        self._merge_fragments()

    def _merge_fragments(self) -> None:
        if not self._defer_merge:
            merge_td_tracer_fragments(self._out_path, self._run_id)

    def _release_tool(self) -> None:
        try:
            sys.monitoring.set_events(self._tid, 0)
        finally:
            try:
                sys.monitoring.register_callback(
                    self._tid, sys.monitoring.events.PY_START, None
                )
            finally:
                sys.monitoring.free_tool_id(self._tid)

    def _relative_python_path(self, filename: str) -> str | None:
        try:
            path = Path(filename).resolve(strict=True)
        except (OSError, RuntimeError):
            return None
        if path.suffix != ".py":
            return None
        if not any(path.is_relative_to(root) for root in self._python_roots):
            return None
        return path.relative_to(self._repo_root).as_posix()

    def _record_fixture_consumers(self, item) -> None:
        fixture_info = getattr(item, "_fixtureinfo", None)
        if fixture_info is not None:
            for fixture_defs in fixture_info.name2fixturedefs.values():
                for fixture_def in fixture_defs or ():
                    self._fixture_consumers[fixture_def].add(item.nodeid)

        request = getattr(item, "_request", None)
        for fixture_def in getattr(request, "_fixture_defs", {}).values():
            self._fixture_consumers[fixture_def].add(item.nodeid)

    def _apply_shared_coverage(self) -> None:
        for fixture_def, paths in self._fixture_coverage.items():
            for test in self._fixture_consumers[fixture_def]:
                self._coverage_by_test[test].update(paths)

    def _environment(self) -> dict[str, Any]:
        torch = sys.modules.get("torch")
        torch_version = getattr(torch, "version", None)
        return {
            "build_environment": os.environ.get("BUILD_ENVIRONMENT"),
            "machine": platform.machine(),
            "platform": sys.platform,
            "python": platform.python_version(),
            "test_config": os.environ.get("TEST_CONFIG"),
            "revision": self._revision(),
            "torch": getattr(torch, "__version__", None),
            "cuda": getattr(torch_version, "cuda", None),
            "hip": getattr(torch_version, "hip", None),
        }

    def _revision(self) -> str | None:
        revision = os.environ.get("GITHUB_SHA")
        if revision is None:
            try:
                result = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=self._repo_root,
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    revision = result.stdout.strip()
            except OSError:
                pass
        torch = sys.modules.get("torch")
        torch_version = getattr(torch, "version", None)
        return revision or getattr(torch_version, "git_version", None)

    def _fragment(self, exitstatus: int) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "run_id": self._run_id,
            "session_id": self._session_id,
            "participant_id": self._participant_id,
            "worker_id": self._worker_id,
            "pid": os.getpid(),
            "state": "finished",
            "exit_status": int(exitstatus),
            "complete": not self._remaining_tests,
            "environment": self._environment(),
            "coverage_by_test": {
                test: sorted(paths)
                for test, paths in sorted(self._coverage_by_test.items())
            },
        }

    def pytest_sessionstart(self, session: pytest.Session) -> None:
        logging.getLogger(__name__).info("TD tracer output: %s", self._out_path)

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_collection(self, session: pytest.Session):
        self._collecting = True
        try:
            yield
        finally:
            self._collecting = False

    @pytest.hookimpl(hookwrapper=True)
    def pytest_make_collect_report(self, collector):
        previous_target = self._collection_target
        collector_path = getattr(collector, "path", None)
        if collector_path is not None and Path(collector_path).is_file():
            self._collection_target = Path(collector_path).resolve()
        try:
            yield
        finally:
            self._collection_target = previous_target

    @pytest.hookimpl(trylast=True)
    def pytest_collection_finish(self, session: pytest.Session) -> None:
        for item in session.items:
            self._remaining_tests.add(item.nodeid)
            self._coverage_by_test.setdefault(item.nodeid, set())
            item_path = Path(item.path).resolve()
            self._coverage_by_test[item.nodeid].update(
                self._collection_coverage[item_path]
            )
            self._record_fixture_consumers(item)

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_fixture_setup(self, fixturedef, request):
        def stop_teardown_trace():
            self._active_fixture_teardowns.discard(fixturedef)

        def start_teardown_trace():
            self._active_fixture_teardowns.add(fixturedef)

        fixturedef.addfinalizer(stop_teardown_trace)
        self._active_fixture_setups.append(fixturedef)
        try:
            yield
        finally:
            self._active_fixture_setups.pop()
            fixturedef.addfinalizer(start_teardown_trace)

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_setup(self, item):
        try:
            yield
        finally:
            self._record_fixture_consumers(item)

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_protocol(self, item, nextitem):
        previous_test = self._current_test
        previous_env_test = os.environ.get(TD_TRACER_CURRENT_TEST_ENV)
        self._coverage_by_test[item.nodeid].update(self._shared_collection_coverage)
        self._current_test = item.nodeid
        os.environ[TD_TRACER_CURRENT_TEST_ENV] = item.nodeid
        try:
            yield
        finally:
            self._current_test = previous_test
            if previous_env_test is None:
                os.environ.pop(TD_TRACER_CURRENT_TEST_ENV, None)
            else:
                os.environ[TD_TRACER_CURRENT_TEST_ENV] = previous_env_test

    def pytest_runtest_logfinish(self, nodeid, location) -> None:
        self._remaining_tests.discard(nodeid)

    @pytest.hookimpl(optionalhook=True)
    def pytest_configure_node(self, node) -> None:
        participant_id = uuid.uuid4().hex
        node.workerinput[_WORKER_RUN_ID_KEY] = self._run_id
        node.workerinput[_WORKER_PARTICIPANT_ID_KEY] = participant_id
        running_path = self._fragment_dir / f"{participant_id}.running"
        _atomic_write_json(
            running_path,
            {
                "schema_version": _SCHEMA_VERSION,
                "run_id": self._run_id,
                "participant_id": participant_id,
                "state": "running",
            },
        )
        self._merge_fragments()

    @pytest.hookimpl(optionalhook=True)
    def pytest_testnodedown(self, node, error) -> None:
        fragment = node.workeroutput.get(_WORKER_OUTPUT_KEY)
        if fragment is None:
            return
        participant_id = fragment["participant_id"]
        _atomic_write_json(self._fragment_dir / f"{participant_id}.json", fragment)
        (self._fragment_dir / f"{participant_id}.running").unlink(missing_ok=True)
        self._merge_fragments()

    def pytest_sessionfinish(self, session, exitstatus):
        self._apply_shared_coverage()
        fragment = self._fragment(exitstatus)
        if self._is_xdist_worker:
            self._config.workeroutput[_WORKER_OUTPUT_KEY] = fragment
            return

        _atomic_write_json(self._fragment_path, fragment)
        self._running_path.unlink(missing_ok=True)
        self._merge_fragments()

    def pytest_unconfigure(self, config: Config) -> None:
        self._release_tool()

    def _on_py_start(self, code, instruction_offset):
        if code not in self._code_paths:
            self._code_paths[code] = self._relative_python_path(code.co_filename)
        relative_path = self._code_paths[code]
        if relative_path is None:
            return

        destinations = []
        for fixture_def in self._active_fixture_setups:
            destinations.append(self._fixture_coverage[fixture_def])
        for fixture_def in self._active_fixture_teardowns:
            destinations.append(self._fixture_coverage[fixture_def])
        if not destinations:
            if self._current_test is not None:
                destinations.append(self._coverage_by_test[self._current_test])
            if self._collection_target is not None:
                destinations.append(self._collection_coverage[self._collection_target])
            elif self._collecting and self._current_test is None:
                destinations.append(self._shared_collection_coverage)
        if self._parent_test is not None:
            destinations.append(self._coverage_by_test[self._parent_test])

        added = False
        for destination in destinations:
            if relative_path not in destination:
                destination.add(relative_path)
                added = True
        if added:
            logging.getLogger(__name__).debug(
                "PY_START %s -> %s -> %s",
                self._current_test,
                code.co_name,
                relative_path,
            )
