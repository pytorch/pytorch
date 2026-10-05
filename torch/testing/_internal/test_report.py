"""The test report that feeds the ``tests.*`` ClickHouse tables, one JSON Lines file
per test process: a ``report`` line (schema version, run context, environment,
flags, properties), then one ``run`` line per test execution.
``tools/testing/hud/test_run_report.md`` documents the format.

``common_utils.run_tests`` and ``test/run_test.py`` register this module as a pytest
plugin: ``-p torch.testing._internal.test_report --report-dir=<dir>``, and the
writer names its file after the directory, ``<dir>/<name>-<random>.jsonl``,
so every process of a test file writes its own. Runs are appended to the file as
they finish, so a process that dies leaves every finished line on disk, and the
test it was running becomes a synthetic ``crashed`` or ``timed_out`` run: from
``run_test.py`` (``finalize_report``), the ``--subprocess`` parent
(``record_dead_subprocess``), or the xdist controller. Each line is a complete
record, so there is no closing marker: a line the process was cut off writing
does not parse and is skipped, and every other line counts. Writing the report
never changes a test's outcome: any error turns it off and the tests run on.
"""

from __future__ import annotations

import contextlib
import inspect
import json
import os
import platform
import re
import signal
import subprocess
import sys
import sysconfig
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

import pytest

import torch


if TYPE_CHECKING:
    from collections.abc import Generator, Iterator
    from typing import IO

    from _pytest.config import Config
    from _pytest.config.argparsing import Parser
    from _pytest.main import Session
    from _pytest.reports import TestReport

REPORT_SUFFIX = ".jsonl"
# Bumped on a breaking change to the lines; the ingester reads it from the report line.
SCHEMA_VERSION = "0.1"
# Where run_test.py reads the stepcurrent plugin's files (test/conftest.py); the
# report's path and in-flight test are published next to them.
STEPCURRENT_CACHE_DIR = "cache/stepcurrent"


@dataclass
class Environment:
    """What tests.environments identifies a process by."""

    os: str
    os_version: str
    cpu_architecture: str
    cpu_capability: str
    python_version: str
    cc_compiler: str
    cc_compiler_version: str
    accelerator: str
    accelerator_version: str
    device_count: int
    device_name: str
    flags: dict[str, str]
    properties: dict[str, str]

    def identity(self) -> dict[str, Any]:
        names = (
            "os", "os_version", "cpu_architecture", "cpu_capability", "python_version",
            "cc_compiler", "cc_compiler_version", "accelerator", "accelerator_version",
            "device_count", "device_name",
        )  # fmt: skip
        return {name: getattr(self, name) for name in names}


def capture_environment() -> Environment:
    from torch.testing._internal.common_utils import TestEnvironment

    os_name, os_version, os_release = _os()
    cc_compiler, cc_version = _compiler(torch.__config__.show())
    accelerator, acc_version = _accelerator()
    try:
        device_count = torch.accelerator.device_count()
    except Exception:
        device_count = 0
    raw_name, memory, driver = _device(accelerator) if device_count else ("", "", "")
    cpu_capability = torch.backends.cpu.get_cpu_capability().lower().replace(" ", "")
    if cpu_capability == "avx512" and torch.cpu._is_amx_tile_supported():
        cpu_capability = "amx"
    properties = {
        "torch_version": torch.__version__,
        "os_release": os_release,
        "device_memory_mib": memory,
        "driver_version": driver,
        "host_memory_mib": _host_memory_mib(),
        "build_environment": os.environ.get("BUILD_ENVIRONMENT", ""),
        "test_config": os.environ.get("TEST_CONFIG", ""),
        "runner_name": os.environ.get("RUNNER_NAME", ""),
    }
    return Environment(
        os=os_name,
        os_version=os_version,
        cpu_architecture=_cpu_architecture(),
        cpu_capability=cpu_capability,
        python_version=_python_version(),
        cc_compiler=cc_compiler,
        cc_compiler_version=cc_version,
        accelerator=accelerator,
        accelerator_version=acc_version,
        device_count=device_count,
        device_name=raw_name,
        flags=dict(TestEnvironment.env_var_values),
        properties={k: v for k, v in properties.items() if v},
    )


def _os() -> tuple[str, str, str]:
    system = platform.system()
    if system == "Linux":
        try:
            release = platform.freedesktop_os_release()
        except OSError:
            release = {}
        return "linux", release.get("VERSION_ID", ""), release.get("PRETTY_NAME", "")
    if system == "Darwin":
        version = platform.mac_ver()[0]
        return "macos", ".".join(version.split(".")[:2]), version
    if system == "Windows":
        version = platform.version()
        return "windows", ".".join(version.split(".")[:2]), version
    return system.lower(), "", ""


def _cpu_architecture() -> str:
    machine = platform.machine().lower()
    return {"amd64": "x86_64", "arm64": "aarch64"}.get(machine, machine)


def _python_version() -> str:
    free_threaded = "t" if sysconfig.get_config_var("Py_GIL_DISABLED") else ""
    return f"{sys.version_info.major}.{sys.version_info.minor}{free_threaded}"


def _compiler(config: str) -> tuple[str, str]:
    # clang defines __GNUC__ too, so torch.__config__.show() prints a GCC line for
    # clang builds; the clang and MSVC lines take precedence.
    for name, pattern in (("clang", r"clang"), ("msvc", r"MSVC"), ("gcc", r"GCC")):
        match = re.search(rf"^  - {pattern} (\S+)", config, re.MULTILINE)
        if match:
            version = match.group(1)
            # MSVC prints _MSC_FULL_VER, e.g. 194134120 for version 19.41.
            major = version[:2] if name == "msvc" else version.split(".")[0]
            return name, major
    return "", ""


def _accelerator() -> tuple[str, str]:
    if torch.version.cuda:
        return "cuda", ".".join(torch.version.cuda.split(".")[:2])
    if torch.version.hip:
        return "rocm", ".".join(torch.version.hip.split(".")[:2])
    xpu = str(getattr(torch.version, "xpu", "") or "")
    if xpu:
        # The SYCL version packed as major * 10000 + minor * 100 + patch.
        version = xpu
        if xpu.isdigit() and len(xpu) == 8:
            version = f"{xpu[:4]}.{int(xpu[4:6])}"
        return "xpu", version
    if torch.backends.mps.is_built():
        return "mps", ""
    return "cpu", ""


def _device(accelerator: str) -> tuple[str, str, str]:
    """Raw model name, memory in MiB and driver version of the first visible device,
    read without creating a device context, which would poison fork."""
    try:
        if accelerator == "cuda":
            return _nvml_device()
        if accelerator == "rocm":
            return _amdsmi_device()
        if accelerator == "xpu":
            return _xpu_smi_device()
        if accelerator == "mps":
            return torch.backends.mps.get_name(), "", ""
    except Exception:
        pass
    return "", "", ""


def _first_visible_index() -> int:
    visible = torch.cuda._parse_visible_devices()
    return visible[0] if visible and isinstance(visible[0], int) else 0


def _nvml_device() -> tuple[str, str, str]:
    import ctypes

    class Memory(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in ("total", "free", "used")]

    nvml = ctypes.CDLL("nvml.dll" if sys.platform == "win32" else "libnvidia-ml.so.1")
    if nvml.nvmlInit() != 0:
        return "", "", ""
    try:
        handle = ctypes.c_void_p()
        index = ctypes.c_uint(_first_visible_index())
        if nvml.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(handle)) != 0:
            return "", "", ""
        name = ctypes.create_string_buffer(96)
        driver = ctypes.create_string_buffer(80)
        memory = Memory()
        name_value = (
            name.value.decode() if nvml.nvmlDeviceGetName(handle, name, 96) == 0 else ""
        )
        memory_value = (
            str(memory.total >> 20)
            if nvml.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory)) == 0
            else ""
        )
        driver_value = (
            driver.value.decode()
            if nvml.nvmlSystemGetDriverVersion(driver, 80) == 0
            else ""
        )
        return name_value, memory_value, driver_value
    finally:
        nvml.nvmlShutdown()


def _amdsmi_device() -> tuple[str, str, str]:
    import amdsmi  # type: ignore[import-not-found]

    amdsmi.amdsmi_init()
    try:
        handle = amdsmi.amdsmi_get_processor_handles()[_first_visible_index()]
        name = amdsmi.amdsmi_get_gpu_asic_info(handle)["market_name"]
        memory = driver = ""
        try:
            vram_type = amdsmi.AmdSmiMemoryType.VRAM
            vram = amdsmi.amdsmi_get_gpu_memory_total(handle, vram_type)
            memory = str(int(vram) >> 20)
            driver = amdsmi.amdsmi_get_gpu_driver_info(handle)["driver_version"]
        except Exception:
            pass
        return name, memory, driver
    finally:
        amdsmi.amdsmi_shut_down()


def _xpu_smi_device() -> tuple[str, str, str]:
    output = subprocess.check_output(
        ["xpu-smi", "discovery", "-j"], text=True, timeout=30
    )
    discovery = json.loads(output)
    return discovery["device_list"][0]["device_name"], "", ""


def _host_memory_mib() -> str:
    try:
        limit = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if limit.isdigit():
            return str(int(limit) >> 20)
    except OSError:
        pass
    try:
        return str((os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) >> 20)
    except (AttributeError, OSError, ValueError):
        return ""


def identity(nodeid: str) -> tuple[str, str, str]:
    """(file, suite, case_name) of a pytest node id, as tests.tests stores them.

    Python: ``test/test_torch.py::TestTorch::test_add[1]`` is the launched file
    (the classes test_jit.py imports from jit/ keep test_jit.py), the innermost
    class and the test name with its parameters. pytest-cpp:
    ``build/bin/test_api::ModulesTest.Linear`` is the binary as run_test.py names
    it, the gtest suite and the test name.
    """
    path, _, rest = nodeid.partition("::")
    if not path.endswith(".py"):
        suite, _, case_name = rest.partition(".")
        return f"cpp/{Path(path).stem}", suite, case_name
    head, bracket, params = rest.partition("[")
    parts = head.split("::")
    suite = parts[-2] if len(parts) > 1 else ""
    return path, suite, parts[-1] + bracket + params


def _job_id() -> int:
    # JOB_ID is set by the workflow's job-id lookup, which can fail.
    job = os.environ.get("JOB_ID", "")
    return int(job) if job.isdigit() else 0


def _timestamp(seconds: float) -> int:
    return int(seconds * 1000)


def _json_safe(value: Any) -> Any:
    # Escape lone surrogates before text writers encode the JSON as UTF-8.
    if isinstance(value, str):
        return value.encode("utf-8", "backslashreplace").decode()
    if isinstance(value, dict):
        return {_json_safe(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _line(record: dict[str, Any]) -> str:
    return json.dumps(_json_safe(record), separators=(",", ":")) + "\n"


def _report_record() -> dict[str, Any]:
    environment = capture_environment()
    return {
        "type": "report",
        "schema_version": SCHEMA_VERSION,
        "repo": os.environ.get("GITHUB_REPOSITORY", ""),
        "github_workflow_job_id": _job_id(),
        "environment": environment.identity(),
        "flags": dict(sorted(environment.flags.items())),
        "properties": environment.properties,
    }


def _fallback_declared_case_name(nodeid: str) -> str:
    file, _, case_name = identity(nodeid)
    name = case_name.partition("[")[0]
    return re.sub(r"/\d+$", "", name) if file.startswith("cpp/") else name


def _item_declared_case_name(item: Any) -> str:
    fallback = _fallback_declared_case_name(item.nodeid)
    try:
        if not item.nodeid.partition("::")[0].endswith(".py"):
            return fallback
        function = item.obj
        if function is None:
            return fallback
        name = getattr(inspect.unwrap(function), "__name__", fallback)
        _, _, case_name = identity(item.nodeid)
        if case_name == name or case_name.startswith((name + "_", name + "[")):
            return name
        return fallback
    except Exception:
        return fallback


def run_record(
    nodeid: str,
    rerun_number: int,
    outcome: str,
    started: float,
    ended: float,
    outcome_summary: str = "",
    declared_case_name: str | None = None,
) -> dict[str, Any]:
    file, suite, case_name = identity(nodeid)
    return {
        "type": "run",
        "schema_version": SCHEMA_VERSION,
        "file": file,
        "suite": suite,
        "case_name": case_name,
        "declared_case_name": declared_case_name
        or _fallback_declared_case_name(nodeid),
        "rerun_number": rerun_number,
        "outcome": outcome,
        "outcome_summary": outcome_summary,
        "started_at": _timestamp(started),
        "ended_at": _timestamp(ended),
        "properties": {},
    }


# Reporting must never change a test's outcome: the first error anywhere in this
# module turns the report off for the rest of the process, and the tests run on.
_disabled = False
_writer: ReportWriter | None = None


def _disable(error: Exception) -> None:
    global _disabled
    if _disabled:
        return
    _disabled = True
    try:
        print(f"test_report: disabled after error: {error!r}", file=sys.stderr)
        if _writer is not None and _writer.file is not None:
            report_file, _writer.file = _writer.file, None
            report_file.close()
    except Exception:
        pass


@contextlib.contextmanager
def _guard() -> Iterator[None]:
    try:
        yield
    except Exception as error:
        _disable(error)


def _report_path(directory: str) -> str:
    name = os.path.basename(os.path.normpath(directory))
    return os.path.join(directory, f"{name}-{os.urandom(8).hex()}{REPORT_SUFFIX}")


def _synthetic_run_record(
    inflight: dict[str, Any], outcome: str, message: str
) -> dict[str, Any]:
    return run_record(
        inflight["nodeid"],
        inflight["rerun_number"],
        outcome,
        inflight["started_at"],
        time.time(),
        outcome_summary=message,
        declared_case_name=inflight.get("declared_case_name"),
    )


def finalize_report(
    path: str, inflight: dict[str, Any] | None, outcome: str, message: str
) -> None:
    """Record the test a dead process was running. ``inflight`` is the
    ``report_inflight`` record the writer published; it becomes a synthetic run
    with ``outcome`` crashed or timed_out, unless the process died after writing
    that run. A line the process was cut off writing is ended first, so it stays
    one unparsable line instead of merging with this one."""
    if not inflight:
        return
    with open(path, "ab+") as f:
        f.seek(0)
        content = f.read()
        for line in reversed(content.splitlines()):
            try:
                last = json.loads(line)
            except ValueError:
                continue
            if last.get("type") == "run":
                keys = ("file", "suite", "case_name", "rerun_number")
                done = tuple(last.get(key) for key in keys)
                if done == (*identity(inflight["nodeid"]), inflight["rerun_number"]):
                    return
                break
        if content and not content.endswith(b"\n"):
            f.write(b"\n")
        f.write(_line(_synthetic_run_record(inflight, outcome, message)).encode())


def _rootdir_relative(path: str) -> str:
    # pytest's rootdir is the nearest directory with pytest.ini, the repo root.
    absolute = Path(path).resolve()
    for parent in absolute.parents:
        if (parent / "pytest.ini").is_file():
            return absolute.relative_to(parent).as_posix()
    return path


def record_dead_subprocess(
    directory: str, nodeid: str, exit_code: int, started: float
) -> None:
    """Record the final attempt of a ``--subprocess`` child (``common_utils.run_tests``)
    that crashed or timed out, in a report file of the parent's own. The parent
    has a cwd-relative node id; the child's own lines use the rootdir-relative one."""
    if _disabled or not (exit_code < 0 or exit_code == 124):
        return
    with _guard():
        path, sep, rest = nodeid.partition("::")
        signals = {s.value: s.name for s in signal.Signals}
        name = f" ({signals[-exit_code]})" if -exit_code in signals else ""
        inflight = {
            "nodeid": _rootdir_relative(path) + sep + rest,
            "rerun_number": 0,
            "started_at": started,
        }
        os.makedirs(directory, exist_ok=True)
        with open(_report_path(directory), "w", encoding="utf-8", newline="\n") as f:
            f.write(_line(_report_record()))
            f.write(
                _line(
                    _synthetic_run_record(
                        inflight,
                        "timed_out" if exit_code == 124 else "crashed",
                        f"the test process exited with code {exit_code}{name}",
                    )
                )
            )


@dataclass
class _Run:
    started: float
    declared_case_name: str
    ended: float = 0.0
    failed_phase: str = ""
    skipped: bool = False
    wasxfail: bool = False
    crashed: bool = False
    outcome_summary: str = ""

    def outcome(self) -> str:
        if self.crashed:
            return "crashed"
        if self.failed_phase:
            return "failed" if self.failed_phase == "call" else "error"
        if self.skipped:
            return "xfailed" if self.wasxfail else "skipped"
        return "xpassed" if self.wasxfail else "passed"


class ReportWriter:
    """pytest plugin: writes one run line per execution of a test to the report."""

    def __init__(self, path: str, config: Config) -> None:
        # run_test.py reads the published path from another working directory.
        self.path = os.path.abspath(path)
        self.config = config
        self.file: IO[str] | None = None
        self.runs: dict[str, _Run] = {}
        self.rerun_numbers: Counter[str] = Counter()
        self.declared_case_names: dict[str, str] = {}
        # Only run_test_retries reads the published in-flight test, and it never
        # runs xdist, so an xdist controller publishes nothing.
        self.xdist = bool(config.getoption("numprocesses", default=None))
        self.cache: Any = None
        self.cache_dir = ""

    def _publish(self, name: str, value: Any) -> None:
        if self.cache is not None:
            self.cache.set(f"{self.cache_dir}/{name}", value)

    def pytest_sessionstart(self, session: Session) -> None:
        if _disabled:
            return
        with _guard():
            # Read here, not at configure: test/conftest.py copies --scs and --rs
            # into this option in its own pytest_configure.
            key = self.config.getoption("stepcurrent", default=None)
            if key and not self.xdist:
                self.cache = getattr(self.config, "cache", None)
                self.cache_dir = f"{STEPCURRENT_CACHE_DIR}/{key}"
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            self.file = open(self.path, "w", encoding="utf-8", newline="\n")  # noqa: SIM115
            self.file.write(_line(_report_record()))
            self.file.flush()
            self._publish("report_path", self.path)

    def pytest_collection_finish(self, session: Session) -> None:
        if _disabled:
            return
        with _guard():
            # Empty on an xdist controller; the workers send the names instead
            # (pytest_runtest_makereport).
            self.declared_case_names = {
                item.nodeid: _item_declared_case_name(item) for item in session.items
            }

    def pytest_runtest_logstart(self, nodeid: str, location: Any) -> None:
        if _disabled:
            return
        with _guard():
            started = time.time()
            declared_case_name = self.declared_case_names.get(
                nodeid
            ) or _fallback_declared_case_name(nodeid)
            self.runs[nodeid] = _Run(started, declared_case_name)
            inflight = {
                "nodeid": nodeid,
                "declared_case_name": declared_case_name,
                "started_at": started,
                "rerun_number": self.rerun_numbers[nodeid],
            }
            self._publish("report_inflight", inflight)

    # First, so test/conftest.py's LogXMLReruns can't rewrite a skip's longrepr
    # before it is read.
    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_logreport(self, report: TestReport) -> None:
        if _disabled:
            return
        with _guard():
            self._logreport(report)

    def _logreport(self, report: TestReport) -> None:
        if report.when not in ("setup", "call", "teardown"):
            # xdist's report for a test whose worker crashed: when "???", no
            # times, and nothing else follows for this run.
            run = self.runs.get(report.nodeid) or _Run(
                time.time(), _fallback_declared_case_name(report.nodeid)
            )
            self.runs[report.nodeid] = run
            run.ended = time.time()
            run.crashed = True
            run.outcome_summary = str(report.longrepr)
            self._finish(report.nodeid, run)
            return
        run = self.runs.get(report.nodeid)
        if run is None:
            return
        if report.when == "setup":
            run.started = report.start
            declared_case_name = getattr(
                report, "_test_run_report_declared_case_name", None
            )
            if declared_case_name:
                run.declared_case_name = declared_case_name
        run.ended = report.stop
        subtest = getattr(report, "context", None) is not None
        if report.failed or report.outcome == "rerun":
            if not run.failed_phase:
                run.failed_phase = report.when
                run.outcome_summary = _failure_summary(report)
        elif report.skipped and not subtest and not run.skipped:
            from _pytest.terminal import _get_raw_skip_reason

            run.skipped = True
            run.wasxfail = hasattr(report, "wasxfail")
            run.outcome_summary = _get_raw_skip_reason(report)
        elif report.when == "call" and not subtest:
            run.wasxfail = hasattr(report, "wasxfail")
        # pytest-rerunfailures logs the failing phase as "rerun" and skips the rest
        # of that run, so it ends here; otherwise the run ends at teardown.
        if report.outcome == "rerun" or report.when == "teardown":
            self._finish(report.nodeid, run)

    def _finish(self, nodeid: str, run: _Run) -> None:
        del self.runs[nodeid]
        if self.file is not None:
            record = run_record(
                nodeid,
                self.rerun_numbers[nodeid],
                run.outcome(),
                run.started,
                run.ended,
                outcome_summary=run.outcome_summary,
                declared_case_name=run.declared_case_name,
            )
            self.file.write(_line(record))
            self.file.flush()
        self.rerun_numbers[nodeid] += 1

    def pytest_sessionfinish(self, session: Session) -> None:
        if _disabled:
            return
        with _guard():
            if self.file is not None:
                self.file.close()
                self.file = None
            # A run still open here was interrupted; run_test.py decides whether
            # that was its timeout and records it.
            if not self.runs:
                self._publish("report_inflight", None)


def _failure_summary(report: TestReport) -> str:
    crash = getattr(report.longrepr, "reprcrash", None)
    return crash.message if crash is not None else str(report.longrepr)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: Any, call: Any) -> Generator[None, Any, None]:
    # Runs where the test runs, xdist workers included, and the name travels on
    # the report to the process that writes. Never touches the wrapped result
    # when the hook itself raised.
    outcome = yield
    if _disabled or call.when != "setup" or outcome.excinfo is not None:
        return
    with _guard():
        name = _item_declared_case_name(item)
        outcome.get_result()._test_run_report_declared_case_name = name


def pytest_addoption(parser: Parser) -> None:
    parser.addoption(
        "--report-dir",
        action="store",
        default=None,
        metavar="dir",
        help="write the test report that feeds the tests.* tables into this directory",
    )


def pytest_configure(config: Config) -> None:
    global _writer
    with _guard():
        directory = config.getoption("report_dir")
        # xdist workers forward their reports to the controller, which writes; a
        # collect-only session (run_tests listing the tests of a --subprocess file)
        # runs nothing.
        worker = hasattr(config, "workerinput")
        if directory and not worker and not config.getoption("collectonly"):
            _writer = ReportWriter(_report_path(directory), config)
            config.pluginmanager.register(_writer, "test_report_writer")
