"""The test report that feeds the ``tests.*`` ClickHouse tables, one JSON Lines file
per test process: a ``report`` line (schema version, run context, environment,
flags, properties), then one ``run`` line per test execution.
``tools/testing/hud/test_run_report.md`` documents the format.

``common_utils.run_tests`` and ``test/run_test.py`` register this module as a pytest
plugin: ``-p torch.testing._internal.test_report --report-dir=<dir>``, and the
writer names its file after the directory, ``<dir>/<name>-<random>.jsonl``,
so every process of a test file writes its own. Runs are appended to the file as
they finish, so a process that dies leaves every finished line on disk;
``run_test.py`` then appends the in-flight test as a
synthetic ``crashed`` or ``timed_out`` run (``finalize_report``). Each line is a
complete record, so there is no closing marker: a line the process was cut off
writing does not parse and is skipped, and every other line counts.
"""

from __future__ import annotations

import inspect
import json
import os
import platform
import re
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
    from typing import IO

    from _pytest.config import Config
    from _pytest.config.argparsing import Parser
    from _pytest.main import Session
    from _pytest.reports import TestReport

REPORT_SUFFIX = ".jsonl"
# Bumped on a breaking change to the lines; the ingester reads it from the report line.
SCHEMA_VERSION = 1
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
    cc_compiler, cc_version, _ = _compiler(torch.__config__.show())
    accelerator, acc_version, _ = _accelerator()
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


def _compiler(config: str) -> tuple[str, str, str]:
    # clang defines __GNUC__ too, so torch.__config__.show() prints a GCC line for
    # clang builds; the clang and MSVC lines take precedence.
    for name, pattern in (("clang", r"clang"), ("msvc", r"MSVC"), ("gcc", r"GCC")):
        match = re.search(rf"^  - {pattern} (\S+)", config, re.MULTILINE)
        if match:
            version = match.group(1)
            # MSVC prints _MSC_FULL_VER, e.g. 194134120 for version 19.41.
            major = version[:2] if name == "msvc" else version.split(".")[0]
            return name, major, version
    return "", "", ""


def _accelerator() -> tuple[str, str, str]:
    if torch.version.cuda:
        return "cuda", ".".join(torch.version.cuda.split(".")[:2]), torch.version.cuda
    if torch.version.hip:
        return "rocm", ".".join(torch.version.hip.split(".")[:2]), torch.version.hip
    xpu = str(getattr(torch.version, "xpu", "") or "")
    if xpu:
        # The SYCL version packed as major * 10000 + minor * 100 + patch.
        version = xpu
        if xpu.isdigit() and len(xpu) == 8:
            version = f"{xpu[:4]}.{int(xpu[4:6])}"
        return "xpu", version, xpu
    if torch.backends.mps.is_built():
        return "mps", "", ""
    return "cpu", "", ""


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
            cmd = ["sysctl", "-n", "machdep.cpu.brand_string"]
            return subprocess.check_output(cmd, text=True).strip(), "", ""
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
        nvml.nvmlDeviceGetName(handle, name, 96)
        nvml.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory))
        nvml.nvmlSystemGetDriverVersion(driver, 80)
        return name.value.decode(), str(memory.total >> 20), driver.value.decode()
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
    import json

    output = subprocess.check_output(["xpu-smi", "discovery", "-j"], text=True)
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


def _line(record: dict[str, Any]) -> str:
    # json.dumps escapes control characters and lone surrogates, so any message
    # or traceback round-trips without the replacement XML needs.
    return json.dumps(record, separators=(",", ":")) + "\n"


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
        return getattr(inspect.unwrap(function), "__name__", fallback)
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


def finalize_report(
    path: str, inflight: dict[str, Any] | None, outcome: str, message: str
) -> None:
    """Record the test a dead process was running. ``inflight`` is the
    ``report_inflight`` record the writer published; it becomes a synthetic run
    with ``outcome`` crashed or timed_out. A line the process was cut off writing is
    ended first, so it stays one unparsable line instead of merging with this one."""
    if not inflight:
        return
    with open(path, "ab+") as f:
        size = f.seek(0, os.SEEK_END)
        if size:
            f.seek(size - 1)
            if f.read(1) != b"\n":
                f.write(b"\n")
        record = run_record(
            inflight["nodeid"],
            inflight["rerun_number"],
            outcome,
            inflight["started_at"],
            time.time(),
            outcome_summary=message,
            declared_case_name=inflight.get("declared_case_name"),
        )
        f.write(_line(record).encode())


@dataclass
class _Run:
    started: float
    declared_case_name: str
    ended: float = 0.0
    failed_phase: str = ""
    skipped: bool = False
    wasxfail: bool = False
    outcome_summary: str = ""

    def outcome(self) -> str:
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
        self.file: IO[str] | None = None
        self.runs: dict[str, _Run] = {}
        self.rerun_numbers: Counter[str] = Counter()
        # Published for run_test.py (see finalize_report) when the stepcurrent
        # plugin of test/conftest.py is active.
        key = config.getoption("stepcurrent", default=None)
        self.cache = getattr(config, "cache", None) if key else None
        self.cache_dir = f"{STEPCURRENT_CACHE_DIR}/{key}"

    def _publish(self, name: str, value: Any) -> None:
        if self.cache is not None:
            self.cache.set(f"{self.cache_dir}/{name}", value)

    def pytest_sessionstart(self, session: Session) -> None:
        environment = capture_environment()
        context = {
            "repo": os.environ.get("GITHUB_REPOSITORY", ""),
            "github_workflow_job_id": _job_id(),
        }
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self.file = open(self.path, "w", encoding="utf-8")  # noqa: SIM115
        environment_record = environment.identity()
        report = {
            "type": "report",
            "schema_version": SCHEMA_VERSION,
            **context,
            "environment": environment_record,
            "flags": dict(sorted(environment.flags.items())),
            "properties": environment.properties,
        }
        self.file.write(_line(report))
        self.file.flush()
        self._publish("report_path", self.path)

    def pytest_runtest_logstart(self, nodeid: str, location: Any) -> None:
        started = time.time()
        declared_case_name = _fallback_declared_case_name(nodeid)
        self.runs[nodeid] = _Run(started=started, declared_case_name=declared_case_name)
        rerun = self.rerun_numbers[nodeid]
        inflight = {
            "nodeid": nodeid,
            "declared_case_name": declared_case_name,
            "started_at": started,
            "rerun_number": rerun,
        }
        self._publish("report_inflight", inflight)

    def pytest_runtest_logreport(self, report: TestReport) -> None:
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
                inflight = {
                    "nodeid": report.nodeid,
                    "declared_case_name": declared_case_name,
                    "started_at": run.started,
                    "rerun_number": self.rerun_numbers[report.nodeid],
                }
                self._publish("report_inflight", inflight)
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
def pytest_runtest_makereport(item: Any, call: Any):
    outcome = yield
    report = outcome.get_result()
    report._test_run_report_declared_case_name = _item_declared_case_name(item)


def pytest_addoption(parser: Parser) -> None:
    parser.addoption(
        "--report-dir",
        action="store",
        default=None,
        metavar="dir",
        help="write the test report that feeds the tests.* tables into this directory",
    )


def pytest_configure(config: Config) -> None:
    directory = config.getoption("report_dir")
    # xdist workers forward their reports to the controller, which writes; a
    # collect-only session (run_tests listing the tests of a --subprocess file)
    # runs nothing.
    worker = hasattr(config, "workerinput")
    if directory and not worker and not config.getoption("collectonly"):
        name = os.path.basename(os.path.normpath(directory))
        path = os.path.join(directory, f"{name}-{os.urandom(8).hex()}{REPORT_SUFFIX}")
        config.pluginmanager.register(ReportWriter(path, config), "test_report_writer")
