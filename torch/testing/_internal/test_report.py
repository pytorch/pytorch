"""The test report that feeds the ``tests.*`` ClickHouse tables, one file per test
process. ``tools/testing/hud/report.xml`` documents the format.

``common_utils.run_tests`` and ``test/run_test.py`` register this module as a pytest
plugin: ``-p torch.testing._internal.test_report --report-dir=<dir>``, and the
writer names its file after the directory, ``<dir>/<name>-<random>.report.xml``,
so every process of a test file writes its own. Attempts
are appended to the file as they finish, so a process that dies leaves every
finished attempt on disk; ``run_test.py`` then appends the in-flight test as a
synthetic ``crashed`` or ``timed_out`` attempt and closes the file
(``finalize_report``). A report without the closing tag is a process that died
before ``run_test.py`` could do that, and its complete attempts still count.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
import sysconfig
import time
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING
from xml.sax.saxutils import quoteattr

import torch


if TYPE_CHECKING:
    from typing import IO

    from _pytest.config import Config
    from _pytest.config.argparsing import Parser
    from _pytest.main import Session
    from _pytest.reports import TestReport

REPORT_SUFFIX = ".report.xml"
FOOTER = "</report>\n"
# Where run_test.py reads the stepcurrent plugin's files (test/conftest.py); the
# report's path and in-flight test are published next to them.
STEPCURRENT_CACHE_DIR = "cache/stepcurrent"
_MODELS = re.compile(
    r"\b(A10G|A100|H100|H200|B200|B300|L4|L40S|L40|T4|V100|MI\d{3}[A-Z]?|M\d+|Max \d{4})\b"
)
# XML 1.0 forbids most control characters, and lone surrogates cannot be encoded.
_NOT_XML = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]")


@dataclass
class Environment:
    """What tests.environments identifies a process by (section 4.1 of the proposal),
    plus report-only facts for debugging."""

    os: str
    os_version: str
    cpu_architecture: str
    cpu_capability: str
    python_version: str
    cc_compiler: str
    cc_compiler_version: str
    accelerator: str
    accelerator_version: str
    device_name: str
    device_count: int
    flags: dict[str, str]
    properties: dict[str, str]

    def identity(self) -> dict[str, str]:
        names = (
            "os", "os_version", "cpu_architecture", "cpu_capability", "python_version",
            "cc_compiler", "cc_compiler_version", "accelerator", "accelerator_version",
            "device_name", "device_count",
        )  # fmt: skip
        return {name: str(getattr(self, name)) for name in names}


def capture_environment() -> Environment:
    from torch.testing._internal.common_utils import TestEnvironment

    os_name, os_version, os_release = _os()
    cc_compiler, cc_version, cc_version_full = _compiler(torch.__config__.show())
    accelerator, acc_version, acc_version_full = _accelerator()
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
        "cc_compiler_version_full": cc_version_full,
        "accelerator_version_full": acc_version_full,
        "device_name_raw": raw_name,
        "device_memory_mib": memory,
        "driver_version": driver,
        "host_memory_mib": _host_memory_mib(),
        "build_environment": os.environ.get("BUILD_ENVIRONMENT", ""),
        "test_config": os.environ.get("TEST_CONFIG", ""),
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
        device_name=normalize_device_name(raw_name) if raw_name else "",
        device_count=device_count,
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


def normalize_device_name(raw: str) -> str:
    """The model behind a vendor's device string: "NVIDIA H100 80GB HBM3" is h100,
    "AMD Instinct MI300X" is mi300x, "Apple M1 Pro" is m1. Unknown strings are
    kept whole, lower-cased with dashes; the raw string is in the report."""
    match = _MODELS.search(raw)
    if match:
        return match.group(1).replace(" ", "").lower()
    return re.sub(r"\s+", "-", raw.strip().lower())


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


def _timestamp(seconds: float) -> str:
    stamp = datetime.fromtimestamp(seconds, timezone.utc)
    return stamp.isoformat(timespec="milliseconds")


def _xml_text(text: str) -> str:
    return _NOT_XML.sub("\ufffd", text)


def attempt_element(
    nodeid: str,
    rerun_number: int,
    outcome: str,
    started: float,
    ended: float,
    message: str = "",
    text: str = "",
    phase: str = "",
) -> str:
    file, suite, case_name = identity(nodeid)
    element = ET.Element(
        "attempt",
        file=file,
        suite=suite,
        case_name=case_name,
        rerun_number=str(rerun_number),
        outcome=outcome,
        started_at=_timestamp(started),
        ended_at=_timestamp(ended),
    )
    if phase:
        element.set("phase", phase)
    if message:
        element.set("message", _xml_text(message))
    if text:
        element.text = _xml_text(text)
    return "  " + ET.tostring(element, encoding="unicode") + "\n"


def finalize_report(
    path: str, inflight: dict[str, Any] | None, outcome: str, message: str
) -> None:
    """Close the report of a process that died. ``inflight`` is the
    ``report_inflight`` record the writer published for the test that was running;
    it becomes a synthetic attempt with ``outcome`` crashed or timed_out. A footer
    pytest wrote while handling the timeout's SIGINT is replaced."""
    footer = FOOTER.encode()
    with open(path, "rb+") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - len(footer)))
        if f.read() == footer:
            size -= len(footer)
            f.truncate(size)
        f.seek(size)
        if inflight:
            attempt = attempt_element(
                inflight["nodeid"],
                inflight["rerun_number"],
                outcome,
                inflight["started_at"],
                time.time(),
                message=message,
            )
            f.write(attempt.encode())
        f.write(footer)


@dataclass
class _Attempt:
    started: float
    ended: float = 0.0
    failed_phase: str = ""
    skipped: bool = False
    wasxfail: bool = False
    message: str = ""
    text: str = ""

    def outcome(self) -> str:
        if self.failed_phase:
            return "failed" if self.failed_phase == "call" else "error"
        if self.skipped:
            return "xfailed" if self.wasxfail else "skipped"
        return "xpassed" if self.wasxfail else "passed"


class ReportWriter:
    """pytest plugin: streams one <attempt> per execution of a test to the report."""

    def __init__(self, path: str, config: Config) -> None:
        # run_test.py reads the published path from another working directory.
        self.path = os.path.abspath(path)
        self.file: IO[str] | None = None
        self.attempts: dict[str, _Attempt] = {}
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
            "github_workflow_job_id": os.environ.get("JOB_ID", ""),
        }
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self.file = open(self.path, "w", encoding="utf-8")  # noqa: SIM115
        attrs = "".join(f" {k}={quoteattr(v)}" for k, v in context.items() if v)
        self.file.write(f'<?xml version="1.0" encoding="utf-8"?>\n<report{attrs}>\n')
        env_element = ET.Element("environment", environment.identity())
        for name, value in sorted(environment.flags.items()):
            ET.SubElement(env_element, "flag", name=name, value=_xml_text(value))
        properties = ET.Element("properties")
        for name, value in environment.properties.items():
            ET.SubElement(properties, "property", name=name, value=_xml_text(value))
        for element in (env_element, properties):
            ET.indent(element, space="  ", level=1)
            self.file.write("  " + ET.tostring(element, encoding="unicode") + "\n")
        self.file.flush()
        self._publish("report_path", self.path)

    def pytest_runtest_logstart(self, nodeid: str, location: Any) -> None:
        started = time.time()
        self.attempts[nodeid] = _Attempt(started=started)
        rerun = self.rerun_numbers[nodeid]
        inflight = {"nodeid": nodeid, "started_at": started, "rerun_number": rerun}
        self._publish("report_inflight", inflight)

    def pytest_runtest_logreport(self, report: TestReport) -> None:
        attempt = self.attempts.get(report.nodeid)
        if attempt is None:
            return
        if report.when == "setup":
            attempt.started = report.start
        attempt.ended = report.stop
        subtest = getattr(report, "context", None) is not None
        if report.failed or report.outcome == "rerun":
            if not attempt.failed_phase:
                attempt.failed_phase = report.when
                attempt.message, attempt.text = _failure_text(report)
        elif report.skipped and not subtest and not attempt.skipped:
            from _pytest.terminal import _get_raw_skip_reason

            attempt.skipped = True
            attempt.wasxfail = hasattr(report, "wasxfail")
            attempt.message = _get_raw_skip_reason(report)
        elif report.when == "call" and not subtest:
            attempt.wasxfail = hasattr(report, "wasxfail")
        # pytest-rerunfailures logs the failing phase as "rerun" and skips the rest
        # of that attempt, so it ends here; otherwise the attempt ends at teardown.
        if report.outcome == "rerun" or report.when == "teardown":
            self._finish(report.nodeid, attempt)

    def _finish(self, nodeid: str, attempt: _Attempt) -> None:
        del self.attempts[nodeid]
        phase = attempt.failed_phase if attempt.failed_phase not in ("", "call") else ""
        if self.file is not None:
            self.file.write(
                attempt_element(
                    nodeid,
                    self.rerun_numbers[nodeid],
                    attempt.outcome(),
                    attempt.started,
                    attempt.ended,
                    message=attempt.message,
                    text=attempt.text,
                    phase=phase,
                )
            )
            self.file.flush()
        self.rerun_numbers[nodeid] += 1

    def pytest_sessionfinish(self, session: Session) -> None:
        if self.file is not None:
            self.file.write(FOOTER)
            self.file.close()
            self.file = None
        # An attempt still open here was interrupted; run_test.py decides whether
        # that was its timeout and records it.
        if not self.attempts:
            self._publish("report_inflight", None)


def _failure_text(report: TestReport) -> tuple[str, str]:
    crash = getattr(report.longrepr, "reprcrash", None)
    text = str(report.longrepr)
    return (crash.message if crash is not None else text), text


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
