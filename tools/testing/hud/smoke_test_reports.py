"""Run test files of every kind and show the test reports they produce.

Clears the repo-root test-reports/ and run_test.py's stepcurrent cache, runs
the sets below through test/run_test.py with CI=1 (which turns the reports on),
then parses every *.report.xml and prints one line per report. The junit files
under test/test-reports are left alone.

    python tools/testing/hud/smoke_test_reports.py
    python tools/testing/hud/smoke_test_reports.py --only python cpp
    python tools/testing/hud/smoke_test_reports.py --only python -- --dynamo

Arguments after "--" are passed to every run_test.py invocation.

The sets were picked from main's CUDA jobs (tests.all_test_runs, Oct 2026) as
files that run in seconds and still cover the shapes the writer must handle.
"""

from __future__ import annotations

import argparse
import collections
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
REPORTS_DIR = REPO_ROOT / "test-reports"
STEPCURRENT_CACHE = REPO_ROOT / ".pytest_cache" / "v" / "cache" / "stepcurrent"
RUNS = {
    # Python test files, one process each unless noted:
    #   test_type_info             plain unittest classes
    #   test_complex               device-generic, CPU and CUDA classes in one process
    #   test_subclass              xfail and parametrize ids with brackets
    #   torch_np/test_dtype        29 skipped and 29 xfailed of 44
    #   functorch/test_vmap_registrations  1,575 attempts in a few seconds, 150 xfailed
    #   test_accelerator           subTest and xfail on the accelerator
    #   test_cuda_nvml_based_avail --subprocess handler: one process and report per test
    #   test_ci_sanity_check_fail  fails on purpose: reruns, then run_test.py's retry
    #                              launches (--rs), three reports of failed attempts
    #   distributed/...            MultiProcessTestCase files spawning 2 and 4 ranks
    "python": [
        "-i",
        "test_type_info",
        "test_complex",
        "test_subclass",
        "torch_np/test_dtype",
        "functorch/test_vmap_registrations",
        "test_accelerator",
        "test_cuda_nvml_based_avail",
        "test_ci_sanity_check_fail",
        "distributed/tensor/test_api",
        "distributed/algorithms/ddp_comm_hooks/test_ddp_hooks",
    ],
    # The same file under a harness flag: a second environment (flags) for the
    # same test ids. Other small files fail under inductor or need triton.
    "inductor": ["--inductor", "-i", "test_type_info"],
    # gtest binaries through pytest-cpp with xdist workers, including typed
    # (OptionalTest/0.Empty) and value-parameterized (BFloat16RNETest/0) names.
    "cpp": [
        "--cpp",
        "-i",
        "cpp/atest",
        "cpp/c10_intrusive_ptr_test",
        "cpp/c10_optional_test",
        "cpp/c10_bfloat16_test",
    ],
}


def run_test(args: list[str]) -> int:
    cmd = [sys.executable, "test/run_test.py", *args]
    print(f"$ CI=1 {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=REPO_ROOT, env={**os.environ, "CI": "1"}).returncode


def describe(report: Path) -> str:
    closed = report.read_bytes().rstrip().endswith(b"</report>")
    try:
        root = ET.parse(report).getroot()
    except ET.ParseError as e:
        return f"PARSE ERROR {e}"
    env = root.find("environment")
    if env is None:
        return "NO <environment> ELEMENT"
    device = f"{env.get('accelerator')} {env.get('device_name') or '-'}"
    device += f" x{env.get('device_count')}"
    flags = " ".join(f"{f.get('name')}={f.get('value')}" for f in env.iter("flag"))
    outcomes = collections.Counter(a.get("outcome") for a in root.iter("attempt"))
    counts = " ".join(f"{k}={v}" for k, v in sorted(outcomes.items()))
    reruns = sum(a.get("rerun_number") != "0" for a in root.iter("attempt"))
    state = "closed" if closed else "OPEN"
    summary = f"{state:6} {sum(outcomes.values()):5} attempts {reruns:3} reruns"
    return f"{summary}  {counts or '-':34} {device}  flags: {flags or '-'}"


def main() -> int:
    formatter = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=formatter)
    parser.add_argument("--only", nargs="+", choices=sorted(RUNS), default=sorted(RUNS))
    parser.add_argument("run_test_args", nargs=argparse.REMAINDER, help="after --")
    args = parser.parse_args()
    extra = args.run_test_args
    if extra[:1] == ["--"]:
        extra = extra[1:]

    for path in (REPORTS_DIR, STEPCURRENT_CACHE):
        shutil.rmtree(path, ignore_errors=True)
    print(f"cleared {REPORTS_DIR} and {STEPCURRENT_CACHE}")

    exit_codes = {name: run_test([*RUNS[name], *extra]) for name in args.only}

    reports = sorted(REPORTS_DIR.rglob("*.report.xml"))
    print(f"\n{len(reports)} reports under {REPORTS_DIR}:")
    for report in reports:
        print(f"  {report.relative_to(REPORTS_DIR)}\n      {describe(report)}")
    print(f"\nrun_test.py exit codes: {exit_codes}")
    return 1 if any(exit_codes.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
