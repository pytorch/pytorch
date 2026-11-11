"""Runs one PyTorch test file the way test/run_test.py does: from test/, as a script.

    run_test.py <file> <options>                                      runs the file
    run_test.py --list <file> <inputs> <cases> <unused>               lists its cases
    run_test.py --run <file> <case> <inputs> <xml> <log> <status> <unused>
                                                                      runs that case

The last two are the actions of torch_test.bzl. <inputs> names the Python files
among the action's inputs, and the action writes to <unused> those that none of
its processes opened: Bazel does not run it again when only they change.
"""

import fcntl
import os
import re
import signal
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
import zlib
from collections import Counter
from pathlib import Path

# tools/bazel/site is on the path under Bazel, not for the type checker.
# pyrefly: ignore [missing-import]
from sitecustomize import RECORD


# The limit common_utils gives a case that it runs in a process of its own.
CASE_TIMEOUT = 15 * 60
PYTHON = [sys.executable, "-bb"]


def list_cases(file: str, listed: Path) -> None:
    """Writes the file's entry of CASES in tests.bzl (see torch_cases in torch_test.bzl).

    A line for each case: a name that can be a target's, and its pytest id.
    """
    command = [*PYTHON, file, "--use-pytest", "--collect-only", "-q"]
    listing = subprocess.run(command, capture_output=True, timeout=CASE_TIMEOUT)
    lines = listing.stdout.decode(errors="replace").splitlines()
    # pytest lists an id twice when two parameters of a test have one name.
    cases = sorted({line for line in lines if line.startswith(f"test/{file}::")})
    if not cases and listing.returncode != 0:
        # Running the file is what reports why it has no cases to list.
        cases = [f"test/{file}"]
    names = [case.removeprefix("test/").removeprefix(f"{file}::") for case in cases]
    names = [re.sub(r"[^A-Za-z0-9_.-]", "_", name.replace("::", ".")) for name in names]
    count = Counter(names)
    with open(listed, "w") as out:
        out.write(f'    "{file.removesuffix(".py")}": {{\n')
        for name, case in zip(names, cases):
            if len(name) > 200 or count[name] > 1:
                # Too long for a file name, or by now the same as another case's.
                name = f"{name[:190]}~{zlib.crc32(case.encode()):08x}"
            quoted = case.replace("\\", "\\\\").replace('"', '\\"')
            out.write(f'        "{name}": "{quoted}",\n')
        out.write("    },\n")


def error(case: str, message: str) -> ET.Element:
    """A test suite of one failed case, named the way pytest's report names it."""
    # The type checker's environment has no pytest.
    # pyrefly: ignore [missing-import]
    from _pytest.junitxml import mangle_test_address

    *classes, name = mangle_test_address(case)
    suite = ET.Element("testsuite", name="run_test.py")
    testcase = ET.SubElement(suite, "testcase", classname=".".join(classes), name=name)
    ET.SubElement(testcase, "error", message=message)
    return suite


def run_case(file: str, case: str, result: Path, log: Path, status: Path) -> None:
    """Runs the case and reports it in `result`, even when its process dies."""
    scratch = Path(tempfile.mkdtemp())
    report = scratch / "report.xml"
    (scratch / "case").write_text(case)
    select = [f"--nodeid-file={scratch / 'case'}", f"--junit-xml-reruns={report}"]
    command = [*PYTHON, file, "--use-pytest", *select]
    with open(log, "wb") as out:
        test = subprocess.Popen(command, stdout=out, stderr=out, start_new_session=True)
        try:
            ending = f"exit code {test.wait(timeout=CASE_TIMEOUT)}"
        except subprocess.TimeoutExpired:
            os.killpg(test.pid, signal.SIGKILL)
            test.wait()
            ending = f"no result after {CASE_TIMEOUT} s"
    suites = ET.Element("testsuites")
    # pytest writes its report once it is through the case.
    if report.exists():
        suites.extend(ET.parse(report).getroot())
    if not report.exists() or ending not in ("exit code 0", "exit code 1"):
        suites.append(error(case, ending))
    ET.ElementTree(suites).write(result)
    status.write_text("0" if ending == "exit code 0" else "1")


def unopened(inputs: list[str], start: Path) -> list[str]:
    """The inputs that sitecustomize.py did not see any process open."""
    opened = set(Path(RECORD).read_text().splitlines())
    # A process that resolves links opens a file under the name of its source.
    sources = {os.path.realpath(start / path): path for path in inputs}
    used = {sources.get(path) or os.path.relpath(path, start) for path in opened}
    # Every Python process opens some file: an empty record is one nothing kept.
    return [path for path in inputs if path not in used] if opened else []


if __name__ == "__main__":
    args = sys.argv[1:]
    start = Path.cwd()
    # Bazel's Python launcher exports PYTHONSAFEPATH, but tests import the
    # helper modules that sit next to them.
    os.environ.pop("PYTHONSAFEPATH", None)
    # An activated venv has its interpreter and its wheels' scripts on PATH, and
    # tests launch python, python3 and ninja by name. Bazel's venv has only a
    # python3, and it keeps each wheel in its own directory, scripts in bin/.
    python = Path(tempfile.mkdtemp(), "python")
    python.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    python.chmod(0o755)
    wheels = [Path(p).parent for p in sys.path if Path(p).name == "site-packages"]
    scripts = [str(wheel / "bin") for wheel in wheels if (wheel / "bin").is_dir()]
    paths = [str(python.parent), os.path.dirname(sys.executable), *scripts]
    os.environ["PATH"] = os.pathsep.join([*paths, os.environ["PATH"]])
    os.chdir(Path(os.environ["RUNFILES_DIR"], "_main/test"))
    if args[0] not in ("--list", "--run"):
        os.execv(sys.executable, [*PYTHON, *args])
    # Bazel gives a test action an empty home directory, and a build action none.
    os.environ["HOME"] = tempfile.mkdtemp()
    # From here on, sitecustomize.py records what every process opens. In a sandbox
    # the record is this action's alone. Without one, the actions running at once
    # share it: the one holding the lock goes by a record that names too much, and
    # the others report nothing as unused.
    record = os.open(RECORD, os.O_WRONLY | os.O_CREAT)
    try:
        fcntl.flock(record, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.ftruncate(record, 0)
        alone = True
    except BlockingIOError:
        alone = False
    mode, file, *rest = args
    if mode == "--list":
        inputs, listed, unused = (start / path for path in rest)
        list_cases(file, listed)
    else:
        case, *rest = rest
        inputs, result, log, status, unused = (start / path for path in rest)
        run_case(file, case, result, log, status)
    files = inputs.read_text().split() if alone else []
    unused.write_text("".join(f"{path}\n" for path in unopened(files, start)))
