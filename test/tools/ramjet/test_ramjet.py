# Owner(s): ["module: devx"]

import contextlib
import dataclasses
import getpass
import gzip
import importlib.util
import io
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest import mock

from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    skipIfWindows,
    TestCase,
)
from torch.testing._internal.tci import machine


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from tools.ramjet import collect, common, inputs, lineage, logger, sync, times
from tools.testing.test_selections import THRESHOLD


HAS_MODAL = importlib.util.find_spec("modal") is not None
if HAS_MODAL:
    import tools.ramjet.__main__ as cli
    from tools.ramjet import build, progress, run, test, worker

sys.path.remove(str(REPO_ROOT))

GIT_ENV = {
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_AUTHOR_NAME": "ramjet",
    "GIT_AUTHOR_EMAIL": "ramjet@example.com",
    "GIT_COMMITTER_NAME": "ramjet",
    "GIT_COMMITTER_EMAIL": "ramjet@example.com",
}
RATES = {
    "cpu_hour_cost": Decimal("0.0473"),
    "cpu_hour_cost_sandbox": Decimal("0.1419"),
    "mem_gib_hour_cost": Decimal("0.008"),
    "mem_gib_hour_cost_sandbox": Decimal("0.024"),
    "gpu_hour_cost_l4": Decimal("0.80"),
    "gpu_hour_cost_a10g": Decimal("1.10"),
    "gpu_hour_cost_a100_40gb": Decimal("2.10"),
    "gpu_hour_cost_a100_80gb": Decimal("2.50"),
    "gpu_hour_cost_b200": Decimal("6.25"),
    "gpu_hour_cost_h100": Decimal("3.95"),
    "gpu_hour_cost_rtx6000": Decimal("3.03"),
}
APP = "https://modal.com/id/ap-1"  # @lint-ignore: no app has this address
CMAKE_LISTS = """\
cmake_minimum_required(VERSION 3.18)
project(demo C)
add_library(demo demo.c)
"""
# Stands in for a test file: what collect reads is what its launcher prints, what
# tci's plugin writes of the tests it deselects, and what ramjet's writes of those
# marked @serialTest: the second test, and one that tci deselects. The second
# test's parametrization is how the file was launched: its arguments and its
# PYTHONPATH
TEST_FILE = """\
import json
import os
import sys

print("Running 2 items in this shard")
print("test/test_file.py::TestFile::test_first")
launched = [*sys.argv[1:], os.environ.get("PYTHONPATH", "no PYTHONPATH")]
deselected = launched.pop(5).removeprefix("--tci-deselected=")
marked = launched.pop(7).removeprefix("--ramjet-serial=")
second = "test/test_file.py::TestFile::test_second"
print(f"{second}[{' '.join(launched)}]")
unmet = {"[two]": "@tci.needs(tci.gpus >= 2)", "_rocm": "@tci.only(tci.rocm)"}
with open(deselected, "w") as file:
    json.dump({second + test: constraint for test, constraint in unmet.items()}, file)
with open(marked, "w") as file:
    file.write(f"{second}[{' '.join(launched)}]\\n{second}[two]")
"""
# Stands in for the ci-metrics wrapper: in a directory outside the repository, it
# keeps each query it is sent, and it answers all of them with the same rows. The
# queries come at once and the shell's printf writes a line at a time, so each is
# added to the others in one write
WRAPPER = """\
#!/bin/sh
for query; do :; done
printf '%s\\n' "$query" > {0}/query.$$
cat {0}/query.$$ >> {0}/queries
cat {0}/rows.json
"""
L4 = {"accelerator": "cuda", "gpus": 1, "device": "NVIDIA L4", "gpu_memory": 22.5}
# Stands in for a test file: its test's name is how it runs, which is the same in a
# process of its own and in a fork of collect.py's, where common_utils is imported
# before the file runs
HOW = """\
import os
import sys

import torch.testing._internal.common_utils as common_utils

main = sys.modules["__main__"]
how = [__name__, sys.argv[0], sys.argv[1]]
how += [os.path.relpath(path) for path in (__file__, sys.path[0])]
how += [str(vars(main) is globals()), str(common_utils.__main__ is main)]
print("test/test_how.py::" + " ".join(how))
"""
# A test file with tests marked @serialTest, as pytest sees them
MARKED = """\
import pytest


@pytest.mark.serial
def test_alone(): ...


def test_plain(): ...


class TestClass:
    @pytest.mark.serial
    def test_alone(self): ...
"""
# Stands in for a test file that sets the environment before it imports what reads
# it: its test's name is what common_utils read
ENV = """\
import os

os.environ["PYTORCH_TEST_WITH_SLOW"] = "1"
from torch.testing._internal import common_utils

print(f"test/test_env.py::slow tests {common_utils.TEST_WITH_SLOW}")
"""
# Stands in for run_test.py in a container: its report says how it was launched, with
# how many test processes at once and how many threads each, and the test process it
# leaves behind outlives it unless its whole session is ended.
# As torchci does while tests run, it leaves a report with a line for a run that
# ended and one still being written, and the tests in flight: one of this shard and
# one of another. And it says which test processes it starts and which have ended,
# as run_test.py does: test_a's two and nn/test_b's one start, test_a's first ends
# and starts again
RUN_TEST = """\
import json
import os
import subprocess
import sys
import time
from pathlib import Path

options = dict(arg.split("=", 1) for arg in sys.argv[1:] if "=" in arg)
launched = {"argv": sys.argv[1:], "cwd": os.getcwd()}
launched["PYTHONPATH"] = os.environ.get("PYTHONPATH")
sizes = ("_NUM_TEST_PROCS", "_NUM_THREADS")
launched["sized"] = {k: v for k, v in os.environ.items() if k.endswith(sizes)}
launched["ids"] = Path(options["--nodeid-file"]).read_text()
launched["child"] = subprocess.Popen(["sleep", "600"]).pid
reports = Path(options["--save-torchci-reports"])
reports.mkdir()
(reports / "test_a-1.report.jsonl").write_text(json.dumps(launched))
test = {"file": "test/test_a.py", "suite": "A"}
ended = json.dumps({"type": "run", **test, "case_name": "test_one"})
(reports / "test_a-2.report.jsonl").write_text(ended + "\\n" + ended[:9])
three = {"file": "test/nn/test_b.py", "suite": "B", "case_name": "test_three"}
flights = [("a", reports, {**test, "case_name": "test_two"})]
for key, where, flying in [*flights, ("nn/b", reports.parent / "other", three)]:
    cache = Path(".pytest_cache/v/cache/stepcurrent", key)
    cache.mkdir(parents=True, exist_ok=True)
    flight = {"path": f"{where}/a.jsonl", "test": flying}
    (cache / "report_inflight").write_text(json.dumps(flight))
said = ["Running parallel tests on 3 processes", "Running test_a 1/2"]
said += ["Running test_a 2/2", "Running nn/test_b 1/1", "Finished test_a 1/2"]
for line in [*said, "Running test_a 1/2"]:
    print(f"{line} ... [2026-11-11 17:04:43][7.5]", file=sys.stderr, flush=True)
print("ran", flush=True)
time.sleep(600 if "--hang" in sys.argv else 0.5)
sys.exit(3)
"""


def run_line(nodeid: str, outcome: str, said: str = "", at: int = 0) -> str:
    """A line of a torchci report: one run of a test, at a time, and what it said."""
    path, bracket, parameters = nodeid.partition("[")  # which may hold a :: too
    file, *classes, case = path.split("::")
    case += bracket + parameters
    test = {"file": file, "suite": classes[-1] if classes else "", "case_name": case}
    told = {"outcome": outcome, "outcome_summary": said, "started_at": at}
    return json.dumps({"type": "run", **test, **told, "rerun_number": 0})


def plain(text: str) -> str:
    """What a terminal shows of the text, without the styles it names."""
    return re.sub(r"\033\[[\d;]*m", "", text)


def shows(written: str) -> list[str]:
    """The rows a terminal shows once this is written to it: it follows the newlines
    and the control sequences that move or erase, and takes no notice of the rest."""
    rows, at, column = [""], 0, 0
    for token in re.findall(r"\033\[\??[\d;]*[A-Za-z]|\033\][^\a]*\a|[\s\S]", written):
        if token == "\n":
            at, column = at + 1, 0
            rows += [""] * (at + 1 - len(rows))
        elif token == "\r":
            column = 0
        elif token.startswith("\033[") and token.endswith("A"):
            at -= int(token[2:-1])
        elif token == "\033[K":
            rows[at] = rows[at][:column]
        elif token == "\033[J":
            rows[at:] = [rows[at][:column]]
        elif not token.startswith("\033"):
            rows[at] = rows[at][:column].ljust(column) + token + rows[at][column + 1 :]
            column += 1
    return rows


def told(queue: Any) -> list[Any]:
    """What a container has put on its queue, which stands in for Modal's, since the
    last look."""
    put = queue.from_id.return_value.put
    said = [call.args[0] for call in put.call_args_list]
    put.reset_mock()
    return said


def wait_for(test: unittest.TestCase, logs: Any, message: str) -> None:
    """Wait for another thread to log a message, given as a regular expression."""
    deadline = time.monotonic() + 60
    while not any(re.fullmatch(message, r.getMessage()) for r in logs.records):
        test.assertLess(time.monotonic(), deadline, f"never logged: {message}")
        time.sleep(0.01)


@contextlib.contextmanager
def exits(test: unittest.TestCase, error: str) -> Iterator[None]:
    """The block logs the error and exits, as ramjet does on every failure."""
    with test.assertLogs("ramjet", "ERROR") as logs, test.assertRaises(SystemExit):
        yield
    test.assertIn(error, "\n".join(logs.output))


def files(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): path.read_text()
        for path in root.rglob("*")
        if path.is_file() and ".git" not in path.parts
    }


class LocalRunner:
    """Stands in for a runner sandbox: its /pytorch is a local directory, and it
    shares your filesystem, so files copied in are already there."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.filesystem = self

    def copy_from_local(self, local_path: str, remote_path: str) -> None:
        pass

    def exec(self, *args: str, workdir: str, text: bool = True) -> Any:
        cwd = self.root / Path(workdir).relative_to("/pytorch")
        pipe = subprocess.PIPE  # like a sandbox's process, its output is there to read
        return subprocess.Popen(args, cwd=cwd, stdout=pipe, stderr=pipe, text=text)


@skipIfWindows(msg="ramjet runs on Linux and macOS")
class TestRepo(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def setUp(self) -> None:
        super().setUp()
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.tmp = Path(scratch.name)
        patcher = mock.patch.dict(os.environ, GIT_ENV)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.repo = self.tmp / "repo"
        self.repo.mkdir()
        self.addCleanup(os.chdir, os.getcwd())
        os.chdir(self.repo)
        common.repo_root.cache_clear()  # each test has its own repo
        self.addCleanup(common.repo_root.cache_clear)
        common.git("init", "-q", "-b", "main")
        for path in ["kept.py", "edited.py", "deleted.py", "module"]:
            self.write(path, f"{path}\n")
        common.git("add", "--all")
        common.git("commit", "-q", "-m", "base")
        self.base = common.git("rev-parse", "HEAD")

    def write(self, path: str, text: str) -> None:
        (self.repo / path).parent.mkdir(parents=True, exist_ok=True)
        (self.repo / path).write_text(text)

    def answer(self, rows: list[list[Any]]) -> None:
        """Have CI's times be these rows: a test, its seconds here and anywhere."""
        self.write(times.WRAPPER, WRAPPER.format(self.tmp))
        (self.repo / times.WRAPPER).chmod(0o755)
        (self.tmp / "rows.json").write_text(json.dumps({"rows": rows}))

    def edit(self) -> None:
        self.write("edited.py", "edited\n")
        self.write("new dir/new file.py", "added\n")
        (self.repo / "deleted.py").unlink()
        (self.repo / "module").unlink()
        self.write("module/__init__.py", "now a package\n")

    def test_snapshot_tree(self) -> None:
        index = Path(common.git("rev-parse", "--git-path", "index"))
        before = index.read_bytes()
        self.edit()
        commit = sync.snapshot_tree()
        self.assertEqual(index.read_bytes(), before)
        self.assertEqual(common.git("rev-parse", "HEAD"), self.base)
        self.assertEqual(common.git("rev-parse", f"{commit}^"), self.base)
        self.assertEqual(common.git("show", f"{commit}:new dir/new file.py"), "added")
        changes = {
            "deleted.py": "D",
            "edited.py": "M",
            "module": "D",
            "module/__init__.py": "A",
            "new dir/new file.py": "A",
        }
        self.assertEqual(sync.diff(self.base, commit), changes)
        self.assertEqual(sync.diff(commit, commit), {})

    def test_ramjet_dir(self) -> None:
        with mock.patch.object(tempfile, "tempdir", str(self.tmp)):
            path = common.ramjet_dir()
            user_dir = self.tmp / f"ramjet-{getpass.getuser()}"
            self.assertEqual(path.parent, user_dir)
            self.assertRegex(path.name, r"^repo-[0-9a-f]{8}$")
            self.assertEqual(user_dir.stat().st_mode & 0o777, 0o700)
            self.assertEqual(common.ramjet_dir(), path)
            with mock.patch("os.getuid", return_value=os.getuid() + 1):
                with exits(self, "another user"):
                    common.ramjet_dir()

    def test_merge_base(self) -> None:
        common.git("remote", "add", "origin", "https://github.com/someone/pytorch.git")
        with exits(self, "no git remote"):
            sync.merge_base("HEAD")
        common.git("remote", "add", "upstream", "git@github.com:pytorch/pytorch.git")
        common.git("update-ref", "refs/remotes/upstream/main", self.base)
        self.write("edited.py", "a local commit\n")
        common.git("commit", "-q", "-am", "local")
        self.edit()
        self.assertEqual(sync.merge_base(sync.snapshot_tree()), self.base)

    def test_bundle_reaches_a_partial_clone(self) -> None:
        # Like the builder's clone of pytorch/pytorch: it has the base, but no blobs
        builder = self.tmp / "builder"
        common.git("config", "uploadpack.allowFilter", "true")
        clone = ["clone", "-q", "--filter=blob:none", "--no-checkout"]
        common.git(*clone, self.repo.as_uri(), str(builder))
        self.edit()
        commit = sync.snapshot_tree()
        bundle = sync.bundle(commit, self.base, self.tmp)
        os.chdir(builder)
        common.git("fetch", "-q", str(bundle), "refs/worktree/ramjet")
        common.git("checkout", "-q", "--force", "--detach", "FETCH_HEAD")
        self.assertEqual(files(builder), files(self.repo))

    @unittest.skipUnless(HAS_MODAL, "needs modal")
    def test_image_for(self) -> None:
        clones: list[bool] = []

        def fake_build(*args: Any, clone: bool) -> tuple[str, float]:
            clones.append(clone)
            args[-1].write_text("lib.cpp\n")  # the inputs, as the builder exports them
            return f"im-{len(clones)}", 180.0  # the image, and the builder's seconds

        ci_image = build.CIImage("ci:a", python="3.11")
        for patcher in [
            mock.patch.object(build, "ci_images", return_value={"13.2": ci_image}),
            mock.patch.object(build, "toolchain"),
            mock.patch.object(build, "build", side_effect=fake_build),
            mock.patch.object(sync, "merge_base", return_value=self.base),
            mock.patch.object(cli.modal.Image, "from_id"),
        ]:
            patcher.start()
            self.addCleanup(patcher.stop)

        def image_for(
            days_later: float = 0, python: str | None = None
        ) -> tuple[lineage.Build, float]:
            now = time.time() + days_later * 24 * 3600
            date = f"{now:.0f} +0000"  # git's clock too, which dates the snapshot
            env = {"GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date}
            with mock.patch.dict(os.environ, env):
                with mock.patch.object(cli.time, "time", return_value=now):
                    commit = sync.snapshot_tree()
                    return cli.image_for(commit, self.tmp, "13.2", python, {})

        self.write("lib.cpp", "int a;\n")
        first, seconds = image_for()
        self.assertEqual((first.image_id, clones, seconds), ("im-1", [True], 180.0))
        self.write("edited.py", "a Python edit\n")
        kept = [self.tmp / "lineage.json", *self.tmp.glob("*.inputs")]
        for path in kept:
            os.utime(path, (0, 0))  # old enough for a cleaner of /tmp to delete
        self.assertEqual(image_for(), (first, 0.0))  # no build: no builder ran
        self.assertEqual([path.stat().st_mtime > 0 for path in kept], [True, True])
        self.write("lib.cpp", "int b;\n")
        self.write(".ci/docker/README.md", "a new CI image tag\n")  # same lineage
        self.assertEqual((image_for()[0].image_id, clones), ("im-2", [True, False]))
        self.assertEqual(image_for(days_later=1)[0].image_id, "im-2")
        self.assertEqual(image_for(days_later=4.1)[0].image_id, "im-3")  # renewed
        self.assertEqual(image_for(days_later=10)[0].image_id, "im-4")  # expired
        self.assertEqual(image_for(days_later=10, python="3.11")[0].image_id, "im-4")
        self.assertEqual(image_for(days_later=10, python="3.13")[0].image_id, "im-5")
        self.assertEqual(clones, [True, False, False, True, True])

    @unittest.skipUnless(HAS_MODAL, "needs modal")
    def test_sync_files(self) -> None:
        runner = self.tmp / "runner"
        runner.mkdir()
        sync.archive(self.base, ["."], self.tmp / "base.tar")
        subprocess.run(["tar", "-xf", self.tmp / "base.tar"], cwd=runner, check=True)
        self.edit()
        commit = sync.snapshot_tree()
        run.sync_files(LocalRunner(runner), self.base, commit, self.tmp)
        self.assertEqual(files(runner), files(self.repo))

    def test_times(self) -> None:
        files = {"test/test_a.py": 3, "test/it's.py": 2}
        with self.assertLogs("ramjet", "WARNING") as logs:  # no wrapper to run
            self.assertEqual(times.fetch(files, "NVIDIA L4", 1), {})
        self.assertIn("no test times from CI:\n", logs.output[0])
        failing = "#!/bin/sh\necho gh: not logged in >&2\necho gcx.error\nexit 1\n"
        self.write(times.WRAPPER, failing)
        (self.repo / times.WRAPPER).chmod(0o755)
        with self.assertLogs("ramjet", "WARNING") as logs:
            self.assertEqual(times.fetch(files, "NVIDIA L4", 1), {})
        message = "no test times from CI:\ngh: not logged in\ngcx.error"
        self.assertEqual(logs.records[0].getMessage(), message)
        one, two = "test/test_a.py::A::test_one", "test/test_a.py::test_two"
        self.answer([[one, 1.5, 2.0], [two, None, 0.25]])
        fetched = times.fetch(files, "NVIDIA L4", 1)
        self.assertEqual(fetched, {one: (1.5, 2.0), two: (None, 0.25)})
        query = (self.tmp / "queries").read_text()
        self.assertIn("e.device_name = 'NVIDIA L4' AND e.device_count = 1", query)
        self.assertIn("t.file IN ('test/test_a.py', 'test/it\\'s.py')", query)
        with mock.patch.object(times, "BATCH", 4):  # a query for each file, then
            times.fetch(files, "", 0)
        queries = (self.tmp / "queries").read_text().split("SELECT")[2:]
        self.assertEqual(len(queries), 2)
        self.assertIn("e.device_name = '' AND e.device_count = 0", queries[0])
        (self.tmp / "rows.json").write_text('{"rows": null}')  # CI ran none of them
        with self.assertNoLogs("ramjet", "WARNING"):
            self.assertEqual(times.fetch(files, "NVIDIA L4", 1), {})

    @unittest.skipUnless(HAS_MODAL, "needs modal")
    def test_run_tests(self) -> None:
        container = self.tmp / "container"  # its /pytorch, at the build image's commit
        sync.archive(self.base, ["."], self.tmp / "base.tar")

        def boot() -> None:
            shutil.rmtree(container, ignore_errors=True)
            container.mkdir()
            unpack = ["tar", "-xf", self.tmp / "base.tar"]
            subprocess.run(unpack, cwd=container, check=True)

        boot()
        self.edit()
        self.write("test/test_file.py", TEST_FILE)
        self.write("test/test_broken.py", "raise ImportError('no such module')\n")
        plugins = f"-p {worker.TCI} -p {worker.SERIAL}"
        launched = f"--use-pytest --collect-only --import-slow-tests {plugins}"
        launched += " -k second -v"
        second = "test/test_file.py::TestFile::test_second"
        nodeid = f"{second}[{launched} --verbosity=-1 no PYTHONPATH]"
        self.answer([[nodeid, 2.0, 3.0]])
        commit = sync.snapshot_tree()

        asked: list[list[str]] = []

        def local_collect(calls: Any, **options: Any) -> Any:
            unordered = {"order_outputs": False, "return_exceptions": True}
            self.assertEqual(options, unordered)
            for named, pytest_args, told in calls:
                self.assertIs(told, queue.object_id)
                started = worker.Worker()  # this process stands in for the container
                started.sync()
                asked.append(named)
                yield started.collect(named, pytest_args, told)

        modal = mock.MagicMock()  # stands in for Modal
        modal.App.return_value.get_dashboard_url.return_value = APP
        image = modal.Image.from_id.return_value.run_commands.return_value

        def add_local_file(local: str, remote: str) -> Any:
            shutil.copyfile(local, remote)  # where the container finds it
            return image

        image.add_local_file.side_effect = add_local_file
        workers = modal.App.return_value.cls.return_value.return_value
        workers.return_value.collect.starmap = local_collect
        for method in (workers.return_value.collect, workers.return_value.run):
            method.get_current_stats.return_value.num_total_runners = 1
        queue = modal.Queue.ephemeral.return_value.__enter__.return_value
        queue.get_many.return_value = []

        def local_starmap(calls: Any, **options: Any) -> Any:
            # A container's 2 cores fit one test process, with CI's least threads
            sized = {"procs": 1, "threads": test.THREADS}
            args = ["-k", "second", "-v"]
            self.assertEqual(options["kwargs"], {"args": args, **sized})
            for number, tests, told, where, started in calls:  # each test passes
                self.assertIs(started, queue.object_id)
                # One process, so run_test.py hears of less than it splits a file at
                self.assertEqual(told, {"test_file": 300.0})
                # ... which runs the test marked @serialTest too: with one at a
                # time, run_test.py runs every file alone
                self.assertEqual(where, {nodeid: ("test_file", 0, False)})
                ids = [nodeid for group in tests.values() for nodeid in group]
                report = "\n".join(run_line(nodeid, "passed") for nodeid in ids)
                done = {"shard": number, "code": 0, "seconds": 9.0, "container": "ta-2"}
                said = {"reports": {"a.jsonl": report}, "output": nothing}
                used = {"cores": 2.0, "memory": 4096, "gpu_memory": 1024}
                yield {**done, **said, **used}

        nothing = gzip.compress(b"")
        workers.return_value.run.starmap = local_starmap

        def fetch(*args: Any) -> Any:
            # collect has run to its end, where its progress lines stop
            names = [thread.name for thread in threading.enumerate()]
            self.assertNotIn("ramjet-progress", names)
            return mock.DEFAULT  # what times.fetch returns

        modal_sets = {"PYTHONPATH": "/pkg/:/root/", "MODAL_TASK_ID": "ta-1"}
        allows = {"split": 600, "alone": ["test_other"]}
        allows["runs"] = ["test_file", "test_broken"]
        for patcher in [
            mock.patch.object(test, "modal", modal),
            mock.patch.object(worker.modal, "Queue"),
            mock.patch.object(times, "fetch", wraps=times.fetch, side_effect=fetch),
            mock.patch.object(worker, "ROOT", str(container)),
            mock.patch.object(worker, "CHANGES", str(self.tmp / "changes.tar")),
            mock.patch.object(worker, "DELETED", str(self.tmp / "deleted")),
            mock.patch.object(worker, "machine", return_value=L4),
            mock.patch.object(worker, "allows", return_value=allows),
            mock.patch.dict(os.environ, modal_sets),
        ]:
            patcher.start()
            self.addCleanup(patcher.stop)

        build_image = lineage.Build("im-1", self.base, taken=0.0)
        sizes = (test.CORES_PER_GPU, test.MIB_PER_CORE)
        resources = run.resources("L4", 2, None, *sizes)
        names = {
            "test_file": ["test/test_file.py::TestFile::test_second"],
            "test_broken": ["test/test_broken.py"],
            "test_elsewhere": ["test/test_elsewhere.py"],  # not run in the container
        }
        work = (build_image, commit, resources, names, ["-k", "second", "-v"])
        args = (*work, self.tmp, {"PATH": "/bin"})  # none of yours for a container
        # What a container bills an hour, and the dollars the run's build billed
        rate, built = Decimal("3.60"), Decimal("1.25")
        with self.assertLogs(test.log, "DEBUG") as logs:
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(test.run_tests(*args, True, rate, built), 1)
        self.assertEqual(asked, [list(names)])  # one call names every file
        # One call at a time, and no wait for shards when listing
        options = {"image": image, "timeout": worker.CALL_SECONDS, "gpu": "L4"}
        options.update(cpu=2, memory=8192, env={})
        idle = {"scaledown_window": test.SCALEDOWN_SECONDS}
        modal.App.return_value.cls.assert_called_with(**options, **idle)
        listed = [json.loads(line) for line in stdout.getvalue().splitlines()]
        self.assertEqual(listed, [{"nodeid": nodeid, "shard": 1}])
        messages = [record.getMessage() for record in logs.records]
        self.assertEqual(messages[0], "collecting 3 files on L4")
        self.assertEqual(logs.records[0].levelname, "DEBUG")
        took = r"^test/test_%s\.py took \d+\.\d s on container ta-1$"
        self.assertRegex(messages[1], took % "file")
        # Of the two tests tci left out, the one under the node id asked for
        needs = "@tci.needs(tci.gpus >= 2)"
        self.assertEqual(messages[2], f"left out {second}[two]: {needs}")
        self.assertRegex(messages[3], took % "broken")
        self.assertIn("test/test_broken.py failed:", messages[4])
        self.assertIn("ImportError: no such module", messages[4])
        # The file that run_test.py doesn't run in the container is not launched
        unrun = "left out test/test_elsewhere.py: run_test.py doesn't run it"
        skipped = "left out 1 file that run_test.py doesn't run on L4"
        self.assertEqual(messages[5:7], [unrun, skipped])
        self.assertEqual(logs.records[5].levelname, "DEBUG")
        left_out = ["left out 1 test that L4 can't run:", f"  1 with {needs}"]
        self.assertEqual(messages[7:9], left_out)
        # CI took 2 s for the test on an L4, which is the machine it was asked about.
        # The L4's memory has room for three test processes, and 2 cores fit one
        sources = "1 test with 1 x NVIDIA L4, 0 on other machines, 0 guessed"
        seconds = round(test.START_SECONDS + 2 * test.SLOWDOWN)
        planned = [
            "running 1 test process at once: 2 cores fit no more",
            "L4 fits 3 with --cpu 4",
            "test processes that fit: 1 in 2 cores, 1 in 8 GiB, 3 in a GPU's 22 GiB",
            "test processes at once: 1, and 1 of the files alone",
            f"CI's times: {sources}",
            f"shard 1, about {seconds} s: 1 test of 1 file",
            # What the run comes to, at $3.60 an hour: the price is magenta
            f"1 test • 1 shard • about {seconds} s • estimate $0.01",
        ]
        self.assertEqual([plain(message) for message in messages[9:]], planned)
        self.assertTrue(messages[-1].endswith("estimate \033[35m$0.01\033[0m"))
        self.assertEqual(logs.records[-1].levelname, "DEBUG")
        query = (self.tmp / "queries").read_text()
        self.assertIn("e.device_name = 'NVIDIA L4' AND e.device_count = 1", query)
        self.assertIn("t.file IN ('test/test_file.py')", query)
        self.assertEqual(files(container), files(self.repo))  # synced first
        # Without --list the shard runs: the files that couldn't be collected still fail
        boot()
        with self.assertLogs(test.log, "INFO") as logs:
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(test.run_tests(*args, False, rate, built), 1)
        self.assertEqual(stdout.getvalue(), "")  # no test failed
        messages = [plain(record.getMessage()) for record in logs.records]
        self.assertNotIn(planned[-1], messages)  # which --log-level debug shows
        (kept,) = self.tmp.glob("test-*")
        # The price is the run's: the build's $1.25 and what the container billed
        self.assertRegex(messages[-4], r"^\$1\.2\d for 1 test in \d+ s:$")
        self.assertEqual(messages[-3:], ["1 passed", str(kept), APP])
        saved = sorted(str(path.relative_to(kept)) for path in kept.rglob("*"))
        self.assertEqual(saved, ["reports", "reports/a.jsonl", "shard-1.log"])
        # The container that collected waits for a shard
        idle = {"scaledown_window": test.HANDOVER_SECONDS}
        modal.App.return_value.cls.assert_called_with(**options, **idle)
        modal.App.return_value.cls.return_value.assert_called_with(worker.Worker)
        # With no test to run there is nothing to plan: the files are counted. Of
        # your variables, the containers get those that torch and its tests read
        none = {"test_file": ["test/test_file.py::TestFile::test_none"]}
        passed = {"TORCH_LOGS": "+dynamo, guards", "PYTORCH_TEST_WITH_SLOW": "1"}
        others = {"PATH": "/bin", "CUDA_LAUNCH_BLOCKING": "1", "MY_TORCH": "1"}
        empty = (build_image, commit, resources, none, [], self.tmp, passed | others)
        boot()
        with self.assertLogs(test.log, "INFO") as logs:
            self.assertEqual(test.run_tests(*empty, True, rate, built), 0)
        logged = [record.getMessage() for record in logs.records]
        passing = "passing PYTORCH_TEST_WITH_SLOW=1 TORCH_LOGS='+dynamo, guards'"
        self.assertEqual(logged, [passing, "found 0 tests in 1 file"])
        listing = {**options, "env": passed, "scaledown_window": test.SCALEDOWN_SECONDS}
        modal.App.return_value.cls.assert_called_with(**listing)
        # If Modal loses the call, no file is collected
        lost = ConnectionError("container lost")
        workers.return_value.collect.starmap = mock.Mock(side_effect=lost)
        with self.assertLogs(test.log, "INFO") as logs:
            self.assertEqual(test.run_tests(*args, True, rate, built), 1)
        logged = [record.getMessage() for record in logs.records]
        gone = [f"collecting failed:\n{lost!r}", "found 0 tests in 0 files"]
        self.assertEqual(logged, gone)
        # Modal ends its app on Ctrl-C and swallows it: the Ctrl-C goes on all the same
        modal.App.return_value.run.return_value.__exit__.return_value = True
        stopped = mock.Mock(side_effect=KeyboardInterrupt)
        workers.return_value.collect.starmap = stopped
        with self.assertLogs(test.log, "DEBUG"), self.assertRaises(KeyboardInterrupt):
            test.run_tests(*args, False, rate, built)

    @unittest.skipUnless(HAS_MODAL, "needs modal")
    @unittest.skipUnless(shutil.which("setsid"), "needs util-linux's setsid")
    def test_execute(self) -> None:
        (self.repo / "sub").mkdir()
        script = 'printf %s "$1" > out; echo to stdout; echo to stderr >&2; exit 3'
        command = ["sh", "-c", script, "sh", "a b"]
        said = io.TextIOWrapper(io.BytesIO())
        with mock.patch.object(run, "PID_FILE", str(self.tmp / "pid")):
            with mock.patch.object(sys, "stdout", said):
                self.assertEqual(run.execute(LocalRunner(self.repo), command, "sub"), 3)
        self.assertEqual((self.repo / "sub/out").read_text(), "a b")  # as given
        # Both of its streams, whole and in the order it wrote them
        self.assertEqual(said.buffer.getvalue(), b"to stdout\nto stderr\n")
        self.assertRegex((self.tmp / "pid").read_text(), r"^\d+\n$")


class TestLineage(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_build_config(self) -> None:
        environ = {
            "BUILD_TEST": "0",
            "CMAKE_PREFIX_PATH": "/opt/conda",
            "CUDA_PATH": "/usr/local/cuda",
            "DEBUG": "1",
            "MAX_JOBS": "8",
            "TORCH_CUDA_ARCH_LIST": "9.0",
            "USE_CUDA": "1",
        }
        names = ["BUILD_TEST", "DEBUG", "TORCH_CUDA_ARCH_LIST", "USE_CUDA"]
        self.assertEqual(lineage.build_config(environ), {n: environ[n] for n in names})

    def test_key(self) -> None:
        config = {"USE_CUDA": "1", "DEBUG": "1"}
        key = lineage.key("ci-image:a", config)
        self.assertEqual(key, lineage.key("ci-image:a", dict(reversed(config.items()))))
        self.assertNotEqual(key, lineage.key("ci-image:b", config))
        self.assertNotEqual(key, lineage.key("ci-image:a", {"USE_CUDA": "1"}))

    def test_usable_and_due(self) -> None:
        build = lineage.Build("im-1", "abc", taken=0.0)
        self.assertIsNone(lineage.usable(None, 0.0))
        self.assertIs(lineage.usable(build, lineage.EXPIRES_AFTER - 1), build)
        self.assertIsNone(lineage.usable(build, lineage.EXPIRES_AFTER))
        self.assertFalse(lineage.due(build, lineage.RENEW_AFTER))
        self.assertTrue(lineage.due(build, lineage.RENEW_AFTER + 1))

    def test_read_by_build(self) -> None:
        read = frozenset({"aten/src/ATen/native/Linear.cpp", "torchgen/gen.py"})
        cases = [
            ("torch/nn/modules/linear.py", "M", False),
            ("test/test_new.py", "A", False),
            ("torch/csrc/README.md", "M", False),
            ("aten/src/ATen/native/Linear.cpp", "M", True),
            ("torchgen/gen.py", "D", True),
            ("torch/csrc/new.cpp", "A", True),
            ("caffe2/CMakeLists.txt", "M", True),
            ("cmake/Modules/FindFoo.cmake", "M", True),
            ("setup.py", "M", True),
            (".ci/docker/ci_commit_pins/triton.txt", "M", True),  # a Triton pin bump
            ("third_party/fbgemm", "M", True),  # a submodule bump
            ("torch/_native/ops.py", "M", True),
        ]
        for path, status, builds in cases:
            with self.subTest(path=path, status=status):
                found = lineage.read_by_build({path: status}, read)
                self.assertEqual(found, [path] if builds else [])
        changes = {
            "setup.py": "M",
            "torch/nn/functional.py": "M",
            "aten/src/ATen/native/Linear.cpp": "M",
        }
        found = lineage.read_by_build(changes, read)
        self.assertEqual(found, ["setup.py", "aten/src/ATen/native/Linear.cpp"])

    def test_load_and_save(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch, "lineage.json")
            self.assertEqual(lineage.load(path), {})
            builds = {"key": lineage.Build("im-1", "abc", 1.5)}
            lineage.save(path, builds)
            self.assertEqual(lineage.load(path), builds)


@skipIfWindows(msg="ramjet runs on Linux and macOS")
@unittest.skipUnless(shutil.which("cmake") and shutil.which("ninja"), "needs CMake")
class TestInputs(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_build_inputs(self) -> None:
        sources = {
            "CMakeLists.txt": CMAKE_LISTS,
            "demo.c": '#include "demo.h"\nint demo(void) { return DEMO; }\n',
            "demo.h": "#define DEMO 1\n",
        }
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch).resolve()
            for name, text in sources.items():
                (root / name).write_text(text)
            build = root / "build"
            cmake = ["cmake", "-G", "Ninja", "-S", root, "-B", build]
            subprocess.run(cmake, check=True, capture_output=True)
            subprocess.run(["ninja", "-C", build], check=True, capture_output=True)
            self.assertEqual(inputs.build_inputs(str(root)), sorted(sources))


@unittest.skipUnless(HAS_MODAL, "needs modal")
class TestRun(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_resources(self) -> None:
        self.assertEqual(run.resources(None, None, None), (None, 8, 65536))
        self.assertEqual(run.resources("H100:2", None, None), ("H100:2", 16, 131072))
        self.assertEqual(run.resources("B200:8", None, None), ("B200:8", 64, 344064))
        self.assertEqual(run.resources(None, 32, None), (None, 32, 262144))
        self.assertEqual(run.resources("B200:8", 2, None), ("B200:8", 2, 16384))
        self.assertEqual(run.resources("H100", 64, 8192), ("H100", 64, 8192))
        # test's containers are smaller
        sizes = (test.CORES_PER_GPU, test.MIB_PER_CORE)
        self.assertEqual(run.resources(None, None, None, *sizes), (None, 4, 16384))
        two = run.resources("H100:2", None, None, *sizes)
        self.assertEqual(two, ("H100:2", 8, 32768))
        self.assertEqual(run.resources("L4", 8, None, *sizes), ("L4", 8, 32768))

    def test_hourly_rate(self) -> None:
        def rate(*request: Any) -> Decimal:  # from --gpu, --cpu and --memory
            return run.hourly_rate(run.resources(*request), RATES)

        core, gib = RATES["cpu_hour_cost_sandbox"], RATES["mem_gib_hour_cost_sandbox"]
        host = 8 * core + 64 * gib  # 8 cores and 64 GiB
        self.assertEqual(rate(None, None, None), host)
        self.assertEqual(rate(None, 32, None), 32 * core + 256 * gib)
        self.assertEqual(rate(None, 8, 8192), 8 * core + 8 * gib)
        gpus = {
            "H100": "h100",
            "H100!": "h100",
            "A10": "a10g",
            "A100": "a100_40gb",
            "A100-80GB": "a100_80gb",
            "RTX-PRO-6000": "rtx6000",
            "B200+": "b200",
        }
        for gpu, name in gpus.items():
            with self.subTest(gpu=gpu):
                gpu_rate = RATES[f"gpu_hour_cost_{name}"]
                self.assertEqual(rate(gpu, None, None), host + gpu_rate)
        no_h100 = {k: v for k, v in RATES.items() if k != "gpu_hour_cost_h100"}
        with exits(self, "no GPU named h100"):
            run.hourly_rate(run.resources("H100", None, None), no_h100)
        # One of test's containers bills its cores and memory as a Function
        small = run.resources("L4", 4, 16384)
        function = 4 * RATES["cpu_hour_cost"] + 16 * RATES["mem_gib_hour_cost"]
        billed = run.hourly_rate(small, RATES, kind="")
        self.assertEqual(billed, function + RATES["gpu_hour_cost_l4"])
        # The builder bills its 64 cores and the 160 GiB it asks for: 3 minutes here
        self.assertEqual(build.cost(180.0, RATES), (64 * core + 160 * gib) / 20)

    def test_start(self) -> None:
        size = run.resources("L4", None, None)
        with mock.patch.object(run, "modal") as modal:
            run.start(lineage.Build("im-1", "commit", 0.0), size, 60)
        # Runners have an app of their own, and each says whose it is
        modal.App.lookup.assert_called_once_with("ramjet-run", create_if_missing=True)
        asked = modal.Sandbox.create.call_args.kwargs
        self.assertEqual((asked["gpu"], asked["idle_timeout"]), ("L4", 60))
        self.assertEqual(asked["tags"], {"user": getpass.getuser()})

    def test_idle_timeout(self) -> None:
        def idle(gpu: str | None) -> int:
            return run.idle_timeout(run.resources(gpu, None, None), RATES)

        self.assertEqual(idle(None), run.IDLE_MAX_SECONDS)
        self.assertEqual(idle("H100"), run.IDLE_MAX_SECONDS)
        # 8 B200s with 64 cores and 336 GiB bill $67.15 an hour: $2.50 is 134 s
        self.assertEqual(idle("B200:8"), 134)

    def test_show(self) -> None:
        h100, cpu = run.resources("H100:8", None, None), run.resources(None, None, None)
        with self.assertLogs(run.log, "INFO") as logs:
            run.show("sb-new", False, h100, Decimal("48.75"))
            run.show("sb-warm", True, cpu, Decimal("2.672"))
        self.assertEqual(
            [record.getMessage() for record in logs.records],
            [
                "sandbox sb-new (new): \033[31m$48.75/h\033[0m • H100 × 8 • 64 cores • 336 GiB",
                "sandbox sb-warm (warm): \033[31m$2.67/h\033[0m • no GPU • 8 cores • 64 GiB",
            ],
        )

    def test_runs_on(self) -> None:
        cuda13 = "7.5;8.0;8.6;9.0;10.0;12.0+PTX"
        cuda12 = "5.0;6.0;7.0;7.5;8.0;8.6;9.0"
        for name, gpu in run.GPUS.items():
            with self.subTest(gpu=name):
                blackwell = gpu.capability >= (10, 0)
                self.assertTrue(run.runs_on(gpu.capability, cuda13))
                self.assertEqual(run.runs_on(gpu.capability, cuda12), not blackwell)
        self.assertFalse(run.runs_on((10, 0), "9.0"))
        self.assertTrue(run.runs_on((10, 0), "9.0+PTX"))  # the driver compiles PTX
        self.assertTrue(run.runs_on((8, 9), "7.5 8.6"))  # CI separates with spaces
        self.assertTrue(run.runs_on((10, 0), "Hopper"))  # not judged
        with exits(self, "can't run on B200:8"):
            run.check_archs("B200:8", cuda12)
        run.check_archs(None, cuda12)


@unittest.skipUnless(HAS_MODAL, "needs modal")
class TestBuild(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_ci_image_name(self) -> None:
        names = [
            "pytorch-linux-jammy-cuda12.6-cudnn9-py3.11-gcc11",
            "pytorch-linux-jammy-cuda13.2-cudnn9-py3-gcc11",
            "pytorch-linux-jammy-cuda13.2-cudnn9-py3-gcc11-inductor-benchmarks",
            "pytorch-linux-jammy-cuda13.2-cudnn9-py3.12-gcc11",
            "pytorch-linux-jammy-cuda13.4-cudnn9-py3.11-clang20",
        ]
        found = build.CI_IMAGE_NAME.findall(",\n".join(names) + ",")
        self.assertEqual(found, [(names[0], "12.6"), (names[1], "13.2")])

    def test_ci_python(self) -> None:
        script = (
            "  pytorch-linux-jammy-cuda13.2-cudnn9-py3-gcc11)\n"
            "    CUDA_VERSION=13.2.2\n"
            "    ANACONDA_PYTHON_VERSION=3.11\n"
            "    ;;\n"
            "  pytorch-linux-jammy-cuda13.2-cudnn9-py3.12-gcc11)\n"
            "    ANACONDA_PYTHON_VERSION=3.12\n"
            "    ;;\n"
        )
        name = "pytorch-linux-jammy-cuda13.2-cudnn9-py3-gcc11"
        self.assertEqual(build.ci_python(script, name), "3.11")
        with exits(self, "no Python"):
            build.ci_python(script, name.replace("13.2", "12.8"))

    def test_build(self) -> None:
        builder = mock.MagicMock()
        builder.exec.return_value.stdout = []
        builder.exec.return_value.wait.return_value = 0
        builder.snapshot_filesystem.return_value.object_id = "im-1"
        # The builder bills from when Modal is asked for it until it is gone
        clock = mock.patch.object(build.time, "monotonic", side_effect=[10.0, 190.0])
        with tempfile.TemporaryDirectory() as tmp, clock:
            bundle, inputs = Path(tmp) / "build.bundle", Path(tmp) / "inputs"
            with mock.patch.object(build, "modal") as modal:
                modal.Sandbox.create.return_value = builder
                built = build.build(None, bundle, {}, inputs, clone=False)
        self.assertEqual(built, ("im-1", 180.0))
        builder.terminate.assert_called_once()
        # Builders have an app of their own, and each says whose it is
        modal.App.lookup.assert_called_once_with("ramjet-build", create_if_missing=True)
        tags = modal.Sandbox.create.call_args.kwargs["tags"]
        self.assertEqual(tags, {"user": getpass.getuser()})


@unittest.skipUnless(HAS_MODAL, "needs modal")
class TestCli(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_parse_args(self) -> None:
        def parse(*argv: str) -> tuple[object, ...]:
            options, command = cli.parse_args(list(argv))
            return (*vars(options).values(), command)  # in the order they're added

        cudas, stable = list(build.ci_images()), build.stable_cuda()
        pythons = build.python_versions()
        self.assertIn(stable, cudas)
        defaults = (None, None, stable, None, "info")  # every option but --gpu
        self.assertEqual(parse("run", "ls"), (None, *defaults, "run", ["ls"]))
        command = ["pytest", "test/test_nn.py", "-k", "linear or conv"]
        argv = ["--gpu", "H100", "run", *command]
        self.assertEqual(parse(*argv), ("H100", *defaults, "run", command))
        cuda, python = f"--build-cuda={cudas[-1]}", f"--python-version={pythons[-1]}"
        options = ["--cpu=16", "--memory=131072", cuda, python, "--log-level=debug"]
        argv = ["--gpu=b200+:8", *options, "run", "nvidia-smi"]  # in any case
        expected = ("B200+:8", 16, 131072, cudas[-1], pythons[-1], "debug", "run")
        self.assertEqual(parse(*argv), (*expected, argv[-1:]))
        # What follows run is the command's, its options and its --s too
        command = ["a", "--gpu", "L4", "-h", "--", "b", "--", "test"]
        self.assertEqual(parse("run", *command), (None, *defaults, "run", command))
        argv = ["run", "--", "a", "--", "b"]  # a -- right after run is not
        self.assertEqual(parse(*argv), (None, *defaults, "run", ["a", "--", "b"]))
        # test: its own options and the paths, then pytest's arguments after --
        paths = ["test/dynamo", "test/test_nn.py::TestNN"]
        listing = ("test", True, paths)
        argv = ["--gpu", "L4", "test", "--list", *paths]
        self.assertEqual(parse(*argv), ("L4", *defaults, *listing, []))
        argv = ["test", *paths, "--list", "--", "-k", "linear", "--", "-x"]
        pytest_args = ["-k", "linear", "--", "-x"]
        self.assertEqual(parse(*argv), (None, *defaults, *listing, pytest_args))
        running = ("test", False, paths)  # without --list, test runs them
        self.assertEqual(parse("test", *paths), (None, *defaults, *running, []))
        bad = [
            ["run"],  # no command
            ["--gpu", "L4", "run", "--"],
            ["ls"],  # the command follows run
            ["run", "--gpu", "L4", "ls"],  # --gpu goes before run
            ["--gpu", "run", "ls"],
            ["--gpu", "Z100", "run", "ls"],
            ["--gpu", "A10:5", "run", "ls"],  # A10s go up to 4
            ["--cpu=3", "run", "ls"],
            ["--memory=1000", "run", "ls"],
            ["--build-cuda=9.9", "run", "ls"],
            ["--python-version=3.9", "run", "ls"],
            ["--log-level=dev", "run", "ls"],
            ["test", "--list"],  # no paths
            ["test"],
            ["test", "--gpu", "L4", "--list", "test/dynamo"],  # --gpu goes before test
            ["test/dynamo"],
        ]
        for argv in bad:
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                parse(*argv)
        # Without a subcommand, the help on stderr
        for argv in [[], ["--gpu", "L4"], ["--", "ls"]]:
            said = io.StringIO()
            with self.subTest(argv=argv), contextlib.redirect_stderr(said):
                with self.assertRaises(SystemExit) as stop:
                    parse(*argv)
                self.assertEqual(stop.exception.code, 2)
                self.assertIn("subcommands:", said.getvalue())


class TestLogger(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_lines(self) -> None:
        log, level, written = logger.log, logger.log.level, io.StringIO()
        log.addHandler(logger._HANDLER)
        self.addCleanup(log.removeHandler, logger._HANDLER)
        self.addCleanup(log.setLevel, level)
        log.setLevel("INFO")
        # A terminal shows the log: the line that asks for it rings its bell
        with mock.patch.object(logger.sys, "stderr") as stderr:
            self.assertIs(logger.terminal(), stderr)
            with logger.through(written):
                log.info("5 tests: %s", logger.bold("1 failed"), extra=logger.RING)
                log.info("4 passed", extra=logger.CONTINUED)
                log.info("/tmp/kept", extra={**logger.CONTINUED, "margin": "Logs"})
                log.info("as it is", extra=logger.VERBATIM)
            stderr.isatty.return_value = False
            self.assertIsNone(logger.terminal())
        rung, under, named, verbatim = written.getvalue().splitlines()
        start = r"^\aramjet \d\d:\d\d:\d\d  info "
        self.assertRegex(plain(rung), start + "5 tests: 1 failed$")
        self.assertIn("\033[1m1 failed\033[0m", rung)
        self.assertEqual(logger.MARGIN, len(plain(rung)) - len("\a5 tests: 1 failed"))
        self.assertEqual(under, " " * logger.MARGIN + "4 passed")
        # A row's name ends where the level does
        self.assertEqual(plain(named), " " * (logger.MARGIN - 5) + "Logs /tmp/kept")
        self.assertIn("\033[2mLogs\033[0m", named)
        self.assertEqual(verbatim, "as it is")
        # No terminal shows a log that is not set up, or that stops at warnings
        self.assertIs(logger._HANDLER.stream, sys.stderr)
        with mock.patch.object(logger.sys, "stderr"):
            log.setLevel("WARNING")
            self.assertIsNone(logger.terminal())
            log.setLevel("INFO")
            log.removeHandler(logger._HANDLER)
            self.assertIsNone(logger.terminal())


@skipIfWindows(msg="ramjet runs on Linux and macOS")
@unittest.skipUnless(HAS_MODAL, "needs modal")
class TestTest(TestCase):
    hw_classification = HardwareClassification.GENERIC
    THIS = "tools/ramjet/test_ramjet"  # this file, as run_test.py names it

    def setUp(self) -> None:
        super().setUp()
        self.addCleanup(os.chdir, os.getcwd())
        os.chdir(REPO_ROOT)
        common.repo_root.cache_clear()
        self.addCleanup(common.repo_root.cache_clear)

    def test_names(self) -> None:
        file_id = f"test/{self.THIS}.py"
        self.assertEqual(test.names([file_id]), {self.THIS: [file_id]})
        self.assertEqual(test.names(["test/tools/ramjet"]), {self.THIS: [file_id]})
        node_id = f"{file_id}::TestCli::test_parse_args"
        both = [node_id, file_id]
        self.assertEqual(test.names(both), {self.THIS: both})
        os.chdir("test/tools")  # paths are from where you are
        self.assertEqual(test.names(["ramjet/test_ramjet.py"]), {self.THIS: [file_id]})
        os.chdir(REPO_ROOT)
        everything = test.names(["test"])
        self.assertIn(self.THIS, everything)
        self.assertIn("test_torch", everything)
        # run_test.py also knows names that are not test files, such as doctests
        no_file = [n for n in everything if not Path(f"test/{n}.py").is_file()]
        self.assertEqual(no_file, [])
        bad = [
            "test/fx",  # run by test_fx.py, not on their own
            "test/fx/test_common_passes.py",
            "test/missing.py",
            "test/tools/ramjet::TestCli",  # a node id names a file
            "tools/ramjet",  # not under test/
        ]
        for path in bad:
            with self.subTest(path=path), exits(self, f"run pytest {path}"):
                test.names([path])

    def test_collect(self) -> None:
        rules = worker.allows

        def allows() -> dict[str, Any]:
            """How run_test.py runs files here, as if it ran one that is not there."""
            real = rules()
            return {**real, "runs": [*real["runs"], "tools/ramjet/missing"]}

        queues = mock.patch.object(worker.modal, "Queue")
        with tempfile.NamedTemporaryFile() as nothing, queues as queue:
            # No file changed or deleted
            here = {"ROOT": str(REPO_ROOT), "CHANGES": nothing.name, "allows": allows}
            with mock.patch.multiple(worker, DELETED=nothing.name, **here):
                started = worker.Worker()
                started.sync()
                # This file, launched as run_test.py launches it, one not there, and
                # one that run_test.py doesn't run here, which is not launched
                three = [self.THIS, "tools/ramjet/missing", "tools/ramjet/unrun"]
                args = ["-k", "test_names", "-v"]
                collected, missing, unrun = started.collect(three, args, "qu-1")
                done = told(queue)
                # More than it lists at once: by forks of one process, and the file
                # that fails there by a process of its own after all
                with mock.patch.object(worker, "AT_ONCE", 1):
                    forked, failed, _ = started.collect(three, args, "qu-1")
                # One that can't say how far it is collects all the same
                queue.from_id.return_value.put.side_effect = ConnectionError("no queue")
                with mock.patch.object(worker, "COLLECT_SECONDS", 0.01):
                    (slow,) = started.collect([self.THIS], [], "qu-1")
        queue.from_id.assert_called_with("qu-1")
        ids = [f"test/{self.THIS}.py::TestTest::test_names"]
        self.assertEqual((collected["ids"], collected["error"]), (ids, ""))
        # On the queue: each file, as it is done
        self.assertEqual(done, [three[2], *three[:2]])
        self.assertEqual((unrun["ids"], unrun["error"], unrun["seconds"]), ([], "", 0))
        runs = [file["runs"] for file in (collected, missing, unrun)]
        self.assertEqual(runs, [True, True, False])
        self.assertEqual(forked, {**collected, "seconds": forked["seconds"]})
        self.assertEqual(failed, {**missing, "seconds": failed["seconds"]})
        self.assertEqual(collected["left_out"], {})  # no test here has a constraint
        self.assertGreater(collected["seconds"], 0)
        read = json.loads(json.dumps(dataclasses.asdict(machine.capture())))
        # With how run_test.py runs files, which it runs this one beside
        self.assertEqual(collected["machine"], {**read, "split": THRESHOLD})
        self.assertEqual((collected["serial"], collected["alone"]), ([], False))
        self.assertEqual(missing["ids"], [])
        self.assertIn("missing.py", missing["error"])
        gave_up = ([], "still collecting after 0.01 s")
        self.assertEqual((slow["ids"], slow["error"]), gave_up)

    def test_fork(self) -> None:
        # Stand-ins for test files in a container's /pytorch: most lead with
        # common_utils, which collect.py then imports before it forks them
        leading = "import torch.testing._internal.common_utils\n"
        files = {
            "test_env": ENV,
            "test_how": HOW,
            "test_quiet": leading,
            "test_same": leading,
            "test_signal": "import os, signal\nos.kill(os.getpid(), signal.SIGKILL)\n",
            "test_slow": f"{HOW}import time\ntime.sleep(600)\n",
        }
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        root = Path(scratch.name)
        (root / "test").mkdir()
        (root / "nothing").touch()  # changed or deleted
        (root / "dies.py").write_text("raise SystemExit(1)\n")
        for name, text in files.items():
            (root / f"test/{name}.py").write_text(text)
        here = {"ROOT": str(root), "CHANGES": str(root / "nothing")}
        allows = {"split": 600, "alone": ["test_quiet"], "runs": list(files)}
        for patcher in [
            mock.patch.multiple(worker, DELETED=str(root / "nothing"), **here),
            mock.patch.object(worker, "machine", return_value=L4),
            mock.patch.object(worker, "allows", return_value=allows),
            mock.patch.object(worker, "AT_ONCE", 2),
        ]:
            patcher.start()
            self.addCleanup(patcher.stop)
        queue = mock.patch.object(worker.modal, "Queue").start()
        self.addCleanup(mock.patch.stopall)
        started = worker.Worker()
        started.sync()
        two = ["test_env", "test_how"]  # each by a process of its own
        usual = started.collect(two, [], "qu-1")
        told(queue)
        env, how, quiet, _, killed = started.collect(list(files)[:5], [], "qu-1")
        done = told(queue)
        # Done first: the files whose forks end, as they do, then the one that
        # fails in its fork, which a signal ends: it says nothing
        self.assertEqual(sorted(done[:4]), list(files)[:4])
        self.assertEqual(done[4], "test_signal")
        self.assertEqual(killed["error"], "python exited with -9")
        # A file is forked once collect.py has imported what the file leads with,
        # and no more: one that sets the environment first finds it read as it set it
        self.assertEqual(env["ids"], ["test/test_env.py::slow tests True"])
        said = "__main__ test_how.py --use-pytest test_how.py . True True"
        self.assertEqual(how["ids"], [f"test/test_how.py::{said}"])
        both = [{**file, "seconds": 0} for file in (env, how)]
        self.assertEqual(both, [{**file, "seconds": 0} for file in usual])
        # A file that starts no pytest has no tests
        nothing = {"name": "test_quiet", "ids": [], "alone": True}
        self.assertEqual(quiet, {**how, **nothing, "seconds": quiet["seconds"]})
        # A second for each file: enough in a fork that has common_utils already.
        # The file that takes longer is ended, and then fails on its own too
        three = ["test_how", "test_slow", "test_quiet"]
        with mock.patch.object(worker, "COLLECT_SECONDS", 1):
            again, slow, _ = started.collect(three, [], "qu-1")
            done = told(queue)
            # If collect.py dies, each file is listed by a process of its own
            with mock.patch.object(worker, "FORKS", str(root / "dies.py")):
                alone = started.collect(three, [], "qu-1")
        self.assertEqual(done, [three[0], three[2], three[1]])
        self.assertEqual(again["ids"], how["ids"])
        self.assertEqual(slow["error"], "still collecting after 1 s")
        self.assertEqual([found["name"] for found in alone], three)
        self.assertEqual(alone[1]["error"], slow["error"])

    def test_leads(self) -> None:
        # What a file imports of torch before it does anything else, with the
        # packages: what it imports after a statement, or after a module beside it,
        # is not what it leads with
        texts = {
            "a.py": '"""Its docstring."""\nimport os, torch.nn as nn\nfrom torch import fx\n',
            "b.py": "# -*- coding: utf-8 -*-\nfrom torch.fx import (\n    Graph,\n)\nX = 1\nimport torch.nn\n",
            "c.py": "import numpy\nimport typing\nfrom a import nn\nimport torch\n",
            "d.py": "import os\nif os.name:\n    import torch\n",
            "e.py": "from . import a\nimport torch\n",
            "f.py": "not python\n",
        }
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "typing").mkdir()  # as test/typing, which is no package
            for name, text in texts.items():
                Path(tmp, name).write_text(text)
            paths = [f"{tmp}/{name}" for name in [*texts, "none.py"]]
            led = [sorted(collect.leads(path)) for path in paths]
            torch_nn, torch_fx = ["torch", "torch.nn"], ["torch", "torch.fx"]
            self.assertEqual(led, [torch_nn, torch_fx, [], [], [], [], []])
            # A module that needs CUDA to import is left to each file, as is one that
            # fails to import
            Path(tmp, "needs_cuda.py").write_text("import torch\ntorch.cuda.init()\n")
            with mock.patch.object(sys, "path", [tmp, *sys.path]):
                for name in ("needs_cuda", "no_such_module", "json"):
                    collect.bring(name)
        self.assertNotIn("needs_cuda", sys.modules)
        self.assertNotIn(collect.NVML_CHECK, os.environ)

    def test_serial(self) -> None:
        # pytest's plugin, as a container loads it to collect a file: it writes the
        # tests marked @serialTest, also those that the other arguments leave out
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "test_marked.py").write_text(MARKED)
            plugin = ["-p", worker.SERIAL, f"--ramjet-serial={tmp}/marked"]
            pytest = [sys.executable, "-m", "pytest", "--collect-only", "-q", tmp]
            output = {"capture_output": True, "text": True}
            collect = [*pytest, *plugin, "-k", "plain"]
            done = subprocess.run(collect, cwd=REPO_ROOT, **output)
            marked = Path(tmp, "marked").read_text()
        self.assertEqual(done.returncode, 0, done.stderr + done.stdout)
        self.assertIn("test_marked.py::test_plain", done.stdout)
        self.assertNotIn("test_alone", done.stdout)
        both = ["test_marked.py::test_alone", "test_marked.py::TestClass::test_alone"]
        self.assertEqual(marked.split("\n"), both)

    def test_timed(self) -> None:
        kept = {
            "test_a": ["test/test_a.py::A::test_here", "test/test_a.py::A::test_other"],
            "test_b": [f"test/test_b.py::B::test_{n}" for n in range(4)],
            "test_new": ["test/test_new.py::New::test_new"],
        }
        ci = {
            "test/test_a.py::A::test_here": (4.0, 1.0),
            "test/test_a.py::A::test_other": (None, 2.0),
            "test/test_b.py::B::test_0": (1.0, 5.0),
            "test/test_b.py::B::test_2": (3.0, 5.0),
        }
        with (
            mock.patch.object(times, "fetch", return_value=ci) as fetch,
            self.assertLogs(test.log, "DEBUG") as logs,
        ):
            timed = test.timed(kept, L4)
        fetch.assert_called_once_with(
            {"test/test_a.py": 2, "test/test_b.py": 4, "test/test_new.py": 1},
            "NVIDIA L4",
            1,
        )
        # CI's seconds: here, else anywhere, else its file's median, else a guess
        expected = {
            "test_a": [4.0, 2.0],
            "test_b": [1.0, 2.0, 3.0, 2.0],
            "test_new": [test.GUESS_SECONDS],
        }
        seconds = {name: [s / test.SLOWDOWN for _, s in timed[name]] for name in timed}
        self.assertEqual(seconds, expected)
        ids = {name: [nodeid for nodeid, _ in tests] for name, tests in timed.items()}
        self.assertEqual(ids, kept)
        sources = "3 tests with 1 x NVIDIA L4, 1 on other machines, 3 guessed"
        logged = [f"{r.levelname} {r.getMessage()}" for r in logs.records]
        self.assertEqual(logged, [f"DEBUG CI's times: {sources}"])

    def test_at_once(self) -> None:
        l4 = {"gpus": 1, "gpu_memory": 22.5}
        h100, b200 = {"gpus": 1, "gpu_memory": 79.6}, {"gpus": 1, "gpu_memory": 179.0}
        no_gpu = {"gpus": 0, "gpu_memory": None}
        cores = "running 3 test processes at once: 4 cores fit no more"
        memory = "running 3 test processes at once: 16 GiB fit no more"
        one = "running 1 test process at once: 1 core fits no more"
        more = "H100 fits 11 with --memory 65536"
        # --gpu, --cpu and --memory, the machine a container reads itself as, the
        # test processes it runs at once, and what ramjet says when its GPU has
        # room for more than its cores and memory fit
        cases = [
            ((None, None, None), no_gpu, 3, []),  # as CI runs at once
            ((None, 64, None), no_gpu, 48, []),
            ((None, 2, None), no_gpu, 1, []),
            ((None, 1, None), no_gpu, 1, []),  # one at least
            ((None, 16, 16384), no_gpu, 3, []),  # the memory fits no more
            (("L4", None, None), l4, 3, []),  # as CI runs at once on its L4
            (("T4", None, None), {"gpus": 1, "gpu_memory": 15.0}, 2, []),  # and T4
            (("L4", 16, None), l4, 3, []),  # the GPU's memory fits no more
            (("L4:2", None, None), {"gpus": 2, "gpu_memory": 22.5}, 3, []),  # of one
            (("H100", None, None), h100, 3, [cores, "H100 fits 11 with --cpu 16"]),
            (("H100", 16, None), h100, 11, []),
            (("H100", 16, 16384), h100, 3, [memory, more]),
            (("H100:8", None, None), {"gpus": 8, "gpu_memory": 79.6}, 11, []),
            # The cores that fit the nearest to the 25 that the GPU has room for
            (("B200", None, None), b200, 3, [cores, "B200 fits 24 with --cpu 32"]),
            (("B200", 32, None), b200, 24, []),
            (("A10", 1, None), l4, 1, [one, "A10 fits 3 with --cpu 4"]),
        ]
        sizes = (test.CORES_PER_GPU, test.MIB_PER_CORE)
        for asked, read, procs, says in cases:
            with self.subTest(asked=asked), self.assertLogs(test.log, "DEBUG") as logs:
                resources = run.resources(*asked, *sizes)
                self.assertEqual(test.at_once(resources, read), procs)
            said = [r for r in logs.records if r.levelname == "INFO"]
            self.assertEqual([r.getMessage() for r in said], says)
            # What to ask for goes on under the line
            under = [getattr(r, "continued", False) for r in said]
            self.assertEqual(under, [False, True][: len(said)])
        fit = "test processes that fit: 0 in 1 core, 0 in 4 GiB, 3 in a GPU's 22 GiB"
        self.assertEqual(logs.records[-1].getMessage(), fit)

    def test_meter(self) -> None:
        spans = {0.2: "1 s", 40: "40 s", 59.6: "1 min", 359: "6 min", 3600: "1 h"}
        spans[4350] = "1 h 12 min"
        self.assertEqual({s: progress.span(s) for s in spans}, spans)
        self.assertEqual(progress.clock(149.9), "2:29")
        meter = progress.Meter("L4", Decimal("3.60"), began=0.0, asked=0.0)
        asked = [10.0, 70.0, 100.0]
        with mock.patch.object(progress.time, "monotonic", side_effect=asked):
            meter.told(2, "collecting")  # after 10 s without a container
            meter.told(1, "collecting")  # after a minute with two
            meter.told(0, "running")  # after 30 s with one
        self.assertEqual((meter.containers, meter.billed), (0, 150.0))
        spent = {"waiting": 10.0, "collecting": 60.0, "running": 30.0}
        self.assertEqual(meter.spent, spent)
        self.assertEqual(meter.cost(100.0), "$0.15")
        meter.containers = 2  # which bill since Modal last said
        self.assertEqual(meter.cost(150.0), "$0.25")
        meter.built = Decimal("0.65")  # the run's build, which its price starts at
        self.assertEqual(meter.cost(150.0), "$0.90")
        self.assertEqual([meter.named(n) for n in range(3)], ["no L4", "1 L4", "2 L4"])
        cpu = progress.Meter(None, Decimal("0.25"))
        named = ["no container", "1 container", "2 containers"]
        self.assertEqual([cpu.named(n) for n in range(3)], named)
        # Of those the run can use now, which the two there are may be more than
        of = [plain(progress.paint(meter.of(n))) for n in (0, 1, 7)]
        self.assertEqual(of, ["2 L4", "1/1 L4", "2/7 L4"])
        of = [plain(progress.paint(cpu.of(n))) for n in (0, 1, 1500)]
        self.assertEqual(of, ["no container", "0/1 container", "0/1,500 containers"])
        self.assertEqual(progress.paint(meter.of(7)), "2/7\033[2m L4\033[0m")
        where = "10 s with no L4 • 1 min collecting • 30 s running"
        self.assertEqual(meter.where(), where)

    def test_collecting(self) -> None:
        def rows(state: Any, now: float, width: int = 60) -> list[str]:
            return [plain(progress.paint(row)) for row in state.rows(now, width)]

        meter = progress.Meter("L4", Decimal("1.12"), began=0.0, asked=0.0)
        state = progress.Collecting(meter, 152)
        waited = "no L4 yet • %s • nothing billed"
        self.assertEqual(rows(state, 120.0), [waited % "2:00"])
        # After TIP_SECONDS: the other GPUs CI tests on, if it tests on this one
        free = "Ctrl-C is free while nothing runs"
        tip = [waited % "9:22", f"try --gpu A10 or T4: {free}"]
        self.assertEqual(rows(state, 562.0), tip)
        other = progress.Meter("H100", Decimal("4.50"), began=0.0, asked=0.0)
        self.assertEqual(rows(progress.Collecting(other, 152), 562.0)[1], free)
        self.assertEqual(progress.line(state, 562.0), waited % "9:22")
        # A run that built has billed its build by then
        built = progress.Meter("L4", Decimal("1.12"), Decimal("0.65"), 0.0, 0.0)
        after = progress.Collecting(built, 152)
        self.assertEqual(rows(after, 120.0), ["no L4 yet • 2:00 • $0.65"])
        self.assertEqual(state.title(), "ramjet: no L4 yet")
        self.assertFalse(state.working(562.0))
        self.assertEqual(state.news(), (0, 0))
        # The container has 52 files back, and how long collecting has taken goes on
        # from the wait for it
        meter.containers, meter.asked, meter.billed = 1, 900.0, 120.0
        state.back.update(map(str, range(52)))
        facts = "collecting 34% • 52/152 files • 1/1 L4 • 15:01 • $0.04"
        self.assertEqual(rows(state, 901.0), [facts])
        self.assertEqual(progress.line(state, 901.0), facts)
        # In a row too narrow for them, facts give way: how far the files are, then
        # the cost, how long it has taken and the containers
        rest = facts[:11] + facts[17:]
        narrow = {53: rest, 47: rest[:-8], 39: rest[:-16], 31: rest[:23]}
        self.assertEqual({w: rows(state, 901.0, w)[0] for w in narrow}, narrow)
        self.assertEqual(rows(state, 901.0, 5), ["collecting 52/152 files"])
        self.assertEqual(state.title(), "ramjet: collecting")
        self.assertTrue(state.working(901.0))
        self.assertEqual(state.news(), (1, 3))
        # On a terminal: after the start of a line of the log, with a mark that
        # turns, and to a space short of the terminal's end at most
        (first,) = progress.drawn(state, 901.0, 13, progress.ROWS, 120)
        start = r"^ramjet \d\d:\d\d:\d\d     ⠸ "
        self.assertRegex(plain(first), start + re.escape(facts) + "$")
        self.assertIn("collecting \033[1m34%\033[0m", first)
        self.assertIn("52\033[2m/152 files\033[0m", first)
        self.assertTrue(first.endswith("\033[35m$0.04\033[0m"))
        (first,) = progress.drawn(state, 901.0, 13, progress.ROWS, 70)
        self.assertTrue(plain(first).endswith(f" ⠸ {rest[:-8]}"))
        meter.containers = 0  # the mark turns only while containers work
        (first,) = progress.drawn(state, 901.0, 13, 8, 80)
        self.assertRegex(plain(first), r"   · collecting 34% • 52/152 files")
        self.assertIn(" • 0/1 L4 • 15:01 • $0.04", rows(state, 901.0)[0])
        # Of three calls that aren't back, two have a container
        state.calls, meter.containers = 3, 2
        self.assertIn(" • 2/3 L4 • 15:01 • $0.04", rows(state, 901.0)[0])

    def test_running(self) -> None:
        def rows(now: float, width: int = 66) -> list[str]:
            return [plain(progress.paint(row)) for row in state.rows(now, width)]

        def wide(now: float, most: int = progress.ROWS, width: int = 60) -> list[str]:
            return [plain(progress.paint(r)) for r in state.wide(now, most, width)]

        def shard(*processes: tuple[worker.Process, int]) -> test.Shard:
            """A shard of 100 s whose test processes each run so many tests."""
            where = {f"{p}{n}": p for p, size in processes for n in range(size)}
            tests: dict[str, list[str]] = {}
            for nodeid, (name, _, _) in where.items():
                tests.setdefault(name, []).append(nodeid)
            return test.Shard(100.0, tests, {}, where)

        wrapped = ("dynamo/test_nested_graph_breaks_wrapped_once_more", 0, False)
        a, c = ("dynamo/test_a", 0, False), ("nn/test_c", 0, False)
        b1, b2 = ("test_b", 0, False), ("test_b", 1, False)
        d, alone = ("dynamo/test_d", 0, False), ("dynamo/test_d", 0, True)
        e = ("test_e", 0, False)
        shards = [
            shard((a, 10)),
            shard((b1, 12), (b2, 7), (c, 1)),
            shard((d, 28), (alone, 2)),
            shard((wrapped, 38), (e, 2)),
        ]
        meter = progress.Meter("L4", Decimal("3.60"), began=0.0, asked=180.0)
        meter.containers, meter.billed = 2, 100.0
        state = progress.Running(meter, shards, began=60.0)
        self.assertEqual(state.sizes, [10, 20, 30, 40])
        self.assertEqual(state.shares[1], {b1: 12, b2: 7, c: 1})
        state.back.append({"shard": 1, "seconds": 80.0})
        # A container is on a shard since it first said so. Then it says how far the
        # shard's test processes are: how many tests of each have ended, and which
        # run. One runs since it was first heard of
        case = ("test/test_b.py", "TestB", "test_case")
        on_d = {d: ("test/dynamo/test_d.py", "TestD", "test_case")}
        # The first of test_b's has ended its tests. A third is not of the shard
        done, extra = {b1: 12, b2: 3}, ("test_b", 2, False)
        running = {b1: None, b2: case, c: None, extra: None}
        times = [0.0, 100.0, 150.0, 160.0, 170.0, 175.0]
        with mock.patch.object(progress.time, "monotonic", side_effect=times):
            state.report(1, "ta-1", {}, {})
            state.report(2, "ta-2", {}, {})
            state.report(3, "ta-1", {}, {})
            state.report(2, "ta-2", {}, {b1: None, b2: None})
            state.report(2, "ta-2", done, running)
            state.report(3, "ta-1", {d: 5}, on_d)
        # The shards that aren't back, the longest on its shard first
        on = progress.Flight
        heard = {b1: 160.0, b2: 160.0, c: 170.0, extra: 170.0}
        on_b = on(2, 100.0, done, running, heard)
        self.assertEqual(state.flying(), [on_b, on(3, 150.0, {d: 5}, on_d, {d: 175.0})])
        state.failed = 2
        # The tests that have ended: a shard's 10, and 20 of the shards that run.
        # The two containers are of the three that the shards not back could each
        # have. The shards went out two minutes ago
        facts = "testing 30% • 30/100 tests • 2 FAILED • 2/3 L4 • 2:00 • $0.10"
        self.assertEqual(rows(180.0), [facts])
        # The price is the run's, not the tests': it starts at what collecting cost
        collected = progress.Collecting(meter, 152, back={"test_a"})
        self.assertEqual(state.facts(180.0)["cost"], collected.facts(180.0)["cost"])
        (painted,) = state.rows(180.0, 66)
        styled = "\033[1m30%\033[0m", "30\033[2m/100 tests\033[0m"
        styled += "\033[31m2 FAILED\033[0m", "\033[35m$0.10\033[0m"
        self.assertEqual([s for s in styled if s in progress.paint(painted)], [*styled])
        self.assertEqual(state.title(), "ramjet 30% • 2 failed")
        self.assertEqual(state.news(), (2, 3))
        # In a row too narrow for them, facts give way: how far the tests are, then
        # the cost, how long they have run and the containers. A line of the log,
        # which starts two columns in, is too narrow for the first
        rest = facts[:8] + facts[14:]
        self.assertEqual(progress.line(state, 180.0), rest)
        narrow = {60: rest, 54: rest[:-8], 46: rest[:-15], 39: rest[:31]}
        self.assertEqual({w: rows(180.0, w)[0] for w in narrow}, narrow)
        self.assertEqual(rows(180.0, 5), ["testing 30/100 tests • 2 FAILED"])
        self.assertIn("2/3 L4 • 4:00 • ", rows(300.0)[0])
        # A row for each test process that runs and has tests left, the longest on
        # its file first: its container, how long it has been on its file, how many
        # of its tests have ended, and the test it runs, or before its first, its
        # file. A container is its runner's name and its number, in the order the
        # containers came: ta-2 is the second
        processes = [
            " L4 2  0:20 [ 3/ 7] test_b.py::TestB::test_case",
            " L4 2  0:10 [ 0/ 1] test_c.py",
            " L4 1  0:05 [ 5/28] test_d.py::TestD::test_case",
        ]
        self.assertEqual(wide(180.0), processes)
        # The container is dim, as the count of tests is
        dimmed = "\033[2m L4 2 \033[0m 0:20 \033[2m[ 3/ 7] \033[0mtest_b.py"
        self.assertIn(dimmed, progress.paint(state.wide(180.0, progress.ROWS, 60)[0]))
        # As wide as the terminal at most, less a space: a name is cut at its end
        self.assertEqual(wide(180.0, width=30)[0], " L4 2  0:20 [ 3/ 7] test_b.p…")
        drawn = progress.drawn(state, 180.0, 0, progress.ROWS, 89)
        self.assertEqual([plain(row) for row in drawn[1:]], processes)
        self.assertTrue(plain(drawn[0]).endswith(f" ⠋ {facts}"))
        # No more rows than fit: the other processes are counted, from the column
        # the containers start in
        more = " … and 2 more running"
        short = [" L4 2  0:20 [3/7] test_b.py::TestB::test_case", more]
        self.assertEqual(wide(180.0, most=1), short)
        self.assertEqual(wide(180.0, most=0), [])
        # Modal moves a shard to another container: the first one's rows go. A
        # process has ended no more tests than it has
        long = (f"test/{wrapped[0]}.py", "TestW", "test_case")
        with mock.patch.object(progress.time, "monotonic", return_value=190.0):
            state.report(2, "ta-3", {}, {})
            state.report(4, "ta-2", {wrapped: 9, e: 99}, {wrapped: long, e: None})
        moved = [
            " L4 1  0:25 [ 5/28] test_d.py::TestD::test_case",
            " L4 2  0:10 [ 9/38] test_nested_graph_breaks_wrapped_once_…",
        ]
        self.assertEqual(wide(200.0), moved)
        self.assertEqual(state.tests(), (26, 100))
        # A process that has ended has no row, whatever tests it has left. The one
        # of test_d is started again, for the file's tests marked @serialTest
        with mock.patch.object(progress.time, "monotonic", return_value=210.0):
            state.report(3, "ta-1", {d: 20}, {alone: None})
            again = [moved[1].replace("0:10", "0:30"), " L4 1  0:10 [ 0/ 2] test_d.py"]
            self.assertEqual(wide(220.0), again)
            state.report(3, "ta-1", {d: 20, alone: 1}, {alone: on_d[d]})
        last = "0:10 [ 1/ 2] test_d.py::TestD::test_case"
        self.assertEqual(wide(220.0)[1], f" L4 1  {last}")
        # The shard moves on until the tenth container has it: every number is then
        # as wide as its two digits. Without a GPU, a runner is a CPU
        with mock.patch.object(progress.time, "monotonic", return_value=210.0):
            for n in range(4, 11):
                state.report(3, f"ta-{n}", {d: 20, alone: 1}, {alone: on_d[d]})
        tenth = [
            " L4  2  0:30 [ 9/38] test_nested_graph_breaks_wrapped_once…",
            f" L4 10  {last}",
        ]
        self.assertEqual(wide(220.0), tenth)
        with mock.patch.object(meter, "gpu", None):
            self.assertEqual(wide(220.0)[1], f" CPU 10  {last}")
        # A container more than the shards not back isn't counted. With every shard
        # back, the run can use none: those there are
        state.back.append({"shard": 4, "seconds": 20.0})
        meter.containers = 3
        self.assertIn(" • 2/2 L4 • 15:40 • $", rows(1000.0)[0])
        state.back.extend({"shard": shard, "seconds": 20.0} for shard in (2, 3))
        self.assertIn(" • 3 L4 • 15:40 • $", rows(1000.0)[0])
        self.assertEqual(progress.Running(meter, shards).working(0.0), False)

    def test_block(self) -> None:
        terminal = io.StringIO()
        with progress.Block(terminal) as block:
            self.assertEqual(terminal.getvalue(), "\033[?25l\033[22;2t")  # no cursor
            block.draw(["one", "two"], "ramjet 1%")
            drawn = terminal.getvalue()
            self.assertEqual(shows(drawn), ["one", "two", ""])
            self.assertIn("\033]2;ramjet 1%\a\033[?7l", drawn)  # named; rows are cut
            block.draw(["one", "two"], "ramjet 1%")
            self.assertEqual(terminal.getvalue(), drawn)  # the same: nothing to draw
            block.write("a long line of the log\nand its second\n")
            block.draw(["three"], "ramjet 1%")
            rows = ["a long line of the log", "and its second", "three", ""]
            self.assertEqual(shows(terminal.getvalue()), rows)
            self.assertEqual(terminal.getvalue().count("\033]2;"), 1)
            size = (progress.ROWS, progress.COLUMNS)  # as in 24 rows of 80
            self.assertEqual(block.room(), size)
        self.assertEqual(shows(terminal.getvalue()), rows[:2] + [""])
        self.assertTrue(terminal.getvalue().endswith("\033[23;2t\033[?25h"))

    def test_shown(self) -> None:
        meter = progress.Meter("L4", Decimal("1.12"))
        state = progress.Collecting(meter, 10)
        asked: list[int] = []

        def ask() -> None:
            asked.append(len(asked))
            if len(asked) == 2:  # as when Modal can't be asked
                raise ConnectionError("no answer")
            meter.containers = min(len(asked), 3)

        # No terminal: a line when there is news, and never two within APART_SECONDS
        often = mock.patch.multiple(progress, POLL_SECONDS=0.01, APART_SECONDS=0)
        with often, self.assertLogs(progress.log, "DEBUG") as logs:
            with progress.shown(state, ask):
                deadline = time.monotonic() + 60
                while len(logs.records) < 3:  # the third poll's containers are news
                    self.assertLess(time.monotonic(), deadline, "no third line")
                    time.sleep(0.01)
                time.sleep(0.05)  # no line without news
            first, second, third = (record.getMessage() for record in logs.records)
            self.assertEqual(second, "no word from Modal: ConnectionError('no answer')")
            line = r"^  collecting 0% • 0/10 files • 1/1 L4 • 0:\d\d • \$0\.0\d$"
            self.assertRegex(first, line)
            self.assertRegex(third, line)
            polls = len(asked)
            time.sleep(0.05)
            self.assertEqual(len(asked), polls)  # it stopped with the block
        # ... but one after QUIET_SECONDS without any
        quiet = {"POLL_SECONDS": 0.01, "APART_SECONDS": 0, "QUIET_SECONDS": 0.03}
        heard = self.assertLogs(progress.log, "INFO")
        with mock.patch.multiple(progress, **quiet):
            with heard as logs, progress.shown(state, ask):
                deadline = time.monotonic() + 60
                while len(logs.records) < 2:
                    self.assertLess(time.monotonic(), deadline, "no second line")
                    time.sleep(0.01)
        for record in logs.records[:2]:  # the same, but for how long it has taken
            self.assertRegex(record.getMessage(), line)
        # Modal is asked once more as the block ends
        del asked[:]
        with progress.shown(state, ask):
            pass
        self.assertEqual(asked, [0])
        # On a terminal: the block under the log's lines, gone when it ends
        terminal = io.StringIO()
        logger.log.addHandler(logger._HANDLER)
        self.addCleanup(logger.log.removeHandler, logger._HANDLER)
        fast = mock.patch.object(progress, "FRAME_SECONDS", 0.01)
        with fast, mock.patch.object(progress, "terminal", return_value=terminal):
            with progress.shown(state, ask):
                deadline = time.monotonic() + 60
                while "0/10 files" not in plain(terminal.getvalue()):
                    self.assertLess(time.monotonic(), deadline, "no block")
                    time.sleep(0.01)
                logger.log.error("a line")
                line, facts, _ = shows(plain(terminal.getvalue()))
        self.assertRegex(line, r"^ramjet \d\d:\d\d:\d\d error a line$")
        start = r"^ramjet \d\d:\d\d:\d\d     . "
        row = r"collecting 0% • 0/10 files • 1/1 L4 • \d:\d\d • \$0\.0\d$"
        self.assertRegex(facts, start + row)
        self.assertEqual(shows(plain(terminal.getvalue())), [line, ""])
        self.assertIs(logger._HANDLER.stream, sys.stderr)

    def test_calls(self) -> None:
        workers = mock.MagicMock()  # stands in for the class on Modal
        lost = ConnectionError("container lost")
        names = ["test_a", "test_b", "test_c", "test_d", "test_e"]
        files = [{"name": name, "ids": [f"{name}::test"]} for name in names]
        collector = workers.return_value.collect
        collector.get_current_stats.return_value.num_total_runners = 0
        queues = mock.patch.object(test.modal, "Queue").start()
        self.addCleanup(mock.patch.stopall)
        queue = queues.ephemeral.return_value.__enter__.return_value
        queue.get_many.return_value = []

        def collect(calls: Any, **options: Any) -> Any:
            # Two files a call at most: three calls at once, each with files that
            # follow each other, and with the queue to say on how far it is
            shares = [names[:1], names[1:3], names[3:]]
            given = [(share, ["-v"], queue.object_id) for share in shares]
            unordered = {"order_outputs": False, "return_exceptions": True}
            self.assertEqual((calls, options), (given, unordered))
            wait_for(self, logs, r"  no L4 yet • 0:\d\d • nothing billed")
            collector.get_current_stats.return_value.num_total_runners = 2
            took = r" L4 • 0:\d\d • \$0\.00"
            wait_for(self, logs, r"  collecting 0% • 0/5 files • 2/3" + took)
            # Each file as it is done, which is in no order: read again at the next
            # poll, as when Modal starts a call anew and its container says it again
            queue.get_many.return_value = ["test_e", "test_b"]
            wait_for(self, logs, r"  collecting 40% • 2/5 files • 2/3" + took)
            yield files[3:]  # the calls come back in no order either
            wait_for(self, logs, r"  collecting 60% • 3/5 files • 2/2" + took)
            yield lost
            yield files[:1]

        collector.starmap = collect
        meter = progress.Meter("L4", Decimal("1.12"))
        # A line at every poll, as if each had news
        often = {"POLL_SECONDS": 0.01, "APART_SECONDS": 0, "QUIET_SECONDS": 0}
        often = mock.patch.multiple(progress, **often)
        few = mock.patch.object(test, "CALL_FILES", 2)
        with few, often, self.assertLogs(test.log, "INFO") as logs:
            found = test.collect(workers, names, ["-v"], meter)
        # In the order of names, without the files of the call that Modal lost
        self.assertEqual(found, [files[0], *files[3:]])
        failed = f"collecting failed:\n{lost!r}"
        self.assertIn(failed, [record.getMessage() for record in logs.records])
        queue.get_many.assert_called_with(test.QUEUE_ITEMS, block=False)
        queue.get_many.return_value = []
        # If Modal takes no call, no file's tests are back
        collector.starmap = mock.Mock(side_effect=lost)
        with self.assertLogs(test.log, "ERROR") as logs:
            self.assertEqual(test.collect(workers, names, ["-v"], meter), [])
        self.assertEqual(logs.records[0].getMessage(), failed)
        # Shards: one call each, to the same containers, first done first. Each
        # container says on a queue how far its shard's test processes are
        long, short = {"test_a": ["a1", "a2"]}, {"test_b": ["b1"]}
        told = {"test_a": 900.0}  # what run_test.py hears that the file takes
        second = ("test_a", 1, False)  # of the two processes it runs the file as
        halves = {"a1": ("test_a", 0, False), "a2": second}
        whole = {"b1": ("test_b", 0, False)}
        shards = [test.Shard(9.0, long, told, halves)]
        shards.append(test.Shard(5.0, short, {}, whole))
        state = progress.Running(meter, shards)
        runner = workers.return_value.run
        stats = runner.get_current_stats.return_value
        stats.num_total_runners = 0

        def ran(calls: Any, **options: Any) -> Any:
            asked = {"kwargs": told_all, "return_exceptions": True}
            started = queue.object_id
            call = (1, long, told, halves, started)
            self.assertEqual(calls, [call, (2, short, {}, whole, started)])
            self.assertEqual(options, {**asked, "order_outputs": False})
            took = r" L4 • 0:\d\d • \$0\.00"
            wait_for(self, logs, r"  testing 0% • 0/3 tests • 0/2" + took)
            stats.num_total_runners = 1
            queue.get_many.return_value = [(2, "ta-1", {}, {})]
            wait_for(self, logs, r"  testing 0% • 0/3 tests • 1/2" + took)
            self.assertEqual(state.flying()[0].shard, 2)
            yield {"shard": 2, "seconds": 4.0}
            # The tests that have ended: the shard's, and one of the next shard
            a2 = ("test/test_a.py", "A", "a2")
            queue.get_many.return_value = [(1, "ta-1", {second: 1}, {second: a2})]
            wait_for(self, logs, r"  testing 66% • 2/3 tests • 1/1" + took)
            yield lost

        runner.starmap = ran
        # What every shard's container is told: three test processes of four threads
        told_all = {"args": ["-v"], "procs": 3, "threads": 4}
        with often, self.assertLogs(test.log, "INFO") as logs:
            results = list(test.run(workers, told_all, state))
        self.assertEqual(results, [{"shard": 2, "seconds": 4.0}, lost])
        queue.get_many.assert_called_with(test.QUEUE_ITEMS, block=False)
        self.assertEqual(meter.spent.keys(), {"waiting", "collecting", "running"})

    def test_outcomes(self) -> None:
        a = "test/test_a.py::A::"
        boom = "AssertionError: boom\n\nmore"
        first = [
            json.dumps({"type": "report", "schema_version": "1.0"}),
            run_line(f"{a}test_pass", "passed"),
            run_line(f"{a}test_fail", "failed", "AssertionError: first", at=10),
            run_line(f"{a}test_flaky", "passed", at=21),  # after it failed, below
            run_line(f"{a}test_flaky", "error", "RuntimeError: once", at=20),
            run_line(f"{a}test_skip", "skipped", "no CUDA"),
            run_line(f"{a}test_other", "failed", "not sent"),
            run_line("test/test_a.py::test_function", "xfailed", "known"),
            run_line("test/test_a.py::Outer::Inner::test_nested", "passed"),
            run_line(f"{a}test_param[a::b]", "passed"),
        ]
        # A later process ran the failure again, and died on the test after it
        second = [
            run_line(f"{a}test_fail", "failed", boom, at=50),
            run_line(f"{a}test_crash", "crashed", "exit code -6 (SIGIOT)", at=60),
            run_line(f"{a}test_missing", "passed")[:40],
        ]
        reports = {"1.jsonl": "\n".join(first), "2.jsonl": "\n".join(second)}
        sent = {
            f"{a}test_pass": ("passed", ""),
            f"{a}test_fail": ("failed", boom),
            f"{a}test_flaky": ("flaky", "RuntimeError: once"),
            f"{a}test_skip": ("skipped", "no CUDA"),
            "test/test_a.py::test_function": ("xfailed", "known"),
            "test/test_a.py::Outer::Inner::test_nested": ("passed", ""),
            f"{a}test_param[a::b]": ("passed", ""),
            f"{a}test_crash": ("crashed", "exit code -6 (SIGIOT)"),
            f"{a}test_missing": ("not_run", ""),  # the line of its run is torn
        }
        self.assertEqual(test.outcomes(reports, list(sent)), sent)

    def test_execute(self) -> None:
        a, flaky = "test/test_a.py::A::", "test/test_b.py::B::test_flaky"
        ok, bad, gone = f"{a}test_pass", f"{a}test_fail", f"{a}test_gone"
        lost = "test/test_c.py::C::test_lost"
        shards = [
            test.Shard(120.0, {"test_a": [ok, bad, gone]}),
            test.Shard(60.0, {"test_b": [flaky]}),
            test.Shard(30.0, {"test_c": [lost]}),
        ]
        boom = "AssertionError: boom\n\nand more"
        one = [run_line(ok, "passed"), run_line(bad, "failed", boom)]
        two = [run_line(flaky, "failed", "once", at=1), run_line(flaky, "passed", at=2)]
        told = {"a.jsonl": "\n".join(one), "b.jsonl": "\n".join(two)}
        reports = {1: {"a.jsonl": told["a.jsonl"]}, 2: {"b.jsonl": told["b.jsonl"]}}
        codes = {1: 1, 2: 0}
        error = ConnectionError("container lost")

        def ran(number: int, seconds: float) -> Any:
            done = {"shard": number, "code": codes[number], "seconds": seconds}
            output = gzip.compress(f"output {number}\n".encode())
            told = {"reports": reports[number], "output": output}
            # Shard 2 used more cores and memory than a container asks for
            used = {"cores": 2.5 * number, "memory": 9000 * number, "gpu_memory": 512}
            return {**done, **told, **used, "container": f"ta-{number}"}

        workers = mock.MagicMock()
        results = [ran(2, 50.0), error, ran(1, 130.0)]
        runner = workers.return_value.run
        runner.starmap.return_value = results
        runner.get_current_stats.return_value.num_total_runners = 0
        queues = mock.patch.object(test.modal, "Queue").start()
        self.addCleanup(mock.patch.stopall)
        queues.ephemeral.return_value.__enter__.return_value.get_many.return_value = []
        meter = progress.Meter("L4", Decimal("3.60"))
        asked = run.resources("L4", None, None, test.CORES_PER_GPU, test.MIB_PER_CORE)
        told_all = {"args": ["-v"], "procs": 3, "threads": 4}
        with tempfile.TemporaryDirectory() as tmp:
            kept = Path(tmp, "test-1")
            work = (workers, shards, told_all, kept, meter, asked, APP)
            with self.assertLogs(test.log, "DEBUG") as logs:
                with contextlib.redirect_stdout(io.StringIO()) as stdout:
                    self.assertEqual(test.execute(*work), 3)
            saved = {str(p.relative_to(kept)): p.read_text() for p in kept.rglob("*.*")}
        logged = [f"{r.levelname} {plain(r.getMessage())}" for r in logs.records]
        expected = [
            "DEBUG shard 2 took 50 s on container ta-2: exit code 0",
            "DEBUG shard 2 used 5.0 cores, 18000 MiB, and 512 MiB of a GPU",
            f"WARNING flaky: {flaky}",
            "ERROR Modal lost a shard: ConnectionError('container lost')",
            "DEBUG shard 1 took 130 s on container ta-1: exit code 1",
            "DEBUG shard 1 used 2.5 cores, 9000 MiB, and 512 MiB of a GPU",
            f"ERROR failed: {bad}",
            "ERROR   AssertionError: boom",
            f"ERROR shard 1 exited 1: 1 test didn't run, see {kept}/shard-1.log",
            "WARNING 1 shard used more than a container's 4 cores",
            "WARNING shard 2 the most, 5.0 cores: see --cpu",
            "WARNING 1 shard used more than a container's 16 GiB of memory",
            "WARNING shard 2 the most, 17.6 GiB: see --memory",
            "DEBUG on 2 L4: 1 s with no L4",  # where the time went: no container came
            # The summary, which nothing follows: the states that need no look, those
            # that fail the run, the directory and the app's address
            "INFO $0.00 for 5 tests in 1 s:",
            "INFO 1 passed • 1 flaky",
            "INFO 1 FAILED • 2 NOT RUN",
            f"INFO {kept}",
            f"INFO {APP}",
        ]
        self.assertEqual(logged, expected)
        # Of the states, only those that fail the run have a color
        fine, failed = (r.getMessage() for r in logs.records[15:17])
        dot = logger.dim(" • ")
        self.assertEqual(fine, dot.join(["1 passed", "1 flaky"]))
        self.assertEqual(failed, dot.join(map(logger.red, ["1 FAILED", "2 NOT RUN"])))
        # The first failure and the summary ring the bell. The rows that go on under
        # a line have no start of their own: the last two have a name there
        records = list(enumerate(logs.records))
        ringing = [i for i, r in records if getattr(r, "ring", False)]
        self.assertEqual(ringing, [6, 14])
        under = [i for i, r in records if getattr(r, "continued", False)]
        self.assertEqual(under, [10, 12, 15, 16, 17, 18])
        names = [getattr(r, "margin", None) for r in logs.records[15:]]
        self.assertEqual(names, [None, None, "Logs", "Modal"])
        failing = {bad: "failed", gone: "not_run", lost: "not_run"}
        printed = [json.loads(line) for line in stdout.getvalue().splitlines()]
        rows = [{"nodeid": nodeid, "outcome": o} for nodeid, o in failing.items()]
        self.assertEqual(printed, rows)
        files = {"shard-1.log": "output 1\n", "shard-2.log": "output 2\n"}
        kept_reports = {f"reports/{name}": report for name, report in told.items()}
        self.assertEqual(saved, {**files, **kept_reports})

        # Ctrl-C, before a shard is back, while one is awaited or while one that is
        # back is handled: the summary of the shards handled before, without states
        # if there are none, and the Ctrl-C goes on
        def awaited(calls: Any, **options: Any) -> Any:
            yield ran(2, 50.0)
            raise KeyboardInterrupt

        def handled(calls: Any, **options: Any) -> Any:
            yield ran(2, 50.0)
            with mock.patch.object(test, "outcomes", side_effect=KeyboardInterrupt):
                yield ran(1, 130.0)

        none = mock.Mock(side_effect=KeyboardInterrupt)
        one = ["INFO $0.00 for 1 test in 1 s:", "INFO 1 flaky"]
        cases = [
            (none, "no", 5, ["INFO $0.00 for 0 tests in 1 s"]),
            (awaited, 1, 4, one),
            (handled, 2, 4, one),
        ]
        for starmap, used, left, counted in cases:
            runner.starmap = starmap
            with tempfile.TemporaryDirectory() as tmp:
                work = (workers, shards, told_all, Path(tmp, "out"), meter, asked, APP)
                with self.assertLogs(test.log, "DEBUG") as logs:
                    with contextlib.redirect_stdout(io.StringIO()) as stdout:
                        with self.assertRaises(KeyboardInterrupt):
                            test.execute(*work)
            logged = [f"{r.levelname} {plain(r.getMessage())}" for r in logs.records]
            warned = f"WARNING stopped by Ctrl-C: {left} of 5 tests did not run"
            self.assertIn(warned, logged)
            where = f"DEBUG on {used} L4: 1 s with no L4"
            stopped = [where, *counted, f"INFO {tmp}/out", f"INFO {APP}"]
            self.assertEqual(logged[-len(stopped) :], stopped)
            self.assertEqual(stdout.getvalue(), "")  # of those, none failed
            self.assertFalse(any(getattr(r, "ring", False) for r in logs.records))
        # Every state a test can end in, in the summary's order
        d, again = "test/test_d.py::D::test_", "test/test_d.py::D::test_flaky"
        ended = ["passed", "xfailed", "skipped", *test.FAILED]
        lines = [run_line(d + state, state) for state in ended]
        lines += [run_line(again, "failed"), run_line(again, "passed", at=1)]
        reports[1] = {"d.jsonl": "\n".join(lines)}
        runner.starmap = mock.Mock(return_value=[ran(1, 9.0)])
        sent = [d + state for state in (*ended, "flaky", "not_run")]
        shards = [test.Shard(60.0, {"test_d": sent})]
        with tempfile.TemporaryDirectory() as tmp:
            work = (workers, shards, told_all, Path(tmp, "out"), meter, asked, APP)
            with self.assertLogs(test.log, "INFO") as logs:
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(test.execute(*work), 6)
        summary = [
            "$0.00 for 10 tests in 1 s:",
            "1 passed • 1 flaky • 1 xfailed • 1 skipped",
            "1 FAILED • 1 ERROR • 1 CRASHED • 1 TIMED OUT • 1 XPASSED • 1 NOT RUN",
        ]
        self.assertEqual([plain(r.getMessage()) for r in logs.records[-5:-2]], summary)

    def test_run_shard(self) -> None:
        one, two = "test/test_a.py::A::test_one", "test/test_a.py::A::test_two"
        three = "test/nn/test_b.py::B::test_three"
        tests = {"test_a": [one, two], "nn/test_b": [three]}
        told = {"test_a": 1500.0, "nn/test_b": 300.0}
        # The test process that runs each: test_a is two, and nn/test_b one
        first, second = ("test_a", 0, False), ("test_a", 1, False)
        other = ("nn/test_b", 0, False)
        where = {one: first, two: second, three: other}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (root / "test").mkdir()
            (root / "test/run_test.py").write_text(RUN_TEST)
            there = {"ROOT": str(root), "DISABLED": "print('fetched', flush=True)"}
            modal_sets = {"PYTHONPATH": "/pkg/:/root/", "MODAL_TASK_ID": "ta-9"}
            # As many threads as the container has cores, for each library
            modal_sets |= {"OMP_NUM_THREADS": "64", "OPENBLAS_NUM_THREADS": "64"}
            there["TELL_SECONDS"] = 0.1
            there["GPU_MEMORY"] = "echo 512 4096"  # stands in for nvidia-smi
            queues = mock.patch.object(worker.modal, "Queue")
            with mock.patch.multiple(worker, **there), queues as queue:
                with mock.patch.dict(os.environ, modal_sets):
                    started = worker.Worker()
                    # Two test processes at once, of five threads each
                    shard = (3, tests, told, where, "qu-1", 2, 5)
                    ran = started.run(*shard, ["-k", "x"])
                    # run_test.py reads the files' times where CI keeps them
                    known = json.loads((root / worker.TIMES).read_text())
                    self.assertEqual(known, {"default": {"default": told}})
                    # The container says on the queue that it starts the shard, and
                    # then how far its test processes are. A test of test_a's first
                    # has ended. Those that run: test_a's second, with its test in
                    # flight, nn/test_b's, and test_a's first, which ended and was
                    # started again for its tests marked @serialTest
                    queue.from_id.assert_called_once_with("qu-1")
                    put = queue.from_id.return_value.put
                    said = [call.args[0] for call in put.call_args_list]
                    self.assertEqual(said[0], (3, "ta-9", {}, {}))
                    running = {second: worker.identity(two), other: None}
                    running[("test_a", 0, True)] = None
                    self.assertEqual(said[-1], (3, "ta-9", {first: 1}, running))
                    put.assert_called_with(said[-1], block=False)
                    # One that hangs is given up on, with every process it started,
                    # and one that can't say how far it is runs all the same
                    put.side_effect = ConnectionError("no queue")
                    with mock.patch.object(worker, "GIVE_UP_SECONDS", 2):
                        shard = (4, tests, told, where, "qu-1", 2, 5)
                        hung = started.run(*shard, ["--hang"])
        self.assertEqual((ran["shard"], ran["code"], ran["container"]), (3, 3, "ta-9"))
        printed = gzip.decompress(ran["output"]).decode().splitlines()
        self.assertEqual((printed[0], printed[-1], len(printed)), ("fetched", "ran", 8))
        self.assertGreater(ran["seconds"], 0)
        # What the shard used: memory is the machine's, as a container reads its own
        self.assertGreater(ran["memory"], 0)
        self.assertGreaterEqual(ran["cores"], 0)
        self.assertEqual(ran["gpu_memory"], 4096)
        launched = json.loads(ran["reports"]["test_a-1.report.jsonl"])
        ours, theirs = launched["argv"][:6], launched["argv"][6:]
        reports = ours.pop(2).removeprefix("--save-torchci-reports=")
        self.assertEqual(Path(reports).name, "reports")
        named = ["--keep-going", "--pipe-logs", "-i", "test_a", "nn/test_b"]
        self.assertEqual(ours, named)
        ids = theirs.pop(1).removeprefix("--nodeid-file=")
        self.assertEqual(Path(ids).parent, Path(reports).parent)
        lists = ["--", "--import-slow-tests", "--import-disabled-tests", "-k", "x"]
        self.assertEqual(theirs, lists)
        self.assertEqual(launched["ids"], "\n".join([one, two, three]))
        self.assertEqual((launched["cwd"], launched["PYTHONPATH"]), (str(root), None))
        # run_test.py is told how many test processes to run at once, and every
        # library how many threads each has: torch's, and those Modal sets
        sized = launched["sized"]
        self.assertEqual(sized.pop("PYTORCH_NUM_TEST_PROCS"), "2")
        libraries = {"OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"}
        self.assertLessEqual(libraries, sized.keys())
        self.assertEqual(set(sized.values()), {"5"})
        self.assertEqual((hung["shard"], hung["code"]), (4, None))
        # A container fetches CI's list of disabled tests once
        self.assertNotIn(b"fetched", gzip.decompress(hung["output"]))
        left = json.loads(hung["reports"]["test_a-1.report.jsonl"])["child"]
        deadline = time.monotonic() + 60
        while Path(f"/proc/{left}").exists():  # the kernel takes a moment to end it
            self.assertLess(time.monotonic(), deadline, "the shard's process lives on")
            time.sleep(0.01)

    def test_sharded(self) -> None:
        start, split = test.START_SECONDS, 600
        # CI's plugin, which gives each test of a file to one of its processes
        path = REPO_ROOT / "test/pytest_shard_custom.py"
        spec = importlib.util.spec_from_file_location("pytest_shard_custom", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        plugin = module.PytestShardPlugin(None)

        def pick(name: str, hashes: list[int]) -> list[str]:
            """A node id of the file for each hash, as its remainder by 6: by 3 and
            by 2, the process of three and of two that CI's plugin gives it."""
            ids = (f"test/{name}.py::T::test_{n}" for n in itertools.count())
            items = (mock.Mock(nodeid=nodeid) for nodeid in ids)
            kept = plugin.filter_items_by_shard
            mine = (next(i for i in items if kept([i], h + 1, 6)) for h in hashes)
            return [item.nodeid for item in mine]

        big, two = pick("big", [0, 1, 2, 3, 4, 5]), pick("two", [0, 1])
        timed = {
            "big": [(nodeid, 100.0) for nodeid in big],  # three processes, twice
            "two": [(nodeid, 100.0) for nodeid in two],  # two processes
            "a": [("a::0", 100.0)],
            "b": [("b::0", 60.0)],
            "c": [("c::0", 30.0)],
            "solo": [("solo::0", 15.0), ("solo::1", 15.0)],
            "huge": [("huge::0", 400.0)],  # longer than a shard
        }
        shards = test.sharded(timed, ["solo"], {}, 3, split)
        planned = [(round(p.seconds - start, 1), p.tests, p.times) for p in shards]
        one, three = split / 2, 2.5 * split  # what run_test.py hears a file takes
        solo = {"solo": ["solo::0", "solo::1"]}
        expected = [
            (400.0, {"huge": ["huge::0"]}, {"huge": one}),
            # run_test.py starts the processes in the order of their files' names,
            # three at once: c takes the place of the first of big's that ends
            (130.0 + start, {"big": big[:3], "c": ["c::0"]}, {"big": three, "c": one}),
            # solo runs alone, once big's processes have ended
            (130.0 + start, {"big": big[3:], **solo}, {"big": three, "solo": one}),
            # b would keep the second of two's processes waiting
            (100.0, {"two": two, "a": ["a::0"]}, {"two": 1.5 * split, "a": one}),
            (60.0, {"b": ["b::0"]}, {"b": one}),
        ]
        self.assertEqual(planned, expected)
        # The test process that runs each test: one of the three that big is, by
        # CI's plugin, and the one that solo is
        where = {nodeid: ("big", n, False) for n, nodeid in enumerate(big[3:])}
        where |= dict.fromkeys(solo["solo"], ("solo", 0, False))
        self.assertEqual(shards[2].where, where)
        # Tests marked @serialTest run after the others, alone: big starts once more
        # for each of its processes that has one, here the second and the third,
        # whether the test is of this shard or not
        short, elsewhere = big[4], big[5]
        tests = [*timed["big"][:3], (short, 5.0)]
        marked = {"big": {short, elsewhere}}
        (shard,) = test.sharded({"big": tests}, [], marked, 3, split)
        self.assertEqual(round(shard.seconds, 1), round(100.0 + 3 * start + 5.0, 1))
        self.assertEqual(shard.tests, {"big": [*big[:3], short]})
        self.assertEqual(shard.times, {"big": three})
        where = {nodeid: ("big", n, False) for n, nodeid in enumerate(big[:3])}
        self.assertEqual(shard.where, {**where, short: ("big", 1, True)})


if __name__ == "__main__":
    run_tests()
