# Owner(s): ["module: devx"]

import contextlib
import getpass
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
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


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from tools.ramjet import common, inputs, lineage, sync


HAS_MODAL = importlib.util.find_spec("modal") is not None
if HAS_MODAL:
    import tools.ramjet.__main__ as cli
    from tools.ramjet import build, run

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
    "cpu_hour_cost_sandbox": Decimal("0.1419"),
    "mem_gib_hour_cost_sandbox": Decimal("0.024"),
    "gpu_hour_cost_a10g": Decimal("1.10"),
    "gpu_hour_cost_a100_40gb": Decimal("2.10"),
    "gpu_hour_cost_a100_80gb": Decimal("2.50"),
    "gpu_hour_cost_b200": Decimal("6.25"),
    "gpu_hour_cost_h100": Decimal("3.95"),
    "gpu_hour_cost_rtx6000": Decimal("3.03"),
}
CMAKE_LISTS = """\
cmake_minimum_required(VERSION 3.18)
project(demo C)
add_library(demo demo.c)
"""


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

    def exec(self, *args: str, workdir: str, **streams: Any) -> subprocess.Popen[str]:
        cwd = self.root / Path(workdir).relative_to("/pytorch")
        return subprocess.Popen(args, cwd=cwd, stderr=subprocess.PIPE, text=True)


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

        def fake_build(*args: Any, clone: bool) -> str:
            clones.append(clone)
            args[-1].write_text("lib.cpp\n")  # the inputs, as the builder exports them
            return f"im-{len(clones)}"

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
        ) -> lineage.Build:
            now = time.time() + days_later * 24 * 3600
            date = f"{now:.0f} +0000"  # git's clock too, which dates the snapshot
            env = {"GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date}
            with mock.patch.dict(os.environ, env):
                with mock.patch.object(cli.time, "time", return_value=now):
                    commit = sync.snapshot_tree()
                    return cli.image_for(None, commit, self.tmp, "13.2", python, {})

        self.write("lib.cpp", "int a;\n")
        first = image_for()
        self.assertEqual((first.image_id, clones), ("im-1", [True]))
        self.write("edited.py", "a Python edit\n")
        self.assertEqual(image_for(), first)
        self.write("lib.cpp", "int b;\n")
        self.write(".ci/docker/README.md", "a new CI image tag\n")  # same lineage
        self.assertEqual((image_for().image_id, clones), ("im-2", [True, False]))
        self.assertEqual(image_for(days_later=1).image_id, "im-2")
        self.assertEqual(image_for(days_later=4.1).image_id, "im-3")  # renewed
        self.assertEqual(image_for(days_later=10).image_id, "im-4")  # expired
        self.assertEqual(image_for(days_later=10, python="3.11").image_id, "im-4")
        self.assertEqual(image_for(days_later=10, python="3.13").image_id, "im-5")
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

    @unittest.skipUnless(HAS_MODAL, "needs modal")
    @unittest.skipUnless(shutil.which("setsid"), "needs util-linux's setsid")
    def test_execute(self) -> None:
        (self.repo / "sub").mkdir()
        command = ["sh", "-c", 'printf %s "$1" > out; exit 3', "sh", "a b"]
        with mock.patch.object(run, "PID_FILE", str(self.tmp / "pid")):
            self.assertEqual(run.execute(LocalRunner(self.repo), command, "sub"), 3)
        self.assertEqual((self.repo / "sub/out").read_text(), "a b")  # as given
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
        self.assertEqual(parse("--", "ls"), (None, *defaults, ["ls"]))
        command = ["pytest", "test/test_nn.py", "-k", "linear or conv"]
        argv = ["--gpu", "H100", "--", *command]
        self.assertEqual(parse(*argv), ("H100", *defaults, command))
        cuda, python = f"--build-cuda={cudas[-1]}", f"--python-version={pythons[-1]}"
        options = ["--cpu=16", "--memory=131072", cuda, python, "--log-level=debug"]
        argv = ["--gpu=b200+:8", *options, "--", "nvidia-smi"]  # in any case
        expected = ("B200+:8", 16, 131072, cudas[-1], pythons[-1], "debug", argv[-1:])
        self.assertEqual(parse(*argv), expected)
        argv = ["--", "a", "--", "b"]  # later --s are the command's
        self.assertEqual(parse(*argv), (None, *defaults, ["a", "--", "b"]))
        bad = [
            [],  # no command
            ["--gpu", "L4", "--"],
            ["ls"],  # the command follows a --
            ["--gpu", "--", "ls"],
            ["--gpu", "Z100", "--", "ls"],
            ["--gpu", "A10:5", "--", "ls"],  # A10s go up to 4
            ["--cpu=3", "--", "ls"],
            ["--memory=1000", "--", "ls"],
            ["--build-cuda=9.9", "--", "ls"],
            ["--python-version=3.9", "--", "ls"],
            ["--log-level=dev", "--", "ls"],
        ]
        for argv in bad:
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                parse(*argv)


if __name__ == "__main__":
    run_tests()
