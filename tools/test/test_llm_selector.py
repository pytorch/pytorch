from __future__ import annotations

import io
import json
import re
import shlex
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from tools.testing.discover_tests import BLOCKLISTED_PATTERNS, TESTS
from tools.testing.llm_selector import selection
from tools.testing.llm_selector.selection import (
    DRIVER_MAPPINGS,
    needs_run_all,
    normalize_test_name,
    Selection,
)


sys.path.remove(str(REPO_ROOT))


CANDIDATES = {
    "test_nn",
    "test_ops",
    "dynamo/test_misc",
    "cpython/v3_13/test_set",
    "inductor/test_torchinductor",
}


def answer(**kw: Any) -> dict:
    """A well-formed model answer selecting nothing, with `kw` replacing its keys."""
    return {"run_all": False, "reason": "", "sweeps": [], "dirs": [], "tests": [], **kw}


def inductor_includes(sh: str) -> dict[str, set[str]]:
    """The files each function of the shell script `sh` passes to `run_test.py
    --inductor` by --include or -i, under "" outside a function. Raises
    ValueError on such a call whose files it cannot read."""
    runs: dict[str, set[str]] = {}
    func = ""
    for line in sh.replace("\\\n", " ").splitlines():
        if m := re.match(r"(\w+)\s*\(\)\s*\{", line):
            func = m.group(1)
        elif line.startswith("}"):
            func = ""
        if "run_test.py" not in line:
            continue
        lex = shlex.shlex(line, posix=True, punctuation_chars=True)
        lex.whitespace_split = True
        words = list(lex)
        if (
            not any(w.endswith("run_test.py") for w in words)
            or "--inductor" not in words
        ):
            continue
        tests: list[str] = []
        for i, w in enumerate(words):
            if w in ("--include", "-i"):
                for v in words[i + 1 :]:
                    if v[:1] in "-;&|<>":
                        break
                    tests.append(v)
            elif w.startswith(("--include=", "-i=")):
                raise ValueError(f"unreadable run_test.py --inductor call: {line}")
        if not tests or any("$" in t for t in tests):
            raise ValueError(f"unreadable run_test.py --inductor call: {line}")
        runs.setdefault(func, set()).update(tests)
    return runs


class TestSelection(unittest.TestCase):
    def test_normalize(self) -> None:
        for name in [
            "test_nn",
            "test/test_nn.py",
            "./test/test_nn.py",
            "test_nn [12m]",
            "test_nn.py::TestNN",
        ]:
            self.assertEqual(normalize_test_name(name, CANDIDATES), {"test_nn"})
        self.assertEqual(
            normalize_test_name("dynamo.test_misc", CANDIDATES), {"dynamo/test_misc"}
        )
        self.assertEqual(normalize_test_name("test_nonexistent", CANDIDATES), set())
        # A file a driver runs names the driver, but a misspelling a driver's regex matches is no file.
        aot = {"test_cpp_extensions_aot_ninja", "test_cpp_extensions_aot_no_ninja"}
        cands = CANDIDATES | {"test_jit"} | aot
        paths = {"test/jit/test_freezing.py", "test/test_cpp_extensions_aot.py"}
        for name, expected in [
            ("jit/test_freezing", {"test_jit"}),
            ("test/jit/test_freezing.py", {"test_jit"}),
            ("jit.test_freezing", {"test_jit"}),
            ("test_cpp_extensions_aot", aot),
            ("cpp_extensions_jit", set()),
        ]:
            self.assertEqual(normalize_test_name(name, cands, paths), expected, name)
            self.assertEqual(normalize_test_name(name, cands), set(), name)
            out = answer(tests=[{"file": name}])
            sel = Selection.from_model_output(
                out, cands, [], lambda p: sorted(paths) if p == "^" else []
            )
            self.assertEqual(
                (sel.files, sel.run_all, sel.error is None),
                (expected, not expected, bool(expected)),
                name,
            )
            # Without grep, no file is known.
            sel = Selection.from_model_output(out, cands, [])
            self.assertTrue(sel.run_all and sel.error, name)

    def test_from_model_output(self) -> None:
        out = {
            "run_all": False,
            "reason": "r",
            "sweeps": ["dynamo_wrapped"],
            "dirs": ["test/dynamo/"],
            "tests": [{"file": "test/test_ops.py", "confidence": "high"}],
        }
        sel = Selection.from_model_output(
            out, CANDIDATES, ["torch/_dynamo/x.py", "test/test_nn.py"]
        )
        self.assertEqual(sel.files, {"test_ops", "test_nn", "dynamo/test_misc"})
        self.assertEqual(sel.reason, "r")
        self.assertEqual(sel.sweeps, {"dynamo_wrapped"})
        self.assertFalse(sel.run_all)
        self.assertTrue(sel.covers("inductor/test_torchinductor", "dynamo_wrapped"))
        self.assertFalse(sel.covers("inductor/test_torchinductor", "default"))
        self.assertTrue(sel.covers("test_ops", "default"))
        # The inductor config compiles every test of the files it runs with --inductor.
        bare = Selection.from_model_output(answer(), CANDIDATES | {"test_torch"}, [])
        self.assertTrue(bare.covers("test_torch", "inductor"))
        self.assertFalse(bare.covers("test_torch", "default"))
        self.assertFalse(bare.covers("inductor/test_torchinductor", "inductor"))
        self.assertIsNone(sel.error)
        self.assertEqual(Selection.from_json(sel.to_json()), sel)
        failed = Selection.from_model_output(None, CANDIDATES, [])
        self.assertIsNotNone(failed.error)
        self.assertEqual(Selection.from_json(failed.to_json()), failed)

    def test_always_covered_matches_test_sh(self) -> None:
        sh = (REPO_ROOT / ".ci/pytorch/test.sh").read_text()
        lines = sh.splitlines()
        # The top-level if/elif chain whose *inductor* branch calls test_inductor_shard.
        end = next(
            i
            for i, line in enumerate(lines)
            if line.startswith("  test_inductor_shard ")
        )
        start = max(i for i in range(end) if lines[i].startswith("if [["))
        dispatch = re.compile(
            r'(?:el)?if \[\[ "?\$\{?TEST_CONFIG\}?"? ==? (\S+) \]\]; then$'
        )
        branches = [
            m.group(1) for line in lines[start:end] if (m := dispatch.match(line))
        ]
        # A branch on anything else would hide the configs it catches from the globs.
        others = [
            line
            for line in lines[start:end]
            if re.match(r"(el)?if ", line) and not dispatch.match(line)
        ]
        self.assertEqual(
            others, ['elif [[ "${BUILD_ENVIRONMENT}" == *libtorch* ]]; then']
        )
        globs = tuple(b.strip("*") for b in branches if b.startswith("*"))
        self.assertEqual(
            globs,
            selection.TEST_SH_GLOBS,
            "update selection.TEST_SH_GLOBS to match test.sh's dispatch",
        )
        # These run no --inductor files, so the ALWAYS_COVERED set of a glob they
        # contain only over-covers them.
        exact = {b for b in branches if not b.startswith("*") and "inductor" in b}
        msg = "a new exact inductor branch must run no --inductor files"
        self.assertEqual(exact, {"inductor_avx2", "inductor_cpp_wrapper_cpu"}, msg)
        # Each function that runs files with --inductor is called by exactly one branch of the chain, on a glob.
        includes = inductor_includes(sh)
        callers: dict[str, set[str]] = {f: set() for f in includes}
        head = ""
        for line in lines[start : lines.index("fi", start)]:
            if re.match(r"(el)?if |else$", line):
                head = line
            callers.update(
                (f, callers[f] | {head})
                for f in includes
                if re.match(rf"\s+(\w+=\S+\s+)*{f}(\s|$)", line)
            )
        runs = {}
        for f, tests in includes.items():
            self.assertEqual(len(callers[f]), 1, f)
            m = dispatch.match(next(iter(callers[f])))
            self.assertTrue(m and m.group(1).startswith("*"), callers[f])
            glob = m.group(1).strip("*")
            self.assertIn(glob, selection.TEST_SH_GLOBS, "extend it past *inductor*")
            runs[glob] = runs.get(glob, set()) | tests
        self.assertEqual(
            runs,
            selection.ALWAYS_COVERED,
            "update selection.ALWAYS_COVERED to match test.sh's --inductor runs",
        )
        sel = Selection.from_model_output(answer(), CANDIDATES | {"test_torch"}, [])
        for config, covered in [
            ("inductor_amx", True),
            ("inductor_avx2", True),
            ("inductor_cpp_wrapper", True),
            ("inductor_core", False),
            ("inductor_huggingface", False),
            ("default", False),
        ]:
            self.assertEqual(sel.covers("test_torch", config), covered, config)
        self.assertFalse(sel.covers("test_ops", "inductor_cpp_wrapper"))

    def test_inductor_includes(self) -> None:
        sh = """python test/run_test.py --inductor --include top
f() {
  python test/run_test.py --inductor -i a b \
    c \
    --verbose
  python test/run_test.py --include not_inductor
  # run_test.py --inductor --include commented
  if true; then
    python test/run_test.py -k x --inductor \
      --include d && echo done
  fi
}
g () {
  python test/run_test.py --verbose --inductor -i e
}
"""
        self.assertEqual(
            inductor_includes(sh), {"": {"top"}, "f": {"a", "b", "c", "d"}, "g": {"e"}}
        )
        for bad in (
            'run_test.py --inductor --include "$TESTS"',
            "run_test.py --inductor --verbose",
            "run_test.py --inductor --include=a",
        ):
            with self.assertRaisesRegex(
                ValueError, "unreadable run_test.py --inductor call", msg=bad
            ):
                inductor_includes(bad)
        with self.assertRaisesRegex(ValueError, "No closing quotation"):
            inductor_includes("run_test.py --inductor -i 'a")

    def test_files_the_model_never_saw_are_covered(self) -> None:
        sel = Selection.from_model_output(answer(), CANDIDATES, [])
        self.assertEqual(sel.candidates, CANDIDATES)
        for f in ("test_openreg", "cpp/test_jit"):
            self.assertTrue(sel.covers(f, "default"), f)
        self.assertFalse(sel.covers("test_nn", "default"))
        self.assertEqual(Selection.from_json(sel.to_json()), sel)

    def test_model_run_all_is_kept(self) -> None:
        out = {"run_all": True, "reason": "", "sweeps": [], "dirs": [], "tests": []}
        sel = Selection.from_model_output(out, CANDIDATES, ["torch/a.py"])
        self.assertTrue(sel.run_all)
        self.assertTrue(sel.covers("test_nn", "default"))
        self.assertTrue(Selection.from_json(sel.to_json()).covers("test_nn", "default"))

    def test_fallbacks_run_all(self) -> None:
        sel = Selection.from_model_output(None, CANDIDATES, ["torch/a.py"])
        self.assertEqual((sel.run_all, sel.error), (True, "no model output"))
        out = {"run_all": False, "reason": "", "sweeps": [], "dirs": [], "tests": []}
        sel = Selection.from_model_output(out, CANDIDATES, ["test/run_test.py"])
        self.assertEqual((sel.run_all, sel.error), (True, None))
        self.assertFalse(
            Selection.from_model_output(out, CANDIDATES, ["docs/a.md"]).run_all
        )
        # A malformed answer sets error even when the change runs everything anyway.
        sel = Selection.from_model_output([], CANDIDATES, ["test/run_test.py"])
        self.assertRegex(str(sel.error), "^malformed model output: ")

    def test_load_malformed_file(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "sel.json"
            stderr = io.StringIO()
            with mock.patch("sys.stderr", stderr):
                self.assertIsNone(Selection.load(p))
            self.assertEqual(stderr.getvalue(), "")
            for sel in (
                Selection(candidates={"test_nn"}),
                Selection(run_all=True),
                Selection(run_all=True, error="e"),
                Selection(
                    {"test_nn"},
                    {"dynamo_wrapped", "crossref"},
                    reason="r",
                    enforce=True,
                    candidates={"test_nn", "test_ops"},
                ),
            ):
                p.write_text(json.dumps(sel.to_json()))
                self.assertEqual(Selection.load(p), sel)
            valid = Selection(candidates={"test_nn"}).to_json()
            # Without candidates, a selection would skip every file it does not name.
            bad = [
                {**valid, "version": 2},
                {**valid, "version": True},
                {**valid, "files": "test_nn"},
                {**valid, "files": [1]},
                {**valid, "run_all": "false"},
                {**valid, "enforce": "false"},
                {**valid, "reason": 1},
                {**valid, "candidates": [1]},
                {**valid, "candidates": None},
                {**valid, "error": 1},
                {**valid, "sweeps": ["default"]},
                {**valid, "files": ["test_ops"]},
                {**valid, "error": "e"},
                {**valid, "extra": 1},
            ]
            # run_all and no candidates leave only the type checks to reject these.
            r = Selection(run_all=True).to_json()
            bad += [{**r, "files": "test_nn"}, {**r, "files": [1]}, {**r, "error": 1}]
            bad += [
                {k: v for k, v in valid.items() if k != key}
                for key in ("version", "files", "sweeps", "run_all", "reason")
                + ("enforce", "candidates", "error")
            ]
            for text in (
                "{",
                '{"sweeps": []}',
                "[]",
                "[" * 100_000,
                *map(json.dumps, bad),
            ):
                p.write_text(text)
                stderr = io.StringIO()
                with mock.patch("sys.stderr", stderr):
                    self.assertIsNone(Selection.load(p), text[:100])
                self.assertTrue(stderr.getvalue().startswith("::warning::"), text[:100])
            sel = Selection({"test_nn"}, candidates={"test_nn", "test_ops"})
            p.write_text(json.dumps(sel.to_json()))
            self.assertEqual(Selection.load(p), sel)

    def test_covers(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be run_all"):
            Selection()
        cands = {"test_nn", "test_ops", "test_torch"}
        sel = Selection({"test_nn"}, {"dynamo_wrapped"}, candidates=cands)
        # The model was not shown test_openreg, so it always runs.
        for f, config, covered in [
            ("test_openreg", "default", True),
            ("test_nn", "default", True),
            ("test_ops", "default", False),
            ("test_ops", "dynamo_wrapped", True),
            ("test_ops", "crossref", False),
        ]:
            self.assertEqual(sel.covers(f, config), covered, (f, config))
        self.assertTrue(Selection(run_all=True).covers("test_ops", "default"))
        # The first of TEST_SH_GLOBS a config contains decides its ALWAYS_COVERED.
        for config, torch, ops in [
            ("inductor", True, True),
            ("inductor_amx", True, True),
            ("inductor_cpp_wrapper", True, False),
            ("inductor_core", False, False),
            ("inductor_huggingface", False, False),
        ]:
            got = (sel.covers("test_torch", config), sel.covers("test_ops", config))
            self.assertEqual(got, (torch, ops), config)

    def test_run_all_patterns(self) -> None:
        for f in [
            ".ci/docker/common/install_cuda.sh",
            ".ci/docker/requirements-ci.txt",
            ".ci/docker/requirements-docs.txt",
            ".ci/docker/java/jni.h",
            ".ci/docker/build.sh",
            ".ci/docker/ci_commit_pins/triton.txt",
            ".ci/docker/ubuntu/Dockerfile",
            ".ci/docker/ubuntu-rocm/Dockerfile",
            ".ci/docker/ubuntu-xpu/Dockerfile",
            ".ci/pytorch/test.sh",
            ".ci/pytorch/multigpu-test.sh",
            ".ci/pytorch/win-test.sh",
            ".ci/pytorch/macos-test.sh",
            ".ci/pytorch/macos-common.sh",
            ".ci/pytorch/build.sh",
            ".ci/pytorch/common.sh",
            ".ci/pytorch/common-build.sh",
            ".ci/pytorch/common_utils.sh",
            ".ci/pytorch/torch_trace.sh",
            ".ci/pytorch/win-build.sh",
            ".ci/pytorch/macos-build.sh",
            ".ci/pytorch/rocm_utils.sh",
            ".ci/pytorch/win-test-helpers/build_pytorch.bat",
            ".github/ci_commit_pins/torch_tpu.txt",
            "tools/stats/export_test_times.py",
            ".ci/pytorch/win-test-helpers/setup_pytorch_env.bat",
            ".ci/pytorch/win-test-helpers/test_python_shard.bat",
            ".ci/pytorch/win-test-helpers/installation-helpers/activate_miniconda3.bat",
            ".ci/pytorch/win-test-helpers/installation-helpers/install_magma.bat",
            ".ci/pytorch/win-test-helpers/installation-helpers/install_sccache.bat",
            ".github/workflows/_linux-test.yml",
            ".github/workflows/_win-test.yml",
            ".github/workflows/_mac-test.yml",
            ".github/workflows/_rocm-test.yml",
            ".github/workflows/_rocm-test-no-dind.yml",
            ".github/workflows/_xpu-test.yml",
            ".github/workflows/_s390x-test.yml",
            ".github/workflows/_tpu-test.yml",
            ".github/workflows/_linux-build.yml",
            ".github/workflows/_win-build.yml",
            ".github/workflows/_mac-build.yml",
            ".github/workflows/_s390x-build.yml",
            ".github/actions/setup-rocm/action.yml",
            ".github/actions/setup-win/action.yml",
            ".github/actions/setup-xpu/action.yml",
            ".github/actions/filter-test-configs/action.yml",
            ".github/actions/download-td-artifacts/action.yml",
            ".github/actions/download-build-artifacts/action.yml",
            ".github/scripts/filter_test_configs.py",
            ".github/scripts/parse_ref.py",
            ".github/scripts/pytest_cache.py",
            ".github/scripts/pytest_caching_utils.py",
            ".github/scripts/file_io_utils.py",
            ".github/actions/pytest-cache-download/action.yml",
            ".github/actions/pytest-cache-upload/action.yml",
            ".github/workflows/pull.yml",
            ".github/workflows/trunk.yml",
            ".github/workflows/periodic.yml",
            ".github/workflows/slow.yml",
            ".github/workflows/inductor.yml",
            ".github/workflows/inductor-periodic.yml",
            ".github/actions/reuse-old-whl/action.yml",
            ".github/workflows/target_determination.yml",
            ".github/workflows/llm_td_retrieval.yml",
            "tools/native_aot/build_stage2.py",
            ".ci/pytorch/generate_ck_config_h.cmake",
            ".github/ci_commit_pins/vision.txt",
            ".github/ci_commit_pins/audio.txt",
            ".github/ci_commit_pins/spmd_types.txt",
            "torch/testing/_internal/torchci/plugin.py",
            "test/run_test.py",
            "test/conftest.py",
            "test/torch_np/conftest.py",
            "test/slow_tests.json",
            "torch/testing/_internal/dynamo_test_failures.py",
            "test/pytest_shard_custom.py",
            "pytest.ini",
            "torch/testing/_internal/common_utils.py",
            "torch/testing/_internal/common_device_type.py",
            "torch/testing/_internal/common_dtype.py",
            "torch/testing/_internal/common_cuda.py",
            "torch/testing/_comparison.py",
            "torch/testing/__init__.py",
            "torch/testing/_creation.py",
            "torch/testing/_utils.py",
            "scripts/onnx/test.sh",
            ".ci/docker/triton_version.txt",
            ".ci/docker/triton_xpu_version.txt",
            ".ci/pytorch/numba-cuda-13.patch",
            ".ci/pytorch/win-test-helpers/test_python_jit_legacy.bat",
            "tools/testing/discover_tests.py",
            "tools/stats/import_test_stats.py",
            "tools/stats/upload_metrics.py",
            "tools/stats/upload_test_stats.py",
            "tools/stats/upload_stats_lib.py",
            "tools/stats/test_dashboard.py",
            "test/cpp_extensions/open_registration_extension/torch_openreg/pyproject.toml",
        ]:
            self.assertTrue(needs_run_all([f]), f)
            self.assertTrue(
                (REPO_ROOT / f).exists(),
                f"{f} is gone: update selection.RUN_ALL_PATTERNS",
            )
        # pytest reads the first of these it finds from the test's directory up.
        for f in [
            "conftest.py",
            "test/pytest.ini",
            "test/.pytest.ini",
            "test/dynamo/pyproject.toml",
            "test/tox.ini",
            "test/setup.cfg",
        ]:
            self.assertTrue(needs_run_all([f]), f)
        # Every caller of a test workflow sets the test matrix of its jobs.
        for wf in (REPO_ROOT / ".github/workflows").glob("*.yml"):
            called = [
                line.split("uses: ./", 1)[1].split()[0]
                for line in wf.read_text().splitlines()
                if "uses: ./.github/workflows/_" in line
            ]
            if any("-test" in c and needs_run_all([c]) for c in called):
                self.assertTrue(needs_run_all([f".github/workflows/{wf.name}"]), wf)
        for f in [
            "docs/source/index.rst",
            "test/data/setup.cfg.in",
            ".github/workflows/lint.yml",
            ".github/workflows/_binary-build-linux.yml",
            ".github/workflows/docs-build.yml",
            ".github/actions/setup-linux/action.yml",
            # A cross-build image; no test job runs in it.
            ".ci/docker/ubuntu-cross-riscv/Dockerfile",
            ".github/ci_commit_pins/vllm.txt",
        ]:
            self.assertFalse(needs_run_all([f]), f)
        self.assertTrue(needs_run_all(["docs/source/index.rst", "test/run_test.py"]))
        self.assertFalse(needs_run_all([]))

    def test_dirs(self) -> None:
        candidates = CANDIDATES | {"dynamo_foo/test_a"}
        for d in ["dynamo", "dynamo/", "dynamo/*", "dynamo/**", " test/dynamo/* "]:
            sel = Selection.from_model_output(answer(dirs=[d]), candidates, [])
            self.assertEqual(sel.files, {"dynamo/test_misc"}, d)
        sel = Selection.from_model_output(answer(dirs=["cpython"]), candidates, [])
        self.assertEqual(sel.files, {"cpython/v3_13/test_set"})
        # A blocklisted directory selects its drivers.
        jit = {"test_jit", "test_jit_profiling", "test_jit_legacy"}
        for d, expected in [("jit", jit), ("test/jit/", jit), ("fx", {"test_fx"})]:
            out = answer(dirs=[d])
            sel = Selection.from_model_output(out, candidates | jit | {"test_fx"}, [])
            self.assertEqual((sel.files, sel.error), (expected, None), d)
        # With grep, the drivers of the files under the directory.
        paths = [
            "test/jit/test_fuser_common.py",
            "test/cpp_extensions/dangling_impl_extension.cpp",
        ]
        cands = candidates | jit | {"test_jit_fuser_te", "test_dispatch"}
        for d, driver in [
            ("jit", "test_jit_fuser_te"),
            ("cpp_extensions", "test_dispatch"),
        ]:
            sel = Selection.from_model_output(
                answer(dirs=[d]), cands, [], lambda p: paths
            )
            self.assertIn(driver, sel.files, d)

    def test_parse_model_output_rejects_malformed_answers(self) -> None:
        malformed: list[Any] = [
            [],
            "tests",
            *({k: v for k, v in answer().items() if k != key} for key in answer()),
            answer(tests="test_nn"),
            answer(tests=["test_nn"]),
            answer(tests=[{"file": 3}]),
            answer(dirs="dynamo"),
            answer(sweeps=[1]),
            answer(dirs=None),
            answer(sweeps=None),
            answer(dirs=[1]),
            answer(sweeps=["default"]),
            answer(run_all=0),
            answer(reason=None),
            answer(reason=["r"]),
            answer(dirs=["nn"]),
            # An empty prefix must not match every candidate.
            *(answer(dirs=[d]) for d in ("", "/", "/*")),
        ]
        for out in malformed:
            with self.assertRaises(ValueError, msg=repr(out)):
                selection.parse_model_output(out, CANDIDATES)

    def test_edited_tests_select_their_driver(self) -> None:
        aot, autoload = (
            {"test_cpp_extensions_aot_ninja", "test_cpp_extensions_aot_no_ninja"},
            {"test_autoload_enable", "test_autoload_disable"},
        )
        jit = {"test_jit", "test_jit_profiling", "test_jit_legacy"}
        fuser, dispatch = {"test_jit_fuser_te"}, {"test_dispatch"}
        candidates = CANDIDATES | jit | fuser | dispatch | aot | autoload
        dangling = "test/cpp_extensions/dangling_impl_extension.cpp"
        for changed, expected in [
            ("test/jit/test_freezing.py", jit),
            ("test/test_jit.py", jit),
            ("test/jit/test_fuser_common.py", jit | fuser),
            ("test/cpp_extensions/extension.cpp", aot | autoload),
            (dangling, aot | autoload | dispatch),
            ("test/test_cpp_extensions_aot.py", aot),
            ("test/test_autoload.py", autoload),
            ("test/test_jit_fuser_te.py", fuser),
            ("test/test_nnapi.py", jit),
        ]:
            sel = Selection.from_model_output(answer(), candidates, [changed])
            self.assertEqual(sel.files, expected, changed)

    def test_drivers_match_the_test_list(self) -> None:
        for test in DRIVER_MAPPINGS:
            self.assertIn(test, TESTS)
        for test, regexes in selection.ADDITIONAL_MAPPINGS.items():
            self.assertLessEqual(set(regexes), set(DRIVER_MAPPINGS[test]), test)
        # Directories whose files no candidate runs. torch_openreg/tests runs
        # through test_openreg, which is never a candidate and so is always
        # covered; edits there also select the cpp_extensions drivers.
        undriven = {"custom_backend", "mobile"}
        for pattern in BLOCKLISTED_PATTERNS:
            path = f"test/{pattern}/test_x.py"
            driven = any(
                re.search(r, path) for rs in DRIVER_MAPPINGS.values() for r in rs
            )
            self.assertEqual(driven, pattern not in undriven, pattern)
        # Blocklisted test files that candidates import or run as subprocesses.
        spawn = {f"distributed/test_c10d_spawn_{b}" for b in ("gloo", "nccl", "ucc")}
        launcher_bin = "distributed/launcher/bin/test_script"
        for f, drivers in [
            ("test_jit_fuser", {"test_jit_fuser_legacy"}),
            ("test_nnapi", {"test_jit", "test_jit_profiling", "test_jit_legacy"}),
            ("distributed/test_c10d_spawn", spawn),
            ("onnx/test_models", {"onnx/test_models_onnxruntime"}),
            ("distributed/bin/test_script", {"distributed/test_launcher"}),
            *(
                (f"{launcher_bin}{s}", {"distributed/launcher/test_run"})
                for s in ("", "_init_method", "_is_torchelastic_launched")
            ),
        ]:
            path = f"test/{f}.py"
            self.assertTrue((REPO_ROOT / path).exists(), path)
            self.assertNotIn(f, TESTS)
            self.assertEqual(selection.edited_tests([path], set(TESTS)), drivers, f)
        # test_run runs bin/test_script.sh too.
        sh = f"test/{launcher_bin}.sh"
        self.assertTrue((REPO_ROOT / sh).exists(), sh)
        self.assertEqual(
            selection.edited_tests([sh], set(TESTS)), {"distributed/launcher/test_run"}
        )
        ext = "test/cpp_extensions/libtorch_agn_2_10_extension/csrc/kernel.cpp"
        self.assertLessEqual(
            {
                "cpp_extensions/test_libtorch_agnostic",
                "cpp_extensions/libtorch_agn_2_10_extension/test_version_compatibility",
            },
            selection.edited_tests([ext], set(TESTS)),
        )


if __name__ == "__main__":
    unittest.main()
