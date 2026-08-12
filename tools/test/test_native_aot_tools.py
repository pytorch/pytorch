from __future__ import annotations

import importlib
import json
import os
import tempfile
import unittest

# Ordinary package imports, like every other tools test (CI runs
# `PYTHONPATH=$(pwd) pytest tools/test`). These modules keep their module
# scope torch-free precisely so this works: the Test tools job runs in the
# linter image, which has no built torch.
from tools.native_aot import export, toolchains


_TOOLS_FILE = os.path.abspath(toolchains.__file__)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(_TOOLS_FILE)))


SIDECAR = {
    "prefix": "fakeop_f32_n1024_k8",
    "spec": {"dtype": "float32", "N": 1024, "K": 8, "deterministic": False},
    "tensor_args": [
        {"name": "mX", "dynamic_sizes": [0], "dynamic_strides": [0]},
        {"name": "mOut", "dynamic_sizes": [0, 1], "dynamic_strides": [0]},
    ],
}


class TestExportJobs(unittest.TestCase):
    def test_job_skip_matches_on_spec(self):
        # Skip detection matches the sidecar's recorded spec AND a
        # current source closure; a spec match alone (no/mismatched
        # sources) re-exports.
        import json as _json
        import tempfile

        rel = os.path.relpath(
            os.path.join(os.path.dirname(__file__), "..", "native_aot", "decl.py"),
            export.REPO,
        )
        current = {rel: export._file_hash(os.path.join(export.REPO, rel))}
        with tempfile.TemporaryDirectory() as d:
            point = {"dtype": "float32", "N": 4096}
            job = ("fakeop", "aot_kernel.py", point, d, None)
            self.assertTrue(export._job_needed(job, force=False))
            with open(os.path.join(d, "x.json"), "w") as f:
                _json.dump(
                    {
                        "version": export.SIDECAR_VERSION,
                        "prefix": "x",
                        "spec": point,
                        "sources": current,
                    },
                    f,
                )
            self.assertFalse(export._job_needed(job, force=False))
            self.assertTrue(export._job_needed(job, force=True))
            other = ("fakeop", "aot_kernel.py", {"dtype": "bfloat16"}, d, None)
            self.assertTrue(export._job_needed(other, force=False))

    def test_job_skip_survives_json_round_trip(self):
        # Tuple-valued grid fields read back from the sidecar as lists;
        # skip detection must normalize both sides or such points
        # re-export on every run (the pointwise family's in_dtypes hit
        # this).
        import json as _json
        import tempfile

        rel = os.path.relpath(
            os.path.join(os.path.dirname(__file__), "..", "native_aot", "decl.py"),
            export.REPO,
        )
        current = {rel: export._file_hash(os.path.join(export.REPO, rel))}
        with tempfile.TemporaryDirectory() as d:
            point = {"aten": "add.Tensor", "in_dtypes": ("float32", "bfloat16")}
            job = ("fakeop", "aot_kernel.py", point, d, None)
            sidecar = {
                "version": export.SIDECAR_VERSION,
                "prefix": "x",
                "spec": export._json_normal(point),
                "sources": current,
            }
            with open(os.path.join(d, "x.json"), "w") as f:
                _json.dump(sidecar, f)
            self.assertFalse(export._job_needed(job, force=False))

    def test_run_job_is_module_level(self):
        # The pool pickles the job function by qualified name; a closure
        # or nested function would break only at --jobs > 1.
        # (A real pickle round-trip needs export importable by module
        # name, which this by-path test harness can't provide.)
        self.assertEqual(export._run_job.__qualname__, "_run_job")
        self.assertEqual(export.export_point.__qualname__, "export_point")

    def test_pool_never_forks_after_cuda_init(self):
        # Plain "fork" gives workers a dead CUDA context (measured:
        # is_initialized() False, allocation fails, no exception) because
        # the parent initializes CUDA before the pool starts.
        # Pinned to forkserver, not just "not fork": main() calls
        # set_forkserver_preload unconditionally, which a spawn context
        # would silently ignore.
        self.assertEqual(export.POOL_START_METHOD, "forkserver")

    def test_cutedsl_export_passes_gpu_arch(self):
        # The arch must reach the compiler as a --gpu-arch OPTION, not as
        # process state: that is what lets one worker serve several
        # arches. Appended to any builder-supplied options, never
        # replacing them. Stubs cute.compile/export_to_c because this
        # suite runs in the linter image, which has no DSL installed.
        import sys
        import types
        import unittest.mock as mock
        from typing import Any, cast

        seen = {}

        def fake_compile(fn, *args, **kwargs):
            seen["options"] = kwargs.get("options")
            return types.SimpleNamespace(export_to_c=lambda **kw: None)

        fake_cute = types.ModuleType("cutlass.cute")
        # cast: a ModuleType instance has no declared attributes, so plain
        # assignment fails type checking (idiom from test/test_utils.py).
        cast(Any, fake_cute).compile = fake_compile
        fake_cutlass = types.ModuleType("cutlass")
        cast(Any, fake_cutlass).cute = fake_cute
        tc = toolchains.CuteDslToolchain()
        with (
            mock.patch.dict(
                sys.modules, {"cutlass": fake_cutlass, "cutlass.cute": fake_cute}
            ),
            mock.patch.object(toolchains.CuteDslToolchain, "_warm_up_exporter"),
        ):
            for opts, want in (
                (None, "--gpu-arch sm_90a"),
                ("--enable-assertions", "--enable-assertions --gpu-arch sm_90a"),
            ):
                b = {"prefix": "p", "fn": None, "fake_args": (), "tensor_args": []}
                if opts:
                    b["options"] = opts
                tc.export(b, "/tmp", arch="sm_90a")
                self.assertEqual(seen["options"], want)
            # No arch: builder options pass through untouched (the
            # detect-from-device path must not inject --gpu-arch).
            tc.export(
                {"prefix": "p", "fn": None, "fake_args": (), "tensor_args": []}, "/tmp"
            )
            self.assertIsNone(seen["options"])

    def test_toolchains_declare_their_backend(self):
        # Which build backends a kind can emit for is DATA, not a platform
        # check buried in the gate: that is what lets a ROCm build skip
        # cleanly today and what makes a future ROCm DSL a new class with
        # BACKENDS = ("rocm",) rather than an edit to build_stage2.
        for kind, tc in toolchains.TOOLCHAINS.items():
            self.assertTrue(tc.BACKENDS, f"{kind} declares no BACKENDS")
        self.assertEqual(sorted(toolchains.for_backend("rocm")), [])
        self.assertEqual(
            sorted(toolchains.for_backend("cuda")),
            sorted(toolchains.TOOLCHAINS),
        )

    def test_missing_runtime_is_fatal_not_skipped(self):
        # A declaration whose toolchain targets this build's backend was
        # ASKED for, so a missing runtime must fail rather than quietly
        # ship a wheel with fewer kernels than declared. TORCH_NATIVE_AOT=0
        # is the supported way to build without the DSL wheels.
        import unittest.mock as mock

        with mock.patch.object(
            toolchains.CuteDslToolchain, "missing_runtimes", classmethod(lambda cls: [])
        ):
            b = {"prefix": "p", "fn": None, "fake_args": (), "tensor_args": []}
            with mock.patch.object(export, "load_builder", lambda *a: lambda p: b):
                # Runtime present: gets past the gate (fails later, on the
                # real compile, which this harness cannot do).
                with self.assertRaises(Exception) as cm:
                    export.export_point("fakeop", "aot_kernel.py", {}, "/tmp")
                self.assertNotIn("TORCH_NATIVE_AOT=0", str(cm.exception))

        with mock.patch.object(
            toolchains.CuteDslToolchain,
            "missing_runtimes",
            classmethod(lambda cls: ["cutlass"]),
        ):
            b = {"prefix": "p", "fn": None, "fake_args": (), "tensor_args": []}
            with mock.patch.object(export, "load_builder", lambda *a: lambda p: b):
                with self.assertRaisesRegex(RuntimeError, "cannot export"):
                    export.export_point("fakeop", "aot_kernel.py", {}, "/tmp")

    def test_pool_preload_stays_fork_safe(self):
        # The forkserver's server process is the fork PARENT for every
        # worker, so only modules that are inert in a fork parent may be
        # preloaded. Importing torch initializes no CUDA state; cutlass
        # and triton would build DSL/driver state in the parent that every
        # worker then inherits, which is what this pins down.
        self.assertEqual(export.POOL_PRELOAD, ("torch",))

    def test_json_normal_matches_sidecar_round_trip(self):
        # _json_normal replaces a json.dumps/loads pair, so it must agree
        # with one exactly: any divergence makes a spec mismatch its own
        # sidecar and re-export forever.
        point = {
            "dtype": "float32",
            "in_dtypes": ("float32", "bfloat16"),
            "N": 4096,
            "tma": True,
        }
        self.assertEqual(export._json_normal(point), json.loads(json.dumps(point)))
        self.assertEqual(
            export._json_normal(point)["in_dtypes"], ["float32", "bfloat16"]
        )


class TestArch(unittest.TestCase):
    def test_archs_from_cuda_arch_list(self):
        # TORCH_CUDA_ARCH_LIST -> the EXPORTABLE_ARCHES subset; named,
        # malformed, +PTX and non-exportable entries drop out.
        f = export.archs_from_cuda_arch_list
        self.assertEqual(f("7.5 8.9"), [])
        self.assertEqual(f("9.0a;10.0a"), ["sm_100a"])
        self.assertEqual(f("8.0 9.0 10.0+PTX"), ["sm_100"])
        # Both spellings of a CC are separate nvcc targets, and both are
        # exportable: CI passes "10.0a", the wheel builds pass "10.0".
        self.assertEqual(f("10.0"), ["sm_100"])
        # 10.3 is not exportable while the runtime gate is major-only.
        self.assertEqual(f("Hopper 10.3a"), [])

    def test_archs_from_cuda_arch_list_dedups(self):
        # "10.0;10.0+PTX" names one arch twice. Without dedup the result
        # reads as multi-arch downstream: nested <out>/<arch>/ layout the
        # one-level CMake globs do not walk, and a --jobs 1 hard exit.
        f = export.archs_from_cuda_arch_list
        self.assertEqual(f("10.0;10.0+PTX"), ["sm_100"])
        self.assertEqual(f("10.0a 10.0a"), ["sm_100a"])
        # Distinct spellings are distinct targets, so both survive.
        self.assertEqual(f("10.0;10.0a"), ["sm_100", "sm_100a"])

    def test_collect_jobs_respects_declaration_archs(self):
        # A declaration pinning ARCHS gets no jobs for other arches; an
        # on-device export (arch None) is never filtered.
        import tempfile
        import unittest.mock as mock

        decl_body = (
            'ATEN_OP = "fakeop"\nDISPATCH_KEY = "CUDA"\n'
            'KERNEL_MODULE = "k.py"\nARCHS = ("sm_100a",)\n'
            "def kernel_precompile_grid():\n    return [{'dtype': 'float32'}]\n"
            "def covered_axes(self):\n    return {}\n"
            "def cpp_dispatch(spec):\n    return 'true'\n"
            "def cpp_launch(spec, launch_fn):\n    return launch_fn\n"
        )
        with tempfile.TemporaryDirectory() as ops, tempfile.TemporaryDirectory() as out:
            os.makedirs(os.path.join(ops, "fakeop"))
            with open(os.path.join(ops, "fakeop", "aot.py"), "w") as f:
                f.write(decl_body)
            with mock.patch.object(export, "OPS_DIR", ops):
                blackwell = export._collect_jobs(None, out, False, ["sm_100a"])
                hopper = export._collect_jobs(None, out, False, ["sm_90a"])
                on_device = export._collect_jobs(None, out, False, [None])
        self.assertEqual(len(blackwell), 1)
        self.assertEqual(len(hopper), 0)
        self.assertEqual(len(on_device), 1)

    def test_multi_arch_jobs_nest_per_arch(self):
        # Multi-arch fan-out nests <out>/<arch>/<decl_id>; single arch
        # (or default None) keeps the flat layout.
        import tempfile
        import unittest.mock as mock

        with tempfile.TemporaryDirectory() as ops, tempfile.TemporaryDirectory() as out:
            os.makedirs(os.path.join(ops, "fakeop"))
            with open(os.path.join(ops, "fakeop", "aot.py"), "w") as f:
                f.write(
                    'ATEN_OP = "fakeop"\nDISPATCH_KEY = "CUDA"\n'
                    'KERNEL_MODULE = "k.py"\n'
                    "def kernel_precompile_grid():\n    return [{'dtype': 'float32'}]\n"
                    "def covered_axes(self):\n    return {}\n"
                    "def cpp_dispatch(spec):\n    return 'true'\n"
                    "def cpp_launch(spec, launch_fn):\n    return launch_fn\n"
                )
            with mock.patch.object(export, "OPS_DIR", ops):
                multi = export._collect_jobs(None, out, False, ["sm_90a", "sm_100a"])
                single = export._collect_jobs(None, out, False, [None])
        self.assertEqual(len(multi), 2)
        dirs = sorted(os.path.basename(os.path.dirname(j[3])) for j in multi)
        self.assertEqual(dirs, ["sm_100a", "sm_90a"])
        self.assertEqual({j[4] for j in multi}, {"sm_90a", "sm_100a"})
        (sj,) = single
        self.assertEqual(os.path.basename(sj[3]), "fakeop")
        self.assertIsNone(sj[4])

    def test_empty_dir_is_fine(self):
        # A clean build (or a newly added spec point) has no sidecar and
        # no artifacts; it must export, not fail.
        with tempfile.TemporaryDirectory() as d:
            export._check_no_orphan_artifacts(d)

    def test_artifacts_without_sidecar_are_fatal(self):
        # An export that died between compiling and writing the sidecar.
        # The CMake globs link *.o by pattern, so an undescribed orphan
        # would otherwise be linked silently.
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "k_f32.o"), "w").close()
            with self.assertRaisesRegex(RuntimeError, "no sidecar"):
                export._check_no_orphan_artifacts(d)

    def test_artifacts_with_sidecar_are_fine(self):
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "k.o"), "w").close()
            with open(os.path.join(d, "k.json"), "w") as f:
                json.dump({"prefix": "k"}, f)
            export._check_no_orphan_artifacts(d)

    def test_unreadable_sidecar_is_fatal(self):
        # Present but unparsable: corruption, not an interrupted run.
        # Re-exporting would paper over it (and --force skips the check).
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "k.json")
            with open(path, "w") as f:
                f.write("{truncated")
            with self.assertRaisesRegex(RuntimeError, "could not be read"):
                export._read_sidecar(path)

    def test_cross_product(self):
        pts = export.expand_specs(
            [{"dtype": ["float32", "bfloat16"], "N": [1024, 2048], "K": 8}]
        )
        self.assertEqual(len(pts), 4)
        self.assertIn({"dtype": "float32", "N": 2048, "K": 8}, pts)
        self.assertIn({"dtype": "bfloat16", "N": 1024, "K": 8}, pts)

    def test_multiple_blocks_concatenate(self):
        pts = export.expand_specs([{"N": [1, 2]}, {"N": 3, "extra": True}])
        self.assertEqual(pts, [{"N": 1}, {"N": 2}, {"N": 3, "extra": True}])

    def test_scalar_only_spec(self):
        self.assertEqual(
            export.expand_specs([{"N": 4096, "K": 64}]), [{"N": 4096, "K": 64}]
        )


class TestSourceStaleness(unittest.TestCase):
    def test_closure_covers_shared_declaration_machinery(self):
        # The grid expander and the validating loader decide which spec
        # points exist and what a declaration means, so editing them
        # changes what an artifact MEANS. They live in torchgen (outside
        # the tools/*.py glob) and arrive by ordinary import, so only the
        # sys.modules half of the closure can catch them -- which used to
        # filter on "torch._native" alone and silently missed them.
        # Compared by basename: an editable install can resolve torchgen
        # to a different checkout than REPO, and relpath then yields a
        # ../.. traversal rather than the tidy repo-relative path.
        names = {os.path.basename(p) for p in export.source_closure()}
        for want in (
            "native_aot_spec_grid.py",
            "native_aot_decl.py",
            "toolchains.py",
        ):
            self.assertIn(want, names, f"{want} must invalidate artifacts")

    def test_closure_survives_sys_modules_mutation(self):
        # source_closure hashes files while walking sys.modules, and
        # hashing imports hashlib lazily -- so on a cold interpreter the
        # walk mutates the dict it is iterating and raises "dictionary
        # changed size during iteration". Force that ordering by having
        # the hash step import a module that is definitely not loaded yet.
        import sys
        import unittest.mock as mock

        real_hash = export._file_hash

        def hash_and_import(path):
            importlib.import_module("wave")  # stdlib, unlikely to be loaded
            return real_hash(path)

        sys.modules.pop("wave", None)
        with mock.patch.object(export, "_file_hash", hash_and_import):
            export.source_closure()

    def test_sources_current_roundtrip(self):
        # A sidecar whose recorded closure matches the tree is current;
        # editing any recorded file (or recording none) makes it stale.
        rel = os.path.relpath(
            os.path.join(os.path.dirname(__file__), "..", "native_aot", "decl.py"),
            export.REPO,
        )
        h = export._file_hash(os.path.join(export.REPO, rel))
        good = {"version": export.SIDECAR_VERSION, "sources": {rel: h}}
        self.assertTrue(export.sources_current(good))
        # schema-version mismatch is stale even with current sources
        self.assertFalse(export.sources_current({"version": 0, "sources": {rel: h}}))
        self.assertFalse(export.sources_current({"sources": {rel: "0" * 16}}))
        self.assertFalse(export.sources_current({}))
        self.assertFalse(export.sources_current({"sources": {"no/such/file.py": "aa"}}))

    def test_stale_point_reexports_without_force(self):
        import json as _json

        with tempfile.TemporaryDirectory() as d:
            point = {"dtype": "float32"}
            job = ("fakeop", "aot_kernel.py", point, d, None)
            rel = os.path.relpath(
                os.path.join(os.path.dirname(__file__), "..", "native_aot", "decl.py"),
                export.REPO,
            )
            current = {rel: export._file_hash(os.path.join(export.REPO, rel))}
            with open(os.path.join(d, "x.json"), "w") as f:
                _json.dump(
                    {
                        "version": export.SIDECAR_VERSION,
                        "prefix": "x",
                        "spec": point,
                        "sources": current,
                    },
                    f,
                )
            self.assertFalse(export._job_needed(job, force=False))
            with open(os.path.join(d, "x.json"), "w") as f:
                _json.dump(
                    {"prefix": "x", "spec": point, "sources": {rel: "0" * 16}}, f
                )
            self.assertTrue(export._job_needed(job, force=False))
