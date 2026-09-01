from __future__ import annotations

import importlib
import json
import os
import tempfile
import unittest

# Ordinary package imports: these modules keep their scope torch-free so the
# Test tools job, which runs in the linter image, can import them without torch.
from tools.native_aot import export, toolchains

from torchgen import native_aot_decl


_TOOLS_FILE = os.path.abspath(toolchains.__file__)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(_TOOLS_FILE)))

SIDECAR = {
    "prefix": "fakeop_f32_n1024_k8",
    # Generation reads arch and kind rather than defaulting either, so a fixture
    # missing them is not a sidecar export could have written.
    "arch": "sm_100a",
    "kind": "cutedsl",
    "spec": {"dtype": "float32", "N": 1024, "K": 8, "deterministic": False},
    "tensor_args": [
        {"name": "mX", "dynamic_sizes": [0], "dynamic_strides": [0]},
        {"name": "mOut", "dynamic_sizes": [0, 1], "dynamic_strides": [0]},
    ],
}


def _write_fake_decl(ops_dir, archs_line="", grid="[{'N': 1}]"):
    """One declaration under ops_dir/fakeop/, the minimum _collect_jobs loads:
    one grid point and the two C++ hooks, whose text nothing here reads."""
    os.makedirs(os.path.join(ops_dir, "fakeop"), exist_ok=True)
    with open(os.path.join(ops_dir, "fakeop", "aot.py"), "w") as f:
        f.write(
            'ATEN_OP = "fakeop"\nDISPATCH_KEY = "CUDA"\nKERNEL_MODULE = "k.py"\n'
            + archs_line
            + f"def kernel_precompile_grid():\n    return {grid}\n"
            "def covered_axes(self):\n    return {}\n"
            "def cpp_dispatch(spec):\n    return 'true'\n"
            "def cpp_launch(spec, launch_fn):\n    return launch_fn\n"
        )


def _touch_artifacts(out_dir, prefix, exts=(".o", ".h")):
    """Create the files a sidecar claims. _job_needed verifies they exist,
    so a fixture that writes only the .json would always re-export."""
    for e in exts:
        with open(os.path.join(out_dir, prefix + e), "w") as f:
            f.write("")


class TestExportJobs(unittest.TestCase):
    def test_job_skip_matches_on_spec(self):
        # A spec match alone re-exports: the source closure must match too.
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
            _touch_artifacts(d, "x")
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
        # Tuple-valued grid fields read back as lists, so both sides must be
        # normalized or such points re-export on every run.
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
            _touch_artifacts(d, "x")
            _touch_artifacts(d, "x")
            with open(os.path.join(d, "x.json"), "w") as f:
                _json.dump(sidecar, f)
            self.assertFalse(export._job_needed(job, force=False))

    def test_run_job_is_module_level(self):
        # The pool pickles the job function by qualified name, so a closure would
        # break only at --jobs > 1.
        self.assertEqual(export._run_job.__qualname__, "_run_job")
        self.assertEqual(export.export_point.__qualname__, "export_point")

    def test_pool_never_forks_after_cuda_init(self):
        # A fork parent that has initialized CUDA gives workers a dead context,
        # silently. forkserver specifically: main() calls set_forkserver_preload.
        self.assertEqual(export.POOL_START_METHOD, "forkserver")

    def test_cutedsl_export_passes_gpu_arch(self):
        # A --gpu-arch option rather than process state, which is what lets one
        # worker serve several arches, appended to any builder-supplied options.
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
        # Backends are data, not a platform check in the gate, so a future ROCm DSL
        # is a new class rather than an edit to build_stage2.
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
        # The forkserver's server process is the fork parent, so only modules inert
        # there may be preloaded: cutlass or triton would build state workers inherit.
        self.assertEqual(export.POOL_PRELOAD, ("torch",))

    def test_json_normal_matches_sidecar_round_trip(self):
        # It stands in for a json.dumps/loads pair, so any divergence makes a spec
        # mismatch its own sidecar and re-export forever.
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
    def test_effective_arch_is_per_toolchain(self):
        # Only kinds that declare an ARCH_ENV_VAR read one, so a kind that
        # takes its target another way (Triton: an explicit GPUTarget) is
        # not perturbed by CUTE_DSL_ARCH.
        import unittest.mock as mock

        cutedsl = toolchains.get_toolchain("cutedsl")
        no_env = toolchains.Toolchain()
        self.assertIsNone(no_env.ARCH_ENV_VAR)
        with mock.patch.dict(os.environ, {"CUTE_DSL_ARCH": "sm_90a"}):
            self.assertEqual(export._effective_arch(None, cutedsl), "sm_90a")
            self.assertIsNone(export._effective_arch(None, no_env))
            # An explicit arch always wins over the env var.
            self.assertEqual(export._effective_arch("sm_100a", cutedsl), "sm_100a")

    def test_job_skip_sees_the_arch_env_var(self):
        # Two runs into ONE --out-dir differing only in CUTE_DSL_ARCH. Both
        # pass arch=None, so comparing the raw value would match on spec
        # alone and skip the second run -- leaving the first arch's objects
        # behind a sidecar the caller reads as the second arch.
        import unittest.mock as mock

        rel = os.path.relpath(
            os.path.join(os.path.dirname(__file__), "..", "native_aot", "decl.py"),
            export.REPO,
        )
        current = {rel: export._file_hash(os.path.join(export.REPO, rel))}
        with tempfile.TemporaryDirectory() as d:
            point = {"dtype": "float32", "N": 4096}
            job = ("fakeop", "aot_kernel.py", point, d, None)
            _touch_artifacts(d, "x")
            with open(os.path.join(d, "x.json"), "w") as f:
                json.dump(
                    {
                        "version": export.SIDECAR_VERSION,
                        "prefix": "x",
                        "kind": "cutedsl",
                        "spec": point,
                        "arch": "sm_90a",
                        "sources": current,
                    },
                    f,
                )
            with mock.patch.dict(os.environ, {"CUTE_DSL_ARCH": "sm_90a"}):
                self.assertFalse(export._job_needed(job, force=False))
            with mock.patch.dict(os.environ, {"CUTE_DSL_ARCH": "sm_100a"}):
                self.assertTrue(export._job_needed(job, force=False))
            # Env var gone: an on-device export is not the sm_90a one either.
            with mock.patch.dict(os.environ, {}, clear=True):
                self.assertTrue(export._job_needed(job, force=False))

    def test_job_skip_rejects_arch_less_sidecar_when_env_set(self):
        # A sidecar recording no arch must not satisfy a run whose CUTE_DSL_ARCH
        # names one, or that run inherits objects built for the builder's own GPU.
        import unittest.mock as mock

        rel = os.path.relpath(
            os.path.join(os.path.dirname(__file__), "..", "native_aot", "decl.py"),
            export.REPO,
        )
        current = {rel: export._file_hash(os.path.join(export.REPO, rel))}
        with tempfile.TemporaryDirectory() as d:
            point = {"dtype": "float32", "N": 4096}
            job = ("fakeop", "aot_kernel.py", point, d, None)
            _touch_artifacts(d, "x")
            with open(os.path.join(d, "x.json"), "w") as f:
                json.dump(
                    {
                        "version": export.SIDECAR_VERSION,
                        "prefix": "x",
                        "kind": "cutedsl",
                        "spec": point,
                        "arch": None,
                        "sources": current,
                    },
                    f,
                )
            with mock.patch.dict(os.environ, {"CUTE_DSL_ARCH": "sm_100a"}):
                self.assertTrue(export._job_needed(job, force=False))
            # ...while a genuine on-device re-run still skips.
            with mock.patch.dict(os.environ, {}, clear=True):
                self.assertFalse(export._job_needed(job, force=False))

    def test_cc_of_reads_the_capability_both_spellings_name(self):
        # The exporter matches ARCHS by string while the generator groups by
        # capability, so both spellings of one piece of hardware must parse equal.
        self.assertEqual(native_aot_decl.cc_of("sm_90"), (9, 0))
        self.assertEqual(native_aot_decl.cc_of("sm_103a"), (10, 3))
        self.assertEqual(
            native_aot_decl.cc_of("sm_100a"), native_aot_decl.cc_of("sm_100")
        )

    def test_cc_of_refuses_what_it_cannot_read(self):
        # Each would otherwise compute a plausible capability ("sm_9" -> (0, 9)) and
        # emit a gate no device satisfies. The last four are what str.isdigit() accepts.
        for bad in (
            "sm_9",
            "sm_1000",
            "sm_100f",
            "sm_",
            "",
            "100a",
            "sm_10a0",
            "sm_\uff19\uff10",
            "sm_\u0669\u0660",
            "sm_090",
            "sm_\u00b2\u00b2",
        ):
            with self.subTest(arch=bad):
                with self.assertRaisesRegex(
                    RuntimeError, "cannot read a compute capability"
                ):
                    native_aot_decl.cc_of(bad)

    def test_cc_of_refuses_a_capability_outside_the_known_range(self):
        # Parses fine, but no such hardware: a gate for it is dead code.
        with self.assertRaisesRegex(RuntimeError, "outside the known range"):
            native_aot_decl.cc_of("sm_130")

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
                blackwell = export._collect_jobs(None, out, ["sm_100a"])
                hopper = export._collect_jobs(None, out, ["sm_90a"])
                on_device = export._collect_jobs(None, out, [None])
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
                multi = export._collect_jobs(None, out, ["sm_90a", "sm_100a"])
                single = export._collect_jobs(None, out, [None])
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
        # The grid expander and the loader decide what a declaration means, and arrive
        # by ordinary import, so only the sys.modules half of the closure catches them.
        # By basename, because an editable install can resolve torchgen elsewhere.
        names = {os.path.basename(p) for p in export.source_closure()}
        for want in (
            "native_aot_spec_grid.py",
            "native_aot_decl.py",
            "toolchains.py",
        ):
            self.assertIn(want, names, f"{want} must invalidate artifacts")

    def test_closure_survives_sys_modules_mutation(self):
        # Hashing imports hashlib lazily, so on a cold interpreter the walk mutates the
        # dict it is iterating. Forced by hashing a module that is not loaded yet.
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
            _touch_artifacts(d, "x")
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
            _touch_artifacts(d, "x")
            with open(os.path.join(d, "x.json"), "w") as f:
                _json.dump(
                    {"prefix": "x", "spec": point, "sources": {rel: "0" * 16}}, f
                )
            self.assertTrue(export._job_needed(job, force=False))


class TestLauncherCodegen(unittest.TestCase):
    """CuteDslToolchain.gen_launcher is what puts C++ into the build; these
    pin the properties that were argued over in review."""

    def _launcher(self, **over):
        sc = dict(SIDECAR, **over)
        return toolchains.CuteDslToolchain().gen_launcher(sc)

    def test_read_only_arg_uses_const_data_ptr(self):
        # A mutable data_ptr() would materialize a copy-on-write input.
        targs = [dict(SIDECAR["tensor_args"][0], read_only=True)]
        src = self._launcher(tensor_args=targs)
        self.assertIn("const_cast<void*>(mX.const_data_ptr())", src)
        self.assertNotIn("mX.mutable_data_ptr()", src)

    def test_written_arg_uses_mutable_data_ptr(self):
        src = self._launcher()
        self.assertIn("mOut_s.data = mOut.mutable_data_ptr();", src)

    def test_arg_order_is_tensors_then_scalars_then_stream(self):
        # Must match the exported wrapper's signature exactly.
        src = self._launcher(scalar_args=[{"name": "k", "ctype": "int64_t"}])
        self.assertIn("const at::Tensor& mX, const at::Tensor& mOut, int64_t k", src)
        self.assertIn("&mX_s, &mOut_s, k", src)
        self.assertIn("c10::cuda::CUDAStream(stream).stream()", src)

    def test_shape_slots_narrow_and_strides_stay_64bit(self):
        src = self._launcher()
        self.assertIn("mX_s.dynamic_shapes[0] = static_cast<int32_t>(mX.size(0));", src)
        self.assertIn("mX_s.dynamic_strides[0] = mX.stride(0);", src)

    def test_module_load_is_once_per_process(self):
        src = self._launcher()
        self.assertIn("c10::call_once", src)


# mX with one size and one stride slot, and with none at all: the two claims the
# per-argument checks are made against, with SIDECAR's own mOut beside them.
_MX_ONE_SLOT = [
    {"name": "mX", "dynamic_sizes": [0], "dynamic_strides": [0]},
    SIDECAR["tensor_args"][1],
]
_MX_NO_SLOTS = [{"name": "mX"}, SIDECAR["tensor_args"][1]]


class TestAbiValidation(unittest.TestCase):
    """The launcher is a fixed template and the ABI comes from the DSL, so a width
    or slot-count mismatch is a wrong value at runtime, not a compile error.

    Every fixture here is in the layout the DSL actually emits -- `typedef struct
    { ...fields... } <prefix>_Tensor_<name>_t;`, one struct PER TENSOR ARGUMENT.
    Fixtures shaped like `struct x { ... }` cannot exercise a per-argument check
    at all."""

    def _struct(self, name, shapes=1, strides=1, stride_t="int64_t", shape_t="int32_t"):
        p = SIDECAR["prefix"]
        fields = ["  void* data;"]
        if shapes:
            fields.append(f"  {shape_t} dynamic_shapes[{shapes}];")
        if strides:
            fields.append(f"  {stride_t} dynamic_strides[{strides}];")
        return "typedef struct {\n" + "\n".join(fields) + f"\n}} {p}_Tensor_{name}_t;\n"

    def _header(self, mx=None, mout=None):
        # Defaults match SIDECAR's claims (mX one shape and one stride, mOut two and
        # one); anything else describes an ABI no export could produce.
        return (
            "#pragma once\n#include <stdint.h>\n"
            + (mx if mx is not None else self._struct("mX", shapes=1, strides=1))
            + (mout if mout is not None else self._struct("mOut", shapes=2, strides=1))
        )

    def _mx_header(self, member):
        """A whole header whose mX struct declares `member` where its
        dynamic_strides go, with SIDECAR's mOut struct beside it."""
        p = SIDECAR["prefix"]
        return (
            "#pragma once\n#include <stdint.h>\n"
            "typedef struct {\n  void* data;\n"
            f"  int32_t dynamic_shapes[1];\n  {member}\n}} {p}_Tensor_mX_t;\n"
            + self._struct("mOut", shapes=2, strides=1)
        )

    def _sidecar(self, tmpdir, header: str, **over) -> dict:
        with open(os.path.join(tmpdir, SIDECAR["prefix"] + ".h"), "w") as f:
            f.write(header)
        return dict(SIDECAR, _dir=tmpdir, **over)

    def _refuses(self, header, pattern, **over):
        tc = toolchains.CuteDslToolchain()
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(RuntimeError, pattern):
                tc.validate_abi(self._sidecar(d, header, **over))

    def _accepts(self, header, **over):
        tc = toolchains.CuteDslToolchain()
        with tempfile.TemporaryDirectory() as d:
            tc.validate_abi(self._sidecar(d, header, **over))

    def test_a_matching_header_is_accepted(self):
        # int32 SHAPE slots are expected and handled (explicit cast + size gate);
        # only the stride WIDTH and both slot COUNTS are the subject.
        self._accepts(self._header())

    def test_int32_stride_slots_are_refused(self):
        # aten strides are int64 and the launcher assigns them straight across, so
        # int32 slots truncate silently -- an implicit conversion, no warning.
        self._refuses(
            self._header(mx=self._struct("mX", stride_t="int32_t")),
            "must be declared 64-bit",
        )

    def test_the_width_is_checked_per_argument(self):
        # use_32bit_stride is per argument, so a whole-file search for one
        # "int64_t dynamic_strides" accepts a header that narrows another tensor's.
        self._refuses(
            self._header(mout=self._struct("mOut", shapes=2, stride_t="int32_t")),
            "mOut",
        )

    def test_a_longer_argument_name_does_not_shadow_a_shorter_one(self):
        # A find() of "<prefix>_Tensor_mX_t" also matches inside "..._mX_tile_t", so
        # argument order alone would decide which tensor's struct is read.
        for label, header in (
            (
                "longer first",
                self._struct("mX_tile", shapes=4, strides=3)
                + self._header(mx=self._struct("mX", stride_t="int32_t")),
            ),
            (
                "longer last",
                self._header(mx=self._struct("mX", stride_t="int32_t"))
                + self._struct("mX_tile", shapes=4, strides=3),
            ),
        ):
            with self.subTest(order=label):
                self._refuses(header, "mX's dynamic_strides|must be declared")

    def test_a_slot_count_must_match_exactly(self):
        # Two independent statements about one number. Over-claiming stores past the end
        # of an uninitialized local; under-claiming leaves slots unwritten.
        for label, header in (
            (
                "header declares more",
                self._header(mx=self._struct("mX", shapes=1, strides=2)),
            ),
            (
                "header declares fewer",
                self._header(mx=self._struct("mX", shapes=3, strides=1)),
            ),
        ):
            with self.subTest(case=label):
                # Each side named as its own: the sidecar's key and the header's
                # member differ (dynamic_sizes against dynamic_shapes).
                self._refuses(
                    header, r"sidecar's dynamic_(sizes|strides) claims \d+ slot"
                )

    def test_a_missing_member_must_match_a_zero_claim(self):
        # The DSL omits the array entirely at zero slots, so absent means zero --
        # which still has to equal what the sidecar claims.
        self._refuses(
            self._header(mx=self._struct("mX", shapes=1, strides=0)),
            "declares no dynamic_strides",
        )
        # ...and a tensor claiming nothing is fine with no arrays at all.
        self._accepts(
            self._header(mx=self._struct("mX", shapes=0, strides=0)),
            tensor_args=_MX_NO_SLOTS,
        )

    def test_whitespace_variants_are_accepted(self):
        # `int64_t  dynamic_strides [ 1 ]` is the same declaration, and the DSL's own
        # C-type table stores some spellings with a trailing space.
        for label, decl in (
            ("two spaces", "int64_t  dynamic_strides[1];"),
            ("space before bracket", "int64_t dynamic_strides [1];"),
            ("spaces inside", "int64_t dynamic_strides[ 1 ];"),
            ("newline separated", "int64_t\n  dynamic_strides[1];"),
        ):
            with self.subTest(spelling=label):
                self._accepts(self._mx_header(decl))

    def test_a_non_literal_bound_is_refused(self):
        # Accepting it skipped the count check in silence, which is the mode this
        # guard exists to rule out.
        for bound in ("MX_NDYN", "1 + 1", ""):
            with self.subTest(bound=bound):
                member = f"int64_t dynamic_strides[{bound}];"
                self._refuses(self._mx_header(member), "not a literal count")

    def test_a_struct_declared_twice_is_refused(self):
        # An #ifdef'd 32-bit variant: which widths the compiler sees would depend
        # on the preprocessor, so there is no single answer to give.
        self._refuses(
            self._header() + self._struct("mX", stride_t="int32_t"),
            "declared 2 times",
        )

    def test_an_unparsable_header_is_refused_not_skipped(self):
        # Fails closed on a layout this code cannot read, so an upstream header change
        # cannot switch the guard off silently.
        self._refuses(
            "struct x { int64_t dynamic_strides[1]; };\n", "no `typedef struct"
        )
        self._refuses(
            f"/* ABI: {SIDECAR['prefix']}_Tensor_mX_t* */\n" + self._header(mx=""),
            "no `typedef struct",
        )

    def test_a_malformed_sidecar_is_refused_with_a_useful_message(self):
        # Each of these must name the file and the fix, not raise a bare
        # KeyError/TypeError out of a build step.
        self._refuses(
            self._header(), "names no tensor", tensor_args=[{"dynamic_strides": [0]}]
        )
        self._refuses(self._header(), "not a list", tensor_args={"mX": {}})
        self._refuses(
            self._header(),
            "not a list of",
            tensor_args=[dict(SIDECAR["tensor_args"][0], dynamic_strides="01")],
        )

    def test_a_kernel_with_no_stride_slots_is_not_checked_for_width(self):
        # Nothing to get wrong: the launcher emits no stride assignment at all.
        self._accepts(
            self._header(mx=self._struct("mX", shapes=1, strides=0)),
            tensor_args=[
                {"name": "mX", "dynamic_sizes": [0]},
                SIDECAR["tensor_args"][1],
            ],
        )

    def test_a_comment_is_not_read_as_the_declaration(self):
        # A search over the raw body takes the first textual match, so a commented-out
        # declaration above the real one wins.
        p = SIDECAR["prefix"]

        def hdr(comment, member):
            return (
                "#pragma once\n#include <stdint.h>\n"
                "typedef struct {\n  void* data;\n  int32_t dynamic_shapes[1];\n"
                f"  {comment}\n  {member}\n}} {p}_Tensor_mX_t;\n"
                + self._struct("mOut", shapes=2, strides=1)
            )

        one = _MX_ONE_SLOT
        for label, comment, member, targs in (
            (
                "hides int32",
                "/* int64_t dynamic_strides[1]; */",
                "int32_t dynamic_strides[1];",
                one,
            ),
            (
                "hides an under-claim",
                "/* int64_t dynamic_strides[1]; */",
                "int64_t dynamic_strides[3];",
                one,
            ),
            (
                "hides an over-claim",
                "/* room for int64_t dynamic_strides[3]; */",
                "int64_t dynamic_strides[1];",
                [dict(one[0], dynamic_strides=[0, 1, 2]), one[1]],
            ),
        ):
            with self.subTest(case=label):
                self._refuses(
                    hdr(comment, member), "dynamic_strides", tensor_args=targs
                )
        # ...while a comment BETWEEN the type and the member name is still the
        # same declaration, and must be accepted.
        self._accepts(
            hdr("", "int64_t /* 64-bit */ dynamic_strides[1];"), tensor_args=one
        )

    def test_a_terminator_this_parser_cannot_read_does_not_swallow_the_next_struct(
        self,
    ):
        # `} *Ptr_t;` is ordinary C, and a body regex running past it would check one
        # tensor against another's declarations.
        p = SIDECAR["prefix"]
        self._refuses(
            "#pragma once\n#include <stdint.h>\n"
            "typedef struct {\n  void* data;\n  int64_t dynamic_strides[1];\n"
            f"}} *{p}_TensorPtr_mA_t;\n"
            "typedef struct {\n  void* data;\n  int32_t dynamic_shapes[1];\n"
            f"  int32_t dynamic_strides[1];\n}} {p}_Tensor_mX_t;\n"
            + self._struct("mOut", shapes=2, strides=1),
            "dynamic_strides",
            tensor_args=_MX_ONE_SLOT,
        )

    def test_an_unreadable_struct_is_refused_even_when_nothing_is_claimed(self):
        # "Claims nothing, so nothing to check" is exactly the state that leaves every
        # slot the header does declare unwritten in an uninitialized local.
        p = SIDECAR["prefix"]
        self._refuses(
            "#pragma once\n#include <stdint.h>\n"
            f"typedef struct {p}_Tensor_mX_s {{\n  void* data;\n"
            "  int32_t dynamic_shapes[2];\n  int64_t dynamic_strides[2];\n"
            f"}} {p}_Tensor_mX_t;\n" + self._struct("mOut", shapes=2, strides=1),
            # The tag form is readable, so this refuses on the real problem: the
            # header declares slots the sidecar claims none of.
            r"sidecar's dynamic_\w+ claims 0 slot",
            tensor_args=_MX_NO_SLOTS,
        )
        # ...and where the declaration cannot be parsed at all, a zero claim is
        # refused rather than skipped.
        self._refuses(
            "#pragma once\n#include <stdint.h>\n"
            "typedef struct {\n  void* data;\n  int64_t dynamic_strides[2];\n"
            f"}} *{p}_TensorPtr_mX_t;\n" + self._struct("mOut", shapes=2, strides=1),
            "no `typedef struct",
            tensor_args=_MX_NO_SLOTS,
        )

    def test_an_unreadable_MEMBER_is_refused_even_when_nothing_is_claimed(self):
        # A declaration this parser cannot classify never enters `declared`, and the
        # absent-member arm would read that as zero slots. Both spellings are ordinary C.
        for label, member in (
            ("comma-separated declarator", "int64_t reserved[1], dynamic_strides[1];"),
            (
                "attribute before the name",
                "int64_t __attribute__((aligned(8))) dynamic_strides[1];",
            ),
        ):
            with self.subTest(spelling=label):
                self._refuses(
                    self._mx_header(member),
                    "could not read as a declaration",
                    tensor_args=_MX_NO_SLOTS,
                )

    def test_an_octal_bound_is_not_read_as_decimal(self):
        # C reads [010] as 8 where int("010") is 10, and comparing those counts is this
        # check's whole job, so such a bound is refused rather than parsed.
        self._refuses(
            self._mx_header("int64_t dynamic_strides[010];"),
            "has a leading zero, which C reads as octal",
            tensor_args=[
                {
                    "name": "mX",
                    "dynamic_sizes": [0],
                    "dynamic_strides": list(range(10)),
                },
                SIDECAR["tensor_args"][1],
            ],
        )

    def test_ordinary_c_spellings_of_the_same_declaration_are_accepted(self):
        # Each is what an upstream refactor would emit, and refusing one fails every
        # build. An allowlist, since assuming 64-bit restores the silent truncation.
        p = SIDECAR["prefix"]
        for label, decl in (
            ("long long", "long long dynamic_strides[1];"),
            ("signed long long", "signed long long dynamic_strides[1];"),
            ("std::int64_t", "std::int64_t dynamic_strides[1];"),
            ("u-suffixed bound", "int64_t dynamic_strides[1u];"),
        ):
            with self.subTest(spelling=label):
                self._accepts(self._mx_header(decl), tensor_args=_MX_ONE_SLOT)
        # The tag form and an attribute before the name are the same declaration.
        for label, header in (
            (
                "tag form",
                f"typedef struct Tag_mX {{\n  void* data;\n"
                f"  int32_t dynamic_shapes[1];\n  int64_t dynamic_strides[1];\n"
                f"}} {p}_Tensor_mX_t;\n",
            ),
            (
                "attribute",
                "typedef struct {\n  void* data;\n"
                "  int32_t dynamic_shapes[1];\n  int64_t dynamic_strides[1];\n"
                f"}} __attribute__((aligned(16))) {p}_Tensor_mX_t;\n",
            ),
        ):
            with self.subTest(form=label):
                self._accepts(
                    "#pragma once\n#include <stdint.h>\n"
                    + header
                    + self._struct("mOut", shapes=2, strides=1),
                    tensor_args=_MX_ONE_SLOT,
                )

    def test_an_unrelated_pointer_typedef_does_not_disturb_the_parse(self):
        # `typedef struct {...} *Ptr_t;` is ordinary C, and a body running past its
        # closing brace would merge it into the next struct.
        p = SIDECAR["prefix"]
        self._accepts(
            "#pragma once\n#include <stdint.h>\n"
            "typedef struct {\n  void* data;\n  int64_t dynamic_strides[1];\n"
            f"}} *{p}_TensorPtr_scratch_t;\n" + self._header(),
        )

    def test_an_unknown_stride_spelling_is_refused(self):
        # An allowlist of 64-bit spellings, not a refusal of int32_t: an unrecognized
        # type could be any width. The message says how to extend it.
        for spelling in ("int32_t", "int", "short", "cute_i64", "uint64_t", "float"):
            with self.subTest(type=spelling):
                self._refuses(
                    self._mx_header(f"{spelling} dynamic_strides[1];"),
                    "must be declared 64-bit",
                    tensor_args=_MX_ONE_SLOT,
                )

    def test_a_pathological_bound_is_refused_not_a_traceback(self):
        # int(bound) after bound.isdigit() raised ValueError for '\u00b2' (isdigit
        # is True) and for a bound past CPython's 4300-digit limit.
        for label, bound in (
            ("superscript two", "\u00b2"),
            ("5000 digits", "9" * 5000),
        ):
            with self.subTest(bound=label):
                self._refuses(
                    self._mx_header(f"int64_t dynamic_strides[{bound}];"),
                    "not a literal count",
                    tensor_args=_MX_ONE_SLOT,
                )

    def test_a_non_ascii_header_is_read_not_a_decode_error(self):
        # A plain open() uses the ambient locale, so a valid UTF-8 header raises
        # UnicodeDecodeError under LC_ALL=C.
        tc = toolchains.CuteDslToolchain()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, SIDECAR["prefix"] + ".h")
            # Invalid UTF-8 (a lone 0xFF), not merely non-ASCII: valid UTF-8 fails
            # only under LC_ALL=C, passing locally and breaking in a container.
            with open(path, "wb") as f:
                f.write(b"/* \xff */\n" + self._header().encode("utf-8"))
            tc.validate_abi(dict(SIDECAR, _dir=d))

    def test_a_header_that_cannot_be_read_is_not_silently_skipped(self):
        # Only ABSENT means "nothing to check", or the ABI goes unvalidated. A directory
        # stands in for the unreadable file, mode 000 being readable by root.
        with tempfile.TemporaryDirectory() as d:
            os.mkdir(os.path.join(d, SIDECAR["prefix"] + ".h"))
            with self.assertRaises(OSError):
                toolchains.CuteDslToolchain().validate_abi(dict(SIDECAR, _dir=d))

    def test_absent_header_is_not_an_error(self):
        # Unit fixtures have no header; a real generation always does.
        toolchains.CuteDslToolchain().validate_abi(dict(SIDECAR, _dir="/nonexistent"))


class TestToolchainRegistry(unittest.TestCase):
    def test_cutedsl_registered(self):
        self.assertIn("cutedsl", toolchains.TOOLCHAINS)

    def test_unknown_kind_raises(self):
        with self.assertRaisesRegex(RuntimeError, "unknown toolchain kind"):
            toolchains.get_toolchain("nvfuser")

    def test_builder_validation_names_missing_keys(self):
        tc = toolchains.get_toolchain("cutedsl")
        with self.assertRaisesRegex(RuntimeError, "missing keys.*fake_args"):
            tc.validate_build_result({"prefix": "x", "fn": object(), "tensor_args": []})


class TestDeclarationArchs(unittest.TestCase):
    def test_a_malformed_archs_entry_is_refused_at_load(self):
        # _SM_RE accepts "sm_9" and "sm_1000", which name no capability. Refused by the
        # loader, the only place that knows which file to name.
        for bad in ("sm_9", "sm_1000"):
            with self.subTest(archs=bad):
                with tempfile.TemporaryDirectory() as ops:
                    _write_fake_decl(ops, f"ARCHS = ({bad!r},)\n")
                    path = os.path.join(ops, "fakeop", "aot.py")
                    with self.assertRaisesRegex(RuntimeError, "compute capability"):
                        native_aot_decl.load_declarations(path)
                    # ...and the message names the file, which a refusal from
                    # inside export could not.
                    try:
                        native_aot_decl.load_declarations(path)
                    except RuntimeError as e:
                        self.assertIn("aot.py", str(e))


class TestDeclarationStaleness(unittest.TestCase):
    def test_source_closure_includes_the_declaration(self):
        # Declarations load by file path and never enter sys.modules, so without this a
        # grid edit reuses artifacts built from the previous one.
        with tempfile.TemporaryDirectory() as tmpdir:
            decl_path = os.path.join(tmpdir, "aot.py")
            with open(decl_path, "w") as f:
                f.write("ATEN_OP = 'fake'\n")
            closure = export.source_closure(decl_path)
            rel = os.path.relpath(decl_path, export.REPO)
            self.assertIn(rel, closure)
            self.assertEqual(closure[rel], export._file_hash(decl_path))

    def test_source_closure_without_declaration_is_unchanged(self):
        self.assertNotIn("aot.py", " ".join(export.source_closure()))


class TestStaleGridPointArtifacts(unittest.TestCase):
    def test_sidecar_for_dropped_grid_point_is_fatal(self):
        # Its .o is still matched by the CMake glob and would link with no
        # launcher referencing it.
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "gone.json"), "w") as f:
                json.dump({"prefix": "gone", "spec": {"N": 4096}}, f)
            with self.assertRaisesRegex(RuntimeError, "no longer in the grid"):
                export._check_no_orphan_artifacts(tmpdir, [{"N": 1024}])

    def test_sidecar_still_in_grid_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "live.json"), "w") as f:
                json.dump({"prefix": "live", "spec": {"N": 1024}}, f)
            export._check_no_orphan_artifacts(tmpdir, [{"N": 1024}])

    def test_grid_unknown_skips_the_stale_check(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "live.json"), "w") as f:
                json.dump({"prefix": "live", "spec": {"N": 1024}}, f)
            export._check_no_orphan_artifacts(tmpdir)


class TestRegistryConsistency(unittest.TestCase):
    def test_link_exts_must_be_a_subset_of_artifact_exts(self):
        # Generation links `if ext in link_exts`, so an ext the kind cannot produce
        # contributes nothing: a green link, then an undefined symbol at first call.
        class _Bad(toolchains.Toolchain):
            kind = "bad"
            artifact_exts = (".cubin", ".h")
            link_exts = (".o",)

        with self.assertRaisesRegex(RuntimeError, "not a subset"):
            toolchains._assert_link_exts_are_exportable({"bad": _Bad()})

    def test_link_exts_must_be_declared(self):
        # The realistic mistake is forgetting the attribute, which is a subset of
        # everything, so the subset rule alone would accept a kind that links nothing.
        class _Forgot(toolchains.Toolchain):
            kind = "forgot"
            artifact_exts = (".o", ".h")

        with self.assertRaisesRegex(RuntimeError, "link_exts is not declared"):
            toolchains._assert_link_exts_are_exportable({"forgot": _Forgot()})

        # ...while an EXPLICIT empty tuple stays a real answer, for a kind whose
        # launcher embeds the artifact instead of linking it.
        class _Embeds(toolchains.Toolchain):
            kind = "embeds"
            artifact_exts = (".cubin",)
            link_exts = ()

        toolchains._assert_link_exts_are_exportable({"embeds": _Embeds()})

    def test_the_shipped_toolchains_are_consistent(self):
        # The import-time call, re-run explicitly so this is a test rather than a
        # side effect of collection.
        toolchains._assert_link_exts_are_exportable(toolchains.TOOLCHAINS)


class TestMissingArtifacts(unittest.TestCase):
    def test_missing_artifacts_reexport_despite_current_sidecar(self):
        # Sidecar matches spec/arch/sources, but its .o is gone: skipping
        # here surfaces as a missing include when torch_cuda compiles.
        rel = os.path.relpath(
            os.path.join(os.path.dirname(__file__), "..", "native_aot", "decl.py"),
            export.REPO,
        )
        current = {rel: export._file_hash(os.path.join(export.REPO, rel))}
        point = {"dtype": "float32", "N": 4096}
        sidecar = {
            "version": export.SIDECAR_VERSION,
            "prefix": "x",
            "spec": point,
            "sources": current,
            "kind": "cutedsl",
        }
        with tempfile.TemporaryDirectory() as d:
            job = ("fakeop", "aot_kernel.py", point, d, None)
            with open(os.path.join(d, "x.json"), "w") as f:
                json.dump(sidecar, f)
            _touch_artifacts(d, "x")
            self.assertFalse(export._job_needed(job, force=False))

            os.remove(os.path.join(d, "x.o"))
            self.assertTrue(export._job_needed(job, force=False))

    def test_missing_header_also_reexports(self):
        rel = os.path.relpath(
            os.path.join(os.path.dirname(__file__), "..", "native_aot", "decl.py"),
            export.REPO,
        )
        current = {rel: export._file_hash(os.path.join(export.REPO, rel))}
        point = {"dtype": "float32", "N": 4096}
        with tempfile.TemporaryDirectory() as d:
            job = ("fakeop", "aot_kernel.py", point, d, None)
            with open(os.path.join(d, "x.json"), "w") as f:
                json.dump(
                    {
                        "version": export.SIDECAR_VERSION,
                        "prefix": "x",
                        "spec": point,
                        "sources": current,
                        "kind": "cutedsl",
                    },
                    f,
                )
            _touch_artifacts(d, "x", exts=(".o",))  # .h missing
            self.assertTrue(export._job_needed(job, force=False))


if __name__ == "__main__":
    unittest.main()
