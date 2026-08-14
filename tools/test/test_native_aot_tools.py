from __future__ import annotations

import importlib
import json
import os
import tempfile
import unittest
import unittest.mock as mock

# Ordinary package imports, like every other tools test (CI runs
# `PYTHONPATH=$(pwd) pytest tools/test`). These modules keep their module
# scope torch-free precisely so this works: the Test tools job runs in the
# linter image, which has no built torch.
from tools.native_aot import build_stage2, export, gen_aot_lib, toolchains


_TOOLS_FILE = os.path.abspath(toolchains.__file__)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(_TOOLS_FILE)))


SIDECAR = {
    "prefix": "fakeop_f32_n1024_k8",
    # Every sidecar records the arch it was compiled for and the kind that
    # built it; generation reads both rather than defaulting either, so a
    # fixture missing them is not a sidecar export could have written.
    "arch": "sm_100a",
    "kind": "cutedsl",
    "spec": {"dtype": "float32", "N": 1024, "K": 8, "deterministic": False},
    "tensor_args": [
        {"name": "mX", "dynamic_sizes": [0], "dynamic_strides": [0]},
        {"name": "mOut", "dynamic_sizes": [0, 1], "dynamic_strides": [0]},
    ],
}


def _no_device_arch():
    """Patch out local-device arch detection.

    _effective_arch resolves an unspecified arch to the builder's GPU, so a
    sidecar written without one is legitimately stale on any CUDA machine.
    Tests about spec/source matching say nothing about arch and must not
    depend on whether the runner has a GPU. Imports mock locally to stay
    self-contained."""
    import unittest.mock as mock

    return mock.patch.object(export, "_detected_arch", return_value=None)


def _touch_artifacts(out_dir, prefix, exts=(".o", ".h")):
    """Create the files a sidecar claims. _job_needed verifies they exist,
    so a fixture that writes only the .json would always re-export."""
    for e in exts:
        with open(os.path.join(out_dir, prefix + e), "w") as f:
            f.write("")


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
            _touch_artifacts(d, "x")
            with open(os.path.join(d, "x.json"), "w") as f:
                _json.dump(
                    {
                        "version": export.SIDECAR_VERSION,
                        "prefix": "x",
                        "kind": "cutedsl",
                        "spec": point,
                        "sources": current,
                    },
                    f,
                )
            with _no_device_arch():
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
                "kind": "cutedsl",
                "spec": export._json_normal(point),
                "sources": current,
            }
            _touch_artifacts(d, "x")
            _touch_artifacts(d, "x")
            with open(os.path.join(d, "x.json"), "w") as f:
                _json.dump(sidecar, f)
            with _no_device_arch():
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
    def test_effective_arch_is_per_toolchain(self):
        # Only kinds that declare an ARCH_ENV_VAR read one, so a kind that
        # takes its target another way (Triton: an explicit GPUTarget) is
        # not perturbed by CUTE_DSL_ARCH.
        import unittest.mock as mock

        cutedsl = toolchains.get_toolchain("cutedsl")
        no_env = toolchains.Toolchain()
        self.assertIsNone(no_env.ARCH_ENV_VAR)
        # _no_device_arch: the last fallback is the builder's GPU, which would
        # otherwise mask what this test is about.
        with (
            mock.patch.dict(os.environ, {"CUTE_DSL_ARCH": "sm_90a"}),
            _no_device_arch(),
        ):
            self.assertEqual(export._effective_arch(None, cutedsl), "sm_90a")
            self.assertIsNone(export._effective_arch(None, no_env))
            # An explicit arch always wins over the env var.
            self.assertEqual(export._effective_arch("sm_100a", cutedsl), "sm_100a")
        # Precedence below the env var: local device, so an on-device export
        # still records the arch it compiled for.
        import unittest.mock as _mock

        with _mock.patch.object(export, "_detected_arch", return_value="sm_100"):
            self.assertEqual(export._effective_arch(None, cutedsl), "sm_100")

    def test_arch_tag_is_short(self):
        # The tag lands in every exported C symbol, so its shape is part of
        # the artifact ABI: one underscore dropped, nothing else.
        self.assertEqual(export._arch_tag("sm_100a"), "sm100a")
        self.assertEqual(export._arch_tag("sm_90"), "sm90")

    def test_conflicting_toolchain_arch_vars_are_an_error(self):
        # With no toolchain named, _effective_arch scans every registered kind
        # for its arch variable. Two kinds set to DIFFERENT arches have no
        # single answer, and taking the first would make the artifact tree
        # depend on registry order -- so it must say so instead.
        import unittest.mock as mock

        class _Other(toolchains.Toolchain):
            kind = "other"
            ARCH_ENV_VAR = "OTHER_DSL_ARCH"

        registry = dict(toolchains.TOOLCHAINS, other=_Other())
        with (
            mock.patch.dict(toolchains.TOOLCHAINS, registry, clear=True),
            mock.patch.dict(
                os.environ, {"CUTE_DSL_ARCH": "sm_90a", "OTHER_DSL_ARCH": "sm_100a"}
            ),
            _no_device_arch(),
        ):
            with self.assertRaisesRegex(RuntimeError, "conflicting toolchain arch"):
                export._effective_arch(None)
            # Agreeing values are not a conflict: there is one answer.
            with mock.patch.dict(os.environ, {"OTHER_DSL_ARCH": "sm_90a"}):
                self.assertEqual(export._effective_arch(None), "sm_90a")

    def test_export_prefix_is_arch_qualified(self):
        # Two arches must not produce the same prefix: the exported symbols
        # (cute_dsl_<prefix>_wrapper, <prefix>_Kernel_Module_Load) are derived
        # from it, so an unqualified prefix is a duplicate definition when both
        # arches link into one libtorch_cuda.
        import unittest.mock as mock

        seen = []

        class _FakeTc(toolchains.Toolchain):
            kind = "cutedsl"
            artifact_exts = (".o", ".h")

            def missing_runtimes(self):
                return []

            def validate_build_result(self, b):
                pass

            def export(self, b, out_dir, arch=None):
                seen.append(b["prefix"])
                _touch_artifacts(out_dir, b["prefix"])
                return {"tensor_args": []}

        fake = _FakeTc()
        with tempfile.TemporaryDirectory() as d:
            with (
                mock.patch.object(
                    export,
                    "load_builder",
                    return_value=lambda p: {"prefix": "k", "kind": "cutedsl"},
                ),
                mock.patch.object(toolchains, "get_toolchain", return_value=fake),
            ):
                for arch in ("sm_90a", "sm_100a"):
                    export.export_point("fakeop", "aot_kernel.py", {"n": 1}, d, arch)
        self.assertEqual(seen, ["k__sm90a", "k__sm100a"])

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
        # The reported case, from the other side: a sidecar that recorded no
        # arch at all (an on-device export) must NOT satisfy a run whose
        # CUTE_DSL_ARCH names a target, or the env-var run silently inherits
        # objects built for whatever the builder's GPU was.
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
            # ...and where an arch IS resolvable it does not satisfy an
            # on-device run either: that run knows its arch, the sidecar names
            # none. _detected_arch patched rather than trusted, so the claim
            # does not depend on the runner having a GPU.
            with (
                mock.patch.dict(os.environ, {}, clear=True),
                mock.patch.object(export, "_detected_arch", return_value="sm_100"),
            ):
                self.assertTrue(export._job_needed(job, force=False))
                # Only where no arch can be resolved at all is it a match.
                with _no_device_arch():
                    self.assertFalse(export._job_needed(job, force=False))

    def test_archs_from_cuda_arch_list(self):
        # TORCH_CUDA_ARCH_LIST -> the EXPORTABLE_ARCHES subset; named,
        # malformed, +PTX and non-exportable entries drop out.
        f = export.archs_from_cuda_arch_list
        self.assertEqual(f("7.5 8.9"), [])
        # Hopper and Blackwell together: one tree per arch, selected at
        # runtime by capability.
        self.assertEqual(f("9.0a;10.0a"), ["sm_90a", "sm_100a"])
        self.assertEqual(f("8.0 9.0 10.0+PTX"), ["sm_90", "sm_100"])
        # Both spellings of a CC are separate nvcc targets, and both are
        # exportable: CI passes "10.0a", the wheel builds pass "10.0".
        self.assertEqual(f("10.0"), ["sm_100"])
        # 10.3 stays unexportable -- nothing names it, so shipping it would
        # only grow wheels (the gate itself is major.minor and would be safe).
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
            # _detected_arch patched: the on-device call below resolves the
            # directory from it, and an unpatched run would pass only on a
            # machine with a GPU -- this suite must also pass in the linter
            # image, which has no built torch at all.
            with (
                mock.patch.object(export, "OPS_DIR", ops),
                mock.patch.object(export, "_detected_arch", return_value="sm_100a"),
            ):
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
            # _detected_arch patched, not read from the runner: the layout must
            # be the same shape everywhere, and an unpatched call would make
            # this test depend on whether the machine has a GPU.
            with (
                mock.patch.object(export, "OPS_DIR", ops),
                mock.patch.object(export, "_detected_arch", return_value="sm_100"),
            ):
                multi = export._collect_jobs(None, out, ["sm_90a", "sm_100a"])
                single = export._collect_jobs(None, out, [None])
        self.assertEqual(len(multi), 2)
        dirs = sorted(os.path.basename(os.path.dirname(j[3])) for j in multi)
        self.assertEqual(dirs, ["sm_100a", "sm_90a"])
        self.assertEqual({j[4] for j in multi}, {"sm_90a", "sm_100a"})
        # ONE layout: a single arch nests under its own directory too, so
        # adding an arch to an op is just another directory.
        (sj,) = single
        self.assertEqual(os.path.basename(sj[3]), "fakeop")
        self.assertEqual(os.path.basename(os.path.dirname(sj[3])), "sm_100")
        # The job still carries no explicit arch: the compile target stays the
        # DSL's own device default (which may use arch-conditional features),
        # only the directory is named.
        self.assertIsNone(sj[4])

    def test_by_arch_groups_and_orders_by_capability(self):
        scs = [
            {"prefix": "k__sm100a", "arch": "sm_100a"},
            {"prefix": "k__sm90a", "arch": "sm_90a"},
            {"prefix": "k2__sm90a", "arch": "sm_90a"},
        ]
        groups = gen_aot_lib._by_arch(scs)
        self.assertEqual(list(groups), [(9, 0), (10, 0)])
        self.assertEqual(
            [s["prefix"] for s in groups[(9, 0)]], ["k__sm90a", "k2__sm90a"]
        )

    def test_by_arch_prefers_the_arch_conditional_build(self):
        # Both are valid on 10.0 hardware; the conditional build is the one the
        # kernels were written against, and shipping both would otherwise let
        # directory order decide.
        for order in (
            [("sm_100", "p"), ("sm_100a", "c")],
            [("sm_100a", "c"), ("sm_100", "p")],
        ):
            scs = [{"prefix": n, "arch": a} for a, n in order]
            groups = gen_aot_lib._by_arch(scs)
            self.assertEqual(list(groups), [(10, 0)])
            self.assertEqual([s["prefix"] for s in groups[(10, 0)]], ["c"])

    def test_by_arch_rejects_an_arch_less_sidecar(self):
        # Export names the arch of everything it writes, so this is a tree from
        # before it did. Rejected rather than grouped: a capability nothing
        # matches would emit a branch that declines every call in silence.
        with self.assertRaisesRegex(RuntimeError, "records no arch"):
            gen_aot_lib._by_arch([{"prefix": "old", "arch": None}])

    def test_dropped_tie_break_candidate_gets_no_launcher(self):
        # Both spellings of a capability are exportable ("10.0;10.0a"), so the
        # plain build loses the tie-break -- and a launcher emitted for it would
        # be defined and never called, i.e. -Wunused-function in an anonymous
        # namespace, fatal under CI's WERROR.
        class _Decl:
            ATEN_OP = "fakeop"
            DISPATCH_KEY = "CUDA"
            ARCHS = ("sm_100", "sm_100a")

            @staticmethod
            def cpp_dispatch(spec):
                return "true"

            @staticmethod
            def cpp_launch(spec, launch_fn):
                return f"{launch_fn}(self, out, at::cuda::getCurrentCUDAStream());"

        def sc(arch):
            return dict(
                SIDECAR, prefix=f"fakeop_p__{arch.replace('_', '', 1)}", arch=arch
            )

        src = gen_aot_lib.gen_op(
            "fakeop",
            "CUDA",
            _Decl,
            [sc("sm_100"), sc("sm_100a")],
            "const at::Tensor & self, const at::Tensor & out",
        )
        self.assertIn("launch_fakeop_p__sm100a(", src)
        self.assertNotIn("launch_fakeop_p__sm100(", src)

    def test_cc_of_and_device_match(self):
        self.assertEqual(gen_aot_lib._cc_of("sm_90"), (9, 0))
        self.assertEqual(gen_aot_lib._cc_of("sm_103a"), (10, 3))
        m = gen_aot_lib._device_match(10, 3)
        self.assertIn("major == 10", m)
        self.assertIn("minor == 3", m)


class TestSidecarIntegrity(unittest.TestCase):
    """The sidecar is written after the artifacts, so it is the commit
    marker: absent means not-yet-exported, corrupt or orphaned means the
    tree cannot be trusted."""

    def test_empty_dir_is_fine(self):
        # A clean build (or a newly added spec point) has no sidecar and
        # no artifacts; it must export, not fail.
        with tempfile.TemporaryDirectory() as d:
            export._check_no_orphan_artifacts(d, [])

    def test_artifacts_without_sidecar_are_fatal(self):
        # An export that died between compiling and writing the sidecar.
        # The CMake globs link *.o by pattern, so an undescribed orphan
        # would otherwise be linked silently.
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "k_f32.o"), "w").close()
            with self.assertRaisesRegex(RuntimeError, "no sidecar"):
                export._check_no_orphan_artifacts(d, [])

    def test_artifacts_with_sidecar_are_fine(self):
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "k.o"), "w").close()
            with open(os.path.join(d, "k.json"), "w") as f:
                json.dump({"prefix": "k", "kind": "cutedsl", "spec": {"N": 1}}, f)
            export._check_no_orphan_artifacts(d, [{"N": 1}])

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


class TestLauncherGeneration(unittest.TestCase):
    def test_marshalling_from_sidecar(self):
        src = gen_aot_lib.gen_launcher(SIDECAR)
        # Struct fill: one dynamic size per listed dim, in slot order.
        self.assertIn("mX_s.dynamic_shapes[0] = static_cast<int32_t>(mX.size(0));", src)
        self.assertIn("mX_s.dynamic_strides[0] = mX.stride(0);", src)
        self.assertIn(
            "mOut_s.dynamic_shapes[1] = static_cast<int32_t>(mOut.size(1));", src
        )
        # Lazy module load + wrapper call with all tensor structs.
        self.assertIn("c10::call_once(fakeop_f32_n1024_k8_loaded", src)
        self.assertIn(
            "cute_dsl_fakeop_f32_n1024_k8_wrapper(&fakeop_f32_n1024_k8_module, &mX_s, &mOut_s,",
            src,
        )


class TestScalarArgs(unittest.TestCase):
    def test_launcher_forwards_scalars_by_value(self):
        sidecar = dict(
            SIDECAR,
            tensor_args=[{"name": "mX", "dynamic_sizes": [0]}],
            scalar_args=[{"name": "alpha", "ctype": "float"}],
        )
        src = gen_aot_lib.gen_launcher(sidecar)
        # Scalars appear after tensors in both the params and the call,
        # matching the exported wrapper's argument order.
        self.assertIn("const at::Tensor& mX, float alpha, c10::Stream stream", src)
        self.assertIn("(&fakeop_f32_n1024_k8_module, &mX_s, alpha,", src)
        # The shared contract carries a device-agnostic c10::Stream; the
        # body narrows it to the raw handle the C ABI takes.
        self.assertIn("c10::cuda::CUDAStream(stream).stream()", src)

    def test_no_scalar_args_unchanged(self):
        src = gen_aot_lib.gen_launcher(SIDECAR)
        self.assertIn(
            "const at::Tensor& mX, const at::Tensor& mOut, c10::Stream stream", src
        )


class _FakeDecl:
    """Minimal declaration object for gen_op tests (the real contract
    lives in decl.py; gen_op only touches these attributes)."""

    ATEN_OP = "fakeop"
    DISPATCH_KEY = "CUDA"
    # Set explicitly: fixtures bypass the validating loader, which is
    # what normalizes ARCHS on real declarations. Annotated so
    # subclasses can narrow it without a bad-override.
    ARCHS: tuple[str, ...] = ("sm_90", "sm_90a", "sm_100", "sm_100a")

    @staticmethod
    def cpp_dispatch_prelude():
        return (
            "if (self.scalar_type() != at::kFloat) return false;\n"
            "const int64_t N = self.size(-1);"
        )

    @staticmethod
    def cpp_dispatch(spec):
        return f"N == {spec['N']} && k == {spec['K']}"

    @staticmethod
    def cpp_launch(spec, launch_fn):
        return f"{launch_fn}(self, out, at::cuda::getCurrentCUDAStream());"


class TestInt32SizeGate(unittest.TestCase):
    # The DSL's exported ABI carries int32_t shape slots while aten sizes
    # are int64_t, so the generated gate must DECLINE oversized dims
    # rather than let the launcher's static_cast truncate them.
    def test_gate_covers_plain_and_optional_tensors(self):
        gate = gen_aot_lib._int32_size_gate(
            "const at::Tensor & self, int64_t k, "
            "const std::optional<at::Tensor>& values"
        )
        # Plain tensor: unconditional scan.
        self.assertIn("self.sizes().begin()", gate)
        # Optional tensor: has_value() guarded, arrow deref.
        self.assertIn("values.has_value()", gate)
        self.assertIn("values->sizes().begin()", gate)
        # Declines, never truncates -- and does so as ONE guarded statement,
        # so an oversized dim in any tensor leaves via the same early return.
        self.assertEqual(gate.count("return false;"), 1)
        self.assertIn("if (C10_UNLIKELY(", gate)
        self.assertTrue(gate.rstrip().endswith(")) return false;"), gate)
        # Non-tensor params contribute nothing: a wrongly-included scalar
        # would emit its name in a sizes() probe.
        self.assertNotIn("k.sizes()", gate)

    def test_gate_empty_without_tensors(self):
        self.assertEqual(gen_aot_lib._int32_size_gate("int64_t k, bool largest"), "")

    def test_gate_emitted_into_both_stub_and_covers(self):
        # Both the stub prelude and cpp_covers get it: coverage must not
        # claim a shape the stub will refuse.
        sidecar = dict(SIDECAR, spec={"N": 1024, "K": 8})
        src = gen_aot_lib.gen_op(
            "fakeop",
            "CUDA",
            _FakeDecl,
            [sidecar],
            "const at::Tensor & self, int64_t k",
        )
        self.assertIn("inline bool _naot_dim_too_big", src)
        # _FakeDecl has no cpp_covers, so only the stub gate is emitted.
        # The covers side is covered by test_gate_covers_plain_and_optional.
        self.assertEqual(src.count("// Size gate:"), 1)
        self.assertIn("self.sizes().begin()", src)


class TestAotSourceGeneration(unittest.TestCase):
    def test_full_source(self):
        sidecar = dict(SIDECAR, spec={"N": 1024, "K": 8})
        src = gen_aot_lib.gen_op(
            "fakeop",
            "CUDA",
            _FakeDecl,
            [sidecar],
            "const at::Tensor & self, int64_t k, const at::Tensor & out",
        )
        # Prelude verbatim (indented) ahead of the chain.
        self.assertIn("if (self.scalar_type() != at::kFloat) return false;", src)
        # Dispatch branch wraps the launch with the point's launcher name.
        self.assertIn("if (N == 1024 && k == 8) {", src)
        self.assertIn(
            "launch_fakeop_f32_n1024_k8(self, out, at::cuda::getCurrentCUDAStream());",
            src,
        )
        # Falls through to false: the LAST statement of the kernel body, which
        # is what routes an unmatched call to op.impl. Asserted positionally --
        # the device gate and the size gate each emit a "return false;" too, so
        # a substring check here passes even with no fallback at all.
        body = src.split("bool fakeop_cuda_aot_kernel(", 1)[1]
        body = body[: body.index("\n}\n")]
        self.assertTrue(body.rstrip().endswith("return false;"), body[-200:])
        # Registers on the generated DispatchStub.
        self.assertIn(
            "REGISTER_CUDA_DISPATCH(fakeop_aot_stub, &::fakeop_cuda_aot_kernel)", src
        )
        self.assertIn('#include "fakeop_f32_n1024_k8.h"', src)

    def test_branches_in_sidecar_order(self):
        s1 = dict(SIDECAR, spec={"N": 1024, "K": 8})
        s2 = dict(SIDECAR, prefix="fakeop_f32_n2048_k8", spec={"N": 2048, "K": 8})
        src = gen_aot_lib.gen_op(
            "fakeop", "CUDA", _FakeDecl, [s1, s2], "const at::Tensor & self, int64_t k"
        )
        self.assertLess(src.index("N == 1024"), src.index("N == 2048"))

    def test_prelude_optional(self):
        class NoPrelude(_FakeDecl):
            # The contract's "absent prelude" case; the deliberate
            # callable -> None override trips bad-override.
            cpp_dispatch_prelude = None  # pyrefly: ignore [bad-override]

        sidecar = dict(SIDECAR, spec={"N": 1024, "K": 8})
        src = gen_aot_lib.gen_op(
            "fakeop", "CUDA", NoPrelude, [sidecar], "const at::Tensor & self, int64_t k"
        )
        self.assertNotIn("scalar_type() != at::kFloat", src)
        self.assertIn("if (N == 1024 && k == 8) {", src)

    def test_cpp_covers_emission(self):
        # cpp_covers -> a bool fn over the schema params + a
        # TORCH_LIBRARY_FRAGMENT registration in the _native_aot ns.
        sidecar = dict(SIDECAR, spec={"N": 1024, "K": 8})
        covers = (
            "const at::Tensor & self, int64_t k",
            "covers_fakeop(Tensor self, int k) -> bool",
            "return self.scalar_type() == at::kFloat && k == 8;",
        )
        src = gen_aot_lib.gen_op(
            "fakeop",
            "CUDA",
            _FakeDecl,
            [sidecar],
            "const at::Tensor & self, int64_t k",
            covers,
        )
        self.assertIn(
            "bool fakeop_cuda_covers(const at::Tensor & self, int64_t k) {", src
        )
        self.assertIn("TORCH_LIBRARY_FRAGMENT(_native_aot, m) {", src)
        self.assertIn(
            'm.def("covers_fakeop(Tensor self, int k) -> bool", &::fakeop_cuda_covers);',
            src,
        )

    def test_cpp_covers_absent_no_registration(self):
        sidecar = dict(SIDECAR, spec={"N": 1024, "K": 8})
        src = gen_aot_lib.gen_op(
            "fakeop", "CUDA", _FakeDecl, [sidecar], "const at::Tensor & self, int64_t k"
        )
        self.assertNotIn("TORCH_LIBRARY_FRAGMENT", src)
        self.assertNotIn("fakeop_cuda_covers", src)

    def test_covers_signature_from_schema(self):
        # Functional schema args + out-variant outputs as trailing
        # optionals; SymInt degrades to int.
        params, schema = gen_aot_lib.covers_signature("topk")
        self.assertIn("int64_t k", params)
        self.assertIn("const std::optional<at::Tensor>& values", params)
        self.assertIn("const std::optional<at::Tensor>& indices", params)
        self.assertEqual(
            schema,
            "covers_topk(Tensor self, int k, int dim=-1, bool largest=True, bool sorted=True, Tensor? values=None, Tensor? indices=None) -> bool",
        )

    def test_covers_signature_kwarg_only_and_defaults(self):
        # Pin the tricky schema shapes the per-argument rendering must
        # handle: a kwarg-only section (the '*' marker), a list default
        # ('int[1] dim=[]'), and Scalar defaults -- surgery on the whole
        # schema string mangles these.
        _, schema = gen_aot_lib.covers_signature("sum.dim_IntList")
        self.assertEqual(
            schema,
            "covers_sum_dim_IntList(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor? out=None) -> bool",
        )
        _, schema = gen_aot_lib.covers_signature("amax")
        self.assertEqual(
            schema,
            "covers_amax(Tensor self, int[1] dim=[], bool keepdim=False, Tensor? out=None) -> bool",
        )
        _, schema = gen_aot_lib.covers_signature("add.Tensor")
        self.assertEqual(
            schema,
            "covers_add_Tensor(Tensor self, Tensor other, *, Scalar alpha=1, Tensor? out=None) -> bool",
        )


class TestMultiCapabilitySelector(unittest.TestCase):
    """One generated .cpp serves every arch an op shipped for, so its selector
    is what keeps each artifact on its own hardware. _by_arch's grouping and
    the absence of dead launchers are covered above; this pins the SHAPE of
    what gen_op emits from those groups, which an inverted or misplaced gate
    passes through unchanged."""

    def _body(self):
        # Deliberately passed newest-first: the emitted order must come from
        # the capability, not from directory or argument order.
        s100 = dict(
            SIDECAR, prefix="fakeop_p__sm100a", arch="sm_100a", spec={"N": 1024, "K": 8}
        )
        s90 = dict(
            SIDECAR, prefix="fakeop_p__sm90a", arch="sm_90a", spec={"N": 1024, "K": 8}
        )
        src = gen_aot_lib.gen_op(
            "fakeop",
            "CUDA",
            _FakeDecl,
            [s100, s90],
            "const at::Tensor & self, int64_t k, const at::Tensor & out",
        )
        body = src.split("bool fakeop_cuda_aot_kernel(", 1)[1]
        return body[: body.index("\n}\n")]

    def test_properties_read_once_before_any_gate(self):
        body = self._body()
        self.assertEqual(body.count("= at::cuda::getCurrentDeviceProperties();"), 1)
        self.assertLess(body.index("_naot_props ="), body.index("_naot_props->"))

    def test_early_out_accepts_exactly_the_shipped_capabilities(self):
        body = self._body()
        accept = next(l for l in body.splitlines() if l.startswith("  if (!(("))
        self.assertEqual(accept.count("major =="), 2)
        self.assertIn("major == 9 && _naot_props->minor == 0", accept)
        self.assertIn("major == 10 && _naot_props->minor == 0", accept)

    def test_each_capability_launches_only_its_own_kernels(self):
        # The cross pairing is the failure this whole grouping exists to
        # prevent: loading a module built for other hardware fails inside the
        # launcher instead of declining to aten.
        body = self._body()
        i9 = body.index("_naot_props->major == 9 && _naot_props->minor == 0) {")
        i10 = body.index("_naot_props->major == 10 && _naot_props->minor == 0) {")
        self.assertLess(i9, i10)
        sm90_branch, sm100_branch = body[i9:i10], body[i10:]
        self.assertIn("launch_fakeop_p__sm90a(", sm90_branch)
        self.assertNotIn("launch_fakeop_p__sm100a(", sm90_branch)
        self.assertIn("launch_fakeop_p__sm100a(", sm100_branch)
        self.assertNotIn("launch_fakeop_p__sm90a(", sm100_branch)


class TestReadOnlyInputs(unittest.TestCase):
    # read_only tensor args must go through const_data_ptr in every
    # toolchain's launcher: a mutable data_ptr() materializes
    # copy-on-write inputs on each call.

    def test_cutedsl_launcher(self):
        sc: dict = dict(SIDECAR)
        sc["tensor_args"] = [
            {"name": "mX", "dynamic_sizes": [0], "read_only": True},
            {"name": "mOut", "dynamic_sizes": [0]},
        ]
        src = gen_aot_lib.gen_launcher(sc)
        self.assertIn("mX_s.data = const_cast<void*>(mX.const_data_ptr());", src)
        self.assertIn("mOut_s.data = mOut.mutable_data_ptr();", src)

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

    def test_closure_excludes_the_consumer_tools(self):
        # Neither can change what an artifact MEANS, and hashing them is
        # expensive in a way that is easy to miss: every kernel of every arch
        # re-exports (minutes) for an edit that could not have changed one.
        # gen_aot_lib.py only reads sidecars; build_stage2.py only decides
        # whether stage 2 runs and then relinks -- export.py reads the arch
        # list itself, so the driver passes it nothing kernel-affecting.
        names = {os.path.basename(p) for p in export.source_closure()}
        for unwanted in ("gen_aot_lib.py", "build_stage2.py"):
            self.assertNotIn(unwanted, names)

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
            _touch_artifacts(d, "x")
            with open(os.path.join(d, "x.json"), "w") as f:
                _json.dump(
                    {
                        "version": export.SIDECAR_VERSION,
                        "prefix": "x",
                        "kind": "cutedsl",
                        "spec": point,
                        "sources": current,
                    },
                    f,
                )
            with _no_device_arch():
                self.assertFalse(export._job_needed(job, force=False))
            _touch_artifacts(d, "x")
            with open(os.path.join(d, "x.json"), "w") as f:
                _json.dump(
                    {
                        "prefix": "x",
                        "kind": "cutedsl",
                        "spec": point,
                        "sources": {rel: "0" * 16},
                    },
                    f,
                )
            self.assertTrue(export._job_needed(job, force=False))


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

    def test_cmake_globs_cover_all_toolchains(self):
        # The embedded link block in caffe2/CMakeLists.txt must glob every
        # artifact pattern the toolchains emit (it cannot import this file).
        cmake_path = os.path.join(REPO, "caffe2", "CMakeLists.txt")
        with open(cmake_path) as f:
            cmake = f.read()
        # Exact patterns, not basenames: the depths are the point now that a
        # multi-arch export nests <arch>/<op>/ under the artifacts root, and
        # "*/*.o" is a substring of "*/*/*.o" so a basename check would pass
        # with either depth missing.
        for tc in toolchains.TOOLCHAINS.values():
            for pattern in tc.link_source_globs:
                self.assertIn(f'"${{NATIVE_AOT_ARTIFACTS_DIR}}/{pattern}"', cmake)


CUBIN_SIDECAR = {
    "prefix": "fakemm_f32",
    "kind": "triton",
    "symbol": "_fakemm_kernel",
    "spec": {"dtype": "float32"},
    "launch": {"grid_x": "B_dim*((M+31)/32)"},
    "shared": 512,
    "block_x": 128,
    "args": [
        {"name": "a", "kind": "tensor"},
        {"name": "B_dim", "kind": "scalar", "ctype": "int32_t"},
        {"name": "M", "kind": "scalar", "ctype": "int32_t"},
    ],
}


class TestEndToEndGeneration(unittest.TestCase):
    _DECL = (
        'ATEN_OP = "fakeop"\n'
        'DISPATCH_KEY = "CUDA"\n'
        'KERNEL_MODULE = "kernel.py"\n'
        "def kernel_precompile_grid():\n"
        '    return [{"N": 1024, "K": 8}]\n'
        "def covered_axes(self, k):\n"
        '    return {"N": 0, "K": k}\n'
        "def cpp_dispatch(spec):\n"
        "    return f\"N == {spec['N']}\"\n"
        "def cpp_launch(spec, launch_fn):\n"
        '    return f"{launch_fn}();"\n'
    )

    def test_main_writes_aot_source(self):
        # Artifacts dir with sidecars but no .o (generation is text-only);
        # declaration read from a patched ops dir.
        with tempfile.TemporaryDirectory() as art, tempfile.TemporaryDirectory() as ops:
            # Artifacts live at <root>/<arch>/<decl_id>/ -- the one layout,
            # whatever the arch count.
            art_op = os.path.join(art, "sm_100a", "fakeop")
            os.makedirs(art_op)
            rel = os.path.relpath(
                os.path.join(os.path.dirname(__file__), "..", "native_aot", "decl.py"),
                export.REPO,
            )
            current = {rel: export._file_hash(os.path.join(export.REPO, rel))}
            sidecar = dict(
                SIDECAR,
                spec={"N": 1024, "K": 8},
                sources=current,
                version=export.SIDECAR_VERSION,
            )
            with open(os.path.join(art_op, SIDECAR["prefix"] + ".json"), "w") as f:
                json.dump(sidecar, f)
            os.makedirs(os.path.join(ops, "fakeop"))
            with open(os.path.join(ops, "fakeop", "aot.py"), "w") as f:
                f.write(self._DECL)

            orig_ops, orig_impl, orig_pre = (
                gen_aot_lib.OPS_DIR,
                gen_aot_lib.impl_signature_params,
                gen_aot_lib.precomputed_args,
            )
            gen_aot_lib.OPS_DIR = ops
            gen_aot_lib.impl_signature_params = (
                lambda op: "const at::Tensor & self, int64_t k"
            )
            gen_aot_lib.precomputed_args = lambda op: []
            try:
                import sys

                argv = sys.argv
                sys.argv = ["gen_aot_lib.py", "--artifacts-dir", art]
                try:
                    gen_aot_lib.main()
                finally:
                    sys.argv = argv
            finally:
                gen_aot_lib.OPS_DIR = orig_ops
                gen_aot_lib.impl_signature_params = orig_impl
                gen_aot_lib.precomputed_args = orig_pre

            out = os.path.join(art, "fakeop", "aot_fakeop_cuda.cpp")
            self.assertTrue(os.path.exists(out))
            with open(out) as f:
                self.assertIn("fakeop_cuda_aot_kernel", f.read())

    def test_main_tolerates_missing_artifacts_dir(self):
        # Zero declarations: export creates no artifacts dir at all
        # (stage 2 on a repo state with no ops). main() must no-op, not
        # FileNotFoundError (regressed on the sm100 CI build of the
        # commit before any op lands).
        import sys

        with tempfile.TemporaryDirectory() as d:
            argv = sys.argv
            sys.argv = ["gen_aot_lib.py", "--artifacts-dir", os.path.join(d, "absent")]
            try:
                gen_aot_lib.main()
            finally:
                sys.argv = argv


class TestWheelPatch(unittest.TestCase):
    LIB = "torch/lib/libtorch_cuda.so"

    def _make_wheel(self, d: str, record_entry: bool = True) -> str:
        import base64
        import hashlib
        import zipfile

        def rec(name, data):
            h = base64.urlsafe_b64encode(hashlib.sha256(data).digest())
            return f"{name},sha256={h.rstrip(b'=').decode()},{len(data)}"

        whl = os.path.join(d, "torch-0.0-cp310-linux_x86_64.whl")
        record = "torch-0.0.dist-info/RECORD"
        lines = [rec("torch/__init__.py", b"x"), f"{record},,"]
        if record_entry:
            lines.insert(0, rec(self.LIB, b"OLD"))
        with zipfile.ZipFile(whl, "w") as zf:
            zf.writestr(self.LIB, b"OLD")
            zf.writestr("torch/__init__.py", b"x")
            zf.writestr(record, "\n".join(lines) + "\n")
        return whl

    def test_replaces_lib_and_record(self):
        import zipfile

        with tempfile.TemporaryDirectory() as d:
            whl = self._make_wheel(d)
            lib = os.path.join(d, "libtorch_cuda.so")
            with open(lib, "wb") as f:
                f.write(b"NEW" * 64)
            build_stage2.patch_wheel(whl, lib)
            with zipfile.ZipFile(whl) as zf:
                self.assertEqual(zf.read(self.LIB), b"NEW" * 64)
                lines = zf.read("torch-0.0.dist-info/RECORD").decode().splitlines()
                digest, size = build_stage2._wheel_hash_and_size(lib)
                self.assertIn(f"{self.LIB},{digest},{size}", lines)
                # untouched members keep their RECORD entries; no dup names
                self.assertTrue(any(x.startswith("torch/__init__.py,") for x in lines))
                self.assertEqual(len(zf.namelist()), len(set(zf.namelist())))

    def test_missing_record_entry_fails(self):
        with tempfile.TemporaryDirectory() as d:
            whl = self._make_wheel(d, record_entry=False)
            lib = os.path.join(d, "libtorch_cuda.so")
            with open(lib, "wb") as f:
                f.write(b"NEW")
            with self.assertRaisesRegex(RuntimeError, "RECORD has no entry"):
                build_stage2.patch_wheel(whl, lib)


class TestDeclarationStaleness(unittest.TestCase):
    def test_source_closure_includes_the_declaration(self):
        # Declarations load by file path and never enter sys.modules, so
        # without this a kernel_precompile_grid() edit reuses artifacts
        # built from the old grid.
        with tempfile.TemporaryDirectory() as tmpdir:
            decl_path = os.path.join(tmpdir, "aot.py")
            with open(decl_path, "w") as f:
                f.write("ATEN_OP = 'fake'\n")
            closure = export.source_closure(decl_path)
            rel = os.path.relpath(decl_path, export.REPO)
            self.assertIn(rel, closure)
            self.assertEqual(closure[rel], export._file_hash(decl_path))

    def test_source_closure_omits_a_declaration_not_passed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            decl_path = os.path.join(tmpdir, "aot.py")
            with open(decl_path, "w") as f:
                f.write("ATEN_OP = 'fake'\n")
            self.assertNotIn(
                os.path.relpath(decl_path, export.REPO), export.source_closure()
            )


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


class TestShouldRun(unittest.TestCase):  # _no_missing_runtimes helper
    """build_stage2.should_run decides whether the wheel ships kernels, so
    every skip arm needs to be deliberate rather than incidental."""

    def _run(self, probes, env=None, missing=()):
        # probes: expr -> bool, consulted in place of a real torch subprocess.
        #
        # missing_runtimes is patched because should_run checks it BEFORE the
        # arch logic: in an image without the DSL wheels (the Test tools job)
        # every arch case would otherwise raise the missing-runtime error
        # instead of exercising its own branch. The gate itself is covered by
        # test_missing_runtime_is_fatal below.
        with (
            mock.patch.object(
                build_stage2, "_torch_probe", side_effect=lambda e: probes.get(e, True)
            ),
            mock.patch.object(
                toolchains.Toolchain,
                "missing_runtimes",
                classmethod(lambda cls: list(missing)),
            ),
            mock.patch.dict(os.environ, env or {}, clear=False),
        ):
            return build_stage2.should_run()

    def test_missing_runtime_is_fatal(self):
        # The gate the other cases neutralize: a declared kernel that cannot
        # be built must fail the build, not ship a slower wheel.
        with self.assertRaisesRegex(RuntimeError, "runtimes are not installed"):
            self._run(
                {"torch.version.hip is not None": False},
                {"TORCH_CUDA_ARCH_LIST": "10.0a"},
                missing=("cutlass",),
            )

    def test_disabled_by_env(self):
        self.assertFalse(self._run({}, {"TORCH_NATIVE_AOT": "0"}))

    def test_skips_when_torch_not_importable(self):
        self.assertFalse(self._run({"True": False}))

    def test_skips_when_torch_built_without_cuda(self):
        self.assertFalse(self._run({"torch.backends.cuda.is_built()": False}))

    def test_skips_on_rocm_with_no_rocm_toolchain(self):
        # ROCm has no AOT toolchain, so absent DSL wheels are expected there
        # rather than a missing dependency.
        self.assertFalse(
            self._run(
                {"torch.version.hip is not None": True},
                {"TORCH_CUDA_ARCH_LIST": "10.0a"},
            )
        )

    def test_skips_when_arch_list_has_no_exportable_arch(self):
        # 8.0 and 7.5 are below the kernels' floor (TMA, clusters), so nothing
        # to export. 9.0a IS exportable now, hence not in this list.
        self.assertFalse(
            self._run(
                {"torch.version.hip is not None": False},
                {"TORCH_CUDA_ARCH_LIST": "7.5;8.0"},
            )
        )

    def test_multi_exportable_arch_runs(self):
        # Was fatal while nested per-arch artifacts were walked by neither the
        # generator nor the CMake globs (a kernel-less wheel, silently). Now
        # supported end to end: one tree per arch, a per-capability selector,
        # and link globs at both depths.
        self.assertTrue(
            self._run(
                {"torch.version.hip is not None": False},
                {"TORCH_CUDA_ARCH_LIST": "10.0;10.0a"},
            )
        )

    def test_runs_for_a_single_exportable_arch(self):
        self.assertTrue(
            self._run(
                {"torch.version.hip is not None": False},
                {"TORCH_CUDA_ARCH_LIST": "10.0a"},
            )
        )

    def test_skips_without_arch_list_and_without_a_gpu(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            self.assertFalse(
                self._run(
                    {
                        "torch.version.hip is not None": False,
                        "torch.cuda.is_available()": False,
                    }
                )
            )


class TestInt32GateTypeClassifier(unittest.TestCase):
    def test_plain_and_optional_tensors_are_gated(self):
        gate = gen_aot_lib._int32_size_gate(
            "const at::Tensor & self, const ::std::optional<at::Tensor> & weight"
        )
        self.assertIn("self.sizes()", gate)
        self.assertIn("weight.has_value()", gate)

    def test_unhandled_tensor_like_types_are_refused(self):
        # torchgen renders Tensor? as at::OptionalTensorRef and Tensor[] as
        # at::ITensorListRef in structured impl signatures; neither fits the
        # accessors the gate emits, and guessing would either leave a dim
        # ungated or emit C++ that does not compile.
        for params in (
            "const at::Tensor & self, at::OptionalTensorRef bias",
            "const at::ITensorListRef & tensors",
            "const c10::List<::std::optional<at::Tensor>> & indices",
        ):
            with self.subTest(params=params):
                with self.assertRaisesRegex(RuntimeError, "unhandled tensor-like"):
                    gen_aot_lib._int32_size_gate(params)

    def test_scalar_only_signature_emits_nothing(self):
        self.assertEqual(gen_aot_lib._int32_size_gate("int64_t k, bool largest"), "")


class TestGeneratedVersionScript(unittest.TestCase):
    def test_patterns_are_anchored_on_the_kernel_prefix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = gen_aot_lib.write_version_script(tmpdir, ["topk_f32_n1024_k8"])
            with open(path) as f:
                text = f.read()
        # <prefix>_* covers every symbol the DSL emits for the kernel;
        # enumerating known suffixes left _args_spec/_kernel_info/
        # _function_name/_version in torch_cuda's ABI.
        self.assertIn("topk_f32_n1024_k8_*;", text)
        self.assertIn("_mlir_*topk_f32_n1024_k8*;", text)
        self.assertIn("local:", text)
        # EVERY pattern names the prefix. Asserting the absence of one
        # hand-picked unanchored pattern ("*_cuda_init;") proved nothing -- it
        # was never a candidate. This fails for any pattern that would reach
        # past this kernel's symbols and quietly drop something unrelated out
        # of torch_cuda's ABI. Indented and ";"-terminated selects the pattern
        # lines, not the script's own closing "};" or its comment block.
        patterns = [
            l.strip()
            for l in text.splitlines()
            if l.startswith("    ") and l.strip().endswith(";")
        ]
        self.assertTrue(patterns)
        for p in patterns:
            self.assertIn("topk_f32_n1024_k8", p, f"unanchored pattern: {p}")


class TestEmbeddedSizeReport(unittest.TestCase):
    """Embedded kernel bytes scale with declarations x precompile points x
    arches, so an arch added to TORCH_CUDA_ARCH_LIST can grow the wheel by
    tens of MiB. Nothing else in the build log states it."""

    def test_counts_linked_artifacts_and_skips_headers(self):
        # Fixture built from the exts registered in THIS tree rather than a
        # fixed list: which toolchains exist changes along the stack, so a
        # hardcoded ".cubin" would only hold once Triton has landed.
        linkable = sorted(toolchains.all_artifact_exts() - {".h"})
        mib = 1 << 20
        with tempfile.TemporaryDirectory() as tmpdir:
            d = os.path.join(tmpdir, "sm_100a", "fakeop")
            os.makedirs(d)
            for i, e in enumerate(linkable):
                with open(os.path.join(d, f"k{i}{e}"), "wb") as f:
                    f.truncate(mib)  # sparse: only the reported size matters
            # The ABI header and the sidecar feed the compiler and the
            # generator; neither reaches the shipped library. Deliberately the
            # biggest file here, so counting it would move the MiB figure.
            with open(os.path.join(d, "k.h"), "wb") as f:
                f.truncate(8 * mib)
            open(os.path.join(d, "k.json"), "w").close()
            report = build_stage2._artifact_size(tmpdir)
        self.assertEqual(report, f"{len(linkable)} object(s), {len(linkable)}.0 MiB")

    def test_artifact_exts_are_shared_by_both_sweeps(self):
        # One notion of "kernel artifact": export's per-directory orphan check
        # and generation's no-declaration check both ask this, so a new
        # toolchain cannot be visible to one and invisible to the other.
        exts = toolchains.all_artifact_exts()
        for tc in toolchains.TOOLCHAINS.values():
            for e in tc.artifact_exts:
                self.assertIn(e, exts)


class TestParamListSplitting(unittest.TestCase):
    """Both the size gate and the covers device read pick parameters out of a
    rendered C++ signature, so the split has to survive a template argument
    list that contains a comma."""

    def test_top_level_split_keeps_template_args_intact(self):
        params = (
            "const at::Tensor & self, std::array<bool, 3> mask, "
            "const std::optional<at::Tensor>& out"
        )
        self.assertEqual(
            gen_aot_lib._split_params(params),
            [
                "const at::Tensor & self",
                "std::array<bool, 3> mask",
                "const std::optional<at::Tensor>& out",
            ],
        )

    def test_gate_sees_a_comma_bearing_type_whole(self):
        # The refusal below exists to stop a guess at an unknown tensor-shaped
        # type. It can only be honest if it is handed the WHOLE type: split on
        # every comma, the classifier judges the fragment "at::Tensor>" and
        # names that in the error, sending the next author after a type that
        # does not appear in their signature.
        with self.assertRaisesRegex(RuntimeError, r"std::pair<at::Tensor, at::Tensor>"):
            gen_aot_lib._int32_size_gate(
                "const at::Tensor & self, std::pair<at::Tensor, at::Tensor> pr"
            )

    def test_non_tensor_comma_bearing_type_is_not_gated(self):
        gate = gen_aot_lib._int32_size_gate(
            "const at::Tensor & self, std::array<bool, 3> mask, "
            "const std::optional<at::Tensor>& out"
        )
        self.assertIn("self.sizes().begin()", gate)
        self.assertIn("out.has_value()", gate)
        self.assertNotIn("mask", gate)


class TestSidecarFieldValidation(unittest.TestCase):
    """Generation reads these fields straight into emitted C++, so a value it
    cannot use must fail here rather than as a compiler error in a @generated
    file, or as a wrongly-read field."""

    def _run(self, tmpdir, sc, extra_argv=()):
        d = os.path.join(tmpdir, "sm_100a", "fakeop")
        os.makedirs(d)
        for e in (".o", ".h"):
            open(os.path.join(d, "k__sm100a" + e), "w").close()
        with open(os.path.join(d, "k__sm100a.json"), "w") as f:
            json.dump(sc, f)
        opsdir = os.path.join(tmpdir, "_ops")
        os.makedirs(os.path.join(opsdir, "fakeop"))
        open(os.path.join(opsdir, "fakeop", "aot.py"), "w").close()
        with (
            mock.patch.object(gen_aot_lib, "OPS_DIR", opsdir),
            mock.patch.object(
                gen_aot_lib.decl, "load_declarations", return_value=[_FakeDecl]
            ),
        ):
            gen_aot_lib.main(["--artifacts-dir", tmpdir, *extra_argv])

    def test_prefix_must_be_a_c_identifier(self):
        # The prefix names extern "C" entry points and launch_<prefix>.
        with tempfile.TemporaryDirectory() as tmpdir:
            sc = {
                "version": export.SIDECAR_VERSION,
                "prefix": "k-sm100a",
                "kind": "cutedsl",
                "arch": "sm_100a",
                "spec": {"N": 1},
                "tensor_args": [],
            }
            with self.assertRaisesRegex(RuntimeError, "not a C identifier"):
                self._run(tmpdir, sc)

    def test_schema_version_mismatch_is_not_waivable(self):
        # --allow-stale exists for artifacts whose SOURCES drifted; those still
        # describe themselves in a shape this generator reads. A schema bump
        # may not, so forcing past it would emit from misread fields.
        for argv in ((), ("--allow-stale",)):
            with tempfile.TemporaryDirectory() as tmpdir:
                sc = {
                    "version": export.SIDECAR_VERSION + 1,
                    "prefix": "k__sm100a",
                    "kind": "cutedsl",
                    "arch": "sm_100a",
                    "spec": {"N": 1},
                    "tensor_args": [],
                }
                with self.assertRaisesRegex(RuntimeError, "sidecar schema version"):
                    self._run(tmpdir, sc, argv)

    def test_stale_error_names_the_arches_and_a_command_that_fixes_them(self):
        # A bare `export.py` re-run maintains only the arch it resolves for, so
        # it leaves other arch trees stale forever: the message has to name the
        # arches and the --arch invocation, not just say "re-run export".
        with tempfile.TemporaryDirectory() as tmpdir:
            sc = {
                "version": export.SIDECAR_VERSION,
                "prefix": "k__sm100a",
                "kind": "cutedsl",
                "arch": "sm_100a",
                "spec": {"N": 1},
                "tensor_args": [],
                "sources": {"tools/native_aot/decl.py": "0" * 16},
            }
            with self.assertRaisesRegex(RuntimeError, r"--arch sm_100a"):
                self._run(tmpdir, sc)


class TestSizeGateIsPerToolchain(unittest.TestCase):
    """The i32 size gate belongs to kinds whose exported ABI takes i32
    extents (CuTeDSL), not to every AOT op."""

    class _Decl:
        ATEN_OP = "fakeop"
        DISPATCH_KEY = "CUDA"
        ARCHS = ("sm_100a",)

        def cpp_dispatch(self, spec):
            return "true"

        def cpp_launch(self, spec, fn):
            return f"{fn}(self, k, at::cuda::getCurrentCUDAStream());"

    _SC = {
        "prefix": "p",
        "spec": {"N": 1},
        "tensor_args": [{"name": "self", "dynamic_sizes": [0], "dynamic_strides": [0]}],
        "arch": "sm_100a",
    }

    def _gen(self, kind):
        return gen_aot_lib.gen_op(
            "fakeop",
            "CUDA",
            self._Decl(),
            [dict(self._SC, kind=kind)],
            "const at::Tensor & self, int64_t k",
        )

    def test_narrowing_kind_gets_gate_and_helper(self):
        src = self._gen("cutedsl")
        self.assertIn("// Size gate:", src)
        self.assertIn("inline bool _naot_dim_too_big", src)

    def test_non_narrowing_kind_gets_neither(self):
        # An unused inline helper would otherwise sit in every generated
        # file, and the gate would decline dims the kernel could serve.
        class _Wide(toolchains.CuteDslToolchain):
            kind = "wide"
            NARROWS_SHAPES_TO_INT32 = False

        with mock.patch.dict(toolchains.TOOLCHAINS, {"wide": _Wide()}, clear=False):
            src = self._gen("wide")
        self.assertNotIn("// Size gate:", src)
        self.assertNotIn("_naot_dim_too_big", src)


class TestSelectiveIncludes(unittest.TestCase):
    """torch/library.h pulls the dispatcher in (~110 transitive headers), and
    only the cpp_covers registration needs it."""

    _SC = {
        "prefix": "p",
        "spec": {"N": 1024, "K": 8},  # _FakeDecl.cpp_dispatch reads both
        "kind": "cutedsl",
        "arch": "sm_100a",
        "tensor_args": [{"name": "self", "dynamic_sizes": [0], "dynamic_strides": [0]}],
    }
    _COVERS = (
        "const at::Tensor & self",
        "fakeop(Tensor self) -> bool",
        "return true;",
    )

    def _gen(self, covers):
        return gen_aot_lib.gen_op(
            "fakeop",
            "CUDA",
            _FakeDecl,
            [self._SC],
            "const at::Tensor & self",
            covers,
        )

    def test_library_header_omitted_without_covers(self):
        src = self._gen(None)
        self.assertNotIn("#include <torch/library.h>", src)
        self.assertFalse(
            any(l.startswith("TORCH_LIBRARY_FRAGMENT") for l in src.splitlines())
        )

    def test_library_header_present_with_covers(self):
        src = self._gen(self._COVERS)
        self.assertIn("#include <torch/library.h>", src)
        self.assertTrue(
            any(l.startswith("TORCH_LIBRARY_FRAGMENT") for l in src.splitlines())
        )

    def test_always_needed_headers_are_unconditional(self):
        # These are used by every generated file (stub registration, launcher
        # stream narrowing, the arch gate's device query).
        src = self._gen(None)
        for h in (
            "<ATen/core/Tensor.h>",
            "<ATen/NativeAotStubs.h>",
            "<ATen/cuda/CUDAContext.h>",
            "<c10/cuda/CUDAStream.h>",
        ):
            with self.subTest(header=h):
                self.assertIn(f"#include {h}", src)


class TestOrphanArtifactSafety(unittest.TestCase):
    """Generation must never destroy exported kernels: a .cpp costs nothing to
    regenerate, a .o costs a full export."""

    def _art(self, tmpdir):
        """The one layout: kernels at <root>/<arch>/<decl_id>/, and the
        generated source that covers every arch at <root>/<decl_id>/.
        Returns both, since orphan handling touches each differently -- the
        .cpp is regenerable and gets deleted, the kernels never do."""
        op = os.path.join(tmpdir, "sm_100a", "fakeop")
        os.makedirs(op)
        for fn in ("k.o", "k.h", "k.json"):
            with open(os.path.join(op, fn), "w") as f:
                f.write("x")
        src = os.path.join(tmpdir, "fakeop")
        os.makedirs(src)
        with open(os.path.join(src, "aot_fakeop_cuda.cpp"), "w") as f:
            f.write("x")
        return op, src

    def test_same_prefix_from_two_arch_dirs_is_fatal(self):
        # Prefixes carry their arch, so two arch dirs cannot collide by
        # construction -- but a copied or renamed tree (rsynced artifacts, a
        # hand-made sm_100a.bak) puts the same prefix under two arch dirs, and
        # discovery merges by declaration. Caught here rather than as a
        # launch_<prefix> redefinition, or the same .o globbed twice, much
        # later in the build.
        with tempfile.TemporaryDirectory() as tmpdir:
            for rel in (
                os.path.join("sm_100a", "fakeop"),
                os.path.join("sm_100a_copy", "fakeop"),
            ):
                d = os.path.join(tmpdir, rel)
                os.makedirs(d)
                for fn in ("k__sm100a.o", "k__sm100a.h"):
                    with open(os.path.join(d, fn), "w") as f:
                        f.write("x")
                with open(os.path.join(d, "k__sm100a.json"), "w") as f:
                    json.dump(
                        {
                            "version": export.SIDECAR_VERSION,
                            "prefix": "k__sm100a",
                            "arch": "sm_100a",
                        },
                        f,
                    )
            # The duplicate check runs after a declaration is matched, so
            # fakeop has to BE declared or discovery takes the orphan path
            # first. gen_aot_lib imports export inside main(), so patch the
            # module itself rather than an attribute on gen_aot_lib.
            opsdir = os.path.join(tmpdir, "_ops")
            os.makedirs(os.path.join(opsdir, "fakeop"))
            open(os.path.join(opsdir, "fakeop", "aot.py"), "w").close()
            with (
                mock.patch.object(gen_aot_lib, "OPS_DIR", opsdir),
                mock.patch.object(
                    gen_aot_lib.decl, "load_declarations", return_value=[_FakeDecl]
                ),
                mock.patch.object(export, "sources_current", return_value=True),
            ):
                with self.assertRaisesRegex(RuntimeError, "present in both"):
                    gen_aot_lib.main(["--artifacts-dir", tmpdir])

    def test_no_declarations_at_all_leaves_everything_alone(self):
        # A commit earlier in the stack (or a bisect) declares nothing; that
        # is not the same as every artifact being orphaned.
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            tempfile.TemporaryDirectory() as opsdir,
        ):
            op, src = self._art(tmpdir)
            # An EMPTY ops dir: "declares nothing" must not depend on which
            # commit of the stack is checked out.
            with mock.patch.object(gen_aot_lib, "OPS_DIR", opsdir):
                gen_aot_lib.main(["--artifacts-dir", tmpdir])
            left = sorted(os.listdir(op))
            src_left = sorted(os.listdir(src))
        self.assertIn("k.o", left)
        self.assertIn("k.h", left)
        self.assertIn("aot_fakeop_cuda.cpp", src_left)

    def test_orphan_with_other_declarations_is_fatal_but_keeps_artifacts(self):
        # With declarations present, an unclaimed dir is a real orphan: drop
        # the generated .cpp so it cannot reference a vanished stub, but raise
        # rather than delete the objects the CMake glob would link.
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            tempfile.TemporaryDirectory() as opsdir,
        ):
            op, src = self._art(tmpdir)
            # by_id is built by walking OPS_DIR, so declare something there.
            os.makedirs(os.path.join(opsdir, "otherop"))
            open(os.path.join(opsdir, "otherop", "aot.py"), "w").close()
            with (
                mock.patch.object(gen_aot_lib, "OPS_DIR", opsdir),
                mock.patch.object(
                    gen_aot_lib.decl, "load_declarations", return_value=[_OtherDecl]
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "no declaration"):
                    gen_aot_lib.main(["--artifacts-dir", tmpdir])
            left = sorted(os.listdir(op))
            src_left = sorted(os.listdir(src))
        self.assertIn("k.o", left)
        # The regenerable source is dropped from <root>/<decl_id>/, where it
        # lives now, so the CMake glob cannot compile it against a stub that
        # no longer exists.
        self.assertNotIn("aot_fakeop_cuda.cpp", src_left)


class _OtherDecl(_FakeDecl):
    """A declaration whose decl_id does not match the artifact dir above."""

    ATEN_OP = "otherop"


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
            with _no_device_arch():
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
                        "kind": "cutedsl",
                        "spec": point,
                        "sources": current,
                    },
                    f,
                )
            _touch_artifacts(d, "x", exts=(".o",))  # .h missing
            self.assertTrue(export._job_needed(job, force=False))


if __name__ == "__main__":
    unittest.main()
