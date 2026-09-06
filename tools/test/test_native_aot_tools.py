from __future__ import annotations

import contextlib
import glob
import importlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import types
import unittest
import unittest.mock as mock
import zipfile
from typing import Any

# Ordinary package imports: these modules keep their scope torch-free so the
# Test tools job, which runs in the linter image, can import them without torch.
from tools.native_aot import build_stage2, export, gen_aot_lib, toolchains

from torchgen import native_aot_decl


_TOOLS_FILE = os.path.abspath(toolchains.__file__)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(_TOOLS_FILE)))


# The DSL versions this environment would compile with; sidecars must carry
# them or the skip check treats them as built by a different compiler.
_RUNTIMES = export.runtime_versions("cutedsl")

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


def _no_ambient_arch(device=None):
    """Take the ambient environment out of arch resolution.

    ``device`` is what _detected_arch should report: None for "no GPU", or an sm string
    to exercise the device fallback.

    _effective_arch resolves an unspecified arch from the local GPU or from a
    toolchain's ARCH_ENV_VAR, so tests about spec/source matching must patch both or
    depend on the runner's hardware and shell."""
    env = {
        tc.ARCH_ENV_VAR: "" for tc in toolchains.TOOLCHAINS.values() if tc.ARCH_ENV_VAR
    }
    stack = contextlib.ExitStack()
    stack.enter_context(
        mock.patch.object(export, "_detected_arch", return_value=device)
    )
    # patch.dict cannot unset, and os.getenv("") is falsy, which is what
    # _effective_arch tests -- so an empty value is equivalent to absent here.
    stack.enter_context(mock.patch.dict(os.environ, env, clear=False))
    return stack


def _manifest(artifacts_dir):
    """What the generator told CMake, parsed back: {"arch_list": str|None,
    "sources": [...], "objects": [...]}.

    The generator emits native_aot.cmake -- the CMake CALLS, not data -- so this
    reads the quoted paths out of its target_sources() and the arch list out of its
    header comment. Tests assert the same invariants as when it was a manifest
    file; only the artifact changed."""
    out = {"arch_list": None, "sources": [], "objects": []}
    path = os.path.join(artifacts_dir, gen_aot_lib.CMAKE_INCLUDE)
    with open(path) as f:
        text = f.read()
    if "TORCH_CUDA_ARCH_LIST=" in text:
        m = re.search(r"TORCH_CUDA_ARCH_LIST='([^']*)'", text)
        if m is None:
            raise AssertionError(f"{path}: arch list present but unquoted:\n{text}")
        out["arch_list"] = m.group(1)
    elif "without an arch list" in text:
        # The generator makes no claim -- a hand run or an on-device export. A
        # DISTINCT value from None, which means "the file said nothing at all".
        out["arch_list"] = "absent"
    block = text.partition("target_sources(torch_cuda PRIVATE")[2].partition(")")[0]
    for line in block.splitlines():
        line = line.strip()
        if not (line.startswith('"') and line.endswith('"')):
            continue
        raw = line[1:-1]
        for char in (";", "$", '"', "\\"):
            raw = raw.replace("\\" + char, char)
        (out["objects"] if raw.endswith(".o") else out["sources"]).append(raw)
    return out


_DECL_REL = os.path.relpath(
    os.path.join(os.path.dirname(__file__), "..", "native_aot", "decl.py"), export.REPO
)


def _current_sources():
    """A source closure that matches this tree, for sidecars meant to look
    current."""
    return {_DECL_REL: export._file_hash(os.path.join(export.REPO, _DECL_REL))}


def _write_sidecar(d, point, prefix="x", exts=(".o", ".h"), **over):
    """Write into ``d`` the artifacts and the sidecar a current export would have
    left for ``point``, so _job_needed sees a skippable job. ``over`` replaces or
    adds fields (arch, sources, ...) for the cases that are about one of them."""
    _touch_artifacts(d, prefix, exts=exts)
    sc = {
        "version": export.SIDECAR_VERSION,
        "prefix": prefix,
        "kind": "cutedsl",
        "spec": point,
        "sources": _current_sources(),
        "runtimes": _RUNTIMES,
    }
    sc.update(over)
    with open(os.path.join(d, prefix + ".json"), "w") as f:
        json.dump(sc, f)


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


def _touch_artifacts(out_dir, prefix, exts=(".o", ".h"), tensor_args=None):
    """Create the files a sidecar claims. _job_needed verifies they exist, so a fixture
    writing only the .json would always re-export.

    The .h is derived from the sidecar's own tensor_args -- one struct per tensor, each
    array bound equal to the dims that tensor claims -- because validate_abi requires
    that equality. Hand-picked bounds would describe an ABI no export could produce."""
    args = SIDECAR["tensor_args"] if tensor_args is None else tensor_args
    for e in exts:
        body = ""
        if e == ".h":
            body = "#pragma once\n#include <stdint.h>\n"
            for a in args:
                fields = ["  void* data;"]
                if a.get("dynamic_sizes"):
                    fields.append(
                        f"  int32_t dynamic_shapes[{len(a['dynamic_sizes'])}];"
                    )
                if a.get("dynamic_strides"):
                    fields.append(
                        f"  int64_t dynamic_strides[{len(a['dynamic_strides'])}];"
                    )
                body += (
                    "typedef struct {\n"
                    + "\n".join(fields)
                    + f"\n}} {prefix}_Tensor_{a['name']}_t;\n"
                )
        with open(os.path.join(out_dir, prefix + e), "w") as f:
            f.write(body)


class TestExportJobs(unittest.TestCase):
    def test_job_skip_matches_on_spec(self):
        # A spec match alone re-exports: the source closure must match too.
        with tempfile.TemporaryDirectory() as d:
            point = {"dtype": "float32", "N": 4096}
            job = ("fakeop", "aot_kernel.py", point, d, None)
            self.assertTrue(export._job_needed(job, force=False))
            _write_sidecar(d, point)
            with _no_ambient_arch():
                self.assertFalse(export._job_needed(job, force=False))
                self.assertTrue(export._job_needed(job, force=True))
                other = ("fakeop", "aot_kernel.py", {"dtype": "bfloat16"}, d, None)
                self.assertTrue(export._job_needed(other, force=False))

    def test_job_skip_survives_json_round_trip(self):
        # Tuple-valued grid fields read back as lists, so both sides must be
        # normalized or such points re-export on every run.
        with tempfile.TemporaryDirectory() as d:
            point = {"aten": "add.Tensor", "in_dtypes": ("float32", "bfloat16")}
            job = ("fakeop", "aot_kernel.py", point, d, None)
            _write_sidecar(d, point, spec=export._json_normal(point))
            with _no_ambient_arch():
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
        from typing import cast

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
        # A declaration targeting this backend was asked for, so a missing runtime must
        # fail rather than ship fewer kernels than declared.
        reached = []
        with (
            mock.patch.object(
                toolchains.CuteDslToolchain,
                "missing_runtimes",
                classmethod(lambda cls: []),
            ),
            mock.patch.object(
                toolchains.CuteDslToolchain,
                "export",
                lambda self, b, out_dir, arch=None: reached.append(arch) or {},
            ),
            _no_ambient_arch(device="sm_100a"),
        ):
            b = {"prefix": "p", "fn": None, "fake_args": (), "tensor_args": []}
            with mock.patch.object(export, "load_builder", lambda *a: lambda p: b):
                with tempfile.TemporaryDirectory() as d:
                    export.export_point("fakeop", "aot_kernel.py", {}, d)
        # Runtime present: it got PAST the gate and into the toolchain's export.
        self.assertEqual(len(reached), 1)

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
    def test_effective_arch_cannot_vary_by_kind(self):
        # Why the resolver takes no toolchain: with no --arch, resolution is device
        # detection, and a per-kind answer lets a tree disagree with its sidecars.
        self.assertNotIn("tc", export._effective_arch.__code__.co_varnames)
        with _no_ambient_arch(device="sm_100"):
            self.assertEqual(export._effective_arch(None), "sm_100")
        # An explicit arch wins, whatever a kind's variable says.
        with mock.patch.dict(os.environ, {"CUTE_DSL_ARCH": "sm_90a"}):
            self.assertEqual(export._effective_arch("sm_100a"), "sm_100a")

    def test_arch_tag_is_short(self):
        # The tag lands in every exported C symbol, so its shape is part of
        # the artifact ABI: one underscore dropped, nothing else.
        self.assertEqual(export._arch_tag("sm_100a"), "sm100a")
        self.assertEqual(export._arch_tag("sm_90"), "sm90")

    def test_an_arch_env_var_without_explicit_arch_is_refused(self):
        # An arch variable is per kind, so honouring it would name a tree after one
        # kind's arch while holding another kind's sidecars.
        class _Other(toolchains.Toolchain):
            kind = "other"
            ARCH_ENV_VAR = "OTHER_DSL_ARCH"

        registry = dict(toolchains.TOOLCHAINS, other=_Other())
        with (
            mock.patch.dict(toolchains.TOOLCHAINS, registry, clear=True),
            mock.patch.dict(
                os.environ, {"CUTE_DSL_ARCH": "sm_90a", "OTHER_DSL_ARCH": "sm_100a"}
            ),
            mock.patch.object(export, "_detected_arch", return_value=None),
        ):
            # Named in the message so the user knows WHICH variable to unset, and
            # both are named when both are set.
            with self.assertRaisesRegex(RuntimeError, "CUTE_DSL_ARCH=sm_90a"):
                export._effective_arch(None)
            with self.assertRaisesRegex(RuntimeError, "OTHER_DSL_ARCH=sm_100a"):
                export._effective_arch(None)
            # Agreeing values are refused too: the point is not a conflict
            # between them, it is that neither can speak for the other kinds.
            with mock.patch.dict(os.environ, {"OTHER_DSL_ARCH": "sm_90a"}):
                with self.assertRaisesRegex(RuntimeError, "--arch"):
                    export._effective_arch(None)
            # ...and through the overload the production callers use, since
            # export_point and _job_needed both pass a toolchain.
            for tc in (toolchains.get_toolchain("cutedsl"), _Other()):
                with self.subTest(kind=tc.kind):
                    with self.assertRaisesRegex(RuntimeError, "CUTE_DSL_ARCH=sm_90a"):
                        export._effective_arch(None)
            # An explicit --arch is the way to say it, for every kind at once.
            self.assertEqual(export._effective_arch("sm_100a"), "sm_100a")

    def test_export_prefix_is_arch_qualified(self):
        # Every exported symbol derives from the prefix, so two arches sharing one
        # are duplicate definitions once both link into libtorch_cuda.
        seen = []

        class _FakeTc(toolchains.Toolchain):
            kind = "cutedsl"
            artifact_exts = (".o", ".h")
            # So the sidecar's recorded compiler versions are non-empty below.
            RUNTIME_DISTS = ("nvidia-cutlass-dsl",)

            def missing_runtimes(self):
                return []

            def validate_build_result(self, b):
                pass

            def export(self, b, out_dir, arch=None):
                # The pair, not the prefix alone: the prefix names an arch, and this
                # is the arch the kernel was really compiled for.
                seen.append((b["prefix"], arch))
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
                want = [("k__sm90a", "sm_90a"), ("k__sm100a", "sm_100a")]
                self.assertEqual(seen, want)
                # The sidecar WRITER against what the readers require; every other
                # test runs on hand-written fixtures.
                with open(os.path.join(d, "k__sm100a.json")) as f:
                    written = json.load(f)
                self.assertEqual(written["version"], export.SIDECAR_VERSION)
                self.assertEqual(written["arch"], "sm_100a")
                self.assertEqual(written["kind"], "cutedsl")
                self.assertEqual(written["prefix"], "k__sm100a")
                self.assertEqual(written["spec"], {"n": 1})
                self.assertTrue(written["sources"], "no source closure recorded")
                self.assertEqual(
                    written["runtimes"], export.runtime_versions("cutedsl")
                )
                # ...and what it wrote is what the readers accept.
                self.assertTrue(export.sources_current(written))
                self.assertTrue(export.runtimes_current(written))

                # The on-device path: recording the raw argument would leave
                # "arch": null behind an unqualified prefix, which generation refuses.
                with _no_ambient_arch(device="sm_100"):
                    export.export_point("fakeop", "aot_kernel.py", {"n": 2}, d, None)
                with open(os.path.join(d, "k__sm100.json")) as f:
                    on_device = json.load(f)
                self.assertEqual(on_device["prefix"], "k__sm100")
                self.assertEqual(on_device["arch"], "sm_100")
                # The compile is still told the RAW arch: None lets the DSL take the
                # local device, which is what was just detected.
                self.assertEqual(seen[-1], ("k__sm100", None))

    def test_job_skip_compares_the_arch(self):
        # Two exports into one --out-dir differing only in arch: comparing spec alone
        # skips the second and leaves the first arch's objects behind its sidecar.
        with tempfile.TemporaryDirectory() as d:
            point = {"dtype": "float32", "N": 4096}
            job = ("fakeop", "aot_kernel.py", point, d, None)
            _write_sidecar(d, point, arch="sm_90a")
            # _collect_jobs resolves the arch once and puts it in the job, so the
            # comparison is job-vs-sidecar and needs no ambient variable.
            with _no_ambient_arch():
                same = ("fakeop", "aot_kernel.py", point, d, "sm_90a")
                other = ("fakeop", "aot_kernel.py", point, d, "sm_100a")
                self.assertFalse(export._job_needed(same, force=False))
                self.assertTrue(export._job_needed(other, force=False))
                # An on-device job (arch resolved from the device) is not the
                # sm_90a one either.
                with mock.patch.object(export, "_detected_arch", return_value="sm_100"):
                    self.assertTrue(export._job_needed(job, force=False))

    def test_job_skip_re_exports_when_the_compiler_changed(self):
        # The compiler appears in no file the closure hashes, so without this an
        # upgraded DSL wheel invalidates nothing. runtime_versions is patched: with no
        # wheels installed every version is "absent", which is not staleness.
        with tempfile.TemporaryDirectory() as d:
            point = {"dtype": "float32", "N": 4096}
            job = ("fakeop", "aot_kernel.py", point, d, "sm_100a")
            with (
                _no_ambient_arch(),
                mock.patch.object(
                    export, "runtime_versions", return_value={"a-dsl": "9.9.9"}
                ),
            ):
                _write_sidecar(d, point, arch="sm_100a", runtimes={"a-dsl": "9.9.9"})
                self.assertFalse(export._job_needed(job, force=False))
                _write_sidecar(d, point, arch="sm_100a", runtimes={"a-dsl": "0.0.1"})
                self.assertTrue(export._job_needed(job, force=False))

    def test_job_skip_rejects_arch_less_sidecar_when_env_set(self):
        # A sidecar recording no arch must not satisfy a run whose CUTE_DSL_ARCH
        # names one, or that run inherits objects built for the builder's own GPU.
        with tempfile.TemporaryDirectory() as d:
            point = {"dtype": "float32", "N": 4096}
            job = ("fakeop", "aot_kernel.py", point, d, None)
            _write_sidecar(d, point, arch=None)
            with _no_ambient_arch():
                named = ("fakeop", "aot_kernel.py", point, d, "sm_100a")
                self.assertTrue(export._job_needed(named, force=False))
                # ...nor an on-device run, which knows its arch where the sidecar
                # names none. _detected_arch is patched, not trusted.
                with mock.patch.object(export, "_detected_arch", return_value="sm_100"):
                    self.assertTrue(export._job_needed(job, force=False))
                # Only where no arch can be resolved at all is it a match.
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

    def test_the_detected_arch_is_the_local_capability(self):
        # Without this the gate falls back to ARCHS and advertises hardware nothing was
        # compiled for. No "a" suffix: the gate compares major.minor.
        fake = types.SimpleNamespace(
            cuda=types.SimpleNamespace(
                is_available=lambda: True, get_device_capability=lambda: (10, 3)
            )
        )
        with mock.patch.dict(sys.modules, {"torch": fake}):
            self.assertEqual(export._detected_arch(), "sm_103")
        # No CUDA is "no local arch", not a crash in the middle of a build.
        fake.cuda.is_available = lambda: False
        with mock.patch.dict(sys.modules, {"torch": fake}):
            self.assertIsNone(export._detected_arch())

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
        # "10.0;10.0+PTX" names one arch twice; a repeated entry would read as
        # multi-arch downstream and export a second full set of kernels.
        f = export.archs_from_cuda_arch_list
        self.assertEqual(f("10.0;10.0+PTX"), ["sm_100"])
        self.assertEqual(f("10.0a 10.0a"), ["sm_100a"])

    def test_the_shipped_arches_are_ones_the_tooling_can_target(self):
        # The two sets live beside each other so they cannot drift; the import-time
        # check is what makes that true rather than merely intended.
        self.assertLessEqual(
            set(native_aot_decl.EXPORTABLE_ARCHES), set(native_aot_decl.KNOWN_ARCHES)
        )
        # ...and the exporter's name is the same object, not a copy that could age.
        self.assertIs(export.EXPORTABLE_ARCHES, native_aot_decl.EXPORTABLE_ARCHES)

    def test_archs_from_cuda_arch_list_collapses_one_capability(self):
        # Both spellings are the same hardware and generation uses one, so exporting
        # both compiles a second full set of kernels no launcher references.
        f = export.archs_from_cuda_arch_list
        self.assertEqual(f("10.0;10.0a"), ["sm_100a"])
        self.assertEqual(f("10.0a;10.0"), ["sm_100a"])
        # Different capabilities are untouched, and order is preserved.
        self.assertEqual(f("9.0a;10.0;10.0a"), ["sm_90a", "sm_100a"])
        self.assertEqual(f("10.0a;9.0"), ["sm_100a", "sm_90"])

    def test_collect_jobs_respects_declaration_archs(self):
        # A declaration pinning ARCHS gets no jobs for other arches; an
        # on-device export (arch None) is never filtered.
        with tempfile.TemporaryDirectory() as ops, tempfile.TemporaryDirectory() as out:
            _write_fake_decl(ops, 'ARCHS = ("sm_100a",)\n')
            # _detected_arch patched: the on-device call below resolves its
            # directory from it, and this suite runs where there is no GPU.
            with (
                mock.patch.object(export, "OPS_DIR", ops),
                _no_ambient_arch(device="sm_100a"),
            ):
                blackwell = export._collect_jobs(None, out, ["sm_100a"])
                hopper = export._collect_jobs(None, out, ["sm_90a"])
                on_device = export._collect_jobs(None, out, [None])
        self.assertEqual(len(blackwell), 1)
        self.assertEqual(len(hopper), 0)
        self.assertEqual(len(on_device), 1)

    def test_multi_arch_jobs_nest_per_arch(self):
        # Every job nests under <out>/<arch>/<decl_id>, one arch or several: there is
        # no second, flat layout, which the single-arch and on-device cases pin too.
        with tempfile.TemporaryDirectory() as ops, tempfile.TemporaryDirectory() as out:
            _write_fake_decl(ops)
            # _detected_arch patched, so the layout claim does not depend on the
            # machine having a GPU.
            with (
                mock.patch.object(export, "OPS_DIR", ops),
                # Device-only resolution: an ambient CUTE_DSL_ARCH would outrank
                # it and put the on-device job in a different arch directory.
                _no_ambient_arch(device="sm_100"),
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
        # sm_100a, not the detected sm_100: an on-device export adopts the spelling the
        # DECLARATION claims, so the tree cannot be one it disowns.
        self.assertEqual(os.path.basename(os.path.dirname(sj[3])), "sm_100a")
        self.assertEqual(sj[4], "sm_100a")

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
        # Both are valid on 10.0 hardware, and the conditional build is what the
        # kernels were written against; otherwise directory order would decide.
        for order in (
            [("sm_100", "p"), ("sm_100a", "c")],
            [("sm_100a", "c"), ("sm_100", "p")],
        ):
            scs = [{"prefix": n, "arch": a} for a, n in order]
            groups = gen_aot_lib._by_arch(scs)
            self.assertEqual(list(groups), [(10, 0)])
            self.assertEqual([s["prefix"] for s in groups[(10, 0)]], ["c"])

    def test_by_arch_rejects_an_arch_less_sidecar(self):
        # Export names the arch of everything it writes, so this is an older tree.
        # Rejected rather than grouped: an unmatchable capability declines silently.
        with self.assertRaisesRegex(RuntimeError, "records no arch"):
            gen_aot_lib._by_arch([{"prefix": "old", "arch": None}])

    def test_dropped_tie_break_candidate_gets_no_launcher(self):
        # The plain build loses the tie-break, and a launcher emitted for it would be
        # defined and never called: -Wunused-function, fatal under CI's WERROR.
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

    def test_device_match_renders_the_full_capability(self):
        # cc_of itself is covered above, against native_aot_decl directly.
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
            with contextlib.redirect_stdout(io.StringIO()) as out:
                export._check_no_orphan_artifacts(d, [])
        self.assertEqual(out.getvalue(), "", "a clean directory says nothing")

    def test_artifacts_without_a_sidecar_are_reported_not_fatal(self):
        # An export that died before writing the sidecar. Generation names artifacts
        # from sidecars, so an undescribed orphan costs disk, not payload.
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "k_f32.o"), "w").close()
            with contextlib.redirect_stdout(io.StringIO()) as out:
                export._check_no_orphan_artifacts(d, [])
        self.assertIn("no sidecar claims", out.getvalue())
        self.assertIn("k_f32.o", out.getvalue())

    def test_an_orphan_beside_a_committed_point_is_reported_not_fatal(self):
        # Where an interrupt lands: among points that already committed. Per directory
        # this goes unseen; per artifact and fatal it needs a hand-delete.
        with tempfile.TemporaryDirectory() as d:
            for name in ("k_n1.o", "k_n1.h", "k_n2.h"):
                open(os.path.join(d, name), "w").close()
            with open(os.path.join(d, "k_n1.json"), "w") as f:
                # version, so the stale half runs rather than short-circuiting.
                json.dump(
                    {
                        "version": export.SIDECAR_VERSION,
                        "prefix": "k_n1",
                        "kind": "cutedsl",
                        "spec": {"N": 1},
                    },
                    f,
                )
            with contextlib.redirect_stdout(io.StringIO()) as out:
                export._check_no_orphan_artifacts(d, [{"N": 1}, {"N": 2}])
        self.assertIn("k_n2.h", out.getvalue())
        self.assertIn("no sidecar claims", out.getvalue())

    def test_a_fresh_trees_failed_first_export_is_not_called_corruption(self):
        # What a first export looks like when it dies: the DSL writes the .h before the
        # .o, so a failed compile strands one per point with no sidecar yet.
        with tempfile.TemporaryDirectory() as d:
            for name in ("k_n1.o", "k_n1.h"):
                open(os.path.join(d, name), "w").close()
            with contextlib.redirect_stdout(io.StringIO()) as out:
                export._check_no_orphan_artifacts(d, [{"N": 1}])
        self.assertIn("no sidecar claims", out.getvalue())
        self.assertNotIn("partial copy", out.getvalue())

    def test_artifacts_with_sidecar_are_fine(self):
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "k.o"), "w").close()
            with open(os.path.join(d, "k.json"), "w") as f:
                # version, or the stale half short-circuits at the schema gate and
                # the spec comparison below never runs.
                json.dump(
                    {
                        "version": export.SIDECAR_VERSION,
                        "prefix": "k",
                        "kind": "cutedsl",
                        "spec": {"N": 1},
                    },
                    f,
                )
            with contextlib.redirect_stdout(io.StringIO()) as out:
                export._check_no_orphan_artifacts(d, [{"N": 1}])
        # Silent, not merely non-fatal: claiming per artifact is what stops a healthy
        # directory reporting its own kernels as orphans.
        self.assertEqual(out.getvalue(), "")

    def test_unreadable_sidecar_is_fatal(self):
        # Present but unparsable: corruption, not an interrupted run.
        # Re-exporting would paper over it (and --force skips the check).
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "k.json")
            with open(path, "w") as f:
                f.write("{truncated")
            with self.assertRaisesRegex(RuntimeError, "could not be read"):
                export._read_sidecar(path)


class TestLauncherGeneration(unittest.TestCase):
    def test_read_only_inputs_take_const_data_ptr(self):
        # read_only args must go through const_data_ptr: a mutable data_ptr()
        # materializes copy-on-write inputs on every call.
        sc: dict = dict(SIDECAR)
        sc["tensor_args"] = [
            {"name": "mX", "dynamic_sizes": [0], "read_only": True},
            {"name": "mOut", "dynamic_sizes": [0]},
        ]
        src = gen_aot_lib.gen_launcher(sc)
        self.assertIn("mX_s.data = const_cast<void*>(mX.const_data_ptr());", src)
        self.assertIn("mOut_s.data = mOut.mutable_data_ptr();", src)

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
    # Set explicitly, since fixtures bypass the loader that normalizes ARCHS on
    # real declarations. Annotated so subclasses can narrow it.
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


@contextlib.contextmanager
def _patched_generation(ops, declarations=(_FakeDecl,)):
    """What every gen_aot_lib.main() test patches: the ops dir discovery walks,
    the declarations it loads (None to use the real loader on a written aot.py,
    or a callable to answer per path), and the two native_functions.yaml lookups
    -- "fakeop" is not a real op, so the real ones raise."""
    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(gen_aot_lib, "OPS_DIR", ops))
        if declarations is not None:
            load = (
                declarations
                if callable(declarations)
                else lambda path: list(declarations)
            )
            stack.enter_context(
                mock.patch.object(gen_aot_lib.decl, "load_declarations", load)
            )
        stack.enter_context(
            mock.patch.object(
                gen_aot_lib,
                "impl_signature_params",
                lambda op: "const at::Tensor & self, int64_t k",
            )
        )
        stack.enter_context(
            mock.patch.object(gen_aot_lib, "precomputed_args", lambda op: [])
        )
        yield


class TestInt32SizeGate(unittest.TestCase):
    # The exported ABI carries int32_t shape slots while aten sizes are int64_t, so
    # the gate must decline oversized dims rather than let the cast truncate them.
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

    def test_the_gate_expression_is_exact(self):
        # The WHOLE expression: substrings cannot tell it from an inverted sense, probes
        # joined with && , or only the first tensor gated. Two tensors, so the last shows.
        gate = gen_aot_lib._int32_size_gate(
            "const at::Tensor & self, int64_t k, const at::Tensor & out, "
            "const std::optional<at::Tensor>& weight"
        )
        want = (
            "  if (C10_UNLIKELY("
            "self.sizes().end() != std::find_if(self.sizes().begin(), "
            "self.sizes().end(), _naot_dim_too_big)"
            " || out.sizes().end() != std::find_if(out.sizes().begin(), "
            "out.sizes().end(), _naot_dim_too_big)"
            " || (weight.has_value() && weight->sizes().end() != "
            "std::find_if(weight->sizes().begin(), weight->sizes().end(), "
            "_naot_dim_too_big))"
            ")) return false;"
        )
        self.assertEqual(gate.splitlines()[-1], want)

    def test_gate_empty_without_tensors(self):
        self.assertEqual(gen_aot_lib._int32_size_gate("int64_t k, bool largest"), "")

    def test_gate_emitted_into_the_stub(self):
        sidecar = dict(SIDECAR, spec={"N": 1024, "K": 8})
        src = gen_aot_lib.gen_op(
            "fakeop",
            "CUDA",
            _FakeDecl,
            [sidecar],
            "const at::Tensor & self, int64_t k",
        )
        self.assertIn("inline bool _naot_dim_too_big", src)
        # The comparison, not just the signature: a helper that can never fire leaves
        # every oversized dim to truncate through the launcher's static_cast.
        self.assertIn("return d > std::numeric_limits<int32_t>::max();", src)
        # _FakeDecl declares no cpp_covers, so there is exactly one gate site.
        self.assertEqual(src.count("// Size gate:"), 1)
        self.assertIn("self.sizes().begin()", src)

    def test_covers_gets_the_same_gates_as_the_stub(self):
        # Coverage must be no wider than the stub's acceptance, or the router hands a
        # call to a stub that refuses it and it loses its JIT route.
        class _WithCovers(_FakeDecl):
            @staticmethod
            def cpp_covers():
                return "return true;"

        sidecar = dict(SIDECAR, spec={"N": 1024, "K": 8})
        src = gen_aot_lib.gen_op(
            "fakeop",
            "CUDA",
            _WithCovers,
            [sidecar],
            "const at::Tensor & self, int64_t k",
            covers=(
                "const at::Tensor & self, int64_t k",
                "covers_fakeop(...)-> bool",
                "return true;",
            ),
        )
        # Two of each: once in the kernel, once in the covers predicate.
        self.assertEqual(src.count("// Size gate:"), 2)
        self.assertEqual(src.count("// Device gate:"), 2)
        # ...reading the TENSOR's device, not the current one: the router calls
        # covers before any device guard.
        self.assertIn(
            "const auto* _naot_props = at::cuda::getDeviceProperties(self.device().index());",
            src,
        )
        self.assertIn(
            "const auto* _naot_props = at::cuda::getCurrentDeviceProperties();", src
        )


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
        # The LAST statement of the kernel body, which routes an unmatched call to
        # op.impl. Positional, because the two gates each emit a "return false;" too.
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
            # The contract's "absent prelude" case; callable -> None trips bad-override.
            cpp_dispatch_prelude = None  # pyrefly: ignore [bad-override]

        sidecar = dict(SIDECAR, spec={"N": 1024, "K": 8})
        src = gen_aot_lib.gen_op(
            "fakeop", "CUDA", NoPrelude, [sidecar], "const at::Tensor & self, int64_t k"
        )
        self.assertNotIn("scalar_type() != at::kFloat", src)
        self.assertIn("if (N == 1024 && k == 8) {", src)

    def test_a_hook_that_returns_nothing_emits_nothing(self):
        # The attribute exists, so the callable default does not fire; `or ""` is what
        # keeps "None" out of the generated C++ when a body forgets to return.
        class NoneHooks(_FakeDecl):
            @staticmethod
            def cpp_dispatch_prelude():
                return None

            @staticmethod
            def cpp_helpers():
                return None

        sidecar = dict(SIDECAR, spec={"N": 1024, "K": 8})
        src = gen_aot_lib.gen_op(
            "fakeop", "CUDA", NoneHooks, [sidecar], "const at::Tensor & self, int64_t k"
        )
        self.assertNotIn("None", src)
        self.assertIn("if (N == 1024 && k == 8) {", src)

    def test_the_branch_launches_inside_its_condition(self):
        # The BLOCK, not its three pieces: separate assertions accept a branch that
        # launches before the if, or returns false after launching.
        sidecar = dict(SIDECAR, spec={"N": 1024, "K": 8})
        src = gen_aot_lib.gen_op(
            "fakeop",
            "CUDA",
            _FakeDecl,
            [sidecar],
            "const at::Tensor & self, int64_t k, const at::Tensor & out",
        )
        self.assertIn(
            "    if (N == 1024 && k == 8) {\n"
            "      launch_fakeop_f32_n1024_k8(self, out, "
            "at::cuda::getCurrentCUDAStream());\n"
            "      return true;\n"
            "    }",
            src,
        )

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
        # The declaration's BODY: the is_cuda() comes from the generated guard, and a
        # covers reduced to those guards claims coverage the stub declines.
        self.assertIn("return self.scalar_type() == at::kFloat && k == 8;", src)
        # ...and it ends by declining, so a body that answers only some paths does
        # not fall off the end of a bool function.
        fn = src.split("bool fakeop_cuda_covers(")[1].split("\n}")[0]
        self.assertTrue(
            fn.rstrip().endswith(
                "return false;  // undecided above: not covered, "
                "the same default the stub takes"
            ),
            fn,
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
        # The schema shapes per-argument rendering must handle: a kwarg-only section,
        # a list default ('int[1] dim=[]'), Scalar defaults.
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


class TestStructuredIntrospection(unittest.TestCase):
    """Read off native_functions.yaml, so they are tested against REAL ops the
    way covers_signature already is. Every generation test patches these two out,
    which left `precomputed_args` returning [] and impl_signature_params
    returning nonsense both invisible to the suite."""

    def test_precomputed_args_reports_what_meta_replaces(self):
        # Why the generated header states this: index_add's dim arrives wrapped and
        # sum.dim_IntList's raw, and a raw negative dim declines every dim=-1 call.
        self.assertEqual(gen_aot_lib.precomputed_args("index_add"), ["dim"])
        self.assertEqual(gen_aot_lib.precomputed_args("sum.dim_IntList"), [])

    def test_impl_signature_is_the_structured_impl_signature(self):
        params = gen_aot_lib.impl_signature_params("topk")
        # Outputs last, because meta() allocated them before the stub runs.
        self.assertTrue(params.startswith("const at::Tensor & self, int64_t k"))
        self.assertIn("const at::Tensor & values", params)
        self.assertIn("const at::Tensor & indices", params)

    def test_precompute_note_states_which_case_applies(self):
        # The note goes into every generated file, where an author debugging a
        # silently-declining dim reads it.
        sidecar = dict(SIDECAR, spec={"N": 1024, "K": 8})
        wrapped = gen_aot_lib.gen_op(
            "fakeop",
            "CUDA",
            _FakeDecl,
            [sidecar],
            "const at::Tensor & self, int64_t k",
            None,
            ["dim"],
        )
        self.assertIn("precomputes (impl receives them wrapped/replaced): dim", wrapped)
        raw = gen_aot_lib.gen_op(
            "fakeop",
            "CUDA",
            _FakeDecl,
            [sidecar],
            "const at::Tensor & self, int64_t k",
            None,
            [],
        )
        self.assertIn("precomputes NOTHING", raw)
        self.assertIn("arrive RAW", raw)


class TestAtomicWrites(unittest.TestCase):
    def test_a_failed_write_leaves_neither_a_partial_file_nor_a_tmp(self):
        # CMake reads the emitted file as authoritative, so a half-written one is
        # worse than none; asserting the final content alone would not show that.
        with tempfile.TemporaryDirectory() as d:
            target = os.path.join(d, "manifest.txt")
            with open(target, "w") as f:
                f.write("PREVIOUS\n")
            real_replace = os.replace
            with mock.patch.object(
                gen_aot_lib.os, "replace", side_effect=OSError("simulated failure")
            ):
                with self.assertRaises(OSError):
                    gen_aot_lib._write_atomic(target, "NEW\n")
            self.assertIs(real_replace, os.replace)
            with open(target) as f:
                self.assertEqual(f.read(), "PREVIOUS\n", "target was clobbered")
            self.assertEqual(
                [f for f in os.listdir(d) if f.endswith(".tmp")], [], "left a .tmp"
            )


class TestShippedVsDeclaredArchs(unittest.TestCase):
    def test_shipping_an_arch_the_declaration_disowns_is_fatal(self):
        # The shared fixture declares every arch, so this needs its own narrow one:
        # otherwise a packaging bug emits a gate for hardware the op disowns.
        class _Narrow(_FakeDecl):
            ARCHS = ("sm_100a",)

        s90 = dict(SIDECAR, prefix="fakeop_p__sm90a", arch="sm_90a")
        with self.assertRaisesRegex(RuntimeError, "declaration supports only"):
            gen_aot_lib.gen_op(
                "fakeop", "CUDA", _Narrow, [s90], "const at::Tensor & self, int64_t k"
            )

    def test_a_disowned_tree_beside_a_claimed_one_is_fatal(self):
        # Same capability, so the plain spelling loses the tie-break: the check must run
        # over every exported tree or a stale sm_100 beside sm_100a passes unnoticed.
        class _Narrow(_FakeDecl):
            ARCHS = ("sm_100a",)

        s100 = dict(SIDECAR, prefix="fakeop_p__sm100", arch="sm_100", _dir="out/sm_100")
        s100a = dict(
            SIDECAR, prefix="fakeop_p__sm100a", arch="sm_100a", _dir="out/sm_100a"
        )
        with self.assertRaisesRegex(RuntimeError, "declaration supports only"):
            gen_aot_lib.gen_op(
                "fakeop",
                "CUDA",
                _Narrow,
                [s100, s100a],
                "const at::Tensor & self, int64_t k",
            )
        # The message must name the tree to delete, not the survivor.
        with self.assertRaisesRegex(RuntimeError, r"Delete out/sm_100 --"):
            gen_aot_lib.gen_op(
                "fakeop",
                "CUDA",
                _Narrow,
                [s100, s100a],
                "const at::Tensor & self, int64_t k",
            )

    def test_declaring_more_than_was_shipped_is_fine(self):
        # The other direction is normal: a build targeting one arch ships one
        # tree for an op whose kernels support several.
        s100 = dict(SIDECAR, prefix="fakeop_p__sm100a", arch="sm_100a")
        src = gen_aot_lib.gen_op(
            "fakeop", "CUDA", _FakeDecl, [s100], "const at::Tensor & self, int64_t k"
        )
        self.assertIn("major == 10", src)
        self.assertNotIn("major == 9", src)


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
        # The failure the grouping exists to prevent: loading a module built for other
        # hardware fails inside the launcher instead of declining to aten.
        body = self._body()
        i9 = body.index("_naot_props->major == 9 && _naot_props->minor == 0) {")
        i10 = body.index("_naot_props->major == 10 && _naot_props->minor == 0) {")
        self.assertLess(i9, i10)
        sm90_branch, sm100_branch = body[i9:i10], body[i10:]
        self.assertIn("launch_fakeop_p__sm90a(", sm90_branch)
        self.assertNotIn("launch_fakeop_p__sm100a(", sm90_branch)
        self.assertIn("launch_fakeop_p__sm100a(", sm100_branch)
        self.assertNotIn("launch_fakeop_p__sm90a(", sm100_branch)


class TestSourceClosureAndRuntimes(unittest.TestCase):
    # What makes an artifact stale: the files whose contents decide what it means,
    # and the compiler that built it. The artifact records neither.

    def test_closure_covers_shared_declaration_machinery(self):
        # The grid expander and the loader decide what a declaration means, and arrive
        # by import, so only the sys.modules half of the closure catches them.
        names = {os.path.basename(p) for p in export.source_closure()}
        for want in (
            "native_aot_spec_grid.py",
            "native_aot_decl.py",
            "toolchains.py",
        ):
            self.assertIn(want, names, f"{want} must invalidate artifacts")

    def test_closure_excludes_the_consumer_tools(self):
        # Neither can change what an artifact means. Against fixtures in a patched
        # _HERE, since asserting absence from the tree would pass with no filter.
        with tempfile.TemporaryDirectory() as d:
            for name in ("export.py", "gen_aot_lib.py", "build_stage2.py"):
                with open(os.path.join(d, name), "w") as f:
                    f.write("# fixture\n")
            with mock.patch.object(export, "_HERE", d):
                names = {os.path.basename(p) for p in export.source_closure()}
        # The exporter's own sources ARE hashed, which is what makes the two
        # absences below mean the filter rather than an empty glob.
        self.assertIn("export.py", names)
        for unwanted in ("gen_aot_lib.py", "build_stage2.py"):
            self.assertNotIn(unwanted, names)

    def test_closure_survives_sys_modules_mutation(self):
        # Hashing imports hashlib lazily, so on a cold interpreter the walk mutates the
        # dict it is iterating. Forced by hashing a module that is not loaded yet.
        import sys

        real_hash = export._file_hash

        def hash_and_import(path):
            importlib.import_module("wave")  # stdlib, unlikely to be loaded
            return real_hash(path)

        sys.modules.pop("wave", None)
        with mock.patch.object(export, "_file_hash", hash_and_import):
            export.source_closure()

    def test_runtimes_current_detects_a_compiler_upgrade(self):
        # The DSL's version is in no file the closure hashes, so an upgraded wheel
        # changes nothing on disk. Patched rather than read: with no wheels installed
        # the live call is all-"absent" and takes the ignorance arm instead.
        current = {"nvidia-cutlass-dsl": "4.6.2", "apache-tvm-ffi": "0.1.11"}
        with mock.patch.object(export, "runtime_versions", lambda kind: current):
            self.assertTrue(
                export.runtimes_current({"kind": "cutedsl", "runtimes": current})
            )
            older = dict.fromkeys(current, "0.0.1")
            self.assertFalse(
                export.runtimes_current({"kind": "cutedsl", "runtimes": older})
            )
            # A sidecar predating the record is stale, like one with no closure.
            self.assertFalse(export.runtimes_current({"kind": "cutedsl"}))

    def test_runtimes_current_does_not_call_ignorance_staleness(self):
        # Generation may run where the DSL wheels are absent. Declaring every
        # artifact stale there would fail a build that cannot re-export anyway.
        with mock.patch.object(
            export, "runtime_versions", lambda kind: {"nvidia-cutlass-dsl": "absent"}
        ):
            self.assertTrue(export.runtimes_current({"kind": "cutedsl"}))

    def test_runtime_versions_reads_metadata_not_the_module(self):
        # Distribution names, so the skip path never imports the DSL: module `cutlass`
        # ships in nvidia-cutlass-dsl, which is not derivable.
        self.assertIn(
            "nvidia-cutlass-dsl", toolchains.get_toolchain("cutedsl").RUNTIME_DISTS
        )
        versions = export.runtime_versions("cutedsl")
        # Sorted keys, so two runs on one machine record byte-identical sidecars
        # (a dict whose order drifts would compare unequal and re-export).
        self.assertEqual(list(versions), sorted(versions))
        self.assertEqual(
            sorted(versions), sorted(toolchains.get_toolchain("cutedsl").RUNTIME_DISTS)
        )
        for dist, v in versions.items():
            self.assertTrue(v, f"{dist} recorded an empty version")

        # The VALUE, against a known one: _RUNTIMES is produced by this function, so a
        # garbage version would move the fixture with it.
        from importlib.metadata import PackageNotFoundError

        dists = toolchains.get_toolchain("cutedsl").RUNTIME_DISTS
        with mock.patch("importlib.metadata.version", return_value="1.2.3"):
            got = export.runtime_versions("cutedsl")
        self.assertEqual(got, dict.fromkeys(dists, "1.2.3"))
        # Absent rather than omitted, so a sidecar compiled without the wheel differs
        # from one predating this record.
        with mock.patch("importlib.metadata.version", side_effect=PackageNotFoundError):
            got = export.runtime_versions("cutedsl")
        self.assertEqual(got, dict.fromkeys(dists, "absent"))

    def test_sources_current_roundtrip(self):
        # A sidecar whose recorded closure matches the tree is current;
        # editing any recorded file (or recording none) makes it stale.
        rel = _DECL_REL
        h = export._file_hash(os.path.join(export.REPO, rel))
        good = {"version": export.SIDECAR_VERSION, "sources": {rel: h}}
        self.assertTrue(export.sources_current(good))
        # schema-version mismatch is stale even with current sources
        self.assertFalse(export.sources_current({"version": 0, "sources": {rel: h}}))
        self.assertFalse(export.sources_current({"sources": {rel: "0" * 16}}))
        self.assertFalse(export.sources_current({}))
        self.assertFalse(export.sources_current({"sources": {"no/such/file.py": "aa"}}))

    def test_stale_point_reexports_without_force(self):
        with tempfile.TemporaryDirectory() as d:
            point = {"dtype": "float32"}
            job = ("fakeop", "aot_kernel.py", point, d, None)
            _write_sidecar(d, point)
            with _no_ambient_arch():
                self.assertFalse(export._job_needed(job, force=False))
            _write_sidecar(d, point, sources={_DECL_REL: "0" * 16})
            # Inside the guard like the call above: unguarded, the recorded arch
            # mismatches the detected one and staleness is never reached.
            with _no_ambient_arch():
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

    @contextlib.contextmanager
    def _generated(self, touch=True, header=None, arch_list=None):
        """Run main() over a one-kernel tree; yield (artifacts_dir, error|None).

        ``touch`` writes the artifacts the sidecar claims; ``header`` overwrites
        the ABI header, for the cases where generation must REFUSE and the
        interesting assertion is that nothing was left behind; ``arch_list`` is
        passed through as --arch-list."""
        with tempfile.TemporaryDirectory() as art, tempfile.TemporaryDirectory() as ops:
            art_op = os.path.join(art, "sm_100a", "fakeop")
            os.makedirs(art_op)
            if touch:
                _touch_artifacts(art_op, SIDECAR["prefix"])
            if header is not None:
                with open(os.path.join(art_op, SIDECAR["prefix"] + ".h"), "w") as f:
                    f.write(header)
            sidecar = dict(
                SIDECAR,
                spec={"N": 1024, "K": 8},
                sources=_current_sources(),
                runtimes=_RUNTIMES,
                version=export.SIDECAR_VERSION,
            )
            with open(os.path.join(art_op, SIDECAR["prefix"] + ".json"), "w") as f:
                json.dump(sidecar, f)
            os.makedirs(os.path.join(ops, "fakeop"))
            with open(os.path.join(ops, "fakeop", "aot.py"), "w") as f:
                f.write(self._DECL)
            err = None
            with _patched_generation(ops, declarations=None):
                argv = ["--artifacts-dir", art]
                if arch_list is not None:
                    argv += ["--arch-list", arch_list]
                try:
                    gen_aot_lib.main(argv)
                except Exception as e:
                    err = e
            yield art, err

    def test_a_second_generation_over_the_same_inputs_touches_nothing(self):
        # These files are build inputs: restamping them recompiles the generated
        # sources and relinks torch_cuda, dirtying all ~110 targets that consume it.
        with tempfile.TemporaryDirectory() as art, tempfile.TemporaryDirectory() as ops:
            art_op = os.path.join(art, "sm_100a", "fakeop")
            os.makedirs(art_op)
            _touch_artifacts(art_op, SIDECAR["prefix"])
            sidecar = dict(
                SIDECAR,
                spec={"N": 1024, "K": 8},
                sources=_current_sources(),
                runtimes=_RUNTIMES,
                version=export.SIDECAR_VERSION,
            )
            with open(os.path.join(art_op, SIDECAR["prefix"] + ".json"), "w") as f:
                json.dump(sidecar, f)
            os.makedirs(os.path.join(ops, "fakeop"))
            with open(os.path.join(ops, "fakeop", "aot.py"), "w") as f:
                f.write(self._DECL)

            # By INODE, not mtime: os.replace gives a rewritten file a new inode, while
            # two runs inside one filesystem timestamp tick share an mtime. The include
            # is exempt -- it is rewritten by construction (invalidated first, then
            # written) and only its timestamp is restored, which is what the build reads.
            def stamps():
                out = {}
                for root, _, files in os.walk(art):
                    for name in files:
                        p = os.path.join(root, name)
                        st = os.stat(p)
                        rel = os.path.relpath(p, art)
                        keep = (
                            st.st_mtime_ns
                            if name == gen_aot_lib.CMAKE_INCLUDE
                            else st.st_ino
                        )
                        out[rel] = (keep, st.st_size)
                return out

            with _patched_generation(ops, declarations=None):
                gen_aot_lib.main(["--artifacts-dir", art])
                first = stamps()
                # Past a timestamp tick, so a restamped file is visible as one.
                time.sleep(0.05)
                gen_aot_lib.main(["--artifacts-dir", art])
                self.assertEqual(stamps(), first)

    def test_main_writes_aot_source(self):
        # Artifacts live at <root>/<arch>/<decl_id>/. _generated writes them for real,
        # since generation refuses a sidecar describing a file that is not there.
        with self._generated() as (art, err):
            self.assertIsNone(err)
            art_op = os.path.join(art, "sm_100a", "fakeop")
            out = os.path.join(art, "fakeop", "aot_fakeop_cuda.cpp")
            self.assertTrue(os.path.exists(out))
            with open(out) as f:
                src = f.read()
            self.assertIn("fakeop_cuda_aot_kernel", src)
            # The include reaches from the generated source at <root>/<decl_id>/ into
            # the arch tree; inverted, every generated file fails to compile.
            self.assertIn(f'#include "../sm_100a/fakeop/{SIDECAR["prefix"]}.h"', src)
            # The emitted file is everything CMake reads: the sources to compile, the
            # objects to link, and the arch list they were generated for.
            man = _manifest(art)
            self.assertEqual(man["sources"], [out])
            self.assertEqual(
                man["objects"], [os.path.join(art_op, SIDECAR["prefix"] + ".o")]
            )
            for p in man["sources"] + man["objects"]:
                self.assertTrue(os.path.isabs(p), f"{p} is not absolute")
            # No --arch-list was passed, so the file must claim NOTHING rather than
            # "", which CMake distinguishes. A literal, not os.getenv.
            self.assertEqual(man["arch_list"], "absent")
            # Version-script CONTENTS, not just existence: written from an empty
            # prefix list it localizes nothing, exporting every kernel symbol.
            with open(os.path.join(art, gen_aot_lib.VERSION_SCRIPT)) as f:
                ver = f.read()
            self.assertIn(f"{SIDECAR['prefix']}_*;", ver)
            self.assertIn(f"_mlir_*{SIDECAR['prefix']}*;", ver)

    def test_main_records_the_arch_list_it_was_given(self):
        # A literal, not os.getenv: the recorded claim must come from the flag rather
        # than from the ambient environment.
        with self._generated(arch_list="9.0a;10.0a") as (art, err):
            self.assertIsNone(err)
            self.assertEqual(_manifest(art)["arch_list"], "9.0a;10.0a")

    def test_main_refuses_a_sidecar_whose_object_is_absent(self):
        # The launcher would be emitted and the object missing at link time. A
        # comment in test_main_writes_aot_source claimed to cover this; it
        # creates the artifacts and never reaches the refusal.
        with self._generated(touch=False) as (art, err):
            self.assertIsNotNone(err, "expected a refusal")
            self.assertIn("not on disk", str(err))
            # ...and nothing was left behind for the CMake glob to compile.
            self.assertFalse(glob.glob(os.path.join(art, "*", "aot_*.cpp")))

    def test_main_refuses_an_int32_stride_abi(self):
        # validate_abi's only production call site, since TestAbiValidation invokes the
        # method directly. A real export's layout with mX's strides narrowed to int32.
        pre = SIDECAR["prefix"]
        bad = "".join(
            f"typedef struct {{ void* data; int32_t dynamic_shapes[2];\n"
            f"  {w} dynamic_strides[1]; }} {pre}_Tensor_{t}_t;\n"
            for t, w in (("mX", "int32_t"), ("mOut", "int64_t"))
        )
        with self._generated(header=bad) as (
            art,
            err,
        ):
            self.assertIsNotNone(err, "expected a refusal")
            self.assertIn("must be declared 64-bit", str(err))
            self.assertFalse(glob.glob(os.path.join(art, "*", "aot_*.cpp")))

    def test_main_tolerates_missing_artifacts_dir(self):
        # Zero declarations: export creates no artifacts dir at all, so main() must
        # no-op rather than raise FileNotFoundError.
        import sys

        with tempfile.TemporaryDirectory() as d:
            argv = sys.argv
            sys.argv = ["gen_aot_lib.py", "--artifacts-dir", os.path.join(d, "absent")]
            try:
                gen_aot_lib.main()
            finally:
                sys.argv = argv


def _has_zip64_extra(info: zipfile.ZipInfo) -> bool:
    """Whether a member's central-directory extra carries a 0x0001 (ZIP64) field."""
    extra, i = info.extra, 0
    while i + 4 <= len(extra):
        tag = int.from_bytes(extra[i : i + 2], "little")
        length = int.from_bytes(extra[i + 2 : i + 4], "little")
        if tag == 1:
            return True
        i += 4 + length
    return False


class TestWheelPatch(unittest.TestCase):
    LIB = "torch/lib/libtorch_cuda.so"

    @staticmethod
    def _rec(name, data):
        """A RECORD line, implemented here rather than imported from the code
        under test -- that is what lets the digest assertion actually fail."""
        import base64
        import hashlib

        h = base64.urlsafe_b64encode(hashlib.sha256(data).digest())
        return f"{name},sha256={h.rstrip(b'=').decode()},{len(data)}"

    def _make_wheel(self, d: str, record_entry: bool = True) -> str:
        rec = self._rec
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
        with tempfile.TemporaryDirectory() as d:
            whl = self._make_wheel(d)
            lib = os.path.join(d, "libtorch_cuda.so")
            with open(lib, "wb") as f:
                f.write(b"NEW" * 64)
            build_stage2.patch_wheel(whl, lib)
            with zipfile.ZipFile(whl) as zf:
                self.assertEqual(zf.read(self.LIB), b"NEW" * 64)
                lines = zf.read("torch-0.0.dist-info/RECORD").decode().splitlines()
                # An independent expectation, since computing it as the code does
                # would pass with the digest switched to md5.
                self.assertIn(self._rec(self.LIB, b"NEW" * 64), lines)
                self.assertRegex(self._rec(self.LIB, b"NEW" * 64), r"[-_]")
                # untouched members keep their RECORD entries; no dup names
                self.assertTrue(any(x.startswith("torch/__init__.py,") for x in lines))
                self.assertEqual(len(zf.namelist()), len(set(zf.namelist())))

    def test_compression_is_preserved(self):
        # The replaced member is the biggest thing in the wheel, which every test shard
        # downloads. Untouched members must keep their compression too.
        with tempfile.TemporaryDirectory() as d:
            whl = os.path.join(d, "torch-0.0-cp310-linux_x86_64.whl")
            record = "torch-0.0.dist-info/RECORD"
            # Deliberately mixed: DEFLATED for the lib and one member, STORED
            # for another, so "preserved" cannot be satisfied by a constant.
            with zipfile.ZipFile(whl, "w") as zf:
                zf.writestr(self.LIB, b"OLD" * 999, zipfile.ZIP_DEFLATED)
                zf.writestr("torch/big.py", b"y" * 4096, zipfile.ZIP_DEFLATED)
                zf.writestr("torch/raw.bin", b"z" * 4096, zipfile.ZIP_STORED)
                zf.writestr(record, f"{self.LIB},,\n{record},,\n")
            lib = os.path.join(d, "libtorch_cuda.so")
            with open(lib, "wb") as f:
                f.write(b"NEW" * 999)
            build_stage2.patch_wheel(whl, lib)
            with zipfile.ZipFile(whl) as zf:
                self.assertEqual(
                    zf.getinfo(self.LIB).compress_type, zipfile.ZIP_DEFLATED
                )
                self.assertEqual(
                    zf.getinfo("torch/big.py").compress_type, zipfile.ZIP_DEFLATED
                )
                self.assertEqual(
                    zf.getinfo("torch/raw.bin").compress_type, zipfile.ZIP_STORED
                )
                self.assertEqual(zf.read(self.LIB), b"NEW" * 999)
                self.assertEqual(zf.read("torch/raw.bin"), b"z" * 4096)

    def test_members_are_streamed_not_read_whole(self):
        # A CUDA wheel holds members in the hundreds of MiB, and read()+writestr holds
        # each twice. Asserted by watching for the read that must not happen.
        with tempfile.TemporaryDirectory() as d:
            whl = self._make_wheel(d)
            lib = os.path.join(d, "libtorch_cuda.so")
            with open(lib, "wb") as f:
                f.write(b"NEW")
            real_read = zipfile.ZipFile.read
            reads = []

            def spy(self, name, *a, **kw):
                reads.append(name if isinstance(name, str) else name.filename)
                return real_read(self, name, *a, **kw)

            with mock.patch.object(zipfile.ZipFile, "read", spy):
                build_stage2.patch_wheel(whl, lib)
            # RECORD is read deliberately (it has to be rewritten); the payload
            # members must not be.
            self.assertNotIn("torch/__init__.py", reads)

    def test_a_failed_rewrite_leaves_the_original_wheel_untouched(self):
        # tmp + rename: an interrupted rewrite must not leave a half-written wheel
        # where a valid one was, nor a .naot.tmp beside it.
        with tempfile.TemporaryDirectory() as d:
            whl = self._make_wheel(d)
            with open(whl, "rb") as f:
                before = f.read()
            lib = os.path.join(d, "libtorch_cuda.so")
            with open(lib, "wb") as f:
                f.write(b"NEW")
            with mock.patch.object(
                build_stage2.shutil, "move", side_effect=OSError("simulated failure")
            ):
                with self.assertRaises(OSError):
                    build_stage2.patch_wheel(whl, lib)
            with open(whl, "rb") as f:
                self.assertEqual(f.read(), before, "original wheel was modified")
            self.assertEqual(
                [f for f in os.listdir(d) if f.endswith(".tmp")],
                [],
                "left a temp wheel behind",
            )
            # ...and it is still a readable zip.
            with zipfile.ZipFile(whl) as zf:
                self.assertIsNone(zf.testzip())

    def test_zip64_members_survive_the_copy(self):
        # ZIP64_LIMIT is patched rather than writing 2 GiB, once either side of it.
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(zipfile, "ZIP64_LIMIT", 2):
                whl = self._make_wheel(d)  # every member gets a real ZIP64 extra
            with zipfile.ZipFile(whl) as zf:
                self.assertTrue(
                    all(_has_zip64_extra(i) for i in zf.infolist()),
                    "the fixture is supposed to need ZIP64 everywhere",
                )
            lib = os.path.join(d, "libtorch_cuda.so")
            with open(lib, "wb") as f:
                f.write(b"NEW" * 64)
            build_stage2.patch_wheel(whl, lib)  # ...rewritten below the limit
            with zipfile.ZipFile(whl) as zf:
                self.assertIsNone(zf.testzip())
                self.assertEqual(zf.read(self.LIB), b"NEW" * 64)
                self.assertEqual(
                    [n for n in zf.namelist() if _has_zip64_extra(zf.getinfo(n))],
                    [],
                    "a copied member kept the source's ZIP64 extra field",
                )
                # ...and the library keeps its position, so no later member moves and
                # .dist-info stays where an archiver expects it.
                self.assertEqual(zf.namelist()[0], self.LIB)
            # ...and again with every member above the limit, which is what (b)
            # needs.
            with mock.patch.object(zipfile, "ZIP64_LIMIT", 0):
                build_stage2.patch_wheel(whl, lib)
            with zipfile.ZipFile(whl) as zf:
                self.assertIsNone(zf.testzip())
                self.assertEqual(zf.read(self.LIB), b"NEW" * 64)

    def test_the_staging_name_belongs_to_this_process(self):
        # Two concurrent stage-2 runs would otherwise share one .naot.tmp path, and
        # the loser would rename its half-written archive over the finished wheel.
        seen = []
        with tempfile.TemporaryDirectory() as d:
            whl = self._make_wheel(d)
            lib = os.path.join(d, "libtorch_cuda.so")
            with open(lib, "wb") as f:
                f.write(b"NEW")
            real_move = build_stage2.shutil.move

            def spy(src, dst):
                seen.append(src)
                return real_move(src, dst)

            with mock.patch.object(build_stage2.shutil, "move", spy):
                build_stage2.patch_wheel(whl, lib)
        self.assertEqual(len(seen), 1, seen)
        self.assertIn(str(os.getpid()), os.path.basename(seen[0]))

    def test_the_rewritten_record_keeps_its_compression(self):
        # writestr() with a plain name takes the ZipFile default, which the destination
        # has none of, so the RECORD would be the one member stored uncompressed.
        with tempfile.TemporaryDirectory() as d:
            whl = os.path.join(d, "torch-0.0.0-cp310-cp310-linux_x86_64.whl")
            lib = "torch/lib/libtorch_cuda.so"
            rec = "torch-0.0.0.dist-info/RECORD"
            body = "\n".join(f"torch/f{i}.py,sha256=x,{i}" for i in range(4000))
            with zipfile.ZipFile(whl, "w", zipfile.ZIP_DEFLATED) as z:
                z.writestr(lib, b"old")
                z.writestr(rec, f"{lib},sha256=old,3\n{body}\n")
            new_lib = os.path.join(d, "new.so")
            with open(new_lib, "wb") as f:
                f.write(b"new kernels")
            build_stage2.patch_wheel(whl, new_lib)
            with zipfile.ZipFile(whl) as z:
                info = z.getinfo(rec)
                self.assertEqual(info.compress_type, zipfile.ZIP_DEFLATED)
                self.assertLess(info.compress_size, info.file_size)
                # ...and the rewrite is still correct: sizes recomputed, entry
                # updated, every other line kept, archive readable.
                text = z.read(rec).decode()
                self.assertNotIn("sha256=old", text)
                self.assertEqual(text.count("\n"), 4001)

    def test_non_torch_wheel_is_refused(self):
        # A wrong --wheel argument, or a renamed member, must say so rather than
        # silently produce a wheel with no kernels in it.
        with tempfile.TemporaryDirectory() as d:
            whl = os.path.join(d, "nottorch-0.0-py3-none-any.whl")
            with zipfile.ZipFile(whl, "w") as zf:
                zf.writestr("nottorch/__init__.py", b"x")
                zf.writestr("notorch-0.0.dist-info/RECORD", "")
            lib = os.path.join(d, "libtorch_cuda.so")
            open(lib, "wb").close()
            with self.assertRaisesRegex(RuntimeError, "not a torch wheel"):
                build_stage2.patch_wheel(whl, lib)

    def test_record_without_the_lib_entry_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            whl = self._make_wheel(d, record_entry=False)
            lib = os.path.join(d, "libtorch_cuda.so")
            with open(lib, "wb") as f:
                f.write(b"NEW")
            with self.assertRaisesRegex(RuntimeError, "RECORD has no entry"):
                build_stage2.patch_wheel(whl, lib)


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

    def test_source_closure_omits_a_declaration_not_passed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            decl_path = os.path.join(tmpdir, "aot.py")
            with open(decl_path, "w") as f:
                f.write("ATEN_OP = 'fake'\n")
            # The closure records the declaration it is given, or editing any aot.py
            # would invalidate every other op's artifacts.
            self.assertNotIn(
                os.path.relpath(decl_path, export.REPO), export.source_closure()
            )
            self.assertIn(
                os.path.relpath(decl_path, export.REPO),
                export.source_closure(decl_path),
            )


class TestStaleGridPointArtifacts(unittest.TestCase):
    def test_sidecar_for_dropped_grid_point_is_fatal(self):
        # Its .o is still matched by the CMake glob and would link with no
        # launcher referencing it.
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "gone.json"), "w") as f:
                # version, because the check reads the schema first: without one this
                # fixture is "unreadable" rather than "stale".
                json.dump(
                    {
                        "version": export.SIDECAR_VERSION,
                        "prefix": "gone",
                        "spec": {"N": 4096},
                    },
                    f,
                )
            with self.assertRaisesRegex(RuntimeError, "no longer in the grid"):
                export._check_no_orphan_artifacts(tmpdir, [{"N": 1024}])

    def test_a_sidecar_from_another_schema_is_not_called_stale(self):
        # It cannot be read, so it cannot be judged: otherwise the next
        # SIDECAR_VERSION bump makes an existing tree demand `rm -rf`.
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "old.json"), "w") as f:
                json.dump(
                    {
                        "version": export.SIDECAR_VERSION + 1,
                        "prefix": "old",
                        "spec": {"N": 4096},
                    },
                    f,
                )
            export._check_no_orphan_artifacts(tmpdir, [{"N": 1024}])

    def test_sidecar_still_in_grid_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "live.json"), "w") as f:
                # version, for the reason above: without it the spec comparison never
                # runs and every sidecar reads as stale.
                json.dump(
                    {
                        "version": export.SIDECAR_VERSION,
                        "prefix": "live",
                        "spec": {"N": 1024},
                    },
                    f,
                )
            export._check_no_orphan_artifacts(tmpdir, [{"N": 1024}])


def _version(major, minor):
    """A stand-in for sys.version_info: supports .major/.minor and [:2]."""
    import collections

    vi = collections.namedtuple("vi", "major minor micro releaselevel serial")
    return vi(major, minor, 0, "final", 0)


class TestShouldRun(unittest.TestCase):
    """build_stage2.should_run decides whether the wheel ships kernels, so
    every skip arm needs to be deliberate rather than incidental."""

    # The inputs that otherwise RUN: a CUDA (not ROCm) torch -- all-probes-true
    # means ROCm, which skips for its own reason -- and one exportable arch.
    CUDA = {"torch.version.hip is not None": False}
    ARCH = {"TORCH_CUDA_ARCH_LIST": "10.0a"}

    def _run(
        self,
        probes,
        env=None,
        missing=(),
        declarations=True,
        platform="linux",
        cuda_major="13",
    ):
        # probes: expr -> bool, in place of a real torch subprocess. missing_runtimes is
        # patched, and REPO holds one declaration since should_run skips without any.
        with contextlib.ExitStack() as stack:
            repo = stack.enter_context(tempfile.TemporaryDirectory())
            ops = os.path.join(repo, "torch", "_native", "ops")
            os.makedirs(ops)
            if declarations:
                os.makedirs(os.path.join(ops, "fakeop"))
                open(os.path.join(ops, "fakeop", "aot.py"), "w").close()
            stack.enter_context(mock.patch.object(build_stage2, "REPO", repo))
            # The gate reads export.OPS_DIR (one spelling of the path, shared with
            # the generator), so that is what has to be redirected.
            stack.enter_context(mock.patch.object(export, "OPS_DIR", ops))
            # BUILD_DIR at a temp dir, since both variables below fall back to the
            # CMakeCache.txt, and it carries only the CUDA major the >=13 gate reads.
            build = stack.enter_context(tempfile.TemporaryDirectory())
            if cuda_major is not None:
                with open(os.path.join(build, "CMakeCache.txt"), "w") as f:
                    f.write(f"CUDAToolkit_VERSION_MAJOR:STRING={cuda_major}\n")
            stack.enter_context(mock.patch.object(build_stage2, "BUILD_DIR", build))
            # Neutralize the two variables should_run reads, before the caller's env:
            # both are exported by this commit's Test Plan. "" reads as absent.
            stack.enter_context(
                mock.patch.dict(
                    os.environ,
                    {"TORCH_NATIVE_AOT": "", "TORCH_CUDA_ARCH_LIST": ""},
                    clear=False,
                )
            )
            # sys.platform too, the arm should_run checks first, or every case on a
            # non-Linux box skips for a reason none of them is about.
            stack.enter_context(mock.patch.object(sys, "platform", platform))
            stack.enter_context(
                mock.patch.object(
                    build_stage2,
                    "_torch_probe",
                    side_effect=lambda e: probes.get(e, True),
                )
            )
            stack.enter_context(
                mock.patch.object(
                    toolchains.Toolchain,
                    "missing_runtimes",
                    classmethod(lambda cls: list(missing)),
                )
            )
            stack.enter_context(mock.patch.dict(os.environ, env or {}, clear=False))
            return build_stage2.should_run()

    def _run_on(self, version, free_threaded, missing):
        """should_run() as if running on another interpreter.

        Patches Py_GIL_DISABLED because that is what the gate reads, and what
        pip's tag matching reads: a free-threaded build needs a cp3XXt wheel
        even with the GIL re-enabled, where sys._is_gil_enabled() would say
        otherwise."""
        with (
            mock.patch.object(build_stage2.sys, "version_info", _version(*version)),
            mock.patch.object(
                build_stage2.sysconfig,
                "get_config_var",
                lambda name: 1 if (name == "Py_GIL_DISABLED" and free_threaded) else 0,
            ),
        ):
            return self._run(self.CUDA, self.ARCH, missing=missing)

    def test_a_non_linux_platform_skips(self):
        # Without this arm, a Windows or macOS CUDA build demands wheels that do not
        # exist for it, and everything downstream is ELF anyway. hip False and an
        # exportable arch list, i.e. inputs that otherwise RUN.
        for platform in ("darwin", "win32"):
            with self.subTest(platform=platform):
                self.assertFalse(self._run(self.CUDA, self.ARCH, platform=platform))
        # ...and the same inputs on Linux RUN, so the assertion above is about the
        # platform and not about some other arm firing first.
        self.assertTrue(self._run(self.CUDA, self.ARCH))

    def test_the_non_linux_skip_reason_names_the_platform(self):
        with contextlib.redirect_stderr(io.StringIO()) as err:
            self._run(self.CUDA, self.ARCH, platform="darwin")
        self.assertIn("Linux-only", err.getvalue())
        self.assertIn("darwin", err.getvalue())

    def test_interpreter_without_a_dsl_wheel_skips(self):
        # The release matrix builds interpreters the pinned DSL publishes no wheel for,
        # where RUN makes the CI shell run an install that cannot resolve and `set -ex`
        # kills the build, for a reason nobody can fix.
        for version, ft in (((3, 15), False), ((3, 15), True), ((3, 13), True)):
            with self.subTest(version=version, free_threaded=ft):
                with contextlib.redirect_stderr(io.StringIO()) as err:
                    self.assertFalse(self._run_on(version, ft, missing=("cutlass",)))
                # The reason names the interpreter and says installed wheels are used
                # anyway: it is the one skip a user can neither fix nor override.
                self.assertIn("no DSL wheel for python", err.getvalue())
                tag = f"{version[0]}.{version[1]}{'t' if ft else ''}"
                self.assertIn(tag, err.getvalue())

    def test_supported_interpreters_still_run(self):
        # 3.14t among them, because cp314t wheels exist for the pinned DSL: gating on
        # free-threaded alone would skip it and ship a kernel-free 3.14t CUDA wheel.
        # Free-threaded is not the question; a published tag is.
        for version, ft in (
            ((3, 10), False),
            ((3, 13), False),
            ((3, 14), False),
            ((3, 14), True),
        ):
            with self.subTest(version=version, free_threaded=ft):
                self.assertTrue(self._run_on(version, ft, missing=("cutlass",)))

    def test_interpreter_gate_survives_a_runtime_less_toolchain(self):
        # The gate asks "must this build install something?", so one toolchain missing
        # its runtimes is enough. Two kinds with DIFFERENT answers is the only shape
        # that tells `any` from `all`.
        class _NeedsNothing(toolchains.Toolchain):
            kind = "needsnothing"

        class _NeedsWheels(toolchains.Toolchain):
            kind = "needswheels"
            REQUIRED_RUNTIMES = ("definitely_not_installed",)

        registry = {"needsnothing": _NeedsNothing(), "needswheels": _NeedsWheels()}
        with (
            mock.patch.dict(toolchains.TOOLCHAINS, registry, clear=True),
            # Environment and platform too, like _run, or an exported
            # TORCH_NATIVE_AOT=0 answers at the first arm.
            mock.patch.dict(
                os.environ, {"TORCH_NATIVE_AOT": "", "TORCH_CUDA_ARCH_LIST": ""}
            ),
            mock.patch.object(sys, "platform", "linux"),
            mock.patch.object(build_stage2.sys, "version_info", _version(3, 15)),
            mock.patch.object(build_stage2.sysconfig, "get_config_var", lambda name: 0),
        ):
            # No base-class patch here: each kind answers for itself.
            with contextlib.ExitStack() as stack:
                repo = stack.enter_context(tempfile.TemporaryDirectory())
                d = os.path.join(repo, "torch", "_native", "ops", "fakeop")
                os.makedirs(d)
                open(os.path.join(d, "aot.py"), "w").close()
                stack.enter_context(mock.patch.object(build_stage2, "REPO", repo))
                # OPS_DIR, which is what should_run reads, and a BUILD_DIR whose
                # cache pins the CUDA major, the >=13 gate running first.
                stack.enter_context(
                    mock.patch.object(export, "OPS_DIR", os.path.dirname(d))
                )
                build = stack.enter_context(tempfile.TemporaryDirectory())
                with open(os.path.join(build, "CMakeCache.txt"), "w") as f:
                    f.write("CUDAToolkit_VERSION_MAJOR:STRING=13\n")
                stack.enter_context(mock.patch.object(build_stage2, "BUILD_DIR", build))
                stack.enter_context(
                    mock.patch.object(
                        build_stage2,
                        "_torch_probe",
                        lambda e: e != "torch.version.hip is not None",
                    )
                )
                stack.enter_context(
                    mock.patch.dict(
                        os.environ, {"TORCH_CUDA_ARCH_LIST": "10.0a"}, clear=False
                    )
                )
                self.assertFalse(build_stage2.should_run())

    def test_installed_wheels_beat_the_published_tags(self):
        # The bound is consulted only when the runtimes are absent: if they import they
        # are installable here, so a stale bound cannot drop kernels from a build that
        # has the wheels in hand.
        self.assertTrue(self._run_on((3, 15), True, missing=()))

    def test_missing_runtime_is_fatal(self):
        # A declared kernel that cannot be built must fail the build, not ship a slower
        # wheel -- enforced by require_runtimes, not by should_run, whose answer decides
        # whether to install those runtimes. The message must name INSTALLABLE
        # distributions: `pip install cutlass tvm_ffi` names unrelated PyPI projects.
        with (
            mock.patch.object(build_stage2, "_torch_probe", lambda e: False),
            mock.patch.object(
                toolchains.Toolchain,
                "missing_runtimes",
                classmethod(lambda cls: ["cutlass"]),
            ),
        ):
            with self.assertRaises(RuntimeError) as caught:
                build_stage2.require_runtimes()
            msg = str(caught.exception)
            # The DISTRIBUTIONS, which is what pip takes...
            self.assertIn("nvidia-cutlass-dsl", msg)
            self.assertIn("apache-tvm-ffi", msg)
            # ...which kind is short of what...
            self.assertIn("cutedsl needs cutlass", msg)
            # ...whose build it is, since a ROCm one demands nothing...
            self.assertIn("this cuda build", msg)
            # ...and the one way out that needs no wheel.
            self.assertIn("TORCH_NATIVE_AOT=0", msg)

    def test_installed_runtimes_are_demanded_of_nobody(self):
        # The happy path: with `gaps` unfiltered every registered kind appears in it
        # needing nothing, and the raise fires on a build that has the wheels.
        with mock.patch.object(
            toolchains.Toolchain, "missing_runtimes", classmethod(lambda cls: [])
        ):
            build_stage2.require_runtimes()

    def test_a_backend_with_no_toolchain_demands_nothing(self):
        # for_backend(), not the whole registry: a ROCm build exports nothing and must
        # not fail for want of the CUDA DSL wheels.
        with (
            mock.patch.object(build_stage2, "_torch_probe", lambda e: True),
            mock.patch.object(
                toolchains.Toolchain,
                "missing_runtimes",
                classmethod(lambda cls: ["cutlass"]),
            ),
        ):
            build_stage2.require_runtimes()

    def test_verdict_never_demands_the_runtimes_it_asks_for(self):
        # The verdict tells the CI shells whether to install the DSL wheels, so
        # requiring them here fails the run on the very runtimes it should request.
        self.assertTrue(self._run(self.CUDA, self.ARCH, missing=("cutlass",)))

    def _verdict(self, env):
        """(stdout, stderr) of `--print-verdict`, captured separately."""
        out, err = io.StringIO(), io.StringIO()
        # All-true would mean ROCm, which skips before the arch logic under test.
        probe = lambda e: e != "torch.version.hip is not None"  # noqa: E731
        with contextlib.ExitStack() as stack:
            # Through main() rather than _run, so it needs the same REPO redirect, or
            # the verdict depends on whether the checked-out commit declares anything.
            repo = stack.enter_context(tempfile.TemporaryDirectory())
            ops = os.path.join(repo, "torch", "_native", "ops")
            os.makedirs(os.path.join(ops, "fakeop"))
            open(os.path.join(ops, "fakeop", "aot.py"), "w").close()
            stack.enter_context(mock.patch.object(build_stage2, "REPO", repo))
            stack.enter_context(mock.patch.object(export, "OPS_DIR", ops))
            # ...and the same platform patch, since should_run checks sys.platform
            # first: without it this test failed on a non-Linux box.
            stack.enter_context(mock.patch.object(sys, "platform", "linux"))
            # ...and the same BUILD_DIR redirect, carrying the CUDA major: an empty
            # cache falls through to the installed torch.
            build = stack.enter_context(tempfile.TemporaryDirectory())
            with open(os.path.join(build, "CMakeCache.txt"), "w") as f:
                f.write("CUDAToolkit_VERSION_MAJOR:STRING=13\n")
            stack.enter_context(mock.patch.object(build_stage2, "BUILD_DIR", build))
            # ...and the same env neutralization as _run, applied before the caller's
            # env, which overrides it.
            stack.enter_context(
                mock.patch.dict(
                    os.environ,
                    {"TORCH_NATIVE_AOT": "", "TORCH_CUDA_ARCH_LIST": ""},
                    clear=False,
                )
            )
            stack.enter_context(mock.patch.object(build_stage2, "_torch_probe", probe))
            stack.enter_context(
                mock.patch.object(
                    toolchains.Toolchain,
                    "missing_runtimes",
                    classmethod(lambda cls: []),
                )
            )
            stack.enter_context(mock.patch.dict(os.environ, env, clear=False))
            stack.enter_context(contextlib.redirect_stdout(out))
            stack.enter_context(contextlib.redirect_stderr(err))
            build_stage2.main(["--print-verdict"])
        return out.getvalue(), err.getvalue()

    def test_verdict_stdout_carries_only_the_verdict(self):
        # The shells compare with ==, so any other line on stdout breaks them, and
        # multi-arch is the one case that reports and still proceeds.
        out, err = self._verdict({"TORCH_CUDA_ARCH_LIST": "9.0;10.0"})
        self.assertEqual(out, "RUN\n")
        self.assertIn("multi-arch", err)

    def test_verdict_stdout_is_clean_when_skipping(self):
        out, err = self._verdict({"TORCH_NATIVE_AOT": "0"})
        self.assertEqual(out, "SKIP\n")
        self.assertIn("TORCH_NATIVE_AOT=0", err)

    def test_skips_when_no_declarations_exist(self):
        # No aot.py in the tree means stage 2 would export nothing, so demanding
        # ~190MB of DSL wheels for it is wrong. It is also the state of a bisect.
        self.assertFalse(self._run(self.CUDA, self.ARCH, declarations=False))
        # ...and one declaration is enough to proceed.
        self.assertTrue(self._run(self.CUDA, self.ARCH, declarations=True))

    def test_torch_value_returns_the_marked_value(self):
        # The only place the marker protocol is pinned: keeping the marker in the value
        # would compare "NAOT_VALUE:sm_100" against EXPORTABLE_ARCHES and skip.
        for stdout, want in (
            ("NAOT_VALUE:2\n", "2"),
            ("NAOT_VALUE:sm_100\n", "sm_100"),
            # A line before the marker (a warning, a site hook) must not become
            # the value, and an empty value is still a value.
            ("some warning\nNAOT_VALUE:sm_90a\n", "sm_90a"),
            ("NAOT_VALUE:\n", ""),
        ):
            with self.subTest(stdout=stdout):
                done = subprocess.CompletedProcess([], 0, stdout=stdout, stderr="")
                with mock.patch.object(subprocess, "run", return_value=done):
                    self.assertEqual(build_stage2._torch_value("expr"), want)

    def test_torch_value_is_none_without_a_marker(self):
        # No marker means the expression never evaluated. Stubbed for the reason above:
        # `1 / 0` would import a real torch before it could raise.
        done = subprocess.CompletedProcess([], 1, stdout="", stderr="ZeroDivisionError")
        with mock.patch.object(subprocess, "run", return_value=done):
            self.assertIsNone(build_stage2._torch_value("1 / 0"))

    def test_probe_verdict_comes_from_stdout_not_the_exit_code(self):
        # Why the marker protocol exists: a CUDA torch can segfault in teardown after
        # the expression evaluated, corrupting a verdict taken from the exit code.
        import subprocess as sp

        crashed = sp.CompletedProcess(
            args=[], returncode=-11, stdout="PROBE_OK\n", stderr=""
        )
        with mock.patch.object(build_stage2.subprocess, "run", return_value=crashed):
            self.assertTrue(build_stage2._torch_probe("True"))
        lied = sp.CompletedProcess(
            args=[], returncode=0, stdout="PROBE_NO\n", stderr=""
        )
        with mock.patch.object(build_stage2.subprocess, "run", return_value=lied):
            self.assertFalse(build_stage2._torch_probe("True"))

    def test_probe_runs_outside_the_repo_root(self):
        # cwd=HERE, not the repo root: `python -c` puts the cwd on sys.path, so from
        # there every probe would answer about the source tree. Checked for every one.
        seen = []

        def fake_run(cmd, **kw):
            seen.append(kw.get("cwd"))
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout="PROBE_OK\nNAOT_VALUE:/x/torch\n",
                stderr="",
            )

        with mock.patch.object(build_stage2.subprocess, "run", fake_run):
            build_stage2._torch_probe("True")
            build_stage2._torch_value("1")
            build_stage2._installed_lib_dir()
        self.assertEqual(len(seen), 3, "all three probes should have run")
        for cwd in seen:
            self.assertEqual(cwd, build_stage2.HERE)
            self.assertNotEqual(cwd, build_stage2.REPO)

    def test_probe_diagnostics_stay_off_stdout(self):
        # The real subprocess probe, which every other test here mocks: a raising
        # expression leaves no verdict and a non-empty stderr.
        import contextlib
        import io

        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            self.assertFalse(build_stage2._torch_probe("1 / 0"))
        self.assertEqual(out.getvalue(), "", "diagnostics must not reach stdout")
        self.assertIn("produced no verdict", err.getvalue())

    def test_on_device_export_checks_the_local_arch(self):
        # Without this gate a dev box outside EXPORTABLE_ARCHES exports for its own arch
        # and fails in generation, leaving a tree that will not configure.
        for local, expected in (("sm_86", False), ("sm_120", False), ("sm_100", True)):
            with self.subTest(local=local):
                with (
                    mock.patch.object(
                        build_stage2, "_torch_value", lambda expr, a=local: a
                    ),
                    contextlib.redirect_stderr(io.StringIO()) as err,
                ):
                    self.assertEqual(self._run(self.CUDA), expected)
                # ...and the reason names the arch it found, since the fix is to
                # add it to the arch list or to accept the JIT path.
                if not expected:
                    self.assertIn(f"local GPU is {local}", err.getvalue())

    def test_on_device_arch_is_read_out_of_process(self):
        # Via _torch_value, not export._detected_arch(): that would initialize CUDA in
        # the driver process, where a teardown segfault fails a stage 2 that succeeded.
        with (
            mock.patch.object(export, "_detected_arch", side_effect=AssertionError),
            mock.patch.object(build_stage2, "_torch_value", lambda expr: "sm_100"),
        ):
            self.assertTrue(self._run(self.CUDA))

    def test_disabled_by_env(self):
        # Inputs that otherwise RUN, with the RUN itself as the control: at the
        # all-true probe default these would decline at the ROCm arm instead.
        self.assertFalse(self._run(self.CUDA, {**self.ARCH, "TORCH_NATIVE_AOT": "0"}))
        self.assertTrue(self._run(self.CUDA, self.ARCH))

    def test_skips_when_torch_not_importable(self):
        self.assertFalse(self._run({**self.CUDA, "True": False}, self.ARCH))
        self.assertTrue(self._run(self.CUDA, self.ARCH))

    def test_skips_when_torch_built_without_cuda(self):
        no_cuda = {**self.CUDA, "torch.backends.cuda.is_built()": False}
        self.assertFalse(self._run(no_cuda, self.ARCH))
        self.assertTrue(self._run(self.CUDA, self.ARCH))

    def test_skips_on_rocm_with_no_rocm_toolchain(self):
        # ROCm has no AOT toolchain, so absent DSL wheels are expected there
        # rather than a missing dependency.
        self.assertFalse(self._run({"torch.version.hip is not None": True}, self.ARCH))

    def test_every_skip_names_its_own_reason(self):
        # The report is the only trace a kernel-free wheel leaves. One row per arm,
        # asserting its own phrase and the absence of every other row's.
        cases = {
            "built torch not importable": ({**self.CUDA, "True": False}, self.ARCH, {}),
            "torch built without CUDA": (
                {**self.CUDA, "torch.backends.cuda.is_built()": False},
                self.ARCH,
                {},
            ),
            "no AOT toolchain targets rocm": (
                {"torch.version.hip is not None": True},
                self.ARCH,
                {},
            ),
            "no declarations under torch/_native/ops": (
                self.CUDA,
                self.ARCH,
                {"declarations": False},
            ),
            "has no exportable arch": (self.CUDA, {"TORCH_CUDA_ARCH_LIST": "8.6"}, {}),
            "no local GPU to detect from": (
                {**self.CUDA, "torch.cuda.is_available()": False},
                {},
                {},
            ),
        }
        for phrase, (probes, env, kwargs) in cases.items():
            with self.subTest(phrase=phrase):
                with contextlib.redirect_stderr(io.StringIO()) as err:
                    self.assertFalse(self._run(probes, env, **kwargs))
                reported = err.getvalue()
                self.assertIn(phrase, reported)
                for other in cases:
                    if other != phrase:
                        self.assertNotIn(other, reported)

    def test_cuda_12_skips(self):
        # CUDA 12 tops out at sm_90 and every 13.x config builds sm_90 too, so a 12.x
        # export is a strict subset of what the 13.x wheels ship. The saving includes
        # the wheel install: build.sh calls install_cutlass_dsl only on RUN.
        for major in ("11", "12"):
            with self.subTest(major=major):
                self.assertFalse(self._run(self.CUDA, self.ARCH, cuda_major=major))
        # ...and the same inputs on 13 and on a future 14 RUN, so this is about the
        # major and not some other arm firing first.
        for major in ("13", "14"):
            with self.subTest(major=major):
                self.assertTrue(self._run(self.CUDA, self.ARCH, cuda_major=major))

    def test_an_undeterminable_cuda_major_skips(self):
        # Too old rather than "assume new enough": _dsl_runtime_archive() cannot pick a
        # per-major runtime without it, and guessing links one for another toolkit.
        with mock.patch.object(build_stage2, "_torch_value", lambda expr: ""):
            self.assertFalse(self._run(self.CUDA, self.ARCH, cuda_major=None))

    def test_the_cuda_major_skip_reason_names_the_version(self):
        with contextlib.redirect_stderr(io.StringIO()) as err:
            self._run(self.CUDA, self.ARCH, cuda_major="12")
        self.assertIn("CUDA 12", err.getvalue())
        self.assertIn(f"CUDA {build_stage2._MIN_CUDA_MAJOR} or newer", err.getvalue())

    def test_rocm_is_not_gated_on_a_cuda_major(self):
        # ROCm reports no CUDA version, so a major gate would skip it for a reason that
        # cannot apply. A fake ROCm toolchain is registered because otherwise should_run
        # returns before the gate and the test passes with the backend guard deleted.
        class _RocmToolchain(toolchains.Toolchain):
            kind = "fakerocm"
            BACKENDS = ("rocm",)

        # _torch_value too: with no cache entry the major falls through to the installed
        # torch, which on a CUDA box reports 13.
        with (
            mock.patch.dict(
                toolchains.TOOLCHAINS, {"fakerocm": _RocmToolchain()}, clear=False
            ),
            mock.patch.object(build_stage2, "_torch_value", lambda expr: ""),
        ):
            self.assertTrue(
                self._run(
                    {"torch.version.hip is not None": True}, self.ARCH, cuda_major=None
                )
            )

    def test_skips_when_arch_list_has_no_exportable_arch(self):
        # 8.0 and 7.5 are below the kernels' floor (TMA, clusters), so nothing
        # to export. 9.0a IS exportable now, hence not in this list.
        self.assertFalse(self._run(self.CUDA, {"TORCH_CUDA_ARCH_LIST": "7.5;8.0"}))

    def test_multi_exportable_arch_runs(self):
        # The real list: .ci/manywheel builds x86_64 CUDA 13.x with these, and they
        # resolve to two capabilities, so every release wheel takes this path.
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            ran = self._run(
                self.CUDA, {"TORCH_CUDA_ARCH_LIST": "7.5;8.0;8.6;9.0;10.0;12.0"}
            )
        self.assertTrue(ran)
        # The report is the branch's only observable, and on stderr: this is the one
        # gate that reports AND proceeds, where stdout carries the verdict.
        self.assertIn("multi-arch: sm_90 sm_100", err.getvalue())
        self.assertEqual(out.getvalue(), "")

    def test_both_spellings_of_one_capability_run_as_one_arch(self):
        # The list a builder writes wanting the arch-conditional kernels and naming the
        # plain arch too: it must run, and as one arch, or the embedded bytes double.
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            ran = self._run(self.CUDA, {"TORCH_CUDA_ARCH_LIST": "10.0;10.0a"})
        self.assertTrue(ran)
        self.assertNotIn("multi-arch", err.getvalue())

    def test_runs_for_a_single_exportable_arch(self):
        self.assertTrue(self._run(self.CUDA, self.ARCH))

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

    def test_the_gate_bounds_dims_only_and_not_numel(self):
        # The contract is "every dim the CALLER passes must fit": bounding numel() would
        # decline (2**28, 8), whose derived extent is 8 and is served correctly.
        gate = gen_aot_lib._int32_size_gate(
            "const at::Tensor & self, const ::std::optional<at::Tensor> & weight"
        )
        self.assertIn("self.sizes()", gate)
        self.assertNotIn("numel()", gate)

    def test_unhandled_tensor_like_types_are_refused(self):
        # torchgen renders Tensor? as at::OptionalTensorRef and Tensor[] as
        # at::ITensorListRef, neither of which fits the accessors the gate emits.
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
        # <prefix>_* covers every symbol the DSL emits for the kernel; enumerating known
        # suffixes leaves _args_spec, _kernel_info and friends in torch_cuda's ABI.
        self.assertIn("topk_f32_n1024_k8_*;", text)
        self.assertIn("_mlir_*topk_f32_n1024_k8*;", text)
        # The DIRECTIVE line, not the word, which VER_TMPL's own comment also contains:
        # `global:` here exports every DSL symbol, the inverse of the file's purpose.
        self.assertIn("\n  local:\n", text)
        self.assertNotIn("\n  global:\n", text)
        # EVERY pattern names the prefix, so this fails for any that would reach past
        # this kernel's symbols. Indented and ";"-terminated selects the pattern lines,
        # not the script's closing "};" or its comment block.
        patterns = [
            l.strip()
            for l in text.splitlines()
            if l.startswith("    ") and l.strip().endswith(";")
        ]
        self.assertTrue(patterns)
        for p in patterns:
            self.assertIn("topk_f32_n1024_k8", p, f"unanchored pattern: {p}")


class TestBuildInputsFromTheCMakeCache(unittest.TestCase):
    """CMake resolves TORCH_CUDA_ARCH_LIST and TORCH_NATIVE_AOT from its own
    variables first and the environment only as their default, so a build
    configured with -D... has neither in the environment. Reading only the
    environment made stage 2 record ARCH_LIST_ABSENT for those builds -- which
    caffe2/CMakeLists.txt treats as "no claim, embed what is there" -- so the
    staleness guard was dead exactly where -D is used."""

    @contextlib.contextmanager
    def _build_dir(self, cache_text):
        with tempfile.TemporaryDirectory() as d:
            if cache_text is not None:
                with open(os.path.join(d, "CMakeCache.txt"), "w") as f:
                    f.write(cache_text)
            with (
                mock.patch.object(build_stage2, "BUILD_DIR", d),
                mock.patch.dict(
                    os.environ,
                    {"TORCH_CUDA_ARCH_LIST": "", "TORCH_NATIVE_AOT": ""},
                    clear=False,
                ),
            ):
                yield d

    def _record(self, d, text):
        """Write what a configure would have recorded, where CMake writes it."""
        path = os.path.join(d, build_stage2.ARCH_LIST_RECORD)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(text)

    def test_the_recorded_arch_list_outranks_both_env_and_cache(self):
        # The case that needs the record: -D and the environment disagree, and reading
        # either alone targets an arch the build did not compile for.
        with self._build_dir("TORCH_CUDA_ARCH_LIST:STRING=8.0\n") as d:
            self._record(d, "9.0a")
            with mock.patch.dict(os.environ, {"TORCH_CUDA_ARCH_LIST": "10.0a"}):
                self.assertEqual(build_stage2._arch_list(), "9.0a")

    def test_a_recorded_empty_arch_list_is_not_a_fallback(self):
        # The configure resolving none is an ANSWER: falling back would target an arch
        # from a stale cache or a developer's shell.
        with self._build_dir("TORCH_CUDA_ARCH_LIST:STRING=8.0\n") as d:
            self._record(d, "")
            with mock.patch.dict(os.environ, {"TORCH_CUDA_ARCH_LIST": "10.0a"}):
                self.assertEqual(build_stage2._arch_list(), "")

    def test_a_recorded_list_is_stripped_of_its_newline(self):
        # file(WRITE) content as CMake leaves it, plus whatever an editor adds:
        # archs_from_cuda_arch_list would read "10.0a\n" as an unparsable entry.
        with self._build_dir(None) as d:
            self._record(d, "9.0a;10.0a\n")
            self.assertEqual(build_stage2._arch_list(), "9.0a;10.0a")

    def test_the_configure_records_what_this_reads(self):
        # The WRITER half, pinned by text since this suite cannot run cmake: the reader
        # falls back only when the file is absent, so a rename either side restores the
        # env-then-cache guessing. Written after Dependencies.cmake shadows the cache.
        with open(os.path.join(REPO, "cmake", "Codegen.cmake")) as f:
            cmake = f.read()
        record = build_stage2.ARCH_LIST_RECORD.replace(os.sep, "/")
        self.assertIn(
            f'file(WRITE "${{CMAKE_BINARY_DIR}}/{record}" "${{TORCH_CUDA_ARCH_LIST}}")',
            cmake,
        )
        # Unconditional: guarded on CUDA, a previous CUDA configure's list would stay
        # in place for a CPU one and stage 2 would read that.
        head = cmake.split(f'file(WRITE "${{CMAKE_BINARY_DIR}}/{record}"')[0]
        self.assertNotIn("if(USE_CUDA", head.rsplit("if(", 1)[-1])

    def test_arch_list_comes_from_the_cache_when_the_env_is_unset(self):
        cache = (
            "//From env\n"
            "TORCH_CUDA_ARCH_LIST:STRING=9.0a;10.0a\n"
            "CMAKE_BUILD_TYPE:STRING=Release\n"
        )
        with self._build_dir(cache):
            self.assertEqual(build_stage2._arch_list(), "9.0a;10.0a")

    def test_the_environment_still_wins(self):
        with self._build_dir("TORCH_CUDA_ARCH_LIST:STRING=8.0\n"):
            with mock.patch.dict(os.environ, {"TORCH_CUDA_ARCH_LIST": "10.0a"}):
                self.assertEqual(build_stage2._arch_list(), "10.0a")

    def test_no_cache_and_no_env_is_empty(self):
        # The on-device path: no arch list at all, rather than a crash or a value
        # inherited from some other build tree.
        with self._build_dir(None):
            self.assertEqual(build_stage2._arch_list(), "")

    def test_a_similarly_named_entry_is_not_mistaken_for_it(self):
        # Prefix matching would read TORCH_CUDA_ARCH_LIST_EXTRA here.
        with self._build_dir("TORCH_CUDA_ARCH_LIST_EXTRA:STRING=7.5\n"):
            self.assertEqual(build_stage2._arch_list(), "")

    def test_a_duplicated_entry_is_read_the_way_cmake_reads_it(self):
        # CMake honours the LAST assignment, and appending a line to flip a setting is
        # normal; reading the first has stage 2 relink for a build that embeds nothing.
        with self._build_dir("TORCH_NATIVE_AOT:BOOL=1\nTORCH_NATIVE_AOT:BOOL=0\n"):
            self.assertEqual(build_stage2._cmake_cache_value("TORCH_NATIVE_AOT"), "0")
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertTrue(build_stage2._opted_out())

    def test_the_opt_out_is_honored_from_the_cache(self):
        # caffe2/CMakeLists.txt caches TORCH_NATIVE_AOT so the opt-out survives a
        # reconfigure, so reading only the environment relinks a library CMake declined
        # to embed anything into.
        with self._build_dir("TORCH_NATIVE_AOT:STRING=0\n"):
            with contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertTrue(build_stage2._opted_out())
            # Reported as coming from the CACHE, with the override: the value is
            # invisible to anyone reading their own environment.
            self.assertIn("CMakeCache.txt", err.getvalue())
            self.assertIn("TORCH_NATIVE_AOT=1", err.getvalue())
        with self._build_dir("TORCH_NATIVE_AOT:STRING=1\n"):
            self.assertFalse(build_stage2._opted_out())

    def test_the_environment_can_re_enable_it(self):
        with self._build_dir("TORCH_NATIVE_AOT:STRING=0\n"):
            with mock.patch.dict(os.environ, {"TORCH_NATIVE_AOT": "1"}):
                self.assertFalse(build_stage2._opted_out())


class TestDslRuntimeArchive(unittest.TestCase):
    """Which dialect runtime the CuTeDSL kernel objects link against.

    4.6.x splits the archive per CUDA major (cu12/lib/, cu13/lib/), so the lookup
    has to choose by the build's toolkit rather than take whichever directory it
    finds first. These fixtures are empty files: what is under test is the
    SELECTION, not the archives' contents. It is the one lookup in stage 2 that can
    fail a build."""

    ARCHIVE = "libcuda_dialect_runtime_static.a"

    @contextlib.contextmanager
    def _wheel(self, subdirs, cuda_major="13", env_dir="venv"):
        """A fake nvidia_cutlass_dsl tree with an archive under each of subdirs
        ("" for the pre-4.6 unsplit layout).

        ``env_dir`` names the directory the package sits in, which a caller can spell
        cu12/cu13 to stand in for a venv or conda environment named after a CUDA
        version -- the archive match must not see it."""
        with tempfile.TemporaryDirectory() as d:
            root = os.path.join(d, env_dir, "site-packages", "nvidia_cutlass_dsl")
            bld = os.path.join(d, "build")
            os.makedirs(root)
            os.makedirs(bld)
            for sub in subdirs:
                lib = os.path.join(root, sub, "lib")
                os.makedirs(lib)
                open(os.path.join(lib, self.ARCHIVE), "w").close()
            if cuda_major is not None:
                with open(os.path.join(bld, "CMakeCache.txt"), "w") as f:
                    f.write(f"CUDAToolkit_VERSION_MAJOR:STRING={cuda_major}\n")
            spec = mock.Mock(submodule_search_locations=[root])
            with (
                mock.patch.object(build_stage2, "BUILD_DIR", bld),
                mock.patch("importlib.util.find_spec", return_value=spec),
            ):
                yield root

    def test_picks_the_archive_for_this_major(self):
        with self._wheel(("cu12", "cu13"), cuda_major="13") as root:
            got = build_stage2._dsl_runtime_archive()
        self.assertEqual(got, os.path.join(root, "cu13", "lib", self.ARCHIVE))

    def test_a_wheel_with_no_archive_for_this_major_warns_and_links_one(self):
        # cu12 is a hard dependency and cu13 behind an extra, so a plain install on a
        # CUDA 13 build leaves only cu12: the contract is warn-and-link.
        with self._wheel(("cu12",), cuda_major="13") as root:
            with contextlib.redirect_stderr(io.StringIO()) as err:
                got = build_stage2._dsl_runtime_archive()
        self.assertEqual(got, os.path.join(root, "cu12", "lib", self.ARCHIVE))
        self.assertIn("no dialect runtime for CUDA 13", err.getvalue())
        self.assertIn("cu13", err.getvalue(), "the warning should name the extra")

    def test_the_mismatch_fallback_takes_the_highest_major(self):
        # Deterministic, so two builds of one environment link the same file.
        with self._wheel(("cu12", "cu13"), cuda_major="14") as root:
            with contextlib.redirect_stderr(io.StringIO()):
                got = build_stage2._dsl_runtime_archive()
        self.assertEqual(got, os.path.join(root, "cu13", "lib", self.ARCHIVE))

    def test_a_matching_major_warns_about_nothing(self):
        # Absence of THIS warning, not an empty stream, which couples the assertion to
        # unrelated output.
        with self._wheel(("cu12", "cu13"), cuda_major="13"):
            with contextlib.redirect_stderr(io.StringIO()) as err:
                build_stage2._dsl_runtime_archive()
        self.assertNotIn("no dialect runtime", err.getvalue())

    def test_the_cache_answering_skips_the_torch_probe(self):
        # The torch subprocess is the expensive probe and the only one that can fail
        # noisily; a configured build never needs it.
        with self._wheel(("cu13",), cuda_major="13"):
            with mock.patch.object(
                build_stage2, "_torch_value", side_effect=AssertionError("probed torch")
            ):
                self.assertEqual(build_stage2._cuda_major(), 13)

    def test_an_unsplit_wheel_is_taken_as_is(self):
        # Pre-4.6 shipped one archive at <root>/lib/ with no major in the path.
        with self._wheel(("",), cuda_major="13") as root:
            got = build_stage2._dsl_runtime_archive()
        self.assertEqual(got, os.path.join(root, "lib", self.ARCHIVE))

    def test_no_archive_at_all_is_none_not_an_error(self):
        # No archive means this build embeds no CuTeDSL objects, so the generator emits
        # CMake with no runtime to link -- correct for a Triton-only export.
        with self._wheel(()):
            self.assertIsNone(build_stage2._dsl_runtime_archive())

    def test_no_wheel_is_none(self):
        with mock.patch("importlib.util.find_spec", return_value=None):
            self.assertIsNone(build_stage2._dsl_runtime_archive())

    def test_the_environments_own_directory_name_does_not_select_the_major(self):
        # The match reads components RELATIVE to the package root: on the absolute path
        # an environment named cu12/cu13 matches both archives and walk order decides.
        for env_dir in ("cu12", "cu13"):
            with self.subTest(env_dir=env_dir):
                with self._wheel(("cu12", "cu13"), env_dir=env_dir) as root:
                    got = build_stage2._dsl_runtime_archive()
                self.assertEqual(got, os.path.join(root, "cu13", "lib", self.ARCHIVE))

    def test_an_unsplit_wheel_under_a_cu_named_environment_is_still_unsplit(self):
        # The same absolute-path hazard's other face: a pre-4.6 wheel, whose one archive
        # has no major in its path, must not look split.
        with self._wheel(("",), env_dir="cu12") as root:
            got = build_stage2._dsl_runtime_archive()
        self.assertEqual(got, os.path.join(root, "lib", self.ARCHIVE))


class TestCudaMajor(unittest.TestCase):
    """The >=13 gate and the archive lookup share one answer for "which CUDA is
    this build", so it is read in one place."""

    @contextlib.contextmanager
    def _cache(self, text, torch_reports=""):
        reported = torch_reports
        with tempfile.TemporaryDirectory() as d:
            if text is not None:
                with open(os.path.join(d, "CMakeCache.txt"), "w") as f:
                    f.write(text)
            with (
                mock.patch.object(build_stage2, "BUILD_DIR", d),
                mock.patch.object(build_stage2, "_torch_value", lambda e: reported),
            ):
                yield

    def test_reads_the_toolkit_major_from_the_cache(self):
        with self._cache("CUDAToolkit_VERSION_MAJOR:STRING=13\n"):
            self.assertEqual(build_stage2._cuda_major(), 13)

    def test_falls_back_to_the_full_cached_version(self):
        with self._cache("CUDA_VERSION:STRING=13.2\n"):
            self.assertEqual(build_stage2._cuda_major(), 13)

    def test_falls_back_to_the_installed_torch(self):
        # An unconfigured tree, or a manual stage-2 run against an installed wheel.
        with self._cache(None, torch_reports="12.6"):
            self.assertEqual(build_stage2._cuda_major(), 12)

    def test_no_source_is_none(self):
        with self._cache(None):
            self.assertIsNone(build_stage2._cuda_major())

    def test_a_non_numeric_value_does_not_become_a_major(self):
        # _torch_value answers with an arch string in two of the should_run tests,
        # and int("sm_100") would raise rather than fall through.
        with self._cache("CUDAToolkit_VERSION_MAJOR:STRING=\n", torch_reports="sm_100"):
            self.assertIsNone(build_stage2._cuda_major())


class TestProbeDiagnostics(unittest.TestCase):
    """A probe that produces no verdict must say why: the caller degrades to a
    skip, and a silent one reads as a confident answer about the build."""

    def _probe(self, returncode, stderr):
        done = subprocess.CompletedProcess([], returncode, stdout="", stderr=stderr)
        with (
            mock.patch.object(subprocess, "run", return_value=done),
            contextlib.redirect_stderr(io.StringIO()) as err,
        ):
            build_stage2._torch_probe("torch.backends.cuda.is_built()")
        return err.getvalue()

    def test_a_signal_killed_probe_is_reported(self):
        # An OOM kill or an import-time segfault writes neither stream, so a guard on
        # stderr alone reports nothing and the caller logs a confident wrong reason.
        out = self._probe(-9, "")
        self.assertIn("no verdict", out)
        self.assertIn("signal 9", out)

    def test_the_childs_stderr_is_still_forwarded(self):
        out = self._probe(1, "ImportError: libcudart.so.13: cannot open")
        self.assertIn("libcudart.so.13", out)
        self.assertIn("exit 1", out)

    def test_diagnostics_stay_off_stdout(self):
        done = subprocess.CompletedProcess([], -11, stdout="", stderr="")
        with (
            mock.patch.object(subprocess, "run", return_value=done),
            contextlib.redirect_stdout(io.StringIO()) as out,
            contextlib.redirect_stderr(io.StringIO()),
        ):
            build_stage2._torch_probe("True")
        # --print-verdict writes a machine-read word to stdout and the CI shells
        # compare it with ==.
        self.assertEqual(out.getvalue(), "")


class TestInstalledLibDir(unittest.TestCase):
    def test_the_path_comes_from_a_marked_line(self):
        # Unmarked, any other line the child writes -- a warning, a site hook --
        # becomes part of a filesystem path.
        done = subprocess.CompletedProcess(
            [], 0, stdout="a warning\nNAOT_VALUE:/x/site-packages/torch\n", stderr=""
        )
        with mock.patch.object(subprocess, "run", return_value=done):
            self.assertEqual(
                build_stage2._installed_lib_dir(),
                os.path.join("/x/site-packages/torch", "lib"),
            )

    def test_a_probe_that_cannot_run_at_all_is_still_framed(self):
        # The SPAWN, not the child: a fork returning EAGAIN after a 32-way relink would
        # surface as a bare OSError, and with no timeout the call hangs the build.
        with (
            mock.patch.object(
                subprocess, "run", side_effect=BlockingIOError(11, "try again")
            ),
            contextlib.redirect_stderr(io.StringIO()) as err,
        ):
            with self.assertRaisesRegex(RuntimeError, "cannot locate the installed"):
                build_stage2._installed_lib_dir()
        self.assertIn("could not run", err.getvalue())

    def test_a_failure_reports_the_childs_stderr(self):
        # The traceback was captured and thrown away -- after a full export,
        # generation and relink.
        done = subprocess.CompletedProcess(
            [], 1, stdout="", stderr="ImportError: libcudart.so.13"
        )
        with (
            mock.patch.object(subprocess, "run", return_value=done),
            contextlib.redirect_stderr(io.StringIO()) as err,
        ):
            with self.assertRaisesRegex(RuntimeError, "cannot locate the installed"):
                build_stage2._installed_lib_dir()
        self.assertIn("libcudart.so.13", err.getvalue())


class TestProbeSpawnFailures(unittest.TestCase):
    """A probe that never runs must degrade like one that answers: should_run()
    prints a word the CI shells compare with ==, so a traceback there is neither
    RUN nor SKIP -- they skip the DSL install and the real stage-2 run demands
    it."""

    def _both_probes(self, exc):
        with (
            mock.patch.object(subprocess, "run", side_effect=exc),
            contextlib.redirect_stdout(io.StringIO()) as out,
            contextlib.redirect_stderr(io.StringIO()) as err,
        ):
            verdict = build_stage2._torch_probe("True")
            value = build_stage2._torch_value("1")
        return verdict, value, out.getvalue(), err.getvalue()

    def test_a_spawn_that_fails_is_a_skip_not_a_traceback(self):
        # The SPAWN, not the child: fork returns EAGAIN at the end of a MAX_JOBS build,
        # and sys.executable can be absent when stage 2 runs from a wrapper.
        for exc in (
            BlockingIOError(11, "Resource temporarily unavailable"),
            FileNotFoundError(2, "No such file or directory"),
        ):
            with self.subTest(exc=type(exc).__name__):
                verdict, value, out, err = self._both_probes(exc)
                self.assertFalse(verdict)
                self.assertIsNone(value)
                self.assertEqual(out, "", "diagnostics must not reach stdout")
                self.assertIn("could not run", err)

    def test_a_wedged_probe_is_reported_like_any_other(self):
        verdict, value, out, err = self._both_probes(
            subprocess.TimeoutExpired(cmd=["python"], timeout=1)
        )
        self.assertFalse(verdict)
        self.assertIsNone(value)
        self.assertIn("could not run", err)

    def test_both_probes_bound_the_wait(self):
        # The timeout has to be passed, not merely handled: `import torch` against a
        # wedged driver never returns, and the build hangs to its step limit.
        seen = []

        def fake_run(cmd, **kw):
            seen.append(kw.get("timeout"))
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout="PROBE_OK\nNAOT_VALUE:2\n", stderr=""
            )

        with mock.patch.object(build_stage2.subprocess, "run", fake_run):
            build_stage2._torch_probe("True")
            build_stage2._torch_value("1")
            # ...including the probe that locates the installed torch: find_spec
            # imports the parent package, so that call is a full `import torch` too.
            build_stage2._installed_lib_dir()
        self.assertEqual(seen, [build_stage2._PROBE_TIMEOUT_SECONDS] * 3)


class TestRegistryConsistency(unittest.TestCase):
    def test_artifact_exts_are_shared_by_both_sweeps(self):
        # One notion of "kernel artifact", through both sweeps with a toolchain neither
        # knew about: asserting the union contains each kind's exts is a tautology.
        class _Novel(toolchains.Toolchain):
            kind = "novel"
            artifact_exts = (".novelobj",)

        with mock.patch.dict(toolchains.TOOLCHAINS, {"novel": _Novel()}, clear=False):
            self.assertIn(".novelobj", toolchains.all_artifact_exts())
            # Sweep 1: export's orphan check, which REPORTS an artifact no sidecar
            # claims; naming it is what proves the shared set.
            with tempfile.TemporaryDirectory() as d:
                open(os.path.join(d, "k.novelobj"), "w").close()
                with contextlib.redirect_stdout(io.StringIO()) as said:
                    export._check_no_orphan_artifacts(d, [])
                self.assertIn("k.novelobj", said.getvalue())

            # Sweep 2: generation's no-declaration check. "other" is declared so
            # by_id is non-empty; an empty one means the tree declares nothing.
            class _OtherDecl(_FakeDecl):
                ATEN_OP = "other"

            with tempfile.TemporaryDirectory() as tmpdir:
                art = os.path.join(tmpdir, "sm_100a", "fakeop")
                os.makedirs(art)
                open(os.path.join(art, "k.novelobj"), "w").close()
                ops = os.path.join(tmpdir, "_ops")
                os.makedirs(os.path.join(ops, "other"))
                open(os.path.join(ops, "other", "aot.py"), "w").close()
                with (
                    mock.patch.object(gen_aot_lib, "OPS_DIR", ops),
                    mock.patch.object(
                        gen_aot_lib.decl,
                        "load_declarations",
                        return_value=[_OtherDecl],
                    ),
                ):
                    out = io.StringIO()
                    with contextlib.redirect_stdout(out):
                        gen_aot_lib.main(["--artifacts-dir", tmpdir])
                    # Reported, per arch directory, and NOT fatal: see
                    # TestOrphanArtifactSafety for why.
                    self.assertIn("no declaration", out.getvalue())

    def test_link_exts_must_be_a_subset_of_artifact_exts(self):
        # Generation links `if ext in link_exts`, so an ext the kind cannot produce
        # contributes nothing: a green link, then an undefined symbol at first call.
        class _Bad(toolchains.Toolchain):
            kind = "bad"
            artifact_exts = (".cubin", ".h")
            link_exts = (".o",)

        with self.assertRaisesRegex(RuntimeError, "not a subset"):
            toolchains._assert_link_exts_are_exportable({"bad": _Bad()})

    def test_generation_and_export_look_for_declarations_in_one_place(self):
        # Two modules spell this path independently: diverged, every artifact looks
        # undeclared and the sweep refuses a correct build.
        self.assertEqual(gen_aot_lib.OPS_DIR, export.OPS_DIR)

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


class TestSpecRoundTrip(unittest.TestCase):
    def test_tuple_grid_values_reach_the_declaration_as_tuples(self):
        # The grid distinguishes a list (an axis) from a tuple (one compound value) where
        # JSON has only arrays, so unrestored a tuple-valued cond never fires.
        restored = gen_aot_lib._spec_from_json(
            {"dtypes": ["f32", "bf16"], "N": 1024, "pairs": [[1, 2], [3, 4]]}
        )
        self.assertEqual(restored["dtypes"], ("f32", "bf16"))
        self.assertEqual(restored["pairs"], ((1, 2), (3, 4)))
        self.assertEqual(restored["N"], 1024)

    def test_a_tuple_spec_survives_the_sidecar_round_trip(self):
        # The restore is exact rather than a guess: expand_specs consumed every
        # list as an axis, so a sequence still in a recorded point was a tuple.
        (point,) = export.expand_specs([{"dtypes": ("f32", "bf16"), "N": 1024}])
        self.assertEqual(point["dtypes"], ("f32", "bf16"))
        recorded = json.loads(json.dumps(export._json_normal(point)))
        self.assertEqual(recorded["dtypes"], ["f32", "bf16"])
        self.assertEqual(gen_aot_lib._spec_from_json(recorded), point)

    def test_the_generator_hands_cpp_dispatch_the_restored_spec(self):
        # The CALL SITE, not just _spec_from_json: dropping the restore in gen_op leaves
        # the direct test above green while every tuple-valued cond stops firing.
        seen = {}

        class _TupleDecl(_FakeDecl):
            @staticmethod
            def cpp_dispatch(spec):
                seen.update(spec)
                return "true"

            @staticmethod
            def cpp_launch(spec, launch_fn):
                return launch_fn + "();"

        sc = dict(
            SIDECAR,
            spec=json.loads(json.dumps({"dtypes": ["f32", "bf16"], "N": 1, "K": 8})),
            _dir="/art",
        )
        gen_aot_lib.gen_op(
            op="fakeop",
            key="CUDA",
            d=_TupleDecl,
            sidecars=[sc],
            impl_params="const at::Tensor & self, int64_t k",
        )
        self.assertEqual(seen["dtypes"], ("f32", "bf16"))


class TestClaimedSpellingPreference(unittest.TestCase):
    def test_the_conditional_spelling_wins_when_both_are_claimed(self):
        # Alphabetical order would pick sm_100 over sm_100a, and the generator's
        # tie-break keeps the conditional one, so the plain build compiles and loses.
        both = ("sm_100", "sm_100a", "sm_90", "sm_90a")
        self.assertEqual(export._claimed_spelling("sm_100", both), "sm_100a")
        self.assertEqual(export._claimed_spelling("sm_100a", both), "sm_100a")
        self.assertEqual(export._claimed_spelling("sm_90", both), "sm_90a")

    def test_a_capability_that_is_not_claimed_has_no_spelling(self):
        self.assertIsNone(export._claimed_spelling("sm_103", ("sm_100a",)))

    def test_the_plain_spelling_is_used_when_it_is_the_only_claim(self):
        # A declaration pinning the plain build must not be handed a conditional
        # target its kernels were not written for.
        self.assertEqual(export._claimed_spelling("sm_100a", ("sm_100",)), "sm_100")


class TestSidecarSchemaIsReadFirst(unittest.TestCase):
    """gen_aot_lib refuses a mismatched sidecar SCHEMA before reading any field,
    because a different version need not mean the same thing by a name. The skip
    check has to do the same: it read sc["kind"] first and raised a bare KeyError,
    naming no file and no remedy, where the version field exists precisely so the
    point re-exports."""

    def _job(self, d, sidecar):
        _touch_artifacts(d, "x")
        with open(os.path.join(d, "x.json"), "w") as f:
            json.dump(sidecar, f)
        return (
            "fakeop",
            "aot_kernel.py",
            {"dtype": "float32", "N": 4096},
            d,
            "sm_100a",
        )

    def _sidecar(self, **over):
        sc = {
            "version": export.SIDECAR_VERSION,
            "prefix": "x",
            "kind": "cutedsl",
            "spec": {"dtype": "float32", "N": 4096},
            "arch": "sm_100a",
            "sources": _current_sources(),
            "runtimes": _RUNTIMES,
        }
        sc.update(over)
        return sc

    def test_a_matching_sidecar_still_skips(self):
        # The control: without it, "re-exports" below could be for any reason.
        with tempfile.TemporaryDirectory() as d:
            job = self._job(d, self._sidecar())
            with _no_ambient_arch():
                self.assertFalse(export._job_needed(job, force=False))

    def test_a_sidecar_from_another_schema_re_exports(self):
        with tempfile.TemporaryDirectory() as d:
            job = self._job(d, self._sidecar(version=export.SIDECAR_VERSION + 1))
            with _no_ambient_arch():
                self.assertTrue(export._job_needed(job, force=False))

    def test_a_sidecar_without_a_kind_re_exports_instead_of_raising(self):
        sc = self._sidecar()
        del sc["kind"]
        with tempfile.TemporaryDirectory() as d:
            job = self._job(d, sc)
            with _no_ambient_arch():
                self.assertTrue(export._job_needed(job, force=False))


class TestSourceClosureCoversVendoredKernels(unittest.TestCase):
    def test_a_loaded_vendored_module_is_hashed(self):
        # torch/_vendor/ holds kernel bodies a declaration's build() shares with its JIT
        # wrapper, and the probe file sits where only the PREFIX reaches it: under
        # tools/native_aot/ the glob would hash it whatever the prefixes say.
        with tempfile.TemporaryDirectory() as root:
            here = os.path.join(root, "tools", "native_aot")
            vendored = os.path.join(root, "torch", "_vendor", "quack")
            os.makedirs(here)
            os.makedirs(vendored)
            path = os.path.join(vendored, "rmsnorm.py")
            with open(path, "w") as f:
                f.write("# a stand-in for a vendored kernel body\n")
            mod = types.ModuleType("torch._vendor.quack.rmsnorm")
            mod.__file__ = path
            with (
                mock.patch.object(export, "REPO", root),
                mock.patch.object(export, "_HERE", here),
                mock.patch.dict(
                    sys.modules, {"torch._vendor.quack.rmsnorm": mod}, clear=False
                ),
            ):
                closure = export.source_closure()
            rel = os.path.join("torch", "_vendor", "quack", "rmsnorm.py")
            self.assertIn(rel, closure)
            # ...and the recorded hash is the file's, so an edit invalidates it.
            self.assertEqual(closure[rel], export._file_hash(path))


class TestGeneratedCoversGuards(unittest.TestCase):
    """The covers predicate is registered as a catch-all custom op, so the router
    calls it for ANY dispatch of that op -- CPU tensors included, on hosts with no
    GPU at all."""

    def _gen(self, params="const at::Tensor & self, int64_t k", covers_schema=None):
        # N and K: _FakeDecl.cpp_dispatch reads both.
        sc = dict(
            SIDECAR,
            spec={"N": 1024, "K": 8},
            _dir="/art",
            tensor_args=[{"name": "mX", "dynamic_sizes": [0], "dynamic_strides": [0]}],
        )
        schema = covers_schema or "covers_fakeop(Tensor self, int k) -> bool"
        return gen_aot_lib.gen_op(
            op="fakeop",
            key="CUDA",
            d=_FakeDecl,
            sidecars=[sc],
            impl_params=params,
            covers=(params, schema, "return self.is_cuda();"),
        )

    def test_the_covers_body_checks_is_cuda_before_reading_properties(self):
        # getDeviceProperties(-1) falls back to the CURRENT device, so without the guard
        # a CPU tensor initializes CUDA inside a predicate the router runs every call --
        # and aborts on a GPU-less host where the answer is simply false.
        src = self._gen()
        body = src.split("_covers(", 1)[1]
        guard = body.index("is_cuda()")
        props = body.index("getDeviceProperties")
        self.assertLess(guard, props, "is_cuda must be checked BEFORE the props read")
        self.assertIn("if (!self.is_cuda()) return false;", src)

    def test_covers_reads_the_tensors_device_not_the_current_one(self):
        # The router calls covers before any device guard, so on a mixed-capability
        # host the current device need not be the one the op will run on.
        src = self._gen()
        self.assertIn("getDeviceProperties(self.device().index())", src)

    def test_a_covers_signature_without_a_tensor_is_refused(self):
        # Nothing to ask is_cuda() of, and current-device properties reintroduce the
        # abort above. Refused at generation, where an author sees it.
        with self.assertRaisesRegex(RuntimeError, "plain at::Tensor"):
            self._gen(params="const at::ITensorListRef & tensors, int64_t dim")

    def test_a_double_quoted_schema_default_is_escaped(self):
        # 15 in-tree schemas carry one (str mode="constant" on pad). Unescaped it closes
        # the C++ string literal early, far from the declaration that caused it.
        src = self._gen(
            covers_schema='covers_fakeop(Tensor self, str mode="constant") -> bool'
        )
        self.assertIn(r"str mode=\"constant\"", src)
        # ...and the literal is still one argument: nothing between the escaped
        # quotes ends it.
        line = next(l for l in src.splitlines() if "m.def(" in l)
        self.assertEqual(line.count('", &::'), 1)


class TestSourceCommitOrdering(unittest.TestCase):
    """Generation writes NOTHING until every declaration has passed its refusals,
    and every file it does write is written atomically. Both properties exist
    because the main build compiles whatever is on disk at configure time, and
    stage 2 -- the only writer -- runs after that build."""

    def _tree(self, root, decl_id, prefix, ok=True):
        art = os.path.join(root, "sm_100a", decl_id)
        os.makedirs(art, exist_ok=True)
        _touch_artifacts(art, prefix, exts=(".o", ".h") if ok else (".h",))
        with open(os.path.join(art, prefix + ".json"), "w") as f:
            json.dump(
                dict(
                    SIDECAR,
                    prefix=prefix,
                    spec={"N": 1024, "K": 8},
                    sources=_current_sources(),
                    runtimes=_RUNTIMES,
                    version=export.SIDECAR_VERSION,
                ),
                f,
            )
        return art

    def test_no_source_is_written_when_a_later_declaration_is_refused(self):
        # TWO declarations, the second refusing: with one, the refusal happens before
        # that declaration's own source would be written. What this prevents is the
        # first's FRESH source beside the previous run's objects.
        with tempfile.TemporaryDirectory() as art, tempfile.TemporaryDirectory() as ops:
            self._tree(art, "aaa", "aaa_p__sm100a", ok=True)
            self._tree(art, "bbb", "bbb_p__sm100a", ok=False)  # .o missing

            class _Aaa(_FakeDecl):
                ATEN_OP = "aaa"

            class _Bbb(_FakeDecl):
                ATEN_OP = "bbb"

            for name in ("aaa", "bbb"):
                os.makedirs(os.path.join(ops, name))
                open(os.path.join(ops, name, "aot.py"), "w").close()
            decls = {"aaa": [_Aaa], "bbb": [_Bbb]}
            with _patched_generation(
                ops, lambda path: decls[os.path.basename(os.path.dirname(path))]
            ):
                with self.assertRaisesRegex(RuntimeError, "not on"):
                    gen_aot_lib.main(["--artifacts-dir", art])
            self.assertEqual(glob.glob(os.path.join(art, "*", "aot_*.cpp")), [])
            self.assertFalse(
                os.path.exists(os.path.join(art, gen_aot_lib.CMAKE_INCLUDE)),
                "no native_aot.cmake: its presence marks a finished generation",
            )

    def test_the_old_cmake_is_invalidated_before_any_source_is_written(self):
        # Sources are individually atomic but their paths are deterministic, and the
        # previous file already names them, so a run that dies between two sources
        # leaves a fresh source paired with the previous run's object list.
        order = []
        real = gen_aot_lib._write_atomic
        # Annotated: pyrefly types an unannotated {} as Unknown, and indexing it gives
        # `Unknown | None`, which assertIn's `container` parameter refuses.
        state: dict[str, str] = {}

        with tempfile.TemporaryDirectory() as art, tempfile.TemporaryDirectory() as ops:
            self._tree(art, "fakeop", SIDECAR["prefix"])
            os.makedirs(os.path.join(ops, "fakeop"))
            open(os.path.join(ops, "fakeop", "aot.py"), "w").close()
            stale = os.path.join(art, gen_aot_lib.CMAKE_INCLUDE)
            with open(stale, "w") as f:
                f.write("ARCH_LIST_ABSENT\nOBJECT=/gone/old.o\n")

            def include_now():
                if not os.path.exists(stale):
                    return ""
                with open(stale) as f:
                    return f.read()

            def spy(path, text):
                name = os.path.basename(path)
                order.append(name)
                if name.startswith("aot_"):
                    state.setdefault("cmake_then", include_now())
                return real(path, text)

            with (
                _patched_generation(ops),
                mock.patch.object(gen_aot_lib, "_write_atomic", spy),
            ):
                gen_aot_lib.main(["--artifacts-dir", art])
        # The previous CONTENT goes, not the file: unlinking it drops CMake's configure
        # dependency and leaves the next generation invisible to `cmake --build`.
        self.assertNotIn("OBJECT=/gone/old.o", state["cmake_then"])
        self.assertIn("Nothing to embed", state["cmake_then"])
        # ...and the new one is written LAST, after every source.
        self.assertEqual(order[-1], gen_aot_lib.CMAKE_INCLUDE)

    def test_every_written_file_goes_through_the_atomic_writer(self):
        # TestAtomicWrites exercises _write_atomic directly, which cannot see a call
        # site that stopped using it. All three writers are checked here.
        calls = []
        real = gen_aot_lib._write_atomic

        def spy(path, text):
            calls.append(os.path.basename(path))
            return real(path, text)

        with tempfile.TemporaryDirectory() as art, tempfile.TemporaryDirectory() as ops:
            self._tree(art, "fakeop", SIDECAR["prefix"])
            os.makedirs(os.path.join(ops, "fakeop"))
            open(os.path.join(ops, "fakeop", "aot.py"), "w").close()
            with (
                _patched_generation(ops),
                mock.patch.object(gen_aot_lib, "_write_atomic", spy),
            ):
                gen_aot_lib.main(["--artifacts-dir", art])
        self.assertIn(gen_aot_lib.CMAKE_INCLUDE, calls)
        self.assertIn("native_aot_local.ver", calls)
        # The generated source too: CMake only checks it exists, so a truncated one is
        # compiled by the main build.
        self.assertTrue(
            any(c.startswith("aot_") and c.endswith(".cpp") for c in calls),
            f"the generated source was not written atomically: {calls}",
        )


class TestSizeGateIsPerKindNotPerFile(unittest.TestCase):
    def test_one_narrowing_kind_among_several_emits_the_gate(self):
        # A mixed sidecar list is the only shape that tells `any` from `all` here, and
        # `all` would drop the gate for a file that does carry a narrowing kernel.
        class _Wide(toolchains.Toolchain):
            kind = "wide"
            NARROWS_SHAPES_TO_INT32 = False
            artifact_exts = (".o",)
            link_exts = (".o",)

            def gen_launcher(self, sidecar):
                return f"void launch_{sidecar['prefix']}() {{}}"

            def kernel_includes(self, sidecar):
                return []

        spec = {"N": 1024, "K": 8}
        narrowing = dict(SIDECAR, prefix="k_narrow", spec=spec, _dir="/art")
        wide = dict(SIDECAR, prefix="k_wide", kind="wide", spec=spec, _dir="/art")
        with mock.patch.dict(toolchains.TOOLCHAINS, {"wide": _Wide()}, clear=False):
            src = gen_aot_lib.gen_op(
                op="fakeop",
                key="CUDA",
                d=_FakeDecl,
                sidecars=[wide, narrowing],
                impl_params="const at::Tensor & self, int64_t k",
            )
        self.assertIn("_naot_dim_too_big", src)


class TestWheelRefusal(unittest.TestCase):
    def test_wheel_requires_an_importable_torch(self):
        # --wheel means a CI caller installed torch on the line above, so "not
        # importable" is a broken build rather than "not applicable".
        with (
            mock.patch.object(build_stage2, "_torch_probe", lambda e: False),
            mock.patch.object(build_stage2, "should_run", lambda: False),
        ):
            with self.assertRaisesRegex(RuntimeError, "does not import"):
                build_stage2.main(["--wheel", "/tmp/nonexistent.whl"])

    def test_the_refusal_is_ahead_of_every_applicability_gate(self):
        # The refusal cannot consult should_run, which needs torch: hence the guard.
        called = []
        with (
            mock.patch.object(build_stage2, "_torch_probe", lambda e: False),
            mock.patch.object(
                build_stage2, "should_run", lambda: called.append(1) or False
            ),
        ):
            with self.assertRaises(RuntimeError):
                build_stage2.main(["--wheel", "/tmp/nonexistent.whl"])
        self.assertEqual(called, [])

    def test_without_wheel_an_unimportable_torch_is_a_clean_skip(self):
        # The other half of the same contract: an ordinary build must degrade.
        with mock.patch.object(build_stage2, "_torch_probe", lambda e: False):
            self.assertEqual(build_stage2.main([]), 0)


class TestExportMain(unittest.TestCase):
    """export.main() had no coverage at all, including the TORCH_CUDA_ARCH_LIST
    translation that another comment in the same file depends on: _CLOSURE_EXCLUDED
    justifies keeping build_stage2.py out of the source closure on the grounds
    that "export.py reads the arch list itself, so the driver passes it nothing
    kernel-affecting"."""

    def _main(self, argv, env, jobs_seen):
        with (
            tempfile.TemporaryDirectory() as out,
            mock.patch.dict(os.environ, env, clear=False),
            mock.patch.object(
                export,
                "_collect_jobs",
                lambda ops, root, archs: jobs_seen.append(archs) or [],
            ),
        ):
            export.main([*argv, "--out-dir", out])

    def test_the_arch_list_is_translated_into_arches(self):
        seen = []
        self._main([], {"TORCH_CUDA_ARCH_LIST": "9.0a;10.0a"}, seen)
        self.assertEqual(seen, [["sm_90a", "sm_100a"]])

    def test_an_explicit_arch_wins_over_the_arch_list(self):
        seen = []
        self._main(["--arch", "sm_90a"], {"TORCH_CUDA_ARCH_LIST": "10.0a"}, seen)
        self.assertEqual(seen, [["sm_90a"]])

    def test_no_arch_list_means_on_device(self):
        # archs [None] is what makes _collect_jobs resolve from the device.
        seen = []
        self._main([], {"TORCH_CUDA_ARCH_LIST": ""}, seen)
        self.assertEqual(seen, [[None]])

    def test_a_pending_export_invalidates_the_previous_generation(self):
        # The EXPORT half of "invalidate the previous generation first": artifacts are
        # direct link inputs in build.ninja, so an interrupted export plus a plain
        # `cmake --build` relinks torch_cuda from two revisions' objects.
        from tools.native_aot.gen_aot_lib import CMAKE_INCLUDE

        with tempfile.TemporaryDirectory() as out:
            stale = os.path.join(out, CMAKE_INCLUDE)
            with open(stale, "w") as f:
                f.write("# the previous generation\n")
            job = (
                "fakeop",
                "aot",
                {"N": 1},
                os.path.join(out, "sm_100a", "fakeop"),
                "sm_100a",
            )
            with (
                mock.patch.object(
                    export, "_collect_jobs", lambda ops, root, archs: [job]
                ),
                mock.patch.object(export, "_job_needed", lambda j, force: True),
                mock.patch.object(export, "_run_job", lambda j: "k__sm100a"),
                mock.patch.dict(
                    os.environ, {"TORCH_CUDA_ARCH_LIST": "10.0a"}, clear=False
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                export.main(["--out-dir", out])
            # OVERWRITTEN, not removed: a deleted include is not registered as a
            # configure dependency, so the next generation would be invisible to
            # `cmake --build`. The empty variant keeps the dependency alive.
            self.assertTrue(os.path.exists(stale), "the include must survive")
            with open(stale) as f:
                self.assertEqual(f.read(), gen_aot_lib.NOTHING_TO_EMBED)

    def test_a_refusal_invalidates_the_generation_it_tells_you_to_break(self):
        # Every refusal in _collect_jobs advises `rm -rf <arch tree>`, which the previous
        # generation names every object of, so the next build must not die on a missing
        # source inside a @generated file that names no remedy.
        from tools.native_aot.gen_aot_lib import CMAKE_INCLUDE

        with tempfile.TemporaryDirectory() as out:
            stale = os.path.join(out, CMAKE_INCLUDE)
            with open(stale, "w") as f:
                f.write("# names objects in the tree the user is about to delete\n")

            def refuse(ops, root, archs):
                raise RuntimeError("kernel artifacts with no sidecar; run `rm -rf ...`")

            with (
                mock.patch.object(export, "_collect_jobs", refuse),
                mock.patch.dict(
                    os.environ, {"TORCH_CUDA_ARCH_LIST": "10.0a"}, clear=False
                ),
                contextlib.redirect_stdout(io.StringIO()),
                self.assertRaisesRegex(RuntimeError, "no sidecar"),
            ):
                export.main(["--out-dir", out])
            self.assertTrue(os.path.exists(stale), "the include must survive")
            with open(stale) as f:
                self.assertEqual(f.read(), gen_aot_lib.NOTHING_TO_EMBED)

    def test_an_arch_list_with_no_exportable_arch_exports_nothing(self):
        # Not an error: a CUDA build for Ampere alone simply has no AOT kernels.
        seen = []
        self._main([], {"TORCH_CUDA_ARCH_LIST": "8.0;8.6"}, seen)
        self.assertEqual(seen, [], "_collect_jobs should not even be reached")


class TestCollectJobsRefusals(unittest.TestCase):
    def test_an_unknown_arch_is_refused_before_the_ops_walk(self):
        # The check has to run before the ops walk: with no declaration on disk that
        # walk never runs, so `--arch sm100a` matches nothing and exits 0.
        with (
            tempfile.TemporaryDirectory() as ops,
            tempfile.TemporaryDirectory() as out,
        ):
            with mock.patch.object(export, "OPS_DIR", ops):
                self.assertEqual(os.listdir(ops), [], "no declaration on disk")
                for bad in ("sm100a", "SM_100", "sm_1000", "sm_86", "90"):
                    with self.subTest(arch=bad):
                        with self.assertRaisesRegex(RuntimeError, "not an arch"):
                            export.main(["--arch", bad, "--out-dir", out])
                # ...and a known arch gets past the check (it exports nothing here,
                # which is the honest answer for a tree with no declarations).
                export.main(["--arch", "sm_100a", "--out-dir", out])

    def test_an_unnameable_arch_is_refused(self):
        # There is no unnamed layout: an artifact whose arch nobody can state is one the
        # runtime gate cannot match to hardware.
        with tempfile.TemporaryDirectory() as ops, tempfile.TemporaryDirectory() as out:
            _write_fake_decl(ops)
            with (
                mock.patch.object(export, "OPS_DIR", ops),
                _no_ambient_arch(device=None),
            ):
                with self.assertRaisesRegex(RuntimeError, "cannot determine the arch"):
                    export._collect_jobs(None, out, [None])

    def test_a_malformed_arch_is_refused(self):
        # The explicit path compares ARCHS by string, so without cc_of here `--arch
        # sm100a` matches no declaration, exports nothing and exits 0 -- a typo that
        # looks like a successful build.
        for bad in ("sm100a", "SM_100", "sm_1000", "sm_9", "90", "sm_100+PTX"):
            with self.subTest(arch=bad):
                with (
                    tempfile.TemporaryDirectory() as ops,
                    tempfile.TemporaryDirectory() as out,
                ):
                    _write_fake_decl(ops)
                    with (
                        mock.patch.object(export, "OPS_DIR", ops),
                        _no_ambient_arch(device=None),
                    ):
                        # The exact refusal: "compute capability" alone also
                        # matches cc_of's range error.
                        with self.assertRaisesRegex(
                            RuntimeError, "cannot read a compute capability"
                        ):
                            export._collect_jobs(None, out, [bad])

    def test_a_declaration_that_ships_nothing_says_so(self):
        # ARCHS spelling is load-bearing on the explicit path: a declaration pinning
        # ('sm_100a',) ships nothing for a release list of plain spellings, and with
        # several declarations that is partial -- the matched ops embed and pass the
        # post-relink check while the others are absent, with no tree to complain
        # about.
        with (
            tempfile.TemporaryDirectory() as ops,
            tempfile.TemporaryDirectory() as out,
        ):
            _write_fake_decl(ops, "ARCHS = ('sm_100a',)\n")
            with (
                mock.patch.object(export, "OPS_DIR", ops),
                _no_ambient_arch(device=None),
                contextlib.redirect_stdout(io.StringIO()) as printed,
            ):
                jobs = export._collect_jobs(None, out, ["sm_100"])
        self.assertEqual(jobs, [])
        said = printed.getvalue()
        self.assertIn("declares kernels but none for this build", said)
        self.assertIn("sm_100", said)
        # The report is the only thing that surfaces this, so it has to state the
        # declaration's real ARCHS rather than a fixed illustration.
        self.assertIn("ARCHS (sm_100a)", said)

    def test_a_declaration_that_misses_one_requested_arch_says_so(self):
        # The whole-declaration report is suppressed once a declaration ships for any
        # requested arch, which hides the worst case: the matched arches embed and pass
        # every check while devices of the missed capability fall back to aten
        # silently. Reported per arch, since the declaration claims this capability
        # under the other spelling.
        with (
            tempfile.TemporaryDirectory() as ops,
            tempfile.TemporaryDirectory() as out,
        ):
            _write_fake_decl(ops, "ARCHS = ('sm_90a', 'sm_100a')\n")
            with (
                mock.patch.object(export, "OPS_DIR", ops),
                _no_ambient_arch(device=None),
                contextlib.redirect_stdout(io.StringIO()) as printed,
            ):
                jobs = export._collect_jobs(None, out, ["sm_90a", "sm_100"])
        self.assertEqual(len(jobs), 1, "the arch that DID match must still export")
        said = printed.getvalue()
        self.assertIn("requested sm_100", said)
        self.assertIn("only as sm_100a", said)

    def test_an_arch_the_declaration_does_not_target_stays_quiet(self):
        # The other half: a capability the declaration claims under no spelling is not
        # news, or every partial build reports every op.
        with (
            tempfile.TemporaryDirectory() as ops,
            tempfile.TemporaryDirectory() as out,
        ):
            _write_fake_decl(ops, "ARCHS = ('sm_90a',)\n")
            with (
                mock.patch.object(export, "OPS_DIR", ops),
                _no_ambient_arch(device=None),
                contextlib.redirect_stdout(io.StringIO()) as printed,
            ):
                jobs = export._collect_jobs(None, out, ["sm_90a", "sm_100"])
        self.assertEqual(len(jobs), 1)
        self.assertEqual(printed.getvalue(), "")

    def test_an_on_device_export_that_ships_nothing_says_so(self):
        # Both paths mean the same thing ("export for this machine"), so both must
        # report it: otherwise the user sees `exported 0 kernels` with nothing named.
        with (
            tempfile.TemporaryDirectory() as ops,
            tempfile.TemporaryDirectory() as out,
        ):
            _write_fake_decl(ops, "ARCHS = ('sm_100a',)\n")
            with (
                mock.patch.object(export, "OPS_DIR", ops),
                _no_ambient_arch(device="sm_90"),
                contextlib.redirect_stdout(io.StringIO()) as printed,
            ):
                jobs = export._collect_jobs(None, out, [None])
        self.assertEqual(jobs, [])
        said = printed.getvalue()
        self.assertIn("declares kernels but none for this build", said)
        self.assertIn("sm_90", said)

    def test_a_declaration_that_does_ship_is_not_reported(self):
        # The control: the report must not fire for the ordinary case, or it is
        # noise on every build.
        with (
            tempfile.TemporaryDirectory() as ops,
            tempfile.TemporaryDirectory() as out,
        ):
            _write_fake_decl(ops, "ARCHS = ('sm_100', 'sm_100a')\n")
            with (
                mock.patch.object(export, "OPS_DIR", ops),
                _no_ambient_arch(device=None),
                contextlib.redirect_stdout(io.StringIO()) as printed,
            ):
                jobs = export._collect_jobs(None, out, ["sm_100"])
        self.assertTrue(jobs)
        self.assertNotIn("declares kernels but none", printed.getvalue())

    def test_a_declaration_that_disowns_the_local_capability_is_skipped(self):
        # Skipped, not exported into a tree the declaration disowns: generation refuses
        # such a tree with a "delete and re-export" remedy that rebuilds it identically.
        with tempfile.TemporaryDirectory() as ops, tempfile.TemporaryDirectory() as out:
            _write_fake_decl(ops, "ARCHS = ('sm_90a',)\n")
            with (
                mock.patch.object(export, "OPS_DIR", ops),
                _no_ambient_arch(device="sm_100"),
            ):
                self.assertEqual(export._collect_jobs(None, out, [None]), [])
                self.assertEqual(os.listdir(out), [], "no tree for a disowned arch")

    def test_the_declarations_spelling_is_adopted_for_the_local_capability(self):
        # sm_100 detected, ('sm_100a',) claimed: same capability, so both the tree and
        # the compile target use the declaration's spelling.
        with tempfile.TemporaryDirectory() as ops, tempfile.TemporaryDirectory() as out:
            _write_fake_decl(ops, "ARCHS = ('sm_100a',)\n")
            with (
                mock.patch.object(export, "OPS_DIR", ops),
                _no_ambient_arch(device="sm_100"),
            ):
                (job,) = export._collect_jobs(None, out, [None])
        self.assertEqual(os.path.basename(os.path.dirname(job[3])), "sm_100a")
        self.assertEqual(job[4], "sm_100a")


class TestCiAndCMakeWiring(unittest.TestCase):
    """Text assertions over files this suite cannot execute, so a change to the build
    wiring fails here rather than in a build."""

    def _read(self, rel):
        with open(os.path.join(REPO, rel)) as f:
            return f.read()

    def test_every_install_path_runs_stage_two(self):
        # stage 2 is a post-install step for want of a hook inside the PEP 517 backend,
        # so an install path that skips it builds a torch with no AOT ops at all.
        src = self._read(".spin/cmds.py")
        self.assertIn("def _native_aot_stage2():", src)
        # ...and it has to still RUN stage 2: gutting the body is invisible to a
        # check that only looks for the call sites.
        hook = src[src.index("def _native_aot_stage2():") :].split("\n\n\n")[0]
        self.assertIn("tools/native_aot/build_stage2.py", hook)
        for cmd in ("def develop():", "def install():"):
            with self.subTest(cmd=cmd):
                body = src[src.index(cmd) :].split("\n\n\n")[0]
                # AFTER the install, not merely present: the kernel builders import
                # the installed torch, so swapped, stage 2 relinks the previous one.
                self.assertLess(
                    body.index("_pip_install_cmd("),
                    body.index("_native_aot_stage2()"),
                    "stage 2 has to run after the install it depends on",
                )
        # ...and the same order in the CI shell. Located through the closing marker,
        # since the guard line itself occurs five times in that file.
        sh = self._read(".ci/pytorch/build.sh")
        block_at = sh.rindex(
            'if [[ "$BUILD_ENVIRONMENT" == *cuda* ]]; then',
            0,
            sh.index("fi  # BUILD_ENVIRONMENT == *cuda*"),
        )
        self.assertLess(
            sh.index("pip_install_whl "),
            block_at,
            "the wheel must be installed before stage 2 runs",
        )

    def test_manywheel_installs_the_raw_wheel_before_stage_two(self):
        # The kernel builders import the installed torch, so the raw wheel goes first --
        # inside the cuda guard, or a cpu container installs it for nothing.
        text = self._read(".ci/manywheel/build.sh")
        marker = "fi  # GPU_ARCH_TYPE == cuda*"
        self.assertIn(marker, text)
        block = text.partition(marker)[0]
        block = block[block.rindex('if [[ "${GPU_ARCH_TYPE}" == cuda* ]]; then') :]
        self.assertLess(
            block.index("pip install"),
            block.index("build_stage2.py --wheel"),
            "the raw wheel must be installed before stage 2 runs",
        )
        # Same single owner of the install decision as .ci/pytorch/build.sh.
        self.assertIn("--print-verdict", block)
        self.assertIn("install_cutlass_dsl", block)

    def test_the_verdict_word_the_shells_compare_is_the_one_stage_two_prints(self):
        # Both shells install the DSL wheels only when stage 2 says RUN, comparing with
        # ==, so renaming the word either side skips the install and fails the build.
        with (
            mock.patch.object(build_stage2, "should_run", return_value=True),
            contextlib.redirect_stdout(io.StringIO()) as printed,
        ):
            build_stage2.main(["--print-verdict"])
        word = printed.getvalue().strip()
        self.assertTrue(word, "--print-verdict printed nothing")
        for rel in (".ci/pytorch/build.sh", ".ci/manywheel/build.sh"):
            with self.subTest(rel=rel):
                self.assertIn(f'--print-verdict)" == "{word}"', self._read(rel))

    def test_the_wheel_is_patched_before_it_is_repaired(self):
        # repair_wheel.py produces the PUBLISHED artifact, so stage 2 patches the raw
        # wheel first: reversed, every release wheel ships kernel-free and green.
        sh = self._read(".ci/manywheel/build.sh")
        self.assertLess(
            sh.index("build_stage2.py --wheel"),
            sh.index("repair_wheel.py"),
            "stage 2 must patch the raw wheel before repair_wheel.py copies it out",
        )

    def test_stage_two_is_cuda_guarded_in_both_ci_shells(self):
        # Structure, not the literal: the call must sit between the guard and its fi.
        text = self._read(".ci/pytorch/build.sh")
        # The closing marker must exist: partitioning on a missing separator returns
        # the whole string, and the assertions below would match an unrelated guard.
        marker = "fi  # BUILD_ENVIRONMENT == *cuda*"
        self.assertIn(marker, text)
        block = text.partition(marker)[0]
        guard_at = block.rindex('if [[ "$BUILD_ENVIRONMENT" == *cuda* ]]; then')
        self.assertIn("build_stage2.py --wheel", block[guard_at:])
        self.assertIn("--print-verdict", block[guard_at:])
        self.assertIn(
            'if [[ "${GPU_ARCH_TYPE}" == cuda* ]]; then',
            self._read(".ci/manywheel/build.sh"),
        )

    def test_both_shells_count_wheels_with_nullglob(self):
        # Without it a non-matching glob yields the literal pattern, and the
        # message said "found 1" for an EMPTY directory.
        for rel, pattern in (
            (".ci/pytorch/build.sh", "naot_wheels=(dist/*.whl)"),
            (".ci/manywheel/build.sh", 'naot_wheels=("${RAW_WHEEL_DIR}"/*.whl)'),
        ):
            with self.subTest(rel=rel):
                text = self._read(rel)
                # Order, with all three indices from the START of the file: searching
                # from the enable's own offset is tautological.
                enable = text.index("shopt -s nullglob")
                at = text.index(pattern)
                disable = text.index("shopt -u nullglob")
                self.assertLess(enable, at, "nullglob must be set BEFORE the glob")
                self.assertLess(at, disable, "and unset only AFTER it has expanded")

    def test_the_documented_skip_reasons_match_should_run(self):
        # The count is what keeps docstring, CONTRIBUTING.md and should_run() in step.
        doc = build_stage2.__doc__
        contributing = self._read("CONTRIBUTING.md")
        arms = (
            ("TORCH_NATIVE_AOT=0", "`TORCH_NATIVE_AOT=0`"),
            ("not Linux", "not Linux"),
            ("does not import", "does not import"),
            ("no toolchain targets this backend", "no toolchain targets this backend"),
            ("CUDA older than", "older than 13 or cannot be determined"),
            ("no published DSL wheel", "no published DSL wheel"),
            ("a static torch_cuda", "`BUILD_SHARED_LIBS=OFF`"),
            ("nothing declares kernels", "nothing declares kernels"),
            ("names no exportable arch", "no supported arch is targeted"),
        )
        # The first bullet block after the lead-in, delimited by blank lines: keyed on
        # a following paragraph's wording, this counted that paragraph's bullets too.
        blocks = doc.partition("Keep this list in sync")[2].split("\n\n")
        skips = next(b for b in blocks if b.strip().startswith("* "))
        bullets = [ln for ln in skips.splitlines() if ln.strip().startswith("* ")]
        self.assertEqual(len(bullets), len(arms), f"docstring skip bullets: {bullets}")
        for in_doc, in_contributing in arms:
            with self.subTest(arm=in_doc):
                self.assertIn(in_doc, doc)
                self.assertIn(in_contributing, contributing)


class TestStageTwoRefusesAnImportTimeRebuild(unittest.TestCase):
    """scikit-build-core's editable.rebuild rebuilds torch on import."""

    class ScikitBuildRedirectingFinder:
        rebuild_flag = True

    def test_the_installed_finder_is_read_for_the_rebuild_flag(self):
        with mock.patch.object(
            build_stage2.sys, "meta_path", [self.ScikitBuildRedirectingFinder()]
        ):
            self.assertIsNotNone(build_stage2._editable_rebuild_finder())
            with self.assertRaisesRegex(RuntimeError, "editable.rebuild"):
                build_stage2.main([])

    def test_a_finder_without_the_flag_is_left_alone(self):
        finder = self.ScikitBuildRedirectingFinder()
        finder.rebuild_flag = False
        with mock.patch.object(build_stage2.sys, "meta_path", [finder]):
            self.assertIsNone(build_stage2._editable_rebuild_finder())

    def test_the_opt_out_still_wins(self):
        # Ordered like every other refusal: TORCH_NATIVE_AOT=0 exempts it.
        with (
            mock.patch.object(
                build_stage2.sys, "meta_path", [self.ScikitBuildRedirectingFinder()]
            ),
            mock.patch.dict(os.environ, {"TORCH_NATIVE_AOT": "0"}),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(build_stage2.main([]), 0)


class TestStageTwoArgvContract(unittest.TestCase):
    """What main() passes to its two children, and the invariant that ties them:
    a tree export creates must be one generation was told about."""

    @contextlib.contextmanager
    def _run_main(self, arch_list, cache=None, archive=None):
        """main() with both children captured; returns the recorded calls."""
        calls = []

        def fake_call(cmd, **kw):
            calls.append((cmd, kw.get("env") or {}))
            return 0

        with contextlib.ExitStack() as stack:
            build = stack.enter_context(tempfile.TemporaryDirectory())
            art = os.path.join(build, "native_aot")
            os.makedirs(art)
            if cache is not None:
                with open(os.path.join(build, "CMakeCache.txt"), "w") as f:
                    f.write(cache)
            env = {"TORCH_CUDA_ARCH_LIST": arch_list or "", "TORCH_NATIVE_AOT": ""}
            stack.enter_context(mock.patch.dict(os.environ, env, clear=False))
            stack.enter_context(mock.patch.object(build_stage2, "BUILD_DIR", build))
            stack.enter_context(
                mock.patch.object(build_stage2, "NATIVE_AOT_ARTIFACTS_DIR", art)
            )
            stack.enter_context(
                mock.patch.object(build_stage2, "should_run", lambda: True)
            )
            stack.enter_context(
                mock.patch.object(build_stage2, "require_runtimes", lambda: None)
            )
            # Discovery is a separate subject: unpatched, these reach the real DSL
            # wheel and a real torch probe. What they are about is main()'s argv.
            stack.enter_context(
                mock.patch.object(build_stage2, "_dsl_runtime_archive", lambda: archive)
            )
            stack.enter_context(mock.patch.object(subprocess, "call", fake_call))
            # No generated source, so main() stops before the relink -- the argv is
            # the whole subject here.
            self.assertEqual(build_stage2.main([]), 0)
            yield calls

    def _args_of(self, calls, script):
        ((cmd, env),) = [(c, e) for c, e in calls if any(script in str(a) for a in c)]
        return cmd, env

    def test_both_children_are_given_the_one_artifacts_directory(self):
        # Both children must get the same tree, or export writes A, generation reads B.
        with self._run_main(None) as calls:
            export_cmd, _ = self._args_of(calls, "export.py")
            gen_cmd, _ = self._args_of(calls, "gen_aot_lib.py")
            out_dir = export_cmd[export_cmd.index("--out-dir") + 1]
            artifacts = gen_cmd[gen_cmd.index("--artifacts-dir") + 1]
            self.assertEqual(out_dir, build_stage2.NATIVE_AOT_ARTIFACTS_DIR)
            self.assertEqual(artifacts, out_dir, "both children need one tree")

    def test_the_arch_list_reaches_the_export_child(self):
        # export.py reads TORCH_CUDA_ARCH_LIST itself, so the value has to be in the
        # child's environment even for a -D build where this process never had it.
        with self._run_main(
            None, cache="TORCH_CUDA_ARCH_LIST:STRING=9.0;10.0a\n"
        ) as calls:
            _, env = self._args_of(calls, "export.py")
            self.assertEqual(env.get("TORCH_CUDA_ARCH_LIST"), "9.0;10.0a")

    def test_generation_is_told_both_the_filter_and_the_recorded_list(self):
        with self._run_main("9.0;10.0a") as calls:
            cmd, _ = self._args_of(calls, "gen_aot_lib.py")
            self.assertIn("--archs", cmd)
            self.assertIn("--arch-list", cmd)
            # --arch-list is recorded verbatim for CMake to compare against.
            self.assertEqual(cmd[cmd.index("--arch-list") + 1], "9.0;10.0a")

    def test_the_discovered_archive_is_passed_to_generation(self):
        # The first of the two hops between _dsl_runtime_archive() and the emitted
        # target_link_libraries, each tested on its own: this pins the forwarding.
        archive = "/x/libcuda_dialect_runtime_static.a"
        with self._run_main(None, archive=archive) as calls:
            cmd, _ = self._args_of(calls, "gen_aot_lib.py")
        self.assertIn("--dsl-runtime", cmd)
        self.assertEqual(cmd[cmd.index("--dsl-runtime") + 1], archive)

    def test_an_on_device_run_passes_no_arch_filter(self):
        # Nothing to filter by: the local GPU is the whole arch list, and passing
        # a filter here would be comparing a resolved spelling against itself.
        with self._run_main(None) as calls:
            cmd, _ = self._args_of(calls, "gen_aot_lib.py")
            self.assertNotIn("--archs", cmd)
            self.assertNotIn("--arch-list", cmd)

    def test_every_tree_export_creates_is_one_generation_was_told_about(self):
        # Directory, sidecar, --archs filter and recorded ARCH_LIST must follow from
        # one resolution. The lists are the real ones (manywheel, and the b200 job).
        for arch_list in (
            "7.5;8.0;9.0;10.0;12.0",
            "7.5;8.0;8.6;9.0",
            "10.0a",
            "9.0",
            "9.0a;10.0",
            # Both spellings collapse to one arch; the two sides must agree which.
            "10.0;10.0a",
        ):
            with self.subTest(arch_list=arch_list):
                with self._run_main(arch_list) as calls:
                    cmd, _ = self._args_of(calls, "gen_aot_lib.py")
                told = cmd[cmd.index("--archs") + 1 : cmd.index("--arch-list")]
                with (
                    tempfile.TemporaryDirectory() as ops,
                    tempfile.TemporaryDirectory() as out,
                ):
                    _write_fake_decl(ops)
                    with (
                        mock.patch.object(export, "OPS_DIR", ops),
                        _no_ambient_arch(device=None),
                    ):
                        jobs = export._collect_jobs(
                            None, out, export.archs_from_cuda_arch_list(arch_list)
                        )
                    trees = sorted(
                        {os.path.basename(os.path.dirname(j[3])) for j in jobs}
                    )
                self.assertTrue(trees, "the fixture should export something")
                for tree in trees:
                    self.assertIn(
                        tree,
                        told,
                        f"export created {tree}/ but generation was told to filter "
                        f"on {told}, so that tree would be ignored and nothing "
                        f"embedded",
                    )


class TestRelinkNeverStrandsTheInstalledTorch(unittest.TestCase):
    """main()'s relink half, which copies over the INSTALLED torch.

    A separate fixture from TestStageTwoArgvContract._run_main, which supplies no
    generated source and so stops before the relink."""

    OLD, NEW = b"PREVIOUSLY-INSTALLED", b"RELINKED"

    @contextlib.contextmanager
    def _main(
        self,
        cache=True,
        configure_embeds=True,
        configure_rc=0,
        embedded_after=True,
        kill_swap=False,
        wheel=False,
        verdict=True,
        runtimes=None,
        child_rc=0,
        stale_include=False,
        built=True,
        installed=True,
        grew=0,
        cache_after=None,
        cmake_command=None,
        rebuild_sibling=False,
    ):
        """main() with every child faked, so the ORDER is the production one.

        Yields `outcome` (exception message or return code), `content` (bytes in
        site-packages), `children` (commands launched), `listing` (the installed lib
        dir), `argv` (the reconfigure's command line), `include` (the emitted CMake)
        and `reported` (stderr); one keyword per arm of main()."""
        children = []
        commands = []
        patched = []
        argv = []
        kwargs = {}

        def fake_call(cmd, **kw):
            commands.append(list(cmd))
            children.append(os.path.basename(str(cmd[1])) if len(cmd) > 1 else cmd[0])
            if rebuild_sibling and "--target" in cmd:
                with open(os.path.join(build, "lib", "libtorch_cpu.so"), "wb") as f:
                    f.write(b"REBUILT-DEPENDENCY")
            if grew and "--target" in cmd:
                with open(os.path.join(build, "lib", "libtorch_cuda.so"), "wb") as f:
                    f.truncate(grew)  # sparse: only the reported size matters
            return child_rc

        def fake_run(cmd, **kw):
            children.append("reconfigure")
            commands.append(list(cmd))
            argv.append(list(cmd))
            kwargs.update(kw)
            if cache_after is not None:
                with open(os.path.join(build, "CMakeCache.txt"), "w") as f:
                    f.write(cache_after)
            out = (
                "-- native-AOT: embedding 1 object(s)\n" if configure_embeds else "--\n"
            )
            return subprocess.CompletedProcess(cmd, configure_rc, out, "cmake said no")

        def require_runtimes():
            if runtimes is not None:
                raise RuntimeError(runtimes)

        with contextlib.ExitStack() as stack:
            d = stack.enter_context(tempfile.TemporaryDirectory())
            build = os.path.join(d, "build")
            art = os.path.join(build, "native_aot")
            libdir = os.path.join(d, "site-packages", "torch", "lib")
            os.makedirs(os.path.join(art, "fakeop"))
            os.makedirs(os.path.join(build, "lib"))
            os.makedirs(libdir)
            # A generated source, so main() gets past the "nothing to embed" return.
            open(os.path.join(art, "fakeop", "aot_fakeop_cuda.cpp"), "w").close()
            include = os.path.join(art, gen_aot_lib.CMAKE_INCLUDE)
            if stale_include:
                # What a PREVIOUS run wired up. caffe2/CMakeLists.txt include()s this
                # file unconditionally, so it is linked by every later configure.
                with open(include, "w") as f:
                    f.write('target_sources(torch_cuda PRIVATE "/x/k.o")\n')
            # A dependency of torch_cuda, which the relink may rebuild and stage 2
            # does not install.
            with open(os.path.join(build, "lib", "libtorch_cpu.so"), "wb") as f:
                f.write(b"AS-BUILT")
            if built:
                with open(os.path.join(build, "lib", "libtorch_cuda.so"), "wb") as f:
                    f.write(self.NEW)
            target = os.path.join(libdir, "libtorch_cuda.so")
            if installed:
                with open(target, "wb") as f:
                    f.write(self.OLD)
            if cache:
                with open(os.path.join(build, "CMakeCache.txt"), "w") as f:
                    f.write("CUDAToolkit_VERSION_MAJOR:STRING=13\n")
                    if cmake_command:
                        f.write(f"CMAKE_COMMAND:INTERNAL={cmake_command}\n")
            for obj, name, value in (
                (build_stage2, "BUILD_DIR", build),
                (build_stage2, "NATIVE_AOT_ARTIFACTS_DIR", art),
                (build_stage2, "should_run", lambda: verdict),
                (build_stage2, "require_runtimes", require_runtimes),
                (build_stage2, "_installed_lib_dir", lambda: libdir),
                (build_stage2, "_arch_list", lambda: ""),
                (build_stage2, "_dsl_runtime_archive", lambda: None),
                (build_stage2, "_torch_probe", lambda e: embedded_after),
                (build_stage2, "patch_wheel", lambda w, lib: patched.append((w, lib))),
                (subprocess, "call", fake_call),
                (subprocess, "run", fake_run),
            ):
                stack.enter_context(mock.patch.object(obj, name, value))
            if kill_swap:
                real_replace = os.replace

                def failing_replace(a, b):
                    if str(a).endswith(".tmp"):
                        raise KeyboardInterrupt("killed at the swap")
                    return real_replace(a, b)

                stack.enter_context(mock.patch.object(os, "replace", failing_replace))
            err = stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            args = []
            if wheel:
                whl = os.path.join(d, "torch-0.0.0-cp312-cp312-linux_x86_64.whl")
                open(whl, "w").close()
                args = ["--wheel", whl]
            try:
                outcome = f"returned {build_stage2.main(args)}"
            except (RuntimeError, KeyboardInterrupt) as e:
                outcome = str(e)
            content = None
            if os.path.exists(target):
                with open(target, "rb") as f:
                    content = f.read()
            children += [f"wheel:{w}|{lib}" for w, lib in patched]
            yield types.SimpleNamespace(
                outcome=outcome,
                content=content,
                children=children,
                commands=commands,
                listing=sorted(os.listdir(libdir)),
                argv=argv[-1] if argv else [],
                kwargs=kwargs,
                include=include,
                reported=err.getvalue(),
            )

    def test_a_declined_verdict_does_nothing_at_all(self):
        # Every other fixture patches should_run() True, so this pins the call itself.
        with self._main(verdict=False) as run:
            self.assertEqual(run.outcome, "returned 0")
            self.assertEqual(run.children, [])
            self.assertEqual(run.content, self.OLD)

    def test_a_declining_run_disables_what_a_previous_one_wired_up(self):
        # build_all.sh shares one build/ across eight interpreters (see the docstring).
        with self._main(verdict=False, stale_include=True) as run:
            self.assertEqual(run.outcome, "returned 0")
            with open(run.include) as f:
                emitted = f.read()
        self.assertIn("Nothing to embed", emitted)
        self.assertNotIn("target_sources", emitted)

    def test_a_declining_run_creates_no_include_where_there_was_none(self):
        # A file created where none existed is invisible until the NEXT configure.
        with self._main(verdict=False) as run:
            self.assertFalse(os.path.exists(run.include))

    def test_a_missing_runtime_stops_before_the_export(self):
        # The stack's one deliberate build failure: past the gates with no DSL runtime,
        # failing beats shipping a wheel missing its declared kernels.
        with self._main(runtimes="cutlass is not installed") as run:
            self.assertIn("cutlass is not installed", run.outcome)
            self.assertEqual(run.children, [])
            self.assertEqual(run.content, self.OLD)

    def test_a_build_dir_without_a_cache_is_refused_before_any_cmake(self):
        # `cmake -B` on a missing directory exits 0 and configures FROM SCRATCH.
        with self._main(cache=False) as run:
            self.assertIn("holds no CMakeCache.txt", run.outcome)
            self.assertEqual(run.content, self.OLD)
            self.assertNotIn("reconfigure", run.children)

    def test_a_failing_reconfigure_shows_what_cmake_said(self):
        # CMake prints most of its failure context on stdout, so this arm captures both
        # streams and echoes them; DEVNULL leaves a failing configure unreadable.
        with self._main(configure_rc=1) as run:
            self.assertIn("reconfiguring", run.outcome)
            self.assertIn("exit 1", run.outcome)
            self.assertIn("embedding 1 object(s)", run.reported)
            self.assertIn("cmake said no", run.reported)
            self.assertEqual(run.content, self.OLD)

    def test_a_reconfigure_that_does_not_embed_is_refused_before_the_relink(self):
        # The STATUS line the generated CMake prints is the only pre-relink evidence that
        # the build agrees it should embed, so a disagreement must stop here.
        with self._main(configure_embeds=False) as run:
            self.assertIn("did not report embedding", run.outcome)
            self.assertEqual(run.content, self.OLD)
            self.assertEqual(run.children[-1], "reconfigure")

    def test_a_child_that_dies_names_the_step_and_the_signal(self):
        # check_call's CalledProcessError names neither, and a 32-way export is what an
        # OOM killer reaches for first.
        with self._main(child_rc=-9) as run:
            self.assertIn("exporting kernels", run.outcome)
            self.assertIn("signal 9", run.outcome)
            self.assertEqual(run.content, self.OLD)

    def test_a_relink_that_produced_no_library_says_so(self):
        with self._main(built=False) as run:
            self.assertIn("expected relinked library", run.outcome)
            self.assertEqual(run.content, self.OLD)

    def test_a_torch_whose_lib_never_held_the_library_is_refused(self):
        # _installed_lib_dir found *a* torch; writing a library into a layout that
        # never had one means we are pointed at the wrong environment.
        with self._main(installed=False) as run:
            self.assertIn("not the one this tree built", run.outcome)
            self.assertEqual(run.listing, [])

    def test_the_happy_path_replaces_the_installed_library(self):
        # ...and the guards above do not block the case they exist to protect.
        with self._main() as run:
            self.assertEqual(run.outcome, "returned 0")
            self.assertEqual(run.content, self.NEW)
            self.assertIn("reconfigure", run.children)
            # NOTHING else left behind: a stray copy in site-packages carries a name
            # no RECORD lists, which `pip uninstall` cannot clear.
            self.assertEqual(run.listing, ["libtorch_cuda.so"])

    def test_a_kill_at_the_swap_leaves_the_library_and_no_temporary(self):
        # One os.replace, so the installed library never stops existing.
        with self._main(kill_swap=True) as run:
            self.assertIn("killed at the swap", run.outcome)
            self.assertEqual(run.content, self.OLD)
            self.assertEqual(run.listing, ["libtorch_cuda.so"])

    def test_a_failed_verification_says_what_state_it_leaves(self):
        # No restore copy is kept, so the error has to say how to get back.
        with self._main(embedded_after=False) as run:
            self.assertIn("reports no embedded kernels", run.outcome)
            self.assertIn("reinstall the wheel", run.outcome)
            self.assertEqual(run.content, self.NEW)
            self.assertEqual(run.listing, ["libtorch_cuda.so"])

    def test_the_report_names_the_bytes_the_kernels_added(self):
        # The delta is what the line is for: one added arch can grow a wheel by tens of
        # MiB, and nothing else in the build log states it.
        with self._main(grew=(3 << 20) + len(self.NEW)) as run:
            self.assertEqual(run.outcome, "returned 0")
        self.assertIn("3 MiB relinked into", run.reported)
        self.assertIn("+3 MiB of embedded kernels", run.reported)

    def test_the_reconfigure_forces_status_messages(self):
        # A cached CMAKE_MESSAGE_LOG_LEVEL would hide the marker; the flag wins.
        with self._main() as run:
            self.assertEqual(run.outcome, "returned 0")
            self.assertIn("--log-level=STATUS", run.argv)
            # ...and BOTH streams are captured, or configure.stdout is None and the
            # `in` test below raises TypeError.
            self.assertIs(run.kwargs.get("capture_output"), True)

    def test_a_reconfigure_that_changes_the_cache_is_refused(self):
        # EnvVarForwarding FORCEs env vars into the cache, so a stage-2 run in another
        # environment would relink torch_cuda against different settings.
        drifted = "//From environment\nCUDAToolkit_VERSION_MAJOR:STRING=12\n"
        with self._main(cache_after=drifted) as run:
            self.assertIn("changed this build's configuration", run.outcome)
            self.assertIn("CUDAToolkit_VERSION_MAJOR", run.outcome)
            self.assertIn("From environment", run.outcome)
            # Refused BEFORE the relink, so nothing reached the installed torch.
            self.assertNotIn("--build", run.children)
        self.assertEqual(run.content, self.OLD)

    def test_cache_entries_the_reconfigure_adds_are_not_drift(self):
        # Only a CHANGED value is drift: a reconfigure legitimately adds entries.
        grown = "CUDAToolkit_VERSION_MAJOR:STRING=13\nNEW_ENTRY:BOOL=ON\n"
        with self._main(cache_after=grown) as run:
            self.assertEqual(run.outcome, "returned 0")
        self.assertEqual(run.content, self.NEW)

    def test_a_relink_that_rebuilt_a_dependency_is_refused(self):
        # --target torch_cuda builds its dependencies too, and only torch_cuda ships.
        with self._main(rebuild_sibling=True) as run:
            self.assertIn("also rebuilt libtorch_cpu.so", run.outcome)
        self.assertEqual(run.content, self.OLD)

    def test_a_new_env_sourced_cache_entry_is_drift(self):
        # EnvVarForwarding creates the entry when the build had none.
        added = "//From environment\nWERROR:STRING=1\n"
        with self._main(cache_after=added) as run:
            self.assertIn("WERROR: absent -> STRING=1", run.outcome)
            self.assertNotIn("--build", run.children)

    def test_both_children_run_the_cmake_that_configured_the_build(self):
        # CMAKE_COMMAND is often a pip wheel's cmake, not the one on PATH.
        with self._main(cmake_command=sys.executable) as run:
            self.assertEqual(run.outcome, "returned 0")
        cmakes = [c[0] for c in run.commands if c[1].startswith("--")]
        self.assertEqual(cmakes, [sys.executable, sys.executable], run.commands)

    def test_the_wheel_is_patched_with_the_relinked_library(self):
        # Nothing downstream re-checks the wheel: the AOT tests skip without kernels.
        with self._main(wheel=True) as run:
            self.assertEqual(run.outcome, "returned 0")
            patched = [c for c in run.children if c.startswith("wheel:")]
        self.assertEqual(len(patched), 1, f"main() did not patch: {run.children}")
        whl, lib = patched[0][len("wheel:") :].split("|")
        self.assertTrue(whl.endswith(".whl"), whl)
        # The RELINKED library, not the installed copy: same basename, so a test
        # asserting only the name would pass on either.
        self.assertEqual(lib.split(os.sep)[-3:], ["build", "lib", "libtorch_cuda.so"])

    def test_a_wheel_that_does_not_exist_is_refused_before_the_export(self):
        # should_run is FALSE deliberately: this check sits ahead of it.
        with tempfile.TemporaryDirectory() as d:
            with (
                mock.patch.object(build_stage2, "_torch_probe", lambda e: True),
                mock.patch.object(build_stage2, "should_run", lambda: False),
                mock.patch.dict(os.environ, {"TORCH_NATIVE_AOT": ""}, clear=False),
                mock.patch.object(build_stage2, "NATIVE_AOT_ARTIFACTS_DIR", d),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                with self.assertRaisesRegex(RuntimeError, "does not exist"):
                    build_stage2.main(["--wheel", os.path.join(d, "nope.whl")])


class TestStaticBuildSkips(unittest.TestCase):
    def test_build_shared_libs_off_skips_cleanly(self):
        # caffe2/CMakeLists.txt refuses to embed into a static torch_cuda, CMake
        # discarding target_link_options for an archive. Skipping here answers "not
        # applicable" rather than exporting and then failing every later configure.
        with contextlib.ExitStack() as stack:
            build = stack.enter_context(tempfile.TemporaryDirectory())
            with open(os.path.join(build, "CMakeCache.txt"), "w") as f:
                # The CUDA major too: the >=13 gate runs before this arm and falls
                # through to the installed torch, so the skip would come from the
                # wrong arm wherever torch is missing.
                f.write(
                    "CUDAToolkit_VERSION_MAJOR:STRING=13\nBUILD_SHARED_LIBS:BOOL=OFF\n"
                )
            stack.enter_context(mock.patch.object(build_stage2, "BUILD_DIR", build))
            stack.enter_context(
                mock.patch.dict(
                    os.environ,
                    {"TORCH_NATIVE_AOT": "", "TORCH_CUDA_ARCH_LIST": "10.0a"},
                )
            )
            stack.enter_context(mock.patch.object(sys, "platform", "linux"))
            stack.enter_context(
                mock.patch.object(
                    build_stage2,
                    "_torch_probe",
                    lambda e: e != "torch.version.hip is not None",
                )
            )
            err = stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            self.assertFalse(build_stage2.should_run())
        self.assertIn("BUILD_SHARED_LIBS=OFF", err.getvalue())


class TestCacheReadCannotRaise(unittest.TestCase):
    """should_run() promises never to raise, and the cache read is the first thing
    it does: a traceback from --print-verdict makes the CI shell read "" != RUN,
    skip installing the runtimes, and the real invocation then fail demanding
    them."""

    def test_an_unreadable_cache_is_treated_as_unset(self):
        with contextlib.ExitStack() as stack:
            build = stack.enter_context(tempfile.TemporaryDirectory())
            cache = os.path.join(build, "CMakeCache.txt")
            with open(cache, "w") as f:
                f.write("TORCH_CUDA_ARCH_LIST:STRING=9.0\n")
            os.chmod(cache, 0)
            stack.enter_context(mock.patch.object(build_stage2, "BUILD_DIR", build))
            stack.enter_context(
                mock.patch.dict(os.environ, {"TORCH_CUDA_ARCH_LIST": ""})
            )
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            try:
                self.assertEqual(build_stage2._arch_list(), "")
            finally:
                os.chmod(cache, 0o644)

    def test_a_directory_named_cmakecache_is_treated_as_unset(self):
        with contextlib.ExitStack() as stack:
            build = stack.enter_context(tempfile.TemporaryDirectory())
            os.makedirs(os.path.join(build, "CMakeCache.txt"))
            stack.enter_context(mock.patch.object(build_stage2, "BUILD_DIR", build))
            stack.enter_context(mock.patch.dict(os.environ, {"TORCH_NATIVE_AOT": ""}))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            self.assertFalse(build_stage2._opted_out())


class TestOptOutSpellings(unittest.TestCase):
    """The two sides must agree: a spelling one honours and the other ignores means
    the build embeds kernels while stage 2 thinks it opted out, or the reverse."""

    def _opted(self, value, cached=False):
        with contextlib.ExitStack() as stack:
            build = stack.enter_context(tempfile.TemporaryDirectory())
            if cached:
                with open(os.path.join(build, "CMakeCache.txt"), "w") as f:
                    f.write(f"TORCH_NATIVE_AOT:STRING={value}\n")
            stack.enter_context(mock.patch.object(build_stage2, "BUILD_DIR", build))
            stack.enter_context(
                mock.patch.dict(
                    os.environ, {"TORCH_NATIVE_AOT": "" if cached else value}
                )
            )
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            return build_stage2._opted_out()

    def test_cmake_falsy_spellings_opt_out(self):
        # Declared as a cache STRING, so ccmake offers it as editable text and the
        # idiomatic OFF must not read as "embed".
        for value in ("0", "OFF", "off", "false", "FALSE", "no", "N", "NOTFOUND"):
            for cached in (False, True):
                with self.subTest(value=value, cached=cached):
                    self.assertTrue(self._opted(value, cached))

    def test_truthy_spellings_do_not(self):
        for value in ("1", "ON", "true", "yes", "2"):
            for cached in (False, True):
                with self.subTest(value=value, cached=cached):
                    self.assertFalse(self._opted(value, cached))

    def test_the_value_is_read_exactly_as_cmake_reads_it(self):
        # Every case below matches `cmake -P` with a quoted if(), the form the generated
        # file uses. The realistic one is trailing whitespace: a value out of a
        # $(grep ...) or a folded YAML scalar arrives as "1\n", which CMake calls FALSE,
        # so stripping it here would embed kernels into a build that opted out. Leading
        # whitespace CMake tolerates for a number and not for a constant.
        for value in ("1 ", "1\n", "1\t", "TRUE ", "ON ", " y", "0x0", "1_0", "\t0"):
            with self.subTest(value=value):
                self.assertTrue(self._opted(value))
        for value in (" 1", "\t1", "\n1", "0x1", "0X1", "yEs", "010", "1e0"):
            with self.subTest(value=value):
                self.assertFalse(self._opted(value))

    def test_a_cache_entry_that_is_defined_but_empty_opts_out(self):
        # `-DTORCH_NATIVE_AOT=` writes `TORCH_NATIVE_AOT:UNINITIALIZED=`, which CMake
        # calls defined-and-false, so the generated file embeds nothing and stage 2
        # must agree. Blank in the ENVIRONMENT still means absent (the test below):
        # only in the cache can a key be present with no value.
        with contextlib.ExitStack() as stack:
            build = stack.enter_context(tempfile.TemporaryDirectory())
            with open(os.path.join(build, "CMakeCache.txt"), "w") as f:
                f.write("TORCH_NATIVE_AOT:UNINITIALIZED=\n")
            stack.enter_context(mock.patch.object(build_stage2, "BUILD_DIR", build))
            stack.enter_context(mock.patch.dict(os.environ, {"TORCH_NATIVE_AOT": ""}))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            self.assertTrue(build_stage2._opted_out())

    def test_blank_means_absent_and_falls_through_to_the_cache(self):
        # How a shell blanks a variable it will not unset. CMake treats blank as absent
        # too, so it must fall through to the cache rather than read as "run".
        with contextlib.ExitStack() as stack:
            build = stack.enter_context(tempfile.TemporaryDirectory())
            with open(os.path.join(build, "CMakeCache.txt"), "w") as f:
                f.write("TORCH_NATIVE_AOT:STRING=0\n")
            stack.enter_context(mock.patch.object(build_stage2, "BUILD_DIR", build))
            stack.enter_context(mock.patch.dict(os.environ, {"TORCH_NATIVE_AOT": ""}))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            self.assertTrue(build_stage2._opted_out())


class TestOrphanCheckIsCalled(unittest.TestCase):
    def test_collect_jobs_runs_the_orphan_check(self):
        # Every orphan test calls _check_no_orphan_artifacts directly, so this is what
        # pins its only production call site.
        seen = []
        with (
            tempfile.TemporaryDirectory() as ops,
            tempfile.TemporaryDirectory() as out,
        ):
            _write_fake_decl(ops)
            with (
                mock.patch.object(export, "OPS_DIR", ops),
                mock.patch.object(
                    export,
                    "_check_no_orphan_artifacts",
                    lambda d, points: seen.append(d),
                ),
                _no_ambient_arch(device="sm_100"),
            ):
                export._collect_jobs(None, out, [None])
        self.assertEqual(len(seen), 1, "the orphan check must run per (decl, arch)")
        self.assertTrue(seen[0].endswith(os.path.join("sm_100a", "fakeop")))

    def test_an_undescribed_artifact_is_reported_through_the_call_site(self):
        # End to end through the call site: the report reaches the build log, and
        # does not stop the export -- nothing links an artifact no sidecar names.
        with (
            tempfile.TemporaryDirectory() as ops,
            tempfile.TemporaryDirectory() as out,
        ):
            _write_fake_decl(ops)
            stray = os.path.join(out, "sm_100a", "fakeop")
            os.makedirs(stray)
            open(os.path.join(stray, "leftover.o"), "w").close()
            with (
                mock.patch.object(export, "OPS_DIR", ops),
                _no_ambient_arch(device="sm_100"),
                contextlib.redirect_stdout(io.StringIO()) as said,
            ):
                jobs = export._collect_jobs(None, out, [None])
        self.assertIn("leftover.o", said.getvalue())
        self.assertTrue(jobs, "the export still runs; the orphan is only disk")


class TestModulePathOrder(unittest.TestCase):
    def test_the_repo_root_is_appended_never_prepended(self):
        # The repo root's torch/ SOURCE tree shadows the installed wheel if inserted.
        for mod in ("build_stage2.py", "export.py", "gen_aot_lib.py"):
            with self.subTest(module=mod):
                with open(os.path.join(REPO, "tools", "native_aot", mod)) as f:
                    src = f.read()
                self.assertIn("sys.path.append(REPO)", src)
                # Space-stripped, so both spellings are one check.
                self.assertNotIn("sys.path.insert(0,REPO)", src.replace(" ", ""))


class TestEmittedCMake(unittest.TestCase):
    """caffe2/CMakeLists.txt is one include() and holds no logic, so every decision is
    emitted here. These assert the emitted CALLS, which is what CMake consumes."""

    def _emit(self, tmpdir, dsl=None, arch_list="10.0a", sources=None, objects=None):
        src = os.path.join(tmpdir, "aot_fakeop_cuda.cpp")
        with open(src, "w") as f:
            f.write("// generated\n")
        obj = os.path.join(tmpdir, "k.o")
        with open(obj, "w") as f:
            f.write("")
        ver = os.path.join(tmpdir, "native_aot_local.ver")
        with open(ver, "w") as f:
            f.write("{ local: *; };\n")
        path = gen_aot_lib.write_cmake_include(
            tmpdir,
            [src] if sources is None else sources,
            [obj] if objects is None else objects,
            ver,
            dsl,
            arch_list,
        )
        with open(path) as f:
            return f.read()

    def test_caffe2_cmakelists_holds_no_logic(self):
        # The point of the restructure: one include, and nothing to get wrong. If a
        # future change starts deriving things in CMake again, this fails.
        with open(os.path.join(REPO, "caffe2", "CMakeLists.txt")) as f:
            cmake = f.read()
        naot = [
            l
            for l in cmake.splitlines()
            if "native_aot" in l and not l.strip().startswith("#")
        ]
        self.assertEqual(
            naot,
            ['  include("${CMAKE_BINARY_DIR}/native_aot/native_aot.cmake" OPTIONAL)'],
            "caffe2/CMakeLists.txt should contain exactly one native-AOT line",
        )
        for gone in ("file(STRINGS", "ARCH_LIST_ABSENT", "_native_aot_normalize"):
            with self.subTest(construct=gone):
                self.assertNotIn(gone, cmake)

    def test_the_include_line_agrees_with_the_python_constants(self):
        # Derived from both constants, so moving either fails the literal above.
        rel = os.path.relpath(
            build_stage2.NATIVE_AOT_ARTIFACTS_DIR, build_stage2.BUILD_DIR
        )
        want = f'"${{CMAKE_BINARY_DIR}}/{rel}/{gen_aot_lib.CMAKE_INCLUDE}"'
        with open(os.path.join(REPO, "caffe2", "CMakeLists.txt")) as f:
            self.assertIn(want, f.read())

    def test_the_blocks_are_separated_by_one_blank_line(self):
        # The layout lives in the join, not the templates: gluing the blocks together
        # still emits valid CMake, so nothing else here would notice.
        with tempfile.TemporaryDirectory() as d:
            emitted = self._emit(d, dsl=os.path.join(d, "libdsl.a"))
        for boundary in (
            "endif()\n\n# Linux only",
            "endif()\n\n# Re-run configure",
            'generated source(s)")\n\nset_source_files_properties(',
            "EXTERNAL_OBJECT TRUE)\n\ntarget_sources(",
            ")\n\n# whole-archive is NOT needed",
        ):
            with self.subTest(boundary=boundary.splitlines()[-1]):
                self.assertIn(boundary, emitted)
        self.assertNotIn("\n\n\n", emitted)

    def test_a_non_linux_configure_embeds_nothing(self):
        # The embed is GNU-linker specific, so the file bails once, up front: guarding
        # only the linker blocks would leave target_sources embedding objects a
        # non-Linux link neither version-scripts nor exclude-libs.
        with tempfile.TemporaryDirectory() as d:
            emitted = self._emit(d, dsl=os.path.join(d, "libdsl.a"))
        before, guard, after = emitted.partition("if(NOT UNIX OR APPLE)")
        self.assertTrue(guard, "no platform guard emitted")
        # The CALL, not the word: the comment above the guard names it too.
        self.assertNotIn("target_sources(torch_cuda", before)
        self.assertIn("return()", after.split("endif()")[0])
        # ONE exit: no block below re-tests the platform.
        self.assertNotIn("APPLE", after)

    def test_the_linker_options_are_emitted_de_duplication_safe(self):
        # Three hazards: the compiler driver splits -Wl, (what LINKER: expands to) on
        # COMMAS; target_link_options de-duplicates bare -Xlinker tokens, which stops
        # torch_cuda linking at all; and SHELL: splits its own value on SPACES, so the
        # path must be quoted inside it.
        with tempfile.TemporaryDirectory() as d:
            emitted = self._emit(d, dsl=os.path.join(d, "libdsl.a"))
        self.assertNotIn('"LINKER:', emitted)
        self.assertNotIn('"-Xlinker" "--', emitted)
        for opt in ("--version-script", "--exclude-libs"):
            with self.subTest(option=opt):
                self.assertIn(f'"SHELL:-Xlinker {opt} -Xlinker \\"', emitted)

    def test_paths_are_absolute_and_quoted(self):
        # CMake resolves a relative path against a different directory than the
        # generator, and an unquoted one breaks on the first space.
        with tempfile.TemporaryDirectory() as d:
            rel = os.path.relpath(d)
            emitted = self._emit(rel)
        for line in emitted.splitlines():
            if line.strip().startswith('"') and line.strip().endswith('"'):
                self.assertTrue(
                    line.strip()[1:].startswith("/"), f"not absolute: {line!r}"
                )

    def test_the_install_rpath_property_is_set_on_the_target(self):
        # Stage 2's hand copy over site-packages matches what install writes only
        # because of this property, Dependencies.cmake setting
        # CMAKE_BUILD_WITH_INSTALL_RPATH FALSE globally. Without it the shipped library
        # carries the builder's build-tree RUNPATH.
        with tempfile.TemporaryDirectory() as d:
            emitted = self._emit(d)
        prop = "BUILD_WITH_INSTALL_RPATH TRUE"
        self.assertIn(f"set_target_properties(torch_cuda PROPERTIES {prop})", emitted)

    def test_the_status_line_stage_two_greps_for_is_emitted(self):
        # A contract across two files: stage 2 refuses to relink unless the reconfigure
        # echoes this, so a reworded message fails every stage-2 build with a
        # diagnostic blaming the CMake cache. Asserted on the side that emits it.
        with tempfile.TemporaryDirectory() as d:
            emitted = self._emit(d)
        self.assertIn(f'message(STATUS "{gen_aot_lib.EMBED_STATUS} ', emitted)

    def test_a_semicolon_in_a_path_is_refused(self):
        # Escaping it as \; yields a literal semicolon, but CMake re-splits source
        # lists on it downstream, so the build asks for the prefix before it. Refused
        # here rather than emitted as a file that configures and cannot build.
        with tempfile.TemporaryDirectory() as d:
            odd = os.path.join(d, "semi;colon")
            os.makedirs(odd)
            with self.assertRaisesRegex(RuntimeError, "splits source lists"):
                self._emit(odd)

    def test_spaces_and_dollars_survive_escaped(self):
        # Assert the emitted PATH, with the escaping spelled out here rather than taken
        # from _cmake_str: the prose in the emitted file contains `\$` and `\"` of its
        # own, so a "is there a backslash-dollar anywhere" assertion would pass with the
        # escaper deleted.
        with tempfile.TemporaryDirectory() as d:
            cases = (
                ("has space", "has space"),
                ("has$dollar", "has\\$dollar"),
                # ${...} and $ENV{...}, not just a bare $: the linker options quote
                # their path by hand, and unescaped there CMake expands the reference
                # away -- configure passes, the source compiles, and the link fails on
                # a version script that never existed. A bare $ survives raw, so only
                # these spellings exercise the escaping.
                ("has${brace}x", "has\\${brace}x"),
                ("has$ENV{HOME}x", "has\\$ENV{HOME}x"),
            )
            for name, want in cases:
                with self.subTest(path=name):
                    odd = os.path.join(d, name)
                    os.makedirs(odd, exist_ok=True)
                    emitted = self._emit(odd)
                    self.assertIn(f"{d}/{want}/aot_fakeop_cuda.cpp", emitted)
                    # The version-script option, which is the half that was raw.
                    ver = f'\\"{d}/{want}/native_aot_local.ver\\"'
                    self.assertIn(f"-Xlinker {ver}", emitted)

    def test_a_quote_or_backslash_in_a_path_is_refused(self):
        # Not escapable: the version script reaches the linker inside a SHELL:
        # argument, meeting a second level of quoting, and cmake then fails to parse
        # the emitted include -- a syntax error pointing at caffe2/CMakeLists.txt.
        for name in ('has"quote', "has\\back"):
            with self.subTest(path=name), tempfile.TemporaryDirectory() as d:
                odd = os.path.join(d, name)
                os.makedirs(odd)
                with self.assertRaisesRegex(RuntimeError, "cannot embed"):
                    self._emit(odd)

    def test_the_opt_out_is_emitted_not_left_to_cmake(self):
        # Only CMake runs at configure time and a previous run's file can be on
        # disk, so the check has to exist -- but it is generated, which is what
        # keeps caffe2/CMakeLists.txt free of it. CMake truthiness, so OFF/false/no
        # work like 0, matching build_stage2._OPT_OUT_VALUES.
        with tempfile.TemporaryDirectory() as d:
            emitted = self._emit(d)
        # The BLOCK, in order. Three independent substrings would pass with either arm
        # inverted (embedding when the user opted out), with either return() deleted,
        # or with the cache read unconditionally.
        self.assertIn(
            'if(DEFINED ENV{TORCH_NATIVE_AOT} AND NOT "$ENV{TORCH_NATIVE_AOT}" '
            'STREQUAL "")\n'
            '  if(NOT "$ENV{TORCH_NATIVE_AOT}")\n'
            '    message(STATUS "native-AOT: '
            'TORCH_NATIVE_AOT=$ENV{TORCH_NATIVE_AOT}, not embedding kernels")\n'
            "    return()\n"
            "  endif()\n"
            'elseif(DEFINED TORCH_NATIVE_AOT AND NOT "${TORCH_NATIVE_AOT}")\n'
            '  message(STATUS "native-AOT: TORCH_NATIVE_AOT=${TORCH_NATIVE_AOT}, '
            'not embedding kernels")\n'
            "  return()\n"
            "endif()\n",
            emitted,
        )

    def test_invalidation_keeps_the_include_so_a_later_generation_is_seen(self):
        # CMake keeps a configure dependency on an include()d file only if it
        # existed at configure time, so DELETING it made the next generation
        # invisible to `cmake --build`: it relinked without the kernels, reported
        # success, and stayed that way however often the tree was regenerated.
        # Overwriting reads the same to the build and keeps the dependency.
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, gen_aot_lib.CMAKE_INCLUDE)
            with open(path, "w") as f:
                f.write("target_sources(torch_cuda PRIVATE /x/aot_op_cuda.cpp)\n")
            with contextlib.redirect_stdout(io.StringIO()) as said:
                export._invalidate_generation(d)
            self.assertTrue(os.path.exists(path), "the include must survive")
            with open(path) as f:
                text = f.read()
        self.assertEqual(text, gen_aot_lib.NOTHING_TO_EMBED)
        self.assertIn("invalidated", said.getvalue())

    def test_no_prefixes_writes_no_version_script(self):
        # An empty `local:` block is a syntax error to ld, and the path is one an
        # already-configured build.ninja still names through LINK_DEPENDS, so a
        # relink without a reconfigure would fail on a @generated file.
        with tempfile.TemporaryDirectory() as d:
            stale = os.path.join(d, gen_aot_lib.VERSION_SCRIPT)
            with open(stale, "w") as f:
                f.write("{\n  local:\n    old_*;\n};\n")
            path = gen_aot_lib.write_version_script(d, [])
            self.assertEqual(path, stale)
            self.assertFalse(os.path.exists(stale), "a stale script must go too")

    def test_nothing_is_emitted_when_there_are_no_sources(self):
        # No generated source means no launcher references the objects, so there is
        # nothing to embed -- and the file must not name them, or CMake would link
        # objects nothing calls.
        with tempfile.TemporaryDirectory() as d:
            emitted = self._emit(d, sources=[])
        self.assertNotIn("target_sources", emitted)
        self.assertIn("Nothing to embed", emitted)


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
        # The refusal below can only be honest if handed the WHOLE type: split on every
        # comma, it judges the fragment "at::Tensor>" and names that in the error.
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


class TestArchScopedGeneration(unittest.TestCase):
    """Nothing prunes arch trees, so an incremental build whose TORCH_CUDA_ARCH_LIST
    changed still holds the tree for the dropped arch. It must be neither generated
    from nor linked, and must not fail the build."""

    def _tree(self, tmpdir, arch, prefix, stale_sources=False):
        d = os.path.join(tmpdir, arch, "fakeop")
        os.makedirs(d)
        _touch_artifacts(d, prefix)
        rel = _DECL_REL
        digest = (
            "0" * 16
            if stale_sources
            else export._file_hash(os.path.join(export.REPO, rel))
        )
        with open(os.path.join(d, prefix + ".json"), "w") as f:
            json.dump(
                dict(
                    SIDECAR,
                    version=export.SIDECAR_VERSION,
                    prefix=prefix,
                    arch=arch,
                    spec={"N": 1024, "K": 8},
                    sources={rel: digest},
                    runtimes=_RUNTIMES,
                ),
                f,
            )
        return d

    def _generate(self, tmpdir, argv):
        # The ops dir must not live under the artifacts dir: discovery reads every
        # top-level directory there as an arch tree.
        with tempfile.TemporaryDirectory() as ops:
            os.makedirs(os.path.join(ops, "fakeop"), exist_ok=True)
            open(os.path.join(ops, "fakeop", "aot.py"), "w").close()
            with _patched_generation(ops):
                gen_aot_lib.main(["--artifacts-dir", tmpdir, *argv])

    def test_a_declaration_with_cpp_covers_emits_the_predicate(self):
        # main()'s covers assembly: every other covers test calls gen_op directly, and
        # losing it falls back to a covered_axes that carries neither gate.
        class _CoversDecl(_FakeDecl):
            @staticmethod
            def cpp_covers():
                return "return self.scalar_type() == at::kFloat && k == 8;"

        signature = (
            "const at::Tensor & self, int64_t k",
            "covers_fakeop(Tensor self, int k) -> bool",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            self._tree(tmpdir, "sm_100a", "fakeop_p__sm100a")
            with (
                tempfile.TemporaryDirectory() as ops,
                mock.patch.object(
                    gen_aot_lib, "covers_signature", lambda op: signature
                ),
            ):
                os.makedirs(os.path.join(ops, "fakeop"))
                open(os.path.join(ops, "fakeop", "aot.py"), "w").close()
                with _patched_generation(ops, declarations=(_CoversDecl,)):
                    gen_aot_lib.main(["--artifacts-dir", tmpdir])
            with open(os.path.join(tmpdir, "fakeop", "aot_fakeop_cuda.cpp")) as f:
                src = f.read()
        self.assertIn("bool fakeop_cuda_covers(", src)
        self.assertIn('m.def("covers_fakeop(Tensor self, int k) -> bool"', src)

    def test_the_dsl_runtime_reaches_the_emitted_cmake(self):
        # Through main(), not write_cmake_include directly, so both hops are pinned.
        # Nothing links torch_cuda with --no-undefined, so dropping the archive links
        # green and the first AOT call fails on an undefined symbol.
        with tempfile.TemporaryDirectory() as tmpdir:
            self._tree(tmpdir, "sm_100a", "fakeop_p__sm100a")
            archive = os.path.join(tmpdir, "libcuda_dialect_runtime_static.a")
            open(archive, "w").close()
            self._generate(tmpdir, ["--dsl-runtime", archive])
            with open(os.path.join(tmpdir, gen_aot_lib.CMAKE_INCLUDE)) as f:
                emitted = f.read()
        self.assertIn(f'target_link_libraries(torch_cuda PRIVATE "{archive}")', emitted)
        self.assertIn("--exclude-libs", emitted)

    def test_tree_outside_the_arch_list_is_ignored_not_generated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._tree(tmpdir, "sm_100a", "fakeop_p__sm100a")
            self._tree(tmpdir, "sm_90a", "fakeop_p__sm90a")
            self._generate(tmpdir, ["--archs", "sm_100a"])
            with open(os.path.join(tmpdir, "fakeop", "aot_fakeop_cuda.cpp")) as f:
                src = f.read()
            self.assertIn("launch_fakeop_p__sm100a", src)
            self.assertNotIn("sm90a", src)
            # ...and its objects are not in the link set either.
            listed = " ".join(_manifest(tmpdir)["objects"])
            self.assertIn("sm_100a", listed)
            self.assertNotIn("sm_90a", listed)

    def test_object_list_excludes_the_tie_break_loser(self):
        # Every consumer must use the same surviving set, or the object list carries
        # the dropped arch's objects with no launcher referencing them.
        with tempfile.TemporaryDirectory() as tmpdir:
            self._tree(tmpdir, "sm_100a", "fakeop_p__sm100a")
            self._tree(tmpdir, "sm_100", "fakeop_p__sm100")
            self._generate(tmpdir, [])
            with open(os.path.join(tmpdir, "fakeop", "aot_fakeop_cuda.cpp")) as f:
                src = f.read()
            listed = _manifest(tmpdir)["objects"]
            # One launcher, one object, and they are the same kernel.
            self.assertEqual(len(listed), 1, listed)
            self.assertIn("/sm_100a/", listed[0])
            self.assertIn("launch_fakeop_p__sm100a", src)
            for obj in listed:
                prefix = os.path.basename(obj)[: -len(".o")]
                self.assertIn(f"void launch_{prefix}(", src)

    def test_generated_source_is_deleted_when_the_sidecars_are_gone(self):
        # The tree exists but holds no sidecar -- a partial `rm`. Skipping leaves a
        # source referencing entry points whose object nothing listed.
        with tempfile.TemporaryDirectory() as tmpdir:
            tree = self._tree(tmpdir, "sm_100a", "fakeop_p__sm100a")
            self._generate(tmpdir, [])
            out = os.path.join(tmpdir, "fakeop", "aot_fakeop_cuda.cpp")
            self.assertTrue(os.path.exists(out))
            for f in os.listdir(tree):
                if f.endswith(".json"):
                    os.remove(os.path.join(tree, f))
            self._generate(tmpdir, [])
            self.assertFalse(os.path.exists(out), "stale source survived")
            self.assertEqual(_manifest(tmpdir)["sources"], [])

    def test_emitted_paths_are_absolute_from_a_relative_artifacts_dir(self):
        # The documented usage is `--artifacts-dir build/native_aot`, and CMake
        # resolves a relative path against a different directory than the generator.
        with tempfile.TemporaryDirectory() as tmpdir:
            self._tree(tmpdir, "sm_100a", "fakeop_p__sm100a")
            rel = os.path.relpath(tmpdir, os.getcwd())
            self._generate(rel, [])
            man = _manifest(tmpdir)
            self.assertTrue(man["sources"] and man["objects"])
            for p in man["sources"] + man["objects"]:
                self.assertTrue(os.path.isabs(p), f"{p} is not absolute")

    def test_stale_tree_outside_the_arch_list_does_not_fail_the_build(self):
        # The trap: the ignored tree is ALSO stale, so judging its staleness would stop
        # the build to demand a re-export of an arch this build dropped.
        with tempfile.TemporaryDirectory() as tmpdir:
            self._tree(tmpdir, "sm_100a", "fakeop_p__sm100a")
            self._tree(tmpdir, "sm_90a", "fakeop_p__sm90a", stale_sources=True)
            self._generate(tmpdir, ["--archs", "sm_100a"])
            self.assertTrue(
                os.path.exists(os.path.join(tmpdir, "fakeop", "aot_fakeop_cuda.cpp"))
            )

    def test_generated_source_is_deleted_when_no_artifacts_remain(self):
        # Deleting an arch tree by hand left the .cpp behind, and the link glob
        # then compiled a file whose #include "../<arch>/..." no longer resolves.
        with tempfile.TemporaryDirectory() as tmpdir:
            tree = self._tree(tmpdir, "sm_100a", "fakeop_p__sm100a")
            self._generate(tmpdir, [])
            out = os.path.join(tmpdir, "fakeop", "aot_fakeop_cuda.cpp")
            self.assertTrue(os.path.exists(out))
            import shutil

            shutil.rmtree(os.path.dirname(tree))
            self._generate(tmpdir, [])
            self.assertFalse(os.path.exists(out))


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
        with _patched_generation(opsdir):
            gen_aot_lib.main(["--artifacts-dir", tmpdir, *extra_argv])

    def _sc(self, **over):
        # N and K: _FakeDecl.cpp_dispatch reads both grid axes, so a case that
        # gets PAST the refusals can still generate.
        sc = {
            "version": export.SIDECAR_VERSION,
            "prefix": "k__sm100a",
            "kind": "cutedsl",
            "arch": "sm_100a",
            "spec": {"N": 1, "K": 8},
            "tensor_args": [],
        }
        sc.update(over)
        return sc

    def test_prefix_must_be_a_c_identifier(self):
        # The prefix names extern "C" entry points, so the pattern is spelled out rather
        # than \w, which is Unicode-aware and accepts "h\u00e9llo".
        for prefix in ("k-sm100a", "h\u00e9llo__sm100a", "1k__sm100a"):
            with self.subTest(prefix=prefix), tempfile.TemporaryDirectory() as tmpdir:
                with self.assertRaisesRegex(RuntimeError, "not a C identifier"):
                    self._run(tmpdir, self._sc(prefix=prefix))

    def test_schema_version_mismatch_is_not_waivable(self):
        # --allow-stale exists for artifacts whose SOURCES drifted, which still describe
        # themselves readably; a schema bump may not, so forcing past it misreads.
        for argv in ((), ("--allow-stale",)):
            with tempfile.TemporaryDirectory() as tmpdir:
                sc = self._sc(version=export.SIDECAR_VERSION + 1)
                with self.assertRaisesRegex(RuntimeError, "sidecar schema version"):
                    self._run(tmpdir, sc, argv)

    def test_allow_stale_generates_anyway(self):
        # The permissive half of the flag: the sibling test only proves a schema bump
        # ignores it, not that the escape hatch works.
        with tempfile.TemporaryDirectory() as tmpdir:
            sc = self._sc(sources={"tools/native_aot/decl.py": "0" * 16})
            self._run(tmpdir, sc, ("--allow-stale",))
            self.assertTrue(
                glob.glob(os.path.join(tmpdir, "*", "aot_*.cpp")),
                "--allow-stale must still generate",
            )

    def test_stale_error_names_the_arches_and_a_command_that_fixes_them(self):
        # A bare `export.py` re-run maintains only the arch it resolves for, so the
        # message has to name the arches and the --arch invocation.
        with tempfile.TemporaryDirectory() as tmpdir:
            sc = self._sc(sources={"tools/native_aot/decl.py": "0" * 16})
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

    def _art(self, tmpdir, arches=("sm_100a", "sm_90a")):
        """The one layout: kernels at <root>/<arch>/<decl_id>/, and the
        generated source that covers every arch at <root>/<decl_id>/.
        Returns the FIRST arch dir and the source dir, since orphan handling
        touches each differently -- the .cpp is regenerable and gets deleted, the
        kernels never do.

        TWO arch dirs, each holding a differently-named artifact: with one, a
        report that names art_dirs[0] while listing basenames gathered from every
        tree reads exactly like one that names each directory."""
        op = os.path.join(tmpdir, arches[0], "fakeop")
        for arch in arches:
            d = os.path.join(tmpdir, arch, "fakeop")
            os.makedirs(d)
            for ext in (".o", ".h", ".json"):
                with open(os.path.join(d, f"k_{arch}{ext}"), "w") as f:
                    f.write("x")
        src = os.path.join(tmpdir, "fakeop")
        os.makedirs(src)
        with open(os.path.join(src, "aot_fakeop_cuda.cpp"), "w") as f:
            f.write("x")
        return op, src

    def test_a_sidecar_without_a_kind_is_refused(self):
        # The stale check reads the kind first, so a sidecar naming none would be judged
        # by another toolchain's compiler versions. Written by hand.
        with tempfile.TemporaryDirectory() as tmpdir:
            art = os.path.join(tmpdir, "sm_100a", "fakeop")
            os.makedirs(art)
            _touch_artifacts(art, "k__sm100a")
            with open(os.path.join(art, "k__sm100a.json"), "w") as f:
                json.dump(
                    {
                        "version": export.SIDECAR_VERSION,
                        "prefix": "k__sm100a",
                        "arch": "sm_100a",
                    },
                    f,
                )
            opsdir = os.path.join(tmpdir, "_ops")
            os.makedirs(os.path.join(opsdir, "fakeop"))
            open(os.path.join(opsdir, "fakeop", "aot.py"), "w").close()
            with (
                mock.patch.object(gen_aot_lib, "OPS_DIR", opsdir),
                mock.patch.object(
                    gen_aot_lib.decl, "load_declarations", return_value=[_FakeDecl]
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "names no kind"):
                    gen_aot_lib.main(["--artifacts-dir", tmpdir])

    def test_same_prefix_from_two_arch_dirs_is_fatal(self):
        # Prefixes carry their arch, so this needs a copied tree (rsynced artifacts, a
        # hand-made sm_100a.bak). Caught here, not as a redefinition at link time.
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
                            "kind": "cutedsl",
                        },
                        f,
                    )
            # The duplicate check runs after a declaration is matched, so fakeop must
            # be declared. gen_aot_lib imports export inside main(), so patch that.
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

    def test_archs_cannot_be_given_no_values(self):
        # Stage 2 composes `--archs *archs`, so an empty list yields a bare flag and
        # nargs="*" assigns [], which would pass every tree through the filter.
        with tempfile.TemporaryDirectory() as d:
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    gen_aot_lib.main(["--artifacts-dir", d, "--archs"])

    def test_no_declarations_at_all_leaves_the_artifacts_alone(self):
        # A commit earlier in the stack (or a bisect) declares nothing; that
        # is not the same as every artifact being orphaned.
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            tempfile.TemporaryDirectory() as opsdir,
        ):
            op, src = self._art(tmpdir)
            # An EMPTY ops dir: "declares nothing" must not depend on which
            # commit of the stack is checked out.
            with (
                mock.patch.object(gen_aot_lib, "OPS_DIR", opsdir),
                contextlib.redirect_stdout(io.StringIO()) as printed,
            ):
                gen_aot_lib.main(["--artifacts-dir", tmpdir])
            left = sorted(os.listdir(op))
            src_left = sorted(os.listdir(src))
            said = printed.getvalue()
        # The ARTIFACTS are what must survive a bisect -- they cost a full export.
        self.assertIn("k_sm_100a.o", left)
        self.assertIn("k_sm_100a.h", left)
        # The generated source must not survive: this run emitted a "nothing to embed"
        # include, so a leftover .cpp has stage 2 report kernels and fail the build.
        self.assertNotIn("aot_fakeop_cuda.cpp", src_left)
        # ...and nothing is REPORTED as undeclared: these artifacts predate the
        # declarations rather than being orphaned by them.
        self.assertNotIn("with no declaration", said)

    def test_orphan_with_other_declarations_is_reported_and_keeps_artifacts(self):
        # With declarations present an unclaimed dir is a real orphan: drop the generated
        # .cpp, keep the objects, and report. Not fatal, since the emitted CMake names
        # exact paths so an unnamed artifact cannot be linked anyway.
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
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    gen_aot_lib.main(["--artifacts-dir", tmpdir])
            # Names the directory the files are actually IN, or a multi-arch orphan
            # points at files that are not in the directory named.
            said = out.getvalue()
            self.assertIn("no declaration", said)
            # EVERY arch dir named alongside ITS OWN artifacts, so "delete this
            # directory" does not leave the other trees behind.
            for arch in ("sm_100a", "sm_90a"):
                one = os.path.join(tmpdir, arch, "fakeop")
                line = next(l for l in said.splitlines() if l.startswith(one))
                self.assertIn(f"k_{arch}.o", line)
                other = "sm_90a" if arch == "sm_100a" else "sm_100a"
                self.assertNotIn(f"k_{other}.o", line)
            left = sorted(os.listdir(op))
            src_left = sorted(os.listdir(src))
        self.assertIn("k_sm_100a.o", left)
        # The regenerable source is dropped from <root>/<decl_id>/, so nothing compiles
        # it against a stub that no longer exists.
        self.assertNotIn("aot_fakeop_cuda.cpp", src_left)


class _OtherDecl(_FakeDecl):
    """A declaration whose decl_id does not match the artifact dir above."""

    ATEN_OP = "otherop"


class TestMissingArtifacts(unittest.TestCase):
    def test_missing_artifacts_reexport_despite_current_sidecar(self):
        # Sidecar matches spec/arch/sources, but its .o is gone: skipping
        # here surfaces as a missing include when torch_cuda compiles.
        point = {"dtype": "float32", "N": 4096}
        with tempfile.TemporaryDirectory() as d:
            job = ("fakeop", "aot_kernel.py", point, d, None)
            _write_sidecar(d, point)
            with _no_ambient_arch():
                self.assertFalse(export._job_needed(job, force=False))

                os.remove(os.path.join(d, "x.o"))
                self.assertTrue(export._job_needed(job, force=False))

    def test_missing_header_also_reexports(self):
        point = {"dtype": "float32", "N": 4096}
        with tempfile.TemporaryDirectory() as d:
            job = ("fakeop", "aot_kernel.py", point, d, None)
            _write_sidecar(d, point, exts=(".o",))  # .h missing
            # _no_ambient_arch: the job resolves arch=None, so an exported
            # CUTE_DSL_ARCH would refuse rather than answer.
            with _no_ambient_arch():
                self.assertTrue(export._job_needed(job, force=False))


if __name__ == "__main__":
    unittest.main()
