from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest


TOOLS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "native_aot")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(TOOLS_DIR, f"{name}.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


export = _load("export")
gen_aot_lib = _load("gen_aot_lib")
toolchains = _load("toolchains")


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
            job = ("fakeop", "aot_kernel.py", point, d)
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
            other = ("fakeop", "aot_kernel.py", {"dtype": "bfloat16"}, d)
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
            job = ("fakeop", "aot_kernel.py", point, d)
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
        # The spawn pool pickles the job function by qualified name; a
        # closure or nested function would break only at --jobs > 1.
        # (A real pickle round-trip needs export importable by module
        # name, which this by-path test harness can't provide.)
        self.assertEqual(export._run_job.__qualname__, "_run_job")
        self.assertEqual(export.export_point.__qualname__, "export_point")


class TestArch(unittest.TestCase):
    def test_arch_gate_from_declaration_and_sidecars(self):
        class Pinned(_FakeDecl):
            ARCHS = ("sm_90a", "sm_100a")

        # Shipped subset gates on the subset.
        gate = gen_aot_lib._arch_gate(Pinned, [{"arch": "sm_100a"}])
        self.assertIn("major == 10", gate)
        self.assertNotIn("major == 9", gate)
        # On-device sidecars (no arch) gate on the declaration's ARCHS.
        gate = gen_aot_lib._arch_gate(Pinned, [{"arch": None}])
        self.assertIn("major == 9", gate)
        self.assertIn("major == 10", gate)
        # Shipped arch outside ARCHS is a packaging error.
        with self.assertRaisesRegex(RuntimeError, "supports only"):
            gen_aot_lib._arch_gate(Pinned, [{"arch": "sm_80"}])



class TestSpecExpansion(unittest.TestCase):
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
            "cute_dsl_fakeop_f32_n1024_k8_wrapper(&fakeop_f32_n1024_k8_module, &mX_s, &mOut_s, stream)",
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
        self.assertIn("const at::Tensor& mX, float alpha, cudaStream_t stream", src)
        self.assertIn("(&fakeop_f32_n1024_k8_module, &mX_s, alpha, stream)", src)

    def test_no_scalar_args_unchanged(self):
        src = gen_aot_lib.gen_launcher(SIDECAR)
        self.assertIn(
            "const at::Tensor& mX, const at::Tensor& mOut, cudaStream_t stream", src
        )


class _FakeDecl:
    """Minimal declaration object for gen_op tests (the real contract
    lives in decl.py; gen_op only touches these attributes)."""

    ATEN_OP = "fakeop"
    DISPATCH_KEY = "CUDA"
    # Set explicitly: fixtures bypass the validating loader, which is
    # what normalizes ARCHS on real declarations.
    ARCHS = ("sm_90", "sm_90a", "sm_100", "sm_100a")

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
        # Falls through to false; registers on the generated DispatchStub.
        self.assertIn("return false;", src)
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
            cpp_dispatch_prelude = None

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


class TestReadOnlyInputs(unittest.TestCase):
    # read_only tensor args must go through const_data_ptr in every
    # toolchain's launcher: a mutable data_ptr() materializes
    # copy-on-write inputs on each call.

    def test_cutedsl_launcher(self):
        sc = dict(SIDECAR)
        sc["tensor_args"] = [
            {"name": "mX", "dynamic_sizes": [0], "read_only": True},
            {"name": "mOut", "dynamic_sizes": [0]},
        ]
        src = gen_aot_lib.gen_launcher(sc)
        self.assertIn("mX_s.data = const_cast<void*>(mX.const_data_ptr());", src)
        self.assertIn("mOut_s.data = mOut.data_ptr();", src)


class TestSourceStaleness(unittest.TestCase):
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
            job = ("fakeop", "aot_kernel.py", point, d)
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
        cmake_path = os.path.join(TOOLS_DIR, "..", "..", "caffe2", "CMakeLists.txt")
        with open(cmake_path) as f:
            cmake = f.read()
        for tc in toolchains.TOOLCHAINS.values():
            for pattern in tc.link_source_globs:
                self.assertIn(pattern.split("/")[-1], cmake)


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
            os.makedirs(os.path.join(art, "fakeop"))
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
            with open(
                os.path.join(art, "fakeop", SIDECAR["prefix"] + ".json"), "w"
            ) as f:
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


if __name__ == "__main__":
    unittest.main()
