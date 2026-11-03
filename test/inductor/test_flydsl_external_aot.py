# Owner(s): ["module: inductor"]

import tempfile
from pathlib import Path

from torch._inductor.codegen.external_aot import (
    ExternalAOTBuildSpec,
    ExternalAOTTensorArg,
)
from torch._inductor.codegen.flydsl.aot_backend import (
    FlyDSLAOTBackend,
    FlyDSLAOTExportRequest,
)
from torch._inductor.runtime.flydsl_cache import (
    capture_flydsl_compiles,
    compile_flydsl,
    is_capturing_flydsl_compiles,
)
from torch._inductor.test_case import TestCase
from torch.testing._internal.common_utils import HardwareClassification


class TestFlyDSLExternalAOT(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_compile_capture_is_context_local(self):
        calls = []

        def compiler(jit_func, *args, **kwargs):
            return (jit_func, args, kwargs)

        def observer(compiler_fn, jit_func, args, kwargs):
            calls.append((jit_func, args, kwargs))
            return compiler_fn(jit_func, *args, **kwargs)

        self.assertFalse(is_capturing_flydsl_compiles())
        with capture_flydsl_compiles(observer):
            self.assertTrue(is_capturing_flydsl_compiles())
            result = compile_flydsl(compiler, "kernel", 1, option=2)
        self.assertFalse(is_capturing_flydsl_compiles())
        self.assertEqual(calls, [("kernel", (1,), {"option": 2})])
        self.assertEqual(result, ("kernel", (1,), {"option": 2}))

    def test_export_header_validation(self):
        spec = ExternalAOTBuildSpec(
            kind="flydsl",
            kernel_name="fly_kernel",
            target="gfx950",
            source_hash="source",
            tensor_args=(
                ExternalAOTTensorArg(
                    name="out",
                    source="output",
                    dtype="float32",
                    dynamic_sizes=(0,),
                    dynamic_strides=(0,),
                ),
            ),
        )

        class Compiled:
            def export_to_c(self, *, file_path, file_name, function_prefix):
                root = Path(file_path)
                (root / f"{file_name}.o").write_bytes(b"fly-object")
                (root / f"{file_name}.h").write_text(f"""
typedef struct __attribute__((packed)) {{
  int32_t shape[1];
  int64_t stride[1];
}} {function_prefix}_out_layout_t;
static inline int32_t {function_prefix}_call_auto(
    void* out_data,
    const {function_prefix}_out_layout_t* out_layout,
    void* stream);
""")

        backend = FlyDSLAOTBackend()
        with tempfile.TemporaryDirectory() as directory:
            artifact = backend.export(
                FlyDSLAOTExportRequest(
                    compiled=Compiled(),
                    prefix="fly_kernel",
                ),
                spec,
                directory,
            )
            backend.validate_abi(artifact, spec)
            self.assertEqual(
                artifact.link_args,
                ("-l:libamdhip64.so", "-pthread", "-ldl"),
            )
            header_path = Path(artifact.header_path)
            header_path.write_text(
                header_path.read_text().replace(
                    "void* out_data,\n    const fly_kernel_out_layout_t* out_layout",
                    "const fly_kernel_out_layout_t* out_layout,\n    void* out_data",
                )
            )
            with self.assertRaisesRegex(RuntimeError, "parameters"):
                backend.validate_abi(artifact, spec)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
