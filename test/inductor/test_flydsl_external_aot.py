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
from torch._inductor.test_case import TestCase


class TestFlyDSLExternalAOT(TestCase):
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
                    dynamic_sizes=(0,),
                    dynamic_strides=(0,),
                    read_only=False,
                    mutated=True,
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
            self.assertEqual(artifact.link_libraries, ("libamdhip64.so",))
            self.assertEqual(
                artifact.link_options,
                ("-lamdhip64", "-pthread", "-ldl"),
            )


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
