# Owner(s): ["module: inductor"]

import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from torch._inductor.codegen.common import IndentedBuffer
from torch._inductor.codegen.cutedsl.aot import (
    CuteDSLAOTBackend,
    CuteDSLAOTExportRequest,
)
from torch._inductor.codegen.external_aot import (
    export_external_aot_files,
    ExternalAOTArtifact,
    ExternalAOTBuildSpec,
    ExternalAOTTensorArg,
    publish_external_aot_artifact,
)
from torch._inductor.test_case import TestCase
from torch._inductor.utils import OrderedSet
from torch.testing._internal.common_utils import HardwareClassification


class TestExternalAOT(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def _spec(self):
        return ExternalAOTBuildSpec(
            kind="test",
            kernel_name="kernel",
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

    def test_sidecar_hashes_and_graph_local_publication(self):
        spec = self._spec()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            object_path = root / "kernel.o"
            header_path = root / "kernel.h"
            object_path.write_bytes(b"object")
            header_path.write_text("header")
            artifact = ExternalAOTArtifact.from_files(
                spec=spec,
                object_path=object_path,
                header_path=header_path,
                link_args=(
                    "-Wl,--whole-archive",
                    "libbackend.a",
                    "-Wl,--no-whole-archive",
                ),
                payload={"symbol": "kernel"},
                runtime_versions={"test-dsl": "1.0"},
            )
            artifact.validate_files()
            second_object = root / "second.o"
            second_header = root / "second.h"
            second_object.write_bytes(b"object")
            second_header.write_text("different header")
            second_artifact = ExternalAOTArtifact.from_files(
                spec=spec,
                object_path=second_object,
                header_path=second_header,
                link_args=(
                    "-Wl,--whole-archive",
                    "libbackend.a",
                    "-Wl,--no-whole-archive",
                ),
                payload={"symbol": "kernel"},
                runtime_versions={"test-dsl": "1.0"},
            )
            self.assertNotEqual(artifact.content_hash, second_artifact.content_hash)

            wrapper = SimpleNamespace(
                additional_files=[],
                aot_kernel_objects=OrderedSet(),
                aot_link_args=[],
                deferred_aot_call_wrappers={},
                external_aot_artifacts={},
            )
            deferred = object()
            published = publish_external_aot_artifact(
                wrapper, "kernel", artifact, deferred
            )
            self.assertIs(published, artifact)
            self.assertEqual(wrapper.aot_kernel_objects, {str(object_path)})
            self.assertEqual(
                wrapper.aot_link_args,
                ["-Wl,--whole-archive", "libbackend.a", "-Wl,--no-whole-archive"],
            )
            self.assertEqual(wrapper.additional_files, [])
            self.assertIs(wrapper.deferred_aot_call_wrappers["kernel"], deferred)

            object_path.write_bytes(b"stale")
            with self.assertRaisesRegex(RuntimeError, "stale external AOT object"):
                artifact.validate_files()

    def test_build_spec_digest(self):
        spec = self._spec()
        self.assertEqual(spec.digest, self._spec().digest)
        changed = replace(spec, target="gfx1250")
        self.assertNotEqual(spec.digest, changed.digest)
        changed_dtype = replace(
            spec,
            tensor_args=(replace(spec.tensor_args[0], dtype="float16"),),
        )
        self.assertNotEqual(spec.digest, changed_dtype.digest)

    def test_export_files_are_published_once(self):
        exports = []

        def exporter(directory):
            exports.append(directory)
            (directory / "kernel.o").write_bytes(b"object")
            (directory / "kernel.h").write_text("header")

        with tempfile.TemporaryDirectory() as directory:
            with ThreadPoolExecutor(max_workers=2) as pool:
                paths = list(
                    pool.map(
                        lambda _: export_external_aot_files(
                            directory,
                            "kernel",
                            exporter,
                        ),
                        range(2),
                    )
                )
            self.assertEqual(len(exports), 1)
            self.assertEqual(paths[0], paths[1])

    def test_cutedsl_export_header_validation_and_launcher(self):
        spec = ExternalAOTBuildSpec(
            kind="cutedsl",
            kernel_name="cute_kernel",
            target="sm_90",
            source_hash="source",
            tensor_args=(
                ExternalAOTTensorArg(
                    name="x",
                    source="x",
                    dtype="float32",
                    dynamic_sizes=(0,),
                    dynamic_strides=(0,),
                ),
            ),
        )

        class Compiled:
            def export_to_c(self, *, file_path, file_name, function_prefix):
                root = Path(file_path)
                (root / f"{file_name}.o").write_bytes(b"cute-object")
                (root / f"{file_name}.h").write_text(f"""
typedef struct {{
  void* data;
  int32_t dynamic_shapes[1];
  int64_t dynamic_strides[1];
}} {function_prefix}_Tensor_x_t;
typedef struct {{ void* handle; }} {function_prefix}_Kernel_Module_t;
void {function_prefix}_Kernel_Module_Load(
    {function_prefix}_Kernel_Module_t*);
int32_t cute_dsl_{function_prefix}_wrapper(
    {function_prefix}_Kernel_Module_t*,
    {function_prefix}_Tensor_x_t*,
    cudaStream_t);
""")

        backend = CuteDSLAOTBackend()
        with tempfile.TemporaryDirectory() as directory:
            artifact = backend.export(
                CuteDSLAOTExportRequest(
                    compiled=Compiled(),
                    prefix="cute_kernel",
                ),
                spec,
                directory,
            )
            backend.validate_abi(artifact, spec)
            wrapper = SimpleNamespace(
                prefix=IndentedBuffer(),
                include_extra_header=lambda _header: None,
            )
            backend.make_deferred_call(artifact).generate(wrapper)
            code = wrapper.prefix.getvalue()
            self.assertIn("cute_kernel_Tensor_x_t", code)
            self.assertIn("std::call_once", code)
            self.assertIn("dynamic_shapes[0]", code)
            self.assertIn("dynamic_strides[0]", code)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
