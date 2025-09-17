# Owner(s): ["module: inductor"]
import sys
import tempfile
import zipfile

import torch
import torch._inductor.config
from torch._inductor.test_case import TestCase
from torch.testing import FileCheck
from torch.testing._internal.inductor_utils import HAS_GPU, requires_gpu


class Simple(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 16)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(16, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class TestAOTInductorWindowsCrossCompilation(TestCase):
    @requires_gpu()
    def test_simple_cpp_only(self):
        # rm -r /tmp/torchinductor_shangdiy/
        with torch.no_grad():
            device = "cuda"
            model = Simple().to(device=device)
            example_inputs = (torch.randn(8, 10, device=device),)
            batch_dim = torch.export.Dim("batch", min=1, max=1024)
            exported = torch.export.export(
                model, example_inputs, dynamic_shapes={"x": {0: batch_dim}}
            )
            package_path = torch._inductor.aoti_compile_and_package(
                exported,
                inductor_configs={
                    "aot_inductor.model_name_for_generated_files": "model",
                    "aot_inductor.package_cpp_only": True,
                    "aot_inductor.cross_target_platform": "windows",
                    "aot_inductor.link_libtorch": False,
                    # no fallback ops
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "TRITON,CPP",
                    "max_autotune_conv_backends": "TRITON,CPP",
                },
            )

            with tempfile.TemporaryDirectory() as tmpdir:
                with zipfile.ZipFile(package_path, "r") as zf:
                    zf.extractall(tmpdir)

                makefile = open(
                    f"{tmpdir}/model.wrapper/data/aotinductor/model/CMakeLists.txt"
                )
                makefile_content = makefile.read()

                FileCheck().check("add_library(model SHARED)").check(
                    "target_compile_definitions(model PRIVATE NOMINMAX "
                ).check("USE_CUDA").check("target_compile_options(model").check(
                    """set_target_properties(model PROPERTIES SUFFIX ".pyd" """
                ).check(
                    """LINK_FLAGS "/DEF:${CMAKE_CURRENT_SOURCE_DIR}/windows_symbol_exports.def" )"""
                ).check(
                    "target_sources(model PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/model.wrapper.cpp)"
                ).check(
                    "target_sources(model PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/model_consts.weights.cpp)"
                ).check("embed_gpu_kernel(").check(
                    "add_dependencies(model ${KERNEL_TARGETS})"
                ).check(
                    "target_link_libraries(model PRIVATE ${KERNEL_OBJECT_FILES})"
                ).check(
                    "target_link_options(model PRIVATE )"  # no libtorch
                ).check("target_link_libraries(model PRIVATE CUDA::cudart cuda)").run(
                    makefile_content
                )

                # TODO: actually compile the package in the test later in windows CI

    @requires_gpu()
    def test_simple_so(self):
        with torch.no_grad():
            device = "cuda"
            model = Simple().to(device=device)
            example_inputs = (torch.randn(8, 10, device=device),)
            batch_dim = torch.export.Dim("batch", min=1, max=1024)
            exported = torch.export.export(
                model, example_inputs, dynamic_shapes={"x": {0: batch_dim}}
            )
            package_path = torch._inductor.aoti_compile_and_package(
                exported,
                inductor_configs={
                    "aot_inductor.model_name_for_generated_files": "model",
                    "aot_inductor.cross_target_platform": "windows",
                    "aot_inductor.link_libtorch": False,
                    # no fallback ops
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "TRITON,CPP",
                    "max_autotune_conv_backends": "TRITON,CPP",
                    # simplify things for now
                    "aot_inductor.precompile_headers": False,
                },
            )

            with tempfile.TemporaryDirectory() as tmpdir:
                with zipfile.ZipFile(package_path, "r") as zf:
                    zf.extractall(tmpdir)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_GPU or sys.platform == "darwin":
        run_tests(needs="filelock")
