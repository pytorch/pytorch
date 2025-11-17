# Owner(s): ["module: inductor"]
import tempfile
import unittest
import zipfile

import torch
import torch._inductor.config
from torch._environment import is_fbcode
from torch._inductor.test_case import TestCase
from torch.testing._internal.common_utils import IS_CI
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
    def test_simple_so(self):
        if is_fbcode() or IS_CI:
            raise unittest.SkipTest("requires x86_64-w64-mingw32-gcc")

        # TODO: enable in CI
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
                    "aot_inductor.aoti_shim_library": "executorch",
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

    if HAS_GPU:
        run_tests(needs="filelock")
