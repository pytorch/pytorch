# Owner(s): ["module: inductor"]

import zipfile
from unittest.mock import patch

import torch
from torch._C import FileCheck
from torch._inductor import config, decomposition, lowering
from torch._inductor.test_case import TestCase
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    parametrize,
    run_tests,
)


try:
    from .test_aot_inductor_utils import AOTIRunnerUtil
except ImportError:
    from test_aot_inductor_utils import AOTIRunnerUtil


class NativeLayerNormModule(torch.nn.Module):
    def __init__(self, device, dtype, affine):
        super().__init__()
        self.register_buffer(
            "weight",
            torch.randn(3, 16, device=device, dtype=dtype)
            if affine in ("both", "weight")
            else None,
        )
        self.register_buffer(
            "bias",
            torch.randn(3, 16, device=device, dtype=dtype)
            if affine in ("both", "bias")
            else None,
        )

    def forward(self, x):
        return torch.ops.aten.native_layer_norm.default(
            x, (3, 16), self.weight, self.bias, 1e-4
        )


class NativeLayerNormCShimTests(TestCase):
    hw_classification = HardwareClassification.CPU

    @dtypes(torch.float32, torch.bfloat16, torch.float16)
    @parametrize("affine", ["none", "weight", "bias", "both"])
    @config.patch({"fx_graph_cache": False, "fx_graph_remote_cache": False})
    def test_native_layer_norm_c_shim(self, device, dtype, affine):
        model = NativeLayerNormModule(device, dtype, affine).eval()
        x = torch.randn(2, 16, 3, device=device, dtype=dtype).transpose(1, 2)
        native_layer_norm = torch.ops.aten.native_layer_norm.default
        decomps = dict(decomposition.select_decomp_table())
        decomps.pop(native_layer_norm, None)
        # Exercise the shim without changing the default LayerNorm lowering policy.
        with (
            torch.no_grad(),
            patch(
                "torch._inductor.compile_fx.select_decomp_table", return_value=decomps
            ),
            patch.dict(
                lowering.lowerings,
                {native_layer_norm: lowering.fallback_handler(native_layer_norm)},
            ),
        ):
            package = AOTIRunnerUtil.compile(
                model,
                (x,),
                dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1, max=16)}},
            )
        with zipfile.ZipFile(package) as archive:
            code = "\n".join(
                archive.read(name).decode()
                for name in archive.namelist()
                if name.endswith((".cpp", ".h"))
            )
        FileCheck().check("aoti_torch_cpu_native_layer_norm(").run(code)
        FileCheck().check_not("aoti_torch_proxy_executor_call_function").run(code)
        loaded = AOTIRunnerUtil.load_package(package)
        with torch.no_grad():
            for batch in (2, 5, 9):
                inputs = torch.randn(
                    batch, 16, 3, device=device, dtype=dtype
                ).transpose(1, 2)
                actual = loaded(inputs)
                expected = model(inputs)
                self.assertEqual(actual, expected, atol=0, rtol=0, exact_dtype=True)
                self.assertEqual(
                    [t.contiguous().view(torch.uint8) for t in actual],
                    [t.contiguous().view(torch.uint8) for t in expected],
                )


instantiate_device_type_tests(NativeLayerNormCShimTests, globals(), only_for="cpu")

if __name__ == "__main__":
    run_tests()
