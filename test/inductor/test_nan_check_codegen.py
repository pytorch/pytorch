# Owner(s): ["module: inductor"]

from types import SimpleNamespace

import sympy

import torch
from torch._inductor.codegen.common import SizeArg, TensorArg
from torch._inductor.codegen.triton import TritonKernel
from torch._inductor.test_case import TestCase
from torch._inductor.virtualized import V
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)


@instantiate_parametrized_tests
class TestNanCheckCodegen(TestCase):
    hw_classification = HardwareClassification.GENERIC

    @parametrize("cpp_wrapper", [False, True])
    def test_nan_checks_identify_kernel_and_buffer(self, cpp_wrapper):
        lines = []
        kernel = SimpleNamespace(
            kernel_name="triton_poi_fused_div_7",
            args=SimpleNamespace(
                python_argdefs=lambda: (
                    [],
                    ["arg0_1", "buf0", "s0"],
                    [
                        TensorArg("in_ptr0", "arg0_1", torch.float32),
                        TensorArg("out_ptr0", "buf0", torch.float32),
                        SizeArg("ks0", sympy.Symbol("s0")),
                    ],
                    [],
                )
            ),
        )
        graph = SimpleNamespace(
            cpp_wrapper=cpp_wrapper,
            wrapper_code=SimpleNamespace(writeline=lines.append),
        )
        with V.set_graph_handler(graph):
            TritonKernel.codegen_nan_check(kernel)
        if cpp_wrapper:
            expected = [
                f'AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_check_inf_and_nan("triton_poi_fused_div_7: {name}", {name}));'
                for name in ["arg0_1", "buf0"]
            ]
        else:
            expected = [
                f"assert not {name}.{check}().any().item(), 'triton_poi_fused_div_7: {name} contains {issue}'"
                for name in ["arg0_1", "buf0"]
                for check, issue in [("isnan", "NaN"), ("isinf", "Inf")]
            ]
        self.assertEqual(lines, expected)


if __name__ == "__main__":
    run_tests()
