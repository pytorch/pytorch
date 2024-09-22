# Owner(s): ["module: inductor"]

import logging

import torch
import torch._inductor
from torch._dynamo.utils import counters
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing import FileCheck
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_CUDA
from torch.testing._internal.triton_utils import requires_gpu


class MyModule(torch.nn.Module):
    def __init__(
        self, n_input: int, n_output: int, has_bias: bool, device=GPU_TYPE
    ) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(n_input, n_output, bias=has_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MyModule2(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input1, input2):
        output = torch.bmm(input1, input2)
        return output


class MyModule3(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input1, input2):
        output = torch.mm(input1, input2)
        return output


@requires_gpu
@torch._inductor.config.patch(
    post_grad_fusion_options={
        "decompose_mm_pass": {},
    }
)
@instantiate_parametrized_tests
class TestDecomposeMemMM(TestCase):
    def compare_dict_tensors(self, ref_dict, res_dict, rtol=1e-3, atol=1e-3):
        if len(set(ref_dict.keys())) != len(set(res_dict.keys())):
            return False
        for key1 in ref_dict.keys():
            key2 = "_orig_mod." + key1
            assert key2 in res_dict, f"{key1} does not exist in traced module"
            if not torch.allclose(ref_dict[key1], res_dict[key2], rtol=rtol, atol=atol):
                return False
        return True

    def compare_pred(self, module, traced, input, rtol=1e-3, atol=1e-3):
        ref = module(*input)
        res = traced(*input)
        self.assertEqual(ref, res, rtol=rtol, atol=atol)

    def compare_parameters(self, module, traced, rtol=1e-3, atol=1e-3):
        ref_params = dict(module.named_parameters())
        res_params = dict(traced.named_parameters())
        self.assertTrue(self.compare_dict_tensors(ref_params, res_params, rtol, atol))

    def compare_gradients(self, module, traced, rtol=1e-3, atol=1e-3):
        ref_grad = {key: param.grad for key, param in module.named_parameters()}
        res_grad = {key: param.grad for key, param in traced.named_parameters()}
        self.assertTrue(
            self.compare_dict_tensors(ref_grad, res_grad, rtol=rtol, atol=atol)
        )

    @parametrize(
        "b,m,k,n,should_decompose",
        [(10240, 2, 2, 2, True), (10240, 2, 32, 32, False), (2000, 2, 2, 2, False)],
    )
    def test_decompose_bmm(self, b, m, n, k, should_decompose):
        torch._logging.set_logs(inductor=logging.DEBUG)
        mat1 = torch.randn(b, m, k, device=GPU_TYPE).requires_grad_(True)
        mat2 = torch.randn(b, k, n, device=GPU_TYPE).requires_grad_(True)

        counters.clear()

        module = MyModule2().to(GPU_TYPE)
        traced = torch.compile(module)
        input = [mat1, mat2]
        ref = module(*input)
        res = traced(*input)

        self.compare_pred(module, traced, input)

        expected_val = 1 if should_decompose and HAS_CUDA else 0
        self.assertEqual(
            counters["inductor"]["decompose_bmm"],
            expected_val,
        )

        ref.sum().backward()
        res.sum().backward()
        self.compare_parameters(module, traced)
        self.compare_gradients(module, traced)

        expected_val = 3 if should_decompose and HAS_CUDA else 0
        self.assertEqual(
            counters["inductor"]["decompose_bmm"],
            expected_val,
        )
        counters.clear()

    @parametrize(
        "m,k,n, should_decompose",
        [(20480, 5, 2, True), (20480, 32, 2, False), (2048, 2, 2, False)],
    )
    @parametrize("has_bias", [True, False])
    def test_decompose_linear(self, m, n, k, has_bias, should_decompose):
        torch._logging.set_logs(inductor=logging.DEBUG)
        input = torch.randn(m, k, device=GPU_TYPE).requires_grad_(True)

        counters.clear()

        module = MyModule(k, n, has_bias).to(GPU_TYPE)
        traced = torch.compile(module)
        input = [input]
        ref = module(*input)
        res = traced(*input)

        self.compare_pred(module, traced, input)

        expected_val = 1 if should_decompose and HAS_CUDA else 0
        if has_bias:
            self.assertEqual(
                counters["inductor"]["decompose_addmm"],
                expected_val,
            )
        else:
            self.assertEqual(
                counters["inductor"]["decompose_mm"],
                expected_val,
            )
        decompose_mm_fwd = counters["inductor"]["decompose_mm"]

        ref.sum().backward()
        res.sum().backward()

        self.compare_parameters(module, traced)
        self.compare_gradients(module, traced)

        self.assertEqual(
            counters["inductor"]["decompose_mm"] - decompose_mm_fwd,
            expected_val,
        )
        counters.clear()

    @parametrize(
        "m,k,n, should_decompose",
        [(20480, 5, 2, True), (20480, 32, 2, False), (2048, 2, 2, False)],
    )
    @parametrize("has_bias", [True, False])
    def test_decompose_linear_mixed_precision(
        self, m, n, k, has_bias, should_decompose
    ):
        with torch.amp.autocast(device_type=GPU_TYPE, dtype=torch.bfloat16):
            torch._logging.set_logs(inductor=logging.DEBUG)
            input = torch.randn(m, k, device=GPU_TYPE).requires_grad_(True)

            counters.clear()

            module = MyModule(k, n, has_bias).to(GPU_TYPE)
            traced = torch.compile(module)
            input = [input]
            ref = module(*input)
            res = traced(*input)

            self.compare_pred(module, traced, input)

            expected_val = 1 if should_decompose and HAS_CUDA else 0
            if has_bias:
                self.assertEqual(
                    counters["inductor"]["decompose_addmm"],
                    expected_val,
                )
            else:
                self.assertEqual(
                    counters["inductor"]["decompose_mm"],
                    expected_val,
                )
            decompose_mm_fwd = counters["inductor"]["decompose_mm"]

            ref.sum().backward()
            res.sum().backward()

            self.compare_parameters(module, traced)
            self.compare_gradients(module, traced)

            self.assertEqual(
                counters["inductor"]["decompose_mm"] - decompose_mm_fwd,
                expected_val,
            )
            counters.clear()

    @parametrize(
        "m,k,n, should_decompose",
        [(20480, 5, 2, True), (20480, 32, 2, False), (2048, 2, 2, False)],
    )
    @parametrize("has_bias", [True, False])
    def test_decompose_mm(self, m, n, k, has_bias, should_decompose):
        torch._logging.set_logs(inductor=logging.DEBUG)
        mat1 = torch.randn(m, k, device=GPU_TYPE).requires_grad_(True)
        mat2 = torch.randn(k, n, device=GPU_TYPE).requires_grad_(True)

        counters.clear()

        module = MyModule3().to(GPU_TYPE)
        traced = torch.compile(module)
        input = [mat1, mat2]
        ref = module(*input)
        res = traced(*input)

        self.compare_pred(module, traced, input)

        expected_val = 1 if should_decompose and HAS_CUDA else 0
        self.assertEqual(
            counters["inductor"]["decompose_mm"],
            expected_val,
        )
        decompose_mm_fwd = counters["inductor"]["decompose_mm"]

        ref.sum().backward()
        res.sum().backward()
        self.compare_parameters(module, traced)
        self.compare_gradients(module, traced)

        expected_val = 1 if should_decompose and HAS_CUDA else 0
        self.assertEqual(
            counters["inductor"]["decompose_mm"] - decompose_mm_fwd,
            expected_val,
        )
        counters.clear()

    @parametrize(
        "m,k,n, should_decompose",
        [(20480, 5, 2, True), (20480, 32, 2, False), (2048, 2, 2, False)],
    )
    @parametrize("has_bias", [True, False])
    def test_decompose_mm_mixed_precision(self, m, n, k, has_bias, should_decompose):
        with torch.amp.autocast(device_type=GPU_TYPE, dtype=torch.bfloat16):
            torch._logging.set_logs(inductor=logging.DEBUG)
            mat1 = torch.randn(m, k, device=GPU_TYPE).requires_grad_(True)
            mat2 = torch.randn(k, n, device=GPU_TYPE).requires_grad_(True)

            counters.clear()

            module = MyModule3().to(GPU_TYPE)
            traced = torch.compile(module)
            input = [mat1, mat2]
            ref = module(*input)
            res = traced(*input)

            self.compare_pred(module, traced, input)

            expected_val = 1 if should_decompose and HAS_CUDA else 0
            self.assertEqual(
                counters["inductor"]["decompose_mm"],
                expected_val,
            )
            decompose_mm_fwd = counters["inductor"]["decompose_mm"]

            ref.sum().backward()
            res.sum().backward()
            self.compare_parameters(module, traced)
            self.compare_gradients(module, traced)

            expected_val = 1 if should_decompose and HAS_CUDA else 0
            self.assertEqual(
                counters["inductor"]["decompose_mm"] - decompose_mm_fwd,
                expected_val,
            )
            counters.clear()

    @parametrize("m,k,n, should_decompose", [(20480, 5, 2, True)])
    @parametrize("has_bias", [True, False])
    def test_dynamic_shape(self, m, n, k, has_bias, should_decompose):
        torch._logging.set_logs(inductor=logging.DEBUG)
        input = torch.randn(m, k, device=GPU_TYPE).requires_grad_(True)

        counters.clear()

        module = MyModule(k, n, has_bias).to(GPU_TYPE)
        traced = torch.compile(module, dynamic=True)
        input = [input]
        ref = module(*input)
        res = traced(*input)

        self.compare_pred(module, traced, input)

        expected_val = 1 if should_decompose and HAS_CUDA else 0
        if has_bias:
            self.assertEqual(
                counters["inductor"]["decompose_addmm"],
                expected_val,
            )

        ref.sum().backward()
        res.sum().backward()

        self.compare_parameters(module, traced)
        self.compare_gradients(module, traced)

        expected_val = 0
        if HAS_CUDA:
            expected_val = 1 if has_bias else 2

        self.assertEqual(
            counters["inductor"]["decompose_mm"],
            expected_val,
        )
        counters.clear()

    def test_realize_input(self):
        m = 20480
        k = 5
        n = 2
        torch._logging.set_logs(inductor=logging.DEBUG)
        input1 = torch.randn(m, k, device=GPU_TYPE).T.contiguous()
        input2 = torch.randn(k, n, device=GPU_TYPE)

        @torch.compile()
        def foo(x, y):
            return x.T.contiguous() @ y

        out, code = run_and_get_code(foo, input1, input2)

        if GPU_TYPE == "xpu":
            # only 1 kernel generated on the XPU stack
            FileCheck().check_count(".run(", 1, exactly=True).run(code[0])
        else:
            # two kernels generated
            FileCheck().check_count(".run(", 2, exactly=True).run(code[0])


if __name__ == "__main__":
    run_tests()
