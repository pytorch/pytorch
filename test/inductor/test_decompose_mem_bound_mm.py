# Owner(s): ["module: inductor"]

import logging
import unittest

import torch
import torch._inductor
from torch._dynamo.utils import counters
from torch._inductor.fx_passes.decompose_mem_bound_mm import check_device
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing import FileCheck
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    is_navi3_arch,
    parametrize,
    patch_test_members,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU_AND_TRITON
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


class TestDecomposeAddMM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, z: torch.Tensor, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        return torch.ops.aten.addmm.default(z, x, y)


@requires_gpu
@torch._inductor.config.patch(
    post_grad_fusion_options={
        "decompose_mm_pass": {},
    }
)
@instantiate_parametrized_tests
class TestDecomposeMemMM(TestCase):
    def __init__(self, method_name="runTest", methodName="runTest"):
        super().__init__(method_name, methodName)
        self.atol = 1e-3
        self.rtol = 1e-3

    def setup_tolerance(self, rtol=None, atol=None):
        if rtol is None:
            rtol = self.rtol
        if atol is None:
            atol = self.atol
        self.rtol = rtol
        self.atol = atol

    def compare_dict_tensors(self, ref_dict, res_dict, rtol=None, atol=None):
        self.setup_tolerance(rtol, atol)
        if len(set(ref_dict.keys())) != len(set(res_dict.keys())):
            return False
        for key1 in ref_dict.keys():
            key2 = "_orig_mod." + key1
            assert key2 in res_dict, f"{key1} does not exist in traced module"
            if not torch.allclose(
                ref_dict[key1], res_dict[key2], rtol=self.rtol, atol=self.atol
            ):
                return False
        return True

    def compare_pred(self, module, traced, input, rtol=None, atol=None):
        self.setup_tolerance(rtol, atol)
        ref = module(*input)
        res = traced(*input)
        self.assertEqual(ref, res, rtol=self.rtol, atol=self.atol)

    def compare_parameters(self, module, traced, rtol=None, atol=None):
        self.setup_tolerance(rtol, atol)
        ref_params = dict(module.named_parameters())
        res_params = dict(traced.named_parameters())
        self.assertTrue(
            self.compare_dict_tensors(
                ref_params, res_params, rtol=self.rtol, atol=self.atol
            )
        )

    def compare_gradients(self, module, traced, rtol=None, atol=None):
        self.setup_tolerance(rtol, atol)
        ref_grad = {key: param.grad for key, param in module.named_parameters()}
        res_grad = {key: param.grad for key, param in traced.named_parameters()}
        self.assertTrue(
            self.compare_dict_tensors(
                ref_grad, res_grad, rtol=self.rtol, atol=self.atol
            )
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

        expected_val = 1 if should_decompose and HAS_GPU_AND_TRITON else 0
        self.assertEqual(
            counters["inductor"]["decompose_bmm"],
            expected_val,
        )

        ref.sum().backward()
        res.sum().backward()
        self.compare_parameters(module, traced)
        self.compare_gradients(module, traced)

        expected_val = 3 if should_decompose and HAS_GPU_AND_TRITON else 0
        self.assertEqual(
            counters["inductor"]["decompose_bmm"],
            expected_val,
        )
        counters.clear()

    @parametrize(
        "b,m,k,n,should_decompose",
        [(1, 2, 2, 2, True), (2, 2, 2, 2, False)],
    )
    def test_decompose_bmm_cpu(self, b, m, n, k, should_decompose):
        torch._logging.set_logs(inductor=logging.DEBUG)
        mat1 = torch.randn(b, m, k)
        mat2 = torch.randn(b, k, n)

        counters.clear()

        module = MyModule2()
        traced = torch.compile(module)
        input = [mat1, mat2]
        self.compare_pred(module, traced, input)

        expected_val = 1 if should_decompose else 0
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

        expected_val = 1 if should_decompose and HAS_GPU_AND_TRITON else 0
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

    # We have to increase tolerance for navi3 because all fp16, bf16
    # GEMMs operations have an accuracy issue caused by hardware limitation
    @patch_test_members(
        {
            "atol": 2e-3 if is_navi3_arch() else 1e-3,
            "rtol": 2e-3 if is_navi3_arch() else 1e-3,
        }
    )
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

            expected_val = 1 if should_decompose and HAS_GPU_AND_TRITON else 0
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

        expected_val = 1 if should_decompose and HAS_GPU_AND_TRITON else 0
        self.assertEqual(
            counters["inductor"]["decompose_mm"],
            expected_val,
        )
        decompose_mm_fwd = counters["inductor"]["decompose_mm"]

        ref.sum().backward()
        res.sum().backward()
        self.compare_parameters(module, traced)
        self.compare_gradients(module, traced)

        expected_val = 1 if should_decompose and HAS_GPU_AND_TRITON else 0
        self.assertEqual(
            counters["inductor"]["decompose_mm"] - decompose_mm_fwd,
            expected_val,
        )
        counters.clear()

    @parametrize(
        "m,k,n, should_decompose",
        [(1, 64, 16, True), (2, 64, 16, False), (1, 64, 32, True)],
    )
    def test_decompose_mm_cpu(self, m, n, k, should_decompose):
        torch._logging.set_logs(inductor=logging.DEBUG)
        mat1 = torch.randn(m, k)
        mat2 = torch.randn(k, n)
        counters.clear()

        module = MyModule3()
        traced = torch.compile(module)
        input = [mat1, mat2]
        self.compare_pred(module, traced, input)

        expected_val = 1 if should_decompose else 0
        self.assertEqual(
            counters["inductor"]["decompose_mm"],
            expected_val,
        )
        counters.clear()

    # We have to increase tolerance for navi3 because all fp16, bf16
    # GEMMs operations have an accuracy issue caused by hardware limitation
    @patch_test_members(
        {
            "atol": 3e-3 if is_navi3_arch() else 1e-3,
            "rtol": 4e-3 if is_navi3_arch() else 1e-3,
        }
    )
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

            expected_val = 1 if should_decompose and HAS_GPU_AND_TRITON else 0
            self.assertEqual(
                counters["inductor"]["decompose_mm"],
                expected_val,
            )
            decompose_mm_fwd = counters["inductor"]["decompose_mm"]

            ref.sum().backward()
            res.sum().backward()
            self.compare_parameters(module, traced)
            self.compare_gradients(module, traced)

            expected_val = 1 if should_decompose and HAS_GPU_AND_TRITON else 0
            self.assertEqual(
                counters["inductor"]["decompose_mm"] - decompose_mm_fwd,
                expected_val,
            )
            counters.clear()

    @unittest.skip
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

        expected_val = 1 if should_decompose and HAS_GPU_AND_TRITON else 0
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
        if HAS_GPU_AND_TRITON:
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

        _, code = run_and_get_code(foo, input1, input2)

        # two kernels generated
        FileCheck().check_count(".run(", 2, exactly=True).run(code[0])

    def test_check_device(self):
        m = 5
        k = 5
        n = 2
        torch._logging.set_logs(inductor=logging.DEBUG)

        input1 = torch.randn(m, k, device=GPU_TYPE)
        input2 = torch.randn(k, n, device=GPU_TYPE)
        self.assertTrue(check_device(input1, input2, device=GPU_TYPE))
        self.assertFalse(check_device(input1, input2, device="cpu"))

        input1 = torch.randn(m, k)
        input2 = torch.randn(k, n)
        self.assertTrue(check_device(input1, input2, device="cpu"))
        self.assertFalse(check_device(input1, input2))

        input1 = torch.randn(m, k, device=GPU_TYPE)
        input2 = torch.randn(k, n)
        self.assertFalse(check_device(input1, input2, device="gpu"))
        self.assertFalse(check_device(input1, input2, device="cpu"))

        self.assertFalse(check_device(input1, input2, device="mtia"))

    @torch._inductor.config.patch(
        post_grad_fusion_options={
            "decompose_mm_pass": {"skip_dynamic_shape_dim_check": True},
        }
    )
    def test_dynamic_shape_decompose_addmm(self):
        m, k, n = 19494144, 8, 8
        input = torch.randn(m, k, device=GPU_TYPE).requires_grad_(False)
        weight = torch.randn(k, n, device=GPU_TYPE).requires_grad_(False)
        bias = torch.randn(n, device=GPU_TYPE).requires_grad_(False)

        counters.clear()

        module = TestDecomposeAddMM().to(GPU_TYPE)
        traced = torch.compile(module, dynamic=True)
        input = [bias, input, weight]

        self.compare_pred(module, traced, input)

        self.assertEqual(
            counters["inductor"]["decompose_addmm"],
            1,
        )
        counters.clear()


if __name__ == "__main__":
    run_tests()
