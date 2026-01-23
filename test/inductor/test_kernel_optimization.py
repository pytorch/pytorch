# Owner(s): ["module: inductor"]

import torch
import torch._inductor
from torch._dynamo.utils import counters
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import serialTest
from torch.testing._internal.inductor_utils import GPU_TYPE, requires_gpu


class TestEinsumtoPointwise(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        input: torch.Tensor,
        weights: torch.Tensor,
        bias: torch.Tensor,
        input2: torch.Tensor,
        weights2: torch.Tensor,
        bias2: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.functional.einsum("bni, nio -> bno", input, weights)
        add1 = output.add(bias)
        output2 = torch.functional.einsum("bni, bnio -> bno", input2, weights2)
        add2 = output2 + bias2
        return add1 + add2


class TestKernelOptimization(TestCase):
    def compare_dict_tensors(self, ref_dict, res_dict, rtol=1e-3, atol=1e-3):
        if len(set(ref_dict.keys())) != len(set(res_dict.keys())):
            return False
        for key1 in ref_dict:
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

    @requires_gpu()
    @torch._inductor.config.patch(
        pre_grad_fusion_options={
            "einsum_to_pointwise_pass": {},
        },
        post_grad_fusion_options={},
    )
    @serialTest()  # Needs slightly more memory on GPUs
    def test_einsum_to_pointwise(self):
        counters.clear()
        module = TestEinsumtoPointwise().to(GPU_TYPE)
        input = [
            torch.randn(4096, 9, 512, device=GPU_TYPE, requires_grad=True),
            torch.randn(9, 512, 96, device=GPU_TYPE, requires_grad=True),
            torch.randn(9, 96, device=GPU_TYPE, requires_grad=True),
            torch.randn(4096, 9, 160, device=GPU_TYPE, requires_grad=True),
            torch.randn(4096, 9, 160, 96, device=GPU_TYPE, requires_grad=True),
            torch.randn(4096, 9, 96, device=GPU_TYPE, requires_grad=True),
        ]
        traced = torch.compile(module)
        ref = module(*input)
        res = traced(*input)
        ref.sum().backward()
        res.sum().backward()
        self.compare_pred(module, traced, input)
        self.compare_parameters(module, traced)
        self.compare_gradients(module, traced)
        self.assertEqual(
            counters["inductor"]["einsum_to_pointwise_pass"],
            1,
        )
        counters.clear()


if __name__ == "__main__":
    run_tests()
