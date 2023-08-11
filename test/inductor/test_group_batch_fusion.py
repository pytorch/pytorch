# Owner(s): ["module: inductor"]

import functools
import unittest

import torch
import torch._inductor
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.utils import counters
from torch.testing._internal.inductor_utils import HAS_CUDA

try:
    # importing this will register fbgemm lowerings for inductor
    import deeplearning.fbgemm.fbgemm_gpu.fb.inductor_lowerings  # noqa: F401

    has_fbgemm = True
except Exception:
    has_fbgemm = False
    pass

requires_cuda = functools.partial(unittest.skipIf, not HAS_CUDA, "requires cuda")


class MyModule(torch.nn.Module):
    def __init__(self, z: int, has_bias: bool, device="cuda") -> None:
        super().__init__()
        self.z = z
        self.device = device
        self.seq_len = 10
        self.seq1 = [
            torch.nn.Linear(z, z, has_bias).to(self.device) for _ in range(self.seq_len)
        ]
        self.seq2 = [
            torch.nn.Linear(z, z, has_bias).to(self.device) for _ in range(self.seq_len)
        ]
        self.seq3 = [
            torch.nn.Linear(z, z, has_bias).to(self.device) for _ in range(self.seq_len)
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = [x + 0.1 * i for i in range(self.seq_len)]
        x2 = [self.seq1[i](x1[i]) for i in range(self.seq_len)]
        x3 = [x2[i] - 0.1 * i for i in range(self.seq_len)]
        x4 = [x1[i] for i in range(3)] + [x3[i] for i in range(3, self.seq_len)]
        x5 = [self.seq2[i](x4[i]) for i in range(self.seq_len)]
        x6 = [x5[i] + 0.1 * (self.seq_len - i) for i in range(self.seq_len)]
        x7 = (
            [x1[i] for i in range(4)]
            + [x3[i] for i in range(6, 8)]
            + [x6[i] for i in range(4)]
        )
        x8 = [self.seq3[i](x7[i]) for i in range(self.seq_len)]
        x9 = torch.cat(x8, dim=1)
        return x9


class MyModule2(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear0 = torch.nn.Linear(6, 8)
        self.linear1 = torch.nn.Linear(8, 8)
        self.linear2 = torch.nn.Linear(10, 8)
        self.linear3 = torch.nn.Linear(6, 8)
        self.linear4 = torch.nn.Linear(8, 8)
        self.linear5 = torch.nn.Linear(10, 8)
        self.bn0 = torch.nn.BatchNorm1d(8)
        self.bn1 = torch.nn.BatchNorm1d(8)
        self.bn2 = torch.nn.BatchNorm1d(8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = torch.split(x, [6, 8, 10], dim=1)
        a0 = self.bn0(self.linear0(t[0] + 0.1))
        a1 = self.bn1(self.linear1(t[1] + 0.2))
        a2 = self.bn2(self.linear2(t[2] + 0.3))
        a3 = self.linear3(torch.sin(t[0]))
        a4 = self.linear4(torch.cos(t[1]))
        a5 = self.linear5(torch.sin(t[2] * 0.5))

        b = torch.cat([a0, a1, a2, a3, a4, a5])
        return torch.sigmoid(b)


class MyModule3(torch.nn.Module):
    def __init__(self, device, has_weight=True, has_bias=True):
        super().__init__()
        self.device = device
        self.scale0 = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.randn(10)) for _ in range(5)]
        ).to(self.device)
        self.bias0 = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.randn(10)) for _ in range(5)]
        ).to(self.device)
        self.scale1 = (
            torch.nn.ParameterList(
                [torch.nn.Parameter(torch.randn(5, 10)) for _ in range(5)]
            ).to(self.device)
            if has_weight
            else [None for _ in range(5)]
        )
        self.bias1 = (
            torch.nn.ParameterList(
                [torch.nn.Parameter(torch.randn(5, 10)) for _ in range(5)]
            ).to(self.device)
            if has_bias
            else [None for _ in range(5)]
        )

    def forward(self, x):
        l1_out = torch.split(x.to(self.device), 10, dim=2)
        post_l1 = [
            torch.nn.functional.layer_norm(
                l1_out[i], (10,), weight=self.scale0[i], bias=self.bias0[i]
            )
            for i in range(len(l1_out))
        ]
        l1_out = torch.cat(post_l1, dim=2)

        l2_out = torch.split(l1_out, 10, dim=2)
        post_l2 = [
            torch.nn.functional.layer_norm(
                l2_out[i], (5, 10), weight=self.scale1[i], bias=self.bias1[i]
            )
            for i in range(len(l2_out))
        ]

        return torch.cat(post_l2, dim=2)


class MyModule4(torch.nn.Module):
    def __init__(self, z, device, has_bias):
        super().__init__()
        self.z = z
        self.device = device
        self.has_bias = has_bias
        self.seq_len = 10
        self.weights1 = [
            torch.nn.Parameter(torch.randn(z - i % 5, z)).to(self.device)
            for i in range(self.seq_len)
        ]
        self.weights2 = [
            torch.nn.Parameter(torch.randn(z - i % 5, z)).to(self.device)
            for i in range(self.seq_len)
        ]

        if has_bias:
            self.biases1 = [
                torch.nn.Parameter(torch.randn(z - i % 5)).to(self.device)
                for i in range(self.seq_len)
            ]
            self.biases2 = [
                torch.nn.Parameter(torch.randn(z - i % 5)).to(self.device)
                for i in range(self.seq_len)
            ]

    def forward(self, x):
        x = x + 1.2
        x1 = [
            torch.nn.functional.linear(
                x, self.weights1[i], self.biases1[i] if self.has_bias else None
            )
            for i in range(self.seq_len)
        ]
        x2 = torch.cat(x1, dim=1)
        x3 = torch.split(x2, 10, dim=1)
        x4 = torch.cat(x3)
        x5 = [
            torch.nn.functional.linear(
                x4, self.weights2[i], self.biases2[i] if self.has_bias else None
            )
            for i in range(self.seq_len)
        ]
        x6 = torch.cat(x5, dim=1)
        return torch.sigmoid(x6)


@requires_cuda()
@torch._inductor.config.patch(group_fusion=True, batch_fusion=True)
class TestGroupBatchFusion(TestCase):
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

    @unittest.skipIf(not has_fbgemm, "requires fbgemm")
    def test_group_linear_fusion(self):
        z = 10
        for has_bias in [True, False]:
            counters.clear()
            module = MyModule(z, has_bias).to("cuda")
            input = [torch.randn(z, z, device="cuda")]
            traced = torch.compile(module)
            ref = module(*input)
            res = traced(*input)
            self.compare_pred(module, traced, input)
            self.assertEqual(
                counters["inductor"]["group_fusion"],
                2,
            )
            self.assertEqual(
                counters["inductor"]["batch_fusion"],
                0,
            )
            ref.sum().backward()
            res.sum().backward()
            self.compare_parameters(module, traced)
            self.compare_gradients(module, traced)
            self.assertEqual(
                counters["inductor"]["group_fusion"],
                4,
            )
            self.assertEqual(
                counters["inductor"]["batch_fusion"],
                0,
            )
            counters.clear()

    @unittest.skipIf(not has_fbgemm, "requires fbgemm")
    def test_group_linear_fusion_different_shapes(self):
        counters.clear()
        module = MyModule2().eval().to("cuda")
        input = [torch.rand(4, 24, device="cuda")]
        traced = torch.compile(module)
        ref = module(*input)
        res = traced(*input)
        self.compare_pred(module, traced, input)
        self.assertEqual(
            counters["inductor"]["group_fusion"],
            1,
        )
        self.assertEqual(
            counters["inductor"]["batch_fusion"],
            0,
        )
        ref.sum().backward()
        res.sum().backward()
        self.compare_parameters(module, traced)
        self.compare_gradients(module, traced)
        self.assertEqual(
            counters["inductor"]["group_fusion"],
            2,
        )
        self.assertEqual(
            counters["inductor"]["batch_fusion"],
            0,
        )
        counters.clear()

    def test_batch_layer_norm_fusion(self):
        for has_weight in [True, False]:
            for has_bias in [True, False]:
                counters.clear()
                module = MyModule3("cuda", has_weight, has_bias).to("cuda")
                input = [torch.randn(2, 5, 50, device="cuda")]
                traced = torch.compile(module)
                ref = module(*input)
                res = traced(*input)
                self.compare_pred(module, traced, input)
                self.assertEqual(
                    counters["inductor"]["group_fusion"],
                    0,
                )
                self.assertEqual(counters["inductor"]["batch_fusion"], 2)
                self.assertEqual(
                    counters["inductor"]["scmerge_split_removed"],
                    3,
                )
                self.assertEqual(
                    counters["inductor"]["scmerge_cat_removed"],
                    3,
                )
                ref.sum().backward()
                res.sum().backward()
                self.compare_parameters(module, traced, rtol=1e-8, atol=1e-8)
                self.compare_gradients(module, traced, rtol=1e-8, atol=1e-8)
                counters.clear()

    def test_batch_linear_lhs_fusion(self):
        z = 10
        for has_bias in [True, False]:
            counters.clear()
            module = MyModule4(z, "cuda", has_bias)
            input = [torch.randn(20, z, device="cuda")]
            traced = torch.compile(module)
            ref = module(*input)
            res = traced(*input)
            self.compare_pred(module, traced, input)
            self.assertEqual(counters["inductor"]["batch_fusion"], 2)
            self.assertEqual(
                counters["inductor"]["scmerge_split_removed"],
                1,
            )
            self.assertEqual(
                counters["inductor"]["scmerge_cat_removed"],
                1,
            )
            ref.sum().backward()
            res.sum().backward()
            self.compare_parameters(module, traced, rtol=1e-8, atol=1e-8)
            self.compare_gradients(module, traced, rtol=1e-8, atol=1e-8)
            counters.clear()


if __name__ == "__main__":
    run_tests()
