# Owner(s): ["module: inductor"]

import unittest

import torch
import torch._inductor
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.utils import counters

try:
    # importing this will register fbgemm lowerings for inductor
    import deeplearning.fbgemm.fbgemm_gpu.fb.inductor_lowerings  # noqa: F401

    has_fbgemm = True
except Exception:
    has_fbgemm = False
    pass


class MyModule(torch.nn.Module):
    def __init__(self, z: int, has_bias: bool) -> None:
        super().__init__()
        self.linear0 = torch.nn.Linear(z, z, has_bias)
        self.linear1 = torch.nn.Linear(z, z, has_bias)
        self.linear2 = torch.nn.Linear(z, z, has_bias)
        self.linear3 = torch.nn.Linear(z, z, has_bias)
        self.linear4 = torch.nn.Linear(z, z, has_bias)
        self.linear5 = torch.nn.Linear(z, z, has_bias)
        self.bn0 = torch.nn.BatchNorm1d(z)
        self.bn1 = torch.nn.BatchNorm1d(z)
        self.bn2 = torch.nn.BatchNorm1d(z)
        self.bn3 = torch.nn.BatchNorm1d(z)
        self.bn4 = torch.nn.BatchNorm1d(z)

    def forward(
        self,
        t0: torch.Tensor,
        t1: torch.Tensor,
        t2: torch.Tensor,
    ) -> torch.Tensor:
        a0 = self.bn0(self.linear0(t0))
        a1 = self.bn1(self.linear1(t1))
        a2 = self.bn2(self.linear2(t2))

        b0 = torch.sigmoid(a0)
        b1 = torch.tanh(a1)
        b2 = self.linear3(a2)

        c0 = b0 + b1 + b2
        c1 = torch.relu(b2)

        d0 = self.bn3(self.linear4(c0))
        d1 = self.bn4(self.linear5(c1))
        return d0 + d1


class MyModule2(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear0 = torch.nn.Linear(6, 8)
        self.linear1 = torch.nn.Linear(8, 8, False)
        self.linear2 = torch.nn.Linear(10, 8)
        self.linear3 = torch.nn.Linear(8, 8)
        self.linear4 = torch.nn.Linear(8, 6, False)
        self.linear5 = torch.nn.Linear(8, 6)
        self.bn0 = torch.nn.BatchNorm1d(8)
        self.bn1 = torch.nn.BatchNorm1d(8)
        self.bn2 = torch.nn.BatchNorm1d(8)
        self.bn3 = torch.nn.BatchNorm1d(6)
        self.bn4 = torch.nn.BatchNorm1d(6)

    def forward(
        self,
        t0: torch.Tensor,
        t1: torch.Tensor,
        t2: torch.Tensor,
    ) -> torch.Tensor:
        a0 = self.bn0(self.linear0(t0))
        a1 = self.bn1(self.linear1(t1))
        a2 = self.bn2(self.linear2(t2))

        b0 = torch.sigmoid(a0)
        b1 = torch.tanh(a1)
        b2 = self.linear3(a2)

        c0 = b0 + b1 + b2
        c1 = torch.relu(b2)

        d0 = self.bn3(self.linear4(c0))
        d1 = self.bn4(self.linear5(c1))
        return d0 + d1


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


@unittest.skipIf(not has_fbgemm, "requires fbgemm")
@torch._inductor.config.patch(group_fusion=True, batch_fusion=True)
class TestGroupFusion(TestCase):
    def compare_dict_tensors(self, ref_dict, res_dict):
        if len(set(ref_dict.keys())) != len(set(res_dict.keys())):
            return False
        for key1 in ref_dict.keys():
            key2 = "_orig_mod." + key1
            assert key2 in res_dict, f"{key1} does not exist in traced module"
            if not torch.allclose(ref_dict[key1], res_dict[key2], rtol=1e-3, atol=1e-3):
                return False
        return True

    def compare_pred(self, module, traced, input):
        ref = module(*input)
        res = traced(*input)
        self.assertEqual(ref, res, rtol=1e-3, atol=1e-3)

    def compare_parameters(self, module, traced):
        ref_params = dict(module.named_parameters())
        res_params = dict(traced.named_parameters())
        self.assertTrue(self.compare_dict_tensors(ref_params, res_params))

    def compare_gradients(self, module, traced):
        ref_grad = {key: param.grad for key, param in module.named_parameters()}
        res_grad = {key: param.grad for key, param in traced.named_parameters()}
        self.assertTrue(self.compare_dict_tensors(ref_grad, res_grad))

    def test_group_linear_fusion(self):
        z = 16
        for has_bias in [True, False]:
            counters.clear()
            module = MyModule(z, has_bias).eval().to("cuda")
            input = [
                torch.randn(4, z, device="cuda"),
                torch.randn(4, z, device="cuda"),
                torch.randn(4, z, device="cuda"),
            ]
            traced = torch.compile(module)
            ref = module(*input)
            res = traced(*input)
            self.compare_pred(module, traced, input)
            self.assertEqual(
                counters["inductor"]["group_fusion"],
                2 if has_bias else 0,
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
                2 if has_bias else 0,
            )
            self.assertEqual(
                counters["inductor"]["batch_fusion"],
                0,
            )
            counters.clear()

    def test_group_linear_fusion_different_shapes(self):
        counters.clear()
        module = MyModule2().eval().to("cuda")
        input = [
            torch.randn(4, 6, device="cuda"),
            torch.randn(4, 8, device="cuda"),
            torch.randn(4, 10, device="cuda"),
        ]
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
            1,
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
                module = MyModule3("cuda", has_weight, has_bias).eval().to("cuda")
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
                self.compare_parameters(module, traced)
                self.compare_gradients(module, traced)
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
                counters.clear()


if __name__ == "__main__":
    run_tests()
