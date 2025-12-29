# Owner(s): ["module: inductor"]

import logging

import numpy as np

import torch
import torch._inductor
import torch._inductor.fx_passes.group_batch_fusion
from torch._dynamo.utils import counters
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import IS_LINUX
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU, requires_gpu


log = logging.getLogger(__name__)


class TargetCPModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        relued = torch.relu(x1)
        tanhed = torch.tanh(relued)
        tensor = torch.matmul(
            tanhed,
            x2,
        )
        return tensor


class FeedforwardNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(1, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        self.fc4 = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        tanh_x = torch.tanh(x)
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(tanh_x))
        x = self.fc4(x)
        return x


class LayernormNN(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, normalized_shape, weight, bias):
        x = torch.nn.functional.layer_norm(
            input=input,
            normalized_shape=normalized_shape,
            weight=weight,
            bias=bias,
            eps=1e-5,
        )
        return x


class TestQuantization(TestCase):
    def compare_dict_tensors(self, ref_dict, res_dict, rtol=1e-3, atol=1e-3):
        if len(set(ref_dict.keys())) != len(set(res_dict.keys())):
            return False
        for key1 in ref_dict:
            key2 = "_orig_mod." + key1
            assert key2 in res_dict, f"{key1} does not exist in traced module"
            # if both of them are None, continue
            if (
                not isinstance(ref_dict[key1], torch.Tensor)
                and not isinstance(res_dict[key2], torch.Tensor)
                and ref_dict[key1] is None
                and res_dict[key2] is None
            ):
                log.info(
                    "None found with key1 and value 1: %s, %s, key2 and value2 %s, %s",
                    key1,
                    ref_dict[key1],
                    key2,
                    res_dict[key2],
                )
                continue
            elif not torch.allclose(
                ref_dict[key1], res_dict[key2], rtol=rtol, atol=atol, equal_nan=True
            ):
                log.info(
                    "gradient mismatch for eager and compiled modules, with eager: %s and compiled: %s",
                    ref_dict[key1],
                    res_dict[key2],
                )
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
        pre_grad_fusion_options={},
        post_grad_fusion_options={
            "activation_quantization_aten_pass": {
                "quant_type": "torch.float8_e5m2",
                "use_scaling": True,
                "size_in_mb": 0.0,
                "exclude_primals": True,
                "allowed_dtypes": "torch.bfloat16;torch.float32",
            },
        },
    )
    def test_activation_quantization_aten_with_scaling(self):
        counters.clear()
        module = TargetCPModule().to(GPU_TYPE)
        input = [
            torch.rand(
                (16, 10), requires_grad=True, device=GPU_TYPE, dtype=torch.bfloat16
            ),
            torch.rand(
                (10, 16), requires_grad=True, device=GPU_TYPE, dtype=torch.bfloat16
            ),
        ]
        traced = torch.compile(module)
        ref = module(*input)
        res = traced(*input)
        self.compare_pred(module, traced, input)
        ref.sum().backward()
        res.sum().backward()
        self.compare_parameters(module, traced)
        self.compare_gradients(module, traced)
        self.assertEqual(
            counters["inductor"]["activation_quantization_fwd_aten_pass"], 1
        )
        self.assertEqual(
            counters["inductor"]["activation_quantization_bwd_aten_pass"], 1
        )
        self.assertTrue(torch.allclose(ref, res))
        counters.clear()

        module = FeedforwardNN().to(GPU_TYPE)
        X = np.linspace(-10, 10, 100).reshape(-1, 1).astype(np.float32)
        input = [
            torch.from_numpy(X).to(GPU_TYPE),
        ]
        traced = torch.compile(module)
        ref = module(*input)
        res = traced(*input)
        self.compare_pred(module, traced, input)
        ref.sum().backward()
        res.sum().backward()
        self.compare_parameters(module, traced)
        self.compare_gradients(module, traced)
        self.assertEqual(
            counters["inductor"]["activation_quantization_fwd_aten_pass"], 1
        )
        self.assertEqual(
            counters["inductor"]["activation_quantization_bwd_aten_pass"], 1
        )
        self.assertTrue(torch.allclose(ref, res))
        counters.clear()

    @requires_gpu()
    @torch._inductor.config.patch(
        pre_grad_fusion_options={},
        post_grad_fusion_options={
            "activation_quantization_aten_pass": {
                "quant_type": "torch.float8_e5m2",
                "use_scaling": False,
                "size_in_mb": 0.0,
                "exclude_primals": True,
                "allowed_dtypes": "torch.bfloat16;torch.float32",
            },
        },
    )
    def test_activation_quantization_aten_without_scaling(self):
        counters.clear()

        module = LayernormNN().to(GPU_TYPE)
        normalized_shape = [256]
        input = [
            torch.randn(
                (1, 3, 256), requires_grad=True, device=GPU_TYPE, dtype=torch.bfloat16
            ),
            normalized_shape,
            torch.randn(
                *normalized_shape,
                requires_grad=True,
                device=GPU_TYPE,
                dtype=torch.bfloat16,
            ),
            torch.randn(
                *normalized_shape,
                requires_grad=True,
                device=GPU_TYPE,
                dtype=torch.bfloat16,
            ),
        ]
        traced = torch.compile(module)
        ref = module(*input)
        res = traced(*input)
        self.compare_pred(module, traced, input)
        ref.sum().backward()
        res.sum().backward()
        self.compare_parameters(module, traced)
        self.compare_gradients(module, traced)
        self.assertEqual(
            counters["inductor"]["activation_quantization_fwd_aten_pass"], 1
        )
        self.assertEqual(
            counters["inductor"]["activation_quantization_bwd_aten_pass"], 1
        )
        self.assertTrue(torch.allclose(ref, res))
        counters.clear()


if __name__ == "__main__":
    if IS_LINUX and HAS_GPU:
        run_tests()
