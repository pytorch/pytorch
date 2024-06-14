# Owner(s): ["module: inductor"]
import copy
import importlib
import itertools
import os
import sys
import unittest

import torch
from torch import nn

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

from torch._dynamo.utils import counters
from torch._inductor import config as inductor_config
from torch._inductor.test_case import TestCase

from torch.testing._internal.common_utils import IS_CI, IS_WINDOWS, TEST_WITH_ASAN

from torch.testing._internal.inductor_utils import HAS_CPU, HAS_CUDA

if IS_WINDOWS and IS_CI:
    sys.stderr.write(
        "Windows CI does not have necessary dependencies for test_torchinductor yet\n"
    )
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires sympy/functorch/filelock")

importlib.import_module("functorch")
importlib.import_module("filelock")

from inductor.test_torchinductor import copy_tests


class ConvOp(nn.Module):
    expected_optimization_count = 1

    def __init__(
        self,
        conv_class,
        bn_class,
        use_bias,
        in_channels,
        out_channels,
        device,
        **kwargs,
    ):
        super().__init__()
        self.conv = conv_class(in_channels, out_channels, bias=use_bias, **kwargs).to(
            device
        )
        self.bn = bn_class(out_channels).to(device)

    def forward(self, x):
        x = self.conv(x)
        return self.bn(x)


class MultiUserConvOp(nn.Module):
    expected_optimization_count = 3

    def __init__(
        self,
        conv_class,
        bn_class,
        use_bias,
        in_channels,
        out_channels,
        device,
        **kwargs,
    ):
        super().__init__()
        self.conv1 = conv_class(in_channels, out_channels, bias=use_bias, **kwargs).to(
            device
        )
        self.bn1 = bn_class(out_channels).to(device)
        self.conv2 = conv_class(out_channels, out_channels, bias=use_bias, **kwargs).to(
            device
        )
        self.bn2 = bn_class(out_channels).to(device)
        self.conv3 = conv_class(out_channels, out_channels, bias=use_bias, **kwargs).to(
            device
        )
        self.bn3 = bn_class(out_channels).to(device)

    def forward(self, x):
        # this conv-bn pair can use efficient_conv_bn_eval
        x = self.bn1(self.conv1(input=x))
        # this conv-bn pair cannot use efficient_conv_bn_eval feature
        # just for the second forward of the `self.conv2`
        x = self.bn2(input=self.conv2(self.conv2(x)))
        # this conv-bn pair can use efficient_conv_bn_eval feature
        # just for the first forward of the `self.bn3`
        # test for multiple users of one computation node
        x = self.bn3(input=self.conv3(input=x))
        x = self.bn3(x) + x
        return x


class EfficientConvBNEvalTemplate(TestCase):
    @inductor_config.patch({"efficient_conv_bn_eval_fx_passes": True})
    def test_basic(self):
        def test_conv_bn_eval(
            test_class, use_bias, module, sync_bn, decompose_nn_module
        ):
            from functorch import make_fx
            from torch._dispatch.python import enable_python_dispatcher

            kwargs = {"kernel_size": 3, "stride": 2} if module[0] != nn.Linear else {}
            mod_eager = test_class(
                module[0],
                module[1],
                use_bias,
                3,
                32,
                self.device,
                **kwargs,
            ).eval()
            # Copy module to test backward
            mod_optimized = copy.deepcopy(mod_eager)
            if sync_bn:
                mod_eager = nn.SyncBatchNorm.convert_sync_batchnorm(mod_eager).eval()
                mod_optimized = nn.SyncBatchNorm.convert_sync_batchnorm(
                    mod_optimized
                ).eval()
            torch._dynamo.reset()

            inps = [4, 3]
            # Conv shape goes from big to small, and ConvTranspose shape goes from small to big
            spatial_d = (
                4 if issubclass(module[0], nn.modules.conv._ConvTransposeNd) else 96
            )
            if module[0] == nn.Conv1d or module[0] == nn.ConvTranspose1d:
                inps += [spatial_d] * 1
            if module[0] == nn.Conv2d or module[0] == nn.ConvTranspose2d:
                inps += [spatial_d] * 2
            if module[0] == nn.Conv3d or module[0] == nn.ConvTranspose3d:
                inps += [spatial_d] * 3
            inp = torch.rand(inps).to(self.device)

            if decompose_nn_module:
                with enable_python_dispatcher():
                    mod_optimized = make_fx(mod_optimized, pre_dispatch=True)(inp)
            mod_optimized = torch.compile(mod_optimized)

            original_value = counters["inductor"]["efficient_conv_bn_eval"]

            optim_eager = torch.optim.SGD(mod_eager.parameters(), lr=1e-3)
            optim_optimized = torch.optim.SGD(mod_optimized.parameters(), lr=1e-3)

            optim_eager.zero_grad()
            optim_optimized.zero_grad()

            # test forward
            out_eager = mod_eager(inp)
            out_optimized = mod_optimized(inp)

            self.assertEqual(out_optimized, out_eager, atol=3e-04, rtol=1e-5)

            out_eager.mean().backward()
            out_optimized.mean().backward()

            optim_eager.step()
            optim_optimized.step()
            # test forward (by testing forward again after one training iteration)
            inp_bw = torch.rand_like(inp)
            out_eager_bw = mod_eager(inp_bw)
            out_optimized_bw = mod_optimized(inp_bw)

            self.assertEqual(out_eager_bw, out_optimized_bw, atol=3e-04, rtol=1e-5)
            current_value = counters["inductor"]["efficient_conv_bn_eval"]
            self.assertEqual(
                current_value - original_value, test_class.expected_optimization_count
            )

        conv_bias = [True, False]
        modules = [
            (nn.Linear, nn.BatchNorm1d),
            (nn.Conv1d, nn.BatchNorm1d),
            (nn.Conv2d, nn.BatchNorm2d),
            (nn.Conv3d, nn.BatchNorm3d),
            (nn.ConvTranspose1d, nn.BatchNorm1d),
            (nn.ConvTranspose2d, nn.BatchNorm2d),
            (nn.ConvTranspose3d, nn.BatchNorm3d),
        ]
        test_classes = [ConvOp, MultiUserConvOp]
        sync_bns = [False, True]
        decompose_nn_modules = [False, True]
        for (
            test_class,
            use_bias,
            module,
            sync_bn,
            decompose_nn_module,
        ) in itertools.product(
            test_classes,
            conv_bias,
            modules,
            sync_bns,
            decompose_nn_modules,
        ):
            test_conv_bn_eval(
                test_class, use_bias, module, sync_bn, decompose_nn_module
            )


if HAS_CPU and not torch.backends.mps.is_available():

    class EfficientConvBNEvalCpuTests(TestCase):
        device = "cpu"

    copy_tests(EfficientConvBNEvalTemplate, EfficientConvBNEvalCpuTests, "cpu")

if HAS_CUDA and not TEST_WITH_ASAN:

    class EfficientConvBNEvalCudaTests(TestCase):
        device = "cuda"

    copy_tests(EfficientConvBNEvalTemplate, EfficientConvBNEvalCudaTests, "cuda")

del EfficientConvBNEvalTemplate

if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_CPU or HAS_CUDA:
        run_tests(needs="filelock")
