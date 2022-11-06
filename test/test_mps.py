# -*- coding: utf-8 -*-
# Owner(s): ["module: mps"]

import sys
import math
import random
import unittest
import warnings
import subprocess
import tempfile
import os
import pprint
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from collections import defaultdict
from torch._six import inf
from torch.nn import Parameter
from torch.testing._internal import opinfo
from torch.testing._internal.common_utils import \
    (gradcheck, gradgradcheck, run_tests, TestCase, download_file, IS_CI,
     TEST_WITH_UBSAN, dtype_abbrs, skipIfSlowGradcheckEnv, TEST_WITH_ASAN, suppress_warnings)
from torch.testing import make_tensor
from torch.testing._comparison import TensorLikePair
from torch.testing._internal.common_dtype import get_all_dtypes, integral_types
import torch.backends.mps
from torch.distributions import Uniform, Exponential
from functools import partial

from torch.testing._internal.common_methods_invocations import (
    op_db,
    UnaryUfuncInfo,
    ReductionOpInfo,
    SpectralFuncInfo,
    BinaryUfuncInfo,
)
from torch.testing._internal.common_device_type import ops, instantiate_device_type_tests, onlyMPS
from torch.testing._internal.common_nn import NNTestCase
import numpy as np
import torch
import torch.utils._pytree as pytree


# Copied from `test_ops.py` for the purposes of duplicating `test_numpy_ref`
_ref_test_ops = tuple(
    filter(
        lambda op: not isinstance(
            op, (UnaryUfuncInfo, ReductionOpInfo, SpectralFuncInfo, BinaryUfuncInfo)
        )
        and op.ref is not None,
        op_db,
    )
)

# Same logic as test_cuda.py
if not torch.backends.mps.is_available():
    print('MPS not available, skipping tests', file=sys.stderr)
    TestCase = object  # noqa: F811
    NNTestCase = object  # noqa: F811

class MPSReluTest(TestCase):
    def _npRelu(self, np_features):
        return np.maximum(np_features, np.zeros(np_features.shape)).astype(np_features.dtype)

    def testNpRelu(self):
        torch.testing.assert_close(
            np.array([[0., 0.7, 0.0, 0.3, 0.0], [0.1, 0.0, 0.5, 0.0, 0.9]]),
            self._npRelu(
                np.array([[-0.9, 0.7, -0.5, 0.3, -0.1], [0.1, -0.3, 0.5, -0.7,
                                                         0.9]])))

    def _testRelu(self, np_features, device):
        np_relu = self._npRelu(np_features)
        # Convert the numpy array to a PyTorch Tensor,
        # and move the Tensor to the CPU/GPU based on the "device" parameter
        py_tensor = torch.from_numpy(np_features).to(device)
        py_relu = torch.nn.ReLU(inplace=False)(py_tensor)
        py_relu_cpu = py_relu.to("cpu")

        self.assertEqual(np_relu, py_relu_cpu)

    def _testReluInPlace(self, np_features, device):
        np_relu = self._npRelu(np_features)
        # Convert the numpy array to a PyTorch Tensor,
        # and move the Tensor to the CPU/GPU based on the "device" parameter
        py_tensor = torch.from_numpy(np_features).to(device)
        py_relu = torch.nn.ReLU(inplace=True)(py_tensor)
        py_relu_cpu = py_relu.to("cpu")

        self.assertEqual(np_relu, py_relu_cpu)
        # Inplace Relu modifies the initial input and it should match the output of Relu
        self.assertEqual(np_relu, py_tensor.to("cpu"))

    def testNumbersCPU(self):
        for t in [np.int32]:
            # Force execution on CPU even if a GPU kernel is available for the type.
            self._testRelu(
                np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
                device="cpu")
            self._testReluInPlace(
                np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
                device="cpu")

    def testNumbersGPU(self):
        for t in [np.float16, np.float32]:
            self._testRelu(
                np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
                device="mps")
            self._testReluInPlace(
                np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
                device="mps")

class MatmulTest(TestCase):
    def _helper(self, shape_tensor_1, shape_tensor_2, expand_tensor_1_shape=None, expand_tensor_2_shape=None):
        if expand_tensor_1_shape:
            tensor1_mps = torch.randn(shape_tensor_1, device="mps").expand(expand_tensor_1_shape)
        else:
            tensor1_mps = torch.randn(shape_tensor_1, device="mps")

        if expand_tensor_2_shape:
            tensor2_mps = torch.randn(shape_tensor_2, device="mps").expand(expand_tensor_2_shape)
        else:
            tensor2_mps = torch.randn(shape_tensor_2, device="mps")

        tensor1_cpu = tensor1_mps.to("cpu")
        tensor2_cpu = tensor2_mps.to("cpu")

        matmul_cpu = torch.matmul(tensor1_cpu, tensor2_cpu)
        matmul_mps = torch.matmul(tensor1_mps, tensor2_mps)

        self.assertEqual(matmul_cpu, matmul_mps.to("cpu"))

    def test_vector_x_vector(self):
        # uses `dot`
        self._helper(3, 3)

    def test_matrix_x_vector(self):
        # uses `addmv`
        self._helper((3, 4), 4)

    def test_batched_matrix_x_broadcasted_vector(self):
        self._helper((10, 3, 4), 4)

    def test_batched_matrix_x_batched_matrix(self):
        # uses `bmm.out`
        self._helper((10, 3, 4), (10, 4, 5))

    def test_batched_matrix_x_broadcasted_matrix(self):
        self._helper((10, 3, 4), (4, 5))


class MPSLeakyReluTest(TestCase):
    def _npLeakyRelu(self, np_features, negative_slope=0.1):
        return np.maximum(np_features, negative_slope * np_features).astype(np_features.dtype)

    def testNpLeakyRelu(self):
        torch.testing.assert_close(
            np.array([[-0.09, 0.7, -0.05, 0.3, -0.01],
                      [0.1, -0.03, 0.5, -0.07, 0.9]]),
            self._npLeakyRelu(
                np.array([[-0.9, 0.7, -0.5, 0.3, -0.1], [0.1, -0.3, 0.5, -0.7,
                                                         0.9]]),
                negative_slope=0.1))

    def _testLeakyRelu(self, np_features, negative_slope, device):
        cpu_x = torch.from_numpy(np_features).requires_grad_()
        mps_x = torch.from_numpy(np_features).to('mps').requires_grad_()
        relu_op = torch.nn.LeakyReLU(negative_slope)

        cpu_leaky_relu = relu_op(cpu_x)
        mps_leaky_relu = relu_op(mps_x)
        torch.testing.assert_close(cpu_leaky_relu, mps_leaky_relu.to('cpu'))

        # test backward pass
        cpu_grad = torch.ones_like(cpu_leaky_relu)
        mps_grad = cpu_grad.to('mps')
        cpu_leaky_relu.backward(gradient=cpu_grad)
        mps_leaky_relu.backward(gradient=mps_grad)
        torch.testing.assert_close(cpu_x.grad, mps_x.grad.to('cpu'))

    def testNumbersCPU(self):
        for t in [np.float32]:
            self._testLeakyRelu(
                np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
                negative_slope=0.2,
                device="cpu")


class TestAvgPool(TestCase):
    def _sum_pool2d(self, x, kernel_size):
        windows = torch.nn.functional.unfold(x, kernel_size=kernel_size, stride=kernel_size)
        return torch.sum(windows, dim=1)

    def _sum_pool3d(self, x, kernel_size):
        # Because unfold does not support 3D sliding window we will split tensor to multiple tensors and calculate sum
        h = kernel_size[0]
        splited_x = [t.sum(0) for t in x.split(h) if t.size(0) == h]
        # sum_pool2d assumes tensor in (1, 1, n, m) view, so unsqueeze two times
        splited_x = [self._sum_pool2d(t.unsqueeze(0).unsqueeze(0), kernel_size[1:]) for t in splited_x]
        joined_x = torch.cat(splited_x)
        return joined_x.view(1, joined_x.numel())

    def _avg_pool2d(self, x, kernel_size):
        size = reduce((lambda x, y: x * y), kernel_size)
        return self._sum_pool2d(x, kernel_size) / size

    def _avg_pool3d(self, x, kernel_size):
        size = reduce((lambda x, y: x * y), kernel_size)
        return self._sum_pool3d(x, kernel_size) / size

    def test_avg_pool2d_with_zero_divisor(self):
        self.assertRaisesRegex(RuntimeError, "divisor must be not zero",
                               lambda: F.avg_pool2d(torch.zeros(3, 3, 3), (2, 2), divisor_override=0))

    def test_doubletensor_avg_pool2d_with_divisor(self):
        n, m = 3, 3
        input = torch.rand(1, 1, n, m)
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                for divisor in [1, 7, i * j]:
                    actual = F.avg_pool2d(input[0], (i, j), divisor_override=divisor)
                    actual = actual.view(1, actual.numel())
                    expected = self._sum_pool2d(input, (i, j)) / divisor
                    self.assertEqual(actual, expected, rtol=0, atol=1e-5)

    def test_avg_pool2d_ceil_mode(self):
        # Regression test for gh-36977
        x = 10 * torch.randn((1, 16, 4, 4))
        y = torch.nn.functional.avg_pool2d(
            x, ceil_mode=True, count_include_pad=True, kernel_size=(1, 2),
            padding=(0, 1), stride=2)
        self.assertTrue(not torch.isnan(y).any())
        y = torch.nn.functional.avg_pool2d(
            x.to('mps'), ceil_mode=True, count_include_pad=True, kernel_size=(1, 2),
            padding=(0, 1), stride=2)
        self.assertTrue(not torch.isnan(y).any())


class TestMPS(TestCase):
    def test_exp(self, device="mps", dtype=torch.float):
        for v in (2, -2) + ((1j, 1 + 1j) if dtype.is_complex else ()):
            b = torch.arange(18, device="cpu") / 3 * math.pi
            a = torch.tensor(v, dtype=dtype, device="cpu") * b
            a = a.to(dtype).to("mps")
            self.compare_with_numpy(torch.exp, np.exp, a)

    def test_exp1(self, device="mps", dtype=torch.float):
        input = torch.tensor([-0.1, 3.0, -0.9]).to('mps')
        output = torch.exp(input).to('cpu')

    def _testLeakyRelu(self, np_features, negative_slope, device):
        cpu_x = torch.from_numpy(np_features).requires_grad_()
        mps_x = torch.from_numpy(np_features).to('mps').requires_grad_()
        relu_op = torch.nn.LeakyReLU(negative_slope)

        cpu_leaky_relu = relu_op(cpu_x)
        mps_leaky_relu = relu_op(mps_x)
        torch.testing.assert_close(cpu_leaky_relu, mps_leaky_relu.to('cpu'))

        # test backward pass
        cpu_grad = torch.ones_like(cpu_leaky_relu)
        mps_grad = cpu_grad.to('mps')
        cpu_leaky_relu.backward(gradient=cpu_grad)
        mps_leaky_relu.backward(gradient=mps_grad)
        torch.testing.assert_close(cpu_x.grad, mps_x.grad.to('cpu'))

    def testNumbersGPU(self):
        for t in [np.float32]:
            self._testLeakyRelu(
                np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
                negative_slope=0.1,
                device="mps")

    def test_fill(self):

        def helper(val, shape):
            tensor = torch.zeros(shape, device='mps')
            tensor_mps = tensor.fill_(val)
            tensor_mps = torch.tanh(tensor_mps)

            tensor_0 = torch.zeros(shape, device='cpu')
            tensor_cpu = tensor_0.fill_(val)
            tensor_cpu = torch.tanh(tensor_cpu)

            self.assertEqual(tensor_mps, tensor_cpu)

        helper(0, [1024])
        helper(0.2, [2, 3])

    def test_mm(self):
        B = torch.ones(5, 6).to("mps")
        C = torch.ones(6, 5).to("mps")
        D = torch.mm(B, C).cpu()
        torch.testing.assert_close(D, torch.full((5, 5), 6.0))

    def test_addmm(self):
        A = torch.ones(5, 5).to("mps")
        B = torch.ones(5, 6).to("mps")
        C = torch.ones(6, 5).to("mps")
        D = torch.addmm(A, B, C).to("cpu")
        torch.testing.assert_close(D, torch.full((5, 5), 7.0))

    def test_bmm(self):
        batch1_cpu = torch.randn(10, 3, 4)
        batch2_cpu = torch.randn(10, 4, 5)

        batch1_mps = batch1_cpu.detach().clone().to("mps")
        batch2_mps = batch2_cpu.detach().clone().to("mps")

        output_cpu = torch.bmm(batch1_cpu, batch2_cpu)
        output_mps = torch.bmm(batch1_mps, batch2_mps)

        self.assertEqual(output_cpu, output_mps)
        self.assertEqual(output_cpu.size(), output_mps.size())

    def test_addbmm(self):
        M_cpu = torch.randn(3, 5)
        batch1_cpu = torch.randn(10, 3, 4)
        batch2_cpu = torch.randn(10, 4, 5)

        M_mps = M_cpu.detach().clone().to("mps")
        batch1_mps = batch1_cpu.detach().clone().to("mps")
        batch2_mps = batch2_cpu.detach().clone().to("mps")

        output_cpu = torch.addbmm(M_cpu, batch1_cpu, batch2_cpu)
        output_mps = torch.addbmm(M_mps, batch1_mps, batch2_mps)

        self.assertEqual(output_cpu, output_mps)
        self.assertEqual(output_cpu.size(), output_mps.size())

    def test_baddbmm(self):
        def helper(input_shape, batch1_shape, batch2_shape):
            M_cpu = torch.randn(input_shape)
            batch1_cpu = torch.randn(batch1_shape)
            batch2_cpu = torch.randn(batch2_shape)
            alpha = 1.2
            beta = 0.8

            M_mps = M_cpu.detach().clone().to("mps")
            batch1_mps = batch1_cpu.detach().clone().to("mps")
            batch2_mps = batch2_cpu.detach().clone().to("mps")

            output_cpu = torch.baddbmm(M_cpu, batch1_cpu, batch2_cpu, beta=beta, alpha=alpha)
            output_mps = torch.baddbmm(M_mps, batch1_mps, batch2_mps, beta=beta, alpha=alpha)

            self.assertEqual(output_cpu, output_mps)
            self.assertEqual(output_cpu.size(), output_mps.size())

        helper(input_shape=(3, 5), batch1_shape=(10, 3, 4), batch2_shape=(10, 4, 5))
        helper(input_shape=(10, 3, 5), batch1_shape=(10, 3, 4), batch2_shape=(10, 4, 5))
        helper(input_shape=(1, 77, 77), batch1_shape=(8, 77, 64), batch2_shape=(8, 64, 77))

    def test_local_scalar_dense_mps(self):
        x_cpu = torch.randn(1)
        y_mps = x_cpu.to("mps")
        torch.testing.assert_close(x_cpu.item(), y_mps.item())

    def test_linear_1d_weight(self):
        device = 'cpu'
        projected = torch.rand([8]).to(device)
        x = torch.rand([1, 2, 2, 8]).to(device)
        x_mps = x.to('mps')
        projected_mps = projected.to('mps')
        linear = F.linear(x, projected)
        linear_mps = F.linear(x_mps, projected_mps)

        self.assertEqual(linear, linear_mps)

        projected = torch.rand([1, 8]).to(device)
        x = torch.rand([1, 2, 2, 8]).to(device)
        x_mps = x.to('mps')
        projected_mps = projected.to('mps')
        linear = F.linear(x, projected)
        linear_mps = F.linear(x_mps, projected_mps)

        self.assertEqual(linear, linear_mps)

    def _linear_helper(self, in_features, out_features, shape, bias=True, backward_pass=False):
        cpu_linear = torch.nn.Linear(in_features=in_features, out_features=out_features, device="cpu", bias=bias)
        mps_linear = torch.nn.Linear(in_features=in_features, out_features=out_features, device="mps", bias=bias)

        # Use the same weights and bias as the ones from the cpu
        mps_linear.weight.data = cpu_linear.weight.data.detach().clone().to("mps")

        if bias:
            mps_linear.bias.data = cpu_linear.bias.data.detach().clone().to("mps")

        linear_mps_input = torch.randn(shape).to('mps')
        linear_cpu_input = linear_mps_input.detach().clone().to('cpu')

        if backward_pass:
            linear_mps_input = linear_mps_input.requires_grad_()
            linear_cpu_input = linear_cpu_input.requires_grad_()

        linear_cpu_output = cpu_linear(linear_cpu_input)
        linear_mps_output = mps_linear(linear_mps_input)

        self.assertEqual(linear_cpu_output, linear_mps_output.to('cpu'))
        self.assertEqual(linear_cpu_output.size(), linear_mps_output.size())

        if backward_pass:
            cpu_grad = torch.ones_like(linear_cpu_output)
            grad = cpu_grad.to('mps')

            linear_cpu_output.backward(gradient=cpu_grad)
            linear_mps_output.backward(gradient=grad)

            self.assertEqual(linear_cpu_input.grad.size(), linear_mps_input.grad.size())
            self.assertEqual(linear_cpu_input.grad, linear_mps_input.grad.to("cpu"), atol=8e-04, rtol=10.4e-05)

            self.assertEqual(cpu_linear.weight.grad.size(), mps_linear.weight.grad.size())
            self.assertEqual(cpu_linear.weight.grad, mps_linear.weight.grad.to("cpu"), atol=8e-04, rtol=10.4e-05)
            if bias:
                self.assertEqual(cpu_linear.bias.grad.size(), mps_linear.bias.grad.size())
                self.assertEqual(cpu_linear.bias.grad, mps_linear.bias.grad.to("cpu"), atol=8e-04, rtol=10.4e-05)

    def test_linear1D(self):
        self._linear_helper(in_features=2, out_features=3, shape=([2]), bias=True, backward_pass=False)

    def test_linear1D_backward(self):
        self._linear_helper(in_features=2, out_features=3, shape=([2]), bias=True, backward_pass=True)

    def test_linear2D(self):
        self._linear_helper(in_features=2, out_features=3, shape=((4, 2)), bias=True, backward_pass=False)

    def test_linear2D_backward(self):
        self._linear_helper(in_features=2, out_features=3, shape=((4, 2)), bias=True, backward_pass=True)

    def test_linear2D_no_bias(self):
        self._linear_helper(in_features=2, out_features=3, shape=((4, 2)), bias=False, backward_pass=False)

    def test_linear2D_no_bias_backward(self):
        self._linear_helper(in_features=2, out_features=3, shape=((4, 2)), bias=False, backward_pass=True)

    def test_linear3D(self):
        self._linear_helper(in_features=2, out_features=3, shape=((4, 5, 2)), bias=True, backward_pass=False)

    def test_linear3D_backward(self):
        self._linear_helper(in_features=2, out_features=3, shape=((4, 5, 2)), bias=True, backward_pass=True)

    def test_linear3D_no_bias(self):
        self._linear_helper(in_features=2, out_features=3, shape=((4, 5, 2)), bias=True, backward_pass=False)

    def test_linear3D_no_bias_backward(self):
        self._linear_helper(in_features=2, out_features=3, shape=((4, 5, 2)), bias=True, backward_pass=True)

    def test_uniform(self):
        low = torch.zeros(5, 5, requires_grad=True)
        high = (torch.ones(5, 5) * 3).requires_grad_()
        low_1d = torch.zeros(1, requires_grad=True)
        high_1d = (torch.ones(1) * 3).requires_grad_()
        self.assertEqual(Uniform(low, high).sample().size(), (5, 5))
        self.assertEqual(Uniform(low, high).sample((7,)).size(), (7, 5, 5))
        self.assertEqual(Uniform(low_1d, high_1d).sample().size(), (1,))
        self.assertEqual(Uniform(low_1d, high_1d).sample((1,)).size(), (1, 1))
        self.assertEqual(Uniform(0.0, 1.0).sample((1,)).size(), (1,))

        # Check log_prob computation when value outside range
        uniform = Uniform(low_1d, high_1d, validate_args=False)
        above_high = torch.tensor([4.0])
        below_low = torch.tensor([-1.0])
        self.assertEqual(uniform.log_prob(above_high).item(), -inf)
        self.assertEqual(uniform.log_prob(below_low).item(), -inf)

        # check cdf computation when value outside range
        self.assertEqual(uniform.cdf(below_low).item(), 0)
        self.assertEqual(uniform.cdf(above_high).item(), 1)

        state = torch.get_rng_state()
        rand = low.new(low.size()).uniform_()
        torch.set_rng_state(state)
        u = Uniform(low, high).rsample()
        u.backward(torch.ones_like(u))
        self.assertEqual(low.grad, 1 - rand)
        self.assertEqual(high.grad, rand)
        low.grad.zero_()
        high.grad.zero_()

    # Test forward maxpool2d
    def test_max_pool2d(self):
        def helper(shape, ks, padding=0, dilation=1, ceil_mode=False, return_indices=False, test_ties=False):

            cpu_x = None
            if(test_ties):
                cpu_x = torch.ones(shape, device='cpu', dtype=torch.float, requires_grad=True)
            else:
                cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            pool = torch.nn.MaxPool2d(kernel_size=ks, padding=padding, dilation=dilation,
                                      ceil_mode=ceil_mode, return_indices=return_indices)

            if(return_indices is False):
                y = pool(x)
                ref_y = pool(cpu_x)

                cpu_grad = torch.ones_like(ref_y)
                grad = cpu_grad.to('mps')

                y.backward(gradient=grad)
                ref_y.backward(gradient=cpu_grad)

                self.assertEqual(y, ref_y)
                self.assertEqual(x.grad, cpu_x.grad)
            else:
                y, idx = pool(x)
                ref_y, ref_idx = pool(cpu_x)

                cpu_grad = torch.ones_like(ref_y)
                grad = cpu_grad.to('mps')

                y.backward(gradient=grad)
                ref_y.backward(gradient=cpu_grad)

                self.assertEqual(y, ref_y)
                self.assertEqual(idx, ref_idx)
                self.assertEqual(x.grad, cpu_x.grad)

        # Test with no batch dimension
        helper((8, 4, 4), ks=2)
        helper((2, 8, 4, 4), ks=2)
        helper((1, 1000, 32, 32), ks=4)
        helper((1, 1000, 1, 4), ks=(1, 4))  # test for max_pool1d
        # Test padding
        helper((1, 1000, 32, 32), ks=4, padding=1)
        helper((1, 1000, 1, 4), ks=(1, 4), padding=(0, 1))  # test for max_pool1d
        # Test dilation
        helper((1, 1000, 32, 32), ks=4, dilation=2)
        helper((1, 1000, 1, 4), ks=(1, 4), padding=(0, 2))  # test for max_pool1d
        # Test ceil mode
        helper((1, 1000, 32, 32), ks=4, ceil_mode=True)
        helper((1, 1000, 1, 4), ks=(1, 4), ceil_mode=True)  # test for max_pool1d

        # Test return indices
        for test_ties in [False, True]:
            # Test with no batch dimension
            helper((8, 4, 4), ks=2, return_indices=True, test_ties=test_ties)
            helper((2, 8, 4, 4), ks=2, return_indices=True, test_ties=test_ties)
            helper((1, 1000, 32, 32), ks=4, return_indices=True, test_ties=test_ties)
            helper((1, 1000, 1, 4), ks=(1, 4), return_indices=True, test_ties=test_ties)  # test for max_pool1d
            # Test padding
            helper((1, 1000, 32, 32), ks=4, padding=1, return_indices=True, test_ties=test_ties)
            helper((1, 1000, 1, 4), ks=(1, 4), padding=(0, 1),
                   return_indices=True, test_ties=test_ties)  # test for max_pool1d
            # Test dilation
            helper((1, 1000, 32, 32), ks=4, dilation=2, return_indices=True, test_ties=test_ties)
            helper((1, 1000, 1, 4), ks=(1, 4), padding=(0, 2),
                   return_indices=True, test_ties=test_ties)  # test for max_pool1d
            # Test ceil mode
            helper((1, 1000, 32, 32), ks=4, ceil_mode=True, return_indices=True, test_ties=test_ties)
            helper((1, 1000, 1, 4), ks=(1, 4), ceil_mode=True,
                   return_indices=True, test_ties=test_ties)  # test for max_pool1d

    def test_adaptive_avg_pool2d_output_size_one(self):
        def helper(size, memory_format):
            x = torch.randint(1, 10, size, dtype=torch.float, device='mps', requires_grad=True)
            if memory_format == 'non_contiguous':
                x = x[::2, ::2, ::2, ::2]
            else:
                x = x.to(memory_format=memory_format)

            net = torch.nn.AdaptiveAvgPool2d((1, 1))
            out = net(x)
            ref_out = x.contiguous().mean((-1, -2)).view((x.size(0), x.size(1), 1, 1))

            out.sum().backward()    # make sure it doesn't crash

            self.assertEqual(out, ref_out)
            if memory_format == torch.channels_last:
                self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))
                c = out.size(1)
                self.assertEqual(out.stride(), [c, 1, c, c])
            else:
                self.assertTrue(out.is_contiguous())
                c = out.size(1)
                self.assertEqual(out.stride(), [c, 1, 1, 1])

        helper((2, 3, 6, 6), torch.contiguous_format)

    def test_masked_fill(self):
        device = "mps"
        dtype = torch.float32
        mask_dtype = torch.bool

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            num_dest = 10
            dst = torch.zeros(num_dest, dtype=dtype, device=device)
            mask = torch.randint(2, (num_dest,), dtype=mask_dtype, device=device)
            val = random.random()
            dst2 = torch.zeros(num_dest, dtype=dtype)
            mask_cpu = mask.to("cpu")

            dst.masked_fill_(mask, val)
            for i in range(num_dest):
                if mask_cpu[i]:
                    dst2[i] = val
            self.assertEqual(dst.to("cpu"), dst2, atol=0, rtol=0)

            # test non-contiguous case
            dst = ((torch.randn(num_dest, num_dest, num_dest) * 10).to(dtype)).permute((2, 0, 1))
            dst2 = dst.contiguous()
            if dtype.is_complex:
                mask = dst.abs() > 0
            else:
                mask = dst > 0
            self.assertTrue(not dst.is_contiguous())
            self.assertTrue(dst2.is_contiguous())
            dst.masked_fill_(mask.to(mask_dtype), val)
            dst2.masked_fill_(mask.to(mask_dtype), val)
            self.assertEqual(dst, dst2, atol=0, rtol=0)

            if mask_dtype == torch.uint8:
                self.assertEqual(len(w), 3)

                warn = 'masked_fill_ received a mask with dtype torch.uint8,'
                for wi in w:
                    self.assertEqual(str(wi.message)[0:52], str(warn))
            else:
                self.assertEqual(len(w), 0)

    def test_nhwc_operation(self):
        def helper(shape, channels_last=False):
            import numpy as np
            np.random.seed(332)
            arr = (256 - 128) * np.random.random_sample(size=shape) + 128
            cpu_x = torch.tensor(arr, device='cpu', dtype=torch.float, requires_grad=True)
            if(channels_last):
                cpu_x = cpu_x.to(memory_format=torch.channels_last)
                cpu_x.retain_grad()
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            # This passes
            self.assertEqual(x, cpu_x)

        helper((2, 2, 2, 2), True)

    # Test forward batch norm
    def test_batch_norm(self):
        def helper(shape, eps=1, momentum=0.1, wts=False, training=False, channels_last=False,
                   track_running_stats=True, test_module=False):

            import numpy as np
            np.random.seed(332)
            arr = (256 - 128) * np.random.random_sample(size=shape) + 128
            cpu_x = torch.tensor(arr, device='cpu', dtype=torch.float, requires_grad=True)
            if(channels_last):
                cpu_x = cpu_x.to(memory_format=torch.channels_last)
                cpu_x.retain_grad()
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            mean_shape = [shape[1]]
            cpu_running_mean = None
            cpu_running_var = None
            running_mean = None
            running_var = None
            if(track_running_stats):
                mean_arr = (240 - 140) * np.random.random_sample(size=mean_shape) + 140
                cpu_running_mean = torch.tensor(mean_arr, device='cpu', dtype=torch.float)
                var_arr = 32 * np.random.random_sample(size=mean_shape)
                cpu_running_var = torch.tensor(var_arr, device='cpu', dtype=torch.float)
                running_mean = cpu_running_mean.detach().clone().to('mps')
                running_var = cpu_running_var.detach().clone().to('mps')

            weight = None
            cpu_weight = None
            bias = None
            cpu_bias = None
            if(wts):
                cpu_weight = torch.randn(mean_shape, device='cpu', dtype=torch.float, requires_grad=True)
                weight = cpu_weight.detach().clone().to('mps').requires_grad_()
                cpu_bias = torch.randn(mean_shape, device='cpu', dtype=torch.float, requires_grad=True)
                bias = cpu_bias.detach().clone().to('mps').requires_grad_()

            y = None
            ref_y = None

            if(not test_module):
                y = torch.nn.functional.batch_norm(x, running_mean, running_var,
                                                   weight=weight,
                                                   bias=bias,
                                                   training=training,
                                                   momentum=momentum, eps=eps)
                ref_y = torch.nn.functional.batch_norm(cpu_x, cpu_running_mean, cpu_running_var,
                                                       weight=cpu_weight,
                                                       bias=cpu_bias,
                                                       training=training,
                                                       momentum=momentum, eps=eps)

            else:

                batchnorm_op = None
                mps_batchnorm_op = None

                if(len(shape) == 3):
                    batchnorm_op = torch.nn.BatchNorm1d(shape[1],
                                                        eps=eps,
                                                        momentum=momentum,
                                                        affine=wts,
                                                        track_running_stats=track_running_stats,
                                                        device='cpu')
                    mps_batchnorm_op = torch.nn.BatchNorm1d(shape[1],
                                                            eps=eps,
                                                            momentum=momentum,
                                                            affine=wts,
                                                            track_running_stats=track_running_stats,
                                                            device='mps')
                elif(len(shape) == 4):
                    batchnorm_op = torch.nn.BatchNorm2d(shape[1],
                                                        eps=eps,
                                                        momentum=momentum,
                                                        affine=wts,
                                                        track_running_stats=track_running_stats,
                                                        device='cpu')
                    mps_batchnorm_op = torch.nn.BatchNorm2d(shape[1],
                                                            eps=eps,
                                                            momentum=momentum,
                                                            affine=wts,
                                                            track_running_stats=track_running_stats,
                                                            device='mps')
                elif(len(shape) == 5):
                    batchnorm_op = torch.nn.BatchNorm3d(shape[1],
                                                        eps=eps,
                                                        momentum=momentum,
                                                        affine=wts,
                                                        track_running_stats=track_running_stats,
                                                        device='cpu')
                    mps_batchnorm_op = torch.nn.BatchNorm3d(shape[1],
                                                            eps=eps,
                                                            momentum=momentum,
                                                            affine=wts,
                                                            track_running_stats=track_running_stats,
                                                            device='mps')

                if(track_running_stats):
                    batchnorm_op.running_mean = cpu_running_mean
                    batchnorm_op.running_var = cpu_running_var
                    mps_batchnorm_op.running_mean = running_mean
                    mps_batchnorm_op.running_var = running_var
                if(wts):
                    batchnorm_op.weight = torch.nn.Parameter(cpu_weight)
                    batchnorm_op.bias = torch.nn.Parameter(cpu_bias)
                    mps_batchnorm_op.weight = torch.nn.Parameter(weight)
                    mps_batchnorm_op.bias = torch.nn.Parameter(bias)

                ref_y = batchnorm_op(cpu_x)
                y = mps_batchnorm_op(x)

            self.assertEqual(y, ref_y)
            if(not test_module):
                self.assertEqual(running_mean, cpu_running_mean)
                self.assertEqual(running_var, cpu_running_var)
            else:
                self.assertEqual(mps_batchnorm_op.running_mean, batchnorm_op.running_mean)
                self.assertEqual(mps_batchnorm_op.running_var, batchnorm_op.running_var)

            cpu_grad = torch.randn(ref_y.shape)
            grad = cpu_grad.to('mps')
            ref_y.backward(gradient=cpu_grad)
            y.backward(gradient=grad)

            self.assertEqual(x.grad, cpu_x.grad)
            if(wts):
                if(not test_module):
                    self.assertEqual(weight.grad, cpu_weight.grad)
                    self.assertEqual(bias.grad, cpu_bias.grad)
                else:
                    self.assertEqual(mps_batchnorm_op.weight.grad, batchnorm_op.weight.grad)
                    self.assertEqual(mps_batchnorm_op.bias.grad, batchnorm_op.bias.grad)

        for shape in [(2, 3, 2, 2), (2, 3, 2, 2, 2), (2, 3, 2)]:
            for test_module in [False, True]:
                for track_running_stats in [True, False]:
                    for channels_last in [False]:
                        if(channels_last and len(shape) != 4):
                            continue
                        # Running stats must be tracked in eval mode
                        if(track_running_stats):
                            helper(shape, eps=0, momentum=1, channels_last=channels_last,
                                   track_running_stats=track_running_stats, test_module=test_module)
                            helper(shape, channels_last=channels_last,
                                   track_running_stats=track_running_stats, test_module=test_module)
                            helper(shape, eps=1e-05, momentum=0.1, wts=False, training=False, channels_last=channels_last,
                                   track_running_stats=track_running_stats, test_module=test_module)
                            helper(shape, eps=0, momentum=1.0, wts=False, training=False, channels_last=channels_last,
                                   track_running_stats=track_running_stats, test_module=test_module)
                            helper(shape, eps=1, momentum=1, wts=True, training=False, channels_last=channels_last,
                                   track_running_stats=track_running_stats, test_module=test_module)
                            helper(shape, eps=3, momentum=0.67, wts=True, training=False, channels_last=channels_last,
                                   track_running_stats=track_running_stats, test_module=test_module)
                        helper(shape, eps=1e-05, momentum=0.1, wts=False, training=True, channels_last=channels_last,
                               track_running_stats=track_running_stats, test_module=test_module)
                        helper(shape, eps=0, momentum=1.0, wts=False, training=True, channels_last=channels_last,
                               track_running_stats=track_running_stats, test_module=test_module)
                        helper(shape, eps=1, momentum=1, wts=True, training=True, channels_last=channels_last,
                               track_running_stats=track_running_stats, test_module=test_module)
                        helper(shape, eps=3, momentum=0.67, wts=True, training=True, channels_last=channels_last,
                               track_running_stats=track_running_stats, test_module=test_module)

    def test_layer_norm(self):
        # TODO: Test non-contiguous
        def helper(input_shape, normalized_shape, eps=1e-05, elementwise_affine=True, dtype=torch.float32):
            cpu_x = torch.randn(input_shape, device='cpu', dtype=dtype, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            cpu_op = torch.nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine, device='cpu', dtype=dtype)
            mps_op = torch.nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine, device='mps', dtype=dtype)
            cpu_wt = torch.randn(normalized_shape, device='cpu', dtype=dtype, requires_grad=True)
            wt = cpu_wt.detach().clone().to('mps').requires_grad_()
            cpu_bias = torch.randn(normalized_shape, device='cpu', dtype=dtype, requires_grad=True)
            bias = cpu_bias.detach().clone().to('mps').requires_grad_()

            if(elementwise_affine):
                cpu_op.weight = torch.nn.Parameter(cpu_wt)
                mps_op.weight = torch.nn.Parameter(wt)
                cpu_op.bias = torch.nn.Parameter(cpu_bias)
                mps_op.bias = torch.nn.Parameter(bias)

            cpu_result = cpu_op(cpu_x)
            result = mps_op(x)

            cpu_grad = torch.randn(cpu_result.shape)
            grad = cpu_grad.to('mps')

            cpu_result.backward(cpu_grad)
            result.backward(grad)

            self.assertEqual(result, cpu_result)
            self.assertEqual(x.grad, cpu_x.grad)
            if(elementwise_affine):
                self.assertEqual(mps_op.weight.grad, cpu_op.weight.grad)
                self.assertEqual(mps_op.bias.grad, cpu_op.bias.grad)

        for elementwise_affine in [True, False]:
            helper((2, 2, 2, 2), (2, 2), elementwise_affine=elementwise_affine)
            helper((2, 3, 4, 5), (4, 5), elementwise_affine=elementwise_affine)
            helper((2, 3, 4, 5, 6), (4, 5, 6), elementwise_affine=elementwise_affine)

    def test_instance_norm(self):
        def helper(shape, eps=1, momentum=0.1, wts=False, channels_last=False, track_running_stats=True, test_module=False):

            import numpy as np
            np.random.seed(332)
            arr = (256 - 128) * np.random.random_sample(size=shape) + 128
            cpu_x = torch.tensor(arr, device='cpu', dtype=torch.float, requires_grad=True)
            if(channels_last):
                cpu_x = cpu_x.to(memory_format=torch.channels_last)
                cpu_x.retain_grad()
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            mean_shape = [shape[1]]
            cpu_running_mean = None
            cpu_running_var = None
            running_mean = None
            running_var = None
            if(track_running_stats):
                mean_arr = (240 - 140) * np.random.random_sample(size=mean_shape) + 140
                cpu_running_mean = torch.tensor(mean_arr, device='cpu', dtype=torch.float)
                var_arr = 32 * np.random.random_sample(size=mean_shape)
                cpu_running_var = torch.tensor(var_arr, device='cpu', dtype=torch.float)
                running_mean = cpu_running_mean.detach().clone().to('mps')
                running_var = cpu_running_var.detach().clone().to('mps')

            weight = None
            cpu_weight = None
            bias = None
            cpu_bias = None
            if(wts):
                cpu_weight = torch.randn(mean_shape, device='cpu', dtype=torch.float, requires_grad=True)
                weight = cpu_weight.detach().clone().to('mps').requires_grad_()
                cpu_bias = torch.randn(mean_shape, device='cpu', dtype=torch.float, requires_grad=True)
                bias = cpu_bias.detach().clone().to('mps').requires_grad_()

            y = None
            ref_y = None

            if(not test_module):
                ref_y = torch.nn.functional.instance_norm(cpu_x, cpu_running_mean, cpu_running_var,
                                                          weight=cpu_weight,
                                                          bias=cpu_bias,
                                                          momentum=momentum, eps=eps)
                y = torch.nn.functional.instance_norm(x, running_mean, running_var,
                                                      weight=weight,
                                                      bias=bias,
                                                      momentum=momentum, eps=eps)

            else:

                instancenorm_op = None
                mps_instancenorm_op = None

                if(len(shape) == 3):
                    instancenorm_op = torch.nn.InstanceNorm1d(shape[1],
                                                              eps=eps,
                                                              momentum=momentum,
                                                              affine=wts,
                                                              track_running_stats=track_running_stats,
                                                              device='cpu')
                    mps_instancenorm_op = torch.nn.InstanceNorm1d(shape[1],
                                                                  eps=eps,
                                                                  momentum=momentum,
                                                                  affine=wts,
                                                                  track_running_stats=track_running_stats,
                                                                  device='mps')
                elif(len(shape) == 4):
                    instancenorm_op = torch.nn.InstanceNorm2d(shape[1],
                                                              eps=eps,
                                                              momentum=momentum,
                                                              affine=wts,
                                                              track_running_stats=track_running_stats,
                                                              device='cpu')
                    mps_instancenorm_op = torch.nn.InstanceNorm2d(shape[1],
                                                                  eps=eps,
                                                                  momentum=momentum,
                                                                  affine=wts,
                                                                  track_running_stats=track_running_stats,
                                                                  device='mps')
                elif(len(shape) == 5):
                    instancenorm_op = torch.nn.InstanceNorm3d(shape[1],
                                                              eps=eps,
                                                              momentum=momentum,
                                                              affine=wts,
                                                              track_running_stats=track_running_stats,
                                                              device='cpu')
                    mps_instancenorm_op = torch.nn.InstanceNorm3d(shape[1],
                                                                  eps=eps,
                                                                  momentum=momentum,
                                                                  affine=wts,
                                                                  track_running_stats=track_running_stats,
                                                                  device='mps')

                if(track_running_stats):
                    instancenorm_op.running_mean = cpu_running_mean
                    instancenorm_op.running_var = cpu_running_var
                    mps_instancenorm_op.running_mean = running_mean
                    mps_instancenorm_op.running_var = running_var
                if(wts):
                    instancenorm_op.weight = torch.nn.Parameter(cpu_weight)
                    instancenorm_op.bias = torch.nn.Parameter(cpu_bias)
                    mps_instancenorm_op.weight = torch.nn.Parameter(weight)
                    mps_instancenorm_op.bias = torch.nn.Parameter(bias)

                ref_y = instancenorm_op(cpu_x)
                y = mps_instancenorm_op(x)

            self.assertEqual(y, ref_y)
            if(not test_module):
                self.assertEqual(running_mean, cpu_running_mean)
                self.assertEqual(running_var, cpu_running_var)
            else:
                self.assertEqual(mps_instancenorm_op.running_mean, instancenorm_op.running_mean)
                self.assertEqual(mps_instancenorm_op.running_var, instancenorm_op.running_var)

            cpu_grad = torch.randn(ref_y.shape)
            grad = cpu_grad.to('mps')
            ref_y.backward(gradient=cpu_grad)
            y.backward(gradient=grad)

            self.assertEqual(x.grad, cpu_x.grad)
            if(wts):
                if(not test_module):
                    self.assertEqual(weight.grad, cpu_weight.grad)
                    self.assertEqual(bias.grad, cpu_bias.grad)
                else:
                    self.assertEqual(mps_instancenorm_op.weight.grad, instancenorm_op.weight.grad)
                    self.assertEqual(mps_instancenorm_op.bias.grad, instancenorm_op.bias.grad)

        for shape in [(2, 3, 2, 2), (2, 3, 2, 2, 2), (2, 3, 2)]:
            for test_module in [False, True]:
                for track_running_stats in [True, False]:
                    for channels_last in [False]:
                        if(channels_last and len(shape) != 4):
                            continue
                        # Running stats must be tracked in eval mode
                        if(track_running_stats):
                            helper(shape, eps=0, momentum=1, channels_last=channels_last,
                                   track_running_stats=track_running_stats, test_module=test_module)
                            helper(shape, channels_last=channels_last,
                                   track_running_stats=track_running_stats, test_module=test_module)
                            helper(shape, eps=1e-05, momentum=0.1, wts=False, channels_last=channels_last,
                                   track_running_stats=track_running_stats, test_module=test_module)
                            helper(shape, eps=0, momentum=1.0, wts=False, channels_last=channels_last,
                                   track_running_stats=track_running_stats, test_module=test_module)
                            helper(shape, eps=1, momentum=1, wts=True, channels_last=channels_last,
                                   track_running_stats=track_running_stats, test_module=test_module)
                            helper(shape, eps=3, momentum=0.67, wts=True, channels_last=channels_last,
                                   track_running_stats=track_running_stats, test_module=test_module)
                        helper(shape, eps=1e-05, momentum=0.1, wts=False, channels_last=channels_last,
                               track_running_stats=track_running_stats, test_module=test_module)
                        helper(shape, eps=0, momentum=1.0, wts=False, channels_last=channels_last,
                               track_running_stats=track_running_stats, test_module=test_module)
                        helper(shape, eps=1, momentum=1, wts=True, channels_last=channels_last,
                               track_running_stats=track_running_stats, test_module=test_module)
                        helper(shape, eps=3, momentum=0.67, wts=True, channels_last=channels_last,
                               track_running_stats=track_running_stats, test_module=test_module)

    # Test conv2d
    def test_conv2d_unit(self):
        def helper(input_shape, wt_shape,
                   stride=1, padding=0,
                   dilation=1, groups=1,
                   bias_shape=None):

            cpu_x = torch.randn(input_shape, device='cpu', dtype=torch.float, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            cpu_wt = torch.randn(wt_shape, device='cpu', dtype=torch.float, requires_grad=True)
            wt = cpu_wt.detach().clone().to('mps').requires_grad_()

            cpu_bias = None
            bias = None

            if(bias_shape is not None):
                cpu_bias = torch.randn(bias_shape, device='cpu', dtype=torch.float, requires_grad=True)
                bias = cpu_bias.detach().clone().to('mps').requires_grad_()

            y = torch.nn.functional.conv2d(x, wt, bias=bias, stride=stride,
                                           padding=padding, dilation=dilation, groups=groups)
            ref_y = torch.nn.functional.conv2d(cpu_x, cpu_wt, bias=cpu_bias, stride=stride,
                                               padding=padding, dilation=dilation, groups=groups)

            cpu_grad = torch.ones_like(ref_y)
            grad = cpu_grad.to('mps')

            y.backward(gradient=grad)
            ref_y.backward(gradient=cpu_grad)

            self.assertEqual(y, ref_y, rtol=2.6e-05, atol=2e-04)
            self.assertEqual(x.grad, cpu_x.grad, rtol=2.6e-06, atol=2e-05)
            self.assertEqual(wt.grad, cpu_wt.grad, atol=8e-04, rtol=10.4e-05)
            if(bias_shape is not None):
                self.assertEqual(bias.grad, cpu_bias.grad, atol=8e-04, rtol=10.4e-05)

        N = 1
        C_in = 3
        C_out = 64
        H = 64
        W = 64
        kH = 4
        kW = 4
        stride = 2
        padding = 1

        helper((N, C_in, H, W), (C_out, C_in, kH, kW), stride=stride, padding=padding)

        N = 4
        C_in = 16
        H = 32
        W = 32

        C_out = 8
        kH = 3
        kW = 3

        for groups in [1, 2, 4]:
            helper((N, C_in, H, W), (C_out, C_in // groups, kH, kW), groups=groups)
            helper((N, C_in, H, W), (C_out, C_in // groups, kH, kW), groups=groups)

            helper((N, C_in, H, W), (C_out, C_in // groups, kH, kW), bias_shape=(C_out), groups=groups)
            helper((N, C_in, H, W), (C_out, C_in // groups, kH, kW), bias_shape=(C_out), groups=groups)

            helper((N, C_in * 2, H * 2, W * 2), (C_out * 2, (C_in * 2) // groups, kH + 2, kW + 2), groups=groups)
            helper((N, C_in * 2, H * 2, W * 2), (C_out * 2, (C_in * 2) // groups, kH + 2, kW + 2), groups=groups)

            helper((N, C_in * 2, H * 2, W * 2), (C_out * 2, (C_in * 2) // groups,
                   kH + 2, kW + 2), bias_shape=(C_out * 2), groups=groups)
            helper((N, C_in * 2, H * 2, W * 2), (C_out * 2, (C_in * 2) // groups,
                   kH + 2, kW + 2), bias_shape=(C_out * 2), groups=groups)

    # Test conv transpose 2d
    def test_conv_transpose2d(self):
        def helper(input_shape, wt_shape,
                   stride=1, padding=0,
                   output_padding=0,
                   dilation=1, groups=1,
                   bias_shape=None):

            cpu_x = torch.randn(input_shape, device='cpu', dtype=torch.float, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            cpu_wt = torch.randn(wt_shape, device='cpu', dtype=torch.float, requires_grad=True)
            wt = cpu_wt.detach().clone().to('mps').requires_grad_()

            cpu_bias = None
            bias = None

            if(bias_shape is not None):
                cpu_bias = torch.randn(bias_shape, device='cpu', dtype=torch.float, requires_grad=True)
                bias = cpu_bias.detach().clone().to('mps').requires_grad_()

            y = torch.nn.functional.conv_transpose2d(
                x, wt, bias=bias, stride=stride, padding=padding, output_padding=output_padding, groups=groups, dilation=dilation)
            ref_y = torch.nn.functional.conv_transpose2d(
                cpu_x, cpu_wt, bias=cpu_bias, stride=stride, padding=padding,
                output_padding=output_padding, groups=groups, dilation=dilation)

            cpu_grad = torch.randn(ref_y.shape)
            grad = cpu_grad.to('mps')

            y.backward(gradient=grad)
            ref_y.backward(gradient=cpu_grad)

            self.assertEqual(y, ref_y, rtol=2.6e-05, atol=2e-04)
            self.assertEqual(x.grad, cpu_x.grad, rtol=2.6e-06, atol=2e-05)
            self.assertEqual(wt.grad, cpu_wt.grad, atol=8e-04, rtol=10.4e-05)

            # if(bias_shape is not None):
            #  print(cpu_bias.grad)
            #  print(bias.grad.to('cpu'))
            #  self.assertEqual(bias.grad, cpu_bias.grad)

        N = 4
        C_in = 2
        H = 32
        W = 32

        C_out = 8
        groups = 1
        kH = 3
        kW = 3

        for stride in [1, 2, 3]:
            for padding in [0, 1, 2]:
                for output_padding in [0, 1, 2]:
                    for dilation in [1, 2]:
                        if(output_padding >= stride or output_padding >= dilation):
                            continue
                        helper((N, C_out, H, W), (C_out, C_in, kH, kW), stride=stride,
                               padding=padding, output_padding=output_padding, dilation=dilation)
                        helper((N, C_out, H, W), (C_out, C_in, kH, kW), stride=stride,
                               padding=padding, output_padding=output_padding, dilation=dilation)

                        helper((N, C_out, H, W), (C_out, C_in, kH, kW), bias_shape=(C_in), stride=stride,
                               padding=padding, output_padding=output_padding, dilation=dilation)
                        helper((N, C_out, H, W), (C_out, C_in, kH, kW), bias_shape=(C_in), stride=stride,
                               padding=padding, output_padding=output_padding, dilation=dilation)

    # Test sigmoid
    def test_sigmoid(self):
        def helper(shape):

            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            sigmoid_op = torch.nn.Sigmoid()

            y = sigmoid_op(x)
            ref_y = sigmoid_op(cpu_x)

            cpu_grad = torch.ones_like(ref_y)
            grad = cpu_grad.to('mps')

            y.backward(gradient=grad)
            ref_y.backward(gradient=cpu_grad)

            self.assertEqual(y, ref_y)
            self.assertEqual(x.grad, cpu_x.grad)

        helper((2, 3, 4, 5))
        helper((2, 3, 4))
        helper((2, 8, 4, 5))

    # Test tanh
    def test_tanh(self):
        def helper(shape):

            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            tanh_op = torch.nn.Tanh()

            y = tanh_op(x)
            ref_y = tanh_op(cpu_x)

            cpu_grad = torch.ones_like(ref_y)
            grad = cpu_grad.to('mps')

            y.backward(gradient=grad)
            ref_y.backward(gradient=cpu_grad)

            self.assertEqual(y, ref_y)
            self.assertEqual(x.grad, cpu_x.grad)

        helper((2, 3, 4, 5))
        helper((2, 3, 4))
        helper((2, 8, 4, 5))

    def test_threshold(self):
        def helper(threshold, value, num_elems, inplace=False, requires_grad=True):
            m = nn.Threshold(threshold=threshold, value=value, inplace=inplace)

            input_cpu = torch.randn(num_elems, requires_grad=requires_grad, dtype=torch.float)
            input_mps = input_cpu.detach().clone().to('mps').requires_grad_(requires_grad)

            output_cpu = m(input_cpu)
            output_mps = m(input_mps)

            cpu_grad = torch.ones_like(output_cpu)
            mps_grad = cpu_grad.to('mps')

            self.assertEqual(output_cpu, output_mps)

            if requires_grad:
                output_cpu.backward(gradient=cpu_grad)
                output_mps.backward(gradient=mps_grad)

                self.assertEqual(input_cpu.grad, input_mps.grad)

        helper(threshold=0.1, value=20, num_elems=2)
        helper(threshold=-0.1, value=10, num_elems=10)
        helper(threshold=0.5, value=-15, num_elems=100)
        helper(threshold=1, value=10, num_elems=100, inplace=True, requires_grad=False)

    # Test pow
    def test_pow(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            x = cpu_x.detach().clone().to('mps')
            cpu_y = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            y = cpu_y.detach().clone().to('mps')
            z = torch.pow(x, y)
            ref_z = torch.pow(cpu_x, cpu_y)

            self.assertEqual(z, ref_z)

            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            x = cpu_x.detach().clone().to('mps')
            exp = random.random()
            z = torch.pow(x, exp)
            ref_z = torch.pow(cpu_x, exp)

            self.assertEqual(z, ref_z)

        helper((2, 8, 4, 5))

    # Test addcmul
    def test_addcmul(self):
        def helper(shape, value):

            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            x = cpu_x.detach().clone().to('mps')

            cpu_y = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            y = cpu_y.detach().clone().to('mps')

            cpu_z = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            z = cpu_z.detach().clone().to('mps')

            y = torch.addcmul(x, y, z, value=value)
            ref_y = torch.addcmul(cpu_x, cpu_y, cpu_z, value=value)

            self.assertEqual(y, ref_y)

        helper((2, 3, 4, 5), 0.1)
        helper((2, 8, 4, 5), 0.1)
        helper((2, 3, 4, 5), 0.2)
        helper((2, 8, 4, 5), 0.2)

    # Test addcdiv
    def test_addcdiv(self):
        def helper(shape, value):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            cpu_y = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            # clamp to avoid division by 0
            cpu_z = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False).clamp_min_(0.1)
            cpu_out = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)

            mps_x = cpu_x.detach().clone().to('mps')
            mps_y = cpu_y.detach().clone().to('mps')
            mps_z = cpu_z.detach().clone().to('mps')
            mps_out = cpu_out.detach().clone().to('mps')

            result_div_mps = torch.addcdiv(mps_x, mps_y, mps_z, value=value)
            result_div_cpu = torch.addcdiv(cpu_x, cpu_y, cpu_z, value=value)
            self.assertEqual(result_div_mps, result_div_cpu)
            # test .out variant
            self.assertEqual(torch.addcdiv(mps_x, mps_y, mps_z, out=mps_out, value=value), result_div_cpu)

        helper((2, 3, 4, 5), 0.1)
        helper((2, 8, 4, 5), 0.2)
        helper((2, 3, 4, 5), 1.0)  # value of 1 should be ignored internally

    def test_buffer_size_match(self):
        # this test shouldn't cause any crash
        size = 16
        cpu_A = torch.rand(size, device='cpu')
        cpu_F = torch.rand(size, size, size, device='cpu')

        mps_A = cpu_A.to('mps')
        mps_F = cpu_F.to('mps')
        self.assertEqual(cpu_A @ cpu_F, mps_A @ mps_F)

    def test_transpose_inplace(self):
        values = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        cpu_x = torch.tensor(values, device='cpu')
        mps_x = torch.tensor(values, device='mps')

        cpu_x.transpose_(0, 1)
        mps_x.transpose_(0, 1)
        self.assertEqual(cpu_x, mps_x.to('cpu'))

    def test_expand_cpu_to_mps_copy(self):
        # https://github.com/pytorch/pytorch/issues/78642

        x = torch.tensor(1).expand([10]).to("mps")
        x_cpu = torch.tensor(1).expand([10])

        self.assertEqual(x_cpu, x.cpu())

    def test_slice(self):
        values = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        cpu_x = torch.tensor(values, device='cpu')
        mps_x = (torch.tensor(values, device='mps', dtype=torch.float))

        cpu_slice1 = cpu_x[:2, :]
        mps_slice1 = mps_x[:2, :]
        self.assertEqual(cpu_slice1, mps_slice1)

        cpu_slice2 = cpu_x[:, :1]
        mps_slice2 = mps_x[:, :1]
        self.assertEqual(cpu_slice2, mps_slice2)

        cpu_slice3 = cpu_x[1:2, :]
        mps_slice3 = mps_x[1:2, :]
        self.assertEqual(cpu_slice3, mps_slice3.to('cpu'))

        cpu_slice4 = cpu_x[1, :]
        mps_slice4 = mps_x[1, :].to('cpu')
        self.assertEqual(cpu_slice4, mps_slice4)

    def test_scalar_from_slice_unary(self):
        # https://github.com/pytorch/pytorch/issues/82543
        tensor_list = torch.tensor([1.0, 1.2], device="mps")

        for scalar in tensor_list:
            r_mps = torch.ceil(scalar)
            r_cpu = torch.ceil(scalar.to("cpu"))
            self.assertEqual(r_mps.cpu(), r_cpu)

    def test_scalar_from_slice_binary(self):
        # https://github.com/pytorch/pytorch/issues/82543
        def helper(binary_op):
            tensor_list = torch.tensor([1.0, 1.2, 2.5, 1.0], device="mps")

            for scalar in tensor_list:
                r_mps = binary_op(scalar, 1.0)
                r_cpu = binary_op(scalar.cpu(), 1.0)
                self.assertEqual(r_mps.cpu(), r_cpu)
        helper(torch.sub)
        helper(torch.add)
        helper(torch.not_equal)
        helper(torch.eq)

    def test_slice_contiguous_view(self):
        # https://github.com/pytorch/pytorch/issues/77750

        def helper(operator):
            t_mps = torch.tensor([1, 2, 3, 4], device="mps")
            t_cpu = torch.tensor([1, 2, 3, 4], device="cpu")

            # contiguous view
            x_mps = t_mps[2:]  # 3, 4
            y_mps = t_mps[:2]  # 1, 2

            x_cpu = t_cpu[2:]
            y_cpu = t_cpu[:2]

            res_mps = res_cpu = None
            if operator == "<=":
                res_mps = x_mps <= y_mps
                res_cpu = x_cpu <= y_cpu
            if operator == "<":
                res_mps = x_mps < y_mps
                res_cpu = x_cpu < y_cpu
            if operator == ">=":
                res_mps = x_mps >= y_mps
                res_cpu = x_cpu >= y_cpu
            if operator == ">":
                res_mps = x_mps >= y_mps
                res_cpu = x_cpu >= y_cpu
            if operator == "==":
                res_mps = x_mps == y_mps
                res_cpu = x_cpu == y_cpu
            if operator == "!=":
                res_mps = x_mps != y_mps
                res_cpu = x_cpu != y_cpu

            self.assertEqual(res_mps, res_cpu)

        for op in ["<=", "<", ">=", ">", "==", "!="]:
            helper(op)

    def test_slice_of_slice(self):
        x = torch.tensor([0.5, 0.5], device="cpu")
        x_mps = torch.tensor([0.5, 0.5], device="mps")

        tensor = x[1][None]
        tensor_mps = x_mps[1][None]

        res = tensor.ne(0)
        res_mps = tensor_mps.ne(0)

        self.assertEqual(res, res_mps)

    def test_index_storage_offset(self):
        # https://github.com/pytorch/pytorch/issues/78107

        a = torch.tensor([8.2670e-01, -1.0293e+00])
        b_cpu = a[0]
        c_cpu = a[1]

        # both 'b' and 'c' are views of 'a'
        # 'b' has a storage offset of 0, while 'c' has a storage offset of 1
        # when copying from 'cpu' to 'mps', c will have a storage_offset of 1 which needs to be taking into account,
        # otherwise it ends with same value as 'b'
        b = b_cpu.to('mps')
        c = c_cpu.to('mps')

        res_mps = b > c
        res_cpu = b_cpu > c_cpu
        self.assertEqual(res_mps, res_cpu)

        res_mps = c > b
        res_cpu = c_cpu > b_cpu
        self.assertEqual(res_mps, res_cpu)

    def test_flatten(self):
        values = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]
        cpu_x = torch.tensor(values, device='cpu')
        mps_x = torch.tensor(values, device='mps')

        cpu_flatten1 = cpu_x.flatten()
        mps_flatten1 = mps_x.flatten().to('cpu')
        self.assertEqual(cpu_flatten1, mps_flatten1)

        cpu_flatten2 = cpu_x.flatten(start_dim=1)
        mps_flatten2 = mps_x.flatten(start_dim=1).to('cpu')
        self.assertEqual(cpu_flatten2, mps_flatten2)

        cpu_flatten3 = cpu_x.flatten(end_dim=1)
        mps_flatten3 = mps_x.flatten(end_dim=1).to('cpu')
        self.assertEqual(cpu_flatten3, mps_flatten3)

    # Test repeat
    def test_repeat(self):
        def helper(shape, repeats):

            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            y = x.repeat(repeats)
            ref_y = cpu_x.repeat(repeats)

            cpu_grad = torch.randn(ref_y.shape)
            grad = cpu_grad.to('mps')

            y.backward(gradient=grad)
            ref_y.backward(gradient=cpu_grad)

            self.assertEqual(y, ref_y)
            self.assertEqual(x.grad, cpu_x.grad)

        helper((2, 3, 4, 5), (2, 3, 4, 5))
        helper((2, 3, 4), (4, 3, 2, 5, 7, 2))
        helper((3, 4, 5), (2, 3, 4, 5))
        helper((3, 4, 5), (2, 2, 2))

    def test_count_nonzero(self):
        def helper(dtype):
            n = [
                [[1, 0, 2], [3, 0, 2], [7, 9, -4]],
                [[0, 2, 3], [3, 2, 1], [2, 0, 0]],
            ]
            cpu_x = torch.tensor(n, dtype=dtype)
            mps_x = torch.tensor(n, dtype=dtype).to('mps')

            # All non-zeros
            self.assertEqual(
                torch.count_nonzero(cpu_x),
                torch.count_nonzero(mps_x)
            )

            # dim=1
            self.assertEqual(
                torch.count_nonzero(cpu_x, dim=1),
                torch.count_nonzero(mps_x, dim=1)
            )

            # dim=(0, 1)
            self.assertEqual(
                torch.count_nonzero(cpu_x, dim=(0, 1)),
                torch.count_nonzero(mps_x, dim=(0, 1))
            )
        helper(torch.int32)
        helper(torch.int64)
        helper(torch.float16)
        helper(torch.float32)

    def _test_module_empty_input(self, module, inp, check_size=True):
        inp.requires_grad_(True)
        out = module(inp)
        gO = torch.rand_like(out)
        out.backward(gO)
        if check_size:
            self.assertEqual(out.size(), inp.size())
        for p in module.parameters():
            if p.requires_grad:
                self.assertEqual(p.grad, torch.zeros_like(p.grad))
        self.assertEqual(inp.grad, torch.zeros_like(inp))

    # Test dtype casting, with and without simultaneous device change
    def test_to(self):
        values = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]
        cpu_x = torch.tensor(values, device='cpu')
        mps_x = torch.tensor(values, device='mps')

        self.assertEqual(cpu_x.int(), mps_x.int().cpu())
        self.assertEqual(cpu_x.bool(), mps_x.bool().cpu())
        self.assertEqual(cpu_x.float(), mps_x.float().cpu())

        self.assertEqual(torch.tensor(1.3, device='mps').int().cpu(),
                         torch.tensor(1, dtype=torch.int32))
        self.assertEqual(torch.tensor(0.0, device='mps').bool().cpu(), torch.tensor(False))
        self.assertEqual(torch.tensor(0.1, device='mps').bool().cpu(), torch.tensor(True))
        self.assertEqual(torch.tensor(0.1, device='mps').bool().int().cpu(),
                         torch.tensor(1, dtype=torch.int32))
        self.assertEqual(torch.tensor(0.1, device='mps').bool().int().float().cpu(),
                         torch.tensor(1.0))
        self.assertEqual(torch.tensor(4.25, device='mps').to('cpu', torch.int),
                         torch.tensor(4, dtype=torch.int32))
        self.assertEqual(torch.tensor(4.25, device='cpu').to('mps', torch.int).cpu(),
                         torch.tensor(4, dtype=torch.int32))
        self.assertEqual(torch.tensor(-8.34, device='cpu').to('mps', torch.int),
                         torch.tensor(-8.34, device='cpu').to('mps').to(torch.int))
        # Cast int8 and uint8 to float and compare results
        # See https://github.com/pytorch/pytorch/issues/80009 for more details
        cpu_byte = torch.tensor([60, 160, 20, 220], dtype=torch.uint8)
        cpu_char = torch.tensor([60, -60, 20, -120], dtype=torch.uint8)
        for x_cpu in [cpu_byte, cpu_char]:
            x_mps = x_cpu.to('mps')
            self.assertEqual(x_mps.to(torch.float32), x_cpu.to(torch.float32))


    def test_setitem_scalar(self) -> None:
        device = 'mps'
        for dtype in [torch.int32, torch.float32, torch.int64]:
            for i in range(3, 6):
                for j in range(3, 6):
                    t = torch.zeros(i, j, dtype=dtype, device=device)
                    self.assertEqual(t.sum(), 0)
                    t[1, 1] = 1
                    t[2, 1] = j
                    t[1, 2] = i
                    self.assertEqual(t[1, 1], 1)
                    self.assertEqual(t[1, 2], i)
                    self.assertEqual(t[2, 1], j)
                    self.assertEqual(t.sum(), 1 + i + j)

    def test_stride_of_strides(self) -> None:
        x = torch.rand(32, 1, device='mps')
        y = x.as_strided(size=(32, 2), stride=(1, 0))
        # Casting stride of strided tensor to CPU use to crash with "buffer is not large enough." assert
        # See https://github.com/pytorch/pytorch/issues/79181#issuecomment-1154683435
        z = y.as_strided(size=(32, 3), stride=(1, 0)).to("cpu")
        self.assertEqual(x.to("cpu").as_strided(size=(32, 3), stride=(1, 0)), z)

    def test_type_casting(self):
        # https://github.com/pytorch/pytorch/issues/81567
        def helper(data, to_dtype):
            a_cpu = torch.tensor(data)
            a_mps = a_cpu.to(torch.device('mps'))

            res_cpu = a_cpu.type(to_dtype)
            res_mps = a_mps.type(to_dtype)
            self.assertEqual(res_cpu, res_mps)

        helper([9.0, 3.0, 5.0, 4.0], torch.LongTensor)
        helper([9.0, 3.0, 5.0, 4.0], torch.FloatTensor)
        helper([9.0, 3.0, 5.0, 4.0], torch.IntTensor)
        helper([9.0, 3.0, 5.0, 4.0], torch.ShortTensor)
        helper([9.0, 3.0, 5.0, 4.0], torch.HalfTensor)
        helper([9.0, 3.0, 5.0, 4.0], torch.CharTensor)
        helper([9.0, 3.0, 5.0, 4.0], torch.ByteTensor)

    def test_to_casting(self):
        # https://github.com/pytorch/pytorch/issues/81567
        def helper(data, to_dtype):
            a_cpu = torch.tensor(data)
            a_mps = a_cpu.to(torch.device('mps'))

            res_cpu = a_cpu.to(to_dtype)
            res_mps = a_mps.to(to_dtype)
            self.assertEqual(res_cpu, res_mps)

        helper([9.0, 3.0, 5.0, 4.0], torch.int64)
        helper([9.0, 3.0, 5.0, 4.0], torch.float)
        helper([9.0, 3.0, 5.0, 4.0], torch.int32)
        helper([9.0, 3.0, 5.0, 4.0], torch.short)
        helper([9.0, 3.0, 5.0, 4.0], torch.half)
        helper([9.0, 3.0, 5.0, 4.0], torch.int8)
        helper([9.0, 3.0, 5.0, 4.0], torch.uint8)

    def test_storage_offset_greater_than_src_nbytes(self):
        # https://github.com/pytorch/pytorch/issues/80844
        n_tensors = 100
        n_tensor_elems = 784
        elems = torch.arange(n_tensors * n_tensor_elems, dtype=torch.float32)

        tensor_list = []
        for i in range(0, n_tensors - 1):
            # create a list of contiguous view tensors (view tensor created by the slice op)
            t = elems[n_tensor_elems * i : n_tensor_elems * (i + 1)]
            tensor_list.append(t)

        for i in range(0, n_tensors - 1):
            t = tensor_list[i].view(1, n_tensor_elems)
            t_mps = t.to("mps")
            self.assertEqual(t, t_mps.cpu(), f"i={i}")

    # See https://github.com/pytorch/pytorch/issues/82427
    # and https://github.com/pytorch/pytorch/issues/83692
    def test_full_bugs(self):
        # Test should not crash
        x = torch.full((3, 3), True, device='mps')
        # torch.full should work for uint8
        y_mps = torch.full((2, 2), 247, device='mps', dtype=torch.uint8)
        y_cpu = torch.full((2, 2), 247, device='cpu', dtype=torch.uint8)
        self.assertEqual(y_mps, y_cpu)

    # See https://github.com/pytorch/pytorch/issues/84995
    def test_div_bugs(self):
        for (dtype, mode) in itertools.product(integral_types(), ['trunc', 'floor']):
            x = torch.tensor(list(range(1, 11)), device='mps', dtype=dtype)
            y = torch.div(x, 101, rounding_mode=mode)
            self.assertEqual(y.sum(), 0)

    # See https://github.com/pytorch/pytorch/issues/82663
    def test_bool_expand(self):
        x = torch.tensor([[1], [0]], dtype=torch.bool, device='mps')
        y = torch.tensor([0, 1], dtype=torch.bool, device='mps')
        self.assertFalse(torch.equal(x.expand(2, 2), y.expand(2, 2)))

    # Empty unary op should return tensor of the same size
    def test_empty_neg(self):
        x = torch.tensor([[]], device='mps')
        y = -x
        self.assertEqual(x, y)

    # See https://github.com/pytorch/pytorch/issues/85675
    def test_cat_non_contiguous(self):
        def rotate_subset(data):
            return torch.concat([data[:, :2], torch.rot90(data[:, 2:])])
        for dtype in MPS_DTYPES:
            if dtype == torch.bool:
                continue
            data = torch.arange(8, dtype=dtype).reshape(2, 4)
            mps_data = data.to("mps")
            cpu_result = rotate_subset(data)
            mps_result = rotate_subset(mps_data)
            self.assertEqual(cpu_result, mps_result.to("cpu"))

    # See https://github.com/pytorch/pytorch/issues/85967
    def test_from_numpy_non_contiguous(self):
        a = np.arange(9).reshape(3, 3)[:, :2]
        t_cpu = torch.tensor(a, device="cpu")
        t_mps = torch.tensor(a, device="mps")
        self.assertEqual(t_cpu, t_mps.to("cpu"))

    # See https://github.com/pytorch/pytorch/issues/86954
    def test_copy_non_contiguous(self):
        x = torch.arange(27).reshape(3, 3, 3).permute(2, 0, 1)
        self.assertFalse(x.is_contiguous())
        y = x.to('mps')
        self.assertFalse(y.is_contiguous())
        self.assertEqual(x, y.to('cpu'))

        x = torch.arange(4**3).reshape(4, 4, 4).permute((2, 0, 1))[1:, ::2]
        y = x.to('mps')
        self.assertEqual(x, y.to('cpu'))

        x = torch.full((4, 4, 4, 4), 13, device="cpu")
        y = torch.full((4, 4, 4, 4), 13, device="mps")
        z = torch.arange(4**4).reshape(4, 4, 4, 4).permute(3, 2, 0, 1)[1::, ::2]
        x.permute(3, 2, 1, 0)[1::, ::2] = z
        # As y is on MPS and z on CPU, this dispatches to a copy operator
        y.permute(3, 2, 1, 0)[1::, ::2] = z
        self.assertEqual(x, y.to('cpu'))



class TestLogical(TestCase):
    def _wrap_tensor(self, x, device="cpu", dtype=None, requires_grad=False):
        return torch.tensor(x, device=device, dtype=dtype, requires_grad=requires_grad)

    def test_logical_not(self):
        def helper(x):
            cpu_x = x
            x = cpu_x.detach().clone().to('mps')

            result = torch.logical_not(x)
            result_cpu = torch.logical_not(cpu_x)

            self.assertEqual(result, result_cpu)

        helper(self._wrap_tensor([1, 1, 0, 0]))
        helper(self._wrap_tensor([1, 1, 0, 0], dtype=torch.float, requires_grad=True))
        helper(self._wrap_tensor([True, True, False, False]))
        helper(self._wrap_tensor(1))
        helper(self._wrap_tensor(0))
        helper(self._wrap_tensor(True))
        helper(self._wrap_tensor(False))

    def test_logical_and(self):
        def helper(x, other):
            cpu_x = x
            x = cpu_x.detach().clone().to('mps')

            cpu_other = other
            other = cpu_other.detach().clone().to('mps')

            result = torch.logical_and(x, other)
            result_cpu = torch.logical_and(cpu_x, cpu_other)
            self.assertEqual(result, result_cpu)

        helper(self._wrap_tensor([1, 1, 0, 0]), self._wrap_tensor(([1, 0, 0, 1])))
        helper(
            self._wrap_tensor([1, 1, 0, 0], dtype=torch.float, requires_grad=True),
            self._wrap_tensor([1, 0, 0, 1], dtype=torch.float)
        )
        helper(self._wrap_tensor([True, True, False, False]), self._wrap_tensor([True, False, False, True]))
        helper(self._wrap_tensor((1, 0, 1, 0)), self._wrap_tensor(1))
        helper(self._wrap_tensor((1, 0, 1, 0)), self._wrap_tensor(0))
        helper(self._wrap_tensor((1, 0, 1, 0)), self._wrap_tensor(True))
        helper(self._wrap_tensor((1, 0, 1, 0)), self._wrap_tensor(False))

    def test_logical_or(self):
        def helper(x, other):
            cpu_x = x
            x = cpu_x.detach().clone().to('mps')

            cpu_other = other
            other = cpu_other.detach().clone().to('mps')

            result = torch.logical_or(x, other)
            result_cpu = torch.logical_or(cpu_x, cpu_other)

            self.assertEqual(result, result_cpu)

        helper(self._wrap_tensor([1, 1, 0, 0]), self._wrap_tensor(([1, 0, 0, 1])))
        helper(
            self._wrap_tensor([1, 1, 0, 0], dtype=torch.float, requires_grad=True),
            self._wrap_tensor([1, 0, 0, 1], dtype=torch.float)
        )
        helper(self._wrap_tensor([True, True, False, False]), self._wrap_tensor([True, False, False, True]))
        helper(self._wrap_tensor((1, 0, 1, 0)), self._wrap_tensor(1))
        helper(self._wrap_tensor((1, 0, 1, 0)), self._wrap_tensor(0))
        helper(self._wrap_tensor((1, 0, 1, 0)), self._wrap_tensor(True))
        helper(self._wrap_tensor((1, 0, 1, 0)), self._wrap_tensor(False))

    def test_logical_xor(self):
        def helper(x, other):
            cpu_x = x
            x = cpu_x.detach().clone().to('mps')

            cpu_other = other
            other = cpu_other.detach().clone().to('mps')

            result = torch.logical_xor(x, other)
            result_cpu = torch.logical_xor(cpu_x, cpu_other)

            self.assertEqual(result, result_cpu)

        helper(self._wrap_tensor([1, 1, 0, 0]), self._wrap_tensor(([1, 0, 0, 1])))
        helper(
            self._wrap_tensor([1, 1, 0, 0], dtype=torch.float, requires_grad=True),
            self._wrap_tensor([1, 0, 0, 1], dtype=torch.float)
        )
        helper(self._wrap_tensor([True, True, False, False]), self._wrap_tensor([True, False, False, True]))
        helper(self._wrap_tensor((1, 0, 1, 0)), self._wrap_tensor(1))
        helper(self._wrap_tensor((1, 0, 1, 0)), self._wrap_tensor(0))
        helper(self._wrap_tensor((1, 0, 1, 0)), self._wrap_tensor(True))
        helper(self._wrap_tensor((1, 0, 1, 0)), self._wrap_tensor(False))


class TestSmoothL1Loss(TestCase):

    def _smooth_l1_loss_helper(self, reduction="mean", requires_grad=False):
        # CPU
        input_cpu = torch.randn(4, 7, requires_grad=requires_grad)
        target_cpu = torch.randn(4, 7)

        # MPS
        input_mps = input_cpu.detach().clone().to('mps').requires_grad_()
        target_mps = target_cpu.detach().clone().to('mps')

        smooth_l1_loss_cpu = F.smooth_l1_loss(input_cpu, target_cpu, beta=1.0, reduction=reduction)
        smooth_l1_loss_mps = F.smooth_l1_loss(input_mps, target_mps, beta=1.0, reduction=reduction)

        self.assertEqual(smooth_l1_loss_cpu, smooth_l1_loss_mps)

        if requires_grad:
            smooth_l1_loss_cpu.backward()
            smooth_l1_loss_mps.backward()
            self.assertEqual(input_cpu.grad, input_mps.grad.to("cpu"))

        return smooth_l1_loss_cpu, smooth_l1_loss_mps

    def test_smooth_l1_loss_reduction_none(self):
        self._smooth_l1_loss_helper(reduction="none")

    def test_smooth_l1_loss_reduction_mean(self):
        self._smooth_l1_loss_helper(reduction="mean")

    def test_smooth_l1_loss_reduction_sum(self):
        self._smooth_l1_loss_helper(reduction="sum")

    def test_smooth_l1_loss_reduction_mean_backward(self):
        self._smooth_l1_loss_helper(reduction="mean", requires_grad=True)

    def test_smooth_l1_loss_reduction_mean_sum_backward(self):
        self._smooth_l1_loss_helper(reduction="sum", requires_grad=True)


class TestNLLLoss(TestCase):
    def test_nll_loss_mismatched_batch(self, device='mps'):
        x = torch.randn((10, 3), requires_grad=True, device=device)
        # t should have size (10,)
        t = torch.zeros((3,), dtype=torch.int64, device=device)
        with self.assertRaisesRegex(ValueError, 'Expected.*batch_size'):
            F.nll_loss(x, t)

    def test_nll_loss_out_of_bounds_ignore_index(self):

        def _test_nll_loss_out_of_bounds_ignore_index(device):
            output = []
            x = torch.tensor([[0.3, 0.5, 0.2], [0.1, 0.7, 0.2], [0.4, 0.5, 0.1], [
                             0.3, 0.5, 0.2], [0.1, 0.7, 0.2], [0.4, 0.5, 0.1]], device=device)
            t = torch.tensor([0, 1, 255, 0, 1, 2], dtype=torch.int64, device=device)
            for reduction in ['mean', 'none']:
                output.append(F.nll_loss(x, t, ignore_index=255, reduction=reduction))
            return output

        output_cpu = _test_nll_loss_out_of_bounds_ignore_index(device='cpu')
        output_mps = _test_nll_loss_out_of_bounds_ignore_index(device='mps')

        for cpu, mps in zip(output_cpu, output_mps):
            self.assertEqual(cpu, mps.to('cpu'))

    def test_nll_loss_invalid_target_dim(self):

        def _test_nll_loss_invalid_target_dim(device):
            output = []
            x = torch.tensor([[0.3, 0.5, 0.2], [0.1, 0.7, 0.2], [0.4, 0.5, 0.1], [
                             0.3, 0.5, 0.2], [0.1, 0.7, 0.2], [0.4, 0.5, 0.1]], device=device)
            t = torch.zeros((6, 2), dtype=torch.int64, device=device)
            with self.assertRaisesRegex(RuntimeError, "1D target tensor expected"):
                F.nll_loss(x, t)

        _test_nll_loss_invalid_target_dim(device='cpu')
        _test_nll_loss_invalid_target_dim(device='mps')

    def test_nll_loss_invalid_weights(self):

        def _test_nll_loss_invalid_weights(device):
            x = torch.tensor([[0.3, 0.5, 0.2], [0.1, 0.7, 0.2], [0.4, 0.5, 0.1], [
                             0.3, 0.5, 0.2], [0.1, 0.7, 0.2], [0.4, 0.5, 0.1]], device=device)
            t = torch.tensor([0, 1, 2, 1, 1, 2], dtype=torch.int64, device=device)
            invalid_weights = [
                torch.zeros(4, device=device),
                torch.zeros((1, 3), device=device),
            ]
            msg = "weight tensor should be defined either for all 3 classes or no classes"
            for weight in invalid_weights:
                with self.assertRaisesRegex(RuntimeError, msg):
                    F.nll_loss(x, t, weight=weight)

        _test_nll_loss_invalid_weights(device='cpu')
        _test_nll_loss_invalid_weights(device='mps')

    def _nll_loss_helper(self, input_size, reduction, expected):

        # CPU
        input = torch.rand(input_size, requires_grad=True, device='cpu')
        num_channels = input_size[1]
        target_size = (input_size[0], ) + tuple(input_size[2:])
        target = torch.randint(num_channels, target_size, device='cpu')

        # MPS
        input_mps = input.detach().clone().to('mps').requires_grad_()
        target_mps = target.detach().clone().to('mps')

        output_cpu = F.nll_loss(input, target, reduction=reduction)
        output_mps = F.nll_loss(input_mps, target_mps, reduction=reduction)
        # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
        self.assertEqualIgnoreType(output_cpu, output_mps.to('cpu'))

        output_cpu.sum().backward()
        output_mps.sum().backward()
        self.assertEqual(input.grad, input_mps.grad.to('cpu'))

    def _nll_loss_1d_helper(self, input_size, reduction):

        # CPU
        input = torch.rand(input_size, requires_grad=True, device='cpu')
        num_channels = input_size[0]
        target = torch.randint(num_channels, [], device='cpu')

        # MPS
        input_mps = input.detach().clone().to('mps').requires_grad_()
        target_mps = target.detach().clone().to('mps')

        output_cpu = F.nll_loss(input, target, reduction=reduction)
        output_mps = F.nll_loss(input_mps, target_mps, reduction=reduction)
        # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
        self.assertEqualIgnoreType(output_cpu, output_mps.to('cpu'))

        output_cpu.sum().backward()
        output_mps.sum().backward()
        self.assertEqual(input.grad, input_mps.grad.to('cpu'))

    def test_as_strided(self):
        values = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        values_1 = [[1.0, 1.0], [1.0, 1.0]]
        cpu_x = torch.tensor(values, device='cpu')
        ones1 = torch.tensor(values_1, device='mps')
        x = cpu_x.detach().clone().to('mps').requires_grad_()
        strided_cpu = torch.as_strided(cpu_x, (2, 2), (1, 2))
        strided_mps = torch.as_strided(x, (2, 2), (1, 2))
        self.assertEqual(strided_mps, strided_cpu)
        strided_cpu_out = strided_cpu + ones1.to('cpu')
        strided_mps_out = strided_mps + ones1
        self.assertEqual(strided_cpu_out, strided_mps_out)

        # test with storage offsets
        cpu_x = torch.rand(3, 3, device='cpu')
        mps_x = cpu_x.to('mps')
        strided_cpu1 = torch.as_strided(cpu_x, (2, 2), (1, 2), 0)
        strided_mps1 = torch.as_strided(mps_x, (2, 2), (1, 2), 0)
        strided_cpu2 = torch.as_strided(cpu_x, (2, 2), (1, 2), 1)
        strided_mps2 = torch.as_strided(mps_x, (2, 2), (1, 2), 1)
        strided_cpu_out = strided_cpu1 - strided_cpu2
        strided_mps_out = strided_mps1 - strided_mps2
        self.assertEqual(strided_cpu_out, strided_mps_out)



    def test_sum_backward(self):
        def helper(n, c):
            values = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
            cpu_x = torch.tensor(values, device='cpu', requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            all_sum = torch.sum(x)
            all_sum_cpu = torch.sum(cpu_x)

            all_sum.backward()
            all_sum_cpu.backward()
            self.assertEqual(all_sum, all_sum_cpu)
            self.assertEqual(x.grad, cpu_x.grad)

        helper(3, 3)

    def test_nll_loss_1d(self, device='cpu'):
        self._nll_loss_1d_helper([10], "none")
        self._nll_loss_1d_helper([10], "mean")
        self._nll_loss_1d_helper([10], "sum")

    def test_nll_loss_empty_tensor_reduction_none(self, device='cpu'):
        self._nll_loss_helper([1, 3], "none", torch.empty([0], device=device))
        self._nll_loss_helper([3, 5, 7], "none", torch.empty([5, 7], device=device))
        self._nll_loss_helper([2, 3, 1, 7], "none", torch.empty([2, 1, 7], device=device))
        self._nll_loss_helper([2, 3, 5, 1], "none", torch.empty([2, 5, 1], device=device))
        self._nll_loss_helper([2, 3, 5, 7, 1], "none", torch.empty([2, 5, 7, 1], device=device))

    @unittest.skipIf(TEST_WITH_UBSAN, "division-by-zero error with UBSAN")
    def test_nll_loss_empty_tensor_reduction_mean(self, device='cpu'):
        nan = torch.tensor(float('nan'), device=device)
        self._nll_loss_helper([1, 3], "mean", nan)
        self._nll_loss_helper([1, 3, 5, 7], "mean", nan)
        self._nll_loss_helper([2, 3, 1, 7], "mean", nan)
        self._nll_loss_helper([2, 3, 5, 1], "mean", nan)
        self._nll_loss_helper([2, 3, 5, 7, 1], "mean", nan)

    def test_nll_loss_empty_tensor_reduction_sum(self, device='cpu'):
        zero = torch.tensor(0, device=device)
        self._nll_loss_helper([1, 3], "sum", zero)
        self._nll_loss_helper([1, 3, 5, 7], "sum", zero)
        self._nll_loss_helper([2, 3, 1, 7], "sum", zero)
        self._nll_loss_helper([2, 3, 5, 1], "sum", zero)
        self._nll_loss_helper([2, 3, 5, 7, 1], "sum", zero)

    def test_nll_loss_byte_target_matches_long(self, device='cpu'):
        N, C = 10, 4
        input = torch.randn(N, C, device=device, requires_grad=True)
        target = torch.empty(N, dtype=torch.long, device=device).random_(0, C)

        def compute_result_and_gradient(reduction, target_dtype):
            result, grad = {}, {}
            for dev in ['cpu', 'mps']:
                input_dev = input.to(dev)
                input_ = input_dev.detach()
                input_.requires_grad_()

                target_dev = target.to(dev)

                prob = F.log_softmax(input_, dim=-1)
                loss = nn.NLLLoss(reduction=reduction)
                result[dev] = loss(prob, target_dev.to(target_dtype))
                result[dev].sum().backward()
                grad[dev] = input_.grad

            return result, grad

        for reduction in ["none", "mean", "sum"]:
            result_long, grad_long = compute_result_and_gradient(reduction, torch.long)
            result_byte, grad_byte = compute_result_and_gradient(reduction, torch.uint8)

            self.assertEqual(result_long['mps'].to('cpu'), result_long['cpu'])
            self.assertEqual(grad_long['mps'].to('cpu'), grad_long['cpu'])

    # L1 loss
    def test_l1_loss(self):
        def helper(shape, reduction):
            # create the criterion
            loss = torch.nn.L1Loss(reduction=reduction)

            inputCPU = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            targetCPU = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            inputMPS = inputCPU.detach().clone().to('mps').requires_grad_()
            targetMPS = targetCPU.detach().clone().to('mps')

            # forward pass
            outputCPU = loss(inputCPU, targetCPU)
            outputMPS = loss(inputMPS, targetMPS)
            self.assertEqual(outputCPU, outputMPS)

            # backward pass
            if reduction != 'none':
                # chose 2 just to make the grad_output > 1 in backward pass
                outputCPU.backward(gradient=torch.full_like(outputCPU, 2))
                outputMPS.backward(gradient=torch.full_like(outputMPS, 2))
                self.assertEqual(inputCPU.grad, inputMPS.grad)

        helper([8, 5, 4], 'none')
        helper([7, 5, 2, 4], 'sum')
        # verify if changes in shape would cause cached graph lookup problems
        helper([7, 5, 2, 4, 6], 'sum')
        helper([8, 4, 5, 7, 6], 'mean')

    # Mean Squared Error
    def test_mse_loss(self):
        def helper(shape, reduction):
            # create the criterion
            loss = torch.nn.MSELoss(reduction=reduction)

            inputCPU = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            targetCPU = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            inputMPS = inputCPU.detach().clone().to('mps').requires_grad_()
            targetMPS = targetCPU.detach().clone().to('mps')

            # forward pass
            outputCPU = loss(inputCPU, targetCPU)
            outputMPS = loss(inputMPS, targetMPS)
            self.assertEqual(outputCPU, outputMPS)

            # backward pass
            if reduction != 'none':
                # chose 2 just to make the grad_output > 1 in backward pass
                outputCPU.backward(gradient=torch.full_like(outputCPU, 2))
                outputMPS.backward(gradient=torch.full_like(outputMPS, 2))
                self.assertEqual(inputCPU.grad, inputMPS.grad)

        helper([8, 5, 4], 'none')
        helper([7, 5, 2, 4], 'sum')
        # verify if changes in shape would cause cached graph lookup problems
        helper([7, 5, 2, 4, 6], 'sum')
        helper([8, 4, 5, 7, 6], 'mean')

    # Binary Cross Enropy
    def test_bce_loss_simple(self):
        def helper(shape, reduction):
            # create the criterion
            loss = torch.nn.BCELoss(reduction=reduction)

            # input and target must be within [0..1]
            input_t = np.random.random_sample(size=shape).astype(np.float32)
            target_t = np.random.random_sample(size=shape).astype(np.float32)
            inputCPU = torch.tensor(input_t, device='cpu', dtype=torch.float, requires_grad=True)
            targetCPU = torch.tensor(target_t, device='cpu', dtype=torch.float, requires_grad=False)
            inputMPS = inputCPU.detach().clone().to('mps').requires_grad_()
            targetMPS = targetCPU.detach().clone().to('mps')

            # forward pass
            outputCPU = loss(inputCPU, targetCPU)
            outputMPS = loss(inputMPS, targetMPS)
            self.assertEqual(outputCPU, outputMPS)

            # backward pass
            if reduction != 'none':
                # chose 0.6 just to have the grad_output != 1
                outputCPU.backward(gradient=torch.full_like(outputCPU, 0.6))
                outputMPS.backward(gradient=torch.full_like(outputMPS, 0.6))
                self.assertEqual(inputCPU.grad, inputMPS.grad)

        helper([8, 5, 4], 'none')
        helper([7, 5, 2, 4], 'sum')
        # verify if changes in shape would cause cached graph lookup problems
        helper([7, 5, 2, 4, 6], 'sum')
        helper([8, 4, 5, 7, 6], 'mean')
        helper([1, 1, 32, 32], 'mean')

    def test_bce_loss_always_nonnegative(self):
        target = torch.ones(5, device='mps')
        input = torch.ones(5, device='mps')
        self.assertEqual((nn.BCELoss()(input, target) < 0).sum(), 0)

        target = torch.zeros(5, device='mps')
        input = torch.zeros(5, device='mps')
        self.assertEqual((nn.BCELoss()(input, target) < 0).sum(), 0)

    def test_bce_loss_size_mismatch(self):
        bceloss = nn.BCELoss()
        a = torch.rand(25, device='mps')
        b = torch.rand(25, 1, device='mps')
        with self.assertRaisesRegex(ValueError, r'Using a target size \('):
            bceloss(a, b)

    def test_bce_with_logits_gives_same_result_as_sigmoid_and_bce_loss_large_tensors_with_grad(self):
        x_size = 1024
        y_size = 256
        target = torch.rand(x_size, y_size, device='mps')

        for reduction in ['none', 'mean', 'sum']:
            output_sig = torch.rand(x_size, y_size, device='mps') - 0.5
            output_logits = output_sig.clone().detach()

            output_sig.requires_grad = True
            output_logits.requires_grad = True
            weight = torch.rand(y_size, device='mps')

            loss_sig = nn.BCELoss(weight, reduction=reduction)(
                torch.sigmoid(output_sig), target
            )
            loss_logits = nn.BCEWithLogitsLoss(weight, reduction=reduction)(
                output_logits, target
            )

            self.assertEqual(loss_logits, loss_sig)

            if reduction == 'none':
                grad = torch.rand(x_size, y_size, device='mps')
                loss_sig.backward(grad)
                loss_logits.backward(grad)
            else:
                loss_sig.backward()
                loss_logits.backward()

            self.assertEqual(output_sig.grad, output_logits.grad)

    def test_bce_with_logits_has_correct_grad_at_zero(self):
        output = torch.zeros(3, 1, requires_grad=True, device='mps')
        target = torch.zeros(3, 1, device='mps')
        nn.BCEWithLogitsLoss(reduction='sum')(output, target).backward()
        expected_grad = torch.empty(3, 1, device='mps').fill_(0.5)
        self.assertEqual(output.grad, expected_grad)

    def test_bce_with_logits_broadcasts_weights(self):
        target = torch.rand(16, 4, device='mps')
        output = torch.rand(16, 4, device='mps') - 0.5

        weight = torch.rand(4, device='mps')
        out1 = nn.BCEWithLogitsLoss(weight)(output, target)

        weight = weight.expand(16, 4).contiguous()
        out2 = nn.BCEWithLogitsLoss(weight)(output, target)

        self.assertEqual(out1, out2)

        weight = torch.rand(16, 1, device='mps')
        out1 = nn.BCEWithLogitsLoss(weight)(output, target)

        weight = weight.expand(16, 4).contiguous()
        out2 = nn.BCEWithLogitsLoss(weight)(output, target)

        self.assertEqual(out1, out2)

    def test_bce_with_logits_ones_in_pos_weights_are_the_same_as_none(self):
        target = torch.rand(64, 4, device='mps')
        output = torch.rand(64, 4, device='mps') - 0.5
        pos_weight = torch.ones(64, 4, device='mps')

        self.assertEqual(nn.BCEWithLogitsLoss()(output, target),
                         nn.BCEWithLogitsLoss(pos_weight=pos_weight)(output, target))

    def test_bce_with_logits_broadcasts_pos_weights(self):
        target = torch.rand(64, 4, device='mps')
        output = torch.rand(64, 4, device='mps') - 0.5
        pos_weight = torch.rand(4, device='mps')
        out1 = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(output, target)

        pos_weight1 = pos_weight.expand(1, 4)
        out2 = nn.BCEWithLogitsLoss(pos_weight=pos_weight1)(output, target)

        pos_weight2 = pos_weight.expand(64, 4)
        out3 = nn.BCEWithLogitsLoss(pos_weight=pos_weight2)(output, target)

        self.assertEqual(out1, out2)
        self.assertEqual(out1, out3)

    def test_bce_with_logits_with_pos_weight_has_correct_grad_at_zero(self):
        output = torch.zeros(3, 1, requires_grad=True, device='mps')
        target = torch.zeros(3, 1, device='mps')
        pos_weight = torch.ones(3, 1, device='mps')
        nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='sum')(output, target).backward()
        expected_grad = torch.empty(3, 1, device='mps').fill_(0.5)
        grad = output.grad
        self.assertEqual(grad, expected_grad)

    def test_bce_with_logits_stability(self):
        output = torch.tensor([0., -120.], device='mps')
        target = torch.tensor([0., 1.], device='mps')
        pos_weight = torch.tensor([1., 1.], device='mps')

        out1 = nn.BCEWithLogitsLoss()(output, target)
        self.assertTrue(torch.isfinite(out1).all().item())

        out2 = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(output, target)
        self.assertTrue(torch.isfinite(out2).all().item())

    def test_bce_loss_broadcasts_weights(self):
        sigmoid = nn.Sigmoid()
        target = torch.rand(16, 4, device='mps')
        output = torch.rand(16, 4, device='mps') - 0.5

        weight = torch.rand(4, device='mps')
        out1 = nn.BCELoss(weight)(sigmoid(output), target)

        weight = weight.expand(16, 4).contiguous()
        out2 = nn.BCELoss(weight)(sigmoid(output), target)

        self.assertEqual(out1, out2)

        weight = torch.rand(16, 1, device='mps')
        out1 = nn.BCELoss(weight)(sigmoid(output), target)

        weight = weight.expand(16, 4).contiguous()
        out2 = nn.BCELoss(weight)(sigmoid(output), target)

        self.assertEqual(out1, out2)

    def test_log_softmax(self):
        values = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]
        cpu_x = torch.tensor(values, device='cpu', requires_grad=True)
        mps_x = torch.tensor(values, device='mps', requires_grad=True)

        cpu_log_softmax = F.log_softmax(cpu_x, dim=0)
        mps_log_softmax = F.log_softmax(mps_x, dim=0)
        self.assertEqual(cpu_log_softmax, mps_log_softmax.to('cpu'))

        cpu_grad = torch.ones_like(cpu_log_softmax)
        mps_grad = torch.ones_like(cpu_log_softmax).to('mps')

        cpu_log_softmax.backward(gradient=cpu_grad)
        mps_log_softmax.backward(gradient=mps_grad)

        self.assertEqual(cpu_x.grad, mps_x.grad.to('cpu'))

    def test_eq(self):
        values1 = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]
        values2 = [[[1.0, 2.0, 15.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [0.0, 11.0, 12.0]]]
        mps_x = torch.tensor(values1, device='mps')
        mps_y = torch.tensor(values2, device='mps')
        cpu_x = torch.tensor(values1, device='cpu')
        cpu_y = torch.tensor(values2, device='cpu')
        result_mps = torch.eq(mps_x, mps_y)
        result_cpu = torch.eq(cpu_x, cpu_y)

        self.assertEqual(result_cpu, result_mps.to('cpu'))

    def test_eq_int64(self):
        values1 = [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]
        values2 = [[[1, 2, 15], [4, 5, 6]], [[7, 8, 9], [0, 11, 12]]]
        mps_x = torch.tensor(values1, device='mps')
        mps_y = torch.tensor(values2, device='mps')
        cpu_x = torch.tensor(values1, device='cpu')
        cpu_y = torch.tensor(values2, device='cpu')
        result_mps = torch.eq(mps_x, mps_y)
        result_cpu = torch.eq(cpu_x, cpu_y)

        self.assertEqual(result_cpu, result_mps.to('cpu'))

    def test_ne(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float)
            cpu_y = torch.randn(shape, device='cpu', dtype=torch.float)
            mps_x = cpu_x.detach().clone().to('mps')
            mps_y = cpu_y.detach().clone().to('mps')
            result_mps = torch.ne(mps_x, mps_y)
            result_cpu = torch.ne(cpu_x, cpu_y)

            self.assertEqual(result_cpu, result_mps.to('cpu'))

        helper((2, 3, 4, 5))

    def test_ne_scalar(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float)
            mps_x = cpu_x.detach().clone().to('mps')
            result_mps = torch.ne(mps_x, 0.0)
            result_cpu = torch.ne(cpu_x, 0.0)

            self.assertEqual(result_cpu, result_mps.to('cpu'))

        helper((2, 3, 4, 5))

    def test_lt(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float)
            cpu_y = torch.randn(shape, device='cpu', dtype=torch.float)
            mps_x = cpu_x.detach().clone().to('mps')
            mps_y = cpu_y.detach().clone().to('mps')
            result_mps = torch.lt(mps_x, mps_y)
            result_cpu = torch.lt(cpu_x, cpu_y)

            self.assertEqual(result_cpu, result_mps.to('cpu'))

        helper((2, 3, 4, 5))

    def test_lt_scalar(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float)
            mps_x = cpu_x.detach().clone().to('mps')
            result_mps = torch.lt(mps_x, 0.0)
            result_cpu = torch.lt(cpu_x, 0.0)

            self.assertEqual(result_cpu, result_mps.to('cpu'))

        helper((2, 3, 4, 5))

    def test_le(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float)
            cpu_y = torch.randn(shape, device='cpu', dtype=torch.float)
            mps_x = cpu_x.detach().clone().to('mps')
            mps_y = cpu_y.detach().clone().to('mps')
            result_mps = torch.le(mps_x, mps_y)
            result_cpu = torch.le(cpu_x, cpu_y)

            self.assertEqual(result_cpu, result_mps.to('cpu'))

        helper((2, 3, 4, 5))

    def test_le_scalar(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float)
            mps_x = cpu_x.detach().clone().to('mps')
            result_mps = torch.le(mps_x, 0.0)
            result_cpu = torch.le(cpu_x, 0.0)

            self.assertEqual(result_cpu, result_mps.to('cpu'))

        helper((2, 3, 4, 5))

    def test_ge(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float)
            cpu_y = torch.randn(shape, device='cpu', dtype=torch.float)
            mps_x = cpu_x.detach().clone().to('mps')
            mps_y = cpu_y.detach().clone().to('mps')
            result_mps = torch.ge(mps_x, mps_y)
            result_cpu = torch.ge(cpu_x, cpu_y)

            self.assertEqual(result_cpu, result_mps.to('cpu'))

        helper((2, 3, 4, 5))

    def test_ge_scalar(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float)
            mps_x = cpu_x.detach().clone().to('mps')
            result_mps = torch.ge(mps_x, 0.0)
            result_cpu = torch.ge(cpu_x, 0.0)

            self.assertEqual(result_cpu, result_mps.to('cpu'))

        helper((2, 3, 4, 5))

    def test_gt(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float)
            cpu_y = torch.randn(shape, device='cpu', dtype=torch.float)
            mps_x = cpu_x.detach().clone().to('mps')
            mps_y = cpu_y.detach().clone().to('mps')
            result_mps = torch.gt(mps_x, mps_y)
            result_cpu = torch.gt(cpu_x, cpu_y)

            self.assertEqual(result_cpu, result_mps.to('cpu'))

        helper((2, 3, 4, 5))

    def test_gt_scalar(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float)
            mps_x = cpu_x.detach().clone().to('mps')
            result_mps = torch.gt(mps_x, 0.0)
            result_cpu = torch.gt(cpu_x, 0.0)

            self.assertEqual(result_cpu, result_mps.to('cpu'))

        helper((2, 3, 4, 5))

    # Test forward argmin argmax
    def test_argmin_argmax(self):
        def helper(n, c, h, w, reduction_type, dtype=torch.float32):
            if reduction_type == "max":
                arg_reduction_fn = torch.argmax
            else:
                arg_reduction_fn = torch.argmin

            cpu_x = None
            x = None
            if(dtype not in [torch.float32, torch.bool]):
                cpu_x = torch.randint(50, (n, c, h, w), device='cpu', dtype=dtype, requires_grad=False)
                x = cpu_x.detach().clone().to('mps')
            elif (dtype == torch.bool):
                cpu_x = torch.randint(2, (n, c, h, w), device='cpu', dtype=dtype, requires_grad=False)
                x = cpu_x.detach().clone().to('mps')
            else:
                cpu_x = torch.randn(n, c, h, w, device='cpu', dtype=dtype, requires_grad=True)
                x = cpu_x.detach().clone().to('mps').requires_grad_()

            y = arg_reduction_fn(x)
            ref_y = arg_reduction_fn(cpu_x)
            self.assertEqual(y, ref_y)

            y_0 = arg_reduction_fn(x, dim=0)
            refy_0 = arg_reduction_fn(cpu_x, dim=0)
            self.assertEqual(y_0, refy_0)

            y_0dim = arg_reduction_fn(x, dim=0, keepdim=True)
            refy_0dim = arg_reduction_fn(cpu_x, dim=0, keepdim=True)
            self.assertEqual(y_0dim, refy_0dim)

            y_1 = arg_reduction_fn(x, dim=1)
            refy_1 = arg_reduction_fn(cpu_x, dim=1)
            self.assertEqual(y_1, refy_1)

            y_1dim = arg_reduction_fn(x, dim=1, keepdim=True)
            refy_1dim = arg_reduction_fn(cpu_x, dim=1, keepdim=True)
            self.assertEqual(y_1dim, refy_1dim)

            y_2 = arg_reduction_fn(x, dim=2)
            refy_2 = arg_reduction_fn(cpu_x, dim=2)
            self.assertEqual(y_2, refy_2)

            y_2dim = arg_reduction_fn(x, dim=2, keepdim=True)
            refy_2dim = arg_reduction_fn(cpu_x, dim=2, keepdim=True)
            self.assertEqual(y_2dim, refy_2dim)

            y_3 = arg_reduction_fn(x, dim=3)
            refy_3 = arg_reduction_fn(cpu_x, dim=3)
            self.assertEqual(y_3, refy_3)

            y_3dim = arg_reduction_fn(x, dim=3, keepdim=True)
            refy_3dim = arg_reduction_fn(cpu_x, dim=3, keepdim=True)
            self.assertEqual(y_3dim, refy_3dim)

        helper(2, 8, 4, 4, "max", torch.float32)
        helper(2, 8, 4, 4, "max", torch.int32)
        helper(2, 8, 4, 4, "max", torch.float16)
        helper(2, 8, 4, 4, "max", torch.int64)
        helper(2, 8, 4, 4, "min", torch.float32)
        helper(2, 8, 4, 4, "min", torch.int32)
        helper(2, 8, 4, 4, "min", torch.float16)
        helper(2, 8, 4, 4, "min", torch.int64)

    # Test forward max
    # Note - don't test grad now
    def test_max_el(self):
        def helper(n, c, h, w, dtype=torch.float32):

            if(dtype not in [torch.float32, torch.bool]):
                cpu_x = torch.randint(50, (n, c, h, w), device='cpu', dtype=dtype, requires_grad=False)
                x = cpu_x.detach().clone().to('mps')
            elif (dtype == torch.bool):
                cpu_x = torch.randint(2, (n, c, h, w), device='cpu', dtype=dtype, requires_grad=False)
                x = cpu_x.detach().clone().to('mps')
            else:
                cpu_x = torch.randn(n, c, h, w, device='cpu', dtype=dtype, requires_grad=True)
                x = cpu_x.detach().clone().to('mps')

            ref_y = torch.max(cpu_x)
            y = torch.max(x)
            self.assertEqual(y, ref_y)

            for dim in [0, 1, 2, 3]:
                for keepdim in [True, False]:
                    y, idx = torch.max(x, dim=dim, keepdim=keepdim)
                    refy, refidx = torch.max(cpu_x, dim=dim, keepdim=keepdim)
                    self.assertEqual(y, refy)
                    self.assertEqual(idx, refidx)

            y_0 = torch.ones(c, h, w, device='mps', dtype=dtype)
            idx_0 = torch.ones(c, h, w, device='mps', dtype=torch.int64)
            torch.max(x, dim=0, out=(y_0, idx_0))
            refy_0, refidx_0 = torch.max(cpu_x, dim=0)
            self.assertEqual(y_0, refy_0)
            self.assertEqual(idx_0, refidx_0)

            y_0dim = torch.ones(1, c, h, w, device='mps', dtype=dtype)
            idx_0dim = torch.ones(1, c, h, w, device='mps', dtype=torch.int64)
            torch.max(x, dim=0, keepdim=True, out=(y_0dim, idx_0dim))
            refy_0dim, refidx_0dim = torch.max(cpu_x, dim=0, keepdim=True)
            self.assertEqual(y_0dim, refy_0dim)
            self.assertEqual(idx_0dim, refidx_0dim)

            y_1 = torch.ones(n, h, w, device='mps', dtype=dtype)
            idx_1 = torch.ones(n, h, w, device='mps', dtype=torch.int64)
            torch.max(x, dim=1, out=(y_1, idx_1))
            refy_1, refidx_1 = torch.max(cpu_x, dim=1)
            self.assertEqual(y_1, refy_1)
            self.assertEqual(idx_1, refidx_1)

            y_1dim = torch.ones(n, 1, h, w, device='mps', dtype=dtype)
            idx_1dim = torch.ones(n, 1, h, w, device='mps', dtype=torch.int64)
            torch.max(x, dim=1, keepdim=True, out=(y_1dim, idx_1dim))
            refy_1dim, refidx_1dim = torch.max(cpu_x, keepdim=True, dim=1)
            self.assertEqual(y_1dim, refy_1dim)
            self.assertEqual(idx_1dim, refidx_1dim)

            y_2 = torch.ones(n, c, w, device='mps', dtype=dtype)
            idx_2 = torch.ones(n, c, w, device='mps', dtype=torch.int64)
            torch.max(x, dim=2, out=(y_2, idx_2))
            refy_2, refidx_2 = torch.max(cpu_x, dim=2)
            self.assertEqual(y_2, refy_2)
            self.assertEqual(idx_2, refidx_2)

            y_2dim = torch.ones(n, c, 1, w, device='mps', dtype=dtype)
            idx_2dim = torch.ones(n, c, 1, w, device='mps', dtype=torch.int64)
            torch.max(x, dim=2, keepdim=True, out=(y_2dim, idx_2dim))
            refy_2dim, refidx_2dim = torch.max(cpu_x, dim=2, keepdim=True,)
            self.assertEqual(y_2dim, refy_2dim)
            self.assertEqual(idx_2dim, refidx_2dim)

            y_3 = torch.ones(n, c, h, device='mps', dtype=dtype)
            idx_3 = torch.ones(n, c, h, device='mps', dtype=torch.int64)
            torch.max(x, dim=3, out=(y_3, idx_3))
            refy_3, refidx_3 = torch.max(cpu_x, dim=3)
            self.assertEqual(y_3, refy_3)
            self.assertEqual(idx_3, refidx_3)

            y_3dim = torch.ones(n, c, h, 1, device='mps', dtype=dtype)
            idx_3dim = torch.ones(n, c, h, 1, device='mps', dtype=torch.int64)
            torch.max(x, dim=3, keepdim=True, out=(y_3dim, idx_3dim))
            refy_3dim, refidx_3dim = torch.max(cpu_x, dim=3, keepdim=True,)
            self.assertEqual(y_3dim, refy_3dim)
            self.assertEqual(idx_3dim, refidx_3dim)

        helper(2, 8, 4, 5, torch.float32)
        helper(2, 8, 4, 5, torch.int32)
        # helper(2, 8, 4, 5, torch.int64)
        
    def test_median(self):
        def helper_dtype_int32(n1, n2, n3):
            cpu_x = torch.randint(50, (n1, n2, n3), device='cpu', dtype=torch.int32)
            mps_x = cpu_x.detach().clone().to('mps')

            result_cpu = torch.median(cpu_x)
            result_mps = torch.median(mps_x)

            self.assertEqual(result_cpu, result_mps)
        def helper_dtype_float32(n1, n2, n3):
            cpu_x = torch.randn(n1, n2, n3, device='cpu', dtype=torch.float32)
            mps_x = cpu_x.detach().clone().to('mps')

            result_cpu = torch.median(cpu_x)
            result_mps = torch.median(mps_x)

            self.assertEqual(result_cpu, result_mps)

        helper_dtype_int32(3, 3, 3)
        helper_dtype_int32(2, 2, 2)
        helper_dtype_float32(2, 2, 2)
        helper_dtype_float32(3, 3, 3)
        
    def test_any(self):
        def helper(shape):
            input_xs = []
            prod = 1

            for i in range(len(shape)):
                prod *= shape[i]
            input_xs.append(torch.randn(prod, dtype=torch.float).reshape(shape))
            input_xs.append(torch.arange(0, prod, dtype=torch.float).reshape(shape))
            input_xs.append(torch.ones(prod, dtype=torch.float).reshape(shape))
            input_xs.append(torch.zeros(prod, dtype=torch.float).reshape(shape))
            input_xs.append(torch.arange(0, prod, dtype=torch.int).reshape(shape))
            input_xs.append(torch.ones(prod, dtype=torch.int).reshape(shape))
            input_xs.append(torch.zeros(prod, dtype=torch.int).reshape(shape))
            input_xs.append(torch.arange(0, prod, dtype=torch.int).reshape(shape).bool())
            input_xs.append(torch.ones(prod, dtype=torch.int).reshape(shape).bool())
            input_xs.append(torch.zeros(prod, dtype=torch.int).reshape(shape).bool())

            for i, cpu_x in enumerate(input_xs):
                x = cpu_x.detach().clone().to('mps')
                y = torch.any(x)
                ref_y = torch.any(cpu_x)
                self.assertEqual(y, ref_y)

                y_0 = torch.any(x, dim=0)
                refy_0 = torch.any(cpu_x, dim=0)
                self.assertEqual(y_0, refy_0)

                y_0dim = torch.any(x, dim=0, keepdim=True)
                refy_0dim = torch.any(cpu_x, dim=0, keepdim=True)
                self.assertEqual(y_0dim, refy_0dim)

                y_0dim = torch.any(x, dim=0, keepdim=True)
                refy_0dim = torch.any(cpu_x, dim=0, keepdim=True)
                self.assertEqual(y_0dim, refy_0dim)

                y_1 = torch.any(x, dim=1)
                refy_1 = torch.any(cpu_x, dim=1)
                self.assertEqual(y_1, refy_1)

                y_1dim = torch.any(x, dim=1, keepdim=True)
                refy_1dim = torch.any(cpu_x, dim=1, keepdim=True)
                self.assertEqual(y_1dim, refy_1dim)

                if (len(shape) > 2):
                    y_2 = torch.any(x, dim=2)
                    refy_2 = torch.any(cpu_x, dim=2)
                    self.assertEqual(y_2, refy_2)

                    y_2dim = torch.any(x, dim=2, keepdim=True)
                    refy_2dim = torch.any(cpu_x, dim=2, keepdim=True)
                    self.assertEqual(y_2dim, refy_2dim)

                    y_3 = torch.any(x, dim=3)
                    refy_3 = torch.any(cpu_x, dim=3)
                    self.assertEqual(y_3, refy_3)

                    y_3dim = torch.any(x, dim=3, keepdim=True)
                    refy_3dim = torch.any(cpu_x, dim=3, keepdim=True)
                    self.assertEqual(y_3dim, refy_3dim)
        helper((1, 1, 1, 1))
        helper((1, 1, 3, 3))
        helper((7, 13))
        helper((2, 8, 4, 5))

    def test_all(self):
        def helper(shape):
            input_xs = []
            prod = 1

            for i in range(len(shape)):
                prod *= shape[i]
            input_xs.append(torch.randn(prod, dtype=torch.float).reshape(shape))
            input_xs.append(torch.arange(0, prod, dtype=torch.float).reshape(shape))
            input_xs.append(torch.ones(prod, dtype=torch.float).reshape(shape))
            input_xs.append(torch.zeros(prod, dtype=torch.float).reshape(shape))
            input_xs.append(torch.arange(0, prod, dtype=torch.int).reshape(shape))
            input_xs.append(torch.ones(prod, dtype=torch.int).reshape(shape))
            input_xs.append(torch.zeros(prod, dtype=torch.int).reshape(shape))
            input_xs.append(torch.arange(0, prod, dtype=torch.int).reshape(shape).bool())
            input_xs.append(torch.ones(prod, dtype=torch.int).reshape(shape).bool())
            input_xs.append(torch.zeros(prod, dtype=torch.int).reshape(shape).bool())

            for i, cpu_x in enumerate(input_xs):
                x = cpu_x.detach().clone().to('mps')
                y = torch.all(x)
                ref_y = torch.all(cpu_x)
                self.assertEqual(y, ref_y)

                y_0 = torch.all(x, dim=0)
                refy_0 = torch.all(cpu_x, dim=0)
                self.assertEqual(y_0, refy_0)

                y_0dim = torch.all(x, dim=0, keepdim=True)
                refy_0dim = torch.all(cpu_x, dim=0, keepdim=True)
                self.assertEqual(y_0dim, refy_0dim)

                y_0dim = torch.all(x, dim=0, keepdim=True)
                refy_0dim = torch.all(cpu_x, dim=0, keepdim=True)
                self.assertEqual(y_0dim, refy_0dim)

                y_1 = torch.all(x, dim=1)
                refy_1 = torch.all(cpu_x, dim=1)
                self.assertEqual(y_1, refy_1)

                y_1dim = torch.all(x, dim=1, keepdim=True)
                refy_1dim = torch.all(cpu_x, dim=1, keepdim=True)
                self.assertEqual(y_1dim, refy_1dim)
                if (len(shape) > 2):
                    y_2 = torch.all(x, dim=2)
                    refy_2 = torch.all(cpu_x, dim=2)
                    self.assertEqual(y_2, refy_2)

                    y_2dim = torch.all(x, dim=2, keepdim=True)
                    refy_2dim = torch.all(cpu_x, dim=2, keepdim=True)
                    self.assertEqual(y_2dim, refy_2dim)

                    y_3 = torch.all(x, dim=3)
                    refy_3 = torch.all(cpu_x, dim=3)
                    self.assertEqual(y_3, refy_3)

                    y_3dim = torch.all(x, dim=3, keepdim=True)
                    refy_3dim = torch.all(cpu_x, dim=3, keepdim=True)
                    self.assertEqual(y_3dim, refy_3dim)

        helper((1, 1, 1, 1))
        helper((1, 1, 3, 3))
        helper((7, 13))
        helper((2, 8, 4, 5))

    # Test forward min
    def test_min_el(self):
        def helper(n, c, h, w):
            cpu_x = torch.randn(n, c, h, w, device='cpu', dtype=torch.float, requires_grad=False)
            x = cpu_x.detach().clone().to('mps')

            y = torch.min(x)
            ref_y = torch.min(cpu_x)
            self.assertEqual(y, ref_y)

            y_0, idx_0 = torch.min(x, dim=0)
            refy_0, refidx_0 = torch.min(cpu_x, dim=0)
            self.assertEqual(y_0, refy_0)
            self.assertEqual(idx_0, refidx_0)

            y_0 = torch.ones(c, h, w, device='mps', dtype=torch.float)
            idx_0 = torch.ones(c, h, w, device='mps', dtype=torch.int64)
            torch.min(x, dim=0, out=(y_0, idx_0))
            refy_0, refidx_0 = torch.min(cpu_x, dim=0)
            self.assertEqual(y_0, refy_0)
            self.assertEqual(idx_0, refidx_0)

            y_0dim, idx_0dim = torch.min(x, dim=0, keepdim=True)
            refy_0dim, refidx_0dim = torch.min(cpu_x, dim=0, keepdim=True)
            self.assertEqual(y_0dim, refy_0dim)
            self.assertEqual(idx_0dim, refidx_0dim)

            y_0dim = torch.ones(1, c, h, w, device='mps', dtype=torch.float)
            idx_0dim = torch.ones(1, c, h, w, device='mps', dtype=torch.int64)
            torch.min(x, dim=0, keepdim=True, out=(y_0dim, idx_0dim))
            refy_0dim, refidx_0dim = torch.min(cpu_x, dim=0, keepdim=True)
            self.assertEqual(y_0dim, refy_0dim)
            self.assertEqual(idx_0dim, refidx_0dim)

            y_1, idx_1 = torch.min(x, dim=1)
            refy_1, refidx_1 = torch.min(cpu_x, dim=1)
            self.assertEqual(y_1, refy_1)
            self.assertEqual(idx_1, refidx_1)

            y_1 = torch.ones(n, h, w, device='mps', dtype=torch.float)
            idx_1 = torch.ones(n, h, w, device='mps', dtype=torch.int64)
            torch.min(x, dim=1, out=(y_1, idx_1))
            refy_1, refidx_1 = torch.min(cpu_x, dim=1)
            self.assertEqual(y_1, refy_1)
            self.assertEqual(idx_1, refidx_1)

            y_1dim, idx_1dim = torch.min(x, dim=1, keepdim=True)
            refy_1dim, refidx_1dim = torch.min(cpu_x, dim=1, keepdim=True)
            self.assertEqual(y_1dim, refy_1dim)
            self.assertEqual(idx_1dim, refidx_1dim)

            y_1dim = torch.ones(n, 1, h, w, device='mps', dtype=torch.float)
            idx_1dim = torch.ones(n, 1, h, w, device='mps', dtype=torch.int64)
            torch.min(x, dim=1, keepdim=True, out=(y_1dim, idx_1dim))
            refy_1dim, refidx_1dim = torch.min(cpu_x, keepdim=True, dim=1)
            self.assertEqual(y_1dim, refy_1dim)
            self.assertEqual(idx_1dim, refidx_1dim)

            y_2, idx_2 = torch.min(x, dim=2)
            refy_2, refidx_2 = torch.min(cpu_x, dim=2)
            self.assertEqual(y_2, refy_2)
            self.assertEqual(idx_2, refidx_2)

            y_2 = torch.ones(n, c, w, device='mps', dtype=torch.float)
            idx_2 = torch.ones(n, c, w, device='mps', dtype=torch.int64)
            torch.min(x, dim=2, out=(y_2, idx_2))
            refy_2, refidx_2 = torch.min(cpu_x, dim=2)
            self.assertEqual(y_2, refy_2)
            self.assertEqual(idx_2, refidx_2)

            y_2dim, idx_2dim = torch.min(x, dim=2, keepdim=True)
            refy_2dim, refidx_2dim = torch.min(cpu_x, dim=2, keepdim=True)
            self.assertEqual(y_2dim, refy_2dim)
            self.assertEqual(idx_2dim, refidx_2dim)

            y_2dim = torch.ones(n, c, 1, w, device='mps', dtype=torch.float)
            idx_2dim = torch.ones(n, c, 1, w, device='mps', dtype=torch.int64)
            torch.min(x, dim=2, keepdim=True, out=(y_2dim, idx_2dim))
            refy_2dim, refidx_2dim = torch.min(cpu_x, dim=2, keepdim=True,)
            self.assertEqual(y_2dim, refy_2dim)
            self.assertEqual(idx_2dim, refidx_2dim)

            y_3, idx_3 = torch.min(x, dim=3)
            refy_3, refidx_3 = torch.min(cpu_x, dim=3)
            self.assertEqual(y_3, refy_3)
            self.assertEqual(idx_3, refidx_3)

            y_3 = torch.ones(n, c, h, device='mps', dtype=torch.float)
            idx_3 = torch.ones(n, c, h, device='mps', dtype=torch.int64)
            torch.min(x, dim=3, out=(y_3, idx_3))
            refy_3, refidx_3 = torch.min(cpu_x, dim=3)
            self.assertEqual(y_3, refy_3)
            self.assertEqual(idx_3, refidx_3)

            y_3dim, idx_3dim = torch.min(x, dim=3, keepdim=True)
            refy_3dim, refidx_3dim = torch.min(cpu_x, dim=3, keepdim=True)
            self.assertEqual(y_3dim, refy_3dim)
            self.assertEqual(idx_3dim, refidx_3dim)

            y_3dim = torch.ones(n, c, h, 1, device='mps', dtype=torch.float)
            idx_3dim = torch.ones(n, c, h, 1, device='mps', dtype=torch.int64)
            torch.min(x, dim=3, keepdim=True, out=(y_3dim, idx_3dim))
            refy_3dim, refidx_3dim = torch.min(cpu_x, dim=3, keepdim=True,)
            self.assertEqual(y_3dim, refy_3dim)
            self.assertEqual(idx_3dim, refidx_3dim)

        helper(2, 8, 4, 5)

    # Test forward sum
    def test_sum(self):
        def helper(n, c, h, w, dtype=torch.float32):
            cpu_x = None
            x = None
            if(dtype not in [torch.float32, torch.bool]):
                cpu_x = torch.randint(50, (n, c, h, w), device='cpu', dtype=dtype, requires_grad=False)
                x = cpu_x.detach().clone().to('mps')
            elif (dtype == torch.bool):
                cpu_x = torch.randint(2, (n, c, h, w), device='cpu', dtype=dtype, requires_grad=False)
                x = cpu_x.detach().clone().to('mps')
            else:
                cpu_x = torch.randn(n, c, h, w, device='cpu', dtype=dtype, requires_grad=True)
                x = cpu_x.detach().clone().to('mps').requires_grad_()

            all_sum = torch.sum(x)
            all_sum_cpu = torch.sum(cpu_x)

            self.assertEqual(all_sum, all_sum_cpu)

            nil_dim_sum = torch.sum(x, dim=[])
            nil_dim_sum_cpu = torch.sum(cpu_x, dim=[])

            self.assertEqual(nil_dim_sum, nil_dim_sum_cpu)

            nil_dim_sum_keepdim = torch.sum(x, dim=[], keepdim=True)
            nil_dim_sum_cpu_keepdim = torch.sum(cpu_x, dim=[], keepdim=True)

            self.assertEqual(nil_dim_sum_keepdim, nil_dim_sum_cpu_keepdim)

            zero_dim_sum = torch.sum(x, dim=[0])
            zero_dim_sum_cpu = torch.sum(cpu_x, dim=[0])

            self.assertEqual(zero_dim_sum, zero_dim_sum_cpu)

            zero_dim_sum_keepdim = torch.sum(x, dim=[0], keepdim=True)
            zero_dim_sum_cpu_keepdim = torch.sum(cpu_x, dim=[0], keepdim=True)

            self.assertEqual(zero_dim_sum_keepdim, zero_dim_sum_cpu_keepdim)

            zero_one_dim_sum = torch.sum(x, dim=[0, 1])
            zero_one_dim_sum_cpu = torch.sum(cpu_x, dim=[0, 1])

            self.assertEqual(zero_one_dim_sum, zero_one_dim_sum_cpu)

            zero_one_dim_sum_keepdim = torch.sum(x, dim=[0, 1], keepdim=True)
            zero_one_dim_sum_cpu_keepdim = torch.sum(cpu_x, dim=[0, 1], keepdim=True)

            self.assertEqual(zero_one_dim_sum_keepdim, zero_one_dim_sum_cpu_keepdim)

            two_three_dim_sum = torch.sum(x, dim=[2, 3])
            two_three_dim_sum_cpu = torch.sum(cpu_x, dim=[2, 3])

            self.assertEqual(two_three_dim_sum, two_three_dim_sum_cpu)

            two_three_keepdim_sum = torch.sum(x, dim=[2, 3], keepdim=True)
            two_three_dim_keepsum_cpu = torch.sum(cpu_x, dim=[2, 3], keepdim=True)

            self.assertEqual(two_three_keepdim_sum, two_three_dim_keepsum_cpu)

        helper(2, 8, 4, 5)
        helper(2, 8, 4, 5, dtype=torch.int32)
        helper(2, 8, 4, 5, dtype=torch.int64)
        helper(2, 8, 4, 5, dtype=torch.bool)

    # Test forward prod
    def test_prod(self):
        def helper(shape, dtype=torch.float32):
            cpu_x = None
            x = None
            if(dtype not in [torch.float32, torch.bool]):
                cpu_x = torch.randint(1, 6, shape, device='cpu', dtype=dtype, requires_grad=False)
                x = cpu_x.detach().clone().to('mps')
            elif (dtype == torch.bool):
                cpu_x = torch.randint(2, shape, device='cpu', dtype=dtype, requires_grad=False)
                x = cpu_x.detach().clone().to('mps')
            else:
                cpu_x = torch.randn(shape, device='cpu', dtype=dtype, requires_grad=True)
                x = cpu_x.detach().clone().to('mps').requires_grad_()

            all_prod = torch.prod(x)
            all_prod_cpu = torch.prod(cpu_x)

            self.assertEqual(all_prod, all_prod_cpu)

            for dim in range(len(shape)):
                dim_prod = torch.prod(x, dim=dim)
                dim_prod_cpu = torch.prod(cpu_x, dim=dim)

                self.assertEqual(dim_prod, dim_prod_cpu)

                dim_prod_keepdim = torch.prod(x, dim=dim, keepdim=True)
                dim_prod_cpu_keepdim = torch.prod(cpu_x, dim=dim, keepdim=True)

                self.assertEqual(dim_prod_keepdim, dim_prod_cpu_keepdim)

        for dtype in [torch.float32, torch.int32, torch.int64, torch.bool]:
            helper((2, 3), dtype)

    # Test forward mean
    def test_mean(self):
        def helper(n, c, h, w):
            cpu_x = torch.randn(n, c, h, w, device='cpu', dtype=torch.float, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            all_mean = torch.mean(x)
            all_mean_cpu = torch.mean(cpu_x)

            self.assertEqual(all_mean, all_mean_cpu)

            nil_dim_mean = torch.mean(x, dim=[])
            nil_dim_mean_cpu = torch.mean(cpu_x, dim=[])

            self.assertEqual(nil_dim_mean, nil_dim_mean_cpu)

            nil_dim_mean_keepdim = torch.mean(x, dim=[], keepdim=True)
            nil_dim_mean_cpu_keepdim = torch.mean(cpu_x, dim=[], keepdim=True)

            self.assertEqual(nil_dim_mean_keepdim, nil_dim_mean_cpu_keepdim)

            zero_dim_mean = torch.mean(x, dim=[0])
            zero_dim_mean_cpu = torch.mean(cpu_x, dim=[0])

            self.assertEqual(zero_dim_mean, zero_dim_mean_cpu)

            zero_dim_mean_keepdim = torch.mean(x, dim=[0], keepdim=True)
            zero_dim_mean_cpu_keepdim = torch.mean(cpu_x, dim=[0], keepdim=True)

            self.assertEqual(zero_dim_mean_keepdim, zero_dim_mean_cpu_keepdim)

            zero_one_dim_mean = torch.mean(x, dim=[0, 1])
            zero_one_dim_mean_cpu = torch.mean(cpu_x, dim=[0, 1])

            self.assertEqual(zero_one_dim_mean, zero_one_dim_mean_cpu)

            zero_one_dim_mean_keepdim = torch.mean(x, dim=[0, 1], keepdim=True)
            zero_one_dim_mean_cpu_keepdim = torch.mean(cpu_x, dim=[0, 1], keepdim=True)

            self.assertEqual(zero_one_dim_mean_keepdim, zero_one_dim_mean_cpu_keepdim)

            two_three_dim_mean = torch.mean(x, dim=[2, 3])
            two_three_dim_mean_cpu = torch.mean(cpu_x, dim=[2, 3])

            self.assertEqual(two_three_dim_mean, two_three_dim_mean_cpu)

            two_three_keepdim_mean = torch.mean(x, dim=[2, 3], keepdim=True)
            two_three_dim_keepmean_cpu = torch.mean(cpu_x, dim=[2, 3], keepdim=True)

            self.assertEqual(two_three_keepdim_mean, two_three_dim_keepmean_cpu)

        helper(2, 8, 4, 5)

    # Test std
    def test_std(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            x = cpu_x.detach().clone().to('mps')

            all_std = torch.std(x, unbiased=False)
            all_std_cpu = torch.std(cpu_x, unbiased=False)

            self.assertEqual(all_std, all_std_cpu)

            nil_dim_std = torch.std(x, dim=[], unbiased=False)
            nil_dim_std_cpu = torch.std(cpu_x, dim=[], unbiased=False)

            self.assertEqual(nil_dim_std, nil_dim_std_cpu)

            nil_dim_std_keepdim = torch.std(x, dim=[], keepdim=True, unbiased=False)
            nil_dim_std_cpu_keepdim = torch.std(cpu_x, dim=[], keepdim=True, unbiased=False)

            self.assertEqual(nil_dim_std_keepdim, nil_dim_std_cpu_keepdim)

            zero_dim_std = torch.std(x, dim=[0], unbiased=False)
            zero_dim_std_cpu = torch.std(cpu_x, dim=[0], unbiased=False)

            self.assertEqual(zero_dim_std, zero_dim_std_cpu)

            zero_dim_std_keepdim = torch.std(x, dim=[0], keepdim=True, unbiased=False)
            zero_dim_std_cpu_keepdim = torch.std(cpu_x, dim=[0], keepdim=True, unbiased=False)

            self.assertEqual(zero_dim_std_keepdim, zero_dim_std_cpu_keepdim)

            zero_one_dim_std = torch.std(x, dim=[0, 1], unbiased=False)
            zero_one_dim_std_cpu = torch.std(cpu_x, dim=[0, 1], unbiased=False)

            self.assertEqual(zero_one_dim_std, zero_one_dim_std_cpu)

            zero_one_dim_std_keepdim = torch.std(x, dim=[0, 1], keepdim=True, unbiased=False)
            zero_one_dim_std_cpu_keepdim = torch.std(cpu_x, dim=[0, 1], keepdim=True, unbiased=False)

            self.assertEqual(zero_one_dim_std_keepdim, zero_one_dim_std_cpu_keepdim)

            two_three_dim_std = torch.std(x, dim=[2, 3], unbiased=False)
            two_three_dim_std_cpu = torch.std(cpu_x, dim=[2, 3], unbiased=False)

            self.assertEqual(two_three_dim_std, two_three_dim_std_cpu)

            two_three_keepdim_std = torch.std(x, dim=[2, 3], keepdim=True, unbiased=False)
            two_three_dim_keepstd_cpu = torch.std(cpu_x, dim=[2, 3], keepdim=True, unbiased=False)

            self.assertEqual(two_three_keepdim_std, two_three_dim_keepstd_cpu)

            all_std = torch.std(x, unbiased=True)
            all_std_cpu = torch.std(cpu_x, unbiased=True)

            self.assertEqual(all_std, all_std_cpu)

            nil_dim_std = torch.std(x, dim=[], unbiased=True)
            nil_dim_std_cpu = torch.std(cpu_x, dim=[], unbiased=True)

            self.assertEqual(nil_dim_std, nil_dim_std_cpu)

            nil_dim_std_keepdim = torch.std(x, dim=[], keepdim=True, unbiased=True)
            nil_dim_std_cpu_keepdim = torch.std(cpu_x, dim=[], keepdim=True, unbiased=True)

            self.assertEqual(nil_dim_std_keepdim, nil_dim_std_cpu_keepdim)

            zero_dim_std = torch.std(x, dim=[0], unbiased=True)
            zero_dim_std_cpu = torch.std(cpu_x, dim=[0], unbiased=True)

            self.assertEqual(zero_dim_std, zero_dim_std_cpu)

            zero_dim_std_keepdim = torch.std(x, dim=[0], keepdim=True, unbiased=True)
            zero_dim_std_cpu_keepdim = torch.std(cpu_x, dim=[0], keepdim=True, unbiased=True)

            self.assertEqual(zero_dim_std_keepdim, zero_dim_std_cpu_keepdim)

            zero_one_dim_std = torch.std(x, dim=[0, 1], unbiased=True)
            zero_one_dim_std_cpu = torch.std(cpu_x, dim=[0, 1], unbiased=True)

            self.assertEqual(zero_one_dim_std, zero_one_dim_std_cpu)

            zero_one_dim_std_keepdim = torch.std(x, dim=[0, 1], keepdim=True, unbiased=True)
            zero_one_dim_std_cpu_keepdim = torch.std(cpu_x, dim=[0, 1], keepdim=True, unbiased=True)

            self.assertEqual(zero_one_dim_std_keepdim, zero_one_dim_std_cpu_keepdim)

            two_three_dim_std = torch.std(x, dim=[2, 3], unbiased=True)
            two_three_dim_std_cpu = torch.std(cpu_x, dim=[2, 3], unbiased=True)

            self.assertEqual(two_three_dim_std, two_three_dim_std_cpu)

            two_three_keepdim_std = torch.std(x, dim=[2, 3], keepdim=True, unbiased=True)
            two_three_dim_keepstd_cpu = torch.std(cpu_x, dim=[2, 3], keepdim=True, unbiased=True)

            self.assertEqual(two_three_keepdim_std, two_three_dim_keepstd_cpu)

        helper((4, 5, 6, 7))
        # verify if a change in shape of input would cause problems with graph caching
        helper((9, 5, 6, 7))

    # Test var
    def test_var_simple(self):
        def helper():

            shape = [2, 3, 4, 5]

            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            x = cpu_x.detach().clone().to('mps')

            for unbiased in [False, True]:
                for keepdim in [False, True]:

                    zero_dim_var = x.var(-1, keepdim=keepdim, unbiased=unbiased)
                    zero_dim_var_cpu = cpu_x.var(-1, keepdim=keepdim, unbiased=unbiased)

                    self.assertEqual(zero_dim_var, zero_dim_var_cpu)

                    all_var = torch.var(x, unbiased=unbiased)
                    all_var_cpu = torch.var(cpu_x, unbiased=unbiased)

                    self.assertEqual(all_var, all_var_cpu)

                    nil_dim_var = torch.var(x, dim=[], keepdim=keepdim, unbiased=unbiased)
                    nil_dim_var_cpu = torch.var(cpu_x, dim=[], keepdim=keepdim, unbiased=unbiased)

                    self.assertEqual(nil_dim_var, nil_dim_var_cpu)

                    zero_dim_var = torch.var(x, dim=[0], keepdim=keepdim, unbiased=unbiased)
                    zero_dim_var_cpu = torch.var(cpu_x, dim=[0], keepdim=keepdim, unbiased=unbiased)

                    self.assertEqual(zero_dim_var, zero_dim_var_cpu)

                    zero_one_dim_var = torch.var(x, dim=[0, -1], keepdim=keepdim, unbiased=unbiased)
                    zero_one_dim_var_cpu = torch.var(cpu_x, dim=[0, -1], keepdim=keepdim, unbiased=unbiased)

                    self.assertEqual(zero_one_dim_var, zero_one_dim_var_cpu)

                    two_three_dim_var = torch.var(x, dim=[2, 3], keepdim=keepdim, unbiased=unbiased)
                    two_three_dim_var_cpu = torch.var(cpu_x, dim=[2, 3], keepdim=keepdim, unbiased=unbiased)

                    self.assertEqual(two_three_dim_var, two_three_dim_var_cpu)

        helper()

    # Test forward amax
    def test_amax(self):
        def helper(shape, dim, keepdim):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            result = torch.amax(x, dim=dim, keepdim=keepdim)
            result_cpu = torch.amax(cpu_x, dim=dim, keepdim=keepdim)

            cpu_grad = torch.randn(result_cpu.shape)
            grad = cpu_grad.to('mps')

            result_cpu.backward(gradient=cpu_grad)
            result.backward(gradient=grad)

            self.assertEqual(result, result_cpu)
            self.assertEqual(x.grad, cpu_x.grad)

        for dim in ([], [0], [0, 1], [2, 3]):
            for keepdim in [False, True]:
                helper((2, 8, 4, 5), dim, keepdim)

    # Test forward amin
    def test_amin(self):
        def helper(shape, dim, keepdim):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            result = torch.amin(x, dim=dim, keepdim=keepdim)
            result_cpu = torch.amin(cpu_x, dim=dim, keepdim=keepdim)

            cpu_grad = torch.randn(result_cpu.shape)
            grad = cpu_grad.to('mps')

            result_cpu.backward(gradient=cpu_grad)
            result.backward(gradient=grad)

            self.assertEqual(result, result_cpu)
            self.assertEqual(x.grad, cpu_x.grad)

        for dim in ([], [0], [0, 1], [2, 3]):
            for keepdim in [False, True]:
                helper((2, 8, 4, 5), dim, keepdim)

    # Test minimum and maximum
    def test_minimum_maximum(self):
        def helper(n, c, h, w):
            cpu_x = torch.randn(n, c, h, w, device='cpu', dtype=torch.float, requires_grad=False)
            cpu_y = torch.randn(n, c, h, w, device='cpu', dtype=torch.float, requires_grad=False)
            mps_x = cpu_x.detach().clone().to('mps')
            mps_y = cpu_y.detach().clone().to('mps')

            minimum_result_cpu = torch.minimum(cpu_x, cpu_y)
            minimum_result_mps = torch.minimum(mps_x, mps_y)
            self.assertEqual(minimum_result_cpu, minimum_result_mps)

            maximum_result_cpu = torch.maximum(cpu_x, cpu_y)
            maximum_result_mps = torch.maximum(mps_x, mps_y)
            self.assertEqual(maximum_result_cpu, maximum_result_mps)

        helper(1, 1, 4, 5)

    # Test clamp_min
    def test_clamp_min(self):
        def helper(n, c, h, w):
            cpu_x = torch.randn(n, c, h, w, device='cpu', dtype=torch.float, requires_grad=False)
            x = cpu_x.detach().clone().to('mps')

            cpu_min_t = torch.randn(n, c, h, w, device='cpu', dtype=torch.float, requires_grad=False)
            min_t = cpu_min_t.detach().clone().to('mps')

            clamp_min_result = torch.clamp_min(x, min=5.0)
            clamp_min_result_cpu = torch.clamp_min(cpu_x, min=5.0)

            self.assertEqual(clamp_min_result, clamp_min_result_cpu)

            clamp_min_t_result = torch.clamp_min(x, min=min_t)
            clamp_min_t_result_cpu = torch.clamp_min(cpu_x, min=cpu_min_t)

            self.assertEqual(clamp_min_t_result, clamp_min_t_result_cpu)

        helper(2, 8, 4, 5)

    # Test clamp_max

    def test_clamp_max(self):
        def helper(n, c, h, w):
            cpu_x = torch.randn(n, c, h, w, device='cpu', dtype=torch.float, requires_grad=False)
            x = cpu_x.detach().clone().to('mps')

            cpu_max_t = torch.randn(n, c, h, w, device='cpu', dtype=torch.float, requires_grad=False)
            max_t = cpu_max_t.detach().clone().to('mps')

            clamp_max_result = torch.clamp_max(x, max=100.0)
            clamp_max_result_cpu = torch.clamp_max(cpu_x, max=100.0)

            self.assertEqual(clamp_max_result, clamp_max_result_cpu)

            clamp_max_t_result = torch.clamp_max(x, max=max_t)
            clamp_max_t_result_cpu = torch.clamp_max(cpu_x, max=cpu_max_t)

            self.assertEqual(clamp_max_t_result, clamp_max_t_result_cpu)

        helper(2, 8, 4, 5)

    # Test clamp
    def test_clamp(self):
        def helper(n, c, h, w):
            import numpy as np
            upper_bound = 1000
            half_upper_bound = upper_bound / 2

            # x=[0..1000)
            x_arr = upper_bound * np.random.random_sample(size=(n, c, h, w)).astype(np.float32)
            cpu_x = torch.tensor(x_arr, device='cpu', dtype=torch.float, requires_grad=False)
            x = cpu_x.detach().clone().to('mps')

            # x=[0..500)
            min_arr = half_upper_bound * np.random.random_sample(size=(n, c, h, w)).astype(np.float32)
            cpu_min_t = torch.tensor(min_arr, device='cpu', dtype=torch.float, requires_grad=False)
            min_t = cpu_min_t.detach().clone().to('mps')

            # x=[500..1000), to ensure max's are greater than mins
            max_arr = (half_upper_bound * np.random.random_sample(size=(n, c, h, w)).astype(np.float32)) + half_upper_bound
            cpu_max_t = torch.tensor(max_arr, device='cpu', dtype=torch.float, requires_grad=False)
            max_t = cpu_max_t.detach().clone().to('mps')

            # [200..600]: just an arbitrary range between [0..1000]
            clamp_result = torch.clamp(x, min=200.0, max=600.0)
            clamp_result_cpu = torch.clamp(cpu_x, min=200.0, max=600.0)
            self.assertEqual(clamp_result, clamp_result_cpu)

            # test optional scalar refs and cached graph keys by passing only max
            clamp_opt_result = torch.clamp(x, max=600.0)
            clamp_opt_result_cpu = torch.clamp(cpu_x, max=600.0)
            self.assertEqual(clamp_opt_result, clamp_opt_result_cpu)

            clamp_t_result = torch.clamp(x, min=min_t, max=max_t)
            clamp_t_result_cpu = torch.clamp(cpu_x, min=cpu_min_t, max=cpu_max_t)
            self.assertEqual(clamp_t_result, clamp_t_result_cpu)

            # test optional tensor refs and cached graph keys by passing only max
            clamp_topt_result = torch.clamp(x, max=max_t)
            clamp_topt_result_cpu = torch.clamp(cpu_x, max=cpu_max_t)
            self.assertEqual(clamp_topt_result, clamp_topt_result_cpu)

            # test inplace clamping
            x.clamp_(min=200.0, max=600.0)
            cpu_x.clamp_(min=200.0, max=600.0)
            self.assertEqual(cpu_x, x)

        helper(2, 8, 4, 5)

    def test_divmode(self):
        def helper(shape, rounding_mode):
            for dtype in [torch.float32, torch.float16, torch.int32, torch.int64]:
                cpu_x = None
                cpu_y = None
                if(dtype in [torch.float32, torch.float16]):
                    cpu_x = torch.randn(shape, device='cpu', dtype=dtype, requires_grad=False)
                    cpu_y = torch.randn(shape, device='cpu', dtype=dtype, requires_grad=False)
                else:
                    cpu_x = torch.randint(-10, 0, shape, device='cpu', dtype=dtype, requires_grad=False)
                    cpu_y = torch.randint(-10, 0, shape, device='cpu', dtype=dtype, requires_grad=False)

                mps_x = cpu_x.detach().clone().to('mps')
                # clamp to avoid division by 0
                mps_y = cpu_y.detach().clone().to('mps')

                result_div_cpu = torch.div(cpu_x, cpu_y, rounding_mode=rounding_mode)
                result_div_mps = torch.div(mps_x, mps_y, rounding_mode=rounding_mode)
                self.assertEqual(result_div_mps, result_div_cpu)

        helper((2, 8, 4, 5), None)
        helper((2, 8, 4, 5), "floor")
        helper((2, 8, 4, 5), "trunc")

    def test_rounding(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            mps_x = cpu_x.detach().clone().to('mps')

            result_floor_cpu = torch.floor(cpu_x)
            result_floor_mps = torch.floor(mps_x)
            self.assertEqual(result_floor_mps, result_floor_cpu)

            result_ceil_cpu = torch.ceil(cpu_x)
            result_ceil_mps = torch.ceil(mps_x)
            self.assertEqual(result_ceil_mps, result_ceil_cpu)

            result_trunc_cpu = torch.trunc(cpu_x)
            result_trunc_mps = torch.trunc(mps_x)
            self.assertEqual(result_trunc_mps, result_trunc_cpu)

            result_round_cpu = torch.round(cpu_x)
            result_round_mps = torch.round(mps_x)
            self.assertEqual(result_round_mps, result_round_cpu)

        helper((2, 6, 3, 5))
        helper((2, 8, 4, 5))

    def test_expand(self):
        def helper(n, c):
            values = [[1.0], [4.0], [7.0]]
            cpu_x = torch.tensor(values, device='cpu')
            x = cpu_x.detach().clone().to('mps')

            strided_cpu = torch.as_strided(cpu_x, (3, 4), (1, 0))
            strided_mps = torch.as_strided(x, (3, 4), (1, 0))

            self.assertEqual(strided_mps, strided_cpu)

        helper(3, 1)

    def test_select(self):
        def helper(n, c):
            cpu_x = torch.randn(n, c, device='cpu', dtype=torch.float, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            strided_cpu = torch.as_strided(cpu_x, (3, 1), (3, 1))
            strided_mps = torch.as_strided(x, (3, 1), (3, 1))
            self.assertEqual(strided_mps, strided_cpu)

            strided_cpu = torch.as_strided(cpu_x, (1, 3), (3, 1))
            strided_mps = torch.as_strided(x, (1, 3), (3, 1))
            self.assertEqual(strided_mps, strided_cpu)

            strided_cpu = torch.as_strided(cpu_x, (3, 1), (3, 1), storage_offset=1)
            strided_mps = torch.as_strided(x, (3, 1), (3, 1), storage_offset=1)

            self.assertEqual(strided_mps, strided_cpu)

        helper(3, 3)

    def test_assert_topk(self):
        # here the k > 16 raises an error as expected
        with self.assertRaisesRegex(RuntimeError, "Currently topk on mps works only for k<=16"):
            xs = torch.arange(30).to('mps')
            xs.topk(30)
        # for k <= 16 it works fine
        ys_cpu = torch.arange(30)
        ys_mps = ys_cpu.to('mps')
        self.assertEqual(ys_cpu.topk(16), ys_mps.topk(16))

    def test_topk(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            x = cpu_x.detach().clone().to('mps')
            for largest_val in [True, False]:
                if (type(shape) == tuple):
                    for curr_dim in range(0, len(shape)):
                        dim_size = shape[curr_dim]
                        for k in range(1, dim_size + 1):
                            topk_values, topk_indices = torch.topk(x, k, dim=curr_dim, largest=largest_val)
                            topk_values_cpu, topk_indices_cpu = torch.topk(cpu_x, k, dim=curr_dim, largest=largest_val)
                            self.assertEqual(topk_values, topk_values_cpu)
                            self.assertEqual(topk_indices, topk_indices_cpu)
                else:
                    for k in range(1, shape):
                        topk_values, topk_indices = torch.topk(x, k, dim=0, largest=largest_val)
                        topk_values_cpu, topk_indices_cpu = torch.topk(cpu_x, k, dim=0, largest=largest_val)
                        self.assertEqual(topk_values, topk_values_cpu)
                        self.assertEqual(topk_indices, topk_indices_cpu)

        helper(2)
        helper((5, 1))
        helper((1, 5))
        helper((5, 9, 7, 4))

    def test_upsample_nearest_exact2d(self):
        def helper(N, C, H, W):
            inputCPU = torch.arange(N * C * H * W, device='cpu', dtype=torch.float,
                                    requires_grad=True).reshape(N, C, H, W)
            inputCPU.retain_grad()
            inputMPS = inputCPU.detach().clone().to('mps').requires_grad_()

            outputCPU = torch.nn.functional.interpolate(inputCPU, size=(5, 5), mode='nearest-exact')
            outputMPS = torch.nn.functional.interpolate(inputMPS, size=(5, 5), mode='nearest-exact')

            self.assertEqual(outputCPU, outputMPS)

            outputCPU.backward(gradient=torch.full_like(outputCPU, 0.3))
            outputMPS.backward(gradient=torch.full_like(outputMPS, 0.3))

            self.assertEqual(inputCPU.grad, inputMPS.grad)

        helper(1, 1, 4, 4)
        helper(7, 5, 3, 2)

    def test_upsample_nearest2d(self):
        def helper(N, C, H, W):
            inputCPU = torch.arange(N * C * H * W, device='cpu', dtype=torch.float,
                                    requires_grad=True).reshape(N, C, H, W)
            inputCPU.retain_grad()
            inputMPS = inputCPU.detach().to('mps').requires_grad_()

            values = [1, 2, 5, 10, 40]

            for i in values:
                for j in values:
                    upsample_nearest2d = nn.UpsamplingNearest2d(scale_factor=(i, j))

                    outputCPU = upsample_nearest2d(inputCPU)
                    outputMPS = upsample_nearest2d(inputMPS)

                    self.assertEqual(outputCPU, outputMPS)
                    upsample_nearest2d = nn.UpsamplingNearest2d((i * H, j * W))

                    outputCPU = upsample_nearest2d(inputCPU)
                    outputMPS = upsample_nearest2d(inputMPS)

                    self.assertEqual(outputCPU, outputMPS)

                    outputCPU.backward(gradient=torch.full_like(outputCPU, 0.3))
                    outputMPS.backward(gradient=torch.full_like(outputMPS, 0.3))

                    self.assertEqual(inputCPU.grad, inputMPS.grad)

        helper(1, 1, 4, 4)
        helper(7, 5, 3, 2)

    def test_upsample_bilinear2d(self):
        def helper(N, C, H, W):
            inputCPU = torch.arange(N * C * H * W, device='cpu', dtype=torch.float,
                                    requires_grad=True).reshape(N, C, H, W)
            inputCPU.retain_grad()
            inputMPS = inputCPU.detach().clone().to('mps').requires_grad_()

            values = [1, 2, 5, 10, 40]

            for i in values:
                for j in values:
                    upsample_bilinear2d = nn.UpsamplingBilinear2d(scale_factor=(i, j))

                    outputCPU = upsample_bilinear2d(inputCPU)
                    outputMPS = upsample_bilinear2d(inputMPS)

                    self.assertEqual(outputCPU, outputMPS)

                    upsample_bilinear2d = nn.UpsamplingBilinear2d((i * H, j * W))

                    outputCPU = upsample_bilinear2d(inputCPU)
                    outputMPS = upsample_bilinear2d(inputMPS)

                    self.assertEqual(outputCPU, outputMPS)

                    outputCPU.backward(gradient=torch.full_like(outputCPU, 0.3))
                    outputMPS.backward(gradient=torch.full_like(outputMPS, 0.3))

                    self.assertEqual(inputCPU.grad, inputMPS.grad)

        helper(1, 1, 4, 4)
        helper(7, 5, 3, 2)

    def test_upsample_nearest1d(self):
        def helper(N, C, H, W):
            inputCPU = torch.arange(C * H * W, device='cpu', dtype=torch.float,
                                    requires_grad=True).reshape(C, H, W)
            inputMPS = inputCPU.detach().clone().to('mps')

            outputCPU = torch.nn.functional.interpolate(inputCPU, scale_factor=2.0, mode='nearest')
            outputMPS = torch.nn.functional.interpolate(inputMPS, scale_factor=2.0, mode='nearest')

            self.assertEqual(outputCPU, outputMPS)

        helper(1, 1, 4, 4)
        helper(7, 5, 3, 2)

    # Test concat forward
    def test_cat1(self):
        def helper(shape_x, shape_y, shape_z):
            cpu_x = torch.randn(shape_x, device='cpu', dtype=torch.float, requires_grad=False)
            x = cpu_x.detach().clone().to('mps')

            cpu_y = torch.randn(shape_y, device='cpu', dtype=torch.float, requires_grad=False)
            y = cpu_y.detach().clone().to('mps')

            cpu_z = torch.randn(shape_z, device='cpu', dtype=torch.float, requires_grad=False)
            z = cpu_z.detach().clone().to('mps')

            cat = torch.cat([x, y, z], dim=1)
            cat_cpu = torch.cat([cpu_x, cpu_y, cpu_z], dim=1)

            self.assertEqual(cat, cat_cpu)

        helper([2, 2, 4, 5], [2, 3, 4, 5], [2, 5, 4, 5])
        helper([2, 2, 6, 5], [2, 3, 6, 5], [2, 5, 6, 5])
        helper([0, 2, 4, 5], [0, 3, 4, 5], [0, 5, 4, 5])
        helper([2, 2, 6, 5], [0], [2, 5, 6, 5])
        helper([0], [2, 3, 6, 5], [2, 5, 6, 5])
        helper([2, 3, 4, 5], [2, 5, 4, 5], [0])
        helper([2, 2, 6, 5], [2, 0, 6, 5], [2, 5, 6, 5])
        helper([2, 0, 6, 5], [2, 3, 6, 5], [2, 5, 6, 5])
        helper([2, 0, 6, 5], [2, 3, 6, 5], [2, 0, 6, 5])

    def test_constant_pad(self):
        m = torch.nn.ConstantPad2d((-2, -2, -2, -2), 3.5)
        input_cpu = torch.randn(1, 16, 16, 16)
        input_mps = input_cpu.detach().clone().to("mps")
        r_cpu = m(input_cpu)
        r_mps = m(input_mps)
        self.assertEqual(r_cpu, r_mps.to("cpu"))

    def test_circular_pad(self):
        # https://github.com/pytorch/pytorch/issues/80856
        k_cpu = torch.ones(3, 3, 9, 9)
        k_mps = k_cpu.detach().clone().to("mps")

        x_cpu = torch.rand(1, 3, 32, 32)
        x_mps = x_cpu.detach().clone().to("mps")

        x_pad_cpu = F.pad(x_cpu, (2, 2, 2, 2), mode='circular')
        x_pad_mps = F.pad(x_mps, (2, 2, 2, 2), mode='circular')

        y_cpu = F.conv2d(x_pad_cpu, k_cpu)
        y_mps = F.conv2d(x_pad_mps, k_mps)

        self.assertEqual(y_cpu, y_mps.cpu())

    def test_constant_pad_4d_warning(self):
        inputCPU = torch.rand((1, 2, 2, 2, 1, 1))
        inputMPS = inputCPU.detach().clone().to('mps')
        outputCPU = F.pad(inputCPU, [0, 0, 0, 0, 0, 0, 1, 0])
        outputMPS = F.pad(inputMPS, [0, 0, 0, 0, 0, 0, 1, 0])
        self.assertEqual(outputCPU, outputMPS)

    def test_pad(self):
        def helper(shape, padding, op, value=0):
            inputCPU = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            inputCPU.retain_grad()
            inputMPS = inputCPU.detach().clone().to('mps').requires_grad_()

            if (op in [nn.ConstantPad1d, nn.ConstantPad2d, nn.ConstantPad3d]):
                padCriteria = op(padding, value)
            else:
                padCriteria = op(padding)
            outputCPU = padCriteria(inputCPU)
            outputMPS = padCriteria(inputMPS)
            self.assertEqual(outputCPU, outputMPS)

            # backward pass (chose 0.6 just to have the grad_output != 1)
            outputCPU.backward(gradient=torch.full_like(outputCPU, 0.6))
            outputMPS.backward(gradient=torch.full_like(outputMPS, 0.6))
            self.assertEqual(inputCPU.grad, inputMPS.grad)

        # 1D Padding
        helper((2, 4, 3), 2, nn.ReflectionPad1d)
        # verify if a change in shape of input would cause problems with graph caching
        helper((2, 4, 4), (1, 3), nn.ReflectionPad1d)
        # Replication 1D
        helper((2, 1, 6), 3, nn.ReplicationPad1d)
        # Constant Pad 1D
        helper((2, 3, 4), 2, nn.ConstantPad1d)
        # Constant Pad 1D with single dimension input
        helper((16), (1, 2), nn.ConstantPad1d)

        # 2D Padding
        helper((1, 2, 3, 4), (1, 1, 2, 0), nn.ReflectionPad2d)
        # verify if a change in shape of input would cause problems with graph caching
        helper((2, 4, 3, 4), (1, 1, 2, 0), nn.ReflectionPad2d)
        # this should make the padding (2, 2, 2, 2)
        helper((2, 1, 6, 8), 2, nn.ReplicationPad2d)
        # verify if a change in shape of padding would cause problems with graph caching
        helper((2, 1, 6, 8), (2, 4, 3, 5), nn.ReplicationPad2d)
        # Constant Pad 2D
        helper((2, 1, 6, 8), (2, 4, 3, 5), nn.ConstantPad2d)
        # input size < pad size
        helper((1, 2, 3), (0, 0, 0, 1), nn.ConstantPad2d)

        # 3D Padding
        helper((2, 4, 6, 8, 4), (1, 3, 3, 5, 3, 4), nn.ReflectionPad3d)
        # verify if a change in shape of padding would cause problems with graph caching
        helper((2, 4, 6, 8, 4), (1, 3, 3, 5, 3, 4), nn.ReplicationPad3d)
        # Constant Pad 3D
        helper((2, 4, 6, 8, 4), (1, 3, 3, 5, 3, 4), nn.ConstantPad3d)

    # Test stack forward
    def test_stack(self):
        # All shapes must be same
        def helper(shape, dtype=torch.float32):

            x, cpu_x = None, None
            y, cpu_y = None, None
            z, cpu_z = None, None

            if(dtype not in [torch.float32, torch.bool]):
                cpu_x = torch.randint(50, shape, device='cpu', dtype=dtype, requires_grad=False)
                x = cpu_x.detach().clone().to('mps')
                cpu_y = torch.randint(50, shape, device='cpu', dtype=dtype, requires_grad=False)
                y = cpu_y.detach().clone().to('mps')
                cpu_z = torch.randint(50, shape, device='cpu', dtype=dtype, requires_grad=False)
                z = cpu_z.detach().clone().to('mps')
            elif (dtype == torch.bool):
                cpu_x = torch.randint(2, shape, device='cpu', dtype=dtype, requires_grad=False)
                x = cpu_x.detach().clone().to('mps')
                cpu_y = torch.randint(2, shape, device='cpu', dtype=dtype, requires_grad=False)
                y = cpu_y.detach().clone().to('mps')
                cpu_z = torch.randint(2, shape, device='cpu', dtype=dtype, requires_grad=False)
                z = cpu_z.detach().clone().to('mps')
            else:
                cpu_x = torch.randn(shape, device='cpu', dtype=dtype, requires_grad=True)
                x = cpu_x.detach().clone().to('mps').requires_grad_()
                cpu_y = torch.randn(shape, device='cpu', dtype=dtype, requires_grad=True)
                y = cpu_y.detach().clone().to('mps').requires_grad_()
                cpu_z = torch.randn(shape, device='cpu', dtype=dtype, requires_grad=True)
                z = cpu_z.detach().clone().to('mps').requires_grad_()

            stack = torch.stack([x, y, z], dim=1)
            stack_cpu = torch.stack([cpu_x, cpu_y, cpu_z], dim=1)

            self.assertEqual(stack, stack_cpu)

        helper([2, 8, 4, 5])
        helper([2, 8, 4, 5], dtype=torch.float16)
        helper([2, 8, 4, 5], dtype=torch.int32)
        helper([2, 8, 4, 5], dtype=torch.int64)
        helper([2, 8, 4, 5], dtype=torch.bool)
        # Empty test - Currently failing! Empty tensor not handled!
        # helper([0, 2, 4, 5])

    # Test abs
    def test_abs(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            x = cpu_x.detach().clone().to('mps')

            abs_result = torch.abs(x)
            abs_result_cpu = torch.abs(cpu_x)

            self.assertEqual(abs_result, abs_result_cpu)

        helper((2, 8, 4, 5))

    def test_log(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            x = cpu_x.detach().clone().to('mps')

            log_result = torch.log(x)
            log_result_cpu = torch.log(cpu_x)

            self.assertEqual(log_result, log_result_cpu)

        helper((2, 8, 4, 5))

    def test_log_ten(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            x = cpu_x.detach().clone().to('mps')

            log_ten_result = torch.log10(x)
            log_ten_result_cpu = torch.log10(cpu_x)

            self.assertEqual(log_ten_result, log_ten_result_cpu)

        helper((2, 8, 4, 5))

    def test_log_two(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            x = cpu_x.detach().clone().to('mps')

            log_two_result = torch.log2(x)
            log_two_result_cpu = torch.log2(cpu_x)

            self.assertEqual(log_two_result, log_two_result_cpu)

        helper((2, 8, 4, 5))

    def test_log1p(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            x = cpu_x.detach().clone().to('mps')

            log_result = torch.log1p(x)
            log_result_cpu = torch.log1p(cpu_x)

            self.assertEqual(log_result, log_result_cpu)

        helper((2, 8, 4, 5))

    def test_logaddexp(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            x = cpu_x.detach().clone().to('mps')

            cpu_y = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            y = cpu_y.detach().clone().to('mps')

            log_result = torch.logaddexp(x, y)
            log_result_cpu = torch.logaddexp(cpu_x, cpu_y)

            self.assertEqual(log_result, log_result_cpu)

        helper((2, 8, 4, 5))

    def test_logaddexp2(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            x = cpu_x.detach().clone().to('mps')

            cpu_y = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            y = cpu_y.detach().clone().to('mps')

            log_result = torch.logaddexp2(x, y)
            log_result_cpu = torch.logaddexp2(cpu_x, cpu_y)

            self.assertEqual(log_result, log_result_cpu)

        helper((2, 8, 4, 5))

    # Test concat forward
    def test_cat2(self):

        def helper1(shape_x, shape_y, shape_z, shape_w):
            cpu_x = torch.randn(shape_x, device='cpu', dtype=torch.float, requires_grad=False)
            x = cpu_x.detach().clone().to('mps')

            cpu_y = torch.randn(shape_y, device='cpu', dtype=torch.float, requires_grad=False)
            y = cpu_y.detach().clone().to('mps')

            cpu_z = torch.randn(shape_z, device='cpu', dtype=torch.float, requires_grad=False)
            z = cpu_z.detach().clone().to('mps')

            cpu_w = torch.randn(shape_w, device='cpu', dtype=torch.float, requires_grad=False)
            w = cpu_w.detach().clone().to('mps')

            cat = torch.cat([x, y, z, w], dim=1)
            cat_cpu = torch.cat([cpu_x, cpu_y, cpu_z, cpu_w], dim=1)

            self.assertEqual(cat, cat_cpu)

        def helper(shape_x, shape_y, shape_z):
            cpu_x = torch.randn(shape_x, device='cpu', dtype=torch.float, requires_grad=False)
            x = cpu_x.detach().clone().to('mps')

            cpu_y = torch.randn(shape_y, device='cpu', dtype=torch.float, requires_grad=False)
            y = cpu_y.detach().clone().to('mps')

            cpu_z = torch.randn(shape_z, device='cpu', dtype=torch.float, requires_grad=False)
            z = cpu_z.detach().clone().to('mps')

            cat = torch.cat([x, y, z], dim=1)
            cat_cpu = torch.cat([cpu_x, cpu_y, cpu_z], dim=1)

            self.assertEqual(cat, cat_cpu)

        helper([2, 8, 4, 5], [2, 10, 4, 5], [2, 6, 4, 5])
        helper([2, 2, 4, 5], [2, 3, 4, 5], [2, 5, 4, 5])
        # Empty test - Currently failing! Empty tensor not handled!
        # helper([0, 2, 4, 5], [2, 0, 4, 5], [2, 5, 0, 5])

    # Test isnan
    def test_isnan(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            nan_index = [random.randrange(0, shape[0])]
            # make a selected row inf
            cpu_x.index_put_(indices=[torch.tensor(nan_index)], values=torch.tensor(float('nan')))
            x = cpu_x.detach().clone().to('mps')

            isnan_result = torch.isnan(x)
            isnan_result_cpu = torch.isnan(cpu_x)

            self.assertEqual(isnan_result, isnan_result_cpu)

        helper((8, 2, 4, 5))

    # Test reciprocal
    def test_reciprocal(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            reciprocal_result = torch.reciprocal(x)
            reciprocal_result_cpu = torch.reciprocal(cpu_x)

            cpu_grad = torch.ones_like(reciprocal_result_cpu)
            grad = cpu_grad.to('mps')

            reciprocal_result.backward(gradient=grad)
            reciprocal_result_cpu.backward(gradient=cpu_grad)

            self.assertEqual(reciprocal_result, reciprocal_result_cpu)
            self.assertEqual(x.grad, cpu_x.grad)

        helper((2, 8, 4, 5))

    # Test sqrt
    def test_sqrt(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            sqrt_result = torch.sqrt(x)
            sqrt_result_cpu = torch.sqrt(cpu_x)

            cpu_grad = torch.ones_like(sqrt_result_cpu)
            grad = cpu_grad.to('mps')

            sqrt_result.backward(gradient=grad)
            sqrt_result_cpu.backward(gradient=cpu_grad)

            self.assertEqual(sqrt_result, sqrt_result_cpu)
            self.assertEqual(x.grad, cpu_x.grad)

        helper((2, 8, 4, 5))

    # Test selu, elu, celu
    def test_elu(self):
        def helper(shape, alpha=1.0):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            for activation_func in [torch.nn.ELU(alpha=alpha), torch.nn.CELU(alpha=alpha), torch.nn.SELU()]:
                elu_result = activation_func(x)
                elu_result_cpu = activation_func(cpu_x)

                cpu_grad = torch.randn(elu_result_cpu.shape)
                grad = cpu_grad.to('mps')

                elu_result.backward(gradient=grad)
                elu_result_cpu.backward(gradient=cpu_grad)

                self.assertEqual(elu_result, elu_result_cpu)
                self.assertEqual(x.grad, cpu_x.grad)

        # Test empty shape too
        for shape in [[], (2, 3), (2, 8, 4, 5)]:
            for alpha in [0.000001, 1.0, 2.3, 0.34, 23]:
                helper(shape, alpha)

    # Test glu
    def test_glu(self):
        def helper(shape, dim=0):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            for activation_func in [torch.nn.GLU(dim=dim)]:
                glu_result = activation_func(x)
                glu_result_cpu = activation_func(cpu_x)

                cpu_grad = torch.randn(glu_result_cpu.shape)
                grad = cpu_grad.to('mps')

                glu_result.backward(gradient=grad)
                glu_result_cpu.backward(gradient=cpu_grad)

                self.assertEqual(glu_result, glu_result_cpu)
                self.assertEqual(x.grad, cpu_x.grad)

        for shape in [[4], (2, 4), (2, 8, 4, 6)]:
            for dim in range(len(shape)):
                helper(shape, dim)

    # Test softplus
    def test_softplus(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            softplus_result = torch.nn.Softplus(beta=0.5, threshold=0.5)(x)
            softplus_result_cpu = torch.nn.Softplus(beta=0.5, threshold=0.5)(cpu_x)

            cpu_grad = torch.randn(softplus_result.shape)
            grad = cpu_grad.to('mps')

            softplus_result.backward(gradient=grad)
            softplus_result_cpu.backward(gradient=cpu_grad)

            self.assertEqual(softplus_result, softplus_result_cpu)
            self.assertEqual(x.grad, cpu_x.grad)

        # Test empty shape too
        for shape in [(), (2, 3), (10, 10), (2, 3, 4, 5)]:
            helper(shape)

    # Test silu

    def test_silu(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            silu_result = torch.nn.SiLU()(x)
            silu_result_cpu = torch.nn.SiLU()(cpu_x)

            cpu_grad = torch.randn(silu_result_cpu.shape)
            grad = cpu_grad.to('mps')

            silu_result.backward(gradient=grad)
            silu_result_cpu.backward(gradient=cpu_grad)

            self.assertEqual(silu_result, silu_result_cpu)
            self.assertEqual(x.grad, cpu_x.grad)

        # Test empty shape too
        for shape in [[], (2, 3), (2, 8, 4, 5)]:
            helper(shape)

    def test_cast_mps_to_cpu(self):
        def helper(src_dtype, dst_dtype):
            input = torch.rand((1, 3, 128, 128), dtype=src_dtype)
            input_cast_mps = input.to('mps')
            input_cast_cpu = input_cast_mps.to('cpu', dtype=dst_dtype)

            # needs to match the initial Tensor
            self.assertEqual(input_cast_cpu, input.to(dtype=dst_dtype))
        helper(torch.half, torch.float)
        helper(torch.float, torch.half)

    def test_cast_mps_to_mps(self):
        def helper(src_dtype, dst_dtype):
            input_cpu = torch.rand((1, 3, 128, 128), dtype=src_dtype)
            input_mps = input_cpu.to('mps')
            output_mps = input_mps.to(dtype=dst_dtype)
            output_cpu = input_cpu.to(dtype=dst_dtype)
            self.assertEqual(output_mps.cpu(), output_cpu)
        helper(torch.half, torch.float)
        helper(torch.float, torch.half)
        helper(torch.half, torch.long)
        helper(torch.float, torch.int)

    # Test adaptive avg pool2d - when the input size is a multiple of output size
    # Not testing for channels last right now
    def test_adaptive_avg_pool2d_simple(self):
        def helper(input_shape, out_shape, channels_last):
            cpu_x = torch.randn(input_shape, device='cpu', dtype=torch.float, requires_grad=True)
            if(channels_last):
                cpu_x = cpu_x.to(memory_format=torch.channels_last)
                cpu_x.retain_grad()
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            avg_result = torch.nn.AdaptiveAvgPool2d(out_shape)(x)
            avg_result_cpu = torch.nn.AdaptiveAvgPool2d(out_shape)(cpu_x)

            cpu_grad = torch.randn(avg_result_cpu.shape)
            grad = cpu_grad.to('mps')

            avg_result.backward(gradient=grad)
            avg_result_cpu.backward(gradient=cpu_grad)

            self.assertEqual(avg_result, avg_result_cpu)
            self.assertEqual(x.grad, cpu_x.grad)

        helper((2, 2, 4, 4), (2, 2), False)
        helper((2, 2, 9, 9), (3, 3), False)
        helper((2, 2, 9, 9), (9, 9), False)
        helper((2, 2, 16, 16), (2, 2), False)
        helper((2, 2, 16, 16), (2, 16), False)

        helper((2, 16, 16), (4, 4), False)

        # Output shape larger than input shape

        helper((2, 2, 4, 4), (8, 8), False)
        helper((2, 2, 2, 2), (4, 4), False)
        helper((2, 2, 3, 3), (9, 9), False)
        helper((2, 2, 2, 2), (16, 16), False)
        helper((2, 2, 2, 16), (16, 16), False)

        helper((2, 4, 4), (16, 16), False)

        try:
            helper((2, 2, 3, 3), (7, 7), False)
        except Exception as e:
            pass

    # Test max avg pool2d - when the input size is a multiple of output size
    # Not testing for channels last right now
    def test_adaptive_max_pool2d_simple(self):
        def helper(input_shape, out_shape, return_indices, dtype, channels_last=False):
            cpu_x = None
            if(dtype in [torch.float16, torch.float32]):
                cpu_x = torch.randn(input_shape, device='cpu', dtype=dtype, requires_grad=True)
            else:
                cpu_x = torch.randint(50, input_shape, device='cpu', dtype=dtype, requires_grad=True)
            if(channels_last):
                cpu_x = cpu_x.to(memory_format=torch.channels_last)
                cpu_x.retain_grad()
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            max_result, max_indices = None, None
            max_result_cpu, max_indices_cpu = None, None

            if(return_indices):
                max_result, max_indices = torch.nn.AdaptiveMaxPool2d(out_shape, return_indices)(x)
                max_result_cpu, max_indices_cpu = torch.nn.AdaptiveMaxPool2d(out_shape, return_indices)(cpu_x)
            else:
                max_result = torch.nn.AdaptiveMaxPool2d(out_shape, return_indices)(x)
                max_result_cpu = torch.nn.AdaptiveMaxPool2d(out_shape, return_indices)(cpu_x)

            cpu_grad = torch.randn(max_result_cpu.shape)
            grad = cpu_grad.to('mps')

            max_result.backward(gradient=grad)
            max_result_cpu.backward(gradient=cpu_grad)

            self.assertEqual(max_result, max_result_cpu)
            if(return_indices):
                self.assertEqual(max_indices, max_indices_cpu)
            self.assertEqual(x.grad, cpu_x.grad)

        for dtype in [torch.float32]:
            for return_indices in [False, True]:
                helper((2, 2, 4, 4), (2, 2), return_indices, dtype)
                helper((2, 2, 9, 9), (3, 3), return_indices, dtype)
                helper((2, 2, 9, 9), (9, 9), return_indices, dtype)
                helper((2, 2, 16, 16), (2, 2), return_indices, dtype)
                helper((2, 2, 16, 16), (2, 16), return_indices, dtype)
                helper((2, 16, 16), (4, 4), return_indices, dtype)

    def test_gelu_simple(self):
        def helper(shape, dtype=torch.float):
            cpu_x = torch.randn(shape, device='cpu', dtype=dtype, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            gelu_result = torch.nn.GELU()(x)
            # GELU is not supported on CPU, so cast it to float
            gelu_result_cpu = torch.nn.GELU()(cpu_x.to(torch.float))

            cpu_grad = torch.ones_like(gelu_result_cpu)
            grad = cpu_grad.to('mps')

            gelu_result.backward(gradient=grad)
            gelu_result_cpu.backward(gradient=cpu_grad)

            atol = 1e-5 if dtype == torch.float else 1e-2
            rtol = 1e-3 if dtype == torch.float else 1e-2
            self.assertEqual(gelu_result, gelu_result_cpu.to(dtype), atol=atol, rtol=rtol)
            self.assertEqual(x.grad, cpu_x.grad, atol=atol, rtol=rtol)

        # Test empty shape too
        for dtype in [torch.float, torch.half]:
            for shape in [(0, 3), [], (2, 3), (2, 8, 4, 5)]:
                helper(shape, dtype)
        # Test that gelu would raise an assert for integral types
        for dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
            self.assertRaises(RuntimeError, lambda: torch.nn.GELU()(torch.randint(100, (2,), dtype=dtype, device="mps")))

    def test_gelu(self):
        def _test_gelu(n, m, dtype, contiguous, atol=None, rtol=None):
            numpy_dtype = {
                torch.bfloat16: torch.float, torch.float: torch.float, torch.double: torch.double
            }[dtype]
            devices = ['cpu']
            devices += ['mps']

            def _gelu_ref(X):
                return X * stats.norm.cdf(X)

            for d in devices:
                X = torch.rand(n, m, dtype=dtype, requires_grad=True, device=d)[:, ::2]
                res = X
                ref = (X.to(numpy_dtype).cpu().detach().numpy())
                self.assertEqual(res, ref, rtol=rtol, atol=atol, exact_dtype=False)

        for n in [1, 5, 10]:
            for m in [1, 5, 10]:
                _test_gelu(n, m, torch.float32, True)
                _test_gelu(n, m, torch.float32, False)

        # Test multi threaded
        num_threads = torch.get_num_threads()
        torch.set_num_threads(4)
        try:
            _test_gelu(32, 32, torch.float32, False)
        finally:
            torch.set_num_threads(num_threads)

    # Test hardtanh
    def test_hardtanh(self):
        def helper(shape, min_val, max_val, inplace=False):
            cpu_x = None
            x = None

            if(not inplace):
                cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
                x = cpu_x.detach().clone().to('mps').requires_grad_()
            else:
                cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
                x = cpu_x.detach().clone().to('mps')

            hardtanh_result = torch.nn.Hardtanh(min_val=min_val, max_val=max_val, inplace=inplace)(x)
            hardtanh_result_cpu = torch.nn.Hardtanh(min_val=min_val, max_val=max_val, inplace=inplace)(cpu_x)

            self.assertEqual(hardtanh_result, hardtanh_result_cpu)

            if(not inplace):
                cpu_grad = torch.randn(hardtanh_result_cpu.shape)
                grad = cpu_grad.to('mps')
                hardtanh_result.backward(gradient=grad)
                hardtanh_result_cpu.backward(gradient=cpu_grad)
                self.assertEqual(x.grad, cpu_x.grad)

        # Test empty shape too
        for shape in [(0, 3), [], (2, 3), (2, 8, 4, 5)]:
            for min_val, max_val in zip([-1, -2, 3], [1, -1, 4]):
                helper(shape, min_val, max_val)
                helper(shape, min_val, max_val, inplace=True)

    def test_transpose_2D(self):
        values = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        values1 = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        cpu_x = torch.tensor(values, device='cpu')
        mps_x = torch.tensor(values, device='mps')
        mps_x1 = torch.tensor(values1, device='mps')

        cpu_transpose = torch.transpose(cpu_x, 0, 1)
        mps_transpose = torch.transpose(mps_x, 0, 1)
        self.assertEqual(cpu_transpose, mps_transpose.to('cpu'))

    def test_transpose_3D(self):
        values = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]
        cpu_x = torch.tensor(values, device='cpu')
        mps_x = torch.tensor(values, device='mps')

        cpu_transpose1 = torch.transpose(cpu_x, 0, 1)
        mps_transpose1 = torch.transpose(mps_x, 0, 1).to('cpu')
        self.assertEqual(cpu_transpose1, mps_transpose1)

        cpu_transpose2 = torch.transpose(cpu_x, 0, 2)
        mps_transpose2 = torch.transpose(mps_x, 0, 2).to('cpu')
        self.assertEqual(cpu_transpose2, mps_transpose2)

        cpu_transpose3 = torch.transpose(cpu_x, 1, 2)
        mps_transpose3 = torch.transpose(mps_x, 1, 2).to('cpu')
        self.assertEqual(cpu_transpose3, mps_transpose3)


    def test_transpose_4D(self):
        values = [[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]],
                  [[[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]], [[19.0, 20.0, 21.0], [22.0, 23.0, 24.0]]]]
        cpu_x = torch.tensor(values, device='cpu')
        mps_x = torch.tensor(values, device='mps')

        cpu_transpose1 = torch.transpose(cpu_x, 0, 1)
        mps_transpose1 = torch.transpose(mps_x, 0, 1).to('cpu')
        self.assertEqual(cpu_transpose1, mps_transpose1)

        cpu_transpose2 = torch.transpose(cpu_x, 0, 2)
        mps_transpose2 = torch.transpose(mps_x, 0, 2).to('cpu')
        self.assertEqual(cpu_transpose2, mps_transpose2)

        cpu_transpose3 = torch.transpose(cpu_x, 0, 3)
        mps_transpose3 = torch.transpose(mps_x, 0, 3).to('cpu')
        self.assertEqual(cpu_transpose3, mps_transpose3)

        cpu_transpose4 = torch.transpose(cpu_x, 3, 1)
        mps_transpose4 = torch.transpose(mps_x, 3, 1).to('cpu')
        self.assertEqual(cpu_transpose4, mps_transpose4)

        cpu_transpose5 = torch.transpose(cpu_x, 3, 2)
        mps_transpose5 = torch.transpose(mps_x, 3, 2).to('cpu')
        self.assertEqual(cpu_transpose5, mps_transpose5)

        cpu_transpose6 = torch.transpose(cpu_x, 1, 2)
        mps_transpose6 = torch.transpose(mps_x, 1, 2).to('cpu')
        self.assertEqual(cpu_transpose6, mps_transpose6)

    # Test sign
    def test_sign(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            sign_result = torch.sign(x)
            sign_result_cpu = torch.sign(cpu_x)

            cpu_grad = torch.ones_like(sign_result_cpu)
            grad = cpu_grad.to('mps')

            sign_result.backward(gradient=grad)
            sign_result_cpu.backward(gradient=cpu_grad)

            self.assertEqual(sign_result, sign_result_cpu)

        helper((2, 8, 4, 5))

    def test_signbit(self):
        def helper(shape, dtype):
            cpu_x = torch.randn(shape, device='cpu').to(dtype)
            x = cpu_x.clone().to('mps')

            signbit_result = torch.signbit(x)
            signbit_result_cpu = torch.signbit(cpu_x)

            self.assertEqual(signbit_result, signbit_result_cpu)

        helper((2, 8, 4, 5), torch.int)
        helper((2, 8, 4, 5), torch.float)
        helper((2, 8, 4, 5), torch.int64)

    # Test neg
    def test_neg(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            neg_result = torch.neg(x)
            neg_result_cpu = torch.neg(cpu_x)

            cpu_grad = torch.ones_like(neg_result_cpu)
            grad = cpu_grad.to('mps')

            neg_result.backward(gradient=grad)
            neg_result_cpu.backward(gradient=cpu_grad)

            self.assertEqual(neg_result, neg_result_cpu)

        helper((2, 8, 4, 5))

    # Test index add
    def test_index_add(self):
        def helper(shape, dim, index, source_shape, alpha, idx_dtype=torch.int32):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            x = cpu_x.detach().clone().to('mps')

            cpu_idx = torch.tensor(index, device='cpu', dtype=idx_dtype)
            idx = cpu_idx.detach().clone().to('mps')

            cpu_source = torch.randn(source_shape, device='cpu', dtype=torch.float, requires_grad=False)
            source = cpu_source.detach().clone().to('mps')

            idx_result = torch.index_add(x, dim=dim, index=idx, source=source, alpha=alpha)
            idx_result_cpu = torch.index_add(cpu_x, dim=dim, index=cpu_idx, source=cpu_source, alpha=alpha)
            self.assertEqual(idx_result, idx_result_cpu)

        helper((2, 8, 4, 5), 0, [0, 1, 0], (3, 8, 4, 5), 5)
        helper((8, 8, 4, 5), 0, [7], (1, 8, 4, 5), 6.0)
        helper((2, 8, 4, 5), 1, [0, 3, 7], (2, 3, 4, 5), 5)
        helper((2, 8, 4, 5), 2, [3, 0], (2, 8, 2, 5), 3.0)
        helper((2, 8, 4, 5), 3, [2, 3, 0], (2, 8, 4, 3), 4)
        helper((2, 3, 3), -1, [1, 2], (2, 3, 2), 6.0)
        # test result dim=1
        helper((2,), 0, [1], (1,), 6.0)
        helper(2, 0, 1, 1, 6)

    # Test flip
    def test_flip(self):
        def helper(shape, dims):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            x = cpu_x.detach().clone().to('mps')

            flip_result = torch.flip(x, dims=dims)
            flip_result_cpu = torch.flip(cpu_x, dims=dims)

            self.assertEqual(flip_result, flip_result_cpu)

        helper((2, 8, 4, 5), [0])
        helper((8, 8, 4, 5), [0, 1])
        helper((2, 8, 4, 5), (0, 1, 2, 3))
        helper((2, 3, 3), (-1,))
        # empty dims
        helper((2, 8, 4, 5), [])
        # input.numel() == 1
        helper((1,), (0,))
        # input.numel() == 0
        helper((0,), (0,))

    # Test index select
    def test_index_select(self):
        def helper(shape, dim, index, idx_dtype=torch.int32):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            x = cpu_x.detach().clone().to('mps')

            cpu_idx = torch.tensor(index, device='cpu', dtype=idx_dtype)
            idx = cpu_idx.detach().clone().to('mps')

            idx_result = torch.index_select(x, dim=dim, index=idx)
            idx_result_cpu = torch.index_select(cpu_x, dim=dim, index=cpu_idx)

            self.assertEqual(idx_result, idx_result_cpu)

        helper((2, 8, 4, 5), 0, [1])
        helper((8, 8, 4, 5), 0, [0, 3, 2, 7, 6])
        helper((2, 8, 4, 5), 1, [0, 3, 2, 7, 6])
        helper((2, 8, 4, 5), 2, [3, 0, 1])
        helper((2, 8, 4, 5), 3, [2, 3, 0])
        helper((2, 3, 3), -1, [1, 2])

    def test_embedding_dense_backward(self):
        def helper(n, d, m, idx):
            embeddingMPS = nn.Embedding(n, d, max_norm=True, device='mps')
            W_MPS = torch.randn((m, d), requires_grad=True, device='mps')
            idx_MPS = torch.tensor(idx).to('mps')
            a_MPS = embeddingMPS.weight.clone() @ W_MPS.t()  # weight must be cloned for this to be differentiable
            a_MPS.retain_grad()
            b_MPS = embeddingMPS(idx_MPS) @ W_MPS.t()  # modifies weight in-place
            b_MPS.retain_grad()
            out_MPS = (a_MPS.unsqueeze(0) + b_MPS)
            loss_MPS = out_MPS.sigmoid().prod()
            loss_MPS.backward()

            embeddingCPU = nn.Embedding(n, d, max_norm=True, scale_grad_by_freq=True)
            W_CPU = W_MPS.to('cpu')
            idx_CPU = torch.tensor(idx)
            a_CPU = embeddingCPU.weight.clone() @ W_CPU.t()  # weight must be cloned for this to be differentiable
            a_CPU.retain_grad()
            b_CPU = embeddingCPU(idx_CPU) @ W_CPU.t()  # modifies weight in-place
            b_CPU.retain_grad()
            out_CPU = (a_CPU.unsqueeze(0) + b_CPU)
            loss_CPU = out_CPU.sigmoid().prod()
            loss_CPU.backward()

            self.assertEqual(b_CPU.grad, b_MPS.grad)
            self.assertEqual(a_CPU.grad, a_MPS.grad)

        helper(3, 5, 7, [0, 1, 2])
        helper(3, 5, 7, 2)  # test scalar index

    # Test pytorch gather
    def test_gather(self):
        def helper(shape, dim, idx_shape, idx_dtype=torch.int64):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            # Indices should be taken from range of axis along which gathering is done
            idx_np = np.random.randint(0, shape[dim], idx_shape)

            cpu_idx = torch.tensor(idx_np, device='cpu', dtype=idx_dtype)
            idx = cpu_idx.detach().clone().to('mps')

            gather_result = torch.gather(x, dim=dim, index=idx)
            gather_result_cpu = torch.gather(cpu_x, dim=dim, index=cpu_idx)

            cpu_grad = torch.randn(idx_shape, device='cpu', dtype=torch.float)
            grad = cpu_grad.to('mps')
            gather_result.backward(gradient=grad)
            gather_result_cpu.backward(gradient=cpu_grad)

            self.assertEqual(gather_result, gather_result_cpu)
            self.assertEqual(cpu_x.grad, x.grad)

        helper((6, 3, 3), 0, (3, 3, 3))
        helper((2, 3, 3, 3), 0, (10, 3, 3, 3))
        helper((2, 8, 4, 5), 0, (10, 8, 4, 5))
        helper((2, 8, 4, 5), 0, (10, 6, 3, 2))
        helper((8, 8, 4, 5), 0, (6, 8, 4, 5))
        helper((8, 8, 4, 5), 0, (6, 7, 2, 3))
        helper((2, 8, 4, 5), 1, (2, 5, 3, 4))
        helper((2, 8, 4, 5), 2, (1, 8, 10, 3))
        helper((2, 8, 4, 5), 3, (2, 5, 3, 12))

    # Test pytorch gather
    def test_gather_scalar(self):
        idx_dtype = torch.int64
        cpu_x = torch.tensor(3, device='cpu', dtype=torch.float, requires_grad=True)
        x = cpu_x.detach().clone().to('mps').requires_grad_()

        idx_np = [0]

        cpu_idx = torch.tensor(idx_np, device='cpu', dtype=idx_dtype)
        idx = cpu_idx.detach().clone().to('mps')

        gather_result = torch.gather(x, dim=0, index=idx)
        gather_result_cpu = torch.gather(cpu_x, dim=0, index=cpu_idx)

        cpu_grad = torch.randn([1], device='cpu', dtype=torch.float)
        grad = cpu_grad.to('mps')
        gather_result.backward(gradient=grad)
        gather_result_cpu.backward(gradient=cpu_grad)

        self.assertEqual(gather_result, gather_result_cpu)
        self.assertEqual(cpu_x.grad, x.grad)

    # Test pytorch scatter_add and scatter
    def test_scatter_add(self):
        def helper(shape, dim, idx_shape, src_shape, idx_dtype=torch.int64, do_add=True):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            cpu_src = torch.randn(src_shape, device='cpu', dtype=torch.float, requires_grad=True)
            src = cpu_src.detach().clone().to('mps').requires_grad_()

            # Indices should be taken from range of axis along which gathering is done
            idx_np = None
            if(do_add):
                idx_np = np.random.randint(0, shape[dim], idx_shape)
            else:
                idx_np = np.array([[0, 1, 2],
                                   [1, 2, 3],
                                   [2, 3, 4],
                                   [3, 4, 5],
                                   [4, 5, 6]])

            cpu_idx = torch.tensor(idx_np, device='cpu', dtype=idx_dtype)
            idx = cpu_idx.detach().clone().to('mps')

            scatter_result = None
            scatter_result_cpu = None

            if(do_add):
                scatter_result = torch.scatter_add(x, dim=dim, index=idx, src=src)
                scatter_result_cpu = torch.scatter_add(cpu_x, dim=dim, index=cpu_idx, src=cpu_src)
            else:
                scatter_result = torch.scatter(x, dim=dim, index=idx, src=src)
                scatter_result_cpu = torch.scatter(cpu_x, dim=dim, index=cpu_idx, src=cpu_src)

            cpu_grad = None
            grad = None

            if(idx_shape == src_shape):
                cpu_grad = torch.randn(shape, device='cpu', dtype=torch.float)
                grad = cpu_grad.to('mps')
                scatter_result.backward(gradient=grad)
                scatter_result_cpu.backward(gradient=cpu_grad)

            self.assertEqual(scatter_result, scatter_result_cpu)
            if(idx_shape == src_shape):
                self.assertEqual(cpu_x.grad, x.grad)
                self.assertEqual(cpu_src.grad, src.grad)

        helper((2, 3), 0, (5, 3), (5, 3))
        helper((2, 8, 4, 5), 0, (10, 8, 4, 5), (10, 8, 4, 5))
        helper((8, 8, 4, 5), 0, (10, 8, 4, 5), (10, 8, 4, 5))
        helper((8, 8, 4, 5), 0, (4, 7, 3, 2), (4, 7, 3, 2))
        helper((8, 8, 4, 5), 0, (4, 6, 3, 2), (4, 7, 3, 2))
        helper((8, 8, 4, 5), 0, (4, 6, 3, 2), (8, 8, 4, 5))

        helper((2, 8, 4, 5), 1, (2, 20, 4, 5), (2, 20, 4, 5))
        helper((2, 8, 4, 5), 1, (2, 13, 3, 2), (2, 13, 3, 2))
        helper((8, 8, 4, 5), 1, (6, 5, 2, 3), (6, 5, 2, 3))
        helper((8, 8, 4, 5), 1, (3, 4, 2, 2), (6, 5, 2, 3))

        helper((4, 5, 9, 8), 2, (4, 5, 13, 8), (4, 5, 13, 8))
        helper((4, 5, 9, 8), 2, (3, 4, 10, 6), (3, 4, 10, 6))
        helper((4, 5, 9, 8), 2, (3, 3, 7, 5), (3, 4, 10, 6))

        # Test scatter src
        helper((8, 3), 0, (5, 3), (5, 3), do_add=False)
        helper((10, 3), 0, (5, 3), (5, 8), do_add=False)

    # Test pytorch scatter_add and scatter for scalar input
    def test_scatter_add_scalar(self):
        def helper(idx_dtype=torch.int64, do_add=True):
            cpu_x = torch.tensor(2, device='cpu', dtype=torch.float, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            cpu_src = torch.tensor(3, device='cpu', dtype=torch.float, requires_grad=True)
            src = cpu_src.detach().clone().to('mps').requires_grad_()

            # Indices should be taken from range of axis along which gathering is done
            idx_np = [0]

            cpu_idx = torch.tensor(idx_np, device='cpu', dtype=idx_dtype)
            idx = cpu_idx.detach().clone().to('mps')

            scatter_result = None
            scatter_result_cpu = None

            if(do_add):
                scatter_result = torch.scatter_add(x, dim=0, index=idx, src=src)
                scatter_result_cpu = torch.scatter_add(cpu_x, dim=0, index=cpu_idx, src=cpu_src)
            else:
                scatter_result = torch.scatter(x, dim=0, index=idx, src=src)
                scatter_result_cpu = torch.scatter(cpu_x, dim=0, index=cpu_idx, src=cpu_src)

            cpu_grad = None
            grad = None

            cpu_grad = torch.tensor(1.2, device='cpu', dtype=torch.float)
            grad = cpu_grad.to('mps')
            scatter_result.backward(gradient=grad)
            scatter_result_cpu.backward(gradient=cpu_grad)

            self.assertEqual(scatter_result, scatter_result_cpu)
            self.assertEqual(cpu_x.grad, x.grad)
            self.assertEqual(cpu_src.grad, src.grad)

        helper()
        helper(do_add=False)

    # Test pytorch scatter_reduce
    def test_scatter_reduce(self):
        def helper(shape, dim, idx_shape, src_shape, idx_dtype=torch.int64, reduce_str="sum"):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            cpu_src = torch.randn(src_shape, device='cpu', dtype=torch.float, requires_grad=True)
            src = cpu_src.detach().clone().to('mps').requires_grad_()

            # Indices should be taken from range of axis along which gathering is done
            idx_np = np.random.randint(0, shape[dim], idx_shape)

            cpu_idx = torch.tensor(idx_np, device='cpu', dtype=idx_dtype)
            idx = cpu_idx.detach().clone().to('mps')

            scatter_result = torch.scatter(x, dim=dim, index=idx, src=src, reduce=reduce_str)
            scatter_result_cpu = torch.scatter(cpu_x, dim=dim, index=cpu_idx, src=cpu_src, reduce=reduce_str)

            self.assertEqual(scatter_result, scatter_result_cpu)

        # for reduce in ["sum", "prod", "amax", "amin"]:
        for reduce in ["add", "multiply"]:
            helper((2, 3), 0, (5, 3), (5, 3), reduce_str=reduce)
            helper((2, 8, 4, 5), 0, (10, 8, 4, 5), (10, 8, 4, 5), reduce_str=reduce)
            helper((8, 8, 4, 5), 0, (10, 8, 4, 5), (10, 8, 4, 5), reduce_str=reduce)
            helper((8, 8, 4, 5), 0, (4, 7, 3, 2), (4, 7, 3, 2), reduce_str=reduce)
            helper((8, 8, 4, 5), 0, (4, 6, 3, 2), (4, 7, 3, 2), reduce_str=reduce)
            helper((8, 8, 4, 5), 0, (4, 6, 3, 2), (8, 8, 4, 5), reduce_str=reduce)

            helper((2, 8, 4, 5), 1, (2, 20, 4, 5), (2, 20, 4, 5), reduce_str=reduce)
            helper((2, 8, 4, 5), 1, (2, 13, 3, 2), (2, 13, 3, 2), reduce_str=reduce)
            helper((8, 8, 4, 5), 1, (6, 5, 2, 3), (6, 5, 2, 3), reduce_str=reduce)
            helper((8, 8, 4, 5), 1, (3, 4, 2, 2), (6, 5, 2, 3), reduce_str=reduce)

            helper((4, 5, 9, 8), 2, (4, 5, 13, 8), (4, 5, 13, 8), reduce_str=reduce)
            helper((4, 5, 9, 8), 2, (3, 4, 10, 6), (3, 4, 10, 6), reduce_str=reduce)
            helper((4, 5, 9, 8), 2, (3, 3, 7, 5), (3, 4, 10, 6), reduce_str=reduce)

    def test_is_nonzero(self):
        self.assertFalse(torch.is_nonzero(torch.tensor([0.]).to('mps')))
        self.assertTrue(torch.is_nonzero(torch.tensor([1.5]).to('mps')))
        self.assertFalse(torch.is_nonzero(torch.tensor([False]).to('mps')))
        self.assertTrue(torch.is_nonzero(torch.tensor([3]).to('mps')))

    # Test triu
    def test_triu(self):
        def helper(shape, diag=0):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            triu_result = torch.triu(x, diag)
            triu_result_cpu = torch.triu(cpu_x, diag)

            cpu_grad = torch.randn(triu_result_cpu.shape)
            grad = cpu_grad.to('mps')

            triu_result.backward(gradient=grad)
            triu_result_cpu.backward(gradient=cpu_grad)

            self.assertEqual(triu_result, triu_result_cpu)
            self.assertEqual(x.grad, cpu_x.grad)

        helper((2, 8, 4, 5))
        helper((2, 8, 4, 5), diag=1)
        helper((2, 8, 4, 5), diag=2)
        helper((2, 8, 4, 5), diag=3)
        helper((2, 8, 4, 5), diag=-1)
        helper((2, 8, 4, 5), diag=-2)
        helper((2, 8, 4, 5), diag=-3)

    # Test tril
    def test_tril(self):
        def helper(shape, diag=0):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            tril_result = torch.tril(x, diag)
            tril_result_cpu = torch.tril(cpu_x, diag)

            cpu_grad = torch.randn(tril_result_cpu.shape)
            grad = cpu_grad.to('mps')

            tril_result.backward(gradient=grad)
            tril_result_cpu.backward(gradient=cpu_grad)

            self.assertEqual(tril_result, tril_result_cpu)
            self.assertEqual(x.grad, cpu_x.grad)

        helper((2, 8, 4, 5))
        helper((2, 8, 4, 5), diag=1)
        helper((2, 8, 4, 5), diag=2)
        helper((2, 8, 4, 5), diag=3)
        helper((2, 8, 4, 5), diag=-1)
        helper((2, 8, 4, 5), diag=-2)
        helper((2, 8, 4, 5), diag=-3)

    # test eye
    def test_eye(self):
        def helper(n, m, dtype):
            cpu_result = None
            result = None

            if(n == m):
                cpu_result = torch.eye(n, dtype=dtype, device='cpu')
                result = torch.eye(n, dtype=dtype, device='mps')
            else:
                cpu_result = torch.eye(n, m, device='cpu')
                result = torch.eye(n, m, device='mps')

            self.assertEqual(result, cpu_result)

        for dtype in [torch.float32, torch.int32, torch.int64]:
            helper(2, 2, dtype)
            helper(2, 3, dtype)
            helper(0, 2, dtype)
            helper(0, 0, dtype)
            helper(3, 8, dtype)
            helper(8, 3, dtype)

    # Test diag
    def test_diag(self):
        def helper(shape, diag=0):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            diag_result = torch.diag(x, diag)
            diag_result_cpu = torch.diag(cpu_x, diag)

            # cpu_grad = torch.randn(diag_result_cpu.shape)
            # grad = cpu_grad.to('mps')

            # diag_result.backward(gradient=grad)
            # diag_result_cpu.backward(gradient=cpu_grad)

            self.assertEqual(diag_result, diag_result_cpu)
            # self.assertEqual(x.grad, cpu_x.grad)

        for shape in [(5, 5), (5, 6), (6, 5), (5,), (6,)]:
            for diag in [0, 1, 2, 3, 4, -1, -2, -3, -4]:
                helper(shape, diag=diag)

    # Test linspace
    def test_linspace(self):
        def helper(start, end, steps, dtype=torch.float32):
            cpu_result = torch.tensor(np.linspace(start, end, steps), dtype=dtype)
            result = torch.linspace(start, end, steps, dtype=dtype, device='mps')
            self.assertEqual(cpu_result, result)

        for dtype in [torch.float32, torch.int32, torch.uint8, torch.int64]:
            helper(2, 5, 10, dtype)
            helper(2, 2, 10, dtype)
            helper(5, 2, 10, dtype)
            helper(2, 2, 0, dtype)

    # Test argange
    def test_arange(self):
        self.assertEqual(np.arange(10), torch.arange(10, device='mps'))
        self.assertEqual(np.arange(7, 1, -1), torch.arange(7, 1, -1, device='mps'))
        self.assertEqual(np.arange(1, 2, .3, dtype=np.float32), torch.arange(1, 2, .3, device='mps'))
        self.assertEqual(np.arange(6.3, dtype=np.float32), torch.arange(6.3, device='mps'))

    # Test softmax
    def test_softmax(self):
        def helper(shape, dim, channels_last=False):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            if(channels_last):
                cpu_x = cpu_x.to(memory_format=torch.channels_last)
                cpu_x.retain_grad()
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            softmax_result = torch.nn.functional.softmax(x, dim=dim)
            softmax_result_cpu = torch.nn.functional.softmax(cpu_x, dim=dim)

            # Currently NOT testing backward for channels last backward
            cpu_grad = None
            grad = None

            if(not channels_last):
                cpu_grad = torch.randn(shape, device='cpu', dtype=torch.float)
                grad = cpu_grad.to('mps')

                softmax_result.backward(gradient=grad)
                softmax_result_cpu.backward(gradient=cpu_grad)

            self.assertEqual(softmax_result, softmax_result_cpu)
            if(not channels_last):
                self.assertEqual(x.grad, cpu_x.grad)

        def helper2(dim):
            cpu_x = torch.tensor(1.23, device='cpu', dtype=torch.float, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            softmax_result = torch.nn.functional.softmax(x, dim=dim)
            softmax_result_cpu = torch.nn.functional.softmax(cpu_x, dim=dim)

            cpu_grad = torch.tensor(2.34, device='cpu', dtype=torch.float)
            grad = cpu_grad.to('mps')

            softmax_result.backward(gradient=grad)
            softmax_result_cpu.backward(gradient=cpu_grad)

            self.assertEqual(softmax_result, softmax_result_cpu)
            self.assertEqual(x.grad, cpu_x.grad)

        helper2(0)

        for channels_last in [False]:
            for shape in [(2, 4, 8, 5), (3, 4, 6, 7, 2)]:
                if(len(shape) != 4 and channels_last):
                    continue
                for dim in [0, 1, 2, 3, -1, -2, -3]:
                    helper(shape, dim, channels_last)

    # Test sub
    def test_sub(self):
        def helper(shape, alpha):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            x = cpu_x.detach().clone().to('mps')

            cpu_y = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            y = cpu_y.detach().clone().to('mps')

            cpu_out = torch.sub(cpu_x, cpu_y, alpha=alpha)
            out = torch.sub(x, y, alpha=alpha)

            self.assertEqual(out, cpu_out)

        helper((2, 8, 4, 5), 0.1)
        helper((2, 8, 3, 5), 0.1)
        helper((2, 8, 3, 5), 0.2)

    # Test where
    def test_where(self):
        def helper(shape, x_shape, y_shape, cond_dtype=torch.bool, x_dtype=torch.float):

            cpu_cond = torch.randint(2, shape, device='cpu', dtype=cond_dtype, requires_grad=False)
            cond = cpu_cond.detach().clone().to('mps')

            cpu_x = torch.randn(x_shape, device='cpu', dtype=x_dtype, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            cpu_y = torch.randn(y_shape, device='cpu', dtype=x_dtype, requires_grad=True)
            y = cpu_y.detach().clone().to('mps').requires_grad_()

            cpu_out = torch.where(cpu_cond, cpu_x, cpu_y)
            out = torch.where(cond, x, y)

            cpu_grad = torch.randn(cpu_out.shape)
            grad = cpu_grad.to('mps')

            cpu_out.backward(gradient=cpu_grad)
            out.backward(gradient=grad)

            self.assertEqual(out, cpu_out)
            self.assertEqual(x.grad, cpu_x.grad)
            self.assertEqual(y.grad, cpu_y.grad)

        for shape in ([(0, 3), [], (2, 3), (9,)]):
            helper(shape, shape, shape)

        helper((2, 3, 1), (2, 3, 4), (2, 1, 4))
        helper((2, 1, 1), (2, 3, 4), (1, 3, 4))
        helper((1, 1, 1), (1, 1, 4), (2, 3, 1))
        helper([], (1, 1, 4), (2, 3, 1))
        helper([], (2, 3, 4), [])
        helper((5, 2, 3), (2, 3), (2, 3))
        helper((2, 3), (5, 2, 3), (2, 3))
        helper((2, 3), (2, 3), (5, 2, 3))
        helper((2, 3), (5, 2, 3), (6, 5, 2, 3))

    # Test normal
    def test_normal(self):
        def helper(shape, mean=0.0, std=1.0):
            mps_out = torch.normal(mean, std, shape, device='mps')

            mean_array = np.ones(shape)
            mean_array *= mean
            cpu_mean_tensor = torch.tensor(mean_array, device='cpu', dtype=torch.float, requires_grad=False)
            mean_tensor = cpu_mean_tensor.detach().clone().to('mps')

            std_array = np.ones(shape)
            std_array *= std
            cpu_std_tensor = torch.tensor(std_array, device='cpu', dtype=torch.float, requires_grad=False)
            std_tensor = cpu_std_tensor.detach().clone().to('mps')

            # test out
            mps_out = torch.zeros(shape, device='mps')
            torch.normal(mean_tensor, std, out=mps_out)

            mps_out = torch.zeros(shape, device='mps')
            torch.normal(mean, std_tensor, out=mps_out)

            mps_out = torch.zeros(shape, device='mps')
            torch.normal(mean_tensor, std_tensor, out=mps_out)

            # test without out
            mps_out = torch.normal(mean_tensor, std)
            self.assertEqual(mps_out.size(), mean_tensor.size())

            mps_out = torch.normal(mean, std_tensor)
            self.assertEqual(mps_out.size(), std_tensor.size())

            inferred_shape = torch.broadcast_shapes(mean_tensor.size(), std_tensor.size())
            mps_out = torch.normal(mean_tensor, std_tensor)
            self.assertEqual(mps_out.size(), inferred_shape)

        helper((2, 3, 4, 5, 6))
        helper((100, 100), 2.5, 1.2)

    def test_bernoulli(self):
        shape = (10, 10)
        all_ones = torch.ones(shape, device='mps')
        all_zeros = torch.zeros(shape, device='mps')

        prob_tensor = all_ones * 0.5
        # probability of drawing "1" is 0.5
        mps_out = torch.bernoulli(prob_tensor)
        # We can't check reliably the mean and std.
        # Just make sure we don't return constant values
        self.assertNotEqual(mps_out.to('cpu').mean(), 0.)
        self.assertNotEqual(mps_out.to('cpu').std() ** 2, 0.)

        # probability of drawing "1" is 0
        mps_out = torch.bernoulli(all_zeros)
        self.assertEqual(mps_out, all_zeros)

        # probability of drawing "1" is 1
        mps_out = torch.bernoulli(all_ones)
        self.assertEqual(mps_out, all_ones)

    # Test random_.to and random_.from
    def test_random(self):
        def helper(shape, low, high, dtype=torch.int32):

            mps_out = torch.randint(low, high, shape, dtype=dtype, device='mps')

            # We can't check reliably the mean and std.
            # Just make sure we don't return constant values
            self.assertNotEqual(mps_out.to('cpu').float().mean(), 0.)
            self.assertNotEqual(mps_out.to('cpu').float().std(), 0.)

        helper([100, 100], 0, 10)
        helper([100, 100], 23, 89)
        helper([100, 100], 23, 89, dtype=torch.float32)
        helper([100, 100], 23, 89, dtype=torch.int64)
        helper([100, 100], 0, 2, dtype=torch.bool)

    # Test exponential
    def test_exponential(self):
        def helper(shape, lamda, dtype=torch.float32):

            mps_out = torch.zeros(shape, device='mps', dtype=dtype)
            mps_out.exponential_(lamda)

            print(mps_out.to('cpu').float().mean(), 1 / lamda)
            print(mps_out.to('cpu').float().std() ** 2, 1 / (lamda**2))

        for dtype in [torch.float32, torch.float16]:
            helper([100, 100], 2, dtype)
            helper([100, 100], 1, dtype)
            helper([100, 100], 3, dtype)
            helper([100, 100], 0.5, dtype)

    def test_exponential_1(self):
        rate = torch.randn(5, 5).abs().requires_grad_()
        rate_1d = torch.randn(1).abs().requires_grad_()
        self.assertEqual(Exponential(rate).sample().size(), (5, 5))
        self.assertEqual(Exponential(rate).sample((7,)).size(), (7, 5, 5))
        self.assertEqual(Exponential(rate_1d).sample((1,)).size(), (1, 1))
        self.assertEqual(Exponential(rate_1d).sample().size(), (1,))
        self.assertEqual(Exponential(0.2).sample((1,)).size(), (1,))
        self.assertEqual(Exponential(50.0).sample((1,)).size(), (1,))

    # Test add
    def test_add_binary_op(self):
        def helper(shape, alpha):
            for dtype in [torch.float16, torch.float32]:
                cpu_x = torch.randn(shape, device='cpu', dtype=dtype, requires_grad=False)
                mps_x = cpu_x.detach().clone().to('mps')

                cpu_y = torch.randn(shape, device='cpu', dtype=dtype, requires_grad=False)
                mps_y = cpu_y.detach().clone().to('mps')

                cpu_out = torch.add(cpu_x, cpu_y, alpha=alpha)
                mps_out = torch.add(mps_x, mps_y, alpha=alpha)
                # fp16 isn't accurate when alpha is passed
                # TODO: remove or fix 'tol' when we fix problems with fp16
                tol = 1e-3 if dtype is torch.float16 else None
                self.assertEqual(mps_out, cpu_out, rtol=tol, atol=tol)
                # create a scalar tensor
                cpu_s = torch.tensor(2.3, device='cpu', dtype=dtype, requires_grad=False)
                mps_s = cpu_s.detach().clone().to('mps')
                # primary tensor is scalar
                self.assertEqual(torch.add(cpu_s, cpu_y), torch.add(mps_s, mps_y))
                # secondary tensor is scalar
                self.assertEqual(torch.add(cpu_x, cpu_s), torch.add(mps_x, mps_s))

        helper((2, 8, 4, 5), 1.0)
        helper((2, 8, 4, 5), 0.0)
        helper((2, 8, 4, 5), 0.1)
        helper((2, 8, 3, 5), 0.1)
        helper((2, 8, 3, 5), 0.2)

    # Test add
    def test_add_scalars(self):
        def helper(alpha):
            for dtype in [torch.float16, torch.float32]:
                cpu_x = torch.tensor(2.3, device='cpu', dtype=dtype, requires_grad=False)
                x = cpu_x.detach().clone().to('mps')

                cpu_y = torch.tensor(3.4, device='cpu', dtype=dtype, requires_grad=False)
                y = cpu_y.detach().clone().to('mps')

                cpu_out = torch.add(cpu_x, cpu_y, alpha=alpha)
                out = torch.add(x, y, alpha=alpha)
                # fp16 isn't accurate when alpha is passed
                tol = 1e-3 if dtype is torch.float16 else None
                self.assertEqual(out, cpu_out, rtol=tol, atol=tol)

        helper(1.0)
        helper(0.0)
        helper(0.1)
        helper(0.2)

        # Test int32 tensor + int64 scalar add
        # see https://github.com/pytorch/pytorch/issues/79835#issuecomment-1164984534
        x = torch.ones(4, dtype=torch.int32, device='mps')
        self.assertEqual(x + 1, torch.full((4,), 2, dtype=torch.int32, device='mps'))
        self.assertTrue(torch.equal(x + 1.5, torch.full((4,), 2.5, device='mps')))

    def test_types_binary_op(self):
        # Float * Bool
        cpu_x = torch.arange(5, dtype=torch.float32, device="cpu") * torch.tensor([True, False, True, False, True], device="cpu")
        mps_x = torch.arange(5, dtype=torch.float32, device="mps") * torch.tensor([True, False, True, False, True], device="mps")
        self.assertEqual(cpu_x, mps_x)
        # Float * Int64
        cpu_y = torch.arange(5, dtype=torch.float32, device="cpu") * torch.tensor([1, 0, 1, 0, 1], device="cpu")
        mps_y = torch.arange(5, dtype=torch.float32, device="mps") * torch.tensor([1, 0, 1, 0, 1], device="mps")
        self.assertEqual(cpu_y, mps_y)

    def test_unary_ops(self):
        def helper(shape, op):
            for dtypef in [torch.float32]:
                cpu_x = torch.randn(shape, device='cpu', dtype=dtypef, requires_grad=False)
                mps_x = cpu_x.detach().clone().to('mps')
                self.assertEqual(op(cpu_x), op(mps_x))

            for dtypei in [torch.int32, torch.int16]:
                cpu_x = torch.randint(0, 1000, shape, device='cpu', dtype=dtypei, requires_grad=False)
                mps_x = cpu_x.to('mps')
                self.assertEqual(op(cpu_x), op(mps_x), rtol=1e-4, atol=1e-4)

        helper((2, 8, 4, 5), torch.exp)
        helper((2, 8, 3, 5), torch.exp2)
        helper((2, 8, 3, 5), torch.expm1)
        helper((2, 8, 3, 5), torch.log)
        helper((2, 8, 3, 5), torch.cos)

    def test_atan2(self):
        def helper(shape):
            input_cpu = torch.randn(shape)
            input_mps = input_cpu.detach().clone().to("mps")

            other_cpu = torch.randn(shape)
            other_mps = other_cpu.detach().clone().to("mps")

            atan2_cpu = torch.atan2(input_cpu, other_cpu)
            atan2_mps = torch.atan2(input_mps, other_mps)

            self.assertEqual(atan2_cpu, atan2_mps.to("cpu"))

        helper(4)
        helper(10000)
        helper((10000, 40))

    def test_multinomial(self):
        # Test with num_dist = 1
        def helper(probs, compare_mean, compare_var, num_samples=5, replacement=True):
            cpu_prob_tensor = torch.tensor(probs, device='cpu', dtype=torch.float, requires_grad=False)
            prob_tensor = cpu_prob_tensor.detach().clone().to('mps')

            mps_out = torch.multinomial(prob_tensor, num_samples, replacement=replacement)
            if(not replacement):
                print(mps_out.to('cpu'))
            else:
                # Compare "real" with theoretical values
                print(mps_out.to('cpu').float().mean(), compare_mean)
                print(mps_out.to('cpu').float().std() ** 2, compare_var)

        # TODO: Add tests for data types
        helper(np.array([[0., 0., 0., 0.5, 0.5]]), (3 + 4) / 2, (12.5 - 3.5 ** 2), 100000)
        helper(np.array([[.2, .2, .2, .2, .2]]), (0 + 1 + 2 + 3 + 4) / 5, (6 - 2 * 2), 10000)
        helper(np.array([[1, 1, 1, 1, 1]]), (0 + 1 + 2 + 3 + 4) / 5, (6 - 2 * 2), 10000)
        helper(np.array([1, 1, 1, 1, 1]), (0 + 1 + 2 + 3 + 4) / 5, (6 - 2 * 2), 10000)
        helper(np.array([[1, 1, 1, 1, 1, 1, 1]]), 0, 0, 7, False)

class TestNNMPS(NNTestCase):

    def _create_basic_net(self):
        class Layer(nn.Module):
            def __init__(self):
                super(Layer, self).__init__()
                self.layer_dummy_param = Parameter(torch.empty(3, 5))
                self.register_buffer('layer_dummy_buf', torch.zeros(1, 3, 3, 7))

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.l1 = Layer()
                self.dummy_param = Parameter(torch.empty(3, 5))
                self.register_buffer('dummy_buf', torch.zeros(7, 3, 3, 1))

        l = Layer()
        n = Net()
        s = nn.Sequential(n, n)

        return l, n, s

    def test_requires_grad_(self):
        m = self._create_basic_net()[-1]
        assert len(list(m.buffers())) > 0, 'invalid test'
        assert all(not b.requires_grad for b in m.buffers()) > 0, 'invalid test'
        assert len(list(m.parameters())) > 0, 'invalid test'
        assert all(p.requires_grad for p in m.parameters()) > 0, 'invalid test'
        for requires_grad in (False, True):
            self.assertIs(m.requires_grad_(requires_grad), m)
            for p in m.parameters():
                self.assertEqual(p.requires_grad, requires_grad)
            for b in m.buffers():
                self.assertFalse(b.requires_grad)

    def test_module_backcompat(self):
        from torch.serialization import SourceChangeWarning
        path = download_file('https://download.pytorch.org/test_data/linear.pt')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', SourceChangeWarning)
            m = torch.load(path)
        input = torch.randn(2, 3, dtype=torch.float)
        self.assertEqual(m(input).size(), (2, 5))

    def test_conv_backcompat(self):
        from torch.serialization import SourceChangeWarning
        # This file was generated by running on PyTorch 1.0.1 on Python 2:
        #
        #     import torch
        #     from torch import nn
        #     m = nn.Conv2d(1, 1, 1)
        #     torch.save(m, 'legacy_conv2d.pt')
        #
        # NB: This Pickle also contains some Unicode data!
        path = download_file('https://download.pytorch.org/test_data/legacy_conv2d.pt')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', SourceChangeWarning)
            m = torch.load(path, encoding='utf-8')
        input = torch.randn((1, 1, 1, 1), dtype=torch.float)
        self.assertEqual(m(input).size(), (1, 1, 1, 1))

    def test_conv_expand(self):
        device = 'mps'
        input_ = torch.rand(2, 3, 16, 16, device=device)
        kernel = torch.rand(1, 1, 3, 11, device=device)
        tmp_kernel = kernel.expand(-1, 3, -1, -1)
        output = F.conv2d(input_, tmp_kernel, groups=1, padding=0, stride=1)

    # The test should not crash
    def test_permute(self):
        X = torch.randn(5, 5).to('mps')
        torch.log(X)
        X = X.permute(1, 0)
        torch.log(X)

    # Printing of non_contiguous should not crash
    def test_print_non_contiguous(self):
        print(torch.ones(100, 100, device='mps').nonzero())
        print(torch.ones(100, 100, device='mps').nonzero().contiguous())

    def test_zero_grad(self):
        i = torch.randn(2, 5, requires_grad=True)
        module = nn.Linear(5, 5)
        for p in module.parameters():
            p.requires_grad = False
        module.zero_grad()

        module.weight.requires_grad = True
        module.zero_grad()
        self.assertIsNone(module.weight.grad)  # uninitialized grad

        module(i).sum().backward()
        self.assertIsNotNone(module.weight.grad)
        self.assertGreater(module.weight.grad.data.abs().sum(), 0)
        module.zero_grad()
        self.assertEqual(module.weight.grad.data, module.weight.data.clone().zero_())

        module.bias.requires_grad = True
        module.zero_grad()
        self.assertIsNotNone(module.weight.grad)
        self.assertIsNone(module.bias.grad)
        module(i).sum().backward()
        self.assertIsNotNone(module.weight.grad)
        self.assertIsNotNone(module.bias.grad)
        self.assertGreater(module.weight.grad.data.abs().sum(), 0)
        self.assertGreater(module.bias.grad.data.abs().sum(), 0)
        module.zero_grad()
        self.assertEqual(module.weight.grad.data, module.weight.data.clone().zero_())
        self.assertEqual(module.bias.grad.data, module.bias.data.clone().zero_())

        # Force set to None.
        module.zero_grad(set_to_none=True)
        self.assertIsNone(module.weight.grad)

    def test_no_grad(self):
        for dtype in [torch.bfloat16, torch.float, torch.double]:
            module = nn.Conv2d(2, 5, kernel_size=3, padding=1).to(dtype)
            input = torch.randn(1, 2, 10, 10).to(dtype)
            x = input
            y = input.clone()

            output = module(x)
            self.assertTrue(output.requires_grad)
            output.backward(torch.ones(1, 5, 10, 10))

            with torch.no_grad():
                output2 = module(y)
                self.assertFalse(output2.requires_grad)
                self.assertRaises(RuntimeError, lambda: output2.backward(torch.ones(1, 5, 10, 10)))

    def test_invalid_conv1d(self):
        for dtype in [torch.bfloat16, torch.float, torch.double]:
            module = nn.Conv1d(in_channels=3, out_channels=33, kernel_size=10, stride=1, bias=True).to(dtype)
            input = torch.randn(1, 3, 4).to(dtype)
            with self.assertRaisesRegex(RuntimeError,
                                        r'Calculated padded input size per channel: \(4\). ' +
                                        r'Kernel size: \(10\). Kernel size can\'t be greater than actual input size'):
                module(input)

            # Negative stride check
            module = nn.Conv1d(in_channels=3, out_channels=6, kernel_size=3, stride=-1, bias=True).to(dtype)
            input = torch.randn(1, 3, 4).to(dtype)
            with self.assertRaisesRegex(RuntimeError, 'non-positive stride is not supported'):
                module(input)

    def test_conv2d_discontiguous_weight(self):
        # Test for https://github.com/pytorch/pytorch/issues/55781
        x = torch.ones(64, 16, 16, 16)
        weight = torch.arange(0, 1.0, 1 / 2.0 ** 10).reshape(32, 16, 1, 2)[:, :, :, ::2]
        self.assertFalse(weight.is_contiguous())
        y = torch.nn.functional.conv2d(x, weight, None)
        if torch.backends.mkldnn.is_available():
            # Disable MKLDNN explicitly, so that either NNPACK or THCNN will be used
            with torch.backends.mkldnn.flags(enabled=False):
                y_ = torch.nn.functional.conv2d(x, weight, None)
                self.assertEqual(y, y_)
        self.assertEqual(y.sum(), 4186112.)

    def test_invalid_conv2d(self):
        for dtype in [torch.bfloat16, torch.float, torch.double]:
            module = torch.nn.Conv2d(1, 1, kernel_size=3, dilation=2, stride=2).to(dtype)
            input = torch.empty(1, 1, 4, 4).to(dtype)
            self.assertRaises(RuntimeError, lambda: module(input))

            module = nn.Conv2d(in_channels=3, out_channels=33, kernel_size=10, stride=1, bias=True)
            input = torch.randn(1, 3, 1, 1)
            with self.assertRaisesRegex(RuntimeError,
                                        r'Calculated padded input size per channel: \(1 x 1\). ' +
                                        r'Kernel size: \(10 x 10\). Kernel size can\'t be greater than actual input size'):
                module(input)

            # Negative stride check
            module = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=4, stride=-1, bias=True).to(dtype)
            input = torch.randn(1, 3, 4, 4).to(dtype)
            with self.assertRaisesRegex(RuntimeError, 'non-positive stride is not supported'):
                module(input)

            # Zero stride check
            module = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=4, stride=0, bias=True).to(dtype)
            input = torch.randn(1, 3, 4, 4).to(dtype)
            with self.assertRaisesRegex(RuntimeError, 'non-positive stride is not supported'):
                module(input)

            # Input and weights on different devices
            self.assertRaisesRegex(RuntimeError,
                                   'must be on the same device',
                                   lambda: torch.conv2d(torch.rand(1, 3, 32, 32), torch.rand(1, 3, 3, 3, device='mps')))
            self.assertRaisesRegex(RuntimeError,
                                   'Input type \\(MPSFloatType\\) and weight type \\(torch\\.FloatTensor\\) should be the same',
                                   lambda: torch.conv2d(torch.rand(1, 3, 32, 32, device='mps'), torch.rand(1, 3, 3, 3)))


    def test_conv2d_valid_padding(self, device='mps'):
        # Test F.conv2d padding='valid' is the same as no padding
        x = torch.rand(1, 1, 1, 10, device=device).to(torch.float)
        y = torch.rand(1, 1, 1, 4, device=device).to(torch.float)

        expect = F.conv2d(x, y)
        actual = F.conv2d(x, y, padding='valid')
        self.assertEqual(expect.to('cpu'), actual.to('cpu'))

    def test_gemm_permute_transpose(self):
        batch_size = 32
        n = 20
        hidden = 768
        num_attention_heads = 12
        attention_head_size = hidden // num_attention_heads

        def transpose_for_scores(x: torch.Tensor) -> torch.Tensor:
            new_x_shape = x.size()[:-1] + (num_attention_heads, attention_head_size)
            x = x.view(new_x_shape)
            return x.permute(0, 2, 1, 3)

        def attention2(key, *, workaround=False, device):
            key = transpose_for_scores(key)
            res = key.transpose(-1, -2)
            return res

        A = torch.randn(batch_size, n, hidden)
        A_mps = A.detach().clone().to("mps")

        r1 = attention2(A, device="cpu")
        r2 = attention2(A_mps, device="mps")

        r2_cpu = r2.to("cpu")
        self.assertEqual(r1, r2_cpu)

    # def test_conv2d_same_padding(self, device='mps'):
        # x = torch.rand(1, 1, 10, 11, device=device)
        # y = torch.rand(1, 1, 4, 5, device=device)
        # expect = F.conv2d(x, y, padding=(2, 2))[..., 1:, :]
        # actual = F.conv2d(x, y, padding='same')
        # self.assertEqual(expect.to('cpu'), actual.to('cpu'))

        # # With dilation
        # y = torch.rand(1, 1, 3, 4, device=device)
        # expect = F.conv2d(x, y, padding=(2, 3), dilation=2)
        # actual = F.conv2d(x, y, padding='same', dilation=2)
        # self.assertEqual(expect, actual)

        # # Dilation with asymmetric padding
        # y = torch.rand(1, 1, 4, 4, device=device)
        # expect = F.conv2d(x, y, padding=5, dilation=3)[..., 1:, 1:]
        # actual = F.conv2d(x, y, padding='same', dilation=3)
        # self.assertEqual(expect, actual)


class TestConstantPadNd(TestCase):
    def test_preserves_memory_format(self):
        nchw_tensor = torch.rand((1, 2, 5, 3))
        nchw_padded = torch.constant_pad_nd(nchw_tensor, [1, 2], 0.5)
        self.assertTrue(nchw_padded.is_contiguous(memory_format=torch.contiguous_format))

        nhwc_tensor = nchw_tensor.contiguous(memory_format=torch.channels_last)
        nhwc_padded = torch.constant_pad_nd(nhwc_tensor, [1, 2], 0.5)
        self.assertTrue(nhwc_padded.is_contiguous(memory_format=torch.channels_last))


class TestLinalgMPS(TestCase):
    def _test_addmm_addmv(self, f, t, m, v, *, alpha=None, beta=None, transpose_out=False):
        dtype = t.dtype
        numpy_dtype = dtype
        alpha = 1.2 if alpha is None else alpha
        beta = 0.8 if beta is None else beta
        res1 = f(t, m, v, alpha=alpha, beta=beta)
        res2 = torch.full_like(res1, math.nan)
        if transpose_out:
            res2 = res2.t().clone(memory_format=torch.contiguous_format).t()
        f(t, m, v, alpha=alpha, beta=beta, out=res2)
        res3 = alpha * (m.to(numpy_dtype).cpu().numpy() @ v.to(numpy_dtype).cpu().numpy())
        if beta != 0:
            res3 += (torch.mul(t, beta)).to(numpy_dtype).cpu().numpy()
        res3 = torch.from_numpy(res3).to(dtype)
        self.assertEqual(res1, res2)
        self.assertEqual(res1, res3)

    def test_addmm(self, device="mps", dtype=torch.float32):
        M = torch.randn(10, 25, device=device).to(dtype)
        m1 = torch.randn(10, 50, device=device).to(dtype)
        m2 = torch.randn(50, 25, device=device).to(dtype)
        self._test_addmm_addmv(torch.addmm, M, m1, m2)

        # Test beta=0, M=nan
        M = torch.full((10, 25), math.nan, device=device).to(dtype)
        m1 = torch.randn(10, 50, device=device).to(dtype)
        m2 = torch.randn(50, 25, device=device).to(dtype)
        self._test_addmm_addmv(torch.addmm, M, m1, m2, beta=0)

        # Test transpose
        for t1, t2, t3, t4 in itertools.product([True, False], repeat=4):
            def maybe_transpose(cond, m):
                if not cond:
                    return m
                return m.t().clone(memory_format=torch.contiguous_format).t()

        M = maybe_transpose(t1, torch.randn(10, 25, device=device).to(dtype))
        m1 = maybe_transpose(t2, torch.randn(10, 50, device=device).to(dtype))
        m2 = maybe_transpose(t3, torch.randn(50, 25, device=device).to(dtype))
        self._test_addmm_addmv(torch.addmm, M, m1, m2, transpose_out=t4)

class TestGatherScatter(TestCase):
    def test_slicing_with_step(self):
        # Slicing with step
        # https://github.com/pytorch/pytorch/issues/78886
        x_mps = torch.zeros(10, dtype=torch.float32, device="mps")
        x_mps[::2] = 1.0

        x_cpu = torch.zeros(10, dtype=torch.float32, device="cpu")
        x_cpu[::2] = 1.0

        self.assertEqual(x_cpu, x_mps)

    def test_cast_gather_scatter(self):
        for _ in range(0, 50):
            input = np.random.randint(0, 255, size=(5, 5, 4), dtype=np.uint8)
            with torch.no_grad():
                s = torch.tensor(input, dtype=torch.uint8, device="mps").unsqueeze(0)
                s_cpu = torch.tensor(input, dtype=torch.uint8, device="cpu").unsqueeze(0)
                s = s.long()
                s_cpu = s_cpu.long()
                self.assertEqual(s.cpu(), s_cpu)

                s = s.float()
                s_cpu = s_cpu.float()
                self.assertEqual(s.cpu(), s_cpu)

                s /= 255
                s_cpu /= 255
                self.assertEqual(s.cpu(), s_cpu)

    def test_slicing_replace_column(self):
        # https://github.com/pytorch/pytorch/issues/78074
        def _helper(tensor_data):
            x_cpu = torch.tensor(tensor_data)
            x_mps = x_cpu.to('mps')

            x_cpu[:, 0] = 7
            x_mps[:, 0] = 7

            self.assertEqual(x_cpu, x_mps)

        _helper([[1, 2, 3], [4, 5, 6]])
        _helper([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        _helper([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

    def test_inplace_scatter(self):
        # https://github.com/pytorch/pytorch/issues/79672
        a_mps = torch.ones((2, 2),).to(torch.device("mps"))
        b_mps = torch.ones((2, 2),).to(torch.device("mps"))

        a_cpu = torch.ones((2, 2),).to(torch.device("cpu"))
        b_cpu = torch.ones((2, 2),).to(torch.device("cpu"))

        a_mps[:, 0] += b_mps[:, 0]
        a_cpu[:, 0] += b_cpu[:, 0]
        self.assertEqual(a_cpu, a_mps)

        a_mps[:, 0] = a_mps[:, 0] + b_mps[:, 0]
        a_cpu[:, 0] = a_cpu[:, 0] + b_cpu[:, 0]
        self.assertEqual(a_cpu, a_mps)

# These tests were taken from test/test_view_ops.py
# They are subset of those tests as currently only this subset is working.
# This whole `class` will be removed when we add generic device testing. There
# are no additional tests added apart from what is part of test_view_ops.py
class TestViewOpsMPS(TestCase):
    exact_dtype = True

    def is_view_of(self, base, other):
        if (not other._is_view() or
                other is base or
                other._base is not base or
                base.device != other.device):
            return False
        # Note: only validates storage on native device types
        # because some accelerators, like XLA, do not expose storage
        if base.device.type == 'mps':
            if base.storage().data_ptr() != other.storage().data_ptr():
                return False

        return True

    # Returns true if v1 and v2 are views of the same base
    def is_view_of_same_base(self, v1, v2):
        if (not v1._is_view() or v1 is v2):
            return False
        return self.is_view_of(v1._base, v2)

    # Performs transpose if contiguous=True, else returns the input tensor as is
    def _do_transpose(self, x, contiguous=False, dim0=0, dim1=1):
        if contiguous:
            return x
        else:
            return x.transpose(dim0, dim1)

    def test_diagonal_view(self, device="mps"):
        t = torch.ones((5, 5), device=device)
        v = torch.diagonal(t)
        self.assertTrue(self.is_view_of(t, v))

        v[0] = 0
        self.assertEqual(t[0, 0], v[0])

        t = torch.ones((3, 3, 3), device="mps")
        v = torch.diagonal(t, offset=1, dim1=1, dim2=2)
        self.assertTrue(self.is_view_of(t, v))

        v[0, 0] = 0
        self.assertEqual(t[0, 0, 1], v[0, 0])

    def test_select_view(self, device="mps") -> None:
        t = torch.ones((5, 5), device=device)
        v = t.select(0, 2)
        self.assertTrue(self.is_view_of(t, v))

        v[0] = 0
        self.assertEqual(t[2, 0], v[0])

    def test_unbind_view(self, device="mps") -> None:
        t = torch.zeros((5, 5), device=device)
        tup = torch.unbind(t)

        for idx, v in enumerate(tup):
            self.assertTrue(self.is_view_of(t, v))

            v[0] = idx + 1
            self.assertEqual(t[idx, 0], v[0])

    def test_expand_view(self, device="mps") -> None:
        t = torch.ones((5, 1), device=device)
        v = t.expand(5, 5)
        self.assertTrue(self.is_view_of(t, v))

        v[2, 2] = 0
        self.assertEqual(t[2, 0], v[2, 2])

    def test_expand_as_view(self, device="mps"):
        t = torch.ones((5, 1), device=device)
        e = torch.empty((5, 5), device=device)
        v = t.expand_as(e)
        self.assertTrue(self.is_view_of(t, v))

        v[2, 2] = 0
        self.assertEqual(t[2, 0], v[2, 2])

    def test_narrow_view(self, device="mps"):
        t = torch.ones((5, 5), device=device)
        v = torch.narrow(t, 1, 2, 2)
        self.assertTrue(self.is_view_of(t, v))

        v[0, 0] = 0
        self.assertEqual(t[0, 2], v[0, 0])

    def test_permute_view(self, device="mps") -> None:
        t = torch.ones((5, 5), device=device)
        v = t.permute(1, 0)
        self.assertTrue(self.is_view_of(t, v))

        v[0, 1] = 0
        self.assertEqual(t[1, 0], v[0, 1])

    def test_transpose_view(self, device="mps"):
        for fn in (torch.swapdims, torch.swapaxes, torch.transpose):
            t = torch.ones((5, 5), device=device)
            v = fn(t, 0, 1)
            self.assertTrue(self.is_view_of(t, v))

            v[0, 1] = 0
            self.assertEqual(t[1, 0], v[0, 1])

    def test_transpose_inplace_view(self, device="mps"):
        t = torch.ones(5, 5, device=device)
        v = t.view_as(t)
        v = v.swapdims_(0, 1)
        self.assertTrue(self.is_view_of(t, v))
        v[0, 1] = 0
        self.assertEqual(t[1, 0], v[0, 1])

        t = torch.ones(5, 5, device=device)
        v = t.view_as(t)
        v = v.swapaxes_(0, 1)
        self.assertTrue(self.is_view_of(t, v))
        v[0, 1] = 0
        self.assertEqual(t[1, 0], v[0, 1])

        t = torch.ones(5, 5, device=device)
        v = t.view_as(t)
        v = v.transpose_(0, 1)
        self.assertTrue(self.is_view_of(t, v))
        v[0, 1] = 0
        self.assertEqual(t[1, 0], v[0, 1])

    def test_t_view(self, device="mps"):
        t = torch.ones((5, 5), device=device)
        v = t.t()
        self.assertTrue(self.is_view_of(t, v))

        v[0, 1] = 0
        self.assertEqual(t[1, 0], v[0, 1])

    def test_t_inplace_view(self, device="mps"):
        t = torch.ones(5, 5, device=device)
        v = t.view_as(t)
        v = v.t_()
        self.assertTrue(self.is_view_of(t, v))
        v[0, 1] = 0
        self.assertEqual(t[1, 0], v[0, 1])

    def test_T_view(self, device="mps"):
        for op in ("T", "H", "mT", "mH"):
            t = torch.ones((5, 5), device=device)
            v = getattr(t, op)
            self.assertTrue(self.is_view_of(t, v))

            v[0, 1] = 0
            self.assertEqual(t[1, 0], v[0, 1])

    # requires aten::unfold
    # def test_unfold_view(self, device="mps"):
    #     t = torch.ones(10, device=device)
    #     v = t.unfold(0, 3, 2)
    #     self.assertTrue(self.is_view_of(t, v))

    #     v[1, 0] = 0
    #     self.assertEqual(t[2], v[1, 0])

    def test_squeeze_view(self, device="mps"):
        t = torch.ones(5, 1, 5, device=device)
        v = torch.squeeze(t)
        self.assertTrue(self.is_view_of(t, v))
        v[0, 1] = 0
        self.assertTrue(t is v._base)

    def test_squeeze_inplace_view(self, device="mps"):
        t = torch.ones(5, 5, device=device)
        v = t.view_as(t)
        v = v.squeeze_()
        self.assertTrue(self.is_view_of(t, v))
        v[0, 1] = 0
        self.assertTrue(t is v._base)

    def test_unsqueeze_view(self, device="mps"):
        t = torch.ones(5, 5, device=device)
        v = torch.unsqueeze(t, 1)
        self.assertTrue(self.is_view_of(t, v))

        v[0, 0, 1] = 0
        self.assertEqual(t[0, 1], v[0, 0, 1])

    def test_unsqueeze_inplace_view(self, device="mps"):
        t = torch.ones(5, 5, device=device)
        v = t.view_as(t)
        v = v.unsqueeze_(1)
        self.assertTrue(self.is_view_of(t, v))
        v[0, 0, 1] = 0
        self.assertEqual(t[0, 1], v[0, 0, 1])

    def test_as_strided_view(self, device="mps"):
        t = torch.ones(5, 5, device=device)
        v = torch.as_strided(t, (25,), (1,))
        self.assertTrue(self.is_view_of(t, v))

        v[6] = 0
        self.assertEqual(t[1, 1], v[6])

    def test_as_strided_inplace_view(self, device="mps"):
        t = torch.ones(5, 5, device=device)
        v = t.view_as(t)
        v = v.as_strided_((25,), (1,))
        self.assertTrue(self.is_view_of(t, v))
        v[6] = 0
        self.assertEqual(t[1, 1], v[6])

    def test_view_view(self, device="mps"):
        t = torch.ones(5, 5, device=device)
        v = t.view(25)
        self.assertTrue(self.is_view_of(t, v))

        v[6] = 0
        self.assertEqual(t[1, 1], v[6])

    def test_view_as_view(self, device="mps"):
        t = torch.ones(5, 5, device=device)
        e = torch.empty((25,))
        v = t.view_as(e)
        self.assertTrue(self.is_view_of(t, v))

        v[6] = 0
        self.assertEqual(t[1, 1], v[6])

    def test_contiguous_self(self, device="mps"):
        t = torch.ones(5, 5, device=device)
        s = t.contiguous()
        self.assertTrue(s is t)

    def test_contiguous_nonview(self, device="mps"):
        t = torch.ones(5, 5, device=device)
        nv = t.t().contiguous()
        self.assertTrue(not self.is_view_of(t, nv))

        nv[0, 0] = 0
        self.assertNotEqual(t[0, 0], nv[0, 0])

    def test_reshape_view(self, device="mps"):
        t = torch.ones(5, 5, device=device)
        v = torch.reshape(t, (25,))
        self.assertTrue(self.is_view_of(t, v))

        v[6] = 0
        self.assertEqual(t[1, 1], v[6])

    def test_reshape_as_view(self, device="mps"):
        t = torch.ones(5, 5, device=device)
        e = torch.empty((25,), device=device)
        v = t.reshape_as(e)
        self.assertTrue(self.is_view_of(t, v))

        v[6] = 0
        self.assertEqual(t[1, 1], v[6])

    def test_reshape_nonview(self, device="mps"):
        t = torch.ones(5, 5, device=device)
        nv = torch.reshape(t.t(), (25,))
        self.assertTrue(not self.is_view_of(t, nv))

        nv[6] = 0
        self.assertNotEqual(t[1, 1], nv[6])

    def test_flatten_view(self, device="mps"):
        def test_writes_propagate(t, v):
            idx_t = (0,) * t.ndim
            idx_v = (0,) * v.ndim
            v[idx_v] = 0
            self.assertEqual(t[idx_t], v[idx_v])

        t = torch.ones(1, 2, 3, 4, device=device)
        v = t.flatten()
        self.assertTrue(self.is_view_of(t, v))
        test_writes_propagate(t, v)

        # zero-dimensional tensor
        t = torch.tensor(1, device=device)
        v = t.flatten()
        test_writes_propagate(t, v)
        self.assertTrue(self.is_view_of(t, v))

        t = torch.ones(1, 2, 3, 4, device=device).transpose(2, 3)
        v = t.flatten(0, 1)
        test_writes_propagate(t, v)
        self.assertTrue(self.is_view_of_same_base(t, v))

        # stride[i] = stride[i + 1] * size[i + 1] is satisfied for 3 groups:
        t = torch.ones(720, device=device) \
            .as_strided((2, 3, 2, 3, 5, 4), (6, 2, 15, 5, 1, 0))
        #               [--1--|---2---|-3-] [--1--|----2---|-3-]
        v1 = t.flatten(0, 1)
        v2 = v1.flatten(1, 3)
        v3 = v2.flatten(2, 2)
        test_writes_propagate(t, v1)
        self.assertTrue(self.is_view_of_same_base(t, v1))
        test_writes_propagate(t, v2)
        self.assertTrue(self.is_view_of_same_base(t, v2))
        test_writes_propagate(t, v3)
        self.assertTrue(self.is_view_of_same_base(t, v3))

    def test_flatten_nonview(self, device="mps"):
        def assert_is_nonview(t, nv):
            idx_t = (0,) * t.ndim
            idx_nv = (0,) * nv.ndim
            self.assertTrue(not nv._is_view())
            nv[idx_nv] = 0
            self.assertNotEqual(t[idx_t], nv[idx_nv])
        t = torch.ones(2, 3, 2, 3, device=device).transpose(2, 3)
        nv = t.flatten(1, 3)
        assert_is_nonview(t, nv)

        t = torch.ones(2, 2, device=device).T
        nv = t.flatten()
        assert_is_nonview(t, nv)

        # flatten returns the original object if start_dim=end_dim
        t = t = torch.ones(2, 2, device=device)
        nv = t.flatten(1, 1)
        self.assertTrue(t is nv)

    def test_basic_indexing_slice_view(self, device="mps"):
        t = torch.ones(5, 5, device=device)
        v = t[:2, :3]
        self.assertTrue(self.is_view_of(t, v))

        v[0, 0] = 0
        self.assertEqual(t[0, 0], v[0, 0])

    def test_basic_indexing_ellipses_view(self, device="mps"):
        t = torch.ones(5, 5, device=device)
        v = t[..., :2]
        self.assertTrue(self.is_view_of(t, v))

        v[0, 0] = 0
        self.assertEqual(t[0, 0], v[0, 0])

    def test_basic_indexing_newaxis_view(self, device="mps"):
        t = torch.ones(5, 5, device=device)
        v = t[None, :2, 3]
        self.assertTrue(self.is_view_of(t, v))

        v[0, 0] = 0
        self.assertEqual(t[0, 3], v[0, 0])

    def test_chunk_view(self, device="mps"):
        t = torch.zeros(3, 3, device=device)
        l = torch.chunk(t, 3)

        for idx, v in enumerate(l):
            self.assertTrue(self.is_view_of(t, v))

            v[0, 0] = idx + 1
            self.assertEqual(t[idx, 0], v[0, 0])

    def test_split_view(self, device="mps"):
        t = torch.zeros(3, 3, device=device)
        l = torch.split(t, [1, 1, 1])

        for idx, v in enumerate(l):
            self.assertTrue(self.is_view_of(t, v))

            v[0, 0] = idx + 1
            self.assertEqual(t[idx, 0], v[0, 0])

    def test_movedim_view(self, device="mps"):
        def run_test(device, op):
            t = torch.zeros(3, 3, device=device)
            out = op(t)

            self.assertTrue(self.is_view_of(t, out))

            # Randomly change values in output
            # and verify that original is changed
            # as well.
            for _ in range(3):
                idx_1, idx_2 = random.randint(0, 2), random.randint(0, 2)
                out[idx_1, idx_2] = random.random()
                self.assertEqual(t[idx_2, idx_1], out[idx_1, idx_2])

        for fn in [torch.movedim, torch.moveaxis]:
            op = partial(fn, source=(0, 1), destination=(1, 0))
            run_test(device, op)

            op = partial(fn, source=0, destination=1)
            run_test(device, op)

    # Testing that the generated view_copy kernel and its derivative are implemented correctly
    def test_view_copy(self, device="mps"):
        a = torch.randn(4, device=device, requires_grad=True)
        a_ref = a.clone().detach().requires_grad_()
        a_view = a_ref.view(2, 2)
        a_view_copy = torch.view_copy(a, (2, 2))

        # view_copy ops don't preserve view relationship
        self.assertTrue(self.is_view_of(a_ref, a_view))
        self.assertFalse(self.is_view_of(a, a_view_copy))

        a_view_copy.sum().backward()
        a_view.sum().backward()

        # forward and backward give the same shape + result
        self.assertEqual(a_view_copy, a_view)
        self.assertEqual(a.grad, a_ref.grad)

    def test_view_copy_out(self, device="mps"):
        a = torch.randn(2, 2, device=device)
        out = torch.empty(2, device=device)

        torch.diagonal_copy(a, out=out)
        expected = torch.diagonal_copy(a)

        self.assertEqual(expected, out)

        a = torch.randn(4, device=device)
        out1 = torch.empty(2, device=device)
        out2 = torch.empty(2, device=device)

        torch.split_copy(a, 2, out=(out1, out2))
        expected1, expected2 = torch.split_copy(a, 2)

        self.assertEqual(expected1, out1)
        self.assertEqual(expected2, out2)

    def test_detached_view_copy(self, device="mps"):
        # https://github.com/pytorch/pytorch/issues/86052
        x = torch.arange(2)
        # .detach() makes y not a view, but contig tensor
        # with non-zero offset
        y = x[1].detach()
        z = y.to(device)
        self.assertEqual(y, z.cpu())

    def test_empty_reshape(self, device="mps"):
        x = torch.randn(0, 6, device=device)
        self.assertEqual((1, 0, 6, 1, 1), x.reshape(1, 0, 6, 1, 1).shape)
        # should be viewable -- i.e. data_ptr is the same.
        self.assertEqual(x.data_ptr(), x.reshape(1, 0, 6, 1, 1).data_ptr())

        # match NumPy semantics -- don't infer the size of dimension with a degree of freedom
        self.assertRaises(RuntimeError, lambda: x.reshape(0, -1))

    def test_expand(self, device="mps"):
        tensor = torch.rand(1, 8, 1, device=device)
        tensor2 = torch.rand(5, device=device)
        template = torch.rand(4, 8, 5, device=device)
        target = template.size()
        self.assertEqual(tensor.expand_as(template).size(), target)
        self.assertEqual(tensor.expand(4, 8, 5).size(), target)
        self.assertEqual(tensor.expand(target).size(), target)
        self.assertEqual(tensor2.expand_as(template).size(), target)
        self.assertEqual(tensor2.expand(4, 8, 5).size(), target)
        self.assertEqual(tensor2.expand(target).size(), target)

        # test double expand
        self.assertEqual(tensor2.expand(1, 5).expand(2, 2, 5), tensor2.repeat(2, 2, 1))

        # test non-contiguous
        noncontig = torch.randn(5, 2, 1, 3, device=device)[:, 0]
        self.assertFalse(noncontig.is_contiguous())
        self.assertEqual(noncontig.expand(2, 5, 4, 3), noncontig.contiguous().repeat(2, 1, 4, 1))

        # make sure it's compatible with unsqueeze
        expanded = tensor2.expand(1, 1, 5)
        unsqueezed = tensor2.unsqueeze(0).unsqueeze(1)
        self.assertEqual(expanded, unsqueezed)
        self.assertEqual(expanded.stride(), unsqueezed.stride())

        # test -1 as target size
        self.assertEqual(tensor.expand(4, -1, 5), tensor.expand(4, 8, 5))
        self.assertRaises(RuntimeError, lambda: tensor2.expand(-1, -1))

        # test expanding empty to empty
        self.assertEqual(torch.zeros(0, device=device).expand((0,)), torch.zeros(0, device=device))

    def test_view_empty(self, device="mps"):
        x = torch.randn(0, 6, device=device)
        self.assertEqual((1, 0, 6, 1, 1), x.view(1, 0, 6, 1, 1).shape)

    def test_reshape(self, device="mps"):
        x = torch.randn(3, 3, device=device)
        self.assertEqual(x.data_ptr(), x.reshape(-1).data_ptr())
        self.assertEqual(x.data_ptr(), x.reshape(1, 9, 1).data_ptr())
        self.assertEqual(torch.reshape(x, (9,)), x.reshape(9))
        self.assertRaises(RuntimeError, lambda: x.reshape(-1, -1))

        y = torch.randn(4, 4, 4, device=device)[:, 0, :]
        # .data_ptr() on meta tensors is always 0 so they are equal regardless of the reshape
        if device != "meta":
            self.assertNotEqual(y.data_ptr(), y.reshape(-1).data_ptr())
        self.assertEqual(y.contiguous().view(-1), y.reshape(-1))
        self.assertEqual(y.reshape(2, 2, 4).data_ptr(), y.data_ptr())

        s = torch.randn((), device=device)
        self.assertEqual(s.data_ptr(), s.reshape(()).data_ptr())
        self.assertEqual(s.reshape(-1).shape, (1,))
        self.assertRaises(RuntimeError, lambda: s.reshape(2))

        empty = torch.tensor([], device=device)
        self.assertEqual(empty, empty.reshape(-1))
        self.assertEqual(empty, empty.reshape([0]))
        # TODO: fix these once we have multi-dimensional empty tensors
        self.assertEqual(empty.reshape([0, 1]).shape, (0, 1))
        self.assertEqual(empty.reshape([1, -1]).shape, (1, 0))
        self.assertRaises(RuntimeError, lambda: empty.reshape(1))

        x = torch.randn(3, 3, device=device)
        self.assertEqual(x.data_ptr(), x.reshape_as(torch.rand(9)).data_ptr())
        self.assertEqual(x.data_ptr(), x.reshape_as(torch.rand(1, 9, 1)).data_ptr())
        self.assertRaises(RuntimeError, lambda: x.reshape_as(torch.rand(10, device=device)))

    def test_narrow(self, device="mps"):
        x = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        self.assertEqual(x.narrow(0, 0, 1), torch.tensor([[0, 1, 2]]))
        self.assertEqual(x.narrow(0, 0, 2), torch.tensor([[0, 1, 2], [3, 4, 5]]))
        self.assertEqual(x.narrow(0, 1, 1), torch.tensor([[3, 4, 5]]))
        self.assertEqual(x.narrow(0, -1, 1), torch.tensor([[6, 7, 8]]))
        self.assertEqual(x.narrow(0, -2, 2), torch.tensor([[3, 4, 5], [6, 7, 8]]))
        self.assertEqual(x.narrow(0, -3, 3), torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]]))
        self.assertEqual(x.narrow(-1, -1, 1), torch.tensor([[2], [5], [8]]))
        self.assertEqual(x.narrow(-2, -1, 1), torch.tensor([[6, 7, 8]]))

    def test_narrow_tensor(self, device="mps"):
        x = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        self.assertEqual(x.narrow(0, torch.tensor(0), 1), torch.tensor([[0, 1, 2]]))
        with self.assertRaises(Exception):
            x.narrow(0, torch.tensor(0.), 1)
        with self.assertRaises(Exception):
            x.narrow(0, torch.tensor([0]), 1)
        with self.assertRaises(Exception):
            x.narrow(0, torch.tensor([0, 1]), 1)

    def test_t(self, device="mps"):
        # Test 0D tensors
        x = torch.randn(())
        self.assertEqual(x, x.t())
        x = x.to_sparse()
        self.assertEqual(x, x.t())

        # Test 1D tensors
        x = torch.arange(4)
        self.assertEqual(x, x.t())
        x = x.to_sparse()
        self.assertEqual(x, x.t())

        # Test 2D tensors
        x = torch.rand((2, 2))
        self.assertEqual(x.t(), x.transpose(0, 1))
        x = x.to_sparse()
        self.assertEqual(x.t(), x.transpose(0, 1))

        # Test 3D tensor
        x = torch.rand((2, 2, 2))
        with self.assertRaisesRegex(RuntimeError, 'expects a tensor with <= 2 dimensions, but self is 3D'):
            x.t()
        x = x.to_sparse()
        with self.assertRaisesRegex(RuntimeError, 'expects a tensor with <= 2 sparse and 0 dense dimensions'):
            x.t()

    def test_split(self, device="mps"):
        tensor = torch.rand(7, 4)
        split_size = 3
        dim = 0
        target_sizes = ([3, 4], [3, 4], [1, 4])
        splits = tensor.split(split_size, dim)
        start = 0
        for target_size, split in zip(target_sizes, splits):
            self.assertEqual(split.size(), target_size)
            self.assertEqual(tensor.narrow(dim, start, target_size[dim]), split, atol=0, rtol=0)
            start = start + target_size[dim]

        # Variable sections split
        tensor = torch.randn(20, 10)
        dim = 0
        split_sizes = [5, 5, 10]
        target_sizes = ([[5, 10], [5, 10], [10, 10]])
        splits = tensor.split(split_sizes, dim)
        start = 0
        for target_size, split in zip(target_sizes, splits):
            self.assertEqual(split.size(), target_size)
            self.assertEqual(tensor.narrow(dim, start, target_size[dim]), split, atol=0, rtol=0)
            start = start + target_size[dim]

        split_sizes = [2, 2, 6]
        target_sizes = ([20, 2], [20, 2], [20, 6])
        dim = 1
        splits = tensor.split(split_sizes, dim)
        start = 0
        for target_size, split in zip(target_sizes, splits):
            self.assertEqual(split.size(), target_size)
            self.assertEqual(tensor.narrow(dim, start, target_size[dim]), split, atol=0, rtol=0)
            start = start + target_size[dim]

    def test_chunk(self, device="mps"):
        tensor = torch.rand(4, 7)
        num_chunks = 3
        dim = 1
        target_sizes = ([4, 3], [4, 3], [4, 1])
        splits = tensor.chunk(num_chunks, dim)
        start = 0
        for target_size, split in zip(target_sizes, splits):
            self.assertEqual(split.size(), target_size)
            self.assertEqual(tensor.narrow(dim, start, target_size[dim]), split,
                             atol=0, rtol=0)
            start = start + target_size[dim]

        # Invalid chunk sizes
        error_regex = 'chunk expects.*greater than 0'
        with self.assertRaisesRegex(RuntimeError, error_regex):
            tensor.chunk(0)
        with self.assertRaisesRegex(RuntimeError, error_regex):
            tensor.chunk(-2)

    def test_unsqueeze(self, device="mps") -> None:
        x = torch.randn(2, 3, 4)
        y = x.unsqueeze(1)
        self.assertEqual(y, x.view(2, 1, 3, 4))
        y = x.clone().unsqueeze_(2)
        self.assertEqual(y, x.view(2, 3, 1, 4))

        x = x[:, 1]
        self.assertFalse(x.is_contiguous())
        y = x.unsqueeze(1)
        self.assertEqual(y, x.contiguous().view(2, 1, 4))
        y = x.clone().unsqueeze_(2)
        self.assertEqual(y, x.contiguous().view(2, 4, 1))

    # unit test for special case transposed copy (see ATen/native/Copy.cpp for details)
    def test_big_transpose(self, device="mps"):
        t = torch.rand(456, 789, device=device)
        t1 = t.t().contiguous()
        t2 = torch.from_numpy(t.cpu().numpy().transpose())
        self.assertEqual(t1, t2)

    def test_T(self, device="mps"):
        a = torch.randn(2, 3, 4, device=device)
        t1 = a.T
        t2 = a.permute(2, 1, 0)
        self.assertEqual(t2, t1)
        b = torch.randn(10, device=device)
        self.assertEqual(b, b.T)
        scalar = torch.tensor(5, device=device)
        self.assertEqual(scalar, scalar.T)

    def test_transposes(self, device="mps", dtype=torch.float32):
        for op in ("T", "H", "mT", "mH", "adjoint"):
            shapes = ((), (2, 3), (2, 3, 4)) if op[0] == "m" or op == "adjoint" else ((), (2, 3),)
            for shape in shapes:
                a = make_tensor(shape, device=device, dtype=dtype)
                t1 = getattr(a, op)
                if op == "adjoint":
                    t1 = t1()
                t2 = a
                if a.ndim != 0:
                    t2 = t2.transpose(-2, -1)
                if op[-1] == "H" or op == "adjoint":
                    t2 = t2.conj()
                self.assertEqual(t2, t1)

    def test_transposes_errors(self, device="mps", dtype=torch.float32):
        for op in ("H", "mT", "mH", "adjoint"):
            shapes = ((2,), (2, 3, 4)) if op == "H" else ((2,),)
            for shape in shapes:
                a = make_tensor(shape, device=device, dtype=dtype)
                with self.assertRaisesRegex(RuntimeError, "only supported on matrices"):
                    t1 = getattr(a, op)
                    if op == "adjoint":
                        t1 = t1()

    def test_python_types(self, device="mps"):
        a1 = torch.randn((1, 2), device=device, dtype=torch.float32)
        a2 = torch.randn((1, 2), device=device, dtype=torch.float32)
        self.assertEqual(a1.dtype, a2.dtype)

        b1 = torch.arange(10, 20, dtype=torch.int64, device=device)
        b2 = torch.arange(10, 20, dtype=int, device=device)
        self.assertEqual(b1.dtype, b2.dtype)

        c1 = torch.tensor([True, False], dtype=torch.bool, device=device)
        c2 = torch.tensor([True, False], dtype=bool, device=device)
        self.assertEqual(c1.dtype, c2.dtype)

    # TODO: is resize best put in test_view_ops?
    def test_resize_as_preserves_strides(self, device="mps"):
        x = torch.empty(2, 3).t()
        old_strides = x.stride()
        x.resize_as_(x)
        self.assertEqual(x.stride(), old_strides)

    def test_memory_format_resize_as(self, device="mps"):
        def test_helper(shape, memory_format, device="mps"):
            xc = torch.randn(shape, device=device).contiguous(memory_format=memory_format)
            flat = torch.randn(xc.numel(), device=device)
            flat.resize_as_(xc, memory_format=torch.preserve_format)
            self.assertTrue(flat.is_contiguous(memory_format=memory_format))

        test_helper((10, 3, 32, 32), torch.channels_last, device="mps")
        test_helper((3, 10, 3, 32, 32), torch.channels_last_3d, device="mps")

    def test_memory_format_resize_(self, device="mps"):
        def test_helper(shape, numel, memory_format, device="mps"):
            flat = torch.randn(numel, device=device)
            flat.resize_(shape, memory_format=memory_format)
            self.assertTrue(flat.is_contiguous(memory_format=memory_format))

        test_helper((10, 3, 32, 32), 10 * 3 * 32 * 32, torch.channels_last, device="mps")
        test_helper((3, 10, 3, 32, 32), 3 * 10 * 3 * 32 * 32, torch.channels_last_3d, device="mps")

    # TODO: OpInfo this
    def _test_atleast(self, device, torch_fn):
        # 0-dim
        s = torch.tensor(0.5, dtype=torch.double, requires_grad=True)

        gradcheck(lambda x: torch_fn(x), s)
        gradgradcheck(lambda x: torch_fn(x), s)

        # 1-dim
        a = torch.rand(4, dtype=torch.double, requires_grad=True)

        gradcheck(lambda x: torch_fn(x), a)
        gradgradcheck(lambda x: torch_fn(x), a)

        # 2,3,4-dim
        b = torch.rand(4, 3, dtype=torch.double, requires_grad=True)
        c = torch.rand(4, 3, 2, dtype=torch.double, requires_grad=True)
        d = torch.rand(4, 3, 2, 1, dtype=torch.double, requires_grad=True)

        input_tuple = (s, a, b, c, d)
        gradcheck(lambda s, w, x, y, z: torch_fn(s, w, x, y, z), input_tuple)
        gradgradcheck(lambda s, w, x, y, z: torch_fn(s, w, x, y, z), input_tuple)

    def test_atleast_gradient(self, device="mps"):
        self._test_atleast(device, torch.atleast_1d)
        self._test_atleast(device, torch.atleast_2d)
        self._test_atleast(device, torch.atleast_3d)

    def test_view(self, device="mps"):
        tensor = torch.rand(15, device=device)
        template = torch.rand(3, 5, device=device)
        empty = torch.empty(0, device=device)
        target = template.size()
        self.assertEqual(tensor.view_as(template).size(), target)
        self.assertEqual(tensor.view(3, 5).size(), target)
        self.assertEqual(tensor.view(torch.Size([3, 5])).size(), target)
        self.assertEqual(tensor.view(-1, 5).size(), target)
        self.assertEqual(tensor.view(3, -1).size(), target)
        tensor_view = tensor.view(5, 3)
        tensor_view.fill_(random.uniform(0, 1))
        self.assertEqual(empty.view_as(empty), empty)
        self.assertEqual(empty.view(0), empty)
        self.assertEqual(empty.view(0, 3, 0, 1).size(), torch.Size([0, 3, 0, 1]))
        self.assertEqual(empty.view(0, 3, 0, 1).view(0), empty)

        # test size inference with empty tensors
        self.assertEqual(empty.view(-1).size(), torch.Size([0]))
        self.assertEqual(empty.view(10, 3, -1).size(), torch.Size([10, 3, 0]))

        with self.assertRaisesRegex(RuntimeError, r"because the unspecified dimension size -1 can be any value"):
            empty.view(-1, 0)

        with self.assertRaisesRegex(RuntimeError, r"because the unspecified dimension size -1 can be any value"):
            empty.view(3, 0, -1, 0)

        self.assertRaises(RuntimeError, lambda: tensor.view(15, 0))
        self.assertRaises(RuntimeError, lambda: tensor.view(7, -1))
        self.assertRaises(RuntimeError, lambda: tensor.view(15, -1, -1))

    # RuntimeError: Invalid device for storage: mps
    def test_contiguous(self, device="mps"):
        x = torch.randn(1, 16, 5, 5, device=device)
        self.assertTrue(x.is_contiguous())
        stride = list(x.stride())
        stride[0] = 20
        # change the stride in dimension 0. the tensor is still contiguous because size[0] is 1
        x.set_(x.storage(), 0, x.size(), stride)
        self.assertTrue(x.is_contiguous())

    def test_resize_all_dtypes_and_devices(self, device="mps"):
        shape = (2, 2)
        for dt in (torch.half, torch.bfloat16, torch.bool):
            x = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=dt, device=device)
            x.resize_(shape)
            self.assertEqual(shape, x.shape)

    def test_resize_as_all_dtypes_and_devices(self, device="mps"):
        for dt in (torch.half, torch.bfloat16, torch.bool):
            x = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=dt, device=device)
            y = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=dt, device=device)
            x.resize_as_(y)
            self.assertEqual(y.shape, x.shape)

    def test_resize_overflow(self, device="mps"):
        x = torch.empty((), dtype=torch.float64)
        with self.assertRaisesRegex(RuntimeError, 'Storage size calculation overflowed'):
            x.resize_([2, 4, 2**29, 2**29])
        with self.assertRaisesRegex(RuntimeError, 'overflow'):
            x.resize_([8, 8, 2**29, 2**29])

    def test_view_all_dtypes_and_devices(self, device="mps"):
        for dt in (torch.float, torch.bool):
            x = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=dt, device=device)
            self.assertEqual(x.view(6).shape, [6])

class TestConvolutionMPS(TestCase):
    def test_conv1d_all_strides_paddings(self):
        # https://github.com/pytorch/pytorch/issues/82921
        def helper(stride, padding):
            y_cpu = torch.randn(1, 57, 40)
            conv_cpu = nn.Conv1d(57, 20, stride=stride, padding=padding, kernel_size=3, bias=False)
            conv_gpu = copy.deepcopy(conv_cpu).to(device='mps')
            x_cpu = conv_cpu(y_cpu)

            y_gpu = y_cpu.to(device='mps')
            x_gpu = conv_gpu(y_gpu)
            self.assertEqual(x_cpu, x_gpu.cpu())
        for stride in range(1, 4):
            for padding in range(1, 4):
                helper(stride, padding)


    def test_conv1d_channels_last(self):
        # https://github.com/pytorch/pytorch/issues/81557
        model_cpu = torch.nn.Conv1d(1, 128, 3)
        a_cpu = torch.arange((128 * 176), dtype=torch.float32)
        a_cpu = a_cpu.view(128, 176, 1).permute(0, 2, 1)
        out_cpu = model_cpu(a_cpu)

        a_mps = a_cpu.detach().clone().to("mps")
        model_mps = model_cpu.to("mps")
        out_mps = model_mps(a_mps)

        self.assertEqual(out_cpu, out_mps.cpu(), rtol=2.6e-05, atol=2e-04)

    def test_conv_transpose_1d_all_strides(self):
        # https://github.com/pytorch/pytorch/issues/82711
        def helper(stride):
            y_cpu = torch.ones(1, 1, 2)
            deconv_cpu = nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=1, stride=stride, bias=False, padding=1)
            deconv_cpu.weight.data = torch.ones(1, 1, 2)
            deconv_gpu = copy.deepcopy(deconv_cpu).to(device='mps')
            x_cpu = deconv_cpu(y_cpu)

            y_gpu = y_cpu.to(device='mps')
            x_gpu = deconv_gpu(y_gpu)
            self.assertEqual(x_cpu, x_gpu.cpu())
        [helper(stride) for stride in [1, 2, 3]]

    def test_conv_transpose_1d_nn_functional(self):
        # https://github.com/pytorch/pytorch/issues/82563
        tin = torch.rand((1, 512, 1245), dtype=torch.float32)
        tparams = torch.rand((512, 256, 16), dtype=torch.float32)
        tbias = torch.rand((256), dtype=torch.float32)

        device = 'cpu'
        tcpu = torch.nn.functional.conv_transpose1d(tin.to(device), tparams.to(device), tbias.to(device), stride=8, padding=4)

        device = 'mps'
        tgpu = torch.nn.functional.conv_transpose1d(tin.to(device), tparams.to(device), tbias.to(device), stride=8, padding=4)

        self.assertEqual(tcpu, tgpu.cpu(), rtol=2.6e-05, atol=2e-04)

    def test_conv_backward_1d_channels_last(self):
        # https://github.com/pytorch/pytorch/issues/84511
        conv_cpu = torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3)
        conv_mps = copy.deepcopy(conv_cpu).to(device='mps')

        data = torch.rand(1, 176, 1, dtype=torch.float32)
        x_cpu = data.permute(0, 2, 1).contiguous()
        x_mps = data.permute(0, 2, 1).contiguous().to("mps")
        res_cpu = conv_cpu(x_cpu).sum().backward()
        res_mps = conv_mps(x_mps).sum().backward()

        self.assertEqual(res_cpu, res_mps)

    def test_conv1d_contiguous(self):
        model_cpu = torch.nn.Conv1d(1, 128, 3)
        a_cpu = torch.ones(128, 1, 176)
        out_cpu = model_cpu(a_cpu)

        a_mps = a_cpu.detach().clone().to("mps")
        model_mps = model_cpu.to("mps")
        out_mps = model_mps(a_mps)

        self.assertEqual(out_cpu.shape, out_mps.shape)
        self.assertEqual(out_cpu, out_mps.cpu())

    def test_conv2d_all_strides_paddings(self):
        # https://github.com/pytorch/pytorch/issues/83180
        y_cpu = torch.randn(2, 2, 3, 6)
        y_gpu = y_cpu.to(device='mps')
        for strideX in range(1, 4):
            for strideY in range(1, 4):
                conv_cpu = torch.nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, stride=(strideX, strideY))
                conv_gpu = copy.deepcopy(conv_cpu).to(device='mps')
                x_cpu = conv_cpu(y_cpu)
                x_gpu = conv_gpu(y_gpu)
                self.assertEqual(x_cpu, x_gpu.cpu(), rtol=1e-03, atol=1e-05)

    def test_conv2d_single_stride(self):
        y_cpu = torch.randn(2, 2, 3, 6)
        y_gpu = y_cpu.to(device='mps')
        for stride in range(1, 4):
            conv_cpu = torch.nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, stride=stride)
            conv_gpu = copy.deepcopy(conv_cpu).to(device='mps')
            x_cpu = conv_cpu(y_cpu)
            x_gpu = conv_gpu(y_gpu)
            self.assertEqual(x_cpu, x_gpu.cpu(), rtol=1e-03, atol=1e-05)

class TestAdvancedIndexing(TestCase):
    supported_dtypes = [torch.float32, torch.float16, torch.int64, torch.int32, torch.int16, torch.uint8]
    supported_np_dtypes = [np.float32, np.float16, np.int64, np.int32, np.int16, np.uint8]

    def test_masked_select(self):
        x = torch.randn(3, 4)
        x_mps = x.to("mps")
        mask = x.ge(0.5)
        mask_mps = x_mps.ge(0.5)

        res = torch.masked_select(x, mask)
        res_mps = torch.masked_select(x_mps, mask_mps)

        self.assertEqual(res, res_mps)

    # examples from https://www.tutorialspoint.com/numpy/numpy_advanced_indexing.htm
    def test_indexing_get(self):
        def helper(dtype):
            x_cpu = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=dtype)
            x_mps = x_cpu.detach().clone().to("mps")

            y_cpu = x_cpu[[0, 1, 2], [0, 1, 0]]
            y_mps = x_mps[[0, 1, 2], [0, 1, 0]]
            self.assertEqual(y_cpu, y_mps, str(dtype))
        [helper(dtype) for dtype in self.supported_dtypes]

    def test_indexing_select_corners(self):
        def helper(dtype):
            x_cpu = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], dtype=dtype)
            x_mps = x_cpu.detach().clone().to("mps")

            rows_cpu = torch.tensor([[0, 0], [3, 3]])
            rows_mps = rows_cpu.detach().clone().to("mps")

            cols_cpu = torch.tensor([[0, 2], [0, 2]])
            cols_mps = cols_cpu.detach().clone().to("mps")

            res_cpu = x_cpu[rows_cpu, cols_cpu]
            res_mps = x_mps[rows_mps, cols_mps]

            self.assertEqual(res_cpu, res_mps, str(dtype))
        [helper(dtype) for dtype in self.supported_dtypes]

    # FIXME: uint8 fails for this testcase, needs further debugging
    def test_slicing_using_advanced_index_for_column(self):
        def helper(dtype):
            x_cpu = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], dtype=dtype)
            x_mps = x_cpu.detach().clone().to("mps")

            z_cpu = x_cpu[1:4, 1:3]
            z_mps = x_mps[1:4, 1:3]
            self.assertEqual(z_cpu, z_mps, str(dtype))

            # using advanced index for column
            y_cpu = x_cpu[1:4, [1, 2]]
            y_mps = x_mps[1:4, [1, 2]]
            self.assertEqual(y_cpu, y_mps, str(dtype))
        # FIXME: use supported_dtypes once uint8 is fixed
        [helper(dtype) for dtype in [torch.float32, torch.float16, torch.int64, torch.int32, torch.int16]]

    # FIXME: conditional indexing not working
    # def test_boolean_array_indexing_1(self):
    #     def helper(dtype):
    #         x_cpu = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], dtype=dtype)
    #         x_mps = x_cpu.detach().clone().to("mps")

    #         res_cpu = x_cpu[x_cpu > 5]
    #         res_mps = x_mps[x_mps > 5]

    #         print(res_cpu)
    #         print(res_mps)

    #         self.assertEqual(res_cpu, res_mps, str(dtype))
    #     [helper(dtype) for dtype in self.supported_dtypes]


    def test_advanced_indexing_3D_get(self):
        def helper(x_cpu):
            x_mps = x_cpu.detach().clone().to("mps")
            self.assertEqual(x_cpu[[1, 2], 3, :], x_mps[[1, 2], 3, :])
            self.assertEqual(x_cpu[[0, 2], :, :], x_mps[[0, 2], :, :])
            self.assertEqual(x_cpu[:, [1, 0], [1]], x_mps[:, [1, 0], [1]])

        x_cpu = torch.tensor([[[0.1, 0.2, 0.3, 0.4],
                               [0.5, 0.6, 0.7, 0.8],
                               [0.9, 1.0, 1.1, 1.2],
                               [1.3, 1.4, 1.5, 1.6]],

                              [[2.0, 2.1, 2.2, 2.3],
                               [2.4, 2.5, 2.6, 2.7],
                               [2.8, 2.9, 3.0, 3.1],
                               [3.2, 3.3, 3.4, 3.5]],

                              [[4.0, 4.1, 4.2, 4.3],
                               [4.4, 4.5, 4.6, 4.7],
                               [4.8, 4.9, 5.0, 5.1],
                               [5.1, 5.2, 5.3, 5.4]]], device="cpu", dtype=torch.float32)
        helper(x_cpu)
        for idx in range(len(self.supported_np_dtypes)):
            # torch.randn / torch.rand don't work with all dtypes
            # Generate input data for all dtypes on Numpy them move to torch
            input_t = np.random.random_sample(size=[3, 4, 4]).astype(self.supported_np_dtypes[idx])
            inputCPU = torch.tensor(input_t, device='cpu', dtype=self.supported_dtypes[idx])

            helper(inputCPU)

    def test_advanced_indexing_3D_put(self):
        def helper(x_cpu):
            dtype = x_cpu.dtype
            x_mps = x_cpu.detach().clone().to("mps")

            out_tensor_cpu = torch.tensor([88, 99], dtype=dtype, device="cpu")
            out_tensor_cpu_view = out_tensor_cpu[1:]

            out_tensor_mps = torch.tensor([88, 99], dtype=dtype, device="mps")
            out_tensor_mps_view = out_tensor_mps[1:]

            x_cpu[[1, 2], 3, :] = out_tensor_cpu_view
            x_mps[[1, 2], 3, :] = out_tensor_mps_view
            self.assertEqual(x_cpu, x_mps)

            x_cpu[[0, 2], :, :] = out_tensor_cpu_view
            x_mps[[0, 2], :, :] = out_tensor_mps_view
            self.assertEqual(x_cpu, x_mps)

            x_cpu[:, [1, 0], [1]] = out_tensor_cpu_view
            x_mps[:, [1, 0], [1]] = out_tensor_mps_view
            self.assertEqual(x_cpu, x_mps)

        x_cpu = torch.tensor([[[0.1, 0.2, 0.3, 0.4],
                               [0.5, 0.6, 0.7, 0.8],
                               [0.9, 1.0, 1.1, 1.2],
                               [1.3, 1.4, 1.5, 1.6]],

                              [[2.0, 2.1, 2.2, 2.3],
                               [2.4, 2.5, 2.6, 2.7],
                               [2.8, 2.9, 3.0, 3.1],
                               [3.2, 3.3, 3.4, 3.5]],

                              [[4.0, 4.1, 4.2, 4.3],
                               [4.4, 4.5, 4.6, 4.7],
                               [4.8, 4.9, 5.0, 5.1],
                               [5.1, 5.2, 5.3, 5.4]]], device="cpu", dtype=torch.float32)
        helper(x_cpu)
        for idx in range(len(self.supported_np_dtypes)):
            # torch.randn / torch.rand don't work with all dtypes
            # Generate input data for all dtypes on Numpy them move to torch
            input_t = np.random.random_sample(size=[3, 4, 4]).astype(self.supported_np_dtypes[idx])
            inputCPU = torch.tensor(input_t, device='cpu', dtype=self.supported_dtypes[idx])

            helper(inputCPU)

    def test_index_put_with_view_indices(self):
        def helper(dtype):
            target_cpu = torch.zeros([5, 3], device="cpu", dtype=dtype)
            target_mps = torch.zeros([5, 3], device="mps", dtype=dtype)

            indices_cpu = torch.tensor([[0, 1], [0, 1]], dtype=torch.int64, device="cpu")
            indices_mps = torch.tensor([[0, 1], [0, 1]], dtype=torch.int64, device="mps")

            value_cpu = torch.ones(indices_cpu.shape[0], device="cpu", dtype=dtype)
            value_mps = torch.ones(indices_mps.shape[0], device="mps", dtype=dtype)

            target_cpu.index_put_(tuple(indices_cpu.t()), value_cpu, accumulate=True)
            target_mps.index_put_(tuple(indices_mps.t()), value_mps, accumulate=True)

            self.assertEqual(target_cpu, target_mps)

        [helper(dtype) for dtype in [torch.int32, torch.float]]

    # tests from 'test_indexing.py'
    def test_advancedindex_big(self, device="mps"):
        reference = torch.arange(0, 123344, dtype=torch.int, device=device)

        self.assertEqual(reference[[0, 123, 44488, 68807, 123343], ],
                         torch.tensor([0, 123, 44488, 68807, 123343], dtype=torch.int))

    def test_set_item_to_scalar_tensor(self, device="mps"):
        m = random.randint(1, 10)
        n = random.randint(1, 10)
        z = torch.randn([m, n], device=device)
        a = 1.0
        w = torch.tensor(a, requires_grad=True, device=device)
        z[:, 0] = w
        z.sum().backward()
        self.assertEqual(w.grad, m * a)

    def test_single_int(self, device="mps"):
        v = torch.randn(5, 7, 3, device=device)
        self.assertEqual(v[4].shape, (7, 3))

    def test_multiple_int(self, device="mps"):
        v = torch.randn(5, 7, 3, device=device)
        self.assertEqual(v[4].shape, (7, 3))
        self.assertEqual(v[4, :, 1].shape, (7,))

    def test_none(self, device="mps"):
        v = torch.randn(5, 7, 3, device=device)
        self.assertEqual(v[None].shape, (1, 5, 7, 3))
        self.assertEqual(v[:, None].shape, (5, 1, 7, 3))
        self.assertEqual(v[:, None, None].shape, (5, 1, 1, 7, 3))
        self.assertEqual(v[..., None].shape, (5, 7, 3, 1))

    def test_step(self, device="mps"):
        v = torch.arange(10, device=device)
        self.assertEqual(v[::1], v)
        self.assertEqual(v[::2].tolist(), [0, 2, 4, 6, 8])
        self.assertEqual(v[::3].tolist(), [0, 3, 6, 9])
        self.assertEqual(v[::11].tolist(), [0])
        self.assertEqual(v[1:6:2].tolist(), [1, 3, 5])

    def test_step_assignment(self, device="mps"):
        v = torch.zeros(4, 4, device=device)
        v[0, 1::2] = torch.tensor([3., 4.], device=device)
        self.assertEqual(v[0].tolist(), [0, 3, 0, 4])
        self.assertEqual(v[1:].sum(), 0)

    def test_bool_indices(self, device="mps"):
        v = torch.randn(5, 7, 3, device=device)
        boolIndices = torch.tensor([True, False, True, True, False], dtype=torch.bool, device=device)
        self.assertEqual(v[boolIndices].shape, (3, 7, 3))
        self.assertEqual(v[boolIndices], torch.stack([v[0], v[2], v[3]]))

        v = torch.tensor([True, False, True], dtype=torch.bool, device=device)
        boolIndices = torch.tensor([True, False, False], dtype=torch.bool, device=device)
        uint8Indices = torch.tensor([1, 0, 0], dtype=torch.uint8, device=device)
        with warnings.catch_warnings(record=True) as w:
            self.assertEqual(v[boolIndices].shape, v[uint8Indices].shape)
            self.assertEqual(v[boolIndices], v[uint8Indices])
            self.assertEqual(v[boolIndices], torch.tensor([True], dtype=torch.bool, device=device))
            self.assertEqual(len(w), 2)

    def test_bool_indices_accumulate(self, device="mps"):
        mask = torch.zeros(size=(10, ), dtype=torch.uint8, device=device)
        mask = mask > 0
        y = torch.ones(size=(10, 10), device=device)
        y.index_put_((mask, ), y[mask], accumulate=True)
        self.assertEqual(y, torch.ones(size=(10, 10), device=device))

    def test_multiple_bool_indices(self, device="mps"):
        v = torch.randn(5, 7, 3, device=device)
        # note: these broadcast together and are transposed to the first dim
        mask1 = torch.tensor([1, 0, 1, 1, 0], dtype=torch.bool, device=device)
        mask2 = torch.tensor([1, 1, 1], dtype=torch.bool, device=device)
        self.assertEqual(v[mask1, :, mask2].shape, (3, 7))

    def test_byte_mask(self, device="mps"):
        v = torch.randn(5, 7, 3, device=device)
        mask = torch.ByteTensor([1, 0, 1, 1, 0]).to(device)
        with warnings.catch_warnings(record=True) as w:
            self.assertEqual(v[mask].shape, (3, 7, 3))
            self.assertEqual(v[mask], torch.stack([v[0], v[2], v[3]]))
            self.assertEqual(len(w), 2)

        v = torch.tensor([1.], device=device)
        self.assertEqual(v[v == 0], torch.tensor([], device=device))

    def test_byte_mask_accumulate(self, device="mps"):
        mask = torch.zeros(size=(10, ), dtype=torch.uint8, device=device)
        y = torch.ones(size=(10, 10), device=device)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            y.index_put_((mask, ), y[mask], accumulate=True)
            self.assertEqual(y, torch.ones(size=(10, 10), device=device))
            self.assertEqual(len(w), 2)

    def test_index_put_accumulate_expanded_values(self, device="mps"):
        t = torch.zeros((5, 2))
        t_dev = t.to(device)
        indices = [
            torch.tensor([0, 1, 2, 3]),
            torch.tensor([1, ]),
        ]
        indices_dev = [i.to(device) for i in indices]
        values0d = torch.tensor(1.0)
        values1d = torch.tensor([1.0, ])

        out_mps = t_dev.index_put_(indices_dev, values0d.to(device), accumulate=True)
        out_cpu = t.index_put_(indices, values0d, accumulate=True)
        self.assertEqual(out_mps.cpu(), out_cpu)

        out_mps = t_dev.index_put_(indices_dev, values1d.to(device), accumulate=True)
        out_cpu = t.index_put_(indices, values1d, accumulate=True)
        self.assertEqual(out_mps.cpu(), out_cpu)

        t = torch.zeros(4, 3, 2)
        t_dev = t.to(device)

        indices = [
            torch.tensor([0, ]),
            torch.arange(3)[:, None],
            torch.arange(2)[None, :],
        ]
        indices_dev = [i.to(device) for i in indices]
        values1d = torch.tensor([-1.0, -2.0])
        values2d = torch.tensor([[-1.0, -2.0], ])

        out_mps = t_dev.index_put_(indices_dev, values1d.to(device), accumulate=True)
        out_cpu = t.index_put_(indices, values1d, accumulate=True)
        self.assertEqual(out_mps.cpu(), out_cpu)

        out_mps = t_dev.index_put_(indices_dev, values2d.to(device), accumulate=True)
        out_cpu = t.index_put_(indices, values2d, accumulate=True)
        self.assertEqual(out_mps.cpu(), out_cpu)

    def test_index_put_accumulate_non_contiguous(self, device="mps"):
        t = torch.zeros((5, 2, 2))
        t_dev = t.to(device)
        t1 = t_dev[:, 0, :]
        t2 = t[:, 0, :]
        self.assertTrue(not t1.is_contiguous())
        self.assertTrue(not t2.is_contiguous())

        indices = [torch.tensor([0, 1]), ]
        indices_dev = [i.to(device) for i in indices]
        value = torch.randn(2, 2)
        out_mps = t1.index_put_(indices_dev, value.to(device), accumulate=True)
        out_cpu = t2.index_put_(indices, value, accumulate=True)
        self.assertTrue(not t1.is_contiguous())
        self.assertTrue(not t2.is_contiguous())

        self.assertEqual(out_mps.cpu(), out_cpu)

    def test_index_put_accumulate_with_optional_tensors(self, device="mps"):
        # TODO: replace with a better solution.
        # Currently, here using torchscript to put None into indices.
        # on C++ it gives indices as a list of 2 optional tensors: first is null and
        # the second is a valid tensor.
        @torch.jit.script
        def func(x, i, v):
            idx = [None, i]
            x.index_put_(idx, v, accumulate=True)
            return x

        n = 4
        t = torch.arange(n * 2, dtype=torch.float32).reshape(n, 2)
        t_dev = t.to(device)
        indices = torch.tensor([1, 0])
        indices_dev = indices.to(device)
        value0d = torch.tensor(10.0)
        value1d = torch.tensor([1.0, 2.0])

        out_mps = func(t_dev, indices_dev, value0d.to("mps"))
        out_cpu = func(t, indices, value0d)
        self.assertEqual(out_mps.cpu(), out_cpu)

        out_mps = func(t_dev, indices_dev, value1d.to("mps"))
        out_cpu = func(t, indices, value1d)
        self.assertEqual(out_mps.cpu(), out_cpu)

    def test_index_put_accumulate_duplicate_indices(self, device="mps"):
        for i in range(1, 128):
            # generate indices by random walk, this will create indices with
            # lots of duplicates interleaved with each other
            delta = torch.empty(i, dtype=torch.float32, device=device).uniform_(-1, 1)

            indices = delta.cumsum(0).long().to("mps")

            # abs for int64 is not supported on mps, fallback on 'cpu' to calculate it
            input = torch.randn(indices.cpu().abs().max().to("mps") + 1, device=device)
            values = torch.randn(indices.size(0), device=device)
            output = input.index_put((indices,), values, accumulate=True)

            input_list = input.tolist()
            indices_list = indices.tolist()
            values_list = values.tolist()
            for i, v in zip(indices_list, values_list):
                input_list[i] += v

            self.assertEqual(output, input_list)

    def test_multiple_byte_mask(self, device="mps"):
        v = torch.randn(5, 7, 3, device=device)
        # note: these broadcast together and are transposed to the first dim
        mask1 = torch.ByteTensor([1, 0, 1, 1, 0]).to(device)
        mask2 = torch.ByteTensor([1, 1, 1]).to(device)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.assertEqual(v[mask1, :, mask2].shape, (3, 7))
            self.assertEqual(len(w), 2)

    def test_byte_mask2d(self, device="mps"):
        v = torch.randn(5, 7, 3, device=device)
        c = torch.randn(5, 7, device=device)
        num_ones = (c > 0).sum()
        r = v[c > 0]
        self.assertEqual(r.shape, (num_ones, 3))

    # FIXME: conditional indexing not working
    # def test_jit_indexing(self, device="mps"):
    #     def fn1(x):
    #         x[x < 50] = 1.0
    #         return x

    #     def fn2(x):
    #         x[0:50] = 1.0
    #         return x

    #     scripted_fn1 = torch.jit.script(fn1)
    #     scripted_fn2 = torch.jit.script(fn2)
    #     data = torch.arange(100, device=device, dtype=torch.float)
    #     out = scripted_fn1(data.detach().clone())
    #     ref = torch.tensor(np.concatenate((np.ones(50), np.arange(50, 100))), device=device, dtype=torch.float)
    #     self.assertEqual(out, ref)
    #     out = scripted_fn2(data.detach().clone())
    #     self.assertEqual(out, ref)

    def test_int_indices(self, device="mps"):
        v = torch.randn(5, 7, 3, device=device)
        self.assertEqual(v[[0, 4, 2]].shape, (3, 7, 3))
        self.assertEqual(v[:, [0, 4, 2]].shape, (5, 3, 3))
        self.assertEqual(v[:, [[0, 1], [4, 3]]].shape, (5, 2, 2, 3))

    def test_index_put_src_datatype(self):
        def helper(device, dtype):
            src = torch.ones(3, 2, 4, device=device, dtype=dtype)
            vals = torch.ones(3, 2, 4, device=device, dtype=dtype)
            indices = (torch.tensor([0, 2, 1]),)
            res = src.index_put_(indices, vals, accumulate=True)
            self.assertEqual(res.shape, src.shape)
        [helper(device="mps", dtype=dtype) for dtype in [torch.float, torch.int32]]

    def test_index_src_datatype(self):
        def helper(device, dtype):
            orig_dtype = dtype
            if dtype is torch.bool:
                dtype = torch.uint8

            src = torch.ones(3, 2, 4, device=device, dtype=dtype)
            if orig_dtype is torch.bool:
                src = src == 1
            # test index
            res = src[[0, 2, 1], :, :]
            self.assertEqual(res.shape, src.shape)
            # test index_put, no accum
            src[[0, 2, 1], :, :] = res
            self.assertEqual(res.shape, src.shape)
        [helper(device="mps", dtype=dtype) for dtype in [torch.float, torch.float16, torch.long, torch.bool]]

    def test_int_indices2d(self, device="mps"):
        # From the NumPy indexing example
        x = torch.arange(0, 12, device=device).view(4, 3)
        rows = torch.tensor([[0, 0], [3, 3]], device=device)
        columns = torch.tensor([[0, 2], [0, 2]], device=device)
        self.assertEqual(x[rows, columns].tolist(), [[0, 2], [9, 11]])

    def test_int_indices_broadcast(self, device="mps"):
        # From the NumPy indexing example
        x = torch.arange(0, 12, device=device).view(4, 3)
        rows = torch.tensor([0, 3], device=device)
        columns = torch.tensor([0, 2], device=device)
        result = x[rows[:, None], columns]
        self.assertEqual(result.tolist(), [[0, 2], [9, 11]])

    def test_empty_index(self, device="mps"):
        x = torch.arange(0, 12, device=device).view(4, 3)
        idx = torch.tensor([], dtype=torch.long, device=device)
        self.assertEqual(x[idx].numel(), 0)

        # empty assignment should have no effect but not throw an exception
        y = x.clone()
        y[idx] = -1
        self.assertEqual(x, y)

        mask = torch.zeros(4, 3, device=device).bool()
        y[mask] = -1
        self.assertEqual(x, y)

    def test_empty_ndim_index(self, device="mps"):
        x = torch.randn(5, device=device)
        self.assertEqual(torch.empty(0, 2, device=device), x[torch.empty(0, 2, dtype=torch.int64, device=device)])

        x = torch.randn(2, 3, 4, 5, device=device)
        self.assertEqual(torch.empty(2, 0, 6, 4, 5, device=device),
                         x[:, torch.empty(0, 6, dtype=torch.int64, device=device)])

        x = torch.empty(10, 0, device=device)
        self.assertEqual(x[[1, 2]].shape, (2, 0))
        self.assertEqual(x[[], []].shape, (0,))
        with self.assertRaisesRegex(IndexError, 'for dimension with size 0'):
            x[:, [0, 1]]

    def test_empty_ndim_index_bool(self, device="mps"):
        x = torch.randn(5, device=device)
        self.assertRaises(IndexError, lambda: x[torch.empty(0, 2, dtype=torch.uint8, device=device)])

    def test_empty_slice(self, device="mps"):
        x = torch.randn(2, 3, 4, 5, device=device)
        y = x[:, :, :, 1]
        z = y[:, 1:1, :]
        self.assertEqual((2, 0, 4), z.shape)
        # this isn't technically necessary, but matches NumPy stride calculations.
        self.assertEqual((60, 20, 5), z.stride())
        self.assertTrue(z.is_contiguous())

    def test_index_getitem_copy_bools_slices(self, device="mps"):
        true = torch.tensor(1, dtype=torch.uint8, device=device)
        false = torch.tensor(0, dtype=torch.uint8, device=device)

        tensors = [torch.randn(2, 3, device=device), torch.tensor(3., device=device)]

        for a in tensors:
            self.assertNotEqual(a.data_ptr(), a[True].data_ptr())
            self.assertEqual(torch.empty(0, *a.shape), a[False])
            self.assertNotEqual(a.data_ptr(), a[true].data_ptr())
            self.assertEqual(torch.empty(0, *a.shape), a[false])
            self.assertEqual(a.data_ptr(), a[None].data_ptr())
            self.assertEqual(a.data_ptr(), a[...].data_ptr())

    def test_index_setitem_bools_slices(self, device="mps"):
        true = torch.tensor(1, dtype=torch.uint8, device=device)
        false = torch.tensor(0, dtype=torch.uint8, device=device)

        tensors = [torch.randn(2, 3, device=device), torch.tensor(3, device=device)]

        for a in tensors:
            # prefix with a 1,1, to ensure we are compatible with numpy which cuts off prefix 1s
            # (some of these ops already prefix a 1 to the size)
            neg_ones = torch.ones_like(a) * -1
            neg_ones_expanded = neg_ones.unsqueeze(0).unsqueeze(0)
            a[True] = neg_ones_expanded
            self.assertEqual(a, neg_ones)
            a[False] = 5
            self.assertEqual(a, neg_ones)
            a[true] = neg_ones_expanded * 2
            self.assertEqual(a, neg_ones * 2)
            a[false] = 5
            self.assertEqual(a, neg_ones * 2)
            a[None] = neg_ones_expanded * 3
            self.assertEqual(a, neg_ones * 3)
            a[...] = neg_ones_expanded * 4
            self.assertEqual(a, neg_ones * 4)
            if a.dim() == 0:
                with self.assertRaises(IndexError):
                    a[:] = neg_ones_expanded * 5

    def test_index_scalar_with_bool_mask(self, device="mps"):
        a = torch.tensor(1, device=device)
        uintMask = torch.tensor(True, dtype=torch.uint8, device=device)
        boolMask = torch.tensor(True, dtype=torch.bool, device=device)
        self.assertEqual(a[uintMask], a[boolMask])
        self.assertEqual(a[uintMask].dtype, a[boolMask].dtype)

        a = torch.tensor(True, dtype=torch.bool, device=device)
        self.assertEqual(a[uintMask], a[boolMask])
        self.assertEqual(a[uintMask].dtype, a[boolMask].dtype)

    def test_setitem_expansion_error(self, device="mps"):
        true = torch.tensor(True, device=device)
        a = torch.randn(2, 3, device=device)
        # check prefix with  non-1s doesn't work
        a_expanded = a.expand(torch.Size([5, 1]) + a.size())
        # NumPy: ValueError
        with self.assertRaises(RuntimeError):
            a[True] = a_expanded
        with self.assertRaises(RuntimeError):
            a[true] = a_expanded

    def test_getitem_scalars(self, device="mps"):
        zero = torch.tensor(0, dtype=torch.int64, device=device)
        one = torch.tensor(1, dtype=torch.int64, device=device)

        # non-scalar indexed with scalars
        a = torch.randn(2, 3, device=device)
        self.assertEqual(a[0], a[zero])
        self.assertEqual(a[0][1], a[zero][one])
        self.assertEqual(a[0, 1], a[zero, one])
        self.assertEqual(a[0, one], a[zero, 1])

        # indexing by a scalar should slice (not copy)
        self.assertEqual(a[0, 1].data_ptr(), a[zero, one].data_ptr())
        self.assertEqual(a[1].data_ptr(), a[one.int()].data_ptr())
        self.assertEqual(a[1].data_ptr(), a[one.short()].data_ptr())

        # scalar indexed with scalar
        r = torch.randn((), device=device)
        with self.assertRaises(IndexError):
            r[:]
        with self.assertRaises(IndexError):
            r[zero]
        self.assertEqual(r, r[...])

    def test_setitem_scalars(self, device="mps"):
        zero = torch.tensor(0, dtype=torch.int64)

        # non-scalar indexed with scalars
        a = torch.randn(2, 3, device=device)
        a_set_with_number = a.clone()
        a_set_with_scalar = a.clone()
        b = torch.randn(3, device=device)

        a_set_with_number[0] = b
        a_set_with_scalar[zero] = b
        self.assertEqual(a_set_with_number, a_set_with_scalar)
        a[1, zero] = 7.7
        self.assertEqual(7.7, a[1, 0])

        # scalar indexed with scalars
        r = torch.randn((), device=device)
        with self.assertRaises(IndexError):
            r[:] = 8.8
        with self.assertRaises(IndexError):
            r[zero] = 8.8
        r[...] = 9.9
        self.assertEqual(9.9, r)

    def test_basic_advanced_combined(self, device="mps"):
        # From the NumPy indexing example
        x = torch.arange(0, 12, device=device).view(4, 3)
        self.assertEqual(x[1:2, 1:3], x[1:2, [1, 2]])
        self.assertEqual(x[1:2, 1:3].tolist(), [[4, 5]])

        # Check that it is a copy
        unmodified = x.clone()
        x[1:2, [1, 2]].zero_()
        self.assertEqual(x, unmodified)

        # But assignment should modify the original
        unmodified = x.clone()
        x[1:2, [1, 2]] = 0
        self.assertNotEqual(x, unmodified)

    def test_int_assignment(self, device="mps"):
        x = torch.arange(0, 4, device=device).view(2, 2)
        x[1] = 5
        self.assertEqual(x.tolist(), [[0, 1], [5, 5]])

        x = torch.arange(0, 4, device=device).view(2, 2)
        x[1] = torch.arange(5, 7, device=device)
        self.assertEqual(x.tolist(), [[0, 1], [5, 6]])

    def test_byte_tensor_assignment(self, device="mps"):
        x = torch.arange(0., 16, device=device).view(4, 4)
        b = torch.ByteTensor([True, False, True, False]).to(device)
        value = torch.tensor([3., 4., 5., 6.], device=device)

        with warnings.catch_warnings(record=True) as w:
            x[b] = value
            self.assertEqual(len(w), 1)

        self.assertEqual(x[0], value)
        self.assertEqual(x[1], torch.arange(4., 8, device=device))
        self.assertEqual(x[2], value)
        self.assertEqual(x[3], torch.arange(12., 16, device=device))

    def test_variable_slicing(self, device="mps"):
        x = torch.arange(0, 16, device=device).view(4, 4)
        indices = torch.IntTensor([0, 1]).to(device)
        i, j = indices
        self.assertEqual(x[i:j], x[0:1])

    def test_ellipsis_tensor(self, device="mps"):
        x = torch.arange(0, 9, device=device).view(3, 3)
        idx = torch.tensor([0, 2], device=device)
        self.assertEqual(x[..., idx].tolist(), [[0, 2],
                                                [3, 5],
                                                [6, 8]])
        self.assertEqual(x[idx, ...].tolist(), [[0, 1, 2],
                                                [6, 7, 8]])

    def test_invalid_index(self, device="mps"):
        x = torch.arange(0, 16, device=device).view(4, 4)
        self.assertRaisesRegex(TypeError, 'slice indices', lambda: x["0":"1"])

    def test_out_of_bound_index(self, device="mps"):
        x = torch.arange(0, 100, device=device).view(2, 5, 10)
        self.assertRaisesRegex(IndexError, 'index 5 is out of bounds for dimension 1 with size 5', lambda: x[0, 5])
        self.assertRaisesRegex(IndexError, 'index 4 is out of bounds for dimension 0 with size 2', lambda: x[4, 5])
        self.assertRaisesRegex(IndexError, 'index 15 is out of bounds for dimension 2 with size 10',
                               lambda: x[0, 1, 15])
        self.assertRaisesRegex(IndexError, 'index 12 is out of bounds for dimension 2 with size 10',
                               lambda: x[:, :, 12])

    def test_zero_dim_index(self, device="mps"):
        x = torch.tensor(10, device=device)
        self.assertEqual(x, x.item())

        def runner():
            print(x[0])
            return x[0]

        self.assertRaisesRegex(IndexError, 'invalid index', runner)

    def test_cpu_indices(self, device="mps"):
        idx = torch.tensor([0, 1])
        b = torch.zeros(2, device=device)
        x = torch.ones(10, device=device)
        x[idx] = b  # index_put_
        ref = torch.ones(10, device=device)
        ref[:2] = 0
        self.assertEqual(x, ref, atol=0, rtol=0)
        out = x[idx]  # index
        self.assertEqual(out, torch.zeros(2, device=device), atol=0, rtol=0)

class TestRNNMPS(TestCase):
    def test_lstm_1(self, device="mps", dtype=torch.float32):

        rnn = nn.LSTM(1, 4, 2, device="cpu")
        input = torch.randn(2, 3, 1, device="cpu")
        hx = torch.zeros(2, 3, 4, device="cpu")
        cx = torch.zeros(2, 3, 4, device="cpu")

        cpu_output, (cpu_hn, cpu_cn) = rnn(input, (hx, cx))

        rnn = rnn.to(device)
        input = input.to(device)
        hx = hx.to(device)
        cx = cx.to(device)
        output, (hn, cn) = rnn(input, (hx, cx))

        self.assertEqual(cpu_output, output)
        self.assertEqual(cpu_hn, hn)
        self.assertEqual(cpu_cn, cn)

        # test batch_first
        rnn = nn.LSTM(1, 4, 2, device="cpu", batch_first=True)
        input = torch.randn(3, 2, 1, device="cpu")
        hx = torch.zeros(2, 3, 4, device="cpu")
        cx = torch.zeros(2, 3, 4, device="cpu")
        cpu_output, (cpu_hn, cpu_cn) = rnn(input, (hx, cx))

        rnn = rnn.to(device)
        input = input.to(device)
        hx = hx.to(device)
        cx = cx.to(device)
        output, (hn, cn) = rnn(input, (hx, cx))

        self.assertEqual(cpu_output, output)
        self.assertEqual(cpu_hn, hn)
        self.assertEqual(cpu_cn, cn)

    @unittest.skipIf(True, "Backward of lstm returns wrong result")
    def test_lstm_2(self, device="mps", dtype=torch.float32):
        def get_results(device):
            rnn = nn.LSTM(1, 4, 1, device=device)
            inp = torch.randn(2, 3, 1, device=device, requires_grad=True)
            hx = torch.zeros(1, 3, 4, device=device)
            cx = torch.zeros(1, 3, 4, device=device)

            output, _ = rnn(inp, (hx, cx))
            output.sum().backward()

            weight_grad = rnn.weight_ih_l0.grad.clone()
            input_grad = inp.grad.clone()

            return output, weight_grad, input_grad


        cpu_output, cpu_weight_grad, cpu_input_grad = get_results("cpu")
        mps_output, mps_weight_grad, mps_input_grad = get_results("mps")

        self.assertEqual(cpu_output, mps_output)
        self.assertEqual(cpu_input_grad, mps_input_grad)
        self.assertEqual(cpu_weight_grad, mps_weight_grad)

class TestFallbackWarning(TestCase):
    # TODO: Remove once test_testing.py is running on MPS devices
    def test_no_warning_on_import(self):
        out = subprocess.check_output(
            [sys.executable, "-W", "all", "-c", "import torch"],
            stderr=subprocess.STDOUT,
            # On Windows, opening the subprocess with the default CWD makes `import torch`
            # fail, so just set CWD to this script's directory
            cwd=os.path.dirname(os.path.realpath(__file__)),).decode("utf-8")
        self.assertEquals(out, "")

    def _get_not_implemented_op(self):
        # This can be changed once we actually implement `torch.bincount`
        # Should return fn, args, kwargs, string_version
        return (torch.bincount,
                torch.tensor([4], device='mps'), {},
                "torch.bincount(torch.tensor([4, 3, 6, 3, 4], device='mps'))")

    def test_error_on_not_implemented(self):
        fn, args, kwargs, _ = self._get_not_implemented_op()

        with self.assertRaisesRegex(NotImplementedError, "not currently implemented for the MPS device"):
            fn(*args, **kwargs)

    def test_warn_on_not_implemented_with_fallback(self):
        _, _, _, op = self._get_not_implemented_op()
        script = f"""
import os
# MUST happen before pytorch's import
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import warnings

with warnings.catch_warnings(record=True) as w:
    import torch

if len(w) > 0:
    print(w)
    exit(1)

# This should run just fine and raise warning about perf
with warnings.catch_warnings(record=True) as w:
    {op}

if len(w) != 1:
    print(w)
    exit(2)

"""
        try:
            subprocess.check_output(
                [sys.executable, '-W', 'all', '-c', script],
                stderr=subprocess.STDOUT,
                # On Windows, opening the subprocess with the default CWD makes `import torch`
                # fail, so just set CWD to this script's directory
                cwd=os.path.dirname(os.path.realpath(__file__)),)
        except subprocess.CalledProcessError as e:
            if e.returncode == 1:
                self.assertTrue(False, "There was a warning when importing torch when PYTORCH_ENABLE_MPS_FALLBACK is set." +
                                       e.output.decode("utf-8"))
            elif e.returncode == 2:
                self.assertTrue(False, "There wasn't exactly one warning when running not implemented op with "
                                f"PYTORCH_ENABLE_MPS_FALLBACK set. {e.output}")
            else:
                self.assertTrue(False, "Running a not implemented op failed even though PYTORCH_ENABLE_MPS_FALLBACK is set. " +
                                       e.output.decode("utf-8"))

class TestNoRegression(TestCase):
    def test_assert_close(self):
        a = torch.ones(1, device="mps")
        b = torch.zeros(1, device="mps")
        inf = a / b
        nan = b / b

        with self.assertRaisesRegex(AssertionError, "Tensor-likes are not close!"):
            torch.testing.assert_close(a, inf)

        # TODO: The NaN test is failing when all the tests in test_mps are run
        # together but passes when run separately. There seems to be memory
        # corruption which needs to be fixed for this test to be enabled.
        # with self.assertRaisesRegex(AssertionError, "Tensor-likes are not close!"):
            # torch.testing.assert_close(a, nan)

    @unittest.expectedFailure
    def test_mps_compat(self):
        # If this test is successful, that means that all operations in the comparison logic are supported natively on
        # the MPS backend. Please remove this test as well as the compatibility logic in
        # torch.testing._comparison.TensorLikePair._equalize_attributes
        actual = torch.tensor(1.0, device="mps")
        expected = actual.clone()

        # We can't use assert_close or TensorLikePair.compare() directly, since that would hit the compatibility logic
        # in torch.testing._comparison.TensorLikePair._equalize_attributes that we want to circumvent here
        pair = TensorLikePair(actual, expected)
        pair._compare_values(actual, expected)

    def test_double_error(self):
        with self.assertRaisesRegex(TypeError, "the MPS framework doesn't support float64"):
            a = torch.ones(2, dtype=torch.float64, device="mps")

        a = torch.ones(2, device="mps")
        with self.assertRaisesRegex(TypeError, "the MPS framework doesn't support float64"):
            a = a.double()

    def test_legacy_constructor(self):
        a = torch.ones(2, device="mps")

        b = a.new(1)

    def test_serialization_map_location(self):

        # Ensures that cpu Tensor can be loaded on mps
        with tempfile.NamedTemporaryFile() as f:
            x = torch.rand(2)
            torch.save(x, f)

            f.seek(0)
            x2 = torch.load(f, map_location="mps")

            self.assertEqual(x, x2)
            self.assertEqual(x2.device.type, "mps")

        # Ensures that mps Tensors can be loaded on mps
        with tempfile.NamedTemporaryFile() as f:
            x = torch.rand(2, device="mps")
            torch.save(x, f)

            f.seek(0)
            x2 = torch.load(f)

            self.assertEqual(x, x2)
            self.assertEqual(x2.device.type, "mps")

        # Ensures that mps Tensors can be loaded on cpu
        with tempfile.NamedTemporaryFile() as f:
            x = torch.rand(2, device="mps")
            torch.save(x, f)

            f.seek(0)
            x2 = torch.load(f, map_location="cpu")

            self.assertEqual(x, x2)
            self.assertEqual(x2.device.type, "cpu")


MPS_DTYPES = get_all_dtypes()
for t in [torch.double, torch.cdouble, torch.cfloat, torch.int8, torch.bfloat16]:
    del MPS_DTYPES[MPS_DTYPES.index(t)]


class TestConsistency(TestCase):
    # TODO: This is only used while some ops are being added.
    # This list should contain all ops and dtypes eventually
    # This can be generated automatically in the `new_mps_allowlist.txt` file
    # by doing `EXPECTTEST_ACCEPT=1 python test_mps.py TestConsistencyCPU`
    # You most likely do NOT want to modify this manually
    ALLOWLIST_OP = {
        '__getitem__': ['b8', 'f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        '__radd__': ['b8', 'f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        '__rand__': ['b8', 'i16', 'i32', 'i64', 'u8'],
        '__rdiv__': ['f16', 'f32', 'i16', 'i32', 'u8'],
        '__rmatmul__': ['f32'],
        '__rmul__': ['b8', 'f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        '__ror__': ['b8', 'i16', 'i32', 'i64', 'u8'],
        '__rpow__': ['f16'],
        '__rxor__': ['b8', 'i16', 'i32', 'i64', 'u8'],
        'masked.argmax': ['i16', 'i64', 'u8'],
        'masked.argmin': ['i16', 'i64', 'u8'],
        'masked.log_softmax': ['f32'],
        'masked.logaddexp': ['f32'],
        'masked.norm': ['f16', 'f32'],
        'masked.normalize': ['f16', 'f32'],
        'masked.softmax': ['f32'],
        'masked.softmin': ['f32'],
        'masked.std': ['f32'],
        'masked.var': ['f32'],
        'abs': ['f16', 'f32', 'i16', 'i32', 'u8'],
        'acos': ['f32', 'i16', 'i32', 'u8'],
        'acosh': ['f32', 'i16', 'i32', 'u8'],
        'add': ['b8', 'f16', 'f32', 'i16', 'i32', 'i64'],
        'addbmm': ['f32'],
        'addcdiv': ['f32'],
        'addcmul': ['f32', 'i16', 'i32', 'i64', 'u8'],
        'addmm': ['f32'],
        'addmv': ['f32'],
        'addr': ['b8', 'f32', 'i16', 'i32', 'i64', 'u8'],
        'all': ['b8', 'f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        'allclose': ['f16', 'f32'],
        'any': ['b8', 'f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        'arange': ['f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        'argmax': ['f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        'argmin': ['f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        'amax': ['f32'],
        'amix': ['f32'],
        'logsumexp': ['f32'],
        'mean': ['f32'],
        'sum': ['f32'],
        'asin': ['f32', 'i16', 'i32', 'u8'],
        'asinh': ['f32', 'i16', 'i32', 'u8'],
        'atan': ['f32', 'i16', 'i32', 'u8'],
        'atan2': ['f32'],
        'atanh': ['f32', 'i16', 'i32', 'u8'],
        'atleast_1d': ['b8', 'f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        'atleast_2d': ['b8', 'f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        'atleast_3d': ['b8', 'f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        'baddbmm': ['f32'],
        'bitwise_and': ['b8', 'i16', 'i32', 'i64', 'u8'],
        'bitwise_left_shift': ['i16', 'i32', 'i64', 'u8'],
        'bitwise_not': ['b8', 'i16', 'i32', 'i64', 'u8'],
        'bitwise_or': ['b8', 'i16', 'i32', 'i64', 'u8'],
        'bitwise_right_shift': ['i16', 'i32', 'i64', 'u8'],
        'bitwise_xor': ['b8', 'i16', 'i32', 'i64', 'u8'],
        'block_diag': ['b8', 'f16', 'f32', 'i16', 'i32', 'i64'],
        'bmm': ['f32'],
        'broadcast_shapes': ['f32'],
        'cat': ['b8', 'f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        'ceil': ['f32', 'int32', 'int64', 'f16'],
        'char': ['b8', 'u8'],
        'chunk': ['b8', 'f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        'clone': ['b8', 'f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        'column_stack': ['b8', 'f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        'combinations': ['b8', 'f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        'conj': ['b8', 'f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        'conj_physical': ['b8', 'f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        'contiguous': ['b8', 'f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        'corrcoef': ['f32'],
        'cos': ['f32', 'i16', 'i32', 'u8', 'i64'],
        'cosh': ['f32', 'i16', 'i32', 'u8', 'i64'],
        'cov': ['f32'],
        'cumsum': ['f16', 'f32', 'int16', 'int32'],
        'deg2rad': ['b8', 'f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        'diag': ['f32', 'i32'],
        'diag_embed': ['b8', 'f16', 'f32', 'i16', 'i32', 'i64'],
        'diagflat': ['f32', 'i32'],
        'diagonal_scatter': ['b8', 'f16', 'f32', 'i16', 'i32', 'i64'],
        'diff': ['f16', 'f32', 'i16', 'i32', 'i64'],
        'dist': ['f32'],
        'dot': ['f32', 'i16', 'i32', 'i64', 'u8'],
        'einsum': ['f32'],
        'equal': ['b8', 'f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        'erf': ['f32', 'i16', 'i32', 'u8'],
        'exp': ['f32', 'i16', 'i32', 'u8'],
        'exp2': ['f16', 'f32', 'i16', 'i32', 'u8'],
        'eye': ['b8', 'f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        'fill': ['b8', 'f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        'flatten': ['b8', 'f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        'flip': ['f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        'fliplr': ['f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        'flipud': ['f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        'float': ['f32'],
        'floor': ['f32', 'f16', 'i16', 'i32', 'i64'],
        'frac': ['f16', 'f32'],
        'gradient': ['f16', 'f32', 'i16'],
        'half': ['f16'],
        'hstack': ['b8', 'f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        'index_select': ['f32', 'i16', 'i32', 'i64'],
        'int': ['i32'],
        'isclose': ['b8', 'f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        'isfinite': ['b8', 'f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        'isinf': ['b8', 'f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        'isnan': ['b8', 'f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        'isreal': ['b8', 'f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        'kron': ['b8', 'f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        'linalg.matrix_norm': ['f16'],
        'linalg.svd': ['f32'],
        'linalg.vector_norm': ['f16', 'f32'],
        'linspace': ['f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        'log': ['f32', 'i16', 'i32', 'u8'],
        'log10': ['f32', 'i16', 'i32', 'u8'],
        'log2': ['f32', 'i16', 'i32', 'u8'],
        'log_softmax': ['f32'],
        'logaddexp': ['f32'],
        'logaddexp2': ['f32'],
        'logical_not': ['b8', 'f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        'logspace': ['f32', 'i16', 'i32', 'i64', 'u8'],
        'masked_fill': ['f16', 'i16', 'i32', 'i64'],
        'masked_select': ['b8', 'f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        'matmul': ['f32'],
        'mm': ['f32'],
        'mv': ['f32'],
        'neg': ['f16', 'f32', 'i16', 'i32', 'i64'],
        'nn.functional.adaptive_max_pool1d': ['f32'],
        'nn.functional.adaptive_max_pool2d': ['f32'],
        'nn.functional.binary_cross_entropy': ['f32'],
        'nn.functional.binary_cross_entropy_with_logits': ['f32'],
        'nn.functional.celu': ['f32'],
        'nn.functional.conv1d': ['f32'],
        'nn.functional.conv2d': ['f32'],
        'nn.functional.conv_transpose1d': ['f32'],
        'nn.functional.cosine_embedding_loss': ['b8',
                                                'f32',
                                                'i16',
                                                'i32',
                                                'i64'],
        'nn.functional.elu': ['f32'],
        'nn.functional.feature_alpha_dropout': ['b8',
                                                'f16',
                                                'f32',
                                                'i16',
                                                'i32',
                                                'i64',
                                                'u8'],
        'nn.functional.gaussian_nll_loss': ['f32'],
        'nn.functional.glu': ['f32'],
        'nn.functional.group_norm': ['f32'],
        'nn.functional.hardtanh': ['f32', 'i16', 'i32', 'i64'],
        'nn.functional.hinge_embedding_loss': ['f32'],
        'nn.functional.huber_loss': ['f32'],
        'nn.functional.instance_norm': ['f32'],
        'nn.functional.kl_div': ['f32'],
        'nn.functional.l1_loss': ['f16', 'f32'],
        'nn.functional.leaky_relu': ['f32'],
        'nn.functional.linear': ['f32'],
        'nn.functional.local_response_norm': ['f32'],
        'nn.functional.margin_ranking_loss': ['f32', 'i16', 'i32'],
        'nn.functional.mse_loss': ['f16', 'f32'],
        'nn.functional.pad': ['b8', 'f16', 'f32', 'i16', 'i32', 'i64'],
        'nn.functional.pairwise_distance': ['f16',
                                            'f32',
                                            'i16',
                                            'i32',
                                            'i64'],
        'nn.functional.poisson_nll_loss': ['f32', 'i16', 'i32', 'u8'],
        'nn.functional.prelu': ['f32'],
        'nn.functional.relu': ['f32', 'i16', 'i32', 'i64', 'u8'],
        'nn.functional.relu6': ['f32', 'i16', 'i32', 'i64', 'u8'],
        'nn.functional.selu': ['f32'],
        'nn.functional.silu': ['f32'],
        'nn.functional.smooth_l1_loss': ['f16', 'f32'],
        'nn.functional.soft_margin_loss': ['f32'],
        'nn.functional.softmin': ['f32'],
        'nn.functional.softsign': ['f16', 'f32', 'i16', 'u8'],
        'nn.functional.tanhshrink': ['f32', 'i16', 'i32', 'u8'],
        'nn.functional.threshold': ['f32', 'i16', 'i32', 'i64', 'u8'],
        'nn.functional.triplet_margin_loss': ['f32', 'i16', 'i32', 'i64'],
        'nn.functional.triplet_margin_with_distance_loss': ['f32',
                                                            'i16',
                                                            'i32',
                                                            'i64'],
        'nn.functional.upsample_bilinear': ['f32'],
        'norm': ['f32', 'f16'],
        'positive': ['f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        'pow': ['f16'],
        'rad2deg': ['b8', 'f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        'real': ['b8', 'f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        'reciprocal': ['f16', 'f32', 'i16', 'i32', 'u8'],
        'repeat': ['f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        'repeat_interleave': ['b8',
                              'f16',
                              'f32',
                              'i16',
                              'i32',
                              'i64',
                              'u8'],
        'resize_': ['b8', 'i16', 'i32', 'i64', 'u8'],
        'resize_as_': ['b8', 'i16', 'i32', 'i64', 'u8'],
        'resolve_conj': ['b8', 'f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        'resolve_neg': ['b8', 'f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        'rot90': ['f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        'round': ['f32', 'f16', 'i16', 'i32', 'i64'],
        'rsqrt': ['f32', 'i16', 'i32', 'u8'],
        'select_scatter': ['b8', 'f16', 'f32', 'i16', 'i32', 'i64'],
        'sgn': ['b8', 'f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        'short': ['i16'],
        'sigmoid': ['f32'],
        'sign': ['b8', 'f16', 'f32', 'i16', 'i32', 'u8', 'i64'],
        'sin': ['f32', 'i16', 'i32', 'u8'],
        'sinh': ['f32', 'i16', 'i32', 'u8'],
        'slice_scatter': ['b8', 'f16', 'f32', 'i16', 'i32', 'i64'],
        'softmax': ['f32'],
        'special.ndtr': ['b8', 'f32', 'i16', 'i32', 'i64', 'u8'],
        'split': ['b8', 'f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        'sqrt': ['f32', 'i16', 'i32', 'u8'],
        'square': ['f16', 'f32'],
        'squeeze': ['b8', 'f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        'stack': ['b8', 'f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        'sub': ['f32', 'i16', 'i32', 'i64'],
        'sum_to_size': ['b8', 'f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        'svd': ['f32'],
        't': ['b8', 'f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        'tan': ['i16', 'i32', 'u8'],
        'tanh': ['f32', 'i16', 'i32', 'u8'],
        'tensordot': ['f32'],
        'tile': ['f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        'topk': ['f32'],
        'trapz': ['f16', 'f32', 'i16', 'i32', 'i64'],
        'tril': ['b8', 'f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        'tril_indices': ['i32', 'i64'],
        'triu': ['b8', 'f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        'triu_indices': ['i32', 'i64'],
        'true_divide': ['b8', 'f16', 'f32', 'i16', 'u8'],
        'trunc': ['f32'],
        'unbind': ['b8', 'f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        'unflatten': ['b8', 'f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        'unsqueeze': ['b8', 'f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        'view': ['b8', 'f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        'view_as': ['b8', 'f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        'vsplit': ['b8', 'f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        'vstack': ['b8', 'f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        'zero_': ['b8', 'f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        'clamp': ['f32', 'i16', 'i32', 'i64', 'u8'],
        'clamp_max': ['b8', 'f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        'clamp_min': ['b8', 'f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        'logical_and': ['b8', 'f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        'logical_or': ['b8', 'f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        'logical_xor': ['b8', 'f16', 'f32', 'i16', 'i32', 'i64', 'u8'],
        'where': ['f16', 'f32', 'i16', 'i32', 'i64', 'u8']}


    ALLOWLIST_OP_GRAD = {
        '__radd__': ['f16', 'f32'],
        '__rdiv__': ['f16', 'f32'],
        '__rmatmul__': ['f32'],
        '__rmul__': ['f16', 'f32'],
        'masked.log_softmax': ['f32'],
        'masked.logaddexp': ['f32'],
        'masked.softmax': ['f32'],
        'masked.softmin': ['f32'],
        'masked.std': ['f32'],
        'masked.var': ['f32'],
        'abs': ['f16', 'f32'],
        'acos': ['f32'],
        'acosh': ['f32'],
        'add': ['f16', 'f32'],
        'addbmm': ['f32'],
        'addcdiv': ['f32'],
        'addcmul': ['f32'],
        'addmm': ['f32'],
        'addmv': ['f32'],
        'addr': ['f32'],
        'all': ['f16', 'f32'],
        'any': ['f16', 'f32'],
        'arange': ['f16', 'f32'],
        'argmax': ['f16', 'f32'],
        'argmin': ['f16', 'f32'],
        'asin': ['f32'],
        'asinh': ['f32'],
        'atan': ['f32'],
        'atan2': ['f32'],
        'atleast_1d': ['f16', 'f32'],
        'atleast_2d': ['f16', 'f32'],
        'atleast_3d': ['f16', 'f32'],
        'baddbmm': ['f32'],
        'block_diag': ['f16', 'f32'],
        'bmm': ['f32'],
        'broadcast_shapes': ['f32'],
        'cat': ['f16', 'f32'],
        'ceil': ['f32'],
        'chunk': ['f16', 'f32'],
        'clone': ['f16', 'f32'],
        'column_stack': ['f16', 'f32'],
        'conj': ['f16', 'f32'],
        'conj_physical': ['f16', 'f32'],
        'contiguous': ['f16', 'f32'],
        'corrcoef': ['f32'],
        'cos': ['f32'],
        'cosh': ['f32'],
        'cumsum': ['f16', 'f32'],
        'deg2rad': ['f16', 'f32'],
        'diag': ['f32'],
        'diag_embed': ['f16', 'f32'],
        'diagflat': ['f32'],
        'diagonal_scatter': ['f16', 'f32'],
        'diff': ['f16', 'f32'],
        'dist': ['f32'],
        'dot': ['f32'],
        'einsum': ['f32'],
        'erf': ['f32'],
        'exp': ['f32'],
        'exp2': ['f16', 'f32'],
        'fill': ['f16', 'f32'],
        'flatten': ['f16', 'f32'],
        'flip': ['f16', 'f32'],
        'fliplr': ['f16', 'f32'],
        'flipud': ['f16', 'f32'],
        'float': ['f32'],
        'floor': ['f32'],
        'gradient': ['f32'],
        'half': ['f16'],
        'hstack': ['f16', 'f32'],
        'index_select': ['f32'],
        'isclose': ['f16', 'f32'],
        'isfinite': ['f16', 'f32'],
        'isinf': ['f16', 'f32'],
        'isnan': ['f16', 'f32'],
        'isreal': ['f16', 'f32'],
        'kron': ['f32'],
        'linalg.matrix_norm': ['f16'],
        'linalg.svd': ['f32'],
        'linspace': ['f16', 'f32'],
        'log': ['f32'],
        'log10': ['f32'],
        'log2': ['f32'],
        'log_softmax': ['f32'],
        'logaddexp': ['f32'],
        'logical_not': ['f16', 'f32'],
        'logspace': ['f32'],
        'matmul': ['f32'],
        'mm': ['f32'],
        'mv': ['f32'],
        'neg': ['f16', 'f32'],
        'nn.functional.adaptive_max_pool1d': ['f32'],
        'nn.functional.adaptive_max_pool2d': ['f32'],
        'nn.functional.binary_cross_entropy': ['f32'],
        'nn.functional.celu': ['f32'],
        'nn.functional.conv1d': ['f32'],
        'nn.functional.conv2d': ['f32'],
        'nn.functional.conv_transpose1d': ['f32'],
        'nn.functional.cosine_embedding_loss': ['f32'],
        'nn.functional.elu': ['f32'],
        'nn.functional.feature_alpha_dropout': ['f16', 'f32'],
        'nn.functional.glu': ['f32'],
        'nn.functional.hardtanh': ['f32'],
        'nn.functional.hinge_embedding_loss': ['f32'],
        'nn.functional.huber_loss': ['f32'],
        'nn.functional.instance_norm': ['f32'],
        'nn.functional.kl_div': ['f32'],
        'nn.functional.l1_loss': ['f16', 'f32'],
        'nn.functional.leaky_relu': ['f32'],
        'nn.functional.local_response_norm': ['f32'],
        'nn.functional.margin_ranking_loss': ['f32'],
        'nn.functional.mse_loss': ['f32'],
        'nn.functional.pad': ['f16', 'f32'],
        'nn.functional.pairwise_distance': ['f16', 'f32'],
        'nn.functional.poisson_nll_loss': ['f32'],
        'nn.functional.relu': ['f32'],
        'nn.functional.relu6': ['f32'],
        'nn.functional.selu': ['f32'],
        'nn.functional.silu': ['f32'],
        'nn.functional.soft_margin_loss': ['f32'],
        'nn.functional.softmin': ['f32'],
        'nn.functional.softsign': ['f16', 'f32'],
        'nn.functional.threshold': ['f32'],
        'nn.functional.triplet_margin_loss': ['f32'],
        'nn.functional.triplet_margin_with_distance_loss': ['f32'],
        'nn.functional.upsample_bilinear': ['f32'],
        'norm': ['f32', 'f16'],
        'positive': ['f16', 'f32'],
        'rad2deg': ['f16', 'f32'],
        'real': ['f16', 'f32'],
        'reciprocal': ['f16', 'f32'],
        'repeat': ['f16', 'f32'],
        'repeat_interleave': ['f16', 'f32'],
        'resolve_conj': ['f16', 'f32'],
        'resolve_neg': ['f16', 'f32'],
        'round': ['f32'],
        'rsqrt': ['f32'],
        'select_scatter': ['f16', 'f32'],
        'sign': ['f16', 'f32'],
        'sin': ['f32'],
        'sinh': ['f32'],
        'slice_scatter': ['f16', 'f32'],
        'softmax': ['f32'],
        'split': ['f16', 'f32'],
        'sqrt': ['f32'],
        'square': ['f16', 'f32'],
        'squeeze': ['f16', 'f32'],
        'stack': ['f16', 'f32'],
        'sub': ['f32'],
        'sum_to_size': ['f16', 'f32'],
        'svd': ['f32'],
        't': ['f16', 'f32'],
        'tanh': ['f32'],
        'tensordot': ['f32'],
        'tile': ['f16', 'f32'],
        'tril': ['f16', 'f32'],
        'triu': ['f16', 'f32'],
        'true_divide': ['f16', 'f32'],
        'trunc': ['f32'],
        'unbind': ['f16', 'f32'],
        'unflatten': ['f16', 'f32'],
        'unsqueeze': ['f16', 'f32'],
        'view': ['f16', 'f32'],
        'view_as': ['f16', 'f32'],
        'vsplit': ['f16', 'f32'],
        'vstack': ['f16', 'f32'],
        'zero_': ['f16', 'f32']}

    # These ops that are problematic. So never run them even when
    # generating the new allowlist.
    # If the dtype list is None, all dtypes are excluded.
    # All the entries in this list should be removed
    BLOCKLIST = {
        # Functions that hang
        'masked_fill': [torch.bool, torch.uint8, torch.float32], 'where': [torch.bool],
        # + forward when requires_grad=True or running backward
        'masked.mean': [torch.bool, torch.float16],
        'masked.prod': [torch.bool],
        'masked.sum': [torch.bool],

        # Functions that hard crash
        'nn.functional.kl_div': [torch.int16, torch.int32, torch.int64],
        'nn.functional.nll_loss': [torch.float32],
        'nn.functional.padreflect': [torch.float32], 'nn.functional.padreplicate': [torch.float32],
        'std': [torch.float16],
        'stft': [torch.float32], 'var': [torch.float16],
        # + forward when requires_grad=True or running backward
        'index_select': [torch.float16],
        'nn.functional.embedding': [torch.float32, torch.float16],
        '__rpow__': [torch.int64],
        'masked.std': [torch.int32],
        'masked.var': [torch.int32],
        'as_strided_scatter': [torch.uint8],
        'atan2': [torch.int64],
        'bfloat16': None,
        'block_diag': [torch.uint8],
        'byte': None,
        'chalf': None,
        'diag_embed': [torch.uint8],
        'diagonal_scatter': [torch.uint8],
        'index_add': None,
        'log1p': None,
        'long': None,
        'nn.functional.avg_pool1d': [torch.int64],
        'nn.functional.avg_pool2d': [torch.int64],
        'nn.functional.conv1d': [torch.int64],
        'nn.functional.conv2d': [torch.int64],
        'nn.functional.conv_transpose1d': [torch.int64],
        'nn.functional.conv_transpose2d': [torch.int64],
        'nn.functional.conv_transpose3d': [torch.int64, torch.float32],
        'nn.functional.huber_loss': [torch.float16],
        'nn.functional.local_response_norm': [torch.int64],
        'nn.functional.padcircular': [torch.uint8],
        'nn.functional.softplus': [torch.float32],
        'pow': [torch.int64],
        'select_scatter': [torch.uint8],
        'sigmoid': [torch.int64],
        'slice_scatter': [torch.uint8],
        'square': [torch.bool, torch.int16, torch.int32, torch.int64, torch.uint8],  # moved from section below


        # ALLOW_LIST doesn't know about variants
        'nn.functional.padconstant': None,

        # These were moved from ALLOWLIST to BLOCK as they are not working
        # locally
        'tile': ['torch.float16', 'torch.float32', 'torch.int16', 'torch.int32', 'torch.int64', 'torch.uint8'],
        '__radd__': ['torch.bool', 'torch.uint8'],
        '__rmul__': ['torch.uint8'],
        'add': ['torch.bool', 'torch.uint8'],
        'addr': ['torch.int16', 'torch.int32', 'torch.int64', 'torch.uint8'],
        'diag': ['torch.int64'],
        'diagflat': ['torch.int64'],

        # Functions that are flaky
        # These are detected as "ok" by the expect case but actually fail to run sometimes
        'H': None,
        'T': None,
        'as_strided': None,
        'broadcast_tensors': None,
        'broadcast': None,
        'broadcast_to': None,
        'diagonal': None,
        'divfloor_rounding': None,
        'divno_rounding_mode': None,
        'divtrunc_rounding': None,
        'dsplit': None,
        'hsplit': None,
        'empty': None,
        'expand_as': None,
        'expand': None,
        'ge': None,
        'ne': None,
        'le': None,
        'lt': None,
        'gt': None,
        'transpose': None,
        'splitlist_args': None,
        'select': None,
        'reshape': None,
        'reshape_as': None,
        'permute': None,
        'norm': None,
        'nn.functional.pixel_unshuffle': None,
        'nn.functional.pixel_shuffle': None,
        'nn.functional.cross_entropy': None,
        'nn.functional.one_hot': None,
        'narrow': None,
        'movedim': None,
        'minreduction_with_dim': None,
        'minreduction_no_dim': None,
        'minbinary': None,
        'meshgridvariadic_tensors': None,
        'meshgridlist_of_tensors': None,
        'maxreduction_with_dim': None,
        'maxreduction_no_dim': None,
        'maxbinary': None,
        'maximum': None,
        'minimum': None,
        'mT': None,
        'mH': None,
        'outer': None,
        'softmaxwith_dtype': None,
        'rounddecimals_neg_3': None,
        'rounddecimals_3': None,
        'rounddecimals_0': None,
        'normnuc': None,
        'nn.functional.softminwith_dtype': None,
        'nn.functional.feature_alpha_dropoutwith_train': None,
        'log_softmaxwith_dtype': None,
        'split_with_sizes': None,
        'trapezoid': None,
        'eq': None,
        'mul': None,
        'cartesian_prod': None,
        'nonzero': None,
        'bool': None,
        'inner': None,
        'dstack': None,
        'take_along_dim': None,
    }

    # Used for accept mode only
    NEW_ALLOW_LIST = defaultdict(list)
    NEW_ALLOW_LIST_GRAD = defaultdict(list)

    @ops(op_db, allowed_dtypes=MPS_DTYPES)
    def test_output_match(self, device, dtype, op):
        self.assertEqual(device, "cpu")
        if not torch.backends.mps.is_available():
            self.skipTest("MPS is not available")

        key = op.name + op.variant_test_name
        if key in self.BLOCKLIST:
            if self.BLOCKLIST[key] is None or dtype in self.BLOCKLIST[key]:
                self.skipTest(f"Running test with {op.name} hangs so skipping")

        # Make this an expecttest manually
        # When this env variable is set, generate a new ALLOWLIST_OP
        # that reflects the current state of what passes or not
        if os.environ.get("EXPECTTEST_ACCEPT", None) == "1":
            generate_new_truth = True
        else:
            generate_new_truth = False

        run_grad_test = True
        if not generate_new_truth:
            if op.name not in self.ALLOWLIST_OP:
                self.skipTest(f"{op.name} is not in the allow list for test on MPS")
            else:
                if dtype_abbrs[dtype] not in self.ALLOWLIST_OP[op.name]:
                    self.skipTest(f"{op.name} is in the allow list for MPS but {dtype} is excluded")

            if op.name not in self.ALLOWLIST_OP_GRAD or dtype_abbrs[dtype] not in self.ALLOWLIST_OP_GRAD[op.name]:
                run_grad_test = False

        def get_samples():
            return op.sample_inputs(device, dtype, requires_grad=(dtype.is_floating_point or dtype.is_complex))
        cpu_samples = get_samples()

        all_forward_pass = True
        all_backward_pass = True
        for cpu_sample in cpu_samples:
            #
            # Forward check
            #
            forward_failed = False
            try:
                mps_sample = cpu_sample.transform(
                    lambda x: x.detach().to("mps").requires_grad_(x.requires_grad) if isinstance(x, torch.Tensor) else x)

                # TODO: This checks only the function variant. We should also check the method and inplace version
                # when they exist
                cpu_args = [cpu_sample.input] + list(cpu_sample.args)
                cpu_kwargs = cpu_sample.kwargs
                mps_args = [mps_sample.input] + list(mps_sample.args)
                mps_kwargs = mps_sample.kwargs

                cpu_out = op(*cpu_args, **cpu_kwargs)
                mps_out = op(*mps_args, **mps_kwargs)

                if op.name == "nn.functional.conv2d" and dtype == torch.float32:
                    atol = 1e-4
                    rtol = 3e-5
                elif op.name == "add" and dtype == torch.float16:
                    atol = 1e-2
                    rtol = 1e-2
                else:
                    atol = None
                    rtol = None

                self.assertEqual(cpu_out, mps_out, atol=atol, rtol=rtol)

            except Exception as e:
                if not generate_new_truth:
                    raise e
                forward_failed = True
                all_forward_pass = False

            if not (dtype.is_floating_point or dtype.is_complex):
                # Maybe we should error here instead?
                continue

            #
            # Backward check
            #

            # Skip the grad test if it is not part of the allow list
            if not generate_new_truth and not run_grad_test:
                # TODO: maybe there is a way to print only when we have -v
                # if i == 0:
                #     print(f"Skipping gradient check because {op.name} is not on the allow list")
                continue

            try:
                if forward_failed:
                    # We would've failed immediately anyway, but this error is clearer
                    # We error instead of continuing so that all_backward_pass would not be True
                    raise RuntimeError("Forward pass already failed")

                cpu_out = (cpu_out,) if isinstance(cpu_out, torch.Tensor) else tuple(cpu_out)
                mps_out = (mps_out,) if isinstance(mps_out, torch.Tensor) else tuple(mps_out)

                def req_grad(t):
                    return isinstance(t, torch.Tensor) and t.requires_grad

                diff_cpu_out = tuple(t for t in cpu_out if req_grad(t))
                diff_mps_out = tuple(t for t in mps_out if req_grad(t))
                diff_cpu_arg = tuple(t for t in pytree.tree_flatten((cpu_args, cpu_kwargs))[0] if req_grad(t))
                diff_mps_arg = tuple(t for t in pytree.tree_flatten((mps_args, mps_kwargs))[0] if req_grad(t))
                self.assertEqual(len(diff_cpu_out), len(diff_mps_out))
                self.assertEqual(len(diff_cpu_arg), len(diff_mps_arg))

                if len(diff_cpu_out) == 0:
                    continue
                # rand_like does not work with certain dtypes, so cast to double and cast back
                cpu_grad_outputs = tuple(torch.rand_like(t.to(dtype=torch.double)).to(dtype=dtype) for t in diff_cpu_out)
                mps_grad_outputs = tuple(t.to("mps") for t in cpu_grad_outputs)

                # Compare computed gradients with cpu given random grad_output vector
                # Sometimes when the derivative is 0, we just don't bother creating the graph
                # allow_unused is needed in those cases.
                cpu_grad_inputs = torch.autograd.grad(diff_cpu_out, diff_cpu_arg, grad_outputs=cpu_grad_outputs, allow_unused=True)
                mps_grad_inputs = torch.autograd.grad(diff_mps_out, diff_mps_arg, grad_outputs=mps_grad_outputs, allow_unused=True)

                self.assertEqual(cpu_grad_inputs, mps_grad_inputs)
            except Exception as e:
                if not generate_new_truth:
                    raise e
                all_backward_pass = False

        if all_forward_pass and generate_new_truth:
            if dtype_abbrs[dtype] not in self.NEW_ALLOW_LIST[op.name]:
                self.NEW_ALLOW_LIST[op.name].append(dtype_abbrs[dtype])
            # We could write it only once. But I don't know how to detect that the current test is the last one
            # So each test append to the dict and write it.
            with open("new_mps_allowlist.txt", "w") as f:
                pprint.pprint(self.NEW_ALLOW_LIST, stream=f)

        if all_backward_pass and generate_new_truth and dtype.is_floating_point:
            if dtype_abbrs[dtype] not in self.NEW_ALLOW_LIST_GRAD[op.name]:
                self.NEW_ALLOW_LIST_GRAD[op.name].append(dtype_abbrs[dtype])
            # We could write it only once. But I don't know how to detect that the current test is the last one
            # So each test append to the dict and write it.
            with open("new_mps_allowlist_grad.txt", "w") as f:
                pprint.pprint(self.NEW_ALLOW_LIST_GRAD, stream=f)


# Copied from `TestCommon` in `test_ops.py`, just enough to duplicate the `test_numpy_ref` for MPS
@skipIfSlowGradcheckEnv
class TestCommon(TestCase):
    exact_dtype = True

    # Verifies, on teardown, that no OpInfo is still using dynamic dtypes in CI
    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

        if IS_CI:
            err_msg = (
                "The operator(s) below is(are) using dynamic_dtypes in the OpInfo entries."
                "This is OK for testing, but be sure to set the dtypes manually before landing your PR!"
            )
            # Assure no opinfo entry has dynamic_dtypes
            filtered_ops = list(filter(opinfo.utils.is_dynamic_dtype_set, op_db))
            for op in filtered_ops:
                fmt_str = opinfo.utils.str_format_dynamic_dtype(op)
                err_msg += "\n" + fmt_str

            assert len(filtered_ops) == 0, err_msg

    # This is the MPS equivalent of `test_numpy_ref` from `test_ops.py`. It lives over here while
    # MPS still requires some fairly heavy special casing in the test framework.
    # When MPS becomes more consistent, this can probably be merged with that test using
    # `@dtypesIfMPS(torch.float32)`, but for now, the assertions themselves need to be loosened
    @unittest.skipIf(TEST_WITH_ASAN, "Skipped under ASAN")
    @onlyMPS
    @suppress_warnings
    # MPS only supports float32
    @ops(_ref_test_ops, allowed_dtypes=(torch.float32,))
    def test_numpy_ref_mps(self, device, dtype, op):
        # Unlike `test_numpy_ref`, this test compares in `float32` since at the time of this test's creation MPS
        # does not support float64 Tensors.
        # A few ops are currently broken on their reference inputs, but not their sample inputs. These should
        # get patched up and this workaround removed.
        broken_on_ref_inputs = op.name in ['cat', 'clamp', 'where']
        inputs = op.reference_inputs(device, dtype) if not broken_on_ref_inputs else op.sample_inputs(device, dtype)
        for sample_input in inputs:
            self.compare_with_reference(op, op.ref, sample_input)

# TODO: Actually instantiate that test for the "mps" device to better reflect what it is doing.
# This requires mps to be properly registered in the device generic test framework which is not the
# case right now. We can probably use `allow_mps` introduced in https://github.com/pytorch/pytorch/pull/87342
# to achieve this.
instantiate_device_type_tests(TestConsistency, globals(), only_for="cpu")
instantiate_device_type_tests(TestCommon, globals(), allow_mps=True)

if __name__ == "__main__":
    run_tests()
