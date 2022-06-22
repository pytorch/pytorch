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
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from torch._six import inf
from torch.nn import Parameter
from torch.testing._internal.common_utils import run_tests, TestCase, download_file, TEST_WITH_UBSAN
from torch.testing._comparison import TensorLikePair
import torch.backends.mps
from torch.distributions import Uniform, Exponential

from torch.testing._internal.common_nn import NNTestCase
import numpy as np
import torch

# Same logic as test_cuda.py
if not torch.backends.mps.is_available():
    print('MPS not available, skipping tests', file=sys.stderr)
    TestCase = object  # noqa: F811
    NNTestCase = object  # noqa: F811

class MPSReluTest(TestCase):
    def _npRelu(self, np_features):
        return np.maximum(np_features, np.zeros(np_features.shape)).astype(np_features.dtype)

    def testNpRelu(self):
        torch.testing.assert_allclose(
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

        torch.testing.assert_allclose(np_relu, py_relu_cpu)

    def _testReluInPlace(self, np_features, device):
        np_relu = self._npRelu(np_features)
        # Convert the numpy array to a PyTorch Tensor,
        # and move the Tensor to the CPU/GPU based on the "device" parameter
        py_tensor = torch.from_numpy(np_features).to(device)
        py_relu = torch.nn.ReLU(inplace=True)(py_tensor)
        py_relu_cpu = py_relu.to("cpu")

        torch.testing.assert_allclose(np_relu, py_relu_cpu)
        # Inplace Relu modifies the initial input and it should match the output of Relu
        torch.testing.assert_allclose(np_relu, py_tensor.to("cpu"))

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
        torch.testing.assert_allclose(
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
        torch.testing.assert_allclose(cpu_leaky_relu, mps_leaky_relu.to('cpu'))

        # test backward pass
        cpu_grad = torch.ones_like(cpu_leaky_relu)
        mps_grad = cpu_grad.to('mps')
        cpu_leaky_relu.backward(gradient=cpu_grad)
        mps_leaky_relu.backward(gradient=mps_grad)
        torch.testing.assert_allclose(cpu_x.grad, mps_x.grad.to('cpu'))

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
        torch.testing.assert_allclose(cpu_leaky_relu, mps_leaky_relu.to('cpu'))

        # test backward pass
        cpu_grad = torch.ones_like(cpu_leaky_relu)
        mps_grad = cpu_grad.to('mps')
        cpu_leaky_relu.backward(gradient=cpu_grad)
        mps_leaky_relu.backward(gradient=mps_grad)
        torch.testing.assert_allclose(cpu_x.grad, mps_x.grad.to('cpu'))

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
        torch.testing.assert_allclose(D, torch.full((5, 5), 6.0))

    def test_addmm(self):
        A = torch.ones(5, 5).to("mps")
        B = torch.ones(5, 6).to("mps")
        C = torch.ones(6, 5).to("mps")
        D = torch.addmm(A, B, C).to("cpu")
        torch.testing.assert_allclose(D, torch.full((5, 5), 7.0))

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
        torch.testing.assert_allclose(x_cpu.item(), y_mps.item())

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

    def test_as_strided(self):
        def helper(n, c):
            values = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
            values_1 = [[1.0, 1.0], [1.0, 1.0]]
            cpu_x = torch.tensor(values, device='cpu')
            ones1 = torch.tensor(values_1, device='mps')
            x = cpu_x.detach().clone().to('mps').requires_grad_()
            strided_cpu = torch.as_strided(cpu_x, (2, 2), (1, 2))
            strided_mps = torch.as_strided(x, (2, 2), (1, 2))

            self.assertEqual(strided_mps, strided_cpu)

        helper(3, 3)

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

    # Test forward argmax
    def test_argmax(self):
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

            y = torch.argmax(x)
            ref_y = torch.argmax(cpu_x)
            self.assertEqual(y, ref_y)

            y_0 = torch.argmax(x, dim=0)
            refy_0 = torch.argmax(cpu_x, dim=0)
            self.assertEqual(y_0, refy_0)

            y_0dim = torch.argmax(x, dim=0, keepdim=True)
            refy_0dim = torch.argmax(cpu_x, dim=0, keepdim=True)
            self.assertEqual(y_0dim, refy_0dim)

            y_1 = torch.argmax(x, dim=1)
            refy_1 = torch.argmax(cpu_x, dim=1)
            self.assertEqual(y_1, refy_1)

            y_1dim = torch.argmax(x, dim=1, keepdim=True)
            refy_1dim = torch.argmax(cpu_x, dim=1, keepdim=True)
            self.assertEqual(y_1dim, refy_1dim)

            y_2 = torch.argmax(x, dim=2)
            refy_2 = torch.argmax(cpu_x, dim=2)
            self.assertEqual(y_2, refy_2)

            y_2dim = torch.argmax(x, dim=2, keepdim=True)
            refy_2dim = torch.argmax(cpu_x, dim=2, keepdim=True)
            self.assertEqual(y_2dim, refy_2dim)

            y_3 = torch.argmax(x, dim=3)
            refy_3 = torch.argmax(cpu_x, dim=3)
            self.assertEqual(y_3, refy_3)

            y_3dim = torch.argmax(x, dim=3, keepdim=True)
            refy_3dim = torch.argmax(cpu_x, dim=3, keepdim=True)
            self.assertEqual(y_3dim, refy_3dim)

        helper(2, 8, 4, 4, torch.float32)
        helper(2, 8, 4, 4, torch.int32)
        helper(2, 8, 4, 4, torch.float16)
        helper(2, 8, 4, 4, torch.int64)

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

    # Test var
    def test_var(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            x = cpu_x.detach().clone().to('mps')

            all_var = torch.var(x, unbiased=False)
            all_var_cpu = torch.var(cpu_x, unbiased=False)

            self.assertEqual(all_var, all_var_cpu)

            nil_dim_var = torch.var(x, dim=[], unbiased=False)
            nil_dim_var_cpu = torch.var(cpu_x, dim=[], unbiased=False)

            self.assertEqual(nil_dim_var, nil_dim_var_cpu)

            nil_dim_var_keepdim = torch.var(x, dim=[], keepdim=True, unbiased=False)
            nil_dim_var_cpu_keepdim = torch.var(cpu_x, dim=[], keepdim=True, unbiased=False)

            self.assertEqual(nil_dim_var_keepdim, nil_dim_var_cpu_keepdim)

            zero_dim_var = torch.var(x, dim=[0], unbiased=False)
            zero_dim_var_cpu = torch.var(cpu_x, dim=[0], unbiased=False)

            self.assertEqual(zero_dim_var, zero_dim_var_cpu)

            zero_dim_var_keepdim = torch.var(x, dim=[0], keepdim=True, unbiased=False)
            zero_dim_var_cpu_keepdim = torch.var(cpu_x, dim=[0], keepdim=True, unbiased=False)

            self.assertEqual(zero_dim_var_keepdim, zero_dim_var_cpu_keepdim)

            zero_one_dim_var = torch.var(x, dim=[0, 1], unbiased=False)
            zero_one_dim_var_cpu = torch.var(cpu_x, dim=[0, 1], unbiased=False)

            self.assertEqual(zero_one_dim_var, zero_one_dim_var_cpu)

            zero_one_dim_var_keepdim = torch.var(x, dim=[0, 1], keepdim=True, unbiased=False)
            zero_one_dim_var_cpu_keepdim = torch.var(cpu_x, dim=[0, 1], keepdim=True, unbiased=False)

            self.assertEqual(zero_one_dim_var_keepdim, zero_one_dim_var_cpu_keepdim)

            two_three_dim_var = torch.var(x, dim=[2, 3], unbiased=False)
            two_three_dim_var_cpu = torch.var(cpu_x, dim=[2, 3], unbiased=False)

            self.assertEqual(two_three_dim_var, two_three_dim_var_cpu)

            two_three_keepdim_var = torch.var(x, dim=[2, 3], keepdim=True, unbiased=False)
            two_three_dim_keepvar_cpu = torch.var(cpu_x, dim=[2, 3], keepdim=True, unbiased=False)

            self.assertEqual(two_three_keepdim_var, two_three_dim_keepvar_cpu)

            all_var = torch.var(x, unbiased=True)
            all_var_cpu = torch.var(cpu_x, unbiased=True)

            self.assertEqual(all_var, all_var_cpu)

            nil_dim_var = torch.var(x, dim=[], unbiased=True)
            nil_dim_var_cpu = torch.var(cpu_x, dim=[], unbiased=True)

            self.assertEqual(nil_dim_var, nil_dim_var_cpu)

            nil_dim_var_keepdim = torch.var(x, dim=[], keepdim=True, unbiased=True)
            nil_dim_var_cpu_keepdim = torch.var(cpu_x, dim=[], keepdim=True, unbiased=True)

            self.assertEqual(nil_dim_var_keepdim, nil_dim_var_cpu_keepdim)

            zero_dim_var = torch.var(x, dim=[0], unbiased=True)
            zero_dim_var_cpu = torch.var(cpu_x, dim=[0], unbiased=True)

            self.assertEqual(zero_dim_var, zero_dim_var_cpu)

            zero_dim_var_keepdim = torch.var(x, dim=[0], keepdim=True, unbiased=True)
            zero_dim_var_cpu_keepdim = torch.var(cpu_x, dim=[0], keepdim=True, unbiased=True)

            self.assertEqual(zero_dim_var_keepdim, zero_dim_var_cpu_keepdim)

            zero_one_dim_var = torch.var(x, dim=[0, 1], unbiased=True)
            zero_one_dim_var_cpu = torch.var(cpu_x, dim=[0, 1], unbiased=True)

            self.assertEqual(zero_one_dim_var, zero_one_dim_var_cpu)

            zero_one_dim_var_keepdim = torch.var(x, dim=[0, 1], keepdim=True, unbiased=True)
            zero_one_dim_var_cpu_keepdim = torch.var(cpu_x, dim=[0, 1], keepdim=True, unbiased=True)

            self.assertEqual(zero_one_dim_var_keepdim, zero_one_dim_var_cpu_keepdim)

            two_three_dim_var = torch.var(x, dim=[2, 3], unbiased=True)
            two_three_dim_var_cpu = torch.var(cpu_x, dim=[2, 3], unbiased=True)

            self.assertEqual(two_three_dim_var, two_three_dim_var_cpu)

            two_three_keepdim_var = torch.var(x, dim=[2, 3], keepdim=True, unbiased=True)
            two_three_dim_keepvar_cpu = torch.var(cpu_x, dim=[2, 3], keepdim=True, unbiased=True)

            self.assertEqual(two_three_keepdim_var, two_three_dim_keepvar_cpu)

        helper((4, 5, 6, 7))

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
            for dtype in [torch.float32]:
                cpu_x = torch.randn(shape, device='cpu', dtype=dtype, requires_grad=False)
                mps_x = cpu_x.detach().clone().to('mps')
                # clamp to avoid division by 0
                cpu_y = torch.randn(shape, device='cpu', dtype=dtype, requires_grad=False)
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
        # Empty test - Currently failing! Empty tensor not handled!
        # helper([0, 2, 4, 5], [2, 0, 4, 5], [2, 5, 0, 5])

    def test_pad(self):
        def helper(shape, padding, op):
            inputCPU = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            inputCPU.retain_grad()
            inputMPS = inputCPU.detach().clone().to('mps').requires_grad_()

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

        # 2D Padding
        helper((1, 2, 3, 4), (1, 1, 2, 0), nn.ReflectionPad2d)
        # verify if a change in shape of input would cause problems with graph caching
        helper((2, 4, 3, 4), (1, 1, 2, 0), nn.ReflectionPad2d)
        # this should make the padding (2, 2, 2, 2)
        helper((2, 1, 6, 8), 2, nn.ReplicationPad2d)
        # verify if a change in shape of padding would cause problems with graph caching
        helper((2, 1, 6, 8), (2, 4, 3, 5), nn.ReplicationPad2d)

        # 3D Padding
        helper((2, 4, 6, 8, 4), (1, 3, 3, 5, 3, 4), nn.ReflectionPad3d)
        # verify if a change in shape of padding would cause problems with graph caching
        helper((2, 4, 6, 8, 4), (1, 3, 3, 5, 3, 4), nn.ReplicationPad3d)

    # Test stack forward
    def test_stack(self):
        # All shapes must be same
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            x = cpu_x.detach().clone().to('mps')

            cpu_y = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            y = cpu_y.detach().clone().to('mps')

            cpu_z = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            z = cpu_z.detach().clone().to('mps')

            stack = torch.stack([x, y, z], dim=1)
            stack_cpu = torch.stack([cpu_x, cpu_y, cpu_z], dim=1)

            self.assertEqual(stack, stack_cpu)

        helper([2, 8, 4, 5])
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

    # Test softplus

    def test_softplus(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            softplus_result = torch.nn.Softplus(beta=0.5, threshold=0.5)(x)
            softplus_result_cpu = torch.nn.Softplus(beta=0.5, threshold=0.5)(cpu_x)

            self.assertEqual(softplus_result, softplus_result_cpu)

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
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            gelu_result = torch.nn.GELU()(x)
            gelu_result_cpu = torch.nn.GELU()(cpu_x)

            cpu_grad = torch.ones_like(gelu_result_cpu)
            grad = cpu_grad.to('mps')

            gelu_result.backward(gradient=grad)
            gelu_result_cpu.backward(gradient=cpu_grad)

            self.assertEqual(gelu_result, gelu_result_cpu)
            self.assertEqual(x.grad, cpu_x.grad)

        # Test empty shape too
        for shape in [(0, 3), [], (2, 3), (2, 8, 4, 5)]:
            helper(shape)

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
        def helper(n, d, m):
            embeddingMPS = nn.Embedding(n, d, max_norm=True, device='mps')
            W_MPS = torch.randn((m, d), requires_grad=True, device='mps')
            idx_MPS = torch.tensor([0, 1, 2]).to('mps')
            a_MPS = embeddingMPS.weight.clone() @ W_MPS.t()  # weight must be cloned for this to be differentiable
            a_MPS.retain_grad()
            b_MPS = embeddingMPS(idx_MPS) @ W_MPS.t()  # modifies weight in-place
            b_MPS.retain_grad()
            out_MPS = (a_MPS.unsqueeze(0) + b_MPS.unsqueeze(1))
            loss_MPS = out_MPS.sigmoid().prod()
            loss_MPS.backward()

            embeddingCPU = nn.Embedding(n, d, max_norm=True, scale_grad_by_freq=True)
            W_CPU = W_MPS.to('cpu')
            idx_CPU = torch.tensor([0, 1, 2])
            a_CPU = embeddingCPU.weight.clone() @ W_CPU.t()  # weight must be cloned for this to be differentiable
            a_CPU.retain_grad()
            b_CPU = embeddingCPU(idx_CPU) @ W_CPU.t()  # modifies weight in-place
            b_CPU.retain_grad()
            out_CPU = (a_CPU.unsqueeze(0) + b_CPU.unsqueeze(1))
            loss_CPU = out_CPU.sigmoid().prod()
            loss_CPU.backward()

            self.assertEqual(b_CPU.grad, b_MPS.grad)
            self.assertEqual(a_CPU.grad, a_MPS.grad)

        helper(3, 5, 7)

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

    # Test normal
    def test_normal(self):
        def helper(shape, mean=0.0, std=1.0):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            x = cpu_x.detach().clone().to('mps')

            mps_out = torch.normal(mean, std, shape, device='mps')

            mean_array = np.ones(shape)
            mean_array *= mean
            cpu_mean_tensor = torch.tensor(mean_array, device='cpu', dtype=torch.float, requires_grad=False)
            mean_tensor = cpu_mean_tensor.detach().clone().to('mps')

            std_array = np.ones(shape)
            std_array *= std
            cpu_std_tensor = torch.tensor(std_array, device='cpu', dtype=torch.float, requires_grad=False)
            std_tensor = cpu_std_tensor.detach().clone().to('mps')

            mps_out = torch.zeros(shape, device='mps')
            torch.normal(mean_tensor, std, out=mps_out)

            mps_out = torch.zeros(shape, device='mps')
            torch.normal(mean, std_tensor, out=mps_out)

            mps_out = torch.zeros(shape, device='mps')
            torch.normal(mean_tensor, std_tensor, out=mps_out)

        helper((2, 3, 4, 5, 6))
        helper((100, 100), 2.5, 1.2)

    def test_bernoulli(self):
        def helper(shape, prob=0.5):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            x = cpu_x.detach().clone().to('mps')

            prob_array = np.ones(shape)
            prob_array *= prob
            cpu_prob_tensor = torch.tensor(prob_array, device='cpu', dtype=torch.float, requires_grad=False)
            prob_tensor = cpu_prob_tensor.detach().clone().to('mps')

            mps_out = torch.bernoulli(prob_tensor)
            # We can't check reliably the mean and std.
            # Just make sure we don't return constant values
            self.assertNotEqual(mps_out.to('cpu').mean(), 0.)
            self.assertNotEqual(mps_out.to('cpu').std() ** 2, 0.)

            mps_out = torch.zeros(shape, device='mps')
            mps_out = torch.bernoulli(mps_out, prob)

            self.assertNotEqual(mps_out.to('cpu').mean(), 0.)
            self.assertNotEqual(mps_out.to('cpu').std(), 0.)

        helper((100, 100), 0.50)
        helper((100, 100), 0.76)
        helper((100, 100), 0.23)

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


class TestRNNMPS(TestCase):
    def test_lstm_1(self, device="mps", dtype=torch.float32):

        rnn = nn.LSTM(1, 4, 2, device="cpu")
        input = torch.randn(2, 3, 1, device="cpu")
        hx = torch.zeros(2, 3, 4, device="cpu")
        cx = torch.zeros(2, 3, 4, device="cpu")

        cpu_output, _ = rnn(input, (hx, cx))

        device = torch.device("mps")
        rnn = rnn.to(device)
        input = input.to(device)
        hx = hx.to(device)
        cx = cx.to(device)
        output, _ = rnn(input, (hx, cx))
        self.assertEqual(cpu_output, output)

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

        with self.assertRaisesRegex(NotImplementedError, "not current implemented for the MPS device"):
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

        with self.assertRaisesRegex(AssertionError, "Tensor-likes are not close!"):
            torch.testing.assert_close(a, nan)

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




if __name__ == "__main__":
    run_tests()
