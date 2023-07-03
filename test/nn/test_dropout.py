# Owner(s): ["module: nn"]
from itertools import product
import unittest
import random
import itertools


import torch
from torch.testing._internal.common_utils import run_tests, set_default_dtype, \
    instantiate_parametrized_tests
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_nn import NNTestCase, freeze_rng_state
from torch.testing._internal.common_device_type import instantiate_device_type_tests, expectedFailureXLA
import torch.nn.functional as F
import torch.nn as nn

class TestDropoutNN(NNTestCase):
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True

    def _test_alpha_dropout(self, cls, input):
        mean = input.mean()
        std = input.std()

        for p in [0.2, 0.5, 0.8]:
            module = cls(p)
            input_var = input.detach().clone().requires_grad_()
            output = module(input_var)
            # output mean should be close to input mean
            self.assertLess(abs(output.data.mean() - mean), 0.1)
            # output std should be close to input std
            self.assertLess(abs(output.data.std() - std), 0.1)
            output.backward(input)

    def test_AlphaDropout(self):
        # generate random tensor with zero mean and unit std
        input = torch.randn(5000)
        self._test_alpha_dropout(nn.AlphaDropout, input)

    def test_FeatureAlphaDropout(self):
        b = random.randint(1, 5)
        w = random.randint(1, 5)
        h = random.randint(1, 5)
        d = random.randint(1, 2)
        num_features = 1000
        input = torch.randn(num_features, b, d, w, h)
        self._test_alpha_dropout(nn.FeatureAlphaDropout, input)

        # no batch dims
        input = torch.randn(50, 20, 64, 64)
        self._test_alpha_dropout(nn.FeatureAlphaDropout, input)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_native_dropout_corner_case(self):
        for train in [True, False]:
            for p in [0.0, 1.0]:
                for device in ["cuda", "cpu"]:
                    x = torch.randn(5).to(device=device).requires_grad_()
                    x_ref = x.detach().requires_grad_()
                    o = torch.native_dropout(x, p, train)[0]
                    o_ref = torch.dropout(x_ref, p, train)
                    o.sum().backward()
                    o_ref.sum().backward()
                    assert(o.equal(o_ref))
                    assert(x.grad.equal(x_ref.grad))

    def test_invalid_dropout_p(self):
        v = torch.ones(1)
        self.assertRaises(ValueError, lambda: nn.Dropout(-0.1))
        self.assertRaises(ValueError, lambda: nn.Dropout(1.1))
        self.assertRaises(ValueError, lambda: nn.Dropout1d(-0.1))
        self.assertRaises(ValueError, lambda: nn.Dropout1d(1.1))
        self.assertRaises(ValueError, lambda: nn.Dropout2d(-0.1))
        self.assertRaises(ValueError, lambda: nn.Dropout2d(1.1))
        self.assertRaises(ValueError, lambda: nn.Dropout3d(-0.1))
        self.assertRaises(ValueError, lambda: nn.Dropout3d(1.1))
        self.assertRaises(ValueError, lambda: F.dropout(v, -0.1))
        self.assertRaises(ValueError, lambda: F.dropout(v, 1.1))

class TestDropoutNNDeviceType(NNTestCase):
    def _test_dropout(self, cls, device, input, memory_format=torch.contiguous_format):
        p = 0.2
        input = input.to(device).fill_(1 - p)

        module = cls(p)
        input_var = input.clone(memory_format=memory_format).requires_grad_()
        output = module(input_var)
        self.assertTrue(output.is_contiguous(memory_format=memory_format))
        self.assertLess(abs(output.data.mean() - (1 - p)), 0.05)
        output.backward(input)
        self.assertTrue(input_var.grad.is_contiguous(memory_format=memory_format))
        self.assertLess(abs(input_var.grad.data.mean() - (1 - p)), 0.05)

        module = cls(p, True)
        input_var = input.clone(memory_format=memory_format).requires_grad_()
        output = module(input_var + 0)
        self.assertTrue(output.is_contiguous(memory_format=memory_format))
        self.assertLess(abs(output.data.mean() - (1 - p)), 0.05)
        output.backward(input)
        self.assertTrue(input_var.grad.is_contiguous(memory_format=memory_format))
        self.assertLess(abs(input_var.grad.data.mean() - (1 - p)), 0.05)

        # check eval mode doesn't change anything
        for inplace in [True, False]:
            module = cls(p, inplace).eval()
            self.assertEqual(input, module(input))

        # Check that these don't raise errors
        module.__repr__()
        str(module)

    def _test_dropout_discontiguous(self, cls, device, memory_format=torch.contiguous_format):
        # In this test, we verify that dropout preserves the layout and data for different memory formats.
        # We check whether, we get same values for the output of dropout, when the probability
        # of dropout is 0 or very close to 0.
        # Reference: https://github.com/pytorch/pytorch/issues/47176
        close_to_zero_p = 1e-10  # Should be almost zero but not zero, as for p=0 different path is taken
        for p in [0, close_to_zero_p]:
            inp = torch.ones(2, 3, 3, 3, device=device)
            inp_discontiguous = torch.empty(2, 3, 3, 6, device=device, memory_format=memory_format)[..., ::2]
            inp_discontiguous.copy_(inp)
            mod = cls(p=p)
            out = mod(inp_discontiguous)
            if p != 0:  # Zero will keep strides as is based on input.
                # When prob == 0, input stride (54, 18, 6, 2) -> output stride (54, 18, 6, 2)
                # When prob != 0, input stride (54, 18, 6, 2) -> output stride (27, 9, 3, 1)
                self.assertTrue(out.is_contiguous(memory_format=memory_format))
            self.assertEqual(inp_discontiguous, out)

    def _test_dropout_stride_mean_preserve(self, cls, device):
        def invert_perm(p):
            d = {x: i for i, x in enumerate(p)}
            return (d[0], d[1], d[2], d[3])

        inp = torch.ones(2, 3, 4, 5, device=device)
        shifts = [(0, 0), (1, 0), (0, 1), (1, 1)]
        for perm in itertools.permutations((0, 1, 2, 3), r=4):
            for shift in shifts:
                for p in [1e-10, 0.3, 0.5, 0.7]:
                    mod = cls(p=p)
                    permuted_inp = inp.permute(perm).contiguous().permute(invert_perm(perm))
                    permuted_inp = permuted_inp[shift[0]:, shift[1]:, :, :]
                    out = mod(permuted_inp)

                    self.assertTrue(out.permute(perm).is_contiguous())
                    self.assertEqual(inp.mean(), out.mean(), rtol=0.5, atol=0.5)
                    if p == 1e-10:
                        self.assertEqual(permuted_inp, out)
                    else:
                        self.assertNotEqual(permuted_inp, out)

    def test_Dropout(self, device):
        input = torch.empty(1000)
        self._test_dropout(nn.Dropout, device, input)

        self._test_dropout_discontiguous(nn.Dropout, device)
        self._test_dropout_discontiguous(nn.Dropout, device, memory_format=torch.channels_last)

        self._test_dropout_stride_mean_preserve(nn.Dropout, device)

        if self.device_type == 'cuda' or self.device_type == 'cpu':
            input = input.bfloat16()
            self._test_dropout(nn.Dropout, device, input)

    def _test_dropoutNd_no_batch(self, dropout, input):
        input_clone = input.clone()
        with freeze_rng_state():
            res_no_batch = dropout(input)

        with freeze_rng_state():
            res_batched = dropout(input_clone.unsqueeze(0)).squeeze(0)

        self.assertEqual(res_no_batch, res_batched)

    def _test_dropoutNd_channel_zero(self, dropout, input):
        # Verify the number of zeros in a channel is 0 or the number of elements in the channel
        # for a fully positive input tensor
        shape = input.shape
        B = shape[0]
        C = shape[1]
        channel_numel = torch.tensor(shape[2:]).prod()
        result = dropout(input)

        for b, c in product(range(B), range(C)):
            self.assertTrue(result[b, c].count_nonzero() in (0, channel_numel))

    @expectedFailureXLA  # seems like freeze_rng_state is not honoured by XLA
    def test_Dropout1d(self, device):
        with set_default_dtype(torch.double):
            N, C, L = random.randint(10, 15), random.randint(10, 15), random.randint(10, 15)
            input = torch.empty(N, C, L)
            self._test_dropout(nn.Dropout1d, device, input)

            with self.assertRaisesRegex(RuntimeError, "Expected 2D or 3D input, but received a 4D input"):
                nn.Dropout1d(p=0.5)(torch.rand(1, 2, 2, 2, device=device))

            with self.assertRaisesRegex(RuntimeError, "Expected 2D or 3D input, but received a 1D input"):
                nn.Dropout1d(p=0.5)(torch.rand(2, device=device))

            # no batch dims
            input = torch.rand(50, 2, device=device)
            self._test_dropoutNd_no_batch(nn.Dropout1d(p=0.5), input)
            self._test_dropoutNd_no_batch(nn.Dropout1d(p=0.5, inplace=True), input)

            # check that complete channels are dropped
            input = torch.ones(10, 4, 2, device=device)
            self._test_dropoutNd_channel_zero(nn.Dropout1d(p=0.5), input)
            self._test_dropoutNd_channel_zero(nn.Dropout1d(p=0.5, inplace=True), input)

    @expectedFailureXLA  # seems like freeze_rng_state is not honoured by XLA
    def test_Dropout2d(self, device):
        b = random.randint(1, 5)
        w = random.randint(1, 5)
        h = random.randint(1, 5)
        num_features = 1000
        input = torch.empty(num_features, b, w, h)
        self._test_dropout(nn.Dropout2d, device, input)
        self._test_dropout(nn.Dropout2d, device, input, memory_format=torch.channels_last)

        self._test_dropout_discontiguous(nn.Dropout2d, device)
        self._test_dropout_discontiguous(nn.Dropout2d, device, memory_format=torch.channels_last)

        with self.assertWarnsRegex(UserWarning, "Received a 5-D input to dropout2d"):
            nn.Dropout2d(p=0.5)(torch.rand(1, 2, 2, 2, 2, device=device))

        with self.assertWarnsRegex(UserWarning, "Received a 2-D input to dropout2d"):
            nn.Dropout2d(p=0.5)(torch.rand(1, 2, device=device))

        # TODO: Uncomment these lines once no-batch-dim inputs are supported.
        # For now, the historical dropout1d behavior is performed for 3D inputs.
        # See https://github.com/pytorch/pytorch/issues/77081

        # input = torch.rand(50, 2, 2, device=device)
        # self._test_dropoutNd_no_batch(nn.Dropout2d(p=0.5), input)
        # self._test_dropoutNd_no_batch(nn.Dropout2d(p=0.5, inplace=True), input)

        with self.assertWarnsRegex(UserWarning, "assuming that channel-wise 1D dropout behavior is desired"):
            nn.Dropout2d(p=0.5)(torch.rand(1, 2, 2, device=device))

        # check that complete channels are dropped
        input = torch.ones(10, 4, 2, 2, device=device)
        self._test_dropoutNd_channel_zero(nn.Dropout2d(p=0.5), input)
        self._test_dropoutNd_channel_zero(nn.Dropout2d(p=0.5, inplace=True), input)

    @expectedFailureXLA  # seems like freeze_rng_state is not honoured by XLA
    def test_Dropout3d(self, device):
        b = random.randint(1, 5)
        w = random.randint(1, 5)
        h = random.randint(1, 5)
        d = random.randint(1, 2)
        num_features = 1000
        input = torch.empty(num_features, b, d, w, h)
        self._test_dropout(nn.Dropout3d, device, input)

        self._test_dropout_discontiguous(nn.Dropout3d, device)
        self._test_dropout_discontiguous(nn.Dropout3d, device, memory_format=torch.channels_last)

        with self.assertWarnsRegex(UserWarning, "Received a 6-D input to dropout3d"):
            nn.Dropout3d(p=0.5)(torch.rand(1, 2, 2, 2, 2, 2, device=device))

        with self.assertWarnsRegex(UserWarning, "Received a 3-D input to dropout3d"):
            nn.Dropout3d(p=0.5)(torch.rand(1, 2, 2, device=device))

        # no batch dims
        input = torch.rand(50, 2, 2, 2, device=device)
        self._test_dropoutNd_no_batch(nn.Dropout3d(p=0.5), input)
        self._test_dropoutNd_no_batch(nn.Dropout3d(p=0.5, inplace=True), input)

        # check that complete channels are dropped
        input = torch.ones(10, 4, 2, 2, 2, device=device)
        self._test_dropoutNd_channel_zero(nn.Dropout3d(p=0.5), input)
        self._test_dropoutNd_channel_zero(nn.Dropout3d(p=0.5, inplace=True), input)

    def test_empty_dropout(self, device):
        x = torch.tensor([]).to(device)
        out = torch.nn.functional.dropout(x)
        self.assertEqual(out.size(), x.size())

instantiate_device_type_tests(TestDropoutNNDeviceType, globals())
instantiate_parametrized_tests(TestDropoutNN)

if __name__ == '__main__':
    run_tests()
