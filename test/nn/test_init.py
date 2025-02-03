# Owner(s): ["module: nn"]
import math
import random
import string
import unittest
from functools import reduce
from operator import mul

import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.testing._internal.common_utils import (
    run_tests,
    skipIfNoLapack,
    skipIfTorchDynamo,
    slowTest,
    TEST_SCIPY,
    TestCase,
)


if TEST_SCIPY:
    from scipy import stats


class TestNNInit(TestCase):
    def setUp(self):
        super().setUp()
        random.seed(123)

    def _is_normal(self, tensor, mean, std):
        samples = tensor.view(-1).tolist()
        p_value = stats.kstest(samples, "norm", args=(mean, std))[1]
        return p_value > 0.0001

    def _is_trunc_normal(self, tensor, mean, std, a, b):
        # scipy's trunc norm is suited for data drawn from N(0, 1),
        # so we need to transform our data to test it using scipy.
        z_samples = (tensor.view(-1) - mean) / std
        z_samples = z_samples.tolist()
        a0 = (a - mean) / std
        b0 = (b - mean) / std
        p_value = stats.kstest(z_samples, "truncnorm", args=(a0, b0))[1]
        return p_value > 0.0001

    def _is_uniform(self, tensor, a, b):
        samples = tensor.view(-1).tolist()
        p_value = stats.kstest(samples, "uniform", args=(a, (b - a)))[1]
        return p_value > 0.0001

    def _create_random_nd_tensor(self, dims, size_min, size_max):
        size = [random.randint(size_min, size_max) for _ in range(dims)]
        tensor = torch.zeros(size)
        return tensor

    def _random_float(self, a, b):
        return (b - a) * random.random() + a

    def test_calculate_gain_linear(self):
        for fn in [
            "linear",
            "conv1d",
            "conv2d",
            "conv3d",
            "conv_transpose2d",
            "conv_transpose2d",
            "conv_transpose3d",
        ]:
            gain = init.calculate_gain(fn)
            self.assertEqual(gain, 1)

    def test_calculate_gain_nonlinear(self):
        for fn in ["sigmoid", "tanh", "relu", "leaky_relu"]:
            gain = init.calculate_gain(fn)
            if fn == "sigmoid":
                self.assertEqual(gain, 1)
            elif fn == "tanh":  # 5 / 3
                self.assertEqual(gain, 1.6666666666666667)
            elif fn == "relu":  # sqrt(2)
                self.assertEqual(gain, 1.4142135623730951)
            elif fn == "leaky_relu":  # sqrt(2 / 1 + slope^2))
                self.assertEqual(gain, 1.4141428569978354)
            elif fn == "selu":
                self.assertEqual(gain, 0.75)

    def test_calculate_gain_leaky_relu(self):
        for param in [None, 0, 0.01, 10]:
            gain = init.calculate_gain("leaky_relu", param)
            if param is None:  # Default slope is 0.01
                self.assertEqual(gain, 1.4141428569978354)
            elif param == 0:  # No slope = same gain as normal ReLU
                self.assertEqual(gain, 1.4142135623730951)
            elif param == 0.01:
                self.assertEqual(gain, 1.4141428569978354)
            elif param == 10:
                self.assertEqual(gain, 0.14071950894605836)

    def test_calculate_gain_leaky_relu_only_accepts_numbers(self):
        for param in [True, [1], {"a": "b"}]:
            with self.assertRaises(ValueError):
                init.calculate_gain("leaky_relu", param)

    def test_calculate_gain_only_accepts_valid_nonlinearities(self):
        for n in [2, 5, 25]:
            # Generate random strings of lengths that definitely aren't supported
            random_string = "".join(
                [random.choice(string.ascii_lowercase) for i in range(n)]
            )
            with self.assertRaises(ValueError):
                init.calculate_gain(random_string)

    @unittest.skipIf(not TEST_SCIPY, "Scipy not found.")
    @skipIfTorchDynamo("scipy.kstest is failing under dynamo")
    def test_uniform(self):
        for dims in [1, 2, 4]:
            input_tensor = self._create_random_nd_tensor(dims, size_min=30, size_max=50)
            a = self._random_float(-3, 3)
            b = a + self._random_float(1, 5)
            init.uniform_(input_tensor, a=a, b=b)
            assert self._is_uniform(input_tensor, a, b)

    @unittest.skipIf(not TEST_SCIPY, "Scipy not found.")
    @skipIfTorchDynamo("scipy.kstest is failing under dynamo")
    def test_normal(self):
        for dims in [1, 2, 4]:
            input_tensor = self._create_random_nd_tensor(dims, size_min=30, size_max=50)
            mean = self._random_float(-3, 3)
            std = self._random_float(1, 5)
            init.normal_(input_tensor, mean=mean, std=std)

            assert self._is_normal(input_tensor, mean, std)

    @unittest.skipIf(not TEST_SCIPY, "Scipy not found.")
    @skipIfTorchDynamo("scipy.kstest is failing under dynamo")
    def test_trunc_normal(self):
        for dims in [1, 2, 4]:
            input_tensor = self._create_random_nd_tensor(dims, size_min=30, size_max=50)
            mean = self._random_float(-3, 3)
            std = self._random_float(0.01, 1)
            a = self._random_float(mean - 2 * std, mean)
            b = self._random_float(mean, mean + 2 * std)
            init.trunc_normal_(input_tensor, mean=mean, std=std, a=a, b=b)

            assert self._is_trunc_normal(input_tensor, mean, std, a, b)

    @unittest.skipIf(not TEST_SCIPY, "Scipy not found.")
    @skipIfTorchDynamo("scipy.kstest is failing under dynamo")
    def test_trunc_normal_generator(self):
        gen = torch.Generator()
        gen.manual_seed(42)
        input_tensor = torch.empty(5)
        init.trunc_normal_(input_tensor, generator=gen)

        ref = torch.empty(5)
        torch.manual_seed(42)
        init.trunc_normal_(ref)

        self.assertEqual(input_tensor, ref)
        assert self._is_trunc_normal(input_tensor, mean=0, std=1, a=0, b=1)

    def test_constant(self):
        for dims in [1, 2, 4]:
            input_tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=5)
            val = self._random_float(1, 10)
            init.constant_(input_tensor, val)

            self.assertEqual(input_tensor, input_tensor.clone().fill_(val))

    def test_ones_and_zeros(self):
        for init_fn_, val in zip([init.ones_, init.zeros_], [1, 0]):
            for dims in [1, 2, 4]:
                input_tensor = self._create_random_nd_tensor(
                    dims, size_min=1, size_max=5
                )
                init_fn_(input_tensor)

                self.assertEqual(input_tensor, input_tensor.clone().fill_(val))

    def test_eye(self):
        input_tensor = self._create_random_nd_tensor(2, size_min=1, size_max=5)
        init.eye_(input_tensor)

        # Check every single element
        for i in range(input_tensor.size(0)):
            for j in range(input_tensor.size(1)):
                if i == j:
                    assert input_tensor[i][j] == 1
                else:
                    assert input_tensor[i][j] == 0

    def test_eye_only_works_on_2d_inputs(self):
        for dims in [1, 3]:
            with self.assertRaises(ValueError):
                tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=3)
                init.eye_(tensor)

    def test_dirac_properties(self):
        for dims in [3, 4, 5]:
            for groups in [1, 2, 3]:
                # prepare random tensor with random sizes, but fits groups
                a, c, d, e = (random.randint(1, 5) for _ in range(4))
                b = random.randint(
                    1, 5 * groups
                )  # same range as a*groups but all range allowed
                # make sure first dim divides by groups
                input_tensor = torch.randn((a * groups, b, c, d, e)[:dims])

                init.dirac_(input_tensor, groups)

                c_out, c_in = input_tensor.size(0) // groups, input_tensor.size(1)
                min_d = min(c_out, c_in)
                # Check number of nonzeros is equivalent to smallest dim (for each group)
                assert torch.nonzero(input_tensor).size(0) == min_d * groups
                # Check sum of values (can have precision issues, hence assertEqual) is also equivalent
                self.assertEqual(input_tensor.sum(), min_d * groups)

    def test_dirac_identity(self):
        for groups in [1, 3]:
            batch, in_c, out_c, size, kernel_size = (
                8,
                3,
                9,
                5,
                3,
            )  # in_c, out_c must divide by groups
            eff_out_c = out_c // groups

            # Test 1D
            input_var = torch.randn(batch, in_c, size)
            filter_var = torch.zeros(eff_out_c, in_c, kernel_size)
            filter_var = torch.cat([filter_var] * groups)
            init.dirac_(filter_var, groups)
            output_var = F.conv1d(input_var, filter_var)
            input_tensor, output_tensor = (
                input_var.data,
                output_var.data,
            )  # Variables do not support nonzero
            for g in range(groups):
                # Assert in_c outputs are preserved (per each group)
                self.assertEqual(
                    input_tensor[:, :, 1:-1],
                    output_tensor[:, eff_out_c * g : eff_out_c * g + in_c, :],
                )
                # Assert extra outputs are 0
                assert (
                    torch.nonzero(
                        output_tensor[:, eff_out_c * g + in_c : eff_out_c * (g + 1), :]
                    ).numel()
                    == 0
                )

            # Test 2D
            input_var = torch.randn(batch, in_c, size, size)
            filter_var = torch.zeros(eff_out_c, in_c, kernel_size, kernel_size)
            filter_var = torch.cat([filter_var] * groups)
            init.dirac_(filter_var, groups)
            output_var = F.conv2d(input_var, filter_var)
            input_tensor, output_tensor = (
                input_var.data,
                output_var.data,
            )  # Variables do not support nonzero
            for g in range(groups):
                # Assert in_c outputs are preserved (per each group)
                self.assertEqual(
                    input_tensor[:, :, 1:-1, 1:-1],
                    output_tensor[:, eff_out_c * g : eff_out_c * g + in_c, :, :],
                )
                # Assert extra outputs are 0
                assert (
                    torch.nonzero(
                        output_tensor[
                            :, eff_out_c * g + in_c : eff_out_c * (g + 1), :, :
                        ]
                    ).numel()
                    == 0
                )

            # Test 3D
            input_var = torch.randn(batch, in_c, size, size, size)
            filter_var = torch.zeros(
                eff_out_c, in_c, kernel_size, kernel_size, kernel_size
            )
            filter_var = torch.cat([filter_var] * groups)
            init.dirac_(filter_var, groups)
            output_var = F.conv3d(input_var, filter_var)
            input_tensor, output_tensor = input_var.data, output_var.data
            for g in range(groups):
                # Assert in_c outputs are preserved (per each group)
                self.assertEqual(
                    input_tensor[:, :, 1:-1, 1:-1, 1:-1],
                    output_tensor[:, eff_out_c * g : eff_out_c * g + in_c, :, :, :],
                )
                # Assert extra outputs are 0
                assert (
                    torch.nonzero(
                        output_tensor[
                            :, eff_out_c * g + in_c : eff_out_c * (g + 1), :, :, :
                        ]
                    ).numel()
                    == 0
                )

    def test_dirac_only_works_on_3_4_5d_inputs(self):
        for dims in [1, 2, 6]:
            with self.assertRaises(ValueError):
                tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=3)
                init.dirac_(tensor)

    def test_xavier_uniform_errors_on_inputs_smaller_than_2d(self):
        for dims in [0, 1]:
            tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=1)
            with self.assertRaises(ValueError):
                init.xavier_uniform_(tensor)

    def test_xavier_normal_errors_on_inputs_smaller_than_2d(self):
        for dims in [0, 1]:
            tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=1)
            with self.assertRaises(ValueError):
                init.xavier_normal_(tensor)

    @unittest.skipIf(not TEST_SCIPY, "Scipy not found.")
    @slowTest
    def test_xavier_uniform(self):
        for use_gain in [True, False]:
            for dims in [2, 4]:
                input_tensor = self._create_random_nd_tensor(
                    dims, size_min=20, size_max=25
                )
                gain = 1

                if use_gain:
                    gain = self._random_float(0.1, 2)
                    init.xavier_uniform_(input_tensor, gain=gain)
                else:
                    init.xavier_uniform_(input_tensor)

                fan_in = input_tensor.size(1)
                fan_out = input_tensor.size(0)
                if input_tensor.dim() > 2:
                    fan_in *= input_tensor[0, 0].numel()
                    fan_out *= input_tensor[0, 0].numel()

                expected_std = gain * math.sqrt(2.0 / (fan_in + fan_out))
                bounds = expected_std * math.sqrt(3)
                assert self._is_uniform(input_tensor, -bounds, bounds)

    @unittest.skipIf(not TEST_SCIPY, "Scipy not found.")
    @skipIfTorchDynamo("scipy.kstest is failing under dynamo")
    def test_xavier_normal(self):
        for use_gain in [True, False]:
            for dims in [2, 4]:
                input_tensor = self._create_random_nd_tensor(
                    dims, size_min=20, size_max=25
                )
                gain = 1

                if use_gain:
                    gain = self._random_float(0.1, 2)
                    init.xavier_normal_(input_tensor, gain=gain)
                else:
                    init.xavier_normal_(input_tensor)

                fan_in = input_tensor.size(1)
                fan_out = input_tensor.size(0)
                if input_tensor.dim() > 2:
                    fan_in *= input_tensor[0, 0].numel()
                    fan_out *= input_tensor[0, 0].numel()

                expected_std = gain * math.sqrt(2.0 / (fan_in + fan_out))
                assert self._is_normal(input_tensor, 0, expected_std)

    def test_kaiming_uniform_errors_on_inputs_smaller_than_2d(self):
        for dims in [0, 1]:
            with self.assertRaises(ValueError):
                tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=1)
                init.kaiming_uniform_(tensor)

    def test_kaiming_normal_errors_on_inputs_smaller_than_2d(self):
        for dims in [0, 1]:
            with self.assertRaises(ValueError):
                tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=1)
                init.kaiming_normal_(tensor)

    def test_kaiming_uniform_warning_on_0element_tensor(self):
        tensor = torch.empty(0, 1)
        with self.assertWarnsRegex(
            UserWarning, "Initializing zero-element tensors is a no-op"
        ):
            _ = init.kaiming_uniform_(tensor)

    def test_kaiming_normal_warning_on_0element_tensor(self):
        tensor = torch.empty(0, 1)
        with self.assertWarnsRegex(
            UserWarning, "Initializing zero-element tensors is a no-op"
        ):
            _ = init.kaiming_normal_(tensor)

    @unittest.skipIf(not TEST_SCIPY, "Scipy not found.")
    @skipIfTorchDynamo("scipy.kstest is failing under dynamo")
    def test_kaiming_uniform(self):
        for use_a in [True, False]:
            for dims in [2, 4]:
                for mode in ["fan_in", "fan_out"]:
                    input_tensor = self._create_random_nd_tensor(
                        dims, size_min=20, size_max=25
                    )
                    if use_a:
                        a = self._random_float(0.1, 2)
                        init.kaiming_uniform_(input_tensor, a=a, mode=mode)
                    else:
                        a = 0
                        init.kaiming_uniform_(input_tensor, mode=mode)

                    fan_in = input_tensor.size(1)
                    fan_out = input_tensor.size(0)
                    if input_tensor.dim() > 2:
                        fan_in *= input_tensor[0, 0].numel()
                        fan_out *= input_tensor[0, 0].numel()

                    if mode == "fan_in":
                        n = fan_in
                    else:
                        n = fan_out

                    expected_std = math.sqrt(2.0 / ((1 + a**2) * n))
                    bounds = expected_std * math.sqrt(3.0)
                    assert self._is_uniform(input_tensor, -bounds, bounds)

    @unittest.skipIf(not TEST_SCIPY, "Scipy not found.")
    @skipIfTorchDynamo("scipy.kstest is failing under dynamo")
    def test_kaiming_normal(self):
        for use_a in [True, False]:
            for dims in [2, 4]:
                for mode in ["fan_in", "fan_out"]:
                    input_tensor = self._create_random_nd_tensor(
                        dims, size_min=20, size_max=25
                    )
                    if use_a:
                        a = self._random_float(0.1, 2)
                        init.kaiming_normal_(input_tensor, a=a, mode=mode)
                    else:
                        a = 0
                        init.kaiming_normal_(input_tensor, mode=mode)

                    fan_in = input_tensor.size(1)
                    fan_out = input_tensor.size(0)
                    if input_tensor.dim() > 2:
                        fan_in *= input_tensor[0, 0].numel()
                        fan_out *= input_tensor[0, 0].numel()

                    if mode == "fan_in":
                        n = fan_in
                    else:
                        n = fan_out

                    expected_std = math.sqrt(2.0 / ((1 + a**2) * n))
                    assert self._is_normal(input_tensor, 0, expected_std)

    def test_sparse_only_works_on_2d_inputs(self):
        for dims in [1, 3]:
            with self.assertRaises(ValueError):
                sparsity = self._random_float(0.1, 0.9)
                tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=3)
                init.sparse_(tensor, sparsity)

    @unittest.skipIf(not TEST_SCIPY, "Scipy not found.")
    @skipIfTorchDynamo("scipy.kstest is failing under dynamo")
    def test_sparse_default_std(self):
        for use_random_std in [True, False]:
            input_tensor = self._create_random_nd_tensor(2, size_min=30, size_max=35)
            rows = input_tensor.size(0)
            sparsity = self._random_float(0.1, 0.2)

            std = 0.01  # default std
            if use_random_std:
                std = self._random_float(0.01, 0.2)
                init.sparse_(input_tensor, sparsity=sparsity, std=std)
            else:
                init.sparse_(input_tensor, sparsity=sparsity)

            for col_idx in range(input_tensor.size(1)):
                column = input_tensor[:, col_idx]
                assert column[column == 0].nelement() >= math.ceil(sparsity * rows)

            assert self._is_normal(input_tensor[input_tensor != 0], 0, std)

    @skipIfNoLapack
    def test_orthogonal(self):
        for use_gain in [True, False]:
            for tensor_size in [[3, 4], [4, 3], [20, 2, 3, 4], [2, 3, 4, 5]]:
                input_tensor = torch.zeros(tensor_size)
                gain = 1.0

                if use_gain:
                    gain = self._random_float(0.1, 2)
                    init.orthogonal_(input_tensor, gain=gain)
                else:
                    init.orthogonal_(input_tensor)

                rows, cols = tensor_size[0], reduce(mul, tensor_size[1:])
                flattened_tensor = input_tensor.view(rows, cols)
                if rows > cols:
                    self.assertEqual(
                        torch.mm(flattened_tensor.t(), flattened_tensor),
                        torch.eye(cols) * gain**2,
                        atol=1e-6,
                        rtol=0,
                    )
                else:
                    self.assertEqual(
                        torch.mm(flattened_tensor, flattened_tensor.t()),
                        torch.eye(rows) * gain**2,
                        atol=1e-6,
                        rtol=0,
                    )

    def test_deprecation(self):
        x = torch.randn(3, 3)

        def fn():
            init.normal(x)

        with self.assertWarnsRegex(
            FutureWarning,
            "deprecated",
            msg="methods not suffixed with underscore should be deprecated",
        ):
            fn()


if __name__ == "__main__":
    run_tests()
