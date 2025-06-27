# Owner(s): ["module: inductor"]

import unittest.mock as mock

import numpy as np
import pandas as pd
import torch
from torch._inductor.mm_kernel_prediction_model import (
    get_nn_x,
    get_total_gb_feature,
    get_total_gflop_feature,
    ModelWrapper,
    NeuralNetwork,
)
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.inductor_utils import HAS_CPU, HAS_GPU


class TestMMKernelPredictionModel(TestCase):
    def test_get_total_gb_feature(self) -> None:
        # Create a test dataframe
        df = pd.DataFrame(
            {
                "dim_m": [1000, 2000],
                "dim_n": [2000, 4000],
                "dim_k": [3000, 6000],
                "dtype_size": [16, 32],  # 16-bit and 32-bit dtypes
            }
        )

        # Calculate expected results manually
        # Formula: (m*k + k*n + m*n) * dtype_size/8 / 1e9
        expected_results = pd.Series([(2 + 3 + 6) * 2 / 1e3, (8 + 12 + 24) * 4 / 1e3])

        # Get actual results
        actual_results = get_total_gb_feature(df)

        # Compare results
        pd.testing.assert_series_equal(
            actual_results, expected_results, check_names=False
        )

    def test_get_total_gflop_feature(self) -> None:
        # Create a test dataframe
        df = pd.DataFrame(
            {
                "dim_m": [1000, 2000],
                "dim_n": [2000, 4000],
                "dim_k": [3000, 6000],
            }
        )

        # Calculate expected results manually
        # Formula: (2 * m * n * k) / 1e9
        expected_results = pd.Series([2 * 12, 2 * 12 * 8])

        # Get actual results
        actual_results = get_total_gflop_feature(df)

        # Compare results
        pd.testing.assert_series_equal(
            actual_results, expected_results, check_names=False
        )

    def test_nn(self) -> None:
        kernel_overhead = 0.5

        nn = NeuralNetwork(
            n_inputs=3,
            hidden_layer_widths=[4, 5],
            kernel_overhead=0.5,
        )
        # Set all weights to 0
        for param in nn.parameters():
            param.data.fill_(0)
        nn.eval()
        self.assertAlmostEqual(
            nn.log_kernel_overhead, torch.log(torch.tensor(0.5)).item()
        )

        x = torch.ones(1, 3)
        with torch.no_grad():
            log_base_pred = nn.linear_relu_stack(x)
            result = nn(x)

        self.assertAlmostEqual(log_base_pred.item(), 0.0)
        # kernel_overhead + e^0
        self.assertAlmostEqual(torch.exp(result).item(), kernel_overhead + 1.0)

    def test_get_nn_x(self) -> None:
        df = pd.DataFrame(
            {
                "dim_m": [10, 12],
                "dim_n": [20, 22],
                "antartica": ["fly", "boat"],
                "dim_k": [4, 2],
                "dtype_size": [16, 32],
                "config_block_n": [32, 64],
                "config_block_k": [16, 32],
                "config_block_m": [1, 2],
                "config_num_stages": [2, 3],
                "config_num_warps": [4, 8],
                "total_gb": [0.1, 0.2],
                "total_gflop": [0.3, 0.4],
                "flops_per_byte": [0.5, 0.6],
            }
        )

        with self.subTest("Mean and variance not provided"):
            t, mean, std = get_nn_x(df=df)
            expected_t = torch.tensor(
                [
                    [
                        -0.7071,
                        -0.7071,
                        -0.7071,
                        0.7071,
                        -0.7071,
                        -0.7071,
                        -0.7071,
                        -0.7071,
                        -0.7071,
                        -0.7071,
                        -0.7071,
                        -0.7071,
                    ],
                    [
                        0.7071,
                        0.7071,
                        0.7071,
                        -0.7071,
                        0.7071,
                        0.7071,
                        0.7071,
                        0.7071,
                        0.7071,
                        0.7071,
                        0.7071,
                        0.7071,
                    ],
                ]
            )
            self.assertTrue(torch.allclose(t, expected_t, atol=1e-4))

            expected_mean = torch.tensor(
                [
                    3.1192,
                    2.3937,
                    3.0434,
                    1.0397,
                    -1.9560,
                    -1.0601,
                    -0.6020,
                    3.1192,
                    0.3466,
                    3.8123,
                    0.8959,
                    1.7329,
                ],
                dtype=torch.double,
            )
            self.assertTrue(torch.allclose(mean, expected_mean, atol=1e-4))
            expected_std = torch.tensor(
                [
                    0.4901,
                    0.1289,
                    0.0674,
                    0.4901,
                    0.4901,
                    0.2034,
                    0.1289,
                    0.4901,
                    0.4901,
                    0.4901,
                    0.2867,
                    0.4901,
                ],
                dtype=torch.float64,
            )
            self.assertTrue(torch.allclose(std, expected_std, atol=1e-4))

        with self.subTest("Mean and variance provided"):
            mean_ = torch.linspace(1, 2, 12, dtype=torch.float64)
            std_ = torch.linspace(3, 4, 12, dtype=torch.float64)
            t, mean, std = get_nn_x(df=df, mean=mean_, std=std_)
            expected_t = torch.tensor(
                [
                    [
                        0.5909,
                        0.3920,
                        0.5701,
                        0.0347,
                        -1.0900,
                        -0.7696,
                        -0.6314,
                        0.3125,
                        -0.4634,
                        0.4315,
                        -0.3111,
                        -0.1534,
                    ],
                    [
                        0.8219,
                        0.4510,
                        0.6000,
                        -0.1771,
                        -0.8839,
                        -0.6863,
                        -0.5800,
                        0.5031,
                        -0.2774,
                        0.6130,
                        -0.2073,
                        0.0199,
                    ],
                ]
            )
            self.assertTrue(torch.allclose(t, expected_t, atol=1e-4))
            self.assertIs(mean, mean_)
            self.assertIs(std, std_)

        with self.subTest("Check against a known solution"):
            cols = [
                "dtype_size",
                "dim_m",
                "dim_n",
                "dim_k",
                "total_gb",
                "total_gflop",
                "flops_per_byte",
                "config_block_k",
                "config_block_m",
                "config_block_n",
                "config_num_stages",
                "config_num_warps",
            ]
            df = pd.DataFrame({col: [np.exp(1), np.exp(3)] for col in cols})
            t, mean, std = get_nn_x(
                df=df, mean=2 * torch.ones(len(cols)), std=torch.ones(len(cols))
            )
            self.assertEqual(t.shape, (2, len(cols)))
            self.assertTrue(torch.allclose(t[0, :], -torch.ones(len(cols))))
            self.assertTrue(torch.allclose(t[1, :], torch.ones(len(cols))))

    def test_model_wrapper(self) -> None:
        # Don't load the real full model
        with mock.patch("torch._inductor.aoti_load_package") as mock_load:
            mock_load.return_value = NeuralNetwork(
                n_inputs=3, hidden_layer_widths=[4, 4]
            )

            wrapper = ModelWrapper()

        # Test vec method
        class MockConfig:
            def all_kwargs(self):
                return {
                    "BLOCK_M": 128,
                    "BLOCK_N": 64,
                    "BLOCK_K": 32,
                    "num_stages": 3,
                    "num_warps": 4,
                }

        config = MockConfig()
        result = wrapper.vec(256, 128, 64, 16, config)
        expected = (256, 128, 64, 16, 128, 64, 32, 3, 4)
        self.assertEqual(result, expected)

        # Test vec_params method
        params = mock.MagicMock(spec=TritonGEMMConfig)
        params.block_m = 128
        params.block_n = 64
        params.block_k = 32
        params.num_stages = 3
        params.num_warps = 4

        result = ModelWrapper.vec_params(256, 128, 64, 16, params)
        expected = (256, 128, 64, 16, 128, 64, 32, 3, 4)
        self.assertEqual(result, expected)

        # Test encode method
        configs = [config]
        with self.assertRaisesRegex(ValueError, "Unsupported dtype"):
            wrapper.encode(256, 128, 64, torch.int32, configs)

        # Mock the get_nn_x function
        expected_shape = (1, 12)

        result = wrapper.encode(256, 128, 64, torch.float16, configs)
        self.assertEqual(result.shape, expected_shape)

        # bfloat16 should have the same encoding as float16
        result_bfloat = wrapper.encode(256, 128, 64, torch.bfloat16, configs)
        self.assertEqual(result_bfloat.shape, shape)
        self.assertTrue(torch.allclose(result_bfloat, result))

        # Test with float32
        result = wrapper.encode(256, 128, 64, torch.float32, configs)
        self.assertGreater(result[0, 0].item(), result_bfloat[0, 0].item())


if __name__ == "__main__":
    if HAS_GPU or HAS_CPU:
        run_tests()
