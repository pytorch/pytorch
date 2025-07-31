# Owner(s): ["module: inductor"]

import os
import unittest
import unittest.mock as mock
from math import exp
from unittest import skipIf

import torch
from torch._inductor.config import (
    parse_matmul_gemm_autotune_benchmark_space,
    parse_matmul_gemm_autotune_search_space,
)
import torch._inductor.config as config
from torch._inductor.models.mm_kernel_prediction_model import (
    _sanitize_path,
    get_model,
    get_nn_x,
    get_total_gb_feature,
    get_total_gflop_feature,
    ModelWrapper,
    NeuralNetwork,
)
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.inductor_utils import HAS_CPU, HAS_GPU


def mock_device_if_cuda_unavailable(function):
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            return function(*args, **kwargs)
        with mock.patch("torch.device", return_value=torch.device("cpu")):
            return function(*args, **kwargs)

    return wrapper


class TestMMKernelPredictionModel(TestCase):
    def test_get_total_gb_feature(self) -> None:
        # Create a test dict of tensors
        dict_of_features = {
            "dim_m": torch.tensor([1000, 2000]),
            "dim_n": torch.tensor([2000, 4000]),
            "dim_k": torch.tensor([3000, 6000]),
            # 16-bit and 32-bit dtypes
            "dtype_size": torch.tensor([16, 32]),
        }

        # Calculate expected results manually
        # Formula: (m*k + k*n + m*n) * dtype_size/8 / 1e9
        expected_results = torch.tensor(
            [(2 + 3 + 6) * 2 / 1e3, (8 + 12 + 24) * 4 / 1e3]
        )

        actual_results = get_total_gb_feature(dict_of_features=dict_of_features)
        self.assertTrue(torch.allclose(actual_results, expected_results))

    def test_get_total_gflop_feature(self) -> None:
        dict_of_features = {
            "dim_m": torch.tensor([1000, 2000]),
            "dim_n": torch.tensor([2000, 4000]),
            "dim_k": torch.tensor([3000, 6000]),
        }

        expected_results = torch.tensor([12.0, 12.0 * 8])
        actual_results = get_total_gflop_feature(dict_of_features=dict_of_features)
        self.assertTrue(torch.allclose(actual_results, expected_results))

    @mock_device_if_cuda_unavailable
    def test_nn(self) -> None:
        kernel_overhead = 0.5

        nn = NeuralNetwork(
            n_inputs=3,
            hidden_layer_widths=[4, 5],
            kernel_overhead=0.5,
        )
        nn.to(device=torch.device("cuda"))
        # Set all weights to 0
        for param in nn.parameters():
            param.data.fill_(0)
        nn.eval()
        self.assertAlmostEqual(
            nn.log_kernel_overhead, torch.log(torch.tensor(0.5)).item()
        )

        x = torch.ones(1, 3, device=torch.device("cuda"))
        with torch.no_grad():
            log_base_pred = nn.linear_relu_stack(x)
            result = nn(x)

        self.assertAlmostEqual(log_base_pred.item(), 0.0)
        # kernel_overhead + e^0
        self.assertAlmostEqual(torch.exp(result).item(), kernel_overhead + 1.0)

    @mock_device_if_cuda_unavailable
    def test_get_nn_x(self) -> None:
        dict_of_features = {
            "dim_m": [10, 12],
            "dim_n": [20, 22],
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
        dict_of_features = {k: torch.tensor(v) for k, v in dict_of_features.items()}

        with self.subTest("Mean and variance not provided"):
            t, mean, std = get_nn_x(dict_of_features=dict_of_features)
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
                ],
                device=torch.device("cuda"),
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
                device=torch.device("cuda"),
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
                device=torch.device("cuda"),
            )
            self.assertTrue(torch.allclose(std, expected_std, atol=1e-4))

        with self.subTest("Mean and variance provided"):
            mean_ = torch.linspace(
                1, 2, 12, dtype=torch.float64, device=torch.device("cuda")
            )
            std_ = torch.linspace(
                3, 4, 12, dtype=torch.float64, device=torch.device("cuda")
            )
            t, mean, std = get_nn_x(
                dict_of_features=dict_of_features, mean=mean_, std=std_
            )
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
                ],
                device=torch.device("cuda"),
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
            dict_of_features = {col: torch.tensor([exp(1), exp(3)]) for col in cols}
            t, mean, std = get_nn_x(
                dict_of_features=dict_of_features,
                mean=2 * torch.ones(len(cols), device=torch.device("cuda")),
                std=torch.ones(len(cols), device=torch.device("cuda")),
            )
            self.assertEqual(t.shape, (2, len(cols)))
            self.assertTrue(
                torch.allclose(
                    t[0, :], -torch.ones(len(cols), device=torch.device("cuda"))
                )
            )
            self.assertTrue(
                torch.allclose(
                    t[1, :], torch.ones(len(cols), device=torch.device("cuda"))
                )
            )

    def test_sanitize_path(self) -> None:
        """Test the _sanitize_path function with various inputs."""
        # Test basic sanitization
        self.assertEqual(_sanitize_path("NVIDIA H100"), "nvidia_h100")
        self.assertEqual(
            _sanitize_path("  NVIDIA GeForce RTX 4090  "), "nvidia_geforce_rtx_4090"
        )

        # Test special characters
        self.assertEqual(_sanitize_path('Test<>:"/\\|?*File'), "test_file")

        # Test whitespace handling
        self.assertEqual(
            _sanitize_path("Test   Multiple   Spaces"), "test_multiple_spaces"
        )

        # Test tab and newline characters
        self.assertEqual(_sanitize_path("Test\t\n\r\f\vChars"), "test_chars")

        # Test multiple underscores
        self.assertEqual(
            _sanitize_path("Test___Multiple___Underscores"), "test_multiple_underscores"
        )

        # Test leading/trailing underscores
        self.assertEqual(_sanitize_path("___Test___"), "test")

        # Test empty string
        self.assertEqual(_sanitize_path(""), "")
        self.assertEqual(_sanitize_path("   "), "")

    def test_model_wrapper_get_device_model_path(self) -> None:
        """Test the _get_device_model_path method."""
        wrapper = ModelWrapper.__new__(ModelWrapper)  # Create without calling __init__

        # Test with default config
        with mock.patch("torch._inductor.config.fast_autotune_model_directory", None):
            path = wrapper._get_device_model_path("NVIDIA H100")
            expected_dir = os.path.dirname(
                os.path.abspath(__file__).replace(
                    "test/inductor", "torch/_inductor/models"
                )
            )
            expected_path = os.path.join(
                expected_dir, "artifacts", "nvidia_h100_triton_mm.pt2"
            )
            self.assertEqual(path, expected_path)

        # Test with custom config directory
        with mock.patch(
            "torch._inductor.config.fast_autotune_model_directory", "/custom/path"
        ):
            path = wrapper._get_device_model_path("NVIDIA H100")
            expected_path = os.path.join(
                "/custom/path", "artifacts", "nvidia_h100_triton_mm.pt2"
            )
            self.assertEqual(path, expected_path)

    def test_model_wrapper_init_cuda_not_available(self) -> None:
        """Test ModelWrapper initialization when CUDA is not available."""
        with mock.patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(
                RuntimeError, "ModelWrapper created when CUDA is not available"
            ):
                ModelWrapper()

    @mock_device_if_cuda_unavailable
    def test_model_wrapper_init_fallback_model(self) -> None:
        """Test ModelWrapper initialization with fallback to H100 model."""
        mock_model = NeuralNetwork(n_inputs=12, hidden_layer_widths=[64, 32])

        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.get_device_name", return_value="Unknown GPU"),
            mock.patch("os.path.exists") as mock_exists,
        ):
            # Unknown GPU model doesn't exist, so it falls back to H100
            mock_exists.return_value = False

            self.assertRaises(RuntimeError, ModelWrapper)

            # Should have checked if unknown GPU model exists once
            self.assertTrue(mock_exists.call_count >= 1)

    def test_model_wrapper_inference(self) -> None:
        """Test the inference method."""
        mock_model = mock.MagicMock()
        mock_output = torch.tensor([[1.5, 2.0]], device=torch.device("cuda"))
        mock_model.return_value = mock_output

        with (
            mock.patch("torch._inductor.aoti_load_package", return_value=mock_model)
        ):
            wrapper = ModelWrapper()

            input_tensor = torch.randn(2, 12, device=torch.device("cuda"))
            result = wrapper.inference(input_tensor)

            mock_model.assert_called_once_with(input_tensor)
            self.assertEqual(result, mock_output)

    @skipIf(not torch.cuda.is_available(), "CUDA is not available")
    def test_model_wrapper_decode(self) -> None:
        """Test the decode method."""
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch._inductor.aoti_load_package"),
        ):
            wrapper = ModelWrapper()

            input_tensor = torch.tensor([[1.0, 2.0]], device=torch.device("cuda"))
            result = wrapper.decode(input_tensor)

            # decode should return the input tensor unchanged
            self.assertTrue(torch.equal(result, input_tensor))

    @mock_device_if_cuda_unavailable
    def test_model_wrapper_device_name_parameter(self) -> None:
        """Test ModelWrapper initialization with explicit device name."""
        mock_model = NeuralNetwork(n_inputs=12, hidden_layer_widths=[64, 32])

        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("os.path.exists", return_value=True),
            mock.patch(
                "torch._inductor.aoti_load_package", return_value=mock_model
            ) as mock_load,
        ):
            ModelWrapper(device_name="Custom GPU")

            # Should have used the provided device name
            mock_load.assert_called_once()
            # Verify the path contains the sanitized device name
            args = mock_load.call_args[0]
            self.assertIn("custom_gpu_triton_mm.pt2", args[0])

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is not available")
    def test_get_model_function(self) -> None:
        """Test the get_model singleton function."""
        with mock.patch("torch.cuda.is_available", return_value=False):
            result = get_model()
            self.assertIsNone(result)

        mock_model = NeuralNetwork(n_inputs=12, hidden_layer_widths=[64, 32])
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch._inductor.aoti_load_package", return_value=mock_model),
        ):
            # Clear the cache first
            get_model.cache_clear()

            result1 = get_model()
            result2 = get_model()

            # Should return the same instance (cached)
            self.assertIsInstance(result1, ModelWrapper)
            self.assertIs(result1, result2)

    @mock_device_if_cuda_unavailable
    def test_neural_network_forward_edge_cases(self) -> None:
        """Test NeuralNetwork forward method with edge cases."""
        nn = NeuralNetwork(n_inputs=2, hidden_layer_widths=[4], kernel_overhead=0.001)
        nn.to(device=torch.device("cuda"))
        nn.eval()  # Set to evaluation mode to avoid BatchNorm issues with single samples

        # Test with different input shapes
        x_single = torch.randn(1, 2, device=torch.device("cuda"))
        x_batch = torch.randn(5, 2, device=torch.device("cuda"))

        with torch.no_grad():
            result_single = nn(x_single)
            result_batch = nn(x_batch)

        self.assertEqual(result_single.shape, (1, 1))
        self.assertEqual(result_batch.shape, (5, 1))

        # Results should be finite
        self.assertTrue(torch.isfinite(result_single).all())
        self.assertTrue(torch.isfinite(result_batch).all())

    @skipIf(not torch.cuda.is_available(), "CUDA is not available")
    def test_model_wrapper_encode_edge_cases(self) -> None:
        """Test ModelWrapper encode method with edge cases."""
        mock_model = NeuralNetwork(n_inputs=12, hidden_layer_widths=[64, 32])

        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch._inductor.aoti_load_package", return_value=mock_model),
        ):
            wrapper = ModelWrapper()

        class MockConfig:
            def all_kwargs(self):
                return {
                    "BLOCK_M": 64,
                    "BLOCK_N": 32,
                    "BLOCK_K": 16,
                    "num_stages": 2,
                    "num_warps": 2,
                }

        config = MockConfig()

        # Test with very small dimensions
        result = wrapper.encode(1, 1, 1, torch.float16, [config])
        self.assertEqual(result.shape, (1, 12))

        # Test with very large dimensions
        result = wrapper.encode(10000, 10000, 10000, torch.float32, [config])
        self.assertEqual(result.shape, (1, 12))

        # Test with multiple configs
        configs = [config, config, config]
        result = wrapper.encode(256, 128, 64, torch.bfloat16, configs)
        self.assertEqual(result.shape, (3, 12))

    def test_get_total_gb_feature_edge_cases(self) -> None:
        """Test get_total_gb_feature with edge cases."""
        # Test with very small values
        features_small = {
            "dim_m": torch.tensor([1]),
            "dim_n": torch.tensor([1]),
            "dim_k": torch.tensor([1]),
            "dtype_size": torch.tensor([8]),  # 8-bit dtype
        }
        result = get_total_gb_feature(dict_of_features=features_small)
        expected = (1 * 1 + 1 * 1 + 1 * 1) * (8 / 8) / 1e9  # 3 bytes / 1e9
        self.assertAlmostEqual(result[0], expected)

        # Test with very large values
        df_large = {
            "dim_m": torch.tensor([100000]),
            "dim_n": torch.tensor([100000]),
            "dim_k": torch.tensor([100000]),
            "dtype_size": torch.tensor([64]),  # 64-bit dtype
        }
        result = get_total_gb_feature(dict_of_features=df_large)
        expected = (
            (100000 * 100000 + 100000 * 100000 + 100000 * 100000) * (64 / 8) / 1e9
        )
        self.assertAlmostEqual(result[0].item(), expected, places=4)

    def test_get_total_gflop_feature_edge_cases(self) -> None:
        """Test get_total_gflop_feature with edge cases."""
        # Test with very small values
        features_small = {
            "dim_m": torch.tensor([1]),
            "dim_n": torch.tensor([1]),
            "dim_k": torch.tensor([1]),
        }
        result = get_total_gflop_feature(dict_of_features=features_small)
        expected = (2 * 1 * 1 * 1) / 1e9
        self.assertAlmostEqual(result[0], expected)

        # Test with asymmetric dimensions
        features_asym = {
            "dim_m": torch.tensor([1000]),
            "dim_n": torch.tensor([10]),
            "dim_k": torch.tensor([100000]),
        }
        result = get_total_gflop_feature(dict_of_features=features_asym)
        expected = (2 * 1000 * 10 * 100000) / 1e9
        self.assertAlmostEqual(result[0], expected)


class TestConfigParsing(TestCase):
    def setUp(self):
        self.original_env = {}
        env_vars = [
            "TORCHINDUCTOR_MATMUL_GEMM_AUTOTUNE_BENCHMARK_SPACE",
            "TORCHINDUCTOR_FAST_AUTOTUNE",
        ]
        for var in env_vars:
            self.original_env[var] = os.environ.get(var)

    def tearDown(self):
        for var, value in self.original_env.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value

    def test_parse_matmul_gemm_autotune_benchmark_space_default(self):
        """Test that benchmark space defaults to 'SAME' when no env var is set"""
        os.environ.pop("TORCHINDUCTOR_MATMUL_GEMM_AUTOTUNE_BENCHMARK_SPACE", None)
        os.environ.pop("TORCHINDUCTOR_FAST_AUTOTUNE", None)

        result = parse_matmul_gemm_autotune_benchmark_space()
        self.assertEqual(result, "SAME")

    def test_parse_matmul_gemm_autotune_benchmark_space_same(self):
        """Test that benchmark space returns 'SAME' when explicitly set"""
        os.environ["TORCHINDUCTOR_MATMUL_GEMM_AUTOTUNE_BENCHMARK_SPACE"] = "SAME"

        result = parse_matmul_gemm_autotune_benchmark_space()
        self.assertEqual(result, "SAME")

    def test_parse_matmul_gemm_autotune_benchmark_space_default_string(self):
        """Test that benchmark space returns 'DEFAULT' when set to 'DEFAULT'"""
        os.environ["TORCHINDUCTOR_MATMUL_GEMM_AUTOTUNE_BENCHMARK_SPACE"] = "DEFAULT"

        result = parse_matmul_gemm_autotune_benchmark_space()
        self.assertEqual(result, "DEFAULT")

    def test_parse_matmul_gemm_autotune_benchmark_space_integer(self):
        """Test that benchmark space returns integer when set to a number"""
        os.environ["TORCHINDUCTOR_MATMUL_GEMM_AUTOTUNE_BENCHMARK_SPACE"] = "5"

        result = parse_matmul_gemm_autotune_benchmark_space()
        self.assertEqual(result, 5)

    def test_parse_matmul_gemm_autotune_benchmark_space_invalid_fallback(self):
        """Test that benchmark space falls back to 'SAME' when invalid value is set and fast_autotune is not enabled"""
        os.environ["TORCHINDUCTOR_MATMUL_GEMM_AUTOTUNE_BENCHMARK_SPACE"] = "invalid"
        os.environ.pop("TORCHINDUCTOR_FAST_AUTOTUNE", None)

        result = parse_matmul_gemm_autotune_benchmark_space()
        self.assertEqual(result, "SAME")

    def test_parse_matmul_gemm_autotune_search_space_default(self):
        """Test that search space defaults to 'DEFAULT' when benchmark space is 'SAME'"""
        os.environ.pop("TORCHINDUCTOR_MATMUL_GEMM_AUTOTUNE_BENCHMARK_SPACE", None)
        with config.patch("fast_autotune", False):

            result = parse_matmul_gemm_autotune_search_space()
            self.assertEqual(result, "DEFAULT")

    def test_parse_matmul_gemm_autotune_search_space_exhaustive_with_model_config(self):
        """Test that search space returns 'EXHAUSTIVE' when model configs are used (benchmark space != 'SAME')"""
        os.environ["TORCHINDUCTOR_MATMUL_GEMM_AUTOTUNE_BENCHMARK_SPACE"] = "DEFAULT"

        result = parse_matmul_gemm_autotune_search_space()
        self.assertEqual(result, "EXHAUSTIVE")

    def test_parse_matmul_gemm_autotune_search_space_exhaustive_with_integer_config(
        self,
    ):
        """Test that search space returns 'EXHAUSTIVE' when benchmark space is set to an integer"""
        os.environ["TORCHINDUCTOR_MATMUL_GEMM_AUTOTUNE_BENCHMARK_SPACE"] = "10"

        result = parse_matmul_gemm_autotune_search_space()
        self.assertEqual(result, "EXHAUSTIVE")

    def test_parse_matmul_gemm_autotune_search_space_with_explicit_search_space_env(
        self,
    ):
        """Test that search space respects explicit SEARCH_SPACE env var when benchmark space is 'SAME'"""
        # Set benchmark space to SAME but explicitly set search space
        os.environ["TORCHINDUCTOR_MATMUL_GEMM_AUTOTUNE_BENCHMARK_SPACE"] = "SAME"

        with mock.patch.dict(
            os.environ,
            {"TORCHINDUCTOR_MATMUL_GEMM_AUTOTUNE_BENCHMARK_SPACE": "EXHAUSTIVE"},
        ):
            result = parse_matmul_gemm_autotune_search_space()
            self.assertEqual(result, "EXHAUSTIVE")

    def test_parse_matmul_gemm_autotune_benchmark_space(self):
        # with mock.patch.dict(
        #     os.environ, {"TORCHINDUCTOR_MATMUL_GEMM_AUTOTUNE_BENCHMARK_SPACE": "5"}
        # ):
        #     self.assertEqual(parse_matmul_gemm_autotune_benchmark_space(), 5)

        with config.patch({"fast_autotune": True, "matmul_gemm_autotune_benchmark_space": "fish"}):
            self.assertEqual(parse_matmul_gemm_autotune_benchmark_space(), 1)

        with config.patch({"matmul_gemm_autotune_benchmark_space": "fish"}):
            self.assertEqual(parse_matmul_gemm_autotune_benchmark_space(), "SAME")

        self.assertEqual(parse_matmul_gemm_autotune_benchmark_space(), "SAME")

    def test_parse_matmul_gemm_autotune_search_space(self):
        with config.patch("fast_autotune", False):
            self.assertEqual(parse_matmul_gemm_autotune_search_space(), "DEFAULT")

            # Case 2: Benchmarking_space is an int
            with mock.patch.dict(
                os.environ,
                {"TORCHINDUCTOR_MATMUL_GEMM_AUTOTUNE_BENCHMARK_SPACE": "1"},
            ):
                self.assertEqual(parse_matmul_gemm_autotune_search_space(), "EXHAUSTIVE")

            # Case 3: Benchmarking_space is "DEFAULT"
            with mock.patch.dict(
                os.environ,
                {"TORCHINDUCTOR_MATMUL_GEMM_AUTOTUNE_BENCHMARK_SPACE": "Invalid"},
            ):
                self.assertEqual(parse_matmul_gemm_autotune_search_space(), "DEFAULT")

if __name__ == "__main__":
    if HAS_GPU or HAS_CPU:
        run_tests()
