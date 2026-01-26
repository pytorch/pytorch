"""
Unit tests for PyTorch test environment configuration.
"""

import os
import unittest
from unittest.mock import patch

from cli.lib.core.pytorch.env_config import TestEnvironment
from cli.lib.core.pytorch.test_setup import (
    CPUAffinityConfig,
    InductorTestConfig,
    ROCmTestSetup,
    XPUTestSetup,
)


class TestTestEnvironment(unittest.TestCase):
    """Tests for TestEnvironment class."""

    def test_default_initialization(self):
        """Test default initialization with no env vars set."""
        with patch.dict(os.environ, {}, clear=True):
            env = TestEnvironment()
            self.assertEqual(env.build_environment, "")
            self.assertEqual(env.test_config, "default")
            self.assertEqual(env.shard_number, 1)
            self.assertEqual(env.num_test_shards, 1)

    def test_initialization_from_env_vars(self):
        """Test initialization reads from environment variables."""
        test_env = {
            "BUILD_ENVIRONMENT": "linux-focal-cuda12.1-py3.10",
            "TEST_CONFIG": "inductor",
            "SHARD_NUMBER": "3",
            "NUM_TEST_SHARDS": "10",
        }
        with patch.dict(os.environ, test_env, clear=True):
            env = TestEnvironment()
            self.assertEqual(env.build_environment, "linux-focal-cuda12.1-py3.10")
            self.assertEqual(env.test_config, "inductor")
            self.assertEqual(env.shard_number, 3)
            self.assertEqual(env.num_test_shards, 10)

    def test_direct_values_override_env_vars(self):
        """Test that direct constructor values override environment variables."""
        test_env = {
            "BUILD_ENVIRONMENT": "from-env",
            "TEST_CONFIG": "from-env",
            "SHARD_NUMBER": "99",
            "NUM_TEST_SHARDS": "100",
        }
        with patch.dict(os.environ, test_env, clear=True):
            env = TestEnvironment(
                build_environment="from-cli",
                test_config="cli-config",
                shard_number=1,
                num_test_shards=4,
            )
            self.assertEqual(env.build_environment, "from-cli")
            self.assertEqual(env.test_config, "cli-config")
            self.assertEqual(env.shard_number, 1)
            self.assertEqual(env.num_test_shards, 4)

    def test_partial_override(self):
        """Test that only specified values override env vars."""
        test_env = {
            "BUILD_ENVIRONMENT": "from-env",
            "TEST_CONFIG": "from-env",
            "SHARD_NUMBER": "5",
            "NUM_TEST_SHARDS": "10",
        }
        with patch.dict(os.environ, test_env, clear=True):
            # Only override build_environment, rest should come from env
            env = TestEnvironment(build_environment="from-cli")
            self.assertEqual(env.build_environment, "from-cli")
            self.assertEqual(env.test_config, "from-env")
            self.assertEqual(env.shard_number, 5)
            self.assertEqual(env.num_test_shards, 10)

    def test_from_args_factory(self):
        """Test TestEnvironment.from_args() factory method."""

        class MockArgs:
            build_environment = "linux-focal-cuda12.1-py3.10"
            test_config = "inductor"
            shard_id = 2
            num_shards = 8

        with patch.dict(os.environ, {}, clear=True):
            env = TestEnvironment.from_args(MockArgs())
            self.assertEqual(env.build_environment, "linux-focal-cuda12.1-py3.10")
            self.assertEqual(env.test_config, "inductor")
            self.assertEqual(env.shard_number, 2)
            self.assertEqual(env.num_test_shards, 8)

    def test_from_args_with_missing_attrs(self):
        """Test from_args handles missing attributes gracefully."""

        class MockArgs:
            # Only has shard_id, missing other attributes
            shard_id = 3

        test_env = {
            "BUILD_ENVIRONMENT": "from-env",
            "TEST_CONFIG": "from-env",
        }
        with patch.dict(os.environ, test_env, clear=True):
            env = TestEnvironment.from_args(MockArgs())
            # Missing attrs should fall back to env vars
            self.assertEqual(env.build_environment, "from-env")
            self.assertEqual(env.test_config, "from-env")
            self.assertEqual(env.shard_number, 3)  # From args
            self.assertEqual(env.num_test_shards, 1)  # Default

    def test_base_env_always_set(self):
        """Test that base environment variables are always set."""
        with patch.dict(os.environ, {}, clear=True):
            env = TestEnvironment()
            updates = env.get_updates()
            self.assertEqual(updates["TERM"], "vt100")
            self.assertEqual(updates["LANG"], "C.UTF-8")
            self.assertEqual(updates["TORCH_SERIALIZATION_DEBUG"], "1")

    def test_cuda_build_detection(self):
        """Test CUDA build environment detection."""
        test_env = {"BUILD_ENVIRONMENT": "linux-focal-cuda12.1-py3.10"}
        with patch.dict(os.environ, test_env, clear=True):
            env = TestEnvironment()
            self.assertTrue(env.is_cuda)
            self.assertFalse(env.is_rocm)
            self.assertFalse(env.is_xpu)

    def test_rocm_build_detection(self):
        """Test ROCm build environment detection."""
        test_env = {"BUILD_ENVIRONMENT": "linux-focal-rocm5.4-py3.10"}
        with patch.dict(os.environ, test_env, clear=True):
            env = TestEnvironment()
            self.assertFalse(env.is_cuda)
            self.assertTrue(env.is_rocm)
            self.assertFalse(env.is_xpu)

    def test_xpu_build_detection(self):
        """Test XPU build environment detection."""
        test_env = {"BUILD_ENVIRONMENT": "linux-focal-xpu-py3.10"}
        with patch.dict(os.environ, test_env, clear=True):
            env = TestEnvironment()
            self.assertFalse(env.is_cuda)
            self.assertFalse(env.is_rocm)
            self.assertTrue(env.is_xpu)

    def test_asan_build_detection(self):
        """Test ASAN build environment detection."""
        test_env = {"BUILD_ENVIRONMENT": "linux-focal-py3.10-clang10-asan"}
        with patch.dict(os.environ, test_env, clear=True):
            env = TestEnvironment()
            self.assertTrue(env.is_asan)

    def test_device_visibility_default_config(self):
        """Test device visibility for default test config."""
        test_env = {"TEST_CONFIG": "default"}
        with patch.dict(os.environ, test_env, clear=True):
            env = TestEnvironment()
            updates = env.get_updates()
            self.assertEqual(updates["CUDA_VISIBLE_DEVICES"], "0")
            self.assertEqual(updates["HIP_VISIBLE_DEVICES"], "0")

    def test_device_visibility_distributed_rocm(self):
        """Test device visibility for distributed ROCm tests."""
        test_env = {
            "BUILD_ENVIRONMENT": "linux-focal-rocm5.4-py3.10",
            "TEST_CONFIG": "distributed",
        }
        with patch.dict(os.environ, test_env, clear=True):
            env = TestEnvironment()
            updates = env.get_updates()
            self.assertEqual(updates["HIP_VISIBLE_DEVICES"], "0,1,2,3")

    def test_slow_test_flags(self):
        """Test slow test config sets appropriate flags."""
        test_env = {"TEST_CONFIG": "slow"}
        with patch.dict(os.environ, test_env, clear=True):
            env = TestEnvironment()
            updates = env.get_updates()
            self.assertEqual(updates["PYTORCH_TEST_WITH_SLOW"], "1")
            self.assertEqual(updates["PYTORCH_TEST_SKIP_FAST"], "1")

    def test_slow_gradcheck_flags(self):
        """Test slow-gradcheck build sets appropriate flags."""
        test_env = {"BUILD_ENVIRONMENT": "linux-focal-py3.10-slow-gradcheck"}
        with patch.dict(os.environ, test_env, clear=True):
            env = TestEnvironment()
            updates = env.get_updates()
            self.assertEqual(updates["PYTORCH_TEST_WITH_SLOW_GRADCHECK"], "1")
            self.assertEqual(updates["PYTORCH_TEST_CUDA_MEM_LEAK_CHECK"], "1")

    def test_cuda_device_only_for(self):
        """Test CUDA/ROCm builds set device-only-for flag."""
        test_env = {"BUILD_ENVIRONMENT": "linux-focal-cuda12.1-py3.10"}
        with patch.dict(os.environ, test_env, clear=True):
            env = TestEnvironment()
            updates = env.get_updates()
            self.assertEqual(updates["PYTORCH_TESTING_DEVICE_ONLY_FOR"], "cuda")

    def test_xpu_device_only_for(self):
        """Test XPU builds set appropriate flags."""
        test_env = {"BUILD_ENVIRONMENT": "linux-focal-xpu-py3.10"}
        with patch.dict(os.environ, test_env, clear=True):
            env = TestEnvironment()
            updates = env.get_updates()
            self.assertEqual(updates["PYTORCH_TESTING_DEVICE_ONLY_FOR"], "xpu")
            self.assertEqual(updates["PYTHON_TEST_EXTRA_OPTION"], "--xpu")

    def test_crossref_test_flag(self):
        """Test crossref test config sets flag."""
        test_env = {"TEST_CONFIG": "dynamo_crossref"}
        with patch.dict(os.environ, test_env, clear=True):
            env = TestEnvironment()
            updates = env.get_updates()
            self.assertEqual(updates["PYTORCH_TEST_WITH_CROSSREF"], "1")

    def test_valgrind_default_on(self):
        """Test valgrind is ON by default."""
        with patch.dict(os.environ, {}, clear=True):
            env = TestEnvironment()
            updates = env.get_updates()
            self.assertEqual(updates["VALGRIND"], "ON")

    def test_valgrind_off_for_clang9(self):
        """Test valgrind is OFF for clang9 builds."""
        test_env = {"BUILD_ENVIRONMENT": "linux-focal-py3.10-clang9"}
        with patch.dict(os.environ, test_env, clear=True):
            env = TestEnvironment()
            updates = env.get_updates()
            self.assertEqual(updates["VALGRIND"], "OFF")

    def test_valgrind_off_for_rocm(self):
        """Test valgrind is OFF for ROCm builds."""
        test_env = {"BUILD_ENVIRONMENT": "linux-focal-rocm5.4-py3.10"}
        with patch.dict(os.environ, test_env, clear=True):
            env = TestEnvironment()
            updates = env.get_updates()
            self.assertEqual(updates["VALGRIND"], "OFF")

    def test_valgrind_off_for_aarch64(self):
        """Test valgrind is OFF for aarch64 builds."""
        test_env = {"BUILD_ENVIRONMENT": "linux-aarch64-py3.10"}
        with patch.dict(os.environ, test_env, clear=True):
            env = TestEnvironment()
            updates = env.get_updates()
            self.assertEqual(updates["VALGRIND"], "OFF")

    def test_asan_options_set(self):
        """Test ASAN options are set for ASAN builds."""
        test_env = {"BUILD_ENVIRONMENT": "linux-focal-py3.10-asan"}
        with patch.dict(os.environ, test_env, clear=True):
            env = TestEnvironment()
            updates = env.get_updates()
            self.assertIn("ASAN_OPTIONS", updates)
            self.assertIn("detect_leaks=0", updates["ASAN_OPTIONS"])
            self.assertEqual(updates["PYTORCH_TEST_WITH_ASAN"], "1")
            self.assertEqual(updates["PYTORCH_TEST_WITH_UBSAN"], "1")
            self.assertEqual(updates["VALGRIND"], "OFF")

    def test_asan_cuda_extra_option(self):
        """Test ASAN + CUDA builds have extra protect_shadow_gap option."""
        test_env = {"BUILD_ENVIRONMENT": "linux-focal-cuda12.1-py3.10-asan"}
        with patch.dict(os.environ, test_env, clear=True):
            env = TestEnvironment()
            updates = env.get_updates()
            self.assertIn("protect_shadow_gap=0", updates["ASAN_OPTIONS"])

    def test_cpu_capability_no_avx2(self):
        """Test CPU capability for NO_AVX2 config."""
        test_env = {"TEST_CONFIG": "nogpu_NO_AVX2"}
        with patch.dict(os.environ, test_env, clear=True):
            env = TestEnvironment()
            updates = env.get_updates()
            self.assertEqual(updates["ATEN_CPU_CAPABILITY"], "default")

    def test_cpu_capability_avx512(self):
        """Test CPU capability for AVX512 config."""
        test_env = {"TEST_CONFIG": "nogpu_AVX512"}
        with patch.dict(os.environ, test_env, clear=True):
            env = TestEnvironment()
            updates = env.get_updates()
            self.assertEqual(updates["ATEN_CPU_CAPABILITY"], "avx2")

    def test_legacy_driver_flag(self):
        """Test legacy NVIDIA driver flag is set."""
        test_env = {"TEST_CONFIG": "legacy_nvidia_driver"}
        with patch.dict(os.environ, test_env, clear=True):
            env = TestEnvironment()
            updates = env.get_updates()
            self.assertEqual(updates["USE_LEGACY_DRIVER"], "1")

    def test_apply_updates_environ(self):
        """Test apply() updates os.environ."""
        with patch.dict(os.environ, {}, clear=True):
            env = TestEnvironment()
            env.apply()
            self.assertEqual(os.environ.get("TERM"), "vt100")
            self.assertEqual(os.environ.get("LANG"), "C.UTF-8")

    def test_as_dict_merges_environ(self):
        """Test as_dict() merges with existing environ."""
        with patch.dict(os.environ, {"EXISTING_VAR": "existing_value"}, clear=True):
            env = TestEnvironment()
            result = env.as_dict()
            self.assertEqual(result["EXISTING_VAR"], "existing_value")
            self.assertEqual(result["TERM"], "vt100")

    def test_set_custom_env_var(self):
        """Test set() adds custom environment variable."""
        with patch.dict(os.environ, {}, clear=True):
            env = TestEnvironment()
            env.set("CUSTOM_VAR", "custom_value")
            updates = env.get_updates()
            self.assertEqual(updates["CUSTOM_VAR"], "custom_value")

    def test_unset_removes_env_var(self):
        """Test unset() removes environment variable."""
        with patch.dict(os.environ, {}, clear=True):
            env = TestEnvironment()
            env.set("CUSTOM_VAR", "custom_value")
            env.unset("CUSTOM_VAR")
            updates = env.get_updates()
            self.assertNotIn("CUSTOM_VAR", updates)

    def test_is_distributed_test(self):
        """Test is_distributed_test property."""
        test_env = {"TEST_CONFIG": "distributed"}
        with patch.dict(os.environ, test_env, clear=True):
            env = TestEnvironment()
            self.assertTrue(env.is_distributed_test)

    def test_is_inductor_test(self):
        """Test is_inductor_test property."""
        test_env = {"TEST_CONFIG": "inductor_distributed"}
        with patch.dict(os.environ, test_env, clear=True):
            env = TestEnvironment()
            self.assertTrue(env.is_inductor_test)

    def test_is_dynamo_test(self):
        """Test is_dynamo_test property."""
        test_env = {"TEST_CONFIG": "dynamo_wrapped"}
        with patch.dict(os.environ, test_env, clear=True):
            env = TestEnvironment()
            self.assertTrue(env.is_dynamo_test)

    def test_is_benchmark_test(self):
        """Test is_benchmark_test property for various configs."""
        benchmark_configs = ["torchbench", "huggingface", "timm_models", "perf"]
        for config in benchmark_configs:
            test_env = {"TEST_CONFIG": config}
            with patch.dict(os.environ, test_env, clear=True):
                env = TestEnvironment()
                self.assertTrue(
                    env.is_benchmark_test, f"Expected is_benchmark_test=True for {config}"
                )


class TestInductorTestConfig(unittest.TestCase):
    """Tests for InductorTestConfig class."""

    def test_cpp_wrapper_enabled(self):
        """Test C++ wrapper is enabled for inductor_cpp_wrapper config."""
        test_env = {"TEST_CONFIG": "inductor_cpp_wrapper"}
        with patch.dict(os.environ, test_env, clear=True):
            config = InductorTestConfig()
            updates = config.get_updates()
            self.assertEqual(updates["TORCHINDUCTOR_CPP_WRAPPER"], "1")

    def test_cpp_wrapper_not_enabled_by_default(self):
        """Test C++ wrapper is not enabled by default."""
        with patch.dict(os.environ, {"TEST_CONFIG": "default"}, clear=True):
            config = InductorTestConfig()
            updates = config.get_updates()
            self.assertNotIn("TORCHINDUCTOR_CPP_WRAPPER", updates)

    def test_max_autotune_enabled(self):
        """Test max autotune is enabled for appropriate config."""
        test_env = {"TEST_CONFIG": "max_autotune_inductor"}
        with patch.dict(os.environ, test_env, clear=True):
            config = InductorTestConfig()
            updates = config.get_updates()
            self.assertEqual(updates["TORCHINDUCTOR_MAX_AUTOTUNE"], "1")


class TestROCmTestSetup(unittest.TestCase):
    """Tests for ROCmTestSetup class."""

    def test_benchmark_subdir(self):
        """Test MAYBE_ROCM is set for benchmark result directories."""
        subdir = ROCmTestSetup.get_benchmark_subdir()
        self.assertEqual(subdir, "rocm/")


class TestXPUTestSetup(unittest.TestCase):
    """Tests for XPUTestSetup class."""

    def test_identifies_oneapi_scripts(self):
        """Test that XPU config identifies oneAPI scripts."""
        # This test will vary based on whether oneAPI is installed
        config = XPUTestSetup()
        # Just verify the method doesn't crash
        commands = config.get_source_commands()
        self.assertIsInstance(commands, list)


class TestCPUAffinityConfig(unittest.TestCase):
    """Tests for CPUAffinityConfig class."""

    def test_omp_num_threads_set(self):
        """Test OMP_NUM_THREADS is computed."""
        config = CPUAffinityConfig()
        updates = config.get_updates()
        # Should have some value set (may vary by system)
        self.assertIn("OMP_NUM_THREADS", updates)
        # Value should be a positive integer
        self.assertGreater(int(updates["OMP_NUM_THREADS"]), 0)

    def test_taskset_property(self):
        """Test taskset property returns string."""
        config = CPUAffinityConfig()
        # Just verify it doesn't crash and returns a string
        taskset = config.taskset
        self.assertIsInstance(taskset, str)


if __name__ == "__main__":
    unittest.main()
