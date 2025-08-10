"""
Test registry for managing and selecting test suites.

This module provides the central registry that determines which test suite
to run based on the environment configuration.
"""

import logging
from typing import List, Optional

from .base import TestSuite, ConditionalTestSuite, DefaultTestSuite
from .environment import EnvironmentConfig

# Import actual test suite implementations
try:
    from test_suites.python_tests import (
        PythonTestSuite, PythonShardedTestSuite, SmokeTestSuite, 
        JITLegacyTestSuite, Numpy2TestSuite
    )
    from test_suites.inductor_tests import (
        InductorTestSuite, InductorDistributedTestSuite, InductorCppWrapperTestSuite,
        InductorHalideTestSuite, InductorTritonCpuTestSuite, InductorMicrobenchmarkTestSuite,
        InductorAOTITestSuite
    )
    from test_suites.distributed_tests import (
        DistributedTestSuite, DistributedRPCTestSuite, DistributedH100TestSuite,
        InductorDistributedTestSuite as InductorDistTestSuite
    )
    from test_suites.benchmark_tests import (
        OperatorBenchmarkTestSuite, TorchBenchTestSuite, DynamoBenchmarkTestSuite,
        CacheBenchTestSuite, VerifyCacheBenchTestSuite
    )
    from test_suites.coverage_tests import (
        CoveragePythonTestSuite, CoverageCppTestSuite, CoverageDockerTestSuite,
        CoverageDockerSingleTestSuite, CoverageDockerMultiTestSuite
    )
    from test_suites.mobile_tests import (
        MobileOptimizerTestSuite, LiteInterpreterTestSuite, MobileCodegenTestSuite,
        AndroidTestSuite, AndroidNDKTestSuite, IOSTestSuite, MobileCustomBuildTestSuite
    )
    from test_suites.specialized_tests import (
        BackwardTestSuite, XLATestSuite, ONNXTestSuite, JITLegacyTestSuite as JITLegacySpecializedTestSuite,
        Aarch64TestSuite, CrossCompileTestSuite, ROCmTestSuite, ASANTestSuite
    )
except ImportError:
    # Fallback to relative imports if direct imports fail
    from ..test_suites.python_tests import (
        PythonTestSuite, PythonShardedTestSuite, SmokeTestSuite, 
        JITLegacyTestSuite, Numpy2TestSuite
    )
    from ..test_suites.inductor_tests import (
        InductorTestSuite, InductorDistributedTestSuite, InductorCppWrapperTestSuite,
        InductorHalideTestSuite, InductorTritonCpuTestSuite, InductorMicrobenchmarkTestSuite,
        InductorAOTITestSuite
    )
    from ..test_suites.distributed_tests import (
        DistributedTestSuite, DistributedRPCTestSuite, DistributedH100TestSuite,
        InductorDistributedTestSuite as InductorDistTestSuite
    )
    from ..test_suites.benchmark_tests import (
        OperatorBenchmarkTestSuite, TorchBenchTestSuite, DynamoBenchmarkTestSuite,
        CacheBenchTestSuite, VerifyCacheBenchTestSuite
    )
    from ..test_suites.coverage_tests import (
        CoveragePythonTestSuite, CoverageCppTestSuite, CoverageDockerTestSuite,
        CoverageDockerSingleTestSuite, CoverageDockerMultiTestSuite
    )
    from ..test_suites.mobile_tests import (
        MobileOptimizerTestSuite, LiteInterpreterTestSuite, MobileCodegenTestSuite,
        AndroidTestSuite, AndroidNDKTestSuite, IOSTestSuite, MobileCustomBuildTestSuite
    )
    from ..test_suites.specialized_tests import (
        BackwardTestSuite, XLATestSuite, ONNXTestSuite, JITLegacyTestSuite as JITLegacySpecializedTestSuite,
        Aarch64TestSuite, CrossCompileTestSuite, ROCmTestSuite, ASANTestSuite
    )


class TestRegistry:
    """Registry for managing test suites and selecting the appropriate one."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.test_suites: List[TestSuite] = []
        self.default_suite = DefaultTestSuite()
        self._register_test_suites()
    
    def _register_test_suites(self) -> None:
        """Register all available test suites."""
        
        # Python test suites
        self.test_suites.append(PythonTestSuite())
        self.test_suites.append(PythonShardedTestSuite())
        self.test_suites.append(SmokeTestSuite())
        self.test_suites.append(Numpy2TestSuite())

        # Inductor test suites
        self.test_suites.append(InductorTestSuite())
        self.test_suites.append(InductorDistributedTestSuite())
        self.test_suites.append(InductorCppWrapperTestSuite())
        self.test_suites.append(InductorHalideTestSuite())
        self.test_suites.append(InductorTritonCpuTestSuite())
        self.test_suites.append(InductorMicrobenchmarkTestSuite())
        self.test_suites.append(InductorAOTITestSuite())

        # Distributed test suites
        self.test_suites.append(DistributedTestSuite())
        self.test_suites.append(DistributedRPCTestSuite())
        self.test_suites.append(DistributedH100TestSuite())
        self.test_suites.append(InductorDistTestSuite())

        # Benchmark test suites
        self.test_suites.append(OperatorBenchmarkTestSuite())
        self.test_suites.append(TorchBenchTestSuite())
        self.test_suites.append(DynamoBenchmarkTestSuite())
        self.test_suites.append(CacheBenchTestSuite())
        self.test_suites.append(VerifyCacheBenchTestSuite())

        # Coverage test suites
        self.test_suites.append(CoveragePythonTestSuite())
        self.test_suites.append(CoverageCppTestSuite())
        self.test_suites.append(CoverageDockerTestSuite())
        self.test_suites.append(CoverageDockerSingleTestSuite())
        self.test_suites.append(CoverageDockerMultiTestSuite())

        # Mobile test suites
        self.test_suites.append(MobileOptimizerTestSuite())
        self.test_suites.append(LiteInterpreterTestSuite())
        self.test_suites.append(MobileCodegenTestSuite())
        self.test_suites.append(AndroidTestSuite())
        self.test_suites.append(AndroidNDKTestSuite())
        self.test_suites.append(IOSTestSuite())
        self.test_suites.append(MobileCustomBuildTestSuite())

        # Specialized test suites
        self.test_suites.append(BackwardTestSuite())
        self.test_suites.append(XLATestSuite())
        self.test_suites.append(ONNXTestSuite())
        self.test_suites.append(JITLegacySpecializedTestSuite())
        self.test_suites.append(Aarch64TestSuite())
        self.test_suites.append(CrossCompileTestSuite())
        self.test_suites.append(ROCmTestSuite())
        self.test_suites.append(ASANTestSuite())
        
        # XLA test suite
        xla_suite = ConditionalTestSuite(
            name="xla",
            description="XLA integration tests",
            test_config_patterns=["xla"]
        )
        xla_suite.add_test_function("install_torchvision")
        xla_suite.add_test_function("build_xla")
        xla_suite.add_test_function("test_xla")
        self.test_suites.append(xla_suite)
        
        # ExecutorTorch test suite
        executorch_suite = ConditionalTestSuite(
            name="executorch",
            description="ExecutorTorch tests",
            test_config_patterns=["executorch"]
        )
        executorch_suite.add_test_function("test_executorch")
        self.test_suites.append(executorch_suite)
        
        # JIT Legacy test suite
        jit_legacy_suite = ConditionalTestSuite(
            name="jit_legacy",
            description="Legacy JIT tests",
            test_config_patterns=["jit_legacy"]
        )
        jit_legacy_suite.add_test_function("test_python_legacy_jit")
        self.test_suites.append(jit_legacy_suite)
        
        # LibTorch test suite
        libtorch_suite = ConditionalTestSuite(
            name="libtorch",
            description="LibTorch C++ tests",
            build_env_patterns=["libtorch"]
        )
        libtorch_suite.add_test_function("test_libtorch_cpp")
        self.test_suites.append(libtorch_suite)
        
        # Distributed test suite
        distributed_suite = ConditionalTestSuite(
            name="distributed",
            description="Distributed training tests",
            test_config_patterns=["distributed"]
        )
        distributed_suite.add_test_function("test_distributed")
        distributed_suite.add_test_function("test_rpc")
        self.test_suites.append(distributed_suite)
        
        # Operator benchmark test suite
        operator_benchmark_suite = ConditionalTestSuite(
            name="operator_benchmark",
            description="Operator benchmarking tests",
            test_config_patterns=["operator_benchmark"]
        )
        operator_benchmark_suite.add_test_function("test_operator_benchmark")
        self.test_suites.append(operator_benchmark_suite)
        
        # Inductor distributed test suite
        inductor_distributed_suite = ConditionalTestSuite(
            name="inductor_distributed",
            description="Inductor distributed tests",
            test_config_patterns=["inductor_distributed"]
        )
        inductor_distributed_suite.add_test_function("test_inductor_distributed")
        self.test_suites.append(inductor_distributed_suite)
        
        # Inductor test suite
        inductor_suite = ConditionalTestSuite(
            name="inductor",
            description="Inductor compiler tests",
            test_config_patterns=["inductor"],
            exclude_patterns=["perf", "inductor_distributed", "inductor_cpp_wrapper"]
        )
        inductor_suite.add_test_function("test_inductor_shard")
        self.test_suites.append(inductor_suite)
        
        # Inductor C++ wrapper test suite
        inductor_cpp_wrapper_suite = ConditionalTestSuite(
            name="inductor_cpp_wrapper",
            description="Inductor C++ wrapper tests",
            test_config_patterns=["inductor_cpp_wrapper"]
        )
        inductor_cpp_wrapper_suite.add_test_function("test_inductor_cpp_wrapper_shard")
        self.test_suites.append(inductor_cpp_wrapper_suite)
        
        # TorchBench test suite
        torchbench_suite = ConditionalTestSuite(
            name="torchbench",
            description="TorchBench performance tests",
            test_config_patterns=["torchbench"]
        )
        torchbench_suite.add_test_function("install_torchaudio")
        torchbench_suite.add_test_function("install_torchvision")
        torchbench_suite.add_test_function("install_torchao")
        torchbench_suite.add_test_function("test_torchbench_suite")
        self.test_suites.append(torchbench_suite)
        
        # Dynamo wrapped test suite
        dynamo_wrapped_suite = ConditionalTestSuite(
            name="dynamo_wrapped",
            description="Dynamo wrapped tests",
            test_config_patterns=["dynamo_wrapped"]
        )
        dynamo_wrapped_suite.add_test_function("test_dynamo_wrapped_shard")
        self.test_suites.append(dynamo_wrapped_suite)
        
        # Einops test suite
        einops_suite = ConditionalTestSuite(
            name="einops",
            description="Einops integration tests",
            test_config_patterns=["einops"]
        )
        einops_suite.add_test_function("test_einops")
        self.test_suites.append(einops_suite)
        
        # Vulkan test suite
        vulkan_suite = ConditionalTestSuite(
            name="vulkan",
            description="Vulkan backend tests",
            build_env_patterns=["vulkan"]
        )
        vulkan_suite.add_test_function("test_vulkan")
        self.test_suites.append(vulkan_suite)
        
        # Bazel test suite
        bazel_suite = ConditionalTestSuite(
            name="bazel",
            description="Bazel build tests",
            build_env_patterns=["-bazel-"]
        )
        bazel_suite.add_test_function("test_bazel")
        self.test_suites.append(bazel_suite)
        
        # Mobile test suite
        mobile_suite = ConditionalTestSuite(
            name="mobile",
            description="Mobile lightweight dispatch tests",
            build_env_patterns=["-mobile-lightweight-dispatch"]
        )
        mobile_suite.add_test_function("test_libtorch")
        self.test_suites.append(mobile_suite)
        
        # Docs test suite
        docs_suite = ConditionalTestSuite(
            name="docs_test",
            description="Documentation tests",
            test_config_patterns=["docs_test"]
        )
        docs_suite.add_test_function("test_docs_test")
        self.test_suites.append(docs_suite)
        
        # XPU test suite
        xpu_suite = ConditionalTestSuite(
            name="xpu",
            description="XPU backend tests",
            build_env_patterns=["xpu"]
        )
        xpu_suite.add_test_function("install_torchvision")
        xpu_suite.add_test_function("test_python")
        xpu_suite.add_test_function("test_aten")
        xpu_suite.add_test_function("test_xpu_bin")
        self.test_suites.append(xpu_suite)
        
        # Smoke test suite
        smoke_suite = ConditionalTestSuite(
            name="smoke",
            description="Smoke tests",
            test_config_patterns=["smoke"]
        )
        smoke_suite.add_test_function("test_python_smoke")
        self.test_suites.append(smoke_suite)
        
        # H100 test suites
        h100_distributed_suite = ConditionalTestSuite(
            name="h100_distributed",
            description="H100 distributed tests",
            test_config_patterns=["h100_distributed"]
        )
        h100_distributed_suite.add_test_function("test_h100_distributed")
        self.test_suites.append(h100_distributed_suite)
        
        h100_symm_mem_suite = ConditionalTestSuite(
            name="h100_symm_mem",
            description="H100 symmetric memory tests",
            test_config_patterns=["h100-symm-mem"]
        )
        h100_symm_mem_suite.add_test_function("test_h100_symm_mem")
        self.test_suites.append(h100_symm_mem_suite)
        
        h100_cutlass_suite = ConditionalTestSuite(
            name="h100_cutlass_backend",
            description="H100 CUTLASS backend tests",
            test_config_patterns=["h100_cutlass_backend"]
        )
        h100_cutlass_suite.add_test_function("test_h100_cutlass_backend")
        self.test_suites.append(h100_cutlass_suite)
    
    def get_test_suite(self, env_config: EnvironmentConfig) -> Optional[TestSuite]:
        """Get the appropriate test suite for the given environment configuration."""
        
        # Check each registered test suite in order
        for suite in self.test_suites:
            if suite.matches(env_config):
                self.logger.info(f"Selected test suite: {suite.name}")
                return suite
        
        # If no specific suite matches, return the default suite
        self.logger.info("No specific test suite matched, using default suite")
        return self.default_suite
    
    def list_test_suites(self) -> List[str]:
        """Get list of all registered test suite names."""
        return [suite.name for suite in self.test_suites] + [self.default_suite.name]
