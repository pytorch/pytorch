"""
Environment configuration detection and setup.

This module handles the detection of CI environment variables and provides
a clean interface for accessing build and test configuration.
"""

import os
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class EnvironmentConfig:
    """Configuration class that encapsulates CI environment detection."""
    
    def __init__(self):
        """Initialize environment configuration from environment variables."""
        self.logger = logging.getLogger(__name__)
        
        # Core environment variables from the original test.sh
        self.build_environment = os.environ.get('BUILD_ENVIRONMENT', '')
        self.test_config = os.environ.get('TEST_CONFIG', '')
        self.shard_number = int(os.environ.get('SHARD_NUMBER', '1'))
        self.num_test_shards = int(os.environ.get('NUM_TEST_SHARDS', '1'))
        
        # Additional environment variables
        self.pytorch_testing_device_only_for = os.environ.get('PYTORCH_TESTING_DEVICE_ONLY_FOR', '')
        self.torch_serialization_debug = os.environ.get('TORCH_SERIALIZATION_DEBUG', '1')
        self.valgrind = os.environ.get('VALGRIND', 'ON')
        
        # Paths
        self.build_dir = os.environ.get('BUILD_DIR', 'build')
        self.test_reports_dir = os.environ.get('TEST_REPORTS_DIR', 'test-reports')
        
        self._setup_environment()
    
    def _setup_environment(self) -> None:
        """Set up environment variables based on configuration."""
        # Set default environment variables
        os.environ['TORCH_SERIALIZATION_DEBUG'] = self.torch_serialization_debug
        
        # Handle CUDA/ROCm specific settings
        if self.is_cuda_build() or self.is_rocm_build():
            os.environ['PYTORCH_TESTING_DEVICE_ONLY_FOR'] = 'cuda'
        
        # Handle ASAN settings
        if self.is_asan_build():
            asan_options = [
                'detect_leaks=0',
                'symbolize=1',
                'detect_stack_use_after_return=true',
                'strict_init_order=true',
                'detect_odr_violation=1',
                'detect_container_overflow=0',
                'check_initialization_order=true',
                'debug=true'
            ]
            
            if self.is_cuda_build():
                asan_options.append('protect_shadow_gap=0')
            
            os.environ['ASAN_OPTIONS'] = ':'.join(asan_options)
        
        # Handle Valgrind settings
        if self.is_clang9_build() or self.is_xpu_build() or self.is_s390x_build():
            os.environ['VALGRIND'] = 'OFF'
            self.valgrind = 'OFF'
    
    def is_cuda_build(self) -> bool:
        """Check if this is a CUDA build."""
        return 'cuda' in self.build_environment
    
    @property
    def is_cuda(self) -> bool:
        """Property alias for is_cuda_build() for compatibility."""
        return self.is_cuda_build()
    
    def is_rocm_build(self) -> bool:
        """Check if this is a ROCm build."""
        return 'rocm' in self.build_environment
    
    @property
    def is_rocm(self) -> bool:
        """Property alias for is_rocm_build() for compatibility."""
        return self.is_rocm_build()
    
    def is_asan_build(self) -> bool:
        """Check if this is an ASAN build."""
        return 'asan' in self.build_environment
    
    def is_clang9_build(self) -> bool:
        """Check if this is a Clang 9 build."""
        return 'clang9' in self.build_environment
    
    def is_xpu_build(self) -> bool:
        """Check if this is an XPU build."""
        return 'xpu' in self.build_environment
    
    def is_s390x_build(self) -> bool:
        """Check if this is an s390x build."""
        return 's390x' in self.build_environment
    
    def is_libtorch_build(self) -> bool:
        """Check if this is a libtorch build."""
        return 'libtorch' in self.build_environment
    
    def is_bazel_build(self) -> bool:
        """Check if this is a Bazel build."""
        return '-bazel-' in self.build_environment
    
    def is_aarch64_build(self) -> bool:
        """Check if this is an aarch64 build."""
        return 'aarch64' in self.build_environment
    
    def is_vulkan_build(self) -> bool:
        """Check if this is a Vulkan build."""
        return 'vulkan' in self.build_environment
    
    def is_mobile_build(self) -> bool:
        """Check if this is a mobile build."""
        return '-mobile-lightweight-dispatch' in self.build_environment
    
    def get_test_config_patterns(self) -> Dict[str, bool]:
        """Get boolean flags for various test config patterns."""
        return {
            'numpy_2': 'numpy_2' in self.test_config,
            'backward': 'backward' in self.test_config,
            'xla': 'xla' in self.test_config,
            'executorch': 'executorch' in self.test_config,
            'jit_legacy': self.test_config == 'jit_legacy',
            'distributed': self.test_config == 'distributed',
            'operator_benchmark': 'operator_benchmark' in self.test_config,
            'inductor_distributed': 'inductor_distributed' in self.test_config,
            'inductor_halide': 'inductor-halide' in self.test_config,
            'inductor_triton_cpu': 'inductor-triton-cpu' in self.test_config,
            'inductor_micro_benchmark': 'inductor-micro-benchmark' in self.test_config,
            'huggingface': 'huggingface' in self.test_config,
            'timm': 'timm' in self.test_config,
            'cachebench': self.test_config == 'cachebench',
            'verify_cachebench': self.test_config == 'verify_cachebench',
            'torchbench': 'torchbench' in self.test_config,
            'inductor_cpp_wrapper': 'inductor_cpp_wrapper' in self.test_config,
            'inductor': 'inductor' in self.test_config,
            'einops': 'einops' in self.test_config,
            'dynamo_wrapped': 'dynamo_wrapped' in self.test_config,
            'docs_test': self.test_config == 'docs_test',
            'smoke': self.test_config == 'smoke',
            'h100_distributed': self.test_config == 'h100_distributed',
            'h100_symm_mem': self.test_config == 'h100-symm-mem',
            'h100_cutlass_backend': self.test_config == 'h100_cutlass_backend',
            'pr_time_benchmarks': 'pr_time_benchmarks' in self.test_config,
            'dynamo_eager': 'dynamo_eager' in self.test_config,
            'aot_eager': 'aot_eager' in self.test_config,
            'aot_inductor': 'aot_inductor' in self.test_config,
            'max_autotune_inductor': 'max_autotune_inductor' in self.test_config,
            'perf': 'perf' in self.test_config,
        }
    
    def __str__(self) -> str:
        """String representation of the environment configuration."""
        return f"EnvironmentConfig(build={self.build_environment}, test={self.test_config}, shard={self.shard_number}/{self.num_test_shards})"
