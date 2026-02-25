"""
Property mixins for PyTorch test environment configuration.

This module contains property checks for build environment and test configuration.
"""


class BuildEnvironmentProperties:
    """Properties for checking build environment characteristics."""

    build_environment: str

    @property
    def is_cuda(self) -> bool:
        """Check if this is a CUDA build."""
        return "cuda" in self.build_environment

    @property
    def is_rocm(self) -> bool:
        """Check if this is a ROCm build."""
        return "rocm" in self.build_environment

    @property
    def is_xpu(self) -> bool:
        """Check if this is an XPU (Intel GPU) build."""
        return "xpu" in self.build_environment

    @property
    def is_asan(self) -> bool:
        """Check if this is an ASAN (Address Sanitizer) build."""
        return "asan" in self.build_environment

    @property
    def is_debug(self) -> bool:
        """Check if this is a debug build."""
        return "-debug" in self.build_environment

    @property
    def is_bazel(self) -> bool:
        """Check if this is a Bazel build."""
        return "-bazel-" in self.build_environment or "bazel" in self.test_config

    @property
    def is_vulkan(self) -> bool:
        """Check if this is a Vulkan build."""
        return "vulkan" in self.build_environment

    @property
    def is_libtorch(self) -> bool:
        """Check if this is a libtorch build."""
        return "libtorch" in self.build_environment

    @property
    def is_s390x(self) -> bool:
        """Check if this is an s390x build."""
        return "s390x" in self.build_environment

    @property
    def is_aarch64(self) -> bool:
        """Check if this is an aarch64 build."""
        return "aarch64" in self.build_environment


class TestConfigProperties:
    """Properties for checking test configuration characteristics."""

    test_config: str

    @property
    def is_distributed_test(self) -> bool:
        """Check if running distributed tests."""
        return self.test_config == "distributed"

    @property
    def is_inductor_test(self) -> bool:
        """Check if running inductor tests."""
        return "inductor" in self.test_config

    @property
    def is_dynamo_test(self) -> bool:
        """Check if running dynamo tests."""
        return "dynamo" in self.test_config

    @property
    def is_benchmark_test(self) -> bool:
        """Check if running benchmark tests."""
        return any(
            x in self.test_config for x in ["torchbench", "huggingface", "timm", "perf"]
        )


class DerivedProperties:
    """Derived properties computed from environment state."""

    import os

    @property
    def rocm_benchmark_prefix(self) -> str:
        """
        Get the benchmark directory prefix for ROCm builds.

        Returns 'rocm/' for ROCm builds to avoid clashes with CUDA benchmark results,
        empty string otherwise.
        """
        return "rocm/" if self.is_rocm else ""

    @property
    def tests_to_include(self) -> str:
        """Get the TESTS_TO_INCLUDE environment variable if set."""
        import os

        return os.environ.get("TESTS_TO_INCLUDE", "")

    @property
    def include_clause(self) -> str:
        """
        Get the --include clause for test filtering.

        Returns '--include <tests>' if TESTS_TO_INCLUDE is set, empty string otherwise.
        """
        tests = self.tests_to_include
        return f"--include {tests}" if tests else ""

    @property
    def pr_number(self) -> str:
        """
        Get the PR number from environment.

        Checks PR_NUMBER first, then falls back to CIRCLE_PR_NUMBER.
        """
        import os

        return os.environ.get("PR_NUMBER") or os.environ.get("CIRCLE_PR_NUMBER", "")
