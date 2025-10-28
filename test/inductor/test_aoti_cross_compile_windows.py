# Owner(s): ["module: inductor"]
import os
import platform
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch._inductor.config
from torch._environment import is_fbcode
from torch._inductor.test_case import TestCase
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU, requires_gpu


@dataclass
class ModelTestConfig:
    """Configuration for a model test case."""

    name: str
    model_class: type
    example_inputs: tuple[torch.Tensor, ...]
    dynamic_shapes: Optional[dict[str, Any]] = None
    inductor_configs: Optional[dict[str, Any]] = None
    rtol: float = 1e-4
    atol: float = 1e-4


class WindowsCrossCompilationTestFramework:
    """
    Framework for testing cross-compilation from Linux to Windows.

    Provides reusable logic for creating compile and load test methods.
    """

    _base_path: Optional[Path] = None
    _win_torch_libs_path: Optional[str] = None

    @classmethod
    def base_path(cls) -> Path:
        """Get or create the base path for package files."""
        if cls._base_path is None:
            cls._base_path = Path(tempfile.mkdtemp(prefix="aoti_cross_compile_"))
        return cls._base_path

    @classmethod
    def set_base_path(cls, path: Optional[Path | str] = None) -> None:
        """Set the base path for package files."""
        cls._base_path = Path(path) if path else None

    @classmethod
    def set_win_torch_libs_path(cls, path: Optional[str] = None) -> None:
        """Set the path for Windows torch libs."""
        cls._win_torch_libs_path = path

    @classmethod
    def get_package_path(cls, model_name: str) -> str:
        """Get the path for a model's .pt2 package file."""
        package_dir = cls.base_path()
        package_dir.mkdir(parents=True, exist_ok=True)
        return str(package_dir / f"{model_name}_windows.pt2")

    @classmethod
    def get_win_torch_libs_path(cls) -> str:
        """Get the path for Windows torch libs."""
        if cls._win_torch_libs_path is None:
            raise RuntimeError("Windows torch libs path not set")
        return str(cls._win_torch_libs_path)

    @classmethod
    def create_compile_test(cls, config: ModelTestConfig):
        """Create a compile test method for a model configuration."""

        def compile_test(self):
            if platform.system() == "Windows":
                raise unittest.SkipTest(
                    "This test should run on Linux for cross-compilation"
                )

            if is_fbcode():
                raise unittest.SkipTest("requires x86_64-w64-mingw32-gcc")

            self.assertTrue("WINDOWS_CUDA_HOME" in os.environ)

            with torch.no_grad():
                # Windows cross-compilation is only used for GPU.
                # AOTI for CPU should be able to work as native compilation on Windows.
                device = GPU_TYPE
                model = config.model_class().to(device=device)
                example_inputs = config.example_inputs

                # Inputs should already be on GPU_TYPE but ensure they are
                example_inputs = tuple(inp.to(device) for inp in example_inputs)

                # Export the model
                exported = torch.export.export(
                    model, example_inputs, dynamic_shapes=config.dynamic_shapes
                )

                # Prepare inductor configs
                inductor_configs = {
                    "aot_inductor.cross_target_platform": "windows",
                    "aot_inductor.precompile_headers": False,
                    "aot_inductor.package_constants_on_disk_format": "binary_blob",
                    "aot_inductor.package_constants_in_so": False,
                    "aot_inductor.aoti_shim_library_path": cls.get_win_torch_libs_path(),
                }
                if config.inductor_configs:
                    inductor_configs.update(config.inductor_configs)

                # Compile and package directly to the expected location
                package_path = cls.get_package_path(config.name)
                torch._inductor.aoti_compile_and_package(
                    exported,
                    package_path=package_path,
                    inductor_configs=inductor_configs,
                )

                self.assertTrue(
                    os.path.exists(package_path),
                    f"Package file should exist at {package_path}",
                )

        return compile_test

    @classmethod
    def create_load_test(cls, config: ModelTestConfig):
        """Create a load test method for a model configuration."""

        def load_test(self):
            if platform.system() != "Windows":
                raise unittest.SkipTest("This test should run on Windows")

            if is_fbcode():
                raise unittest.SkipTest("requires x86_64-w64-mingw32-gcc")

            if not HAS_GPU:
                raise unittest.SkipTest("Test requires GPU")

            package_path = cls.get_package_path(config.name)
            if not os.path.exists(package_path):
                raise unittest.SkipTest(
                    f"Package file not found at {package_path}. "
                    f"Run test_{config.name}_compile first."
                )

            with torch.no_grad():
                # Windows cross-compilation is only used for GPU.
                # AOTI for CPU should be able to work as native compilation on Windows.
                device = GPU_TYPE

                # Create original model for comparison
                original_model = config.model_class().to(device=device)
                example_inputs = config.example_inputs

                # Inputs should already be on GPU_TYPE but ensure they are
                example_inputs = tuple(inp.to(device) for inp in example_inputs)

                # Load the compiled package
                loaded_model = torch._inductor.aoti_load_package(package_path)

                # Test with the same inputs
                original_output = original_model(*example_inputs)
                loaded_output = loaded_model(*example_inputs)

                # Compare outputs
                torch.testing.assert_close(
                    original_output, loaded_output, rtol=config.rtol, atol=config.atol
                )

        return load_test


def auto_generate_tests(test_class):
    """
    Class decorator to automatically generate compile/load test methods
    from _define_* methods that return ModelTestConfig.
    """
    # Find all _define_* methods that return ModelTestConfig
    define_methods = {}
    for name in dir(test_class):
        if name.startswith("_define_") and callable(getattr(test_class, name)):
            method = getattr(test_class, name)
            # Try to call the method to see if it returns ModelTestConfig
            try:
                # Create a temporary instance to call the method
                temp_instance = test_class.__new__(test_class)
                result = method(temp_instance)
                if isinstance(result, ModelTestConfig):
                    define_methods[name] = result
            except Exception:
                # If method fails, skip it
                pass

    # Generate compile/load methods for each discovered definition
    for define_name, config in define_methods.items():
        model_name = define_name[8:]  # Remove '_define_' prefix

        # Create compile test method
        compile_method_name = f"test_{model_name}_compile"
        compile_method = WindowsCrossCompilationTestFramework.create_compile_test(
            config
        )
        compile_method.__name__ = compile_method_name
        compile_method.__doc__ = f"Step 1: Cross-compile {model_name} model on Linux"
        compile_method = requires_gpu()(compile_method)
        setattr(test_class, compile_method_name, compile_method)

        # Create load test method
        load_method_name = f"test_{model_name}_load"
        load_method = WindowsCrossCompilationTestFramework.create_load_test(config)
        load_method.__name__ = load_method_name
        load_method.__doc__ = f"Step 2: Load and test {model_name} model on Windows"
        load_method = requires_gpu()(load_method)
        setattr(test_class, load_method_name, load_method)

    return test_class


@auto_generate_tests
class TestAOTInductorWindowsCrossCompilation(TestCase):
    """
    Test class for AOT Inductor Windows cross-compilation.

    Define test methods that return ModelTestConfig, and the decorator
    will auto-generate compile/load test methods.
    """

    def _define_simple(self):
        """Define the Simple model and its test configuration."""

        class Simple(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(10, 16)
                self.relu = torch.nn.ReLU()
                self.fc2 = torch.nn.Linear(16, 1)
                self.sigmoid = torch.nn.Sigmoid()

            def forward(self, x):
                x = self.fc1(x)
                x = self.relu(x)
                x = self.fc2(x)
                x = self.sigmoid(x)
                return x

        return ModelTestConfig(
            name="simple",
            model_class=Simple,
            example_inputs=(torch.randn(8, 10, device=GPU_TYPE),),
            dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1, max=1024)}},
        )

    def _define_simple_cnn(self):
        """Define the SimpleCNN model and its test configuration."""

        class SimpleCNN(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 16, 3)
                self.relu = torch.nn.ReLU()
                self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
                self.fc = torch.nn.Linear(16, 10)

            def forward(self, x):
                x = self.conv1(x)
                x = self.relu(x)
                x = self.pool(x)
                x = x.flatten(1)
                x = self.fc(x)
                return x

        return ModelTestConfig(
            name="simple_cnn",
            model_class=SimpleCNN,
            example_inputs=(torch.randn(2, 3, 32, 32, device=GPU_TYPE),),
            dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1, max=16)}},
            rtol=1e-3,
            atol=1e-3,
        )

    def _define_transformer(self):
        """Define the SimpleTransformer model and its test configuration."""

        class SimpleTransformer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Linear(128, 256)
                self.attention = torch.nn.MultiheadAttention(256, 8, batch_first=True)
                self.norm1 = torch.nn.LayerNorm(256)
                self.ffn = torch.nn.Sequential(
                    torch.nn.Linear(256, 1024),
                    torch.nn.ReLU(),
                    torch.nn.Linear(1024, 256),
                )
                self.norm2 = torch.nn.LayerNorm(256)
                self.output = torch.nn.Linear(256, 10)

            def forward(self, x):
                # x shape: (batch, seq_len, input_dim)
                x = self.embedding(x)
                attn_out, _ = self.attention(x, x, x)
                x = self.norm1(x + attn_out)
                ffn_out = self.ffn(x)
                x = self.norm2(x + ffn_out)
                x = x.mean(dim=1)  # Global average pooling
                x = self.output(x)
                return x

        return ModelTestConfig(
            name="transformer",
            model_class=SimpleTransformer,
            example_inputs=(torch.randn(4, 16, 128, device=GPU_TYPE),),
            dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1, max=32)}},
            rtol=1e-3,
            atol=1e-3,
        )


if __name__ == "__main__":
    import sys

    from torch._inductor.test_case import run_tests

    # Check for --package-dir argument and remove it before unittest sees it
    package_dir = None
    win_torch_lib_dir = None
    filtered_argv = []
    i = 0
    while i < len(sys.argv):
        if sys.argv[i] == "--package-dir":
            if i + 1 < len(sys.argv):
                package_dir = sys.argv[i + 1]
                i += 2  # Skip both --package-dir and its value
            else:
                print("Error: --package-dir requires a valid directory path")
                sys.exit(1)
        elif sys.argv[i].startswith("--package-dir="):
            package_dir = sys.argv[i].split("=", 1)[1]
            i += 1
        elif sys.argv[i] == "--win-torch-lib-dir":
            if i + 1 < len(sys.argv):
                win_torch_lib_dir = sys.argv[i + 1]
                i += 2  # Skip both --win-torch-lib-dir and its value
            else:
                print("Error: --win-torch-lib-dir requires a valid directory path")
                sys.exit(1)
        elif sys.argv[i].startswith("--win-torch-lib-dir="):
            win_torch_lib_dir = sys.argv[i].split("=", 1)[1]
            i += 1
        else:
            filtered_argv.append(sys.argv[i])
            i += 1

    # Validate and set the base path for package storage
    if package_dir:
        try:
            package_path = Path(package_dir)
            package_path.mkdir(parents=True, exist_ok=True)
            # Test write access
            test_file = package_path / ".test_write"
            test_file.touch()
            test_file.unlink()
            WindowsCrossCompilationTestFramework.set_base_path(package_path)
        except Exception:
            print("Error: --package-dir requires a valid directory path")
            sys.exit(1)

    # Set Windows torch libs path if provided (only needed for compile tests)
    if win_torch_lib_dir:
        WindowsCrossCompilationTestFramework.set_win_torch_libs_path(win_torch_lib_dir)

    # Update sys.argv to remove our custom arguments
    sys.argv = filtered_argv

    if HAS_GPU:
        run_tests(needs="filelock")
