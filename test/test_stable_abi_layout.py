#!/usr/bin/env python3

import os
import subprocess
import tempfile
from pathlib import Path

import torch
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


def _get_torch_root_dir():
    """Get the root directory of PyTorch source"""
    # Get the directory containing torch module
    torch_dir = Path(torch.__file__).parent
    # Go up to find the root (where setup.py should be)
    current = torch_dir.parent
    while current != current.parent:
        if (current / "setup.py").exists() or (current / "pyproject.toml").exists():
            return str(current)
        current = current.parent
    # Fallback: assume test file is in pytorch/test/
    return str(Path(__file__).parent.parent)


def _create_test_file(version_define: str, output_file: str):
    """Create a test C++ file that includes the dummy.h with specific version"""
    content = f"""#include <iostream>
{version_define}
#include <torch/headeronly/dummy.h>

int main() {{
    // Force instantiation of the Dummy struct to ensure layout is generated
    dummy_types::Dummy d(42);
    std::cout << "Dummy created with id: " << d.get_id() << std::endl;
    return 0;
}}
"""
    with open(output_file, "w") as f:
        f.write(content)


def _compile_and_extract_layout(
    test_file: str, version_name: str, pytorch_root: str, temp_dir: str
) -> tuple[bool, str]:
    """Compile the test file and extract layout information"""
    layout_file = os.path.join(temp_dir, f"layout_{version_name}.txt")

    # Try to get system include paths dynamically from clang
    include_paths = []
    try:
        result = subprocess.run(
            ["clang++", "-E", "-v", "-x", "c++", "-"],
            input="",
            capture_output=True,
            text=True,
        )
        in_include_section = False
        for line in result.stderr.split("\n"):
            if "#include <...> search starts here:" in line:
                in_include_section = True
                continue
            elif "End of search list." in line:
                break
            elif in_include_section and line.strip().startswith("/"):
                # Only add essential system paths
                path = line.strip()
                if any(
                    essential in path
                    for essential in ["/include/c++", "/usr/include", "clang"]
                ):
                    include_paths.append(f"-I{path}")
    except Exception:
        # Minimal fallback - just the most essential paths
        include_paths = [
            "-I/usr/include/c++",  # Standard C++ headers
            "-I/usr/include",  # System headers
        ]

    # Use clang -cc1 with reduced include paths
    cmd = [
        "clang",
        "-cc1",
        "-fdump-record-layouts",
        "-emit-llvm",
        f"-I{pytorch_root}",  # PyTorch headers
        *include_paths,
        "-std=c++17",
        test_file,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        combined_output = result.stdout + "\n" + result.stderr
        with open(layout_file, "w") as f:
            f.write(combined_output)

        return result.returncode == 0, layout_file

    except Exception:
        return False, layout_file


def _extract_layout_info(layout_file: str) -> str:
    """Extract relevant layout information from compiler output"""
    try:
        with open(layout_file, "r") as f:
            content = f.read()

        # Look for struct layout information
        lines = content.split("\n")
        layout_lines = []
        capturing = False

        for line in lines:
            if "struct dummy_types::" in line:
                capturing = True
                layout_lines.append(line)
            elif capturing:
                if line.strip() and not line.startswith(
                    "  "
                ):  # End of struct definition
                    break
                layout_lines.append(line)

        return "\n".join(layout_lines) if layout_lines else ""

    except Exception:
        return ""


class TestStableABILayout(TestCase):
    """Test struct layout changes in PyTorch Stable ABI."""

    def setUp(self):
        super().setUp()
        self.pytorch_root = _get_torch_root_dir()

    @skipIfTorchDynamo("clang compilation not supported in dynamo")
    def test_dummy_struct_layout_changes_between_versions(self):
        """Test that Dummy struct layout changes between v2.8 and v2.9"""

        # Skip test if clang is not available
        try:
            subprocess.run(["clang", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.skipTest("clang compiler not available")

        # Create temporary directory
        with tempfile.TemporaryDirectory(prefix="stable_abi_layout_test_") as temp_dir:
            # Version definitions (targeting 2.8 and 2.9)
            versions = {
                "v2_8": "#define TORCH_TARGET_VERSION ((2ULL << 56) | (8ULL << 48))",
                "v2_9": "#define TORCH_TARGET_VERSION ((2ULL << 56) | (9ULL << 48))",
            }

            layout_files = {}
            success_count = 0

            # Create and compile test files for each version
            for version_name, version_define in versions.items():
                test_file = os.path.join(temp_dir, f"test_{version_name}.cpp")
                _create_test_file(version_define, test_file)

                success, layout_file = _compile_and_extract_layout(
                    test_file, version_name, self.pytorch_root, temp_dir
                )

                if success:
                    layout_files[version_name] = layout_file
                    success_count += 1

            # Skip test if compilation failed
            if success_count < 2:
                self.skipTest("Failed to compile test files with clang")

            # Extract layout information for each version
            layout_contents = {}
            for version_name, layout_file in layout_files.items():
                layout_info = _extract_layout_info(layout_file)
                layout_contents[version_name] = layout_info

            # Get layouts
            v2_8_layout = layout_contents.get("v2_8", "")
            v2_9_layout = layout_contents.get("v2_9", "")

            # Skip test if no layout information was extracted
            if not v2_8_layout or not v2_9_layout:
                self.skipTest("Failed to extract layout information from clang output")

            # Use assertExpectedInline to check v2.8 layout
            self.assertExpectedInline(
                v2_8_layout,
                """\
         0 | struct dummy_types::Dummy
         0 |   int32_t id
           | [sizeof=4, dsize=4, align=4,
           |  nvsize=4, nvalign=4]
""",
            )

            # This should fail in CI!!
            # Use assertExpectedInline to check v2.9 layout
            self.assertExpectedInline(
                v2_9_layout,
                """\
         0 | struct dummy_types::Dummy
         0 |   int32_t id
           | [sizeof=4, dsize=4, align=4,
           |  nvsize=4, nvalign=4]
""",
            )


if __name__ == "__main__":
    run_tests()
