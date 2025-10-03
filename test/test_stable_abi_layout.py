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
) -> str:
    """Compile the test file and extract layout information"""
    layout_file = os.path.join(temp_dir, f"layout_{version_name}.txt")

    # Get system include paths dynamically from clang
    include_paths = []
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
            # Add all system include paths to ensure we get everything
            path = line.strip()
            include_paths.append(f"-I{path}")

    # Try to get additional standard library paths using gcc as well
    # since sometimes clang might miss some paths that gcc would find
    try:
        gcc_result = subprocess.run(
            ["gcc", "-E", "-v", "-x", "c++", "-"],
            input="",
            capture_output=True,
            text=True,
        )
        gcc_in_include_section = False
        for line in gcc_result.stderr.split("\n"):
            if "#include <...> search starts here:" in line:
                gcc_in_include_section = True
                continue
            elif "End of search list." in line:
                break
            elif gcc_in_include_section and line.strip().startswith("/"):
                path = line.strip()
                gcc_flag = f"-I{path}"
                if gcc_flag not in include_paths:
                    include_paths.append(gcc_flag)
    except (subprocess.CalledProcessError, FileNotFoundError):
        # gcc might not be available, that's ok
        pass

    # Add architecture-agnostic fallback paths
    import glob

    fallback_patterns = [
        "/usr/include/c++/*",
        "/usr/include/*/c++/*",
        "/usr/include",
        "/usr/lib/llvm-*/lib/clang/*/include",
    ]

    for pattern in fallback_patterns:
        for path in glob.glob(pattern):
            if os.path.isdir(path):
                fallback_flag = f"-I{path}"
                if fallback_flag not in include_paths:
                    include_paths.append(fallback_flag)

    # Use clang++ with -Xclang to pass frontend flags
    cmd = [
        "clang++",
        "-Xclang",
        "-fdump-record-layouts",  # Pass to clang frontend
        "-c",  # Compile only, don't link
        f"-I{pytorch_root}",  # PyTorch headers
        *include_paths,
        "-std=c++17",
        test_file,
        "-o",
        "/dev/null",  # Discard object file output
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {' '.join(cmd)}")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print(f"STDOUT:\n{e.stdout}")
        if e.stderr:
            print(f"STDERR:\n{e.stderr}")
        raise

    combined_output = result.stdout + "\n" + result.stderr
    with open(layout_file, "w") as f:
        f.write(combined_output)

    return layout_file


def _extract_layout_info(layout_file: str) -> str:
    """Extract relevant layout information from compiler output"""
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
            if line.strip() and not line.startswith("  "):  # End of struct definition
                break
            layout_lines.append(line)

    return "\n".join(layout_lines) if layout_lines else ""


class TestStableABILayout(TestCase):
    """Test struct layout changes in PyTorch Stable ABI."""

    def setUp(self):
        super().setUp()
        self.pytorch_root = _get_torch_root_dir()

    @skipIfTorchDynamo("clang compilation not supported in dynamo")
    def test_dummy_struct_layout_changes_between_versions(self):
        """Test that Dummy struct layout changes between v2.8 and v2.9"""

        # Check that clang is available
        subprocess.run(["clang", "--version"], capture_output=True, check=True)

        # Create temporary directory
        with tempfile.TemporaryDirectory(prefix="stable_abi_layout_test_") as temp_dir:
            # Version definitions (targeting 2.8 and 2.9)
            versions = {
                "v2_8": "#define TORCH_TARGET_VERSION ((2ULL << 56) | (8ULL << 48))",
                "v2_9": "#define TORCH_TARGET_VERSION ((2ULL << 56) | (9ULL << 48))",
            }

            layout_files = {}

            # Create and compile test files for each version
            for version_name, version_define in versions.items():
                test_file = os.path.join(temp_dir, f"test_{version_name}.cpp")
                _create_test_file(version_define, test_file)

                layout_file = _compile_and_extract_layout(
                    test_file, version_name, self.pytorch_root, temp_dir
                )

                layout_files[version_name] = layout_file

            # Extract layout information for each version
            layout_contents = {}
            for version_name, layout_file in layout_files.items():
                layout_info = _extract_layout_info(layout_file)
                layout_contents[version_name] = layout_info

            # Get layouts
            v2_8_layout = layout_contents.get("v2_8", "")
            v2_9_layout = layout_contents.get("v2_9", "")

            # Use assertExpectedInline to check v2.8 layout
            self.assertExpectedInline(
                v2_8_layout,
                """\
         0 | struct dummy_types::Dummy
         0 |   int8_t foo
         4 |   int32_t id
           | [sizeof=8, dsize=8, align=4,
           |  nvsize=8, nvalign=4]
""",
            )

            # This should fail in CI!!
            # Use assertExpectedInline to check v2.9 layout
            self.assertExpectedInline(
                v2_9_layout,
                """\
         0 | struct dummy_types::Dummy
         0 |   int8_t foo
         4 |   int32_t id
           | [sizeof=8, dsize=8, align=4,
           |  nvsize=8, nvalign=4]
""",
            )


if __name__ == "__main__":
    run_tests()
