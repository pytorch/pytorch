#!/usr/bin/env python3
"""Compile Slang shaders to C++ for CPU-side unit tests."""

import os
import subprocess
import sys
from pathlib import Path


SHADER_DIR = Path(__file__).parent.parent / "shaders"
OUTPUT_DIR = Path(__file__).parent.parent / "cpu_tests" / "generated"
SLANGC = os.environ.get("SLANGC", "slangc")


def find_shaders():
    """Find all .slang files recursively."""
    return sorted(SHADER_DIR.rglob("*.slang"))


def compile_to_cpp(slang_path: Path, output_path: Path):
    """Compile a Slang file to C++ for CPU testing."""
    cmd = [
        SLANGC,
        str(slang_path),
        "-target", "cpp",
        "-o", str(output_path),
        "-I", str(SHADER_DIR),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error compiling {slang_path} to C++:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        return False
    return True


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    shaders = find_shaders()
    # Skip common/ modules (compiled as part of importing shaders)
    shaders = [s for s in shaders if "common/" not in str(s)]

    success = 0
    fail = 0

    for shader in shaders:
        rel_path = shader.relative_to(SHADER_DIR)
        name = str(rel_path.with_suffix("")).replace("/", "_").replace("\\", "_")
        output = OUTPUT_DIR / f"{name}_cpu.cpp"

        print(f"Compiling {rel_path} -> {name}_cpu.cpp")
        if compile_to_cpp(shader, output):
            success += 1
        else:
            fail += 1

    print(f"\nCompiled {success} shaders to C++ ({fail} failed)")
    if fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
