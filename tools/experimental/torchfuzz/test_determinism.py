#!/usr/bin/env python3
"""Test to verify fuzzer produces deterministic output with same seed."""

import subprocess
import sys
from pathlib import Path


def run_fuzzer_with_seed(seed):
    """Run the fuzzer with a specific seed and return the generated code."""
    cmd = [sys.executable, "fuzzer.py", "--seed", str(seed)]

    # Clear the output directory first
    torchfuzz_dir = Path("/tmp/torchfuzz")
    if torchfuzz_dir.exists():
        for f in torchfuzz_dir.glob("*.py"):
            f.unlink()

    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=Path(__file__).parent
    )

    if result.returncode != 0:
        print(f"Fuzzer failed with return code {result.returncode}")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        return None

    # Find the generated Python file in /tmp/torchfuzz/
    py_files = list(torchfuzz_dir.glob("fuzz_*.py"))
    if not py_files:
        print("No Python files generated in /tmp/torchfuzz/")
        return None

    # Read the content of the generated file
    with open(py_files[0]) as f:
        return f.read()


def test_deterministic_output():
    """Test that the fuzzer produces identical output for the same seed."""
    seed = 115306  # Use the seed mentioned in the user's issue
    num_runs = 3

    outputs = []

    print(f"Running fuzzer {num_runs} times with seed {seed}...")

    for i in range(num_runs):
        print(f"Run {i + 1}...")
        output = run_fuzzer_with_seed(seed)
        if output is None:
            print(f"Failed to get output from run {i + 1}")
            return False
        outputs.append(output)

    # Compare all outputs
    first_output = outputs[0]
    all_identical = all(output == first_output for output in outputs[1:])

    if all_identical:
        print("✓ SUCCESS: All outputs are identical!")
        print(f"Generated code length: {len(first_output)} characters")
        return True
    else:
        print("✗ FAILURE: Outputs differ between runs!")

        # Show differences for debugging
        for i, output in enumerate(outputs[1:], 2):
            if output != first_output:
                print(f"\nDifferences between run 1 and run {i}:")

                # Simple line-by-line comparison
                lines1 = first_output.splitlines()
                lines2 = output.splitlines()

                min_lines = min(len(lines1), len(lines2))
                for line_num in range(min_lines):
                    if lines1[line_num] != lines2[line_num]:
                        print(f"Line {line_num + 1}:")
                        print(f"  Run 1: {lines1[line_num]}")
                        print(f"  Run {i}: {lines2[line_num]}")
                        break

                if len(lines1) != len(lines2):
                    print(f"Different number of lines: {len(lines1)} vs {len(lines2)}")

        return False


def main():
    """Main function to run the determinism test."""
    print("Testing fuzzer determinism...")
    print("=" * 50)

    success = test_deterministic_output()

    if success:
        print("\n🎉 Test PASSED: Fuzzer is deterministic!")
        sys.exit(0)
    else:
        print("\n❌ Test FAILED: Fuzzer is not deterministic!")
        sys.exit(1)


if __name__ == "__main__":
    main()
