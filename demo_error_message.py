"""
Quick demonstration of the improved error message from issue #146018.

This shows what the error looks like when someone passes the wrong number
of arguments to a Triton kernel.
"""

import sys


class MockLauncher:
    def __init__(self, arg_names):
        self.def_arg_names = arg_names


class MockAutotuner:
    def __init__(self, kernel_name="test_kernel"):
        self.fn = type('obj', (object,), {'__name__': kernel_name})()

    def _validate_launcher_args(self, launcher, args):
        """Validate that the number of arguments matches what the launcher expects."""
        expected_arg_names = getattr(launcher, 'def_arg_names', None)
        if expected_arg_names is None:
            return

        expected_count = len(expected_arg_names)
        actual_count = len(args)

        if actual_count != expected_count:
            raise TypeError(
                f"Kernel '{self.fn.__name__}' expected {expected_count} arguments "
                f"({', '.join(expected_arg_names)}) but got {actual_count}. "
                f"Please check the number of arguments passed to the kernel."
            )


def demo_error():
    """Demonstrate the error message"""
    print("=" * 80)
    print("DEMONSTRATION: Improved Error Message for Wrong Number of Arguments")
    print("=" * 80)
    print()

    # Demo 1: Correct number of args (no error)
    print("✓ Demo 1: Passing correct number of arguments")
    print("-" * 80)
    try:
        autotuner = MockAutotuner("matmul_kernel")
        launcher = MockLauncher(["A", "B", "C"])
        autotuner._validate_launcher_args(launcher, (1, 2, 3))
        print("SUCCESS: Kernel executed with 3 arguments (A, B, C)")
    except TypeError as e:
        print(f"ERROR: {e}")
    print()

    # Demo 2: Too few args
    print("✗ Demo 2: Passing too few arguments")
    print("-" * 80)
    try:
        autotuner = MockAutotuner("conv2d_kernel")
        launcher = MockLauncher(["input", "weight", "bias", "stride", "padding"])
        autotuner._validate_launcher_args(launcher, (1, 2))
        print("SUCCESS: Kernel executed")
    except TypeError as e:
        print(f"ERROR: {e}")
    print()

    # Demo 3: Too many args
    print("✗ Demo 3: Passing too many arguments")
    print("-" * 80)
    try:
        autotuner = MockAutotuner("pooling_kernel")
        launcher = MockLauncher(["input", "output"])
        autotuner._validate_launcher_args(launcher, (1, 2, 3, 4, 5, 6))
        print("SUCCESS: Kernel executed")
    except TypeError as e:
        print(f"ERROR: {e}")
    print()

    print("=" * 80)
    print("KEY IMPROVEMENTS:")
    print("=" * 80)
    print("1. Clear kernel name - shows WHICH kernel failed")
    print("2. Expected count - shows HOW MANY arguments needed")
    print("3. Argument names - shows WHICH arguments are needed")
    print("4. Actual count - shows HOW MANY arguments were provided")
    print("5. Helpful message - guides the developer to fix the issue")
    print()


if __name__ == "__main__":
    demo_error()
