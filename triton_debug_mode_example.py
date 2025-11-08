"""
Example demonstrating how to use DebugMode to record Triton kernel calls.

This example shows how DebugMode can now capture Triton kernels generated
by torch.compile/Inductor at runtime.

IMPORTANT: You must enable config.debug=True to record Triton kernels.
This is disabled by default to avoid performance overhead.
"""

import torch
from torch.utils._debug_mode import DebugMode


def example_function(x, y):
    """A simple function that will be compiled and use Triton kernels."""
    z = x + y
    w = z * 2
    return w.sum()


def main():
    # STEP 1: Enable debug mode to record Triton kernels
    # This uses the general inductor debug flag
    import torch._inductor.config as inductor_config
    inductor_config.debug = True

    # Create some example tensors
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(1024, 1024, device=device)
    y = torch.randn(1024, 1024, device=device)

    # Compile the function to generate Triton kernels
    compiled_fn = torch.compile(example_function)

    # Run with DebugMode to record all operations including Triton kernels
    with DebugMode() as debug:
        result = compiled_fn(x, y)

    # Print the debug trace
    print("=" * 80)
    print("Debug Trace (showing Triton kernel calls):")
    print("=" * 80)
    print(debug.debug_string())
    print("=" * 80)

    # You can also iterate through the operators to filter for Triton kernels
    print("\nTriton Kernels Only:")
    print("=" * 80)
    from torch.utils._debug_mode import _TritonKernelCall

    for op in debug.operators:
        if isinstance(op, _TritonKernelCall):
            print(f"Kernel: {op.kernel_name}")
            print(f"  Args: {op.arg_names}")
            print(f"  Full call: {op.render([])}")
            if op.record:
                print(f"  Record: {op.record}")
            if op.log:
                print(f"  Log: {op.log}")
            print()


if __name__ == "__main__":
    main()
