#!/usr/bin/env python3
"""
Test script for OpTracer - traces DoubleLinear model forward and backward ops.

Usage:
    python test_op_tracer.py
"""

import sys
import os

# Add pytorch to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch._subclasses.op_tracer import OpTracer, trace_model, TracedOp
from torch._subclasses.fake_tensor import FakeTensorMode


# Import DoubleLinear from common_fsdp
from torch.testing._internal.common_fsdp import DoubleLinear


def test_double_linear_tracing():
    """Test tracing DoubleLinear model with forward and backward passes."""
    print("=" * 60)
    print("Testing OpTracer with DoubleLinear")
    print("=" * 60)

    # Create model
    dim = 16
    batch_size = 4
    model = DoubleLinear(dim=dim, use_second_linear=True)

    # Create input tensor
    x = torch.randn(batch_size, dim, requires_grad=True)

    print(f"\nModel: DoubleLinear(dim={dim})")
    print(f"Input shape: {x.shape}")
    print(f"Model structure:")
    print(f"  - lin1: Linear({dim}, {dim})")
    print(f"  - lin2: Linear({dim}, {dim})")
    print(f"  - relu: ReLU()")
    print()

    # Trace the model
    tracer = trace_model(model, (x,), backward=True)

    # Print summary
    print(tracer.summary())

    print("\n" + "=" * 60)
    print("Forward Operations (detailed):")
    print("=" * 60)
    for i, op in enumerate(tracer.get_forward_ops()):
        print(f"{i:3d}. {op.op_name}")
        print(f"     Inputs:  {op.input_shapes} ({op.input_dtypes})")
        print(f"     Outputs: {op.output_shapes} ({op.output_dtypes})")

    print("\n" + "=" * 60)
    print("Backward Operations (detailed):")
    print("=" * 60)
    for i, op in enumerate(tracer.get_backward_ops()):
        print(f"{i:3d}. {op.op_name}")
        print(f"     Inputs:  {op.input_shapes} ({op.input_dtypes})")
        print(f"     Outputs: {op.output_shapes} ({op.output_dtypes})")

    return tracer


def test_manual_tracing():
    """Test manual tracing with explicit control over the tracing context."""
    print("\n" + "=" * 60)
    print("Testing Manual Tracing")
    print("=" * 60)

    dim = 8
    model = DoubleLinear(dim=dim, use_second_linear=False)  # Single output

    fake_mode = FakeTensorMode(allow_non_fake_inputs=False)

    with fake_mode:
        # Convert model to fake tensors
        from torch._subclasses.op_tracer import _convert_module_to_fake
        fake_model = _convert_module_to_fake(model, fake_mode)

        # Create fake input
        x = torch.randn(2, dim, requires_grad=True)
        fake_x = fake_mode.from_tensor(x)
        fake_x.requires_grad_(True)

        # Create tracer with aten filter
        tracer = OpTracer(filter_namespaces=["aten"])

        with tracer:
            # Forward
            print("\nRunning forward pass...")
            output = fake_model(fake_x)
            print(f"Forward ops recorded: {len(tracer.get_forward_ops())}")

            # Backward
            print("Running backward pass...")
            loss = output.sum()
            with tracer.mark_backward():
                loss.backward()
            print(f"Backward ops recorded: {len(tracer.get_backward_ops())}")

    print(f"\nTotal ops: {len(tracer.trace)}")
    print("\nAll operations:")
    for op in tracer.trace:
        print(f"  {op}")

    return tracer


def test_op_statistics():
    """Analyze operation statistics from tracing."""
    print("\n" + "=" * 60)
    print("Operation Statistics")
    print("=" * 60)

    dim = 32
    model = DoubleLinear(dim=dim, use_second_linear=True)
    x = torch.randn(8, dim, requires_grad=True)

    tracer = trace_model(model, (x,), backward=True, filter_namespaces=["aten"])

    # Count operations by type
    op_counts: dict[str, int] = {}
    for op in tracer.trace:
        name = op.op_name
        op_counts[name] = op_counts.get(name, 0) + 1

    print("\nOperation counts:")
    for name, count in sorted(op_counts.items(), key=lambda x: -x[1]):
        print(f"  {name}: {count}")

    # Separate forward/backward counts
    fwd_counts: dict[str, int] = {}
    bwd_counts: dict[str, int] = {}
    for op in tracer.trace:
        name = op.op_name
        if op.is_backward:
            bwd_counts[name] = bwd_counts.get(name, 0) + 1
        else:
            fwd_counts[name] = fwd_counts.get(name, 0) + 1

    print("\nForward operation counts:")
    for name, count in sorted(fwd_counts.items(), key=lambda x: -x[1]):
        print(f"  {name}: {count}")

    print("\nBackward operation counts:")
    for name, count in sorted(bwd_counts.items(), key=lambda x: -x[1]):
        print(f"  {name}: {count}")

    return tracer


def main():
    print("OpTracer Test Suite")
    print("=" * 60)
    print()

    # Run tests
    test_double_linear_tracing()
    test_manual_tracing()
    test_op_statistics()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
