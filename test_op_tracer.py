#!/usr/bin/env python3
"""
Test script for OpTracer - traces Transformer model.

Usage:
    python test_op_tracer.py
"""

import os
import sys


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch._subclasses.op_tracer import trace_model
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    Transformer,
)


def test_transformer_tracing():
    """Test tracing Transformer model with forward and backward passes."""
    print("=" * 60)
    print("Testing OpTracer with Transformer")
    print("=" * 60)

    # Create model with small config for testing
    args = ModelArgs(
        n_layers=2,
        vocab_size=32,
        max_seq_len=16,
        dim=32,
        n_heads=4,
        dropout_p=0.0,
        weight_tying=False,
        checkpoint_activations=False,
    )
    model = Transformer(args)

    # Create input tokens
    batch_size = 2
    seq_len = 8
    tokens = torch.randint(0, args.vocab_size, (batch_size, seq_len))

    print("\nModel: Transformer")
    print(f"  n_layers: {args.n_layers}")
    print(f"  dim: {args.dim}")
    print(f"  n_heads: {args.n_heads}")
    print(f"  vocab_size: {args.vocab_size}")
    print(f"Input shape: {tokens.shape}")
    print()

    # Trace the model (filter to aten ops only for cleaner output)
    tracer = trace_model(model, (tokens,), backward=True, filter_namespaces=["aten"])

    # Print summary
    print(f"Total ops: {len(tracer.trace)}")
    print(f"Forward ops: {len(tracer.get_forward_ops())}")
    print(f"Backward ops: {len(tracer.get_backward_ops())}")

    # Count operations by type
    fwd_counts: dict[str, int] = {}
    bwd_counts: dict[str, int] = {}
    for op in tracer.trace:
        name = op.op_name
        if op.is_backward:
            bwd_counts[name] = bwd_counts.get(name, 0) + 1
        else:
            fwd_counts[name] = fwd_counts.get(name, 0) + 1

    print("\nForward operation counts (top 15):")
    for name, count in sorted(fwd_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {name}: {count}")

    print("\nBackward operation counts (top 15):")
    for name, count in sorted(bwd_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {name}: {count}")

    # Show a sample of forward and backward ops
    print("\n" + "-" * 40)
    print("Sample Forward Operations (first 20):")
    print("-" * 40)
    for op in tracer.get_forward_ops()[:20]:
        print(f"  {op}")

    print("\n" + "-" * 40)
    print("Sample Backward Operations (first 20):")
    print("-" * 40)
    for op in tracer.get_backward_ops()[:20]:
        print(f"  {op}")

    return tracer


if __name__ == "__main__":
    test_transformer_tracing()
