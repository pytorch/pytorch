#!/usr/bin/env python3
"""
Test script for OpTracer - traces Transformer model and verifies with real execution.

Usage:
    python test_op_tracer.py
"""

import os
import sys


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch._subclasses.op_tracer import OpTracer
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    Transformer,
)


def test_forward_backward_execution():
    """Execute forward and backward ops with optimizer, compare loss for 10 steps."""
    print("=" * 60)
    print("Training Loop: Traced vs Reference (10 steps)")
    print("=" * 60)

    torch.manual_seed(123)

    args = ModelArgs(
        n_layers=1,
        vocab_size=32,
        max_seq_len=16,
        dim=32,
        n_heads=4,
        dropout_p=0.0,
        weight_tying=False,
        checkpoint_activations=False,
    )

    batch_size = 2
    seq_len = 4
    num_steps = 10
    lr = 0.01

    # Create two models with identical initial weights
    torch.manual_seed(456)
    model_ref = Transformer(args)
    torch.manual_seed(456)
    model_traced = Transformer(args)

    # Create optimizers
    optimizer_ref = torch.optim.SGD(model_ref.parameters(), lr=lr)
    optimizer_traced = torch.optim.SGD(model_traced.parameters(), lr=lr)

    print(f"\nTraining for {num_steps} steps with lr={lr}")
    print("-" * 60)
    print(
        f"{'Step':>4} | {'Ref Loss':>12} | {'Traced Loss':>12} | {'Diff':>12} | {'Match':>5}"
    )
    print("-" * 60)

    all_match = True

    for step in range(num_steps):
        # Generate same input for both models
        torch.manual_seed(1000 + step)
        tokens = torch.randint(0, args.vocab_size, (batch_size, seq_len))

        # === Reference model (no tracing) ===
        optimizer_ref.zero_grad()
        output_ref = model_ref(tokens)
        loss_ref = output_ref.sum()
        loss_ref.backward()
        optimizer_ref.step()

        # === Traced model ===
        optimizer_traced.zero_grad()

        # Trace forward and backward
        tracer = OpTracer(filter_namespaces=["aten"])
        with tracer:
            output_traced = model_traced(tokens)
            loss_traced = output_traced.sum()

            with tracer.mark_backward():
                loss_traced.backward()

        # Optimizer step (not traced)
        optimizer_traced.step()

        # Compare losses
        diff = abs(loss_ref.item() - loss_traced.item())
        match = diff < 1e-5
        if not match:
            all_match = False

        print(
            f"{step:>4} | {loss_ref.item():>12.6f} | {loss_traced.item():>12.6f} | "
            f"{diff:>12.2e} | {'✓' if match else '✗':>5}"
        )

    print("-" * 60)

    # Final comparison of model weights
    print("\nFinal Weight Comparison:")
    print("-" * 40)
    max_weight_diff = 0.0
    for (name_ref, param_ref), (name_traced, param_traced) in zip(
        model_ref.named_parameters(), model_traced.named_parameters()
    ):
        diff = (param_ref - param_traced).abs().max().item()
        max_weight_diff = max(max_weight_diff, diff)

    print(f"Max weight difference: {max_weight_diff:.2e}")
    if max_weight_diff < 1e-5:
        print("✓ Weights match!")
    else:
        print("✗ Weights differ!")

    # Summary
    print("\n" + "=" * 60)
    if all_match and max_weight_diff < 1e-5:
        print("✓ All losses and weights match across 10 training steps!")
    else:
        print("✗ Some differences detected")
    print("=" * 60)

    return tracer


if __name__ == "__main__":
    test_forward_backward_execution()
