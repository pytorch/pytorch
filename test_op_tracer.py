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
from torch._subclasses.fake_tensor import FakeTensorMode
from torch._subclasses.op_tracer import (
    _convert_module_to_fake,
    create_random_inputs,
    execute_trace,
    OpTracer,
    trace_model,
)
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


def test_transformer_real_execution():
    """Execute traced operations with real inputs and verify correctness."""
    print("\n" + "=" * 60)
    print("Executing Transformer with Real Inputs")
    print("=" * 60)

    # Create model config
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

    # Create input tokens
    batch_size = 2
    seq_len = 8
    torch.manual_seed(42)
    tokens = torch.randint(0, args.vocab_size, (batch_size, seq_len))

    print(f"\nInput tokens shape: {tokens.shape}")
    print(f"Input tokens:\n{tokens}")

    # Step 1: Trace with FakeTensors (use fresh model)
    print("\n" + "-" * 40)
    print("Step 1: Tracing with FakeTensors")
    print("-" * 40)
    model_for_tracing = Transformer(args)
    tracer = trace_model(
        model_for_tracing, (tokens,), backward=True, filter_namespaces=["aten"]
    )
    print(f"Traced {len(tracer.trace)} operations")
    print(f"  Forward: {len(tracer.get_forward_ops())}")
    print(f"  Backward: {len(tracer.get_backward_ops())}")

    # Step 2: Execute with real tensors and trace simultaneously (fresh model)
    print("\n" + "-" * 40)
    print("Step 2: Real Execution with Tracing")
    print("-" * 40)

    model = Transformer(args)

    # Create a tracer for real execution
    real_tracer = OpTracer(filter_namespaces=["aten"])

    with real_tracer:
        # Forward pass
        output = model(tokens)
        print(f"Forward output shape: {output.shape}")
        print(f"Forward output sample: {output[0, 0, :5]}")

        # Backward pass
        loss = output.sum()
        print(f"Loss: {loss.item():.4f}")

        with real_tracer.mark_backward():
            loss.backward()

    print(f"\nReal execution traced {len(real_tracer.trace)} operations")
    print(f"  Forward: {len(real_tracer.get_forward_ops())}")
    print(f"  Backward: {len(real_tracer.get_backward_ops())}")

    # Step 3: Compare traces
    print("\n" + "-" * 40)
    print("Step 3: Comparing Fake vs Real Traces")
    print("-" * 40)

    fake_ops = [op.op_name for op in tracer.trace]
    real_ops = [op.op_name for op in real_tracer.trace]

    print(f"Fake trace ops: {len(fake_ops)}")
    print(f"Real trace ops: {len(real_ops)}")

    # Compare operation sequences
    if fake_ops == real_ops:
        print("✓ Operation sequences match exactly!")
    else:
        print("✗ Operation sequences differ")
        # Find differences
        min_len = min(len(fake_ops), len(real_ops))
        diffs = 0
        for i in range(min_len):
            if fake_ops[i] != real_ops[i]:
                if diffs < 10:
                    print(f"  Diff at {i}: fake={fake_ops[i]}, real={real_ops[i]}")
                diffs += 1
        if diffs > 10:
            print(f"  ... and {diffs - 10} more differences")

    # Step 4: Verify gradients exist
    print("\n" + "-" * 40)
    print("Step 4: Verifying Gradients")
    print("-" * 40)

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"  {name}: grad_norm={grad_norm:.6f}")
        else:
            print(f"  {name}: no gradient")

    return real_tracer


def test_trace_replay():
    """Demonstrate that traced shapes match real execution."""
    print("\n" + "=" * 60)
    print("Trace Shape Verification")
    print("=" * 60)

    args = ModelArgs(
        n_layers=1,  # Smaller for clarity
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
    tokens = torch.randint(0, args.vocab_size, (batch_size, seq_len))

    # Create model for tracing
    model_for_trace = Transformer(args)

    # Trace with fake tensors
    fake_mode = FakeTensorMode(allow_non_fake_inputs=False)

    with fake_mode:
        fake_model = _convert_module_to_fake(model_for_trace, fake_mode)
        fake_tokens = fake_mode.from_tensor(tokens)

        tracer = OpTracer(filter_namespaces=["aten"])
        with tracer:
            fake_output = fake_model(fake_tokens)

    print(f"\nTraced {len(tracer.trace)} forward operations")
    print("\nOperation shapes from trace:")
    for i, op in enumerate(tracer.trace[:15]):
        in_shapes = [list(s) for s in op.input_shapes]
        out_shapes = [list(s) for s in op.output_shapes]
        print(f"  {i:2d}. {op.op_name}")
        print(f"      in:  {in_shapes}")
        print(f"      out: {out_shapes}")

    # Execute with real tensors (fresh model)
    print("\n" + "-" * 40)
    print("Real execution verification:")
    print("-" * 40)

    model_for_real = Transformer(args)
    real_output = model_for_real(tokens)
    print(f"Real output shape: {list(real_output.shape)}")
    print(f"Fake output shape: {list(fake_output.shape)}")

    if list(real_output.shape) == list(fake_output.shape):
        print("✓ Output shapes match!")
    else:
        print("✗ Output shapes differ!")

    return tracer


def test_execute_trace():
    """Test executing traced aten ops with real tensors."""
    print("\n" + "=" * 60)
    print("Executing Traced Aten Ops with Real Inputs")
    print("=" * 60)

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
    tokens = torch.randint(0, args.vocab_size, (batch_size, seq_len))

    # Step 1: Trace the model
    print("\nStep 1: Tracing model with FakeTensors")
    print("-" * 40)
    model = Transformer(args)
    tracer = trace_model(model, (tokens,), backward=False, filter_namespaces=["aten"])
    forward_ops = tracer.get_forward_ops()
    print(f"Traced {len(forward_ops)} forward operations")

    # Step 2: Create random inputs for trace execution
    print("\nStep 2: Creating random inputs")
    print("-" * 40)
    input_tensors = create_random_inputs(forward_ops)
    print(f"Created {len(input_tensors)} input tensors:")
    for tid, tensor in input_tensors.items():
        print(f"  ID {tid}: shape={list(tensor.shape)}, dtype={tensor.dtype}")

    # Step 3: Execute the trace
    print("\nStep 3: Executing trace with real tensors")
    print("-" * 40)
    try:
        result_tensors = execute_trace(forward_ops, input_tensors)
        print(f"Execution completed! Produced {len(result_tensors)} tensors")

        # Show some output tensors
        print("\nSample output tensors:")
        output_ids = forward_ops[-1].output_tensor_ids if forward_ops else []
        for tid in output_ids[:3]:
            if tid in result_tensors:
                t = result_tensors[tid]
                print(f"  ID {tid}: shape={list(t.shape)}, dtype={t.dtype}")
                if t.numel() <= 10:
                    print(f"    values: {t}")
                else:
                    print(f"    sample: {t.flatten()[:5]}...")

        print("\n✓ Trace execution successful!")
    except Exception as e:
        print(f"\n✗ Trace execution failed: {e}")
        import traceback

        traceback.print_exc()

    return tracer


def test_forward_backward_execution():
    """Execute forward and backward ops with optimizer, compare loss for 10 steps."""
    print("\n" + "=" * 60)
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
    print(f"{'Step':>4} | {'Ref Loss':>12} | {'Traced Loss':>12} | {'Diff':>12} | {'Match':>5}")
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
    test_transformer_tracing()
    test_transformer_real_execution()
    test_trace_replay()
    test_execute_trace()
    test_forward_backward_execution()
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
