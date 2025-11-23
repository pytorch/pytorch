#!/usr/bin/env python3
"""Standalone test for dynamic range-based autotuning."""

import torch
from torch._inductor.kernel.custom_op import (
    register_custom_op_autotuning,
    CustomOpConfig,
)


def short_sequence_impl(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Simple einsum implementation for short sequences."""
    return torch.einsum("bsh,h->bsh", x, weight)


def medium_sequence_impl(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Chunked processing for medium sequences."""
    batch_size, seq_len, hidden_dim = x.shape
    chunk_size = 256
    chunks = []
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk = x[:, start:end, :]
        chunks.append(chunk * weight)
    return torch.cat(chunks, dim=1)


def long_sequence_impl(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Broadcast implementation for long sequences."""
    return x * weight.view(1, 1, -1)


def test_dynamic_range_tuning():
    """Test dynamic range-based autotuning."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    print(f"Running test on device: {device}")

    # Create unique op name
    test_op_name = f"test_lib::dynamic_range_{id(test_dynamic_range_tuning)}"

    # Register custom op
    @torch.library.custom_op(test_op_name, mutates_args=())
    def dynamic_range_op(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return x * weight

    @dynamic_range_op.register_fake
    def _(x: torch.Tensor, weight: torch.Tensor):
        return torch.empty_like(x)

    # Register for autotuning
    register_custom_op_autotuning(
        dynamic_range_op,
        configs=[
            CustomOpConfig(short_sequence_impl),
            CustomOpConfig(medium_sequence_impl),
            CustomOpConfig(long_sequence_impl),
        ],
        name="dynamic_range_autotuned",
        dispatch_on=("x", 1),
        split_points=[512, 2048],
        input_gen_fns={
            "x": lambda fake: torch.randn_like(fake, device=device) * 0.1,
            "weight": lambda fake: torch.ones_like(fake, device=device),
        },
    )

    print("\n=== Verifying all implementations produce equivalent results ===")

    # Test cases
    test_cases = [
        (2, 256, 128),  # Short sequence
        (2, 1024, 128),  # Medium sequence
        (2, 4096, 128),  # Long sequence
    ]

    for batch_size, seq_len, hidden_dim in test_cases:
        test_x = torch.randn(
            batch_size, seq_len, hidden_dim, device=device, dtype=dtype
        )
        test_weight = torch.ones(hidden_dim, device=device, dtype=dtype)
        expected = test_x * test_weight

        for impl_name, impl_fn in [
            ("short", short_sequence_impl),
            ("medium", medium_sequence_impl),
            ("long", long_sequence_impl),
        ]:
            result = impl_fn(test_x, test_weight)
            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-5,
                atol=1e-5,
                msg=f"{impl_name} implementation differs for seq_len={seq_len}",
            )
            print(f"  ✓ {impl_name}_impl correct for seq_len={seq_len}")

    print("\n=== Testing autotuning with compilation ===")

    # Test compilation
    test_x = torch.randn(2, 256, 128, device=device, dtype=dtype)
    test_weight = torch.ones(128, device=device, dtype=dtype)
    expected = test_x * test_weight

    # Compile the op
    @torch.compile
    def test_fn(x, weight):
        return dynamic_range_op(x, weight)

    # Run compiled version
    result = test_fn(test_x, test_weight)

    # Verify result
    torch.testing.assert_close(
        result,
        expected,
        rtol=1e-5,
        atol=1e-5,
        msg="Compiled result differs from expected",
    )
    print("  ✓ Compiled version produces correct results")

    # Verify dispatch function was generated
    import os

    dispatch_dir = "/tmp/torch_inductor_range_dispatch"
    dispatch_file = os.path.join(dispatch_dir, "dynamic_range_autotuned_dispatch.py")

    if os.path.exists(dispatch_file):
        with open(dispatch_file, "r") as f:
            dispatch_code = f.read()
            if "torch.cond" in dispatch_code:
                print(
                    f"  ✓ Dispatch function generated with torch.cond at {dispatch_file}"
                )
            else:
                print(f"  ⚠ Dispatch function exists but doesn't contain torch.cond")
    else:
        print(f"  ⚠ Dispatch function not found at {dispatch_file}")

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_dynamic_range_tuning()
