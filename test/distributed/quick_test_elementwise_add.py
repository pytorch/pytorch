#!/usr/bin/env python3
"""Quick verification test for elementwise add extern library."""

import torch
import triton
import triton.language as tl
from torch._extern_triton import (
    requires_elementwise_add_lib,
    scalar_add_f32,
    scalar_add_f16,
)


@requires_elementwise_add_lib
@triton.jit
def add_kernel_f32(
    a_ptr,
    b_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)

    result = scalar_add_f32(a, b)

    tl.store(output_ptr + offsets, result, mask=mask)


@requires_elementwise_add_lib
@triton.jit
def composite_add_kernel_f32(
    a_ptr,
    b_ptr,
    c_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Test chaining: output = (a + b) + c"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = tl.load(c_ptr + offsets, mask=mask)

    temp = scalar_add_f32(a, b)
    result = scalar_add_f32(temp, c)

    tl.store(output_ptr + offsets, result, mask=mask)


def test_basic_f32():
    """Test basic float32 addition."""
    print("Test 1: Basic float32 addition...")
    size = 1024
    a = torch.randn(size, device="cuda", dtype=torch.float32)
    b = torch.randn(size, device="cuda", dtype=torch.float32)
    output = torch.empty_like(a)

    grid = lambda meta: (triton.cdiv(size, meta["BLOCK_SIZE"]),)
    add_kernel_f32[grid](a, b, output, size, BLOCK_SIZE=256)

    expected = a + b
    if torch.allclose(output, expected, rtol=1e-5, atol=1e-5):
        print(f"  ✅ PASSED (max error: {(output - expected).abs().max().item():.2e})")
        return True
    else:
        print(f"  ❌ FAILED (max error: {(output - expected).abs().max().item():.2e})")
        return False


def test_various_sizes():
    """Test various tensor sizes."""
    print("Test 2: Various tensor sizes...")
    sizes = [1, 7, 256, 1024, 4096, 65536]
    all_passed = True

    for size in sizes:
        a = torch.randn(size, device="cuda", dtype=torch.float32)
        b = torch.randn(size, device="cuda", dtype=torch.float32)
        output = torch.empty_like(a)

        grid = lambda meta: (triton.cdiv(size, meta["BLOCK_SIZE"]),)
        add_kernel_f32[grid](a, b, output, size, BLOCK_SIZE=256)

        expected = a + b
        if not torch.allclose(output, expected, rtol=1e-5, atol=1e-5):
            print(f"  ❌ FAILED for size={size}")
            all_passed = False

    if all_passed:
        print(f"  ✅ PASSED all sizes: {sizes}")
    return all_passed


def test_composite():
    """Test chaining multiple extern add operations."""
    print("Test 3: Composite kernel (a + b) + c...")
    size = 1024
    a = torch.randn(size, device="cuda", dtype=torch.float32)
    b = torch.randn(size, device="cuda", dtype=torch.float32)
    c = torch.randn(size, device="cuda", dtype=torch.float32)
    output = torch.empty_like(a)

    grid = lambda meta: (triton.cdiv(size, meta["BLOCK_SIZE"]),)
    composite_add_kernel_f32[grid](a, b, c, output, size, BLOCK_SIZE=256)

    expected = (a + b) + c
    if torch.allclose(output, expected, rtol=1e-5, atol=1e-5):
        print(f"  ✅ PASSED (max error: {(output - expected).abs().max().item():.2e})")
        return True
    else:
        print(f"  ❌ FAILED (max error: {(output - expected).abs().max().item():.2e})")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Elementwise Add Extern Library - Quick Verification Tests")
    print("=" * 60)

    # Check library
    from torch._extern_triton._elementwise_add_triton import ElementwiseAddLibFinder

    lib_path = ElementwiseAddLibFinder.find_device_library()
    print(f"Library path: {lib_path}")
    print()

    results = []
    results.append(test_basic_f32())
    results.append(test_various_sizes())
    results.append(test_composite())

    print()
    print("=" * 60)
    if all(results):
        print("All tests PASSED! ✅")
    else:
        print("Some tests FAILED! ❌")
    print("=" * 60)
