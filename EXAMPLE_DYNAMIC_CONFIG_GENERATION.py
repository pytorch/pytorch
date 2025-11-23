"""
Example: Dynamic Config Generation for Custom Op Autotuning

This example demonstrates how to use the new config_generator feature
to automatically generate shape-aware configurations for custom operations.

Key features:
1. Dynamic k_split selection based on matrix dimensions
2. Using get_k_splits from PyTorch Inductor
3. Automatic optimization for different input shapes
"""

import torch
from torch._inductor.kernel.custom_op import (
    CustomOpConfig,
    extract_input_shapes,
    register_custom_op_autotuning,
)


# ============================================================================
# Example 1: Basic Decompose-K with Dynamic Config Generation
# ============================================================================


def decompose_k_matmul(
    a: torch.Tensor, b: torch.Tensor, k_splits: int = 4
) -> torch.Tensor:
    """Matrix multiplication with k-way decomposition."""
    m, k, n = a.shape[0], a.shape[1], b.shape[1]
    k_parts = k // k_splits

    # Reshape and batch matmul
    a_reshaped = torch.permute(a.reshape(m, k_splits, k_parts), (1, 0, 2))
    b_reshaped = b.reshape(k_splits, k_parts, n)
    result = torch.bmm(a_reshaped, b_reshaped)

    return torch.sum(result, dim=0)


@torch.library.custom_op("mylib::matmul_dk", mutates_args=())
def matmul_dk(a: torch.Tensor, b: torch.Tensor, k_splits: int = 4) -> torch.Tensor:
    """Custom matmul with decompose-k and ReLU."""
    return torch.relu(decompose_k_matmul(a, b, k_splits))


@matmul_dk.register_fake
def _(a: torch.Tensor, b: torch.Tensor, k_splits: int = 4):
    return torch.empty(a.shape[0], b.shape[1], device=a.device, dtype=a.dtype)


# Define dynamic config generator using get_k_splits
def generate_k_split_configs(input_nodes):
    """
    Generate k_split configs based on input matrix dimensions.

    This function uses PyTorch Inductor's get_k_splits to automatically
    determine optimal k_split candidates based on:
    - Powers of 2 (best for GPU memory alignment)
    - Multiples of 32 (good for CUDA warp size)
    - Other valid divisors
    """
    from torch._inductor.utils import get_k_splits

    shapes = extract_input_shapes(input_nodes)

    # Extract matrix dimensions from first two inputs
    m, k = shapes["arg_0"][-2:]
    _, n = shapes["arg_1"][-2:]

    # Get optimal k_splits for these dimensions
    k_splits_list = get_k_splits(m, n, k)

    print(f"[Dynamic Config] Shape (M={m}, K={k}, N={n})")
    print(f"[Dynamic Config] Generated k_splits: {k_splits_list}")

    return [CustomOpConfig(k_splits=k) for k in k_splits_list]


# Register with dynamic config generator
register_custom_op_autotuning(
    matmul_dk,
    config_generator=generate_k_split_configs,
    name="matmul_dk_autotuned",
    input_gen_fns={
        "a": lambda fake: torch.randn_like(fake, device="cuda") * 0.1,
        "b": lambda fake: torch.randn_like(fake, device="cuda") * 0.1,
    },
)


# ============================================================================
# Example 2: Custom Config Generator with Custom Logic
# ============================================================================


@torch.library.custom_op("mylib::custom_mm", mutates_args=())
def custom_mm(a: torch.Tensor, b: torch.Tensor, block_size: int = 16) -> torch.Tensor:
    """Custom matmul with configurable block size."""
    return torch.nn.functional.linear(a, b.t())


@custom_mm.register_fake
def _(a: torch.Tensor, b: torch.Tensor, block_size: int = 16):
    return torch.empty(a.shape[0], b.shape[1], device=a.device, dtype=a.dtype)


def generate_block_size_configs(input_nodes):
    """
    Generate block_size configs based on custom heuristics.

    Example logic:
    - Small matrices: use small blocks (16, 32)
    - Medium matrices: use medium blocks (64, 128)
    - Large matrices: use large blocks (256, 512)
    """
    shapes = extract_input_shapes(input_nodes)
    m, k = shapes["arg_0"][-2:]
    _, n = shapes["arg_1"][-2:]

    # Custom heuristics based on matrix size
    total_size = m * k * n

    if total_size < 1e6:  # Small
        block_sizes = [16, 32]
    elif total_size < 1e8:  # Medium
        block_sizes = [64, 128]
    else:  # Large
        block_sizes = [128, 256, 512]

    print(f"[Custom Heuristic] Size={total_size:.0e}, block_sizes={block_sizes}")

    return [CustomOpConfig(block_size=bs) for bs in block_sizes]


register_custom_op_autotuning(
    custom_mm,
    config_generator=generate_block_size_configs,
    name="custom_mm_autotuned",
    input_gen_fns={
        "a": lambda fake: torch.randn_like(fake, device="cuda"),
        "b": lambda fake: torch.randn_like(fake, device="cuda"),
    },
)


# ============================================================================
# Example 3: Multi-Parameter Config Generation
# ============================================================================


@torch.library.custom_op("mylib::fused_op", mutates_args=())
def fused_op(
    a: torch.Tensor,
    b: torch.Tensor,
    use_fast_path: bool = True,
    tile_size: int = 64,
) -> torch.Tensor:
    """Fused operation with multiple tunable parameters."""
    if use_fast_path:
        return torch.relu(a @ b)
    else:
        return torch.gelu(a @ b)


@fused_op.register_fake
def _(
    a: torch.Tensor, b: torch.Tensor, use_fast_path: bool = True, tile_size: int = 64
):
    return torch.empty(a.shape[0], b.shape[1], device=a.device, dtype=a.dtype)


def generate_multi_param_configs(input_nodes):
    """
    Generate configs with multiple parameters.

    Explores combinations of use_fast_path and tile_size.
    """
    shapes = extract_input_shapes(input_nodes)
    m, k = shapes["arg_0"][-2:]

    configs = []

    # Try both fast paths
    for use_fast in [True, False]:
        # Different tile sizes based on matrix size
        if m < 512:
            tile_sizes = [32, 64]
        else:
            tile_sizes = [64, 128, 256]

        for tile_size in tile_sizes:
            configs.append(CustomOpConfig(use_fast_path=use_fast, tile_size=tile_size))

    print(f"[Multi-param] Generated {len(configs)} configs")
    return configs


register_custom_op_autotuning(
    fused_op,
    config_generator=generate_multi_param_configs,
    name="fused_op_autotuned",
    input_gen_fns={
        "a": lambda fake: torch.randn_like(fake, device="cuda"),
        "b": lambda fake: torch.randn_like(fake, device="cuda"),
    },
)


# ============================================================================
# Demo Usage
# ============================================================================


def demo():
    """Demonstrate dynamic config generation with different shapes."""
    print("\n" + "=" * 70)
    print("Dynamic Config Generation Demo")
    print("=" * 70 + "\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Test different shapes - each will get different configs
    test_shapes = [
        (256, 4096, 512),  # Small K
        (256, 65536, 1024),  # Large K
        (128, 16384, 256),  # Medium K
    ]

    @torch.compile
    def test_model(a, b):
        return matmul_dk(a, b)

    for i, (m, k, n) in enumerate(test_shapes, 1):
        print(f"\nTest {i}: Shape (M={m}, K={k}, N={n})")
        print("-" * 70)

        # Ensure k is divisible by common k_splits
        k = ((k + 255) // 256) * 256

        a = torch.randn(m, k, device=device, dtype=torch.float16)
        b = torch.randn(k, n, device=device, dtype=torch.float16)

        # Reset dynamo for fresh compilation
        torch._dynamo.reset()

        # Compile and run
        from torch._inductor import config

        with config.patch(max_autotune=True):
            result = test_model(a, b)

        print(f"Result shape: {result.shape}")
        print(f"Result dtype: {result.dtype}")
        print()


if __name__ == "__main__":
    demo()

    print("\n" + "=" * 70)
    print("Key Takeaways:")
    print("=" * 70)
    print("""
1. config_generator receives input_nodes (list of IR Buffers)
2. Use extract_input_shapes() to get concrete shapes
3. Return list[CustomOpConfig] with appropriate parameters
4. Configs are generated dynamically per compilation
5. Different shapes automatically get different configs
6. Can reuse existing functions like get_k_splits
7. Supports multiple parameters (k_splits, block_size, etc.)
    """)
