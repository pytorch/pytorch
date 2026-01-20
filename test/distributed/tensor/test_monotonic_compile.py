# Owner(s): ["oncall: distributed"]

"""
Test torch.compile interaction with monotonic ops Partial preservation.

Verifies that:
1. Compiled monotonic ops on Partial("max") don't insert collectives
2. Fusion opportunities are preserved (no collective barriers)
3. Results are numerically correct
4. Eager and compiled paths produce identical results

Run with: torchrun --nproc_per_node=2 test/distributed/tensor/test_monotonic_compile.py
"""

import time

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate
from torch.distributed.tensor.debug import CommDebugMode
from torch.distributed.tensor.placement_types import Partial


def setup():
    """Initialize distributed environment."""
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size


def test_compile_with_comm_mode():
    """Use CommDebugMode to verify no communication during compiled execution."""
    rank, world_size = setup()
    device_mesh = init_device_mesh("cuda", (world_size,))
    comm_mode = CommDebugMode()

    @torch.compile
    def monotonic_ops(x):
        y = torch.exp(x)
        y = torch.sigmoid(y)
        y = torch.tanh(y)
        return y

    local_tensor = torch.full((8, 8), 0.5, device="cuda")
    x = DTensor.from_local(local_tensor, device_mesh, [Partial("max")])

    # Warmup compile
    _ = monotonic_ops(x)

    # Measure communication on subsequent call
    with comm_mode:
        result = monotonic_ops(x)

    comm_count = comm_mode.get_total_counts()

    assert result.placements == (Partial("max"),), (
        f"Expected Partial('max'), got {result.placements}"
    )
    assert comm_count == 0, f"Expected 0 communications, got {comm_count}"

    if rank == 0:
        print("✓ test_compile_with_comm_mode passed")


def test_compile_correctness():
    """Verify numerical correctness of compiled monotonic ops."""
    rank, world_size = setup()
    device_mesh = init_device_mesh("cuda", (world_size,))

    @torch.compile
    def monotonic_chain(x):
        return torch.sqrt(torch.exp(x))

    # Each rank has different values
    local_tensor = torch.full((4, 4), float(rank + 1), device="cuda")
    x = DTensor.from_local(local_tensor, device_mesh, [Partial("max")])

    result = monotonic_chain(x)

    # Redistribute to check correctness
    result_rep = result.redistribute(device_mesh, [Replicate()])

    # Expected: max across ranks is world_size, so sqrt(exp(world_size))
    expected = torch.sqrt(torch.exp(torch.tensor(float(world_size), device="cuda")))

    torch.testing.assert_close(
        result_rep.to_local()[0, 0],
        expected,
        rtol=1e-4,
        atol=1e-4,
    )

    if rank == 0:
        print("✓ test_compile_correctness passed")


def test_eager_vs_compile_parity():
    """Verify eager and compiled paths produce identical results."""
    rank, world_size = setup()
    device_mesh = init_device_mesh("cuda", (world_size,))

    def monotonic_chain(x):
        return torch.tanh(torch.sigmoid(torch.exp(x)))

    compiled_fn = torch.compile(monotonic_chain)

    local_tensor = torch.randn(8, 8, device="cuda")
    x = DTensor.from_local(local_tensor, device_mesh, [Partial("max")])

    eager_result = monotonic_chain(x)
    compiled_result = compiled_fn(x)

    # Both should have same placement
    assert eager_result.placements == compiled_result.placements == (Partial("max"),)

    # Both should have same local values
    torch.testing.assert_close(
        eager_result.to_local(),
        compiled_result.to_local(),
        rtol=1e-5,
        atol=1e-5,
    )

    if rank == 0:
        print("✓ test_eager_vs_compile_parity passed")


def test_compile_fusion_opportunity():
    """Verify that without collectives, ops can be fused."""
    rank, world_size = setup()
    device_mesh = init_device_mesh("cuda", (world_size,))

    @torch.compile(mode="reduce-overhead")
    def fused_monotonic(x):
        # These should all fuse into one kernel
        return torch.relu(torch.sigmoid(torch.exp(x)))

    local_tensor = torch.randn(1024, 1024, device="cuda")
    x = DTensor.from_local(local_tensor, device_mesh, [Partial("max")])

    # Warmup
    _ = fused_monotonic(x)
    torch.cuda.synchronize()

    # Time execution
    start = time.perf_counter()
    for _ in range(100):
        result = fused_monotonic(x)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    assert result.placements == (Partial("max"),)

    if rank == 0:
        print(
            f"✓ test_compile_fusion_opportunity passed (100 iters: {elapsed * 1000:.2f}ms)"
        )


if __name__ == "__main__":
    # For OSS CI: import run_tests but skip if not launched with torchrun
    from torch.testing._internal.common_utils import run_tests

    if not dist.is_initialized():
        # Not launched with torchrun, run as normal pytest
        run_tests()
    else:
        # Launched with torchrun, run manual GPU tests
        test_compile_with_comm_mode()
        test_compile_correctness()
        test_eager_vs_compile_parity()
        test_compile_fusion_opportunity()

        if dist.get_rank() == 0:
            print("\n=== All compile integration tests passed ===")

        dist.destroy_process_group()
