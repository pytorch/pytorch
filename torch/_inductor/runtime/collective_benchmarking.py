# Owner(s): ["module: inductor"]
"""
Collective operation benchmarking utilities for distributed autotuning.

This module provides specialized benchmarking for collective operations
(all_reduce, all_gather, reduce_scatter, etc.) that require synchronization
across multiple ranks during autotuning.
"""

import logging
import time
from typing import Any, Callable, Optional

import torch
import torch.distributed as dist

log = logging.getLogger(__name__)

# Default timeout for collective operations during benchmarking (seconds)
DEFAULT_COLLECTIVE_TIMEOUT = 30.0


# Mapping of collective op names to their torch.ops equivalents
COLLECTIVE_OPS = {
    "torch.ops._c10d_functional.all_reduce.default",
    "torch.ops._c10d_functional.all_reduce_.default",
    "torch.ops._c10d_functional.all_gather_into_tensor.default",
    "torch.ops._c10d_functional.reduce_scatter_tensor.default",
    "torch.ops._c10d_functional.all_to_all_single.default",
    "torch.ops._c10d_functional_autograd.all_reduce.default",
    "torch.ops._c10d_functional_autograd.all_gather_into_tensor.default",
    "torch.ops._c10d_functional_autograd.reduce_scatter_tensor.default",
    "torch.ops._c10d_functional_autograd.all_to_all_single.default",
}


def is_collective_op(op_name: str) -> bool:
    """Check if an operation is a collective operation.

    Args:
        op_name: Name of the operation to check

    Returns:
        True if the operation is a collective op, False otherwise
    """
    return op_name in COLLECTIVE_OPS


def _get_comm_op_from_name(comm_func_name: str) -> Optional[Callable]:
    """Get the actual collective operation function from its name.

    Args:
        comm_func_name: Name of the collective operation

    Returns:
        The collective operation function, or None if not found
    """
    if "all_gather_into_tensor" in comm_func_name:
        return torch.ops._c10d_functional.all_gather_into_tensor.default
    elif "reduce_scatter_tensor" in comm_func_name:
        return torch.ops._c10d_functional.reduce_scatter_tensor.default
    elif "all_reduce" in comm_func_name:
        return torch.ops._c10d_functional.all_reduce_.default
    elif "all_to_all_single" in comm_func_name:
        return torch.ops._c10d_functional.all_to_all_single.default
    else:
        log.warning("Unsupported collective op: %s", comm_func_name)
        return None


def benchmark_collective_op(
    comm_func: Callable,
    comm_func_name: str,
    input_tensors: list[torch.Tensor],
    output_tensor: Optional[torch.Tensor],
    process_group: Optional[dist.ProcessGroup] = None,
    nruns: int = 3,
    estimate: bool = False,
) -> float:
    """Benchmark a collective operation with proper cross-rank synchronization.

    This function ensures all ranks synchronize before and during benchmarking
    to get accurate timing measurements for distributed collective operations.

    Args:
        comm_func: The collective operation function to benchmark
        comm_func_name: Name of the collective operation (for logging)
        input_tensors: Input tensors for the operation
        output_tensor: Output tensor (required for some ops like all_gather)
        process_group: Process group for the collective operation
        nruns: Number of benchmark runs for averaging
        estimate: If True, use time estimator; otherwise use actual timing

    Returns:
        Average time in microseconds for the collective operation

    Raises:
        RuntimeError: If distributed is not initialized
        ValueError: If invalid collective operation name
    """
    if not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed is not initialized. "
            "Collective op benchmarking requires distributed initialization."
        )

    rank = dist.get_rank(process_group)
    device = torch.device(f"cuda:{rank}")

    # Prepare input arguments based on collective operation type
    if "all_gather_into_tensor" in comm_func_name:
        if output_tensor is None:
            raise ValueError("all_gather_into_tensor requires output_tensor, got None")
        input_args = {
            "input_tensor": input_tensors[0],
            "output_tensor": output_tensor,
        }
    elif "reduce_scatter_tensor" in comm_func_name:
        if output_tensor is None:
            raise ValueError("reduce_scatter_tensor requires output_tensor, got None")
        input_args = {"input": input_tensors[0], "output": output_tensor}
    elif "all_reduce" in comm_func_name:
        input_args = {"tensor": input_tensors[0]}
    elif "all_to_all_single" in comm_func_name:
        if output_tensor is None:
            raise ValueError("all_to_all_single requires output_tensor, got None")
        input_args = {"input": input_tensors[0], "output": output_tensor}
    else:
        raise ValueError(f"Unsupported comm func {comm_func_name}")

    # Use time estimator if requested (faster but less accurate)
    if estimate:
        try:
            with dist._time_estimator(group=process_group, device=device) as cm:
                comm_func(**input_args, group=process_group)
            comm_time = cm.estimated_time
            return comm_time
        except Exception as e:
            log.warning(
                f"Time estimator failed for {comm_func_name}: {e}. "
                f"Falling back to actual benchmarking."
            )
            estimate = False

    # Actual benchmarking with synchronization
    # Warmup run
    torch.cuda.synchronize()
    comm_func(**input_args, group=process_group)
    torch.cuda.synchronize()

    # Multiple runs with barrier synchronization
    comm_time = 0.0
    for _ in range(nruns):
        # CRITICAL: Barrier to ensure all ranks start benchmarking simultaneously
        dist.barrier(group=process_group)
        torch.cuda.synchronize()

        # Create CUDA events for accurate timing
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)

        start_evt.record()
        comm_func(**input_args, group=process_group)
        end_evt.record()

        # Wait for operation to complete
        end_evt.synchronize()

        # Get elapsed time in milliseconds, convert to microseconds
        current_run_time = start_evt.elapsed_time(end_evt)
        comm_time += current_run_time

    # Average time in microseconds (ms * 1000)
    comm_time = (comm_time / nruns) * 1000.0

    # Optional: All-reduce to get max time across ranks for conservative estimate
    # This ensures we use the worst-case timing
    if process_group is not None:
        comm_time_tensor = torch.tensor([comm_time], dtype=torch.float32, device=device)
        dist.all_reduce(comm_time_tensor, op=dist.ReduceOp.MAX, group=process_group)
        comm_time = comm_time_tensor.item()

    return comm_time


class CollectiveBenchmarker:
    """Specialized benchmarker for collective operations.

    This class handles the orchestration of benchmarking collective operations
    across multiple ranks, ensuring proper synchronization and timing.
    """

    def __init__(
        self,
        process_group: Optional[dist.ProcessGroup] = None,
        nruns: int = 3,
        estimate: bool = False,
    ):
        """Initialize the collective benchmarker.

        Args:
            process_group: Process group for collective operations
            nruns: Number of benchmark runs for averaging
            estimate: Whether to use time estimator (faster but less accurate)
        """
        self.process_group = process_group
        self.nruns = nruns
        self.estimate = estimate

        if not dist.is_initialized():
            log.warning(
                "Distributed not initialized. "
                "Collective benchmarking will fail at runtime."
            )

    def benchmark(
        self,
        comm_func: Callable,
        comm_func_name: str,
        input_tensors: list[torch.Tensor],
        output_tensor: Optional[torch.Tensor] = None,
    ) -> float:
        """Benchmark a collective operation.

        Args:
            comm_func: The collective operation function
            comm_func_name: Name of the operation
            input_tensors: Input tensors
            output_tensor: Output tensor (if required)

        Returns:
            Benchmark time in microseconds
        """
        return benchmark_collective_op(
            comm_func=comm_func,
            comm_func_name=comm_func_name,
            input_tensors=input_tensors,
            output_tensor=output_tensor,
            process_group=self.process_group,
            nruns=self.nruns,
            estimate=self.estimate,
        )

    def is_distributed_ready(self) -> bool:
        """Check if distributed environment is ready for benchmarking.

        Returns:
            True if distributed is initialized, False otherwise
        """
        return dist.is_initialized()


def sync_with_timeout(
    process_group: Optional[dist.ProcessGroup] = None,
    timeout_seconds: float = DEFAULT_COLLECTIVE_TIMEOUT,
) -> bool:
    """Attempt to synchronize all ranks with a timeout.

    This function tries to perform a barrier operation with a timeout to avoid
    hanging indefinitely if some ranks are not responsive. If the barrier
    times out, returns False to indicate that not all ranks are ready.

    Args:
        process_group: Process group for synchronization
        timeout_seconds: Maximum time to wait for synchronization (seconds)

    Returns:
        True if all ranks synchronized successfully, False if timeout occurred

    Raises:
        RuntimeError: If distributed is not initialized
    """
    if not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed is not initialized. Cannot perform synchronization."
        )

    rank = dist.get_rank(process_group)

    try:
        # Create a work handle for the barrier with timeout
        # Note: barrier() in PyTorch distributed doesn't support timeout directly,
        # so we use a workaround with all_reduce of a dummy tensor

        # Signal that this rank is ready
        ready_tensor = torch.tensor([1], dtype=torch.int32, device=f"cuda:{rank}")

        # Use a work handle with timeout
        work = dist.all_reduce(
            ready_tensor,
            op=dist.ReduceOp.SUM,
            group=process_group,
            async_op=True,
        )

        # Wait with timeout
        start_time = time.time()
        while not work.is_completed():
            if time.time() - start_time > timeout_seconds:
                log.warning(
                    f"[Rank {rank}] Sync timeout after {timeout_seconds}s. "
                    f"Not all ranks are ready for collective benchmarking."
                )
                return False
            time.sleep(0.01)  # Small sleep to avoid busy waiting

        # Verify all ranks are ready
        world_size = dist.get_world_size(process_group)
        expected_sum = world_size

        if ready_tensor.item() != expected_sum:
            log.warning(
                f"[Rank {rank}] Sync verification failed. "
                f"Expected sum {expected_sum}, got {ready_tensor.item()}"
            )
            return False

        return True

    except Exception as e:
        log.error(f"[Rank {rank}] Sync failed with exception: {e}")
        return False


def try_collective_benchmark_with_timeout(
    comm_func: Callable,
    comm_func_name: str,
    input_tensors: list[torch.Tensor],
    output_tensor: Optional[torch.Tensor],
    process_group: Optional[dist.ProcessGroup] = None,
    nruns: int = 3,
    timeout_seconds: float = DEFAULT_COLLECTIVE_TIMEOUT,
) -> Optional[float]:
    """Try to benchmark a collective operation with timeout protection.

    This function first attempts to synchronize all ranks with a timeout.
    If synchronization succeeds, proceeds with benchmarking. If synchronization
    fails or times out, returns None to indicate the benchmark should be skipped.

    Args:
        comm_func: The collective operation function to benchmark
        comm_func_name: Name of the collective operation
        input_tensors: Input tensors for the operation
        output_tensor: Output tensor (required for some ops)
        process_group: Process group for the collective operation
        nruns: Number of benchmark runs for averaging
        timeout_seconds: Maximum time to wait for synchronization

    Returns:
        Benchmark time in microseconds if successful, None if timeout or failure
    """
    # First, try to synchronize all ranks with timeout
    if not sync_with_timeout(process_group, timeout_seconds):
        rank = dist.get_rank(process_group) if dist.is_initialized() else 0
        log.warning(
            f"[Rank {rank}] Skipping benchmark for {comm_func_name} "
            f"due to sync timeout. Some ranks may not be ready."
        )
        return None

    # If sync succeeded, proceed with benchmarking
    try:
        return benchmark_collective_op(
            comm_func=comm_func,
            comm_func_name=comm_func_name,
            input_tensors=input_tensors,
            output_tensor=output_tensor,
            process_group=process_group,
            nruns=nruns,
            estimate=False,
        )
    except Exception as e:
        rank = dist.get_rank(process_group) if dist.is_initialized() else 0
        log.warning(f"[Rank {rank}] Benchmark failed for {comm_func_name}: {e}")
        return None


def get_process_group_info(
    process_group: Optional[dist.ProcessGroup] = None,
) -> dict[str, Any]:
    """Get information about the process group.

    Args:
        process_group: Process group to query (None for default world group)

    Returns:
        Dictionary with rank, world_size, and device information
    """
    if not dist.is_initialized():
        return {
            "rank": 0,
            "world_size": 1,
            "device": "cuda:0",
            "is_initialized": False,
        }

    rank = dist.get_rank(process_group)
    world_size = dist.get_world_size(process_group)
    device = f"cuda:{rank}"

    return {
        "rank": rank,
        "world_size": world_size,
        "device": device,
        "is_initialized": True,
    }
