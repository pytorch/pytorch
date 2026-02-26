# mypy: allow-untyped-defs
"""
Partitioned Scatter Optimization for Reduced Atomic Contention.

This pass transforms high-contention index_put operations by distributing
writes across multiple partitions, reducing atomic contention.
"""

import logging
import math
from typing import Any, Optional

import torch
import torch.fx as fx
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.pattern_matcher import (
    Arg,
    CallFunction,
    Match,
    PatternMatcherPass,
    register_graph_pattern,
)


log = logging.getLogger(__name__)
aten = torch.ops.aten
prims = torch.ops.prims


def _get_min_partitions() -> int:
    """Get minimum partitions from config."""
    return getattr(config, "partitioned_scatter_min_partitions", 2)


def _get_max_partitions() -> int:
    """Get maximum partitions from config."""
    return getattr(config, "partitioned_scatter_max_partitions", 128)


def _get_memory_budget_fraction() -> float:
    """Get memory budget fraction from config."""
    return getattr(config, "partitioned_scatter_memory_budget", 0.10)


partitioned_scatter_patterns = PatternMatcherPass(
    pass_name="partitioned_scatter_optimization"
)


def partitioned_scatter_optimization_pass(graph: fx.Graph) -> fx.Graph:
    """
    Apply partitioned scatter optimization to high-contention index_put operations.

    Reduces atomic contention by distributing writes across multiple buffers.
    Controlled by: config.partitioned_scatter_enabled
    """
    if not getattr(config, "partitioned_scatter_enabled", False):
        return graph

    num_matches = partitioned_scatter_patterns.apply(graph)

    if num_matches > 0:
        log.info(
            "partitioned_scatter_optimization: applied to %d operation(s)",
            num_matches,
        )
        graph.lint()

    return graph


def validate_match(match: Match) -> bool:
    """Check if pattern match should be optimized for scatter."""
    output_node = match.output_node()
    if not output_node or not hasattr(output_node, "args") or len(output_node.args) < 4:
        return False

    # Only apply when accumulating
    if output_node.args[3] is not True:
        log.debug("Skipping: accumulate=False")
        return False

    # Extract metadata
    input_node = output_node.args[0]
    indices_arg = output_node.args[1]

    # Validate input_node is an FX Node
    if not isinstance(input_node, fx.Node):
        return False

    scatter_dim, index_node = _extract_scatter_dim_and_index(indices_arg)
    if scatter_dim is None or index_node is None:
        return False

    # Get tensor shapes and validate
    input_meta = _get_tensor_meta(input_node)
    index_meta = _get_tensor_meta(index_node)
    if not input_meta or not index_meta:
        return False

    # Skip unsupported cases
    if isinstance(input_meta["numel"], torch.SymInt) or isinstance(
        index_meta["numel"], torch.SymInt
    ):
        log.debug("Skipping: dynamic shapes not supported")
        return False

    if input_meta["dtype"] == torch.bool or index_meta["dtype"] == torch.bool:
        log.debug("Skipping: bool dtype not supported")
        return False

    if scatter_dim >= len(input_meta["shape"]):
        log.debug("Skipping: scatter dim %d out of bounds", scatter_dim)
        return False

    # Calculate optimal partitions and check memory
    output_size = input_meta["numel"]
    index_size = index_meta["numel"]

    # Safety check (also done in _estimate_optimal_partitions)
    if output_size == 0 or index_size == 0:
        return False

    contention_ratio = index_size / output_size

    # Check minimum index size threshold
    min_index_size = getattr(config, "partitioned_scatter_min_index_size", 4096)
    if index_size < min_index_size:
        log.debug(
            "Skipping: index size %d below threshold %d", index_size, min_index_size
        )
        return False

    # Get optimal partitions and adjust for memory constraints
    num_partitions = _estimate_optimal_partitions(output_size, index_size)
    num_partitions = _fit_to_memory_budget(
        output_size, num_partitions, input_meta["dtype"]
    )

    # If reduced to < min partitions, optimization not worthwhile
    if num_partitions < _get_min_partitions():
        log.debug("Skipping: insufficient memory for minimum partitions")
        return False

    # Store optimization parameters for replacement
    match._num_partitions = num_partitions  # type: ignore[attr-defined]
    match._scatter_dim = scatter_dim  # type: ignore[attr-defined]
    match._index_node = index_node  # type: ignore[attr-defined]

    log.debug(
        "Applying optimization: %d partitions, dim=%d, contention=%.2f, "
        "output_size=%d, index_size=%d",
        num_partitions,
        scatter_dim,
        contention_ratio,
        output_size,
        index_size,
    )

    return True


@register_graph_pattern(
    CallFunction(aten.index_put.default, Arg(), Arg(), Arg(), True),
    pass_dict=partitioned_scatter_patterns,  # type: ignore[arg-type]
    extra_check=validate_match,
)
@register_graph_pattern(
    CallFunction(aten.index_put_.default, Arg(), Arg(), Arg(), True),
    pass_dict=partitioned_scatter_patterns,  # type: ignore[arg-type]
    extra_check=validate_match,
)
def create_replacement(match: Match, input_tensor, indices, values) -> None:
    """Replace high-contention index_put with partitioned scatter."""
    # Get optimization parameters (set in validate_match)
    num_partitions: int = match._num_partitions  # type: ignore[attr-defined]
    scatter_dim: int = match._scatter_dim  # type: ignore[attr-defined]
    index_node = match._index_node  # type: ignore[attr-defined]

    def repl(input_tensor, index_node, values):
        """Partitioned scatter implementation that will be traced."""
        dim_size = input_tensor.shape[scatter_dim]
        num_operations = index_node.numel()

        # Flatten if needed
        if len(index_node.shape) > 1:
            flat_index = index_node.reshape(num_operations)
            values_ndim = len(index_node.shape)
            flat_values = values.reshape(
                [num_operations] + list(values.shape[values_ndim:])
            )
        else:
            flat_index = index_node
            flat_values = values

        # Generate operation IDs and assign to partitions
        operation_ids = torch.ops.prims.iota.default(
            num_operations,
            start=0,
            step=1,
            dtype=flat_index.dtype,
            device=flat_index.device,
            requires_grad=False,
        )
        partition_ids = torch.ops.aten.bitwise_and.Scalar(
            operation_ids, num_partitions - 1
        )

        # Create expanded buffer
        expanded_shape = list(input_tensor.shape)
        expanded_shape[scatter_dim] *= num_partitions
        expanded_buffer = torch.ops.aten.full.default(
            expanded_shape,
            0,
            dtype=flat_values.dtype,
            layout=torch.strided,
            device=flat_values.device,
            pin_memory=False,
        )

        # Adjust indices for partitioning
        partition_offsets = partition_ids * dim_size
        adjusted_index = flat_index + partition_offsets

        # Reconstruct indices list for scatter
        if isinstance(indices, (list, tuple)):
            adjusted_indices = [
                adjusted_index if i == scatter_dim else idx
                for i, idx in enumerate(indices)
            ]
        else:
            adjusted_indices = [adjusted_index]

        # Scatter with reduced contention
        scattered_buffer = torch.ops.aten.index_put.default(
            expanded_buffer, adjusted_indices, flat_values, True
        )

        # Reshape for reduction
        reduce_shape = list(expanded_shape)
        reduce_shape[scatter_dim] = num_partitions
        reduce_shape.insert(scatter_dim + 1, dim_size)
        reshaped = torch.ops.aten.view.default(scattered_buffer, reduce_shape)

        # Sum across partitions (preserve dtype for int types)
        if flat_values.dtype in [torch.int8, torch.int16, torch.int32, torch.uint8]:
            reduced = torch.ops.aten.sum.dim_IntList(
                reshaped, [scatter_dim], dtype=flat_values.dtype
            )
        else:
            reduced = torch.ops.aten.sum.dim_IntList(reshaped, [scatter_dim])

        # Add to original input
        return input_tensor + reduced

    counters["inductor"]["partitioned_scatter_applied"] += 1
    # pyrefly: ignore [bad-argument-type]
    match.replace_by_example(repl, [input_tensor, index_node, values])


def _get_max_partitions_for_size(output_size: int) -> int:
    """
    Get maximum partitions based on output tensor size.

    Larger tensors use fewer partitions to limit memory overhead.
    """
    if output_size >= 100_000_000:  # >= 100M elements
        return 4
    elif output_size >= 10_000_000:  # >= 10M elements
        return 8
    elif output_size >= 1_000_000:  # >= 1M elements
        return 16
    else:  # < 1M elements
        return _get_max_partitions()


def _estimate_optimal_partitions(output_size: int, index_size: int) -> int:
    """Estimate optimal number of partitions based on contention ratio."""
    # Safety check for edge cases
    if output_size == 0 or index_size == 0:
        return _get_min_partitions()

    contention_ratio = index_size / output_size

    # Size-aware partition limits (larger tensors = fewer partitions to limit memory)
    max_partitions_for_size = _get_max_partitions_for_size(output_size)

    # Contention-based calculation - square root scaling
    # Use max to ensure we never go below min_partitions for the base calculation
    base_partitions = max(_get_min_partitions(), int(math.sqrt(contention_ratio) * 16))

    # Round to power of 2 and apply limits
    partitions = 2 ** math.ceil(math.log2(base_partitions))
    return min(partitions, max_partitions_for_size, _get_max_partitions())


def _fit_to_memory_budget(
    output_size: int, num_partitions: int, dtype: torch.dtype
) -> int:
    """
    Reduce partitions to fit memory budget if needed.

    Returns the maximum number of partitions that fit in memory budget.
    Returns input num_partitions if it fits, or a reduced count, or 0 if
    even min_partitions doesn't fit.
    """
    if not torch.cuda.is_available():
        return num_partitions

    try:
        _, total_memory = torch.cuda.mem_get_info()
        element_bytes = dtype.itemsize if hasattr(dtype, "itemsize") else 4
        budget = total_memory * _get_memory_budget_fraction()

        # Try reducing partitions (must be power of 2) until we fit
        current_partitions = num_partitions
        min_partitions = _get_min_partitions()
        while current_partitions >= min_partitions:
            overhead = output_size * element_bytes * (current_partitions - 1)

            if overhead <= budget:
                # Only format debug string if debug logging is enabled
                if current_partitions < num_partitions and log.isEnabledFor(
                    logging.DEBUG
                ):
                    log.debug(
                        "Reduced partitions from %d to %d to fit memory budget "
                        "(%.2fGB / %.2fGB)",
                        num_partitions,
                        current_partitions,
                        overhead / 1e9,
                        budget / 1e9,
                    )
                return current_partitions

            # Reduce by half (maintain power of 2)
            current_partitions //= 2

        # If min_partitions doesn't fit in memory, return 0
        if log.isEnabledFor(logging.DEBUG):
            overhead = output_size * element_bytes * (min_partitions - 1)
            log.debug(
                "Insufficient memory even for %d partitions: %.2fGB > %.2fGB",
                min_partitions,
                overhead / 1e9,
                budget / 1e9,
            )
        return 0

    except Exception:
        log.debug("Memory check failed, proceeding with %s", num_partitions)
        return num_partitions  # Assume we have enough memory if we can't check


def _extract_scatter_dim_and_index(
    indices_arg: Any,
) -> tuple[Optional[int], Optional[fx.Node]]:
    """Extract scatter dimension and index node from indices argument."""
    # Case 1: Single index → dim=0
    if not isinstance(indices_arg, (list, tuple)):
        return 0, indices_arg

    # List with Nones → position of non-None is dim
    index_node = None
    scatter_dim = None

    # Case 2 -> Find the first non-None index as the scatter dimension
    for dim, idx in enumerate(indices_arg):
        if idx is not None:
            if index_node is not None:
                # Multiple indices not supported
                return None, None
            index_node = idx
            scatter_dim = dim

    return scatter_dim, index_node


def _get_tensor_meta(node: fx.Node) -> Optional[dict[str, Any]]:
    """Extract tensor metadata from FX node."""
    if not hasattr(node, "meta") or "val" not in node.meta:
        return None

    val = node.meta["val"]
    if not isinstance(val, (torch.Tensor, type(val))) or not hasattr(val, "shape"):
        return None

    return {
        "shape": tuple(val.shape),
        "dtype": val.dtype,
        "device": val.device,
        "numel": val.numel(),
    }


__all__ = ["partitioned_scatter_optimization_pass"]
