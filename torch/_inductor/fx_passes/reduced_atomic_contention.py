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
from torch._guards import detect_fake_mode
from torch._inductor import config
from torch._inductor.pattern_matcher import (
    Arg,
    CallFunction,
    init_once_fakemode,
    Match,
    PatternMatcherPass,
    register_graph_pattern,
)


log = logging.getLogger(__name__)
aten = torch.ops.aten
prims = torch.ops.prims

# Constants for partition configuration
MIN_PARTITIONS = 2
MAX_PARTITIONS = 128
MEMORY_BUDGET_FRACTION = 0.10

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

    lazy_init()
    num_matches = partitioned_scatter_patterns.apply(graph)

    if num_matches > 0:
        log.info(
            "partitioned_scatter_optimization: applied to %d operation(s)",
            num_matches,
        )
        graph.lint()

    return graph


@init_once_fakemode
def lazy_init():
    """Register patterns for index_put operations with accumulate=True."""
    # Pattern: index_put(input, indices, values, accumulate=True)
    register_graph_pattern(
        CallFunction(aten.index_put.default, Arg(), Arg(), Arg(), True),
        extra_check=validate_match,
        pass_dict=partitioned_scatter_patterns,  # type: ignore[arg-type]
    )(create_replacement)

    # Pattern: index_put_(input, indices, values, accumulate=True)
    register_graph_pattern(
        CallFunction(aten.index_put_.default, Arg(), Arg(), Arg(), True),
        extra_check=validate_match,
        pass_dict=partitioned_scatter_patterns,  # type: ignore[arg-type]
    )(create_replacement)


def validate_match(match: Match) -> bool:
    """Check if pattern match should be optimized."""
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

    # Only use if index_size is small enough and estimated contention is relevant
    if not (index_size < (min_index_size * 8)) and contention_ratio < 4:
        return False

    # Get optimal partitions and adjust for memory constraints
    num_partitions = _estimate_optimal_partitions(output_size, index_size)
    num_partitions = _fit_to_memory_budget(
        output_size, num_partitions, input_meta["dtype"]
    )

    # If reduced to < 2 partitions, optimization not worthwhile
    if num_partitions < MIN_PARTITIONS:
        log.debug("Skipping: insufficient memory for minimum partitions")
        return False

    # Store optimization parameters for replacement
    match._num_partitions = num_partitions  # type: ignore[attr-defined]
    match._scatter_dim = scatter_dim  # type: ignore[attr-defined]

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


def create_replacement(
    match: Match,
    input_tensor: fx.Node,
    indices: Any,
    values: fx.Node,
) -> fx.Node:
    """Replace high-contention index_put with partitioned scatter."""
    graph = match.graph
    matched_node = match.output_node()

    # Get optimization parameters (dynamically set in validate_match)
    num_partitions: int = getattr(match, "_num_partitions", MIN_PARTITIONS)
    scatter_dim: int = getattr(match, "_scatter_dim", 0)

    # Extract index node and metadata
    _, index_node = _extract_scatter_dim_and_index(indices)
    if index_node is None:
        log.warning("Could not extract index node")
        return matched_node

    input_meta = input_tensor.meta["val"]
    index_meta = index_node.meta["val"]
    values_meta = values.meta["val"]

    # Detect fake mode
    fake_mode = detect_fake_mode([input_meta, index_meta, values_meta])
    if fake_mode is None:
        log.warning("Could not detect fake mode")
        return matched_node

    with graph.inserting_before(matched_node):
        # Flatten indices if needed
        flat_index, flat_values = _flatten_indices_if_needed(
            graph, index_node, values, index_meta, values_meta, fake_mode
        )

        # Create partitioned scatter
        output = _create_partitioned_scatter(
            graph,
            input_tensor,
            indices,
            flat_index,
            flat_values,
            scatter_dim,
            num_partitions,
            input_meta,
            index_meta,
            values_meta,
            fake_mode,
        )

    # Replace original node
    matched_node.replace_all_uses_with(output)
    graph.erase_node(matched_node)

    counters["inductor"]["partitioned_scatter_applied"] += 1
    return output


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
        return MAX_PARTITIONS


def _estimate_optimal_partitions(output_size: int, index_size: int) -> int:
    """Estimate optimal number of partitions based on contention ratio."""
    # Safety check for edge cases
    if output_size == 0 or index_size == 0:
        return MIN_PARTITIONS

    contention_ratio = index_size / output_size

    # Size-aware partition limits (larger tensors = fewer partitions to limit memory)
    max_partitions_for_size = _get_max_partitions_for_size(output_size)

    # Contention-based calculation - square root scaling
    # Use max to ensure we never go below MIN_PARTITIONS for the base calculation
    base_partitions = max(MIN_PARTITIONS, int(math.sqrt(contention_ratio) * 16))

    # Round to power of 2 and apply limits
    partitions = 2 ** math.ceil(math.log2(base_partitions))
    return min(partitions, max_partitions_for_size, MAX_PARTITIONS)


def _fit_to_memory_budget(
    output_size: int, num_partitions: int, dtype: torch.dtype
) -> int:
    """
    Reduce partitions to fit memory budget if needed.

    Returns the maximum number of partitions that fit in memory budget.
    Returns input num_partitions if it fits, or a reduced count, or 0 if
    even MIN_PARTITIONS doesn't fit.
    """
    if not torch.cuda.is_available():
        return num_partitions

    try:
        _, total_memory = torch.cuda.mem_get_info()
        element_bytes = dtype.itemsize if hasattr(dtype, "itemsize") else 4
        budget = total_memory * MEMORY_BUDGET_FRACTION

        # Try reducing partitions (must be power of 2) until we fit
        current_partitions = num_partitions
        while current_partitions >= MIN_PARTITIONS:
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

        # Even MIN_PARTITIONS doesn't fit
        if log.isEnabledFor(logging.DEBUG):
            overhead = output_size * element_bytes * (MIN_PARTITIONS - 1)
            log.debug(
                "Insufficient memory even for %d partitions: %.2fGB > %.2fGB",
                MIN_PARTITIONS,
                overhead / 1e9,
                budget / 1e9,
            )
        return 0

    except Exception as e:
        log.debug(f"Memory check failed: {e}, proceeding with {num_partitions}")
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


def _set_fake_tensor_meta(
    node: fx.Node,
    shape: Any,
    dtype: torch.dtype,
    device: torch.device,
    fake_mode: Any,
) -> None:
    """Set node metadata with FakeTensor."""
    with fake_mode:
        node.meta["val"] = torch.empty(shape, dtype=dtype, device=device)


def _flatten_indices_if_needed(
    graph: fx.Graph,
    index_node: fx.Node,
    values: fx.Node,
    index_meta: Any,
    values_meta: Any,
    fake_mode: Any,
) -> tuple[fx.Node, fx.Node]:
    """Flatten multi-dimensional indices if needed."""
    if len(index_meta.shape) <= 1:
        return index_node, values

    num_operations = index_meta.numel()
    device = index_meta.device

    # Flatten index
    flat_index = graph.call_function(
        aten.reshape.default,
        args=(index_node, [num_operations]),
    )
    _set_fake_tensor_meta(
        flat_index, num_operations, index_meta.dtype, device, fake_mode
    )

    # Flatten values
    flat_values_shape = [num_operations] + list(
        values_meta.shape[len(index_meta.shape) :]
    )
    flat_values = graph.call_function(
        aten.reshape.default,
        args=(values, flat_values_shape),
    )
    _set_fake_tensor_meta(
        flat_values, flat_values_shape, values_meta.dtype, device, fake_mode
    )

    return flat_index, flat_values


def _create_partitioned_scatter(
    graph: fx.Graph,
    input_tensor: fx.Node,
    indices: Any,
    flat_index: fx.Node,
    flat_values: fx.Node,
    scatter_dim: int,
    num_partitions: int,
    input_meta: Any,
    index_meta: Any,
    values_meta: Any,
    fake_mode: Any,
) -> fx.Node:
    """Create the partitioned scatter operation."""
    dim_size = input_meta.shape[scatter_dim]
    num_operations = index_meta.numel()
    device = index_meta.device

    # Generate operation IDs
    operation_ids = graph.call_function(
        prims.iota.default,
        args=(num_operations,),
        kwargs={
            "start": 0,
            "step": 1,
            "dtype": index_meta.dtype,
            "device": device,
            "requires_grad": False,
        },
    )
    _set_fake_tensor_meta(
        operation_ids, num_operations, index_meta.dtype, device, fake_mode
    )

    # Assign to partitions using bitwise AND (equivalent to modulo for power of 2)
    partition_ids = graph.call_function(
        aten.bitwise_and.Scalar,
        args=(operation_ids, num_partitions - 1),
    )
    _set_fake_tensor_meta(
        partition_ids, num_operations, index_meta.dtype, device, fake_mode
    )

    # Create expanded buffer
    expanded_shape = list(input_meta.shape)
    expanded_shape[scatter_dim] *= num_partitions

    expanded_buffer = graph.call_function(
        aten.full.default,
        args=(expanded_shape, 0),
        kwargs={
            "dtype": values_meta.dtype,
            "layout": torch.strided,
            "device": device,
            "pin_memory": False,
        },
    )
    _set_fake_tensor_meta(
        expanded_buffer, expanded_shape, values_meta.dtype, device, fake_mode
    )
    # Tag as part of partitioned scatter optimization
    expanded_buffer.meta["partitioned_scatter_node"] = "buffer"

    # Adjust indices
    partition_offsets = graph.call_function(
        aten.mul.Tensor,
        args=(partition_ids, dim_size),
    )
    _set_fake_tensor_meta(
        partition_offsets, num_operations, index_meta.dtype, device, fake_mode
    )

    adjusted_index = graph.call_function(
        aten.add.Tensor,
        args=(flat_index, partition_offsets),
    )
    _set_fake_tensor_meta(
        adjusted_index, num_operations, index_meta.dtype, device, fake_mode
    )

    # Reconstruct indices list
    adjusted_indices = _reconstruct_indices_list(indices, adjusted_index, scatter_dim)

    # Scatter with reduced contention
    scattered_buffer = graph.call_function(
        aten.index_put.default,
        args=(expanded_buffer, adjusted_indices, flat_values, True),
    )
    _set_fake_tensor_meta(
        scattered_buffer, expanded_shape, values_meta.dtype, device, fake_mode
    )
    # Tag as part of partitioned scatter optimization
    scattered_buffer.meta["partitioned_scatter_node"] = "scatter"

    # Reshape for reduction
    reduce_shape = list(expanded_shape)
    reduce_shape[scatter_dim] = num_partitions
    reduce_shape.insert(scatter_dim + 1, dim_size)

    reshaped = graph.call_function(
        aten.view.default,
        args=(scattered_buffer, reduce_shape),
    )
    _set_fake_tensor_meta(reshaped, reduce_shape, values_meta.dtype, device, fake_mode)

    # Sum across partitions
    if values_meta.dtype in [torch.int8, torch.int16, torch.int32, torch.uint8]:
        reduced = graph.call_function(
            aten.sum.dim_IntList,
            args=(reshaped, [scatter_dim]),
            kwargs={"dtype": values_meta.dtype},
        )
    else:
        reduced = graph.call_function(
            aten.sum.dim_IntList,
            args=(reshaped, [scatter_dim]),
        )
    _set_fake_tensor_meta(
        reduced, input_meta.shape, values_meta.dtype, device, fake_mode
    )
    # Tag as part of partitioned scatter optimization
    reduced.meta["partitioned_scatter_node"] = "reduction"

    # Add to original input
    output = graph.call_function(
        aten.add.Tensor,
        args=(input_tensor, reduced),
    )
    _set_fake_tensor_meta(
        output, input_meta.shape, values_meta.dtype, device, fake_mode
    )

    # Tag the output node with optimization metadata
    # Not currently used, but may help with debugging and
    # future codegen optimizations.
    output.meta["partitioned_scatter_applied"] = True
    output.meta["partitioned_scatter_num_partitions"] = num_partitions
    output.meta["partitioned_scatter_dim"] = scatter_dim

    return output


def _reconstruct_indices_list(
    original_indices: Any,
    adjusted_index: fx.Node,
    scatter_dim: int,
) -> list[Optional[fx.Node]]:
    """
    Reconstruct indices list with adjusted index at correct position.

    Used for handling multi-dimensional indices in partitioned scatter.
    """
    if not isinstance(original_indices, (list, tuple)):
        return [adjusted_index]

    return [
        adjusted_index if i == scatter_dim else idx
        for i, idx in enumerate(original_indices)
    ]


__all__ = ["partitioned_scatter_optimization_pass"]
