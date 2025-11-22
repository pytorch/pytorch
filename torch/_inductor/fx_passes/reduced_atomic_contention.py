"""
Partitioned scatter optimization for index_add/index_put operations.

This optimization reduces atomic contention by distributing scatter operations
across multiple independent buffers (partitions), then reducing the results.

Algorithm:
1. Enumerate scatter operations: operation_id = [0, 1, 2, ..., N-1]
2. Assign to partitions: partition_id = operation_id % num_partitions
3. Create expanded buffers along scatter_dim: size = num_partitions × dim_size
4. Adjust indices: adjusted_idx = original_idx + (partition_id × dim_size)
5. Perform partitioned scatter with reduced contention
6. Reduce across partitions: sum(partitions, dim=scatter_dim)

Plans to expand this file in the future with multiple implementations

"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
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

_index_scatter_patterns = PatternMatcherPass(
    pass_name="partitioned_scatter_optimization"
)


@dataclass
class TensorMetadata:
    """Metadata extracted from FX node."""
    
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    numel: int


@dataclass
class ScatterMetadata:
    """Metadata for a scatter operation."""
    
    output_size: int
    index_size: int
    output_shape: tuple[int, ...]
    index_shape: tuple[int, ...]
    scatter_dim: int
    dtype: torch.dtype
    device: torch.device
    element_bytes: int


class PartitionConfig:
    """Configuration and heuristics for partition-based scatter optimization."""

    MIN_PARTITIONS = 2
    MAX_PARTITIONS = 128
    MEMORY_BUDGET_FRACTION = 0.10

    @staticmethod
    def estimate_optimal_partitions(output_size: int, index_size: int) -> int:
        """
        Estimate optimal number of partitions based on contention ratio.
        
        Higher contention (more indices per output element) benefits from more partitions.
        Returns a power of 2 for efficient modulo via bitwise operations.
        
        Args:
            output_size: Number of elements in output tensor
            index_size: Number of scatter operations
            
        Returns:
            Power of 2 between MIN_PARTITIONS and MAX_PARTITIONS
        """
        if output_size == 0:
            return PartitionConfig.MIN_PARTITIONS
            
        contention_ratio = index_size / output_size
        
        # Heuristic: sqrt(contention) * 8, with minimum of 2
        base = 2 if contention_ratio < 0.5 else int(math.sqrt(contention_ratio) * 8)
        
        # Round up to next power of 2
        partitions = 2 ** math.ceil(math.log2(max(2, base)))
        return min(PartitionConfig.MAX_PARTITIONS, partitions)

    @staticmethod
    def calculate_memory_overhead(
        output_size: int, 
        num_partitions: int, 
        element_bytes: int
    ) -> int:
        """
        Calculate additional memory required for partitioned buffers.
        
        Args:
            output_size: Number of elements in output
            num_partitions: Number of partitions
            element_bytes: Size of each element in bytes
            
        Returns:
            Memory overhead in bytes
        """
        # We create (num_partitions - 1) extra copies of the output
        return output_size * element_bytes * (num_partitions - 1)

    @staticmethod
    def should_optimize(
        output_size: int,
        index_size: int,
        num_partitions: int,
        element_bytes: int = 4,
        available_memory: Optional[int] = None,
    ) -> tuple[bool, str]:
        """
        Determine if optimization should be applied.
        
        Args:
            output_size: Number of elements in output
            index_size: Number of scatter operations
            num_partitions: Proposed number of partitions
            element_bytes: Size of each element in bytes
            available_memory: Available GPU memory in bytes, if known
        
        Returns:
            (should_apply, reason) tuple explaining the decision
        """
        if output_size == 0 or index_size == 0:
            return False, "Empty tensor"

        if num_partitions < PartitionConfig.MIN_PARTITIONS:
            return False, f"Too few partitions ({num_partitions} < {PartitionConfig.MIN_PARTITIONS})"

        if available_memory is not None:
            memory_overhead = PartitionConfig.calculate_memory_overhead(
                output_size, num_partitions, element_bytes
            )
            memory_budget = available_memory * PartitionConfig.MEMORY_BUDGET_FRACTION
            
            if memory_overhead > memory_budget:
                overhead_gb = memory_overhead / 1e9
                budget_gb = memory_budget / 1e9
                available_gb = available_memory / 1e9
                return False, (
                    f"Insufficient memory: overhead={overhead_gb:.2f}GB "
                    f"> budget={budget_gb:.2f}GB "
                    f"({PartitionConfig.MEMORY_BUDGET_FRACTION:.0%} of {available_gb:.2f}GB)"
                )

        return True, "Optimization applicable"


def extract_tensor_metadata(node: fx.Node) -> Optional[TensorMetadata]:
    """
    Extract shape, dtype, and device from FX node.
    
    Args:
        node: FX node to extract metadata from
        
    Returns:
        TensorMetadata if successful, None otherwise
    """
    if not hasattr(node, "meta") or "val" not in node.meta:
        return None
    
    val = node.meta["val"]
    
    # Handle both real tensors and FakeTensors
    is_tensor = isinstance(val, torch.Tensor) or (
        hasattr(val, "__class__") and "Tensor" in val.__class__.__name__
    )
    if not is_tensor:
        return None
    
    # Ensure all required attributes exist
    required_attrs = ["shape", "dtype", "device", "numel"]
    if not all(hasattr(val, attr) for attr in required_attrs):
        return None
    
    return TensorMetadata(
        shape=tuple(val.shape),
        dtype=val.dtype,
        device=val.device,
        numel=val.numel(),
    )


def extract_scatter_dim_and_index(indices_arg: Any) -> tuple[Optional[int], Optional[fx.Node]]:
    """
    Extract scatter dimension and index node from index_put indices argument.
    
    When index_add_(dim, idx, values) is lowered to index_put, the dimension
    information is encoded in the indices list structure:
    - dim=0: indices = [idx]
    - dim=1: indices = [None, idx]
    - dim=2: indices = [None, None, idx]
    
    Args:
        indices_arg: The indices from index_put - can be:
            - Single fx.Node: means dim=0
            - List/tuple with Nones and one fx.Node: position indicates dim
    
    Returns:
        (scatter_dim, index_node) tuple, or (None, None) if cannot extract
    """
    # Case 1: Single index node → dim=0
    if not isinstance(indices_arg, (list, tuple)):
        return 0, indices_arg
    
    # Case 2: List of indices with Nones
    # Find the non-None index and its position
    index_node = None
    scatter_dim = None
    
    for dim, idx in enumerate(indices_arg):
        if idx is not None:
            if index_node is not None:
                # Multiple non-None indices - advanced indexing not supported
                log.debug("Skipping: multi-dimensional indexing not supported")
                return None, None
            index_node = idx
            scatter_dim = dim
    
    if index_node is None:
        log.debug("Skipping: no index found in indices list")
        return None, None
    
    return scatter_dim, index_node


def get_scatter_metadata(match: Match) -> Optional[ScatterMetadata]:
    """
    Extract metadata from matched index_put pattern.
    
    Supports index_put operations on any dimension by analyzing the indices structure.
    
    Args:
        match: Pattern match from FX graph
        
    Returns:
        ScatterMetadata if extraction successful, None otherwise
    """
    try:
        output_node = match.output_node()
        if not output_node or not hasattr(output_node, "args"):
            return None

        args = output_node.args
        if len(args) < 3:
            log.debug("Skipping: invalid args length for index_put")
            return None

        input_node = args[0]
        indices_arg = args[1]
        
        # Extract scatter dimension and index node
        scatter_dim, index_node = extract_scatter_dim_and_index(indices_arg)
        if scatter_dim is None or index_node is None:
            return None

        # Extract tensor metadata
        output_meta = extract_tensor_metadata(input_node)
        index_meta = extract_tensor_metadata(index_node)
        if not output_meta or not index_meta:
            return None
        
        # Validate scatter dimension is within bounds
        if scatter_dim >= len(output_meta.shape):
            log.debug(f"Skipping: scatter dim {scatter_dim} out of bounds for shape {output_meta.shape}")
            return None

        element_bytes = output_meta.dtype.itemsize if hasattr(output_meta.dtype, "itemsize") else 4

        return ScatterMetadata(
            output_size=output_meta.numel,
            index_size=index_meta.numel,
            output_shape=output_meta.shape,
            index_shape=index_meta.shape,
            scatter_dim=scatter_dim,
            dtype=output_meta.dtype,
            device=output_meta.device,
            element_bytes=element_bytes,
        )
    except Exception as e:
        log.debug(f"Error extracting scatter metadata: {e}")
        return None


def validate_match(match: Match) -> bool:
    """
    Validate if pattern match should be optimized.
    
    Checks for:
    - Accumulate mode enabled
    - Valid tensor types and shapes
    - Sufficient contention to benefit from optimization
    - Available memory for expanded buffers
    
    Args:
        match: Pattern match from FX graph
        
    Returns:
        True if optimization should be applied, False otherwise
    """
    output_node = match.output_node()
    if not output_node or not hasattr(output_node, "args"):
        return False
    
    args = output_node.args
    if len(args) < 4:
        log.debug("Skipping: insufficient args for index_put")
        return False

    # Only apply when accumulating (4th arg is accumulate flag)
    if args[3] is not True:
        log.debug("Skipping: accumulate=False")
        return False

    metadata = get_scatter_metadata(match)
    if not metadata:
        return False

    # Skip dynamic shapes - optimization not supported with symbolic sizes
    if isinstance(metadata.output_size, torch.SymInt) or isinstance(metadata.index_size, torch.SymInt):
        log.debug("Skipping: dynamic shapes not supported")
        return False

    # Skip unsupported types
    if metadata.dtype == torch.bool:
        log.debug("Skipping: bool dtype not supported")
        return False

    # Skip scalar or empty tensors
    if not metadata.index_shape or len(metadata.index_shape) == 0:
        log.debug("Skipping: scalar index not supported")
        return False

    if metadata.index_size == 0:
        log.debug("Skipping: empty index tensor")
        return False

    if not metadata.output_shape or len(metadata.output_shape) == 0:
        log.debug("Skipping: scalar output not supported")
        return False

    # Skip boolean indices
    indices_arg = args[1]
    _, index_node = extract_scatter_dim_and_index(indices_arg)
    if index_node is not None:
        index_meta = extract_tensor_metadata(index_node)
        if index_meta and index_meta.dtype == torch.bool:
            log.debug("Skipping: boolean indices not supported")
            return False

    # Estimate optimal partition count
    estimated_partitions = PartitionConfig.estimate_optimal_partitions(
        metadata.output_size, metadata.index_size
    )

    # Adjust for memory constraints if CUDA is available
    if torch.cuda.is_available():
        try:
            _, total_memory = torch.cuda.mem_get_info()
            
            while estimated_partitions >= PartitionConfig.MIN_PARTITIONS:
                overhead = PartitionConfig.calculate_memory_overhead(
                    metadata.output_size, estimated_partitions, metadata.element_bytes
                )
                memory_budget = total_memory * PartitionConfig.MEMORY_BUDGET_FRACTION
                
                if overhead <= memory_budget:
                    break
                estimated_partitions //= 2
            else:
                log.debug("Skipping: insufficient memory for minimum partitions")
                return False
        except Exception as e:
            log.debug(f"Could not check CUDA memory: {e}")

    # Final validation check
    should_apply, reason = PartitionConfig.should_optimize(
        metadata.output_size,
        metadata.index_size,
        estimated_partitions,
        metadata.element_bytes,
    )
    
    if not should_apply:
        log.debug(f"Skipping: {reason}")
        return False

    # Store metadata for use in replacement
    match._adjusted_num_partitions = estimated_partitions
    match._scatter_dim = metadata.scatter_dim
    
    contention_ratio = metadata.index_size / metadata.output_size
    memory_overhead_gb = PartitionConfig.calculate_memory_overhead(
        metadata.output_size, estimated_partitions, metadata.element_bytes
    ) / 1e9
    
    log.debug(
        f"Applying partitioned scatter: {estimated_partitions} partitions, "
        f"dim={metadata.scatter_dim}, contention={contention_ratio:.2f}, "
        f"memory_overhead={memory_overhead_gb:.3f}GB"
    )
    
    return True


def _set_fake_tensor_meta(
    node: fx.Node,
    shape: Any,
    dtype: torch.dtype,
    device: torch.device,
    fake_mode: Any,
) -> None:
    """
    Helper to set node metadata with FakeTensor.
    
    Creates a FakeTensor with the specified properties and assigns it to node.meta["val"].
    This ensures proper metadata propagation in the FX graph during compilation.
    
    Args:
        node: FX node to set metadata on
        shape: Shape of the tensor (can be int for 1D or list/tuple for multi-D)
        dtype: Data type of the tensor
        device: Device of the tensor
        fake_mode: FakeTensorMode context for creating fake tensors
    """
    with fake_mode:
        node.meta["val"] = torch.empty(shape, dtype=dtype, device=device)


def reconstruct_indices_list(
    original_indices: Any,
    adjusted_index: fx.Node,
    scatter_dim: int
) -> list[Optional[fx.Node]]:
    """
    Reconstruct the indices list with the adjusted index at the correct position.
    
    Preserves the structure of the original indices (including None placeholders)
    while replacing the actual index tensor with the adjusted version.
    
    Args:
        original_indices: Original indices from index_put (single node or list)
        adjusted_index: The adjusted index node to insert
        scatter_dim: Which dimension is being scattered to
        
    Returns:
        List suitable for index_put's indices argument
    """
    if not isinstance(original_indices, (list, tuple)):
        # Single index case (dim=0)
        return [adjusted_index]
    
    # Reconstruct list with adjusted index at the correct position
    result = []
    for i, idx in enumerate(original_indices):
        result.append(adjusted_index if i == scatter_dim else idx)
    
    return result


def create_replacement(
    match: Match, 
    input_tensor: fx.Node, 
    indices: Any, 
    values: fx.Node
) -> fx.Node:
    """
    Replace high-contention index_put with partitioned scatter.
    
    Supports scattering along any dimension (dim=0, 1, 2, etc.)
    
    The transformation:
        output = index_put(input, indices, values, accumulate=True)
    
    Becomes:
        # 1. Enumerate operations and assign partitions
        operation_ids = iota(N)
        partition_ids = operation_ids % num_partitions
        
        # 2. Create expanded buffer along scatter dimension
        expanded = zeros([..., num_partitions * dim_size, ...])
        
        # 3. Adjust indices for partitioning
        adjusted_idx = idx + (partition_ids * dim_size)
        
        # 4. Scatter with reduced contention
        scattered = index_put(expanded, [adjusted_idx], values, accumulate=True)
        
        # 5. Reduce partitions and add to input
        reshaped = scattered.view([..., num_partitions, dim_size, ...])
        reduced = sum(reshaped, dim=scatter_dim)
        output = input + reduced
    
    Args:
        match: Pattern match containing optimization metadata
        input_tensor: Input tensor to accumulate into
        indices: Index specification (single index or list with Nones)
        values: Values to scatter
        
    Returns:
        Output node of the transformed graph
    """
    graph = match.graph
    matched_node = match.output_node()
    
    # Retrieve optimization parameters computed during validation
    num_partitions = getattr(match, "_adjusted_num_partitions", PartitionConfig.MIN_PARTITIONS)
    scatter_dim = getattr(match, "_scatter_dim", 0)

    # Extract the actual index node from the indices structure
    _, index_node = extract_scatter_dim_and_index(indices)
    if index_node is None:
        log.warning("Could not extract index node, skipping optimization")
        return matched_node

    # Get metadata from nodes
    input_meta = input_tensor.meta["val"]
    index_meta = index_node.meta["val"]
    values_meta = values.meta["val"]

    dim_size = input_meta.shape[scatter_dim]
    num_operations = index_meta.numel()

    device = index_meta.device

    # Detect fake mode from existing nodes to create proper FakeTensors for metadata
    fake_mode = detect_fake_mode([input_meta, index_meta, values_meta])
    if fake_mode is None:
        log.warning("Could not detect fake mode, skipping optimization")
        return matched_node

    with graph.inserting_before(matched_node):
        # Step 0: Flatten multi-dimensional indices if needed
        # PyTorch index_put internally flattens multi-dim indices, so we do it explicitly
        index_is_multidim = len(index_meta.shape) > 1
        
        if index_is_multidim:
            log.debug(
                f"Flattening multi-dimensional index: {index_meta.shape} -> [{num_operations}]"
            )
            
            # Flatten index: e.g., [8, 512] → [4096]
            flat_index = graph.call_function(
                torch.ops.aten.reshape.default,
                args=(index_node, [num_operations]),
            )
            _set_fake_tensor_meta(flat_index, num_operations, index_meta.dtype, device, fake_mode)
            
            # Flatten values to match: e.g., [8, 512, 64] → [4096, 64]
            # Values should have same leading dimensions as index
            flat_values_shape = [num_operations] + list(values_meta.shape[len(index_meta.shape):])
            flat_values = graph.call_function(
                torch.ops.aten.reshape.default,
                args=(values, flat_values_shape),
            )
            _set_fake_tensor_meta(flat_values, flat_values_shape, values_meta.dtype, device, fake_mode)
            
            log.debug(
                f"Flattened values: {values_meta.shape} -> {flat_values_shape}"
            )
        else:
            # Already 1D, use as-is
            flat_index = index_node
            flat_values = values

        # Step 1: Generate operation IDs [0, 1, 2, ..., N-1]
        operation_ids = graph.call_function(
            torch.ops.prims.iota.default,
            args=(num_operations,),
            kwargs={
                "start": 0,
                "step": 1,
                "dtype": index_meta.dtype,
                "device": device,
                "requires_grad": False,
            },
        )
        _set_fake_tensor_meta(operation_ids, num_operations, index_meta.dtype, device, fake_mode)
        
        # Step 2: Assign each operation to a partition using bitwise AND
        # This is equivalent to modulo but faster: id % num_partitions == id & (num_partitions - 1)
        partition_ids = graph.call_function(
            torch.ops.aten.bitwise_and.Scalar,
            args=(operation_ids, num_partitions - 1),
        )
        _set_fake_tensor_meta(partition_ids, num_operations, index_meta.dtype, device, fake_mode)

        # Step 3: Create expanded buffer (num_partitions copies along scatter_dim)
        expanded_shape = list(input_meta.shape)
        expanded_shape[scatter_dim] *= num_partitions
        
        expanded_buffer = graph.call_function(
            torch.ops.aten.full.default,
            args=(expanded_shape, 0),
            kwargs={
                "dtype": values_meta.dtype,
                "layout": torch.strided,
                "device": device,
                "pin_memory": False,
            },
        )
        _set_fake_tensor_meta(expanded_buffer, expanded_shape, values_meta.dtype, device, fake_mode)

        # Step 4: Adjust indices to point into correct partition
        # Each partition gets its own range: partition 0 gets [0, dim_size), 
        # partition 1 gets [dim_size, 2*dim_size), etc.
        partition_offsets = graph.call_function(
            torch.ops.aten.mul.Tensor, 
            args=(partition_ids, dim_size)
        )
        _set_fake_tensor_meta(partition_offsets, num_operations, index_meta.dtype, device, fake_mode)
        
        adjusted_index = graph.call_function(
            torch.ops.aten.add.Tensor, 
            args=(flat_index, partition_offsets)
        )
        _set_fake_tensor_meta(adjusted_index, num_operations, index_meta.dtype, device, fake_mode)

        # Step 5: Reconstruct indices list with adjusted index at correct position
        adjusted_indices = reconstruct_indices_list(indices, adjusted_index, scatter_dim)

        # Step 6: Perform scatter on expanded buffer with reduced contention
        scattered_buffer = graph.call_function(
            torch.ops.aten.index_put.default,
            args=(expanded_buffer, adjusted_indices, flat_values, True),
        )
        _set_fake_tensor_meta(scattered_buffer, expanded_shape, values_meta.dtype, device, fake_mode)

        # Step 7: Reduce across partitions
        # Reshape to split scatter dimension: [..., num_partitions, dim_size, ...]
        reduce_shape = list(expanded_shape)
        reduce_shape[scatter_dim] = num_partitions
        reduce_shape.insert(scatter_dim + 1, dim_size)
        
        reshaped = graph.call_function(
            torch.ops.aten.view.default, 
            args=(scattered_buffer, reduce_shape)
        )
        _set_fake_tensor_meta(reshaped, reduce_shape, values_meta.dtype, device, fake_mode)
        
        # Sum across partition dimension to merge results
        # For integer dtypes, explicitly preserve dtype to avoid promotion
        if values_meta.dtype in [torch.int8, torch.int16, torch.int32, torch.uint8]:
            reduced = graph.call_function(
                torch.ops.aten.sum.dim_IntList, 
                args=(reshaped, [scatter_dim]),
                kwargs={"dtype": values_meta.dtype},
            )
        else:
            reduced = graph.call_function(
                torch.ops.aten.sum.dim_IntList, 
                args=(reshaped, [scatter_dim])
            )
        output_shape = list(input_meta.shape)
        _set_fake_tensor_meta(reduced, output_shape, values_meta.dtype, device, fake_mode)

        # Step 8: Add reduced result to original input
        output = graph.call_function(
            torch.ops.aten.add.Tensor, 
            args=(input_tensor, reduced)
        )
        _set_fake_tensor_meta(output, output_shape, values_meta.dtype, device, fake_mode)

    # Replace the original node
    matched_node.replace_all_uses_with(output)
    graph.erase_node(matched_node)
    
    # Track metric for successful optimization
    counters["inductor"]["partitioned_scatter_applied"] += 1
    
    return output


@init_once_fakemode
def lazy_init():
    """
    Lazily register patterns for both index_put and index_put_ (in-place).
    
    Matches operations with accumulate=True (4th argument).
    Uses init_once_fakemode to ensure patterns are registered once in fake mode context.
    """
    # Pattern: index_put(input, indices, values, accumulate=True)
    register_graph_pattern(
        CallFunction(torch.ops.aten.index_put.default, Arg(), Arg(), Arg(), True),
        extra_check=validate_match,
        pass_dict=_index_scatter_patterns,
    )(create_replacement)

    # Pattern: index_put_(input, indices, values, accumulate=True)
    register_graph_pattern(
        CallFunction(torch.ops.aten.index_put_.default, Arg(), Arg(), Arg(), True),
        extra_check=validate_match,
        pass_dict=_index_scatter_patterns,
    )(create_replacement)


def partitioned_scatter_optimization_pass(graph: fx.Graph) -> fx.Graph:
    """
    Apply partitioned scatter optimization to high-contention index_put operations.
    
    This pass detects index_put operations with high atomic contention and replaces
    them with a partitioned approach that distributes writes across multiple buffers,
    reducing serialization and improving parallelism.
    
    Controlled by: config.partitioned_scatter_enabled
    Supports: Scattering along any dimension (0, 1, 2, etc.)
    Note: Dynamic shapes (SymInt) are not currently supported
    
    Args:
        graph: FX graph to optimize
        
    Returns:
        Optimized FX graph
    """
    if not getattr(config, "partitioned_scatter_enabled", False):
        return graph

    # Lazy pattern registration with fake mode initialization
    lazy_init()

    num_matches = _index_scatter_patterns.apply(graph)
    
    if num_matches > 0:
        log.info(
            f"partitioned_scatter_optimization: successfully applied to "
            f"{num_matches} operation(s)"
        )
        graph.lint()

    return graph


__all__ = [
    "partitioned_scatter_optimization_pass",
    "PartitionConfig",
    "ScatterMetadata",
    "TensorMetadata",
]
