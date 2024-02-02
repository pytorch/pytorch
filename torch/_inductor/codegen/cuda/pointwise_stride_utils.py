import math
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch

import sympy

import torch._inductor.virtualized as virtualized
from torch._inductor import ir

# Utility functions for working with the index expressions encountered in the Pointwise IRNode's
# 'define by run' inner function, which is accessed via the virtualized.V.ops_handler mechanism.
# required by other modules in this package.


def map_pointwise_index_to_read_strides(
    index_expr: sympy.Expr, master_layout: ir.Layout, flip_mn: bool
) -> List[int]:
    """
    Converts a sympy index expression to a list of strides in accordance with the provided master layout.

    Takes a sympy index expression as input and converts this to a list of strides, aligned to the layout
    specified by `master_layout`. The M and N dimensions can be flipped in the mapping based on the
    `flip_mn` boolean flag.

    Note: This function does not return the offset, but considers it while computing strides.

    Args:
        index_expr (sympy.Expr): The index expression to convert to strides.
        master_layout (ir.Layout): The master layout for stride mapping.
        flip_mn (bool): A flag determining if M and N dimensions should be flipped in the mapping.

    Returns:
        List[int]: A list of strides mapped to the GEMM output master_layout.
    """
    free_symbols = list(index_expr.free_symbols)
    assert len(free_symbols) <= len(
        master_layout.stride
    ), f"Too many free symbols in index expression {index_expr} for layout {master_layout}"
    subs = {sym: 0 for sym in free_symbols}
    # Calculate constant offset first by setting all variable coefficients to zero
    # e.g. if we have "256 + 64 * i0 + i1" and set both i0 and i1 to 0, we get 256 as offset
    offset = index_expr.evalf(subs=subs)
    assert (
        math.isfinite(offset) and offset >= 0.0
    ), f"Invalid offset {offset} in index expression {index_expr}"
    offset = int(offset)
    result_strides = [0] * len(master_layout.stride)
    sym_idx_map = list(range(len(result_strides)))
    # flip m and n dimensions in this mapping if requested by the flip_mn flag
    if flip_mn and len(master_layout.stride) >= 2:
        sym_idx_map[-1] = len(master_layout.stride) - 2
        sym_idx_map[-2] = len(master_layout.stride) - 1
    for i in range(len(free_symbols)):
        sym_name = free_symbols[i].name
        assert sym_name[0] == "i"
        sym_idx = sym_idx_map[int(sym_name[1:])]
        assert sym_idx >= 0 and sym_idx < len(
            master_layout.stride
        ), f"Invalid symbol name {sym_name} in index expression {index_expr} for layout {master_layout}"
        if i > 0:
            subs[free_symbols[i - 1]] = 0
        subs[free_symbols[i]] = 1
        # Result of the calculation below is a stride + offset.
        # Example: index_expr = "256 + 64 * i0 + i1"
        # Now we set all index variables (i0, i1, ...) to 0 except the one we are
        # interested in. So we set i0 to 1 and i1 to 0
        # this gives us 256 + 64, which equals the constant offset ( 256 ) + the stride ( 64 ) of i1
        stride_plus_offset = index_expr.evalf(subs=subs)
        assert (
            math.isfinite(stride_plus_offset) and stride_plus_offset >= 0.0
        ), f"Invalid stride+offset {stride_plus_offset} for symbol {free_symbols[i]} in index expression {index_expr}"
        stride = (
            int(stride_plus_offset) - offset
        )  # need to subtract constant offset to arrive at stride
        result_strides[sym_idx] = stride

    return result_strides


def index_to_stride_dict(index_expr: sympy.Expr) -> Dict[str, int]:
    """
    Interprets a sympy index expression used by Pointwise nodes for load operations, mapping indices
    to stride values and providing an offset.

    This function constructs a dictionary from the provided index expression. Each index ('i0', 'i1', etc.)
    in the expression is mapped to an integer 'stride' value. The function also adds an 'offset' key to the
    dictionary. The offset is calculated by setting all free symbols in the index expression to 0 and
    evaluating the resulting value.

    Args:
        index_expr (sympy expression): The sympy index expression to interpret.

    Returns:
        dict: A dictionary mapping indices ('i0', 'i1', etc.) to their corresponding stride values.
        The dictionary also contains an 'offset' key, which is derived from the value of the index
        expression when all free symbols are set to 0.
    """
    free_symbols = list(index_expr.free_symbols)
    subs = {sym: 0 for sym in free_symbols}
    result = {sym.name: 0 for sym in free_symbols}
    offset = index_expr.evalf(subs=subs)
    assert (
        math.isfinite(offset) and offset >= 0.0
    ), f"Invalid offset {offset} in index expression {index_expr}"
    offset = int(offset)
    result["offset"] = offset
    for i in range(len(free_symbols)):
        sym_name = free_symbols[i].name
        assert sym_name[0] == "i"
        if i > 0:
            subs[free_symbols[i - 1]] = 0
        subs[free_symbols[i]] = 1
        stride_plus_offset = index_expr.evalf(subs=subs)
        assert (
            math.isfinite(stride_plus_offset) and stride_plus_offset >= 0.0
        ), f"Invalid stride {stride_plus_offset} for symbol {free_symbols[i]} in index expression {index_expr}"
        stride = int(stride_plus_offset) - offset
        result[sym_name] = stride
    return result


class _IndexExtractor:
    """
    V.ops handler that extracts load index expressions for each buffer loaded
    and just keeps them in a map for further usage. Used by function extract_pointwise_load_strides
    """

    def __init__(
        self,
    ):
        self.name_index_expr_map: Dict[str, Any] = {}

    def __getattr__(self, name):
        # Ignore V.ops.<whatever> calls we are not interested in
        def ignore(*args, **kwargs):
            pass

        return ignore

    def load(self, name, index_expr):
        self.name_index_expr_map[name] = index_expr


def extract_pointwise_load_strides(
    node: ir.IRNode, reference_buffer: ir.Buffer
) -> Dict[str, Tuple[List[int], int, List[int]]]:
    """
    Extract the strides used to load inputs to a pointwise node, mapped to the corresponding dimensions of a
    reference buffer that is also among the inputs.

    This is similar to `map_pointwise_index_to_read_strides`, but instead of mapping to a given index expression,
    we are mapping to the layout of a given Buffer. This Buffer's name has to appear in the inputs loaded by the
    Pointwise IRNode.

    Args:
        node (ir.IRNode): The node from which to extract the strides.
        reference_buffer (ir.Buffer): The reference buffer to which to map the strides.

    Returns:
        dict: A dictionary that maps the buffer names to a tuple of strides (List[int]), offset (int), and sizes (List[int]).
        Each tuple corresponds to a load instruction encountered in the Pointwise IRNode, except for the reference buffer's.
    """
    if isinstance(node, ir.ComputedBuffer):
        pointwise_node = node.data
    reference_layout = reference_buffer.get_layout()
    assert isinstance(
        reference_layout, ir.FixedLayout
    ), f"Expected FixedLayout from reference_buffer.get_layout(), got {reference_layout}"
    assert isinstance(
        pointwise_node, ir.Pointwise
    ), f"Expected a ComputedBuffer wrapping a Pointwise node, got {pointwise_node}"
    extractor = _IndexExtractor()
    with virtualized.V.set_ops_handler(extractor), patch.object(  # type: ignore[call-arg]
        ir.FlexibleLayout, "allow_indexing", True
    ):
        index = pointwise_node._index(pointwise_node.ranges)
        # Call the inner_fn, which will call back into the extractor using the V.ops_handler mechanism
        # populating extrqctor.name_index_expr_map
        pointwise_node.inner_fn(index)

    name_index_expr_map = extractor.name_index_expr_map
    reference_index_expr: Optional[sympy.Expr] = name_index_expr_map.get(
        reference_buffer.name, None  # type: ignore[arg-type]
    )
    if reference_index_expr is None:
        raise RuntimeError("Reference buffer not loaded by pointwise op")
    reference_strides = index_to_stride_dict(reference_index_expr)
    assert (
        reference_strides["offset"] == 0
    ), f"Reference buffer offset is not 0: {reference_strides['offset']}"
    index_to_reference_dim: Dict[str, int] = {}
    reference_stride_to_dim_map = {
        stride: i for i, stride in enumerate(reference_layout.stride)
    }
    for name, stride in reference_strides.items():
        if name == "offset":
            continue
        if stride == 0:
            continue
        dim = reference_stride_to_dim_map.get(stride, -1)
        if dim == -1:
            raise RuntimeError(
                f"Reference buffer {reference_buffer.name} is being read with stride {stride} that has no corresponding existing dimension."  # noqa: B950
            )
        index_to_reference_dim[name] = dim
    result = {}
    for name, index_expr in name_index_expr_map.items():
        if name == reference_buffer.name:
            continue
        index_name_to_stride_dict = index_to_stride_dict(index_expr)
        assert (
            index_name_to_stride_dict["offset"] == 0
        ), f"Buffer {name} offset is not 0: {index_name_to_stride_dict['offset']}"
        buffer_strides = [0] * len(reference_layout.stride)
        buffer_sizes = [1] * len(reference_layout.size)
        buffer_offset = 0
        for index_name, stride in index_name_to_stride_dict.items():
            if index_name == "offset":
                buffer_offset = stride
                continue
            if stride == 0:
                continue
            dim = index_to_reference_dim.get(index_name, -1)
            if dim == -1:
                raise RuntimeError(
                    f"Buffer {name} is being read with stride {stride} ( index name: {index_name} ) that has no corresponding existing dimension in reference buffer {reference_buffer.name}."  # noqa: B950
                )
            buffer_strides[dim] = stride
            buffer_sizes[dim] = reference_layout.size[
                dim
            ]  # we take the size from the reference layout, all broadcasted dimensions set to 1
        result[name] = (buffer_strides, buffer_offset, buffer_sizes)
    return result


def extract_epilogue_storage_layout(
    epilogue_node: ir.IRNode, gemm_node: ir.Buffer
) -> Tuple[List[int], List[int]]:
    if isinstance(epilogue_node, ir.ComputedBuffer):
        pointwise_node = epilogue_node.data
    target_layout = epilogue_node.get_layout()
    reference_layout = gemm_node.get_layout()
    assert isinstance(
        reference_layout, ir.FixedLayout
    ), f"Expected FixedLayout from reference_buffer.get_layout(), got {reference_layout}"
    assert isinstance(
        pointwise_node, ir.Pointwise
    ), f"Expected a ComputedBuffer wrapping a Pointwise node, got {pointwise_node}"
    extractor = _IndexExtractor()
    with virtualized.V.set_ops_handler(extractor), patch.object(  # type: ignore[call-arg]
        ir.FlexibleLayout, "allow_indexing", True
    ):
        index = pointwise_node._index(pointwise_node.ranges)
        # Call the inner_fn, which will call back into the extractor using the V.ops_handler mechanism
        # populating extrqctor.name_index_expr_map
        pointwise_node.inner_fn(index)

    name_index_expr_map = extractor.name_index_expr_map
    reference_index_expr: Optional[sympy.Expr] = name_index_expr_map.get(
        gemm_node.name, None  # type: ignore[arg-type]
    )
    if reference_index_expr is None:
        raise RuntimeError("Reference buffer not loaded by pointwise op")
    reference_strides = index_to_stride_dict(reference_index_expr)
    gemm_output_strides = [0] * len(reference_layout.stride)
    gemm_output_sizes = list(reference_layout.size)
    for name, stride in reference_strides.items():
        if name == "offset":
            continue
        target_dim = int(name[1:])
        reference_dim = reference_layout.stride.index(stride)
        assert (
            reference_layout.size[reference_dim] == target_layout.size[target_dim]
        ), f"Size mismatch between reference and target layouts for {target_dim=}, {reference_dim=}"
        gemm_output_strides[reference_dim] = target_layout.stride[target_dim]
    return gemm_output_strides, gemm_output_sizes
