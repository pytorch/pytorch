import math
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch

import sympy

import torch._inductor.virtualized as virtualized
from torch._inductor import ir


def map_pointwise_index_to_read_strides(
    index_expr: sympy.Expr, master_layout: ir.Layout, flip_mn: bool
) -> List[int]:
    """
    Converts a sympy index expression to a list of strides, mapped to the master layout
    """
    strides = []
    free_symbols = list(index_expr.free_symbols)
    assert len(free_symbols) <= len(
        master_layout.stride
    ), f"Too many free symbols in index expression {index_expr} for layout {master_layout}"
    subs = {sym: 0 for sym in free_symbols}
    result_strides = [0] * len(master_layout.stride)
    sym_idx_map = list(range(len(result_strides)))
    # flip m and n dimensions in this mapping
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
        stride = index_expr.evalf(subs=subs)
        assert (
            math.isfinite(stride) and stride >= 0.0
        ), f"Invalid stride {stride} for symbol {free_symbols[i]} in index expression {index_expr}"
        stride = int(stride)
        result_strides[sym_idx] = stride

    return result_strides


def map_pointwise_index_to_read_strides(
    index_expr: sympy.Expr, master_layout: ir.Layout, flip_mn: bool
) -> List[int]:
    """
    Converts a sympy index expression to a list of strides, mapped to the GEMM output master layout.
    Can flip M and N dimensions in the mapping if the corresponding flag is set.
    """
    free_symbols = list(index_expr.free_symbols)
    assert len(free_symbols) <= len(
        master_layout.stride
    ), f"Too many free symbols in index expression {index_expr} for layout {master_layout}"
    subs = {sym: 0 for sym in free_symbols}
    result_strides = [0] * len(master_layout.stride)
    sym_idx_map = list(range(len(result_strides)))
    # flip m and n dimensions in this mapping
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
        stride = index_expr.evalf(subs=subs)
        assert (
            math.isfinite(stride) and stride >= 0.0
        ), f"Invalid stride {stride} for symbol {free_symbols[i]} in index expression {index_expr}"
        stride = int(stride)
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
    and just keeps them in a map for further usage.
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


@staticmethod
def extract_pointwise_load_strides(
    node: ir.IRNode, reference_buffer: ir.Buffer
) -> Dict[str, Tuple[List[int], int, List[int]]]:
    """
    Extract the strides used to load inputs to the pointwise op, mapped to the corresponding dimensions of a
    reference buffer that is also among the inputs.
    """
    if isinstance(node, ir.ComputedBuffer):
        pointwise_node = node.data
    reference_layout = node.get_layout()
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
        reference_buffer.name, None
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
                f"Reference buffer {reference_buffer.name} is being read with stride {stride} that has no corresponding existing dimension."
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
                    f"Buffer {name} is being read with stride {stride} ( index name: {index_name} ) that has no corresponding existing dimension in reference buffer {reference_buffer.name}."
                )
            buffer_strides[dim] = stride
            buffer_sizes[dim] = reference_layout.size[
                dim
            ]  # we take the size from the reference layout, all broadcasted dimensions set to 1
        result[name] = (buffer_strides, buffer_offset, buffer_sizes)
    return result
