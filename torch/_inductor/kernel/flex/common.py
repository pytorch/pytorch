# mypy: allow-untyped-defs
"""Common utilities and functions for flex attention kernels"""

import math
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING, Union

import sympy

import torch
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet
from torch.utils._pytree import tree_map, tree_map_only


if TYPE_CHECKING:
    from torch._inductor.codegen.cuda_combined_scheduling import _IntLike
else:
    _IntLike = Union[int, sympy.Expr]


from ...ir import (
    ComputedBuffer,
    ExternKernel,
    FixedLayout,
    FlexibleLayout,
    get_fill_order,
    InputBuffer,
    IRNode,
    MutationLayoutSHOULDREMOVE,
    Scatter,
    ShapeAsConstantBuffer,
    StorageBox,
    Subgraph,
    TensorBox,
)
from ...lowering import (
    _full,
    check_and_broadcast_indices,
    expand,
    index_output_size_and_inner_fn,
    to_dtype,
)
from ...select_algorithm import realize_inputs
from ...utils import load_template


SubgraphResults = Union[list[Optional[ComputedBuffer]], Optional[ComputedBuffer]]


def zeros_and_scatter_lowering(shape: list[int], indices, values):
    """To support backwards on captured buffers we register a specific lowering for our specific custom up"""
    # Always accumulate into fp32 then cast
    grad = _full(0, values.get_device(), torch.float32, shape)
    assert isinstance(grad, TensorBox)
    grad.realize()
    x_size = grad.get_size()
    values = to_dtype(values, grad.get_dtype())
    indices_loaders = [i.make_loader() if i is not None else None for i in indices]
    indices, tensor_indices = check_and_broadcast_indices(indices, grad.get_device())
    # We can use the first one since they are all required to be the same size
    tensor_size = list(indices[tensor_indices[0]].get_size())
    indexed_size = [x_size[i] for i in range(len(indices))]

    expected_vals_size, inner_fn = index_output_size_and_inner_fn(
        x_size,
        indices,
        tensor_indices,
        tensor_size,
        indices_loaders,
        indexed_size,
        None,
        check=True,
    )

    values = expand(values, expected_vals_size)
    device = grad.get_device()
    assert device is not None
    scatter = Scatter(
        device=device,
        dtype=grad.get_dtype(),
        inner_fn=values.make_loader(),
        ranges=expected_vals_size,  # iter_ranges,
        output_indexer=inner_fn,
        scatter_mode="atomic_add",
    )

    buffer = ComputedBuffer(
        name=grad.data.data.name,  # type: ignore[attr-defined]
        layout=MutationLayoutSHOULDREMOVE(grad),
        data=scatter,
    )
    return buffer


def get_fwd_subgraph_outputs(
    subgraph_buffer: SubgraphResults, mask_graph_buffer: SubgraphResults
) -> list[Optional[ComputedBuffer]]:
    subgraph_buffer = (
        # pyrefly: ignore [bad-assignment]
        subgraph_buffer if isinstance(subgraph_buffer, Sequence) else [subgraph_buffer]
    )
    mask_graph_buffer = (
        # pyrefly: ignore [bad-assignment]
        mask_graph_buffer
        if isinstance(mask_graph_buffer, Sequence)
        else [mask_graph_buffer]
    )
    # pyrefly: ignore [not-iterable]
    return [*subgraph_buffer, *mask_graph_buffer]


def build_subgraph_module_buffer(
    args: list[Union[TensorBox, ShapeAsConstantBuffer]],
    graph_module: torch.fx.GraphModule,
) -> SubgraphResults:
    """This function's goal is to take in the required args and produce the subgraph buffer
    The subgraph buffer is a ComputedBuffer that will be inlined into the triton template

    Args:
        args: The args that are passed into the subgraph. Contains both fixed and lifted inputs.
        subgraph: The Subgraph ir for which to produce the output node
    """
    # This one we gotta keep lazy
    from ...subgraph_lowering import PointwiseSubgraphLowering

    pw_subgraph = PointwiseSubgraphLowering(
        graph_module,
        root_graph_lowering=V.graph,
        allowed_mutations=OrderedSet([torch.ops.flex_lib.zeros_and_scatter.default]),
        additional_lowerings={
            torch.ops.flex_lib.zeros_and_scatter.default: zeros_and_scatter_lowering
        },
    )
    with V.set_graph_handler(pw_subgraph):  # type: ignore[arg-type]
        pw_subgraph.run(*args)

    def convert_output_node_to_buffer(output_buffer) -> Optional[ComputedBuffer]:
        if output_buffer is None:
            return None
        if isinstance(output_buffer, ComputedBuffer):
            # These nodes are coming from the output of zeros_and_scatter
            return output_buffer
        assert isinstance(output_buffer, TensorBox), (
            "The output node for flex attention's subgraph must be a TensorBox, but got: ",
            type(output_buffer),
        )
        assert isinstance(output_buffer.data, StorageBox), (
            "The output node for the flex attention subgraph must be a StorageBox, but got: ",
            type(output_buffer),
        )
        device = output_buffer.data.get_device()
        assert device is not None
        subgraph_buffer = ComputedBuffer(
            name=None,
            layout=FlexibleLayout(
                device=device,
                dtype=output_buffer.data.get_dtype(),
                size=output_buffer.data.get_size(),
            ),
            data=output_buffer.data.data,  # type: ignore[arg-type]
        )
        return subgraph_buffer

    return tree_map(convert_output_node_to_buffer, pw_subgraph.graph_outputs)


def build_subgraph_buffer(
    args: list[Union[TensorBox, ShapeAsConstantBuffer]], subgraph: Subgraph
) -> SubgraphResults:
    return build_subgraph_module_buffer(args, subgraph.graph_module)


def maybe_realize(args: list[Optional[IRNode]]):
    """Accepts a list of optional IRNodes and returns a list of realized IRNodes"""
    return tree_map(
        lambda x: (
            realize_inputs(x)
            if x is not None and not isinstance(x, sympy.Symbol)
            else x
        ),
        args,
    )


def freeze_irnodes(tree: Any) -> Any:
    """Freeze layouts for every IRNode contained in a pytree."""

    if tree is None:
        return None

    def _freeze(node: IRNode) -> IRNode:
        try:
            node.freeze_layout()
        except NotImplementedError:
            pass
        return node

    return tree_map_only(IRNode, _freeze, tree)


def create_placeholder(
    name: str,
    dtype: torch.dtype,
    device: torch.device,
    size: Optional[list[int]] = None,
) -> Union[TensorBox, ShapeAsConstantBuffer]:
    """Creates a placeholder input buffers for producing subgraph_output."""
    input_buffer = InputBuffer(
        name=name,
        layout=FixedLayout(
            device,
            dtype,
            size if size else [],
            FlexibleLayout.contiguous_strides(size) if size else [],
        ),
    )
    return TensorBox.create(input_buffer)


def construct_strides(
    sizes: Sequence[_IntLike],
    fill_order: Sequence[int],
) -> Sequence[_IntLike]:
    """From a list of sizes and a fill order, construct the strides of the permuted tensor."""
    # Initialize strides
    assert len(sizes) == len(fill_order), (
        "Length of sizes must match the length of the fill order"
    )
    strides: list[_IntLike] = [0] * len(sizes)

    # Start with stride 1 for the innermost dimension
    current_stride: _IntLike = 1

    # Iterate through the fill order populating strides
    for dim in fill_order:
        strides[dim] = current_stride
        current_stride *= sizes[dim]

    return strides


def infer_dense_strides(
    size: Sequence[_IntLike],
    orig_strides: Sequence[_IntLike],
):
    """This is a mirror of the same function in aten/src/ATen/ExpandUtils.cpp

    Args:
        size: The size of the output tensor
        orig_strides: The strides of the input tensor
    Returns:
        List[int]: Dense non-overlapping strides that preserve the input tensor's layout permutation.
        The returned strides follow the same stride propagation rules as TensorIterator. This matches
        The behavior of empty_like()
    """
    fill_order = get_fill_order(orig_strides, V.graph.sizevars.shape_env)
    return construct_strides(size, fill_order)


def create_indices_fake(x) -> torch.Tensor:
    """Create a fake indices that is used for autotuning."""
    size = [V.graph.sizevars.size_hint(i) for i in x.get_size()]
    indices = torch.arange(0, size[-1], dtype=x.get_dtype(), device=x.get_device())
    indices = indices.expand(size).contiguous()
    return indices


def create_num_blocks_fake_generator(sparse_indices):
    """Create a fake num_blocks that is used for autotuning.

    The idea here is that we need to create a real tensor with real data
    that's representative for benchmarking.
    For example, returning all zeros for the `kv_num_blocks` input would mean
    that we are computing 0 blocks for each row, which would provide bogus
    autotuning results.

    In this case, we choose to use min(16, max_block) blocks, because I
    (Horace) think it'll probably result in pretty representative performance.
    If it's too short then prefetching won't help. If it's too long then
    autotuning will take longer for no good reason.
    """

    def create_num_blocks_fake(x) -> torch.Tensor:
        num_blocks_for_autotuning = V.graph.sizevars.size_hint(sparse_indices.shape[-1])
        size = [V.graph.sizevars.size_hint(i) for i in x.get_size()]
        return torch.full(
            size,
            num_blocks_for_autotuning,
            dtype=x.get_dtype(),
            device=x.get_device(),
        )

    return create_num_blocks_fake


def contiguous_last_dim(x):
    """Ensure that realized IR node has a contiguous stride in the last dimension."""
    strides = x.maybe_get_stride()
    if strides and strides[-1] != 1:
        contiguous_stride_order = list(reversed(range(len(x.get_size()))))
        return ExternKernel.require_stride_order(x, contiguous_stride_order)
    return x


def set_head_dim_values(
    kernel_options: dict[str, Any], qk_head_dim, v_head_dim, graph_sizevars
):
    """
    Mutates kernel options, adding head dimension calculations.

    Args:
        kernel_options: Dictionary to populate with options
        qk_head_dim: Query/Key head dimension
        v_head_dim: Value head dimension
        graph_sizevars: Graph size variables object with guard_int method

    """
    # QK dimensions
    qk_head_dim_static = graph_sizevars.guard_int(qk_head_dim)
    kernel_options.setdefault("QK_HEAD_DIM", qk_head_dim_static)
    kernel_options.setdefault(
        "QK_HEAD_DIM_ROUNDED", next_power_of_two(qk_head_dim_static)
    )

    # V dimensions
    v_head_dim_static = graph_sizevars.guard_int(v_head_dim)
    kernel_options.setdefault("V_HEAD_DIM", v_head_dim_static)
    kernel_options.setdefault(
        "V_HEAD_DIM_ROUNDED", next_power_of_two(v_head_dim_static)
    )

    # Safety flag
    kernel_options.setdefault(
        "SAFE_HEAD_DIM",
        is_power_of_2(qk_head_dim_static) and is_power_of_2(v_head_dim_static),
    )


def is_power_of_2(n):
    return n != 0 and ((n & (n - 1)) == 0)


def next_power_of_two(n):
    if n <= 0:
        return 1
    return 2 ** math.ceil(math.log2(n))


_FLEX_TEMPLATE_DIR = Path(__file__).parent / "templates"
load_flex_template = partial(load_template, template_dir=_FLEX_TEMPLATE_DIR)


# Template strings have been moved to templates/common.py.jinja
