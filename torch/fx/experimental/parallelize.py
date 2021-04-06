from enum import Enum, unique
from typing import Dict, NamedTuple, Tuple, Optional

import torch.fx
from torch.fx.experimental.graph_manipulation import get_shape_and_dtype
from torch.fx.node import Node

# Steps to register new node:
# 1. Modify NodeKind enum class to register new node kind.
# 2. Update _node_kind_key_to_node_kind map to register the new mapping
#    from NodeKindKey to NodeKind.
# 3. [Optional] Update _parallelization_config_map to register the
#    ParallelizationConfig you would like to register for new node kind.
# 4. Add a test case to test_fx_parallelize.py.


@unique
class ParallelTransformKind(Enum):
    DATA = 1
    MODEL = 2
    SPECIAL = 3


class NodeKindKey(NamedTuple):
    """Used op and target together as a key to identify the NodeKind of a node.
    This class is supposed to be used internally, users should not use it directly.
    """

    op: str
    target: str


class ParallelizationConfig(NamedTuple):
    """Used to specify how we would like to parallelize the node."""

    # Map key is the string keyword used to look up nodes from keyworded variables: kwargs.
    # Map value is the dim to split for that input node.
    split_dims: Dict[str, int]
    # result_dim is the dimension on which we are splitting and then
    # concatenating the results.
    result_dim: int


@unique
class NodeKind(Enum):
    LINEAR = 1
    # ----- Register new node kind below -----


# Mapping from NodeKindKey to NodeKind.
_node_kind_key_to_node_kind: Dict[NodeKindKey, NodeKind] = {
    NodeKindKey(op="call_function", target="acc_ops.linear"): NodeKind.LINEAR,
    # ----- Register mapping of NodeKindKey to NodeKind for new node kind below -----
}


_parallelization_config_map = {
    # DATA Parallelization.
    ParallelTransformKind.DATA: {
        # Linear node, i.e., fully connection layer.
        NodeKind.LINEAR:
        # Return a lambda func that accepts node_shape, since parallelization config
        # might require information about node shape.
        lambda node_shape: ParallelizationConfig(
            split_dims={
                # Split input node.
                # Keyword of the input node can be found from fbcode/glow/fb/fx/acc_tracer/acc_ops.py
                # Split input node at dim 0, i.e., split by batch size.
                "input": 0
            },
            # Since we split input node at dim 0, we would like to concatenate the
            # result at dim 0 as well.
            result_dim=0,
        ),
        # ----- Register new node kind below -----
    },
    # MODEL Parallelization.
    ParallelTransformKind.MODEL: {
        # Linear node, i.e., fully connection layer.
        NodeKind.LINEAR:
        # Return a lambda func that accepts node_shape, since parallelization config
        # might require information about node shape.
        lambda node_shape: ParallelizationConfig(
            split_dims={
                # Split weight node.
                # Keyword of the weight node can be found from fbcode/glow/fb/fx/acc_tracer/acc_ops.py
                # Split weight node at dim 0, i.e., split by output features dimension.
                # Note that weight node's shape looks like [out_features, input_features]
                "weight": 0,
                # Split bias node.
                # Keyword of the bias node can be found from fbcode/glow/fb/fx/acc_tracer/acc_ops.py
                # Split bias node at dim 0, i.e., split by output features dimension.
                # Note that bias node's shape looks like [out_features]
                "bias": 0,
            },
            # Since we split weight/bias node at output features dimension, we would like
            # to concatenate the result at last dim as well.
            result_dim=len(node_shape) - 1,
        ),
        # ----- Register new node kind below -----
    },
}


def _get_node_kind_key(node: Node) -> NodeKindKey:
    """Get NodeKindKey that represents the category of this node."""
    node_kind_key = NodeKindKey(
        op=node.op,
        target=node.target
        if isinstance(node.target, str)
        else torch.fx.node._get_qualified_name(node.target).replace("glow.fb.fx.", ""),
    )
    return node_kind_key


def _get_node_kind(node: Node) -> NodeKind:
    """Get NodeKind that represents the category of this node."""
    node_kind_key = _get_node_kind_key(node)
    assert (
        node_kind_key in _node_kind_key_to_node_kind
    ), f"New node kind {node_kind_key} not supported. Please register this new node."
    return _node_kind_key_to_node_kind[node_kind_key]


def _node_to_config(
    parallel_transform_mode: ParallelTransformKind, node: Node
) -> ParallelizationConfig:
    """Get ParallelizationConfig for `node`."""
    assert parallel_transform_mode in _parallelization_config_map

    # Get the node kind for current node, for example: NodeKind.LINEAR.
    node_kind = _get_node_kind(node)
    assert node_kind in _parallelization_config_map[parallel_transform_mode], (
        f"ParallelizationConfig not registered for {node_kind} "
        + f"with ParallelTransformKind {parallel_transform_mode}."
    )
    shape, _ = get_shape_and_dtype(node)
    return _parallelization_config_map[parallel_transform_mode][node_kind](shape)


def _parallelize_and_replace_node(
    mod_traced: torch.fx.GraphModule,
    curr_node: Node,
    num_of_chunks_per_node: int,
    input_node_key: str,
    split_dims: Dict[str, int],
    result_dim: int,
    model_parallel_split_alignment: int = 1,
) -> Node:
    """
    Helper to parallelize a node `curr_node` from `mod_traced` into `num_of_chunks_per_node`
    nodes by slicing its inputs, creating clones of it and changing the inputs
    of the clones to the slices, and then concatenating all of the clones
    together and replacing `curr_node` with the concat. `input_node_key` is the
    input keyword from `curr_node`'s kwargs that will be split (there may be more than one
    input to split, but their splitDim should all have the same size).
    `split_dims` represents what dimension to split for each of the inputs to
    `curr_node`. `result_dim` is the dimension on which we are splitting and then
    concatenating the results. The size of the splits will be increased to a
    multiple of `model_parallel_split_alignment`, if possible. If the result after
    aligning the splits is that the new aligned splits are larger than the original
    requested num splits, then the number of resulting splits may be less than requested.

    Returns the ConcatNode that is created and replaces curr_node.
    """
    assert num_of_chunks_per_node > 1
    input_dim = split_dims[input_node_key]
    input_node = curr_node.kwargs[input_node_key]
    assert isinstance(input_node, Node)
    input_shape, _ = get_shape_and_dtype(input_node)
    assert input_dim >= 0 and input_dim < len(
        input_shape
    ), "Input split dim must be valid."
    batch_size = input_shape[input_dim]
    if batch_size < num_of_chunks_per_node:
        raise RuntimeError(
            """
            Invalid parallelization;
            batch_size {} must be >= num_of_chunks_per_node {} for node {}.
            """.format(
                batch_size,
                num_of_chunks_per_node,
                curr_node.name,
            )
        )

    # 1. Calculate num of chunks to split to, and num of elements per chunk.
    elem_per_chunk = batch_size // num_of_chunks_per_node
    remain = batch_size % num_of_chunks_per_node
    # This alignment will create aligned splits. So for example, if we're
    # splitting 190 by 3, then without alignment it would be {64, 63, 63}.
    # With alignment of 64, it will be {64, 64, 62}.
    aligned_elem_per_chunk = (
        (elem_per_chunk + (model_parallel_split_alignment - 1))
        // model_parallel_split_alignment
    ) * model_parallel_split_alignment
    # Potentially modify num_of_chunks_per_node, if the aligned size times
    # current num_of_chunks_per_node exceeds the total size.
    if model_parallel_split_alignment > 1:
        num_of_chunks_per_node = (
            batch_size + aligned_elem_per_chunk - 1
        ) // aligned_elem_per_chunk

    # 2. Now start to split node.
    split_nodes = []
    for i in range(0, num_of_chunks_per_node):
        # 3. Calculate slice_start and slice_end.
        if model_parallel_split_alignment > 1:
            # If we are using aligned splits, then slice by multiples of the
            # alignment, leaving the rest to the last split. The last split is
            # necessarily smaller than the other splits.
            slice_start = i * aligned_elem_per_chunk
            slice_end = (
                slice_start + aligned_elem_per_chunk
                if (i < num_of_chunks_per_node - 1)
                else batch_size
            )
        else:
            # Otherwise, distribute elements evenly across the splits and sprinkle
            # the remainder evenly as well.
            slice_start = i * elem_per_chunk + min(i, remain)
            slice_end = slice_start + elem_per_chunk + (1 if i < remain else 0)

        # 4. Clone original Node.
        # The cloned Node would keep all of the inputs/members of
        # the original Node. Then change the inputs to be the sliced inputs.
        # # We would like to insert the cloned node after curr_node, and then insert
        # sliced input node before curr_node so that we make sure that the graph is
        # still valid.
        with mod_traced.graph.inserting_after(curr_node):
            clone_node = mod_traced.graph.node_copy(curr_node)
        # We need to insert all the sliced input nodes before curr_node.
        with mod_traced.graph.inserting_before(curr_node):
            # 5. Loop over all of the inputs and slice those inputs that need to be
            # sliced, and set them on the clone.
            for split_node_key, split_dim in split_dims.items():
                input_node_to_slice = curr_node.kwargs[split_node_key]
                assert isinstance(input_node_to_slice, Node)
                input_node_shape, _ = get_shape_and_dtype(input_node_to_slice)
                assert split_dim >= 0 and split_dim < len(
                    input_node_shape
                ), "Input split dim must be valid."
                split_dim_size = input_node_shape[split_dim]
                assert split_dim_size == batch_size, (
                    f"input node {split_node_key}'s split dim {split_dim}'s size {split_dim_size}"
                    f" is different from that of input node {input_node_key}"
                )

                sliced_input_node = mod_traced.graph.call_function(
                    the_function=torch.narrow,
                    args=(
                        input_node_to_slice,
                        split_dim,
                        slice_start,
                        slice_end - slice_start,
                    ),
                    type_expr=input_node.type,
                )
                clone_node.replace_input_with(input_node_to_slice, sliced_input_node)
        split_nodes.append(clone_node)

    # 6. Create concatenate node, and insert it after the last split node.
    # split_nodes[0] comes from the fact that clone node is always inserted into the graph right
    # after curr_node, thus split_nodes[0] is the last split node.
    with mod_traced.graph.inserting_after(split_nodes[0]):
        concat_node = mod_traced.graph.call_function(
            the_function=torch.cat,
            args=(split_nodes, result_dim),
            type_expr=curr_node.type,
        )

    # 7. Finally replace use of curr_node with concat_node.
    curr_node.replace_all_uses_with(concat_node)
    return concat_node


def parallelize_ops(
    mod_traced: torch.fx.GraphModule,
    parallelization_per_node: Dict[
        Node,
        Tuple[ParallelTransformKind, Optional[int], Optional[ParallelizationConfig]],
    ],
    num_of_chunks: int = 1,
    model_parallel_split_alignment: int = 1,
) -> Dict[Node, Node]:
    """
    Perform in-place data or model or special parallel transformation of supported Nodes
    in `mod_traced`.

    For data and model parallelization, parallization is predefined and the
    ParallelizationConfig from the input is ignored. For special parallization,
    ParallelizationConfig from the input will be used.

    .. note::

        mod_traced should be generated using glow.fb.fx.acc_tracer instead of
        torch.fx.symbolic_trace.

    .. note::

        This method only modifies GraphModule.graph, users need to call
        GraphModule.recompile, otherwise the generated code of the GraphModule will
        be out of date.

    Args:

        mod_traced (torch.fx.GraphModule): the GraphModule that we would like to
            parallelize on.
        parallelization_per_node (Dict): Key is the node, value is a tuple specifying
            how we would like to parallelize the Node. For the value tuple: first
            element specifies ParallelTransformKind; second element is an optional
            field specifying how many chunks the node should be split into, if not
            set, fall back to `num_of_chunks`; third element is an optional field
            used for special parallization mode: ParallelTransformKind.SPECIAL.
        num_of_chunks (int): fall back value of how many chunks the node should be
            split into.
        model_parallel_split_alignment (int): The size of the any split will be increased
            to a multiple of `model_parallel_split_alignment`, if possible. If the result
            after aligning the splits is that the new aligned size of a split is larger
            than the original requested, then the final number of resulting splits
            may be less than requested.
            This optionally can increase the size of any model parallel split to
            multiple of the given value.

    Returns:
        A map of nodes to the concat_node that they were replaced with.
    """
    replace_map = {}
    # Used to keep track num of nodes we have successfully transformed.
    # Will raise runtime error if num_processed_nodes is different from size of
    # parallelization_per_node.
    num_processed_nodes = 0

    for curr_node in mod_traced.graph.nodes:
        # Skip if not in parallelization_per_node.
        if curr_node not in parallelization_per_node:
            continue

        assert len(parallelization_per_node[curr_node]) == 3
        # parallelization transform kind for this node.
        parallel_transform_mode = parallelization_per_node[curr_node][0]
        num_processed_nodes += 1

        # Calculate how many pieces we would like to split the node into.
        curr_num_of_chunks = (
            num_of_chunks
            if parallelization_per_node[curr_node][1] is None
            else parallelization_per_node[curr_node][1]
        )
        assert curr_num_of_chunks is not None and curr_num_of_chunks > 0
        if curr_num_of_chunks == 1:
            # We could not split node to less than or equal to 1 chunk. Skip.
            continue

        # Get parallelization_config.
        parallelization_config = (
            parallelization_per_node[curr_node][2]
            if parallel_transform_mode == ParallelTransformKind.SPECIAL
            else _node_to_config(parallel_transform_mode, curr_node)
        )
        assert parallelization_config is not None

        # Generate first_split_node_key and check that split_dims is valid.
        # split_dims is used to communicate what dims to split to
        # _parallelize_and_replace_node func.
        first_split_node_key = ""
        for input_node_key, _ in parallelization_config.split_dims.items():
            if not first_split_node_key:
                first_split_node_key = input_node_key
            assert (
                input_node_key in curr_node.kwargs
            ), f"{input_node_key} not present in kwargs."
            assert isinstance(
                curr_node.kwargs[input_node_key], Node
            ), f"{input_node_key} is not Node."
        # Dispatch run.
        concat_node = _parallelize_and_replace_node(
            mod_traced,
            curr_node,
            curr_num_of_chunks,
            first_split_node_key,
            parallelization_config.split_dims,
            parallelization_config.result_dim,
            model_parallel_split_alignment,
        )
        replace_map[curr_node] = concat_node

    if num_processed_nodes != len(parallelization_per_node):
        raise RuntimeError(
            "Not all Nodes specified in parallelization_per_node were processed."
        )
    # Remove dead nodes.
    mod_traced.graph.eliminate_dead_code()
    # Lint the graph to make sure that it is still valid after parallelization.
    mod_traced.graph.lint()
    return replace_map
