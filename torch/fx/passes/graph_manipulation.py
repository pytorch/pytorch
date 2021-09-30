from typing import Dict, List, NamedTuple, Any, Optional, Tuple

import torch
from torch.fx.passes.param_fetch import lift_lowering_attrs_to_nodes
from torch.fx.graph import Graph
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node, Target, Argument, map_arg, map_aggregate
from torch.fx.node import _get_qualified_name
from torch.fx.passes.shape_prop import ShapeProp

from torch.fx._compatibility import compatibility


@compatibility(is_backward_compatible=False)
def replace_target_nodes_with(
    fx_module: GraphModule,
    old_op: str,
    old_target: Target,
    new_op: str,
    new_target: Target,
):
    """Modifies all nodes in fx_module.graph.nodes which match the specified op code and target,
    and updates them to match the new op code and target"""
    new_graph = Graph()
    val_map: Dict[Node, Node] = {}
    for node in fx_module.graph.nodes:
        if node.op == old_op and node.target == old_target:
            args = map_arg(node.args, lambda n: val_map[n])
            kwargs = map_arg(node.kwargs, lambda n: val_map[n])
            assert isinstance(args, tuple)
            assert isinstance(kwargs, dict)
            val_map[node] = new_graph.create_node(
                new_op, new_target, args, kwargs, node.name
            )
        else:
            val_map[node] = new_graph.node_copy(node, lambda n: val_map[n])
    fx_module.graph = new_graph

@compatibility(is_backward_compatible=False)
class size_bytes(NamedTuple):
    output_size: int
    total_size: int

@compatibility(is_backward_compatible=False)
def get_size_of_all_nodes(
    fx_module: GraphModule, args: Optional[List[torch.Tensor]] = None
) -> None:
    """Given a fx graph module, update each node with its total size (weights + bias + output)
    and its output_size(output). For a non-module node, the total size is the output size.
    return total size"""
    if args is not None:
        # Mark shape and dtype for each node (node.shape and node.dtype)
        ShapeProp(fx_module).propagate(*args)
    # Calculate the total size of the whole fx graph
    total_size_of_graph = 0.0
    for node in fx_module.graph.nodes:
        if node.op == "output":
            break
        node.size_bytes = get_size_of_node(fx_module, node)
    return

@compatibility(is_backward_compatible=False)
def get_tensor_meta(node: Node) -> Any:
    tensor_meta = node.meta.get("tensor_meta")

    if not tensor_meta:
        raise RuntimeError(
            f"Node {node} has no tensor metadata associated with it! "
            f"Check that shape propagation has run."
        )

    return tensor_meta

@compatibility(is_backward_compatible=False)
def get_size_of_node(fx_module: GraphModule, node: Node) -> size_bytes:
    """Given a node with node.dtype and node.shape, return its total size and its output size.
    total_size = weights + bias + output_size
    """
    # Total num of elements
    total_num_of_elems = 0
    # For a module, conside all parameters
    if node.op == "call_module":
        submodule_dict = dict(fx_module.named_modules())
        submodule = submodule_dict[node.target]
        parameters = submodule.named_parameters()
        # Parameters are named tuples
        for name, p in parameters:
            total_num_of_elems += p.numel()
    # Don't forget the output size
    # node.shape is the shape of this node's output
    tensor_meta = get_tensor_meta(node)
    output_elem = tensor_meta.shape.numel()
    total_num_of_elems += output_elem
    # Assume for now if it's quantized then it's qint8 or quint8
    if tensor_meta.is_quantized:
        size_per_elem_bytes = torch._empty_affine_quantized(
            [], dtype=tensor_meta.dtype
        ).element_size()
    else:
        size_per_elem_bytes = torch.tensor([], dtype=tensor_meta.dtype).element_size()
    total_size = size_per_elem_bytes * total_num_of_elems
    output_size = size_per_elem_bytes * output_elem
    return size_bytes(output_size, total_size)

@compatibility(is_backward_compatible=False)
def serialize_shape(shape: torch.Size) -> str:
    return str(list(shape))

@compatibility(is_backward_compatible=False)
def serialize_stride(stride: Tuple[int]) -> str:
    return str(list(stride))

@compatibility(is_backward_compatible=False)
def serialize_tensor_quantization(
    tensor: torch.Tensor, weights: Dict, pcq_prefix: str
) -> Tuple[Dict, Dict]:
    """
    Args:
        tensor: The tensor from which we try to extract quantization information.
        weights: A dict that contains mapping from name to a tensor value.
        pcq_prefix: A string that we would use later on as prefix for per channel quantization information. This
            usually would be the key that we use to store info of `tensor`.

    Returns:
        scheme: Dict that stores the quantization information of `tensor`.
        per_channel_dict: Dict that stores the information of per_channel_scales and
            per_channel_zero_points of `tensor`. This Will be empty if `tensor` is not
            per channel quantized.

    `tensor` is per tensor quantized:
        scheme: {
            "qscheme": str(tensor.qscheme()),
            "q_scale": tensor.q_scale(),
            "q_zero_point": tensor.q_zero_point(),
        }

    `tensor` is per channel quantized:
        scheme: {
            "qscheme": str(tensor.qscheme()),
            "q_per_channel_scales": {pcq_prefix}_per_channel_scales,
            "q_per_channel_zero_points": {pcq_prefix}_per_channel_zero_points,
            "q_per_channel_axis": tensor.q_per_channel_axis()
        }
        per_channel_dict: {
            {pcq_prefix}_per_channel_scales: {
                "dtype": dtype,
                "shape": shape,
                "is_quantized": is_quantized,
                "stride": stride,
            }
            {pcq_prefix}_per_channel_zero_points: {
                "dtype": dtype,
                "shape": shape,
                "is_quantized": is_quantized,
                "stride": stride,
            }
        }
        weights would be updated with {
            {pcq_prefix}_per_channel_scales: tensor.q_per_channel_scales().float()
            {pcq_prefix}_per_channel_zero_points: tensor.q_per_channel_zero_points().int()
        }
    """
    scheme: Dict[str, Any] = {}
    per_channel_dict: Dict[str, Dict] = {}

    if not tensor.is_quantized:
        return scheme, per_channel_dict

    scheme["qscheme"] = str(tensor.qscheme())

    # For per tensor scheme, we stores scale and zero_point.
    if tensor.qscheme() in {torch.per_tensor_affine, torch.per_tensor_symmetric}:
        scheme["q_scale"] = tensor.q_scale()
        scheme["q_zero_point"] = tensor.q_zero_point()

    # For per channel scheme, per_channel_scales and per_channel_zero_points are tensors.
    # We store their tensor value into `weights` and store the name into `scheme`.
    if tensor.qscheme() in {
        torch.per_channel_affine,
        torch.per_channel_affine_float_qparams,
        torch.per_channel_symmetric,
    }:
        # per_channel_scales is float64. Here we save it as float32.
        weights[
            f"{pcq_prefix}_per_channel_scales"
        ] = tensor.q_per_channel_scales().float()
        scheme["q_per_channel_scales"] = f"{pcq_prefix}_per_channel_scales"
        per_channel_dict.update(
            serialize_weight(
                weights[f"{pcq_prefix}_per_channel_scales"],
                weights,
                f"{pcq_prefix}_per_channel_scales",
            )
        )

        # per_channel_zero_point is int64. Here we save it as int32.
        weights[
            f"{pcq_prefix}_per_channel_zero_points"
        ] = tensor.q_per_channel_zero_points().int()
        scheme["q_per_channel_zero_points"] = f"{pcq_prefix}_per_channel_zero_points"
        per_channel_dict.update(
            serialize_weight(
                weights[f"{pcq_prefix}_per_channel_zero_points"],
                weights,
                f"{pcq_prefix}_per_channel_zero_points",
            )
        )

        scheme["q_per_channel_axis"] = tensor.q_per_channel_axis()
    return scheme, per_channel_dict

@compatibility(is_backward_compatible=False)
def serialize_weight(tensor: torch.Tensor, weights: Dict, name: str) -> Dict:
    weight_dict: Dict[str, Dict] = {name: {}}
    weight_dict[name]["dtype"] = str(tensor.dtype)
    weight_dict[name]["shape"] = serialize_shape(tensor.shape)
    weight_dict[name]["requires_grad"] = str(tensor.requires_grad)
    weight_dict[name]["is_quantized"] = tensor.is_quantized
    weight_dict[name]["stride"] = serialize_stride(tensor.stride())

    if tensor.is_quantized:
        quantization_info, per_channel_dict = serialize_tensor_quantization(
            tensor, weights, name
        )
        weight_dict[name].update(quantization_info)
        weight_dict.update(per_channel_dict)

    return weight_dict

@compatibility(is_backward_compatible=False)
def serialize_leaf_module(
    node: Node, weights_metadata: Dict, weights: Dict, name_prefix: str
) -> Dict:
    parameters: Dict[str, Any] = {}

    for p_name, p_value in node.attrs_for_lowering.items():  # type: ignore[attr-defined]
        if isinstance(p_value, torch.Tensor):
            weights_metadata.update(
                serialize_weight(p_value, weights, f"{name_prefix}.{p_name}")
            )
            weights[f"{name_prefix}.{p_name}"] = p_value
        else:
            parameters[p_name] = str(p_value)

    return parameters

@compatibility(is_backward_compatible=False)
def serialize_module(fx_module: GraphModule, weights: Dict, name_prefix="") -> Dict:
    """Recursively Serializes a graph module (fx_module) to a dictionary which is later exported to JSON.
    It also adds all weights the provided weights dictionary by qualified_name.
    Dictionary Schema:
    MODULE
    {
        modules: {module_name: MODULE],
        nodes: [NODE],
        weights {qualified_name: WEIGHT},
    }
    NODE
    {
        shape: [],
        stride: [],
        dtype: dtype,
        is_quantized: bool,
        target: target,
        op_code: op_code,
        name: name,
        args: [],
        kwargs: {}
    }
    WEIGHT
    {
        dtype: dtype,
        is_quantized: bool,
        shape: [],
        QUANTIZATION,
    }
    QUANTIZATION
    {
        qscheme: qscheme,
        q_scale: float,
        q_zero_point: float,
        q_per_channel_scales, [],
        q_per_channel_zero_points: [],
        q_per_channel_axis, int
    }
    """
    serialized_dict: Dict[str, Any] = {}
    serialized_dict["modules"] = {}
    serialized_dict["weights"] = {}
    serialized_dict["nodes"] = []
    submodules = dict(fx_module.named_modules())
    prefix = f"{name_prefix}." if name_prefix else ""

    def add_weight_tensors(named_tensors):
        for name, p in named_tensors:
            if name.startswith("parent.") or not isinstance(p, torch.Tensor):
                continue
            weight_dict = serialize_weight(p, weights, prefix + name)
            serialized_dict["weights"].update(weight_dict)
            weights[prefix + name] = p

    add_weight_tensors(fx_module.named_parameters())
    add_weight_tensors(fx_module.named_buffers())

    def get_node_info(node):
        tensor_meta = get_tensor_meta(node)
        node_rep = {
            "shape": serialize_shape(tensor_meta.shape),
            "dtype": str(tensor_meta.dtype),
            "requires_grad": str(tensor_meta.requires_grad),
            "stride": serialize_stride(tensor_meta.stride),
            "is_quantized": tensor_meta.is_quantized,
        }

        if tensor_meta.is_quantized:
            node_rep["qscheme"] = str(tensor_meta.qscheme)

            if tensor_meta.qscheme in {
                torch.per_tensor_affine,
                torch.per_tensor_symmetric,
            }:
                node_rep["q_scale"] = tensor_meta.q_scale
                node_rep["q_zero_point"] = tensor_meta.q_zero_point

        # Add all extra lowering_info that was provided in node.meta.
        lowering_info = node.meta.get("lowering_info")
        if lowering_info is not None:
            overlapping_keys = node_rep.keys() & lowering_info.keys()
            assert (
                len(overlapping_keys) == 0
            ), f"Overlap found between lowering_info and node_rep: {overlapping_keys}"
            node_rep.update(lowering_info)

        return node_rep

    # Note: lift_lowering_attrs_to_nodes is only used to support leaf modules
    # that cannot currently be symbolically traced into, e.g. batch norm.
    lift_lowering_attrs_to_nodes(fx_module)
    for node in fx_module.graph.nodes:
        node_rep: Dict[str, Any] = {}
        # Get shape/type info, currently not needed for call_module node
        # whose target is a GraphModule and output node.
        if (
            not (
                node.op == "call_module"
                and isinstance(submodules[node.target], GraphModule)
            )
            and node.op != "output"
        ):
            node_rep.update(get_node_info(node))

        # Recurse down into any submodules we are calling.
        if node.op == "call_module":
            if isinstance(submodules[node.target], GraphModule):
                serialized_module = serialize_module(
                    getattr(fx_module, node.target), weights, node.target
                )
                serialized_dict["modules"][node.target] = serialized_module
            else:
                node_rep["parameters"] = serialize_leaf_module(
                    node,
                    serialized_dict["weights"],
                    weights,
                    prefix + node.target,
                )

        if node.op == "call_function":
            node_rep["target"] = _get_qualified_name(node.target)
        else:
            node_rep["target"] = str(node.target)

        # Make sure we capture all constants.
        if node.op == "get_attr":
            # If we are targeting a parent constant we update the target.
            if node.target.startswith("parent."):
                stripped_name = node.target[len("parent.") :]
                node.name = stripped_name
                node_rep["target"] = stripped_name
                weight = serialize_weight(
                    weights[stripped_name], weights, node.target[len("parent.") :]
                )
                # For quantized embedding tables we need to update the shape/type,
                # so we check if the users of this get_attr is a quantized EB and this is the weight for the EB.
                user_targets = {
                    _get_qualified_name(n.target)
                    .replace("torch.fx.experimental.fx_acc.", "")
                    .replace("glow.fb.fx.", ""): n
                    for n in node.users.keys()
                }
                if (
                    "acc_ops.embedding_bag_byte_rowwise_offsets" in user_targets
                    and str(
                        user_targets[
                            "acc_ops.embedding_bag_byte_rowwise_offsets"
                        ].kwargs["weight"]
                    )
                    == stripped_name
                ):
                    weight[stripped_name]["dtype"] = "acc.uint8fused"
                # Same as above, but for the 4 bit version.
                if (
                    "acc_ops.embedding_bag_4bit_rowwise_offsets" in user_targets
                    and str(
                        user_targets[
                            "acc_ops.embedding_bag_4bit_rowwise_offsets"
                        ].kwargs["weight"]
                    )
                    == stripped_name
                ):
                    weight[stripped_name]["dtype"] = "acc.uint4fused"

                serialized_dict["weights"].update(weight)
            else:
                # Find the actual target parameter/buffer from the fx_module.
                submod_path, _, target_name = node.target.rpartition(".")
                submod: Optional[torch.nn.Module] = (
                    fx_module.get_submodule(submod_path) if submod_path else fx_module
                )
                assert submod is not None, f"submod {submod_path} not found"
                target = getattr(submod, target_name, None)
                assert target is not None, f"{target_name} not an attr of {submod_path}"
                qualname = prefix + node.target
                # Check that the target is a tensor, and that we haven't added it already from a leaf module.
                if isinstance(target, torch.Tensor) and qualname not in weights:
                    weight = serialize_weight(target, weights, qualname)
                    serialized_dict["weights"].update(weight)
                    weights[qualname] = target

        node_rep["op_code"] = node.op
        node_rep["name"] = node.name

        def get_user_info(user_node: Argument) -> Any:
            return {"is_node": True, "name": str(user_node)}

        def get_arg_info(arg: Argument) -> Any:
            if isinstance(arg, torch.fx.Node):
                return {"is_node": True, "name": str(arg)}
            elif isinstance(arg, (torch.dtype, torch.memory_format, torch.qscheme)):
                return str(arg)
            else:
                return arg

        def get_output_arg_info(arg: Node) -> Dict[str, Any]:
            node_rep: Dict[str, Any] = get_arg_info(arg)
            node_rep.update(get_node_info(arg))
            return node_rep

        if node.op == "output":
            node_rep["args"] = map_arg(
                node.args,
                get_output_arg_info,
            )

            # If there're multiple outputs then node_rep["args"][0] will be a tuple.
            # In this case we want to unpack the tuple.
            if isinstance(node_rep["args"][0], tuple):
                node_rep["args"] = node_rep["args"][0]
        else:
            node_rep["args"] = map_aggregate(node.args, get_arg_info)

        node_rep["kwargs"] = map_aggregate(node.kwargs, get_arg_info)
        node_rep["users"] = map_aggregate(list(node.users.keys()), get_user_info)
        serialized_dict["nodes"] += [node_rep]

    return serialized_dict
