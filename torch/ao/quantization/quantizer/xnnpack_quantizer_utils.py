import itertools
import operator
from dataclasses import dataclass
from typing import Callable, Dict, List, NamedTuple, Optional

import torch
import torch.nn.functional as F
from torch.ao.quantization.pt2e.graph_utils import find_sequential_partitions
from torch.ao.quantization.quantizer import (
    QuantizationAnnotation,
    QuantizationSpec,
    QuantizationSpecBase,
    SharedQuantizationSpec,
)

from torch.ao.quantization.quantizer.utils import (
    _annotate_input_qspec_map,
    _annotate_output_qspec,
    _is_sym_size_node,
    _node_only_used_for_sym_size,
)
from torch.fx import Node
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions


__all__ = [
    "OperatorConfig",
    "OperatorPatternType",
    "QuantizationConfig",
    "get_input_act_qspec",
    "get_output_act_qspec",
    "get_weight_qspec",
    "get_bias_qspec",
]


# In the absence of better name, just winging it with QuantizationConfig
@dataclass(eq=True, frozen=True)
class QuantizationConfig:
    input_activation: Optional[QuantizationSpec]
    output_activation: Optional[QuantizationSpec]
    weight: Optional[QuantizationSpec]
    bias: Optional[QuantizationSpec]
    # TODO: remove, since we can use observer_or_fake_quant_ctr to express this
    is_qat: bool = False


OperatorPatternType = List[Callable]
OperatorPatternType.__module__ = (
    "torch.ao.quantization.quantizer.xnnpack_quantizer_utils"
)


class OperatorConfig(NamedTuple):
    # fix List[str] with List[List[Union[nn.Module, FunctionType, BuiltinFunctionType]]]
    # Basically we are mapping a quantization config to some list of patterns.
    # a pattern is defined as a list of nn module, function or builtin function names
    # e.g. [nn.Conv2d, torch.relu, torch.add]
    # We have not resolved whether fusion can be considered internal details of the
    # quantizer hence it does not need communication to user.
    # Note this pattern is not really informative since it does not really
    # tell us the graph structure resulting from the list of ops.
    config: QuantizationConfig
    operators: List[OperatorPatternType]


# make everything private for now
__all__ = ["OP_TO_ANNOTATOR"]


def _is_annotated(nodes: List[Node]):
    """
    Given a list of nodes (that represents an operator pattern),
    check if any of the node is annotated, return True if any of the node
    is annotated, otherwise return False
    """
    annotated = False
    for node in nodes:
        annotated = annotated or (
            "quantization_annotation" in node.meta
            and node.meta["quantization_annotation"]._annotated
        )
    return annotated


def _mark_nodes_as_annotated(nodes: List[Node]):
    for node in nodes:
        if node is not None:
            if "quantization_annotation" not in node.meta:
                node.meta["quantization_annotation"] = QuantizationAnnotation()
            node.meta["quantization_annotation"]._annotated = True


def get_input_act_qspec(quantization_config: Optional[QuantizationConfig]):
    if quantization_config is None:
        return None
    if quantization_config.input_activation is None:
        return None
    quantization_spec: QuantizationSpec = quantization_config.input_activation
    assert quantization_spec.qscheme in [
        torch.per_tensor_affine,
        torch.per_tensor_symmetric,
    ]
    return quantization_spec


def get_output_act_qspec(quantization_config: Optional[QuantizationConfig]):
    if quantization_config is None:
        return None
    if quantization_config.output_activation is None:
        return None
    quantization_spec: QuantizationSpec = quantization_config.output_activation
    assert quantization_spec.qscheme in [
        torch.per_tensor_affine,
        torch.per_tensor_symmetric,
    ]
    return quantization_spec


def get_weight_qspec(quantization_config: Optional[QuantizationConfig]):
    if quantization_config is None:
        return None
    assert quantization_config is not None
    if quantization_config.weight is None:
        return None
    quantization_spec: QuantizationSpec = quantization_config.weight
    if quantization_spec.qscheme not in [
        torch.per_tensor_symmetric,
        torch.per_channel_symmetric,
    ]:
        raise ValueError(
            f"Unsupported quantization_spec {quantization_spec} for weight"
        )
    return quantization_spec


def get_bias_qspec(quantization_config: Optional[QuantizationConfig]):
    if quantization_config is None:
        return None
    assert quantization_config is not None
    if quantization_config.bias is None:
        return None
    quantization_spec: QuantizationSpec = quantization_config.bias
    assert (
        quantization_spec.dtype == torch.float
    ), "Only float dtype for bias is supported for bias right now"
    return quantization_spec


def _annotate_linear(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> None:
    module_partitions = get_source_partitions(
        gm.graph, [torch.nn.Linear, torch.nn.functional.linear], filter_fn
    )
    input_act_qspec = get_input_act_qspec(quantization_config)
    output_act_qspec = get_output_act_qspec(quantization_config)
    weight_qspec = get_weight_qspec(quantization_config)
    bias_qspec = get_bias_qspec(quantization_config)
    for partitions in module_partitions.values():
        for p in partitions:
            act_nodes = [
                n for n in p.input_nodes if not _node_only_used_for_sym_size(n, p.nodes)
            ]
            if len(act_nodes) > 1:
                raise ValueError(
                    f"Multiple activation nodes found for partition {p} {act_nodes}"
                )
            if len(act_nodes) == 0:
                raise ValueError(f"No activation node found for partition {p}")
            act_node = act_nodes[0]
            output_node = p.output_nodes[0]
            weight_node = None
            bias_node = None
            for node in p.params:
                weight_or_bias = getattr(gm, node.target)  # type: ignore[arg-type]
                if weight_or_bias.ndim == 2:  # type: ignore[attr-defined]
                    weight_node = node
                if weight_or_bias.ndim == 1:  # type: ignore[attr-defined]
                    bias_node = node
            if weight_node is None:
                raise ValueError("No weight found in Linear pattern")
            # find use of act node within the matched pattern
            act_use_node = None
            # When doing tracing with dynamic shape, we end up with sym_size nodes
            # This nodes do not need quantization, so skip those.
            # We can also have quant workflow throw exception when sym_size nodes
            # are annotated.
            # This is not specific to linear, so in future diffs we should streamline
            # this.
            act_node_users = list(
                filter((lambda x: (_is_sym_size_node(x) is False)), act_node.users)
            )
            act_use_node_in_p = set(act_node_users).intersection(set(p.nodes))
            if len(act_use_node_in_p) != 1:
                raise ValueError(
                    f"Could not find a valid use of act node. All uses {act_use_node_in_p}"
                )
            act_use_node = act_use_node_in_p.pop()
            if _is_annotated([act_use_node]) is False:  # type: ignore[list-item]
                _annotate_input_qspec_map(
                    act_use_node,
                    act_node,
                    input_act_qspec,
                )
            if bias_node and _is_annotated([bias_node]) is False:
                _annotate_output_qspec(bias_node, bias_qspec)
            if _is_annotated([weight_node]) is False:  # type: ignore[list-item]
                _annotate_output_qspec(weight_node, weight_qspec)
            if _is_annotated([output_node]) is False:
                _annotate_output_qspec(output_node, output_act_qspec)
            nodes_to_mark_annotated = list(p.nodes)
            _mark_nodes_as_annotated(nodes_to_mark_annotated)


def _annotate_conv2d(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> None:
    conv_partitions = get_source_partitions(
        gm.graph, [torch.nn.Conv2d, torch.nn.functional.conv2d], filter_fn
    )
    conv_partitions = list(itertools.chain(*conv_partitions.values()))
    for conv_partition in conv_partitions:
        if len(conv_partition.output_nodes) > 1:
            raise ValueError("conv partition has more than one output node")
        conv_node = conv_partition.output_nodes[0]
        if (
            conv_node.op != "call_function"
            or conv_node.target != torch.ops.aten.conv2d.default
        ):
            raise ValueError(f"{conv_node} is not an aten conv2d operator")
        # skip annotation if it is already annotated
        if _is_annotated([conv_node]):
            continue

        input_qspec_map = {}
        input_act = conv_node.args[0]
        assert isinstance(input_act, Node)
        input_qspec_map[input_act] = get_input_act_qspec(quantization_config)

        weight = conv_node.args[1]
        assert isinstance(weight, Node)
        input_qspec_map[weight] = get_weight_qspec(quantization_config)

        bias = conv_node.args[2]
        if isinstance(bias, Node):
            input_qspec_map[bias] = get_bias_qspec(quantization_config)

        conv_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=get_output_act_qspec(quantization_config),
            _annotated=True,
        )


def _annotate_conv2d_relu(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> None:
    fused_partitions = find_sequential_partitions(
        gm, [torch.nn.Conv2d, torch.nn.ReLU], filter_fn
    )
    for fused_partition in fused_partitions:
        conv_partition, relu_partition = fused_partition
        if len(relu_partition.output_nodes) > 1:
            raise ValueError("Relu partition has more than one output node")
        relu_node = relu_partition.output_nodes[0]
        if len(conv_partition.output_nodes) > 1:
            raise ValueError("conv partition has more than one output node")
        conv_node = conv_partition.output_nodes[0]

        if not isinstance(conv_node, Node):
            raise ValueError(f"{conv_node} is not a Node")
        if (
            conv_node.op != "call_function"
            or conv_node.target != torch.ops.aten.conv2d.default
        ):
            raise ValueError(f"{conv_node} is not an aten conv2d operator")
        if relu_node.op != "call_function" or relu_node.target not in [
            torch.ops.aten.relu.default,
            torch.ops.aten.relu_.default,
        ]:
            raise ValueError(f"{relu_node} is not an aten relu operator")

        if _is_annotated([relu_node, conv_node]):
            continue

        input_qspec_map = {}
        input_act = conv_node.args[0]
        assert isinstance(input_act, Node)
        input_qspec_map[input_act] = get_input_act_qspec(quantization_config)

        weight = conv_node.args[1]
        assert isinstance(weight, Node)
        input_qspec_map[weight] = get_weight_qspec(quantization_config)

        bias = conv_node.args[2]
        if isinstance(bias, Node):
            input_qspec_map[bias] = get_bias_qspec(quantization_config)

        conv_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map, _annotated=True
        )
        relu_node.meta["quantization_annotation"] = QuantizationAnnotation(
            output_qspec=get_output_act_qspec(quantization_config),  # type: ignore[arg-type]
            _annotated=True,
        )


def _annotate_conv2d_bn(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> None:
    """
    Find Conv2d + batchnorm parititions
    Note: This is only used for QAT. In PTQ, batchnorm should already be fused into the conv.
    """
    fused_partitions = find_sequential_partitions(
        gm, [torch.nn.Conv2d, torch.nn.BatchNorm2d], filter_fn
    )
    for fused_partition in fused_partitions:
        conv_partition, bn_partition = fused_partition
        if len(conv_partition.output_nodes) > 1:
            raise ValueError("conv partition has more than one output node")
        conv_node = conv_partition.output_nodes[0]
        conv_node_users = list(conv_node.users.keys())
        if len(conv_node_users) > 1:
            raise ValueError(
                "Conv node must be consumed by BN only for it to be fusable."
            )
        if len(bn_partition.output_nodes) > 1:
            raise ValueError("BatchNorm partition has more than one output node")
        bn_output_node = bn_partition.output_nodes[0]

        if _is_annotated([bn_output_node, conv_node]):
            continue

        input_qspec_map = {}
        input_act = conv_node.args[0]
        assert isinstance(input_act, Node)
        input_qspec_map[input_act] = get_input_act_qspec(quantization_config)

        weight = conv_node.args[1]
        assert isinstance(weight, Node)
        input_qspec_map[weight] = get_weight_qspec(quantization_config)

        bias = conv_node.args[2] if len(conv_node.args) > 2 else None
        if isinstance(bias, Node):
            input_qspec_map[bias] = get_bias_qspec(quantization_config)

        conv_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map, _annotated=True
        )

        bn_output_node.meta["quantization_annotation"] = QuantizationAnnotation(
            output_qspec=get_output_act_qspec(quantization_config),  # type: ignore[arg-type]
            _annotated=True,
        )
        nodes_to_mark_annotated = list(conv_partition.nodes)
        nodes_to_mark_annotated.extend(list(bn_partition.nodes))
        _mark_nodes_as_annotated(nodes_to_mark_annotated)


def _annotate_conv2d_bn_relu(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> None:
    """
    Find Conv2d + batchnorm + relu parititions
    Note: This is only used for QAT. In PTQ, batchnorm should already be fused into the conv.
    """
    fused_partitions = find_sequential_partitions(
        gm, [torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.ReLU], filter_fn
    )
    for fused_partition in fused_partitions:
        conv_partition, bn_partition, relu_partition = fused_partition
        if len(relu_partition.output_nodes) > 1:
            raise ValueError("Relu partition has more than one output node")
        relu_node = relu_partition.output_nodes[0]
        if len(conv_partition.output_nodes) > 1:
            raise ValueError("conv partition has more than one output node")
        conv_node = conv_partition.output_nodes[0]
        conv_node_users = list(conv_node.users.keys())
        if len(conv_node_users) > 1:
            raise ValueError(
                "Conv node must be consumed by BN only for it to be fusable."
            )
        if len(bn_partition.output_nodes) > 1:
            raise ValueError("BatchNorm partition has more than one output node")
        bn_output_node = bn_partition.output_nodes[0]

        if _is_annotated([relu_node, bn_output_node, conv_node]):
            continue

        input_qspec_map = {}
        input_act = conv_node.args[0]
        assert isinstance(input_act, Node)
        input_qspec_map[input_act] = get_input_act_qspec(quantization_config)

        weight = conv_node.args[1]
        assert isinstance(weight, Node)
        input_qspec_map[weight] = get_weight_qspec(quantization_config)

        bias = conv_node.args[2]
        if isinstance(bias, Node):
            input_qspec_map[bias] = get_bias_qspec(quantization_config)

        conv_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map, _annotated=True
        )

        relu_node.meta["quantization_annotation"] = QuantizationAnnotation(
            output_qspec=get_output_act_qspec(quantization_config),  # type: ignore[arg-type]
            _annotated=True,
        )
        nodes_to_mark_annotated = list(conv_partition.nodes)
        nodes_to_mark_annotated.extend(list(bn_partition.nodes))
        nodes_to_mark_annotated.extend(list(relu_partition.nodes))
        _mark_nodes_as_annotated(nodes_to_mark_annotated)


def _annotate_gru_io_only(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> None:
    gru_partitions = get_source_partitions(gm.graph, [torch.nn.GRU], filter_fn)
    gru_partitions = list(itertools.chain(*gru_partitions.values()))
    for gru_partition in gru_partitions:
        output_nodes = gru_partition.output_nodes
        input_nodes = gru_partition.input_nodes
        # skip annotation if it is already annotated
        if _is_annotated(input_nodes + output_nodes):
            continue
        # inside each GRU partition, we should be able to annotate each linear
        # subgraph
        input_qspec_map: Dict[Node, QuantizationSpecBase] = {}
        input_act = input_nodes[0]
        input_act_user = list(input_act.users.keys())[0]
        assert isinstance(input_act, Node)
        assert isinstance(input_act_user, Node)
        input_act_user.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map={
                input_act: get_input_act_qspec(quantization_config),
            },
            _annotated=True,
        )

        hidden_state = input_nodes[1]
        hidden_state_user = list(hidden_state.users.keys())[0]
        assert isinstance(hidden_state, Node)
        assert isinstance(hidden_state_user, Node)
        hidden_state_user.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map={
                hidden_state: get_input_act_qspec(quantization_config),
            },
            _annotated=True,
        )

        assert len(output_nodes) == 2, "expecting GRU to have two outputs"
        for output in output_nodes:
            output.meta["quantization_annotation"] = QuantizationAnnotation(
                output_qspec=get_output_act_qspec(quantization_config),
                _annotated=True,
            )
        nodes_to_mark_annotated = list(gru_partition.nodes)
        _mark_nodes_as_annotated(nodes_to_mark_annotated)


def _annotate_max_pool2d(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> None:
    module_partitions = get_source_partitions(
        gm.graph, [torch.nn.MaxPool2d, torch.nn.functional.max_pool2d], filter_fn
    )
    maxpool_partitions = list(itertools.chain(*module_partitions.values()))
    for maxpool_partition in maxpool_partitions:
        output_node = maxpool_partition.output_nodes[0]
        maxpool_node = None
        for n in maxpool_partition.nodes:
            if n.target == torch.ops.aten.max_pool2d.default:
                maxpool_node = n
        assert (
            maxpool_node is not None
        ), "XNNPACKQuantizer only works with torch.ops.aten.max_pool2d.default, "
        "please make sure you are exporting the model correctly"
        if _is_annotated([output_node, maxpool_node]):  # type: ignore[list-item]
            continue

        input_act = maxpool_node.args[0]  # type: ignore[union-attr]
        assert isinstance(input_act, Node)

        # only annotate maxpool when the output of the input node is annotated
        if (
            "quantization_annotation" not in input_act.meta
            or not input_act.meta["quantization_annotation"]._annotated
            or input_act.meta["quantization_annotation"].output_qspec is None
        ):
            continue
        # input and output of maxpool will share quantization parameter with input of maxpool
        act_qspec = SharedQuantizationSpec(input_act)
        # act_qspec = get_act_qspec(quantization_config)
        maxpool_node.meta["quantization_annotation"] = QuantizationAnnotation(  # type: ignore[union-attr]
            input_qspec_map={
                input_act: act_qspec,
            },
            _annotated=True,
        )
        output_node.meta["quantization_annotation"] = QuantizationAnnotation(
            output_qspec=act_qspec,
            _annotated=True,
        )


def _annotate_input_out_obs_sharing_op(
    op: Callable,
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> None:
    module_partitions = get_source_partitions(gm.graph, [op], filter_fn)
    partitions = list(itertools.chain(*module_partitions.values()))
    for partition in partitions:
        io_obs_sharing_node = partition.output_nodes[0]
        if _is_annotated([io_obs_sharing_node]):
            continue

        input_act = io_obs_sharing_node.args[0]
        assert isinstance(input_act, Node)

        # only annotate input output sharing operator
        # when the output of the input node is annotated
        if (
            "quantization_annotation" not in input_act.meta
            or not input_act.meta["quantization_annotation"]._annotated
            or input_act.meta["quantization_annotation"].output_qspec is None
        ):
            continue

        act_qspec = SharedQuantizationSpec(input_act)
        io_obs_sharing_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map={
                input_act: act_qspec,
            },
            output_qspec=act_qspec,
            _annotated=True,
        )


def _annotate_adaptive_avg_pool2d(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> None:
    _annotate_input_out_obs_sharing_op(
        torch.nn.AdaptiveAvgPool2d, gm, quantization_config, filter_fn
    )
    _annotate_input_out_obs_sharing_op(
        F.adaptive_avg_pool2d, gm, quantization_config, filter_fn
    )


def _annotate_add_relu(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> None:
    fused_partitions = find_sequential_partitions(
        gm, [torch.add, torch.nn.ReLU], filter_fn
    )
    for fused_partition in fused_partitions:
        add_partition, relu_partition = fused_partition
        if len(relu_partition.output_nodes) > 1:
            raise ValueError("Relu partition has more than one output node")
        relu_node = relu_partition.output_nodes[0]
        if len(add_partition.output_nodes) > 1:
            raise ValueError("add partition has more than one output node")
        add_node = add_partition.output_nodes[0]

        if _is_annotated([relu_node, add_node]):
            continue

        input_act_qspec = get_input_act_qspec(quantization_config)
        output_act_qspec = get_output_act_qspec(quantization_config)

        input_qspec_map = {}
        input_act0 = add_node.args[0]
        if isinstance(input_act0, Node):
            input_qspec_map[input_act0] = input_act_qspec

        input_act1 = add_node.args[1]
        if isinstance(input_act1, Node):
            input_qspec_map[input_act1] = input_act_qspec

        add_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            _annotated=True,
        )
        relu_node.meta["quantization_annotation"] = QuantizationAnnotation(
            output_qspec=output_act_qspec,
            _annotated=True,
        )


def _annotate_add(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> None:
    add_partitions = get_source_partitions(
        gm.graph, [operator.add, torch.add], filter_fn
    )
    add_partitions = list(itertools.chain(*add_partitions.values()))
    for add_partition in add_partitions:
        add_node = add_partition.output_nodes[0]
        if _is_annotated([add_node]):
            continue

        input_act_qspec = get_input_act_qspec(quantization_config)
        output_act_qspec = get_output_act_qspec(quantization_config)

        input_qspec_map = {}
        input_act0 = add_node.args[0]
        if isinstance(input_act0, Node):
            input_qspec_map[input_act0] = input_act_qspec

        input_act1 = add_node.args[1]
        if isinstance(input_act1, Node):
            input_qspec_map[input_act1] = input_act_qspec

        add_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=output_act_qspec,
            _annotated=True,
        )


OP_TO_ANNOTATOR = {
    "linear": _annotate_linear,
    "conv2d": _annotate_conv2d,
    "conv2d_relu": _annotate_conv2d_relu,
    "conv2d_bn": _annotate_conv2d_bn,
    "conv2d_bn_relu": _annotate_conv2d_bn_relu,
    "max_pool2d": _annotate_max_pool2d,
    "add": _annotate_add,
    "add_relu": _annotate_add_relu,
    "adaptive_avg_pool2d": _annotate_adaptive_avg_pool2d,
    # input output only gru
    "gru_io_only": _annotate_gru_io_only,
}
