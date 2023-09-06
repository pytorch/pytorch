import itertools
import operator
from dataclasses import dataclass
from typing import Callable, Dict, List, NamedTuple, Optional

import torch
import torch.nn.functional as F
from torch.ao.quantization.pt2e.graph_utils import find_sequential_partitions
from torch.ao.quantization.pt2e.utils import _is_sym_size_node
from torch.ao.quantization.quantizer import (
    QuantizationAnnotation,
    QuantizationSpec,
    QuantizationSpecBase,
    SharedQuantizationSpec,
)

from torch.ao.quantization.quantizer.utils import (
    _annotate_input_qspec_map,
    _annotate_output_qspec,
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
    "OP_TO_ANNOTATOR",
    "propagate_annotation",
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
) -> Optional[List[List[Node]]]:
    module_partitions = get_source_partitions(
        gm.graph, [torch.nn.Linear, torch.nn.functional.linear], filter_fn
    )
    input_act_qspec = get_input_act_qspec(quantization_config)
    output_act_qspec = get_output_act_qspec(quantization_config)
    weight_qspec = get_weight_qspec(quantization_config)
    bias_qspec = get_bias_qspec(quantization_config)
    annotated_partitions = []
    for partitions in module_partitions.values():
        for p in partitions:
            annotated_partitions.append(p.nodes)
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
            weight_node_users = list(
                filter((lambda x: (_is_sym_size_node(x) is False)), weight_node.users)
            )
            weight_use_node_in_p = set(weight_node_users).intersection(set(p.nodes))
            if len(weight_use_node_in_p) != 1:
                raise ValueError(
                    f"Could not find a valid use of act node. All uses {weight_use_node_in_p}"
                )
            weight_use_node = weight_use_node_in_p.pop()

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
                _annotate_input_qspec_map(
                    weight_use_node,
                    weight_node,
                    weight_qspec,
                )
            if _is_annotated([output_node]) is False:
                _annotate_output_qspec(output_node, output_act_qspec)
            nodes_to_mark_annotated = list(p.nodes)
            _mark_nodes_as_annotated(nodes_to_mark_annotated)
    return annotated_partitions


def _annotate_conv2d(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    conv_partitions = get_source_partitions(
        gm.graph, [torch.nn.Conv2d, torch.nn.functional.conv2d], filter_fn
    )
    conv_partitions = list(itertools.chain(*conv_partitions.values()))
    annotated_partitions = []
    for conv_partition in conv_partitions:
        annotated_partitions.append(conv_partition.nodes)
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

        bias = conv_node.args[2] if len(conv_node.args) > 2 else None
        if isinstance(bias, Node):
            input_qspec_map[bias] = get_bias_qspec(quantization_config)

        conv_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=get_output_act_qspec(quantization_config),
            _annotated=True,
        )
    return annotated_partitions


def _annotate_conv2d_relu(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    fused_partitions = find_sequential_partitions(
        gm, [torch.nn.Conv2d, torch.nn.ReLU], filter_fn
    )
    annotated_partitions = []
    for fused_partition in fused_partitions:
        conv_partition, relu_partition = fused_partition
        annotated_partitions.append(conv_partition.nodes + relu_partition.nodes)
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

        bias = conv_node.args[2] if len(conv_node.args) > 2 else None
        if isinstance(bias, Node):
            input_qspec_map[bias] = get_bias_qspec(quantization_config)

        conv_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map, _annotated=True
        )
        relu_node.meta["quantization_annotation"] = QuantizationAnnotation(
            output_qspec=get_output_act_qspec(quantization_config),  # type: ignore[arg-type]
            _annotated=True,
        )
    return annotated_partitions


def _annotate_conv2d_bn(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    """
    Find Conv2d + batchnorm parititions
    Note: This is only used for QAT. In PTQ, batchnorm should already be fused into the conv.
    """
    fused_partitions = find_sequential_partitions(
        gm, [torch.nn.Conv2d, torch.nn.BatchNorm2d], filter_fn
    )
    annotated_partitions = []
    for fused_partition in fused_partitions:
        conv_partition, bn_partition = fused_partition
        annotated_partitions.append(conv_partition.nodes + bn_partition.nodes)
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
    return annotated_partitions


def _annotate_conv2d_bn_relu(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    """
    Find Conv2d + batchnorm + relu parititions
    Note: This is only used for QAT. In PTQ, batchnorm should already be fused into the conv.
    """
    fused_partitions = find_sequential_partitions(
        gm, [torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.ReLU], filter_fn
    )
    annotated_partitions = []
    for fused_partition in fused_partitions:
        conv_partition, bn_partition, relu_partition = fused_partition
        annotated_partitions.append(
            conv_partition.nodes + bn_partition.nodes + relu_partition.nodes
        )
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

        bias = conv_node.args[2] if len(conv_node.args) > 2 else None
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
    return annotated_partitions


def _annotate_gru_io_only(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    gru_partitions = get_source_partitions(gm.graph, [torch.nn.GRU], filter_fn)
    gru_partitions = list(itertools.chain(*gru_partitions.values()))
    annotated_partitions = []
    for gru_partition in gru_partitions:
        annotated_partitions.append(gru_partition.nodes)
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
    return annotated_partitions


def _annotate_max_pool2d(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    module_partitions = get_source_partitions(
        gm.graph, [torch.nn.MaxPool2d, torch.nn.functional.max_pool2d], filter_fn
    )
    maxpool_partitions = list(itertools.chain(*module_partitions.values()))
    annotated_partitions = []
    for maxpool_partition in maxpool_partitions:
        annotated_partitions.append(maxpool_partition.nodes)
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
    return annotated_partitions


def _annotate_adaptive_avg_pool2d(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    """Always annotate adaptive_avg_pool2d op"""
    module_partitions = get_source_partitions(
        gm.graph, [torch.nn.AdaptiveAvgPool2d, F.adaptive_avg_pool2d], filter_fn
    )
    partitions = list(itertools.chain(*module_partitions.values()))
    annotated_partitions = []
    for partition in partitions:
        pool_node = partition.output_nodes[0]
        if (
            pool_node.op != "call_function"
            or pool_node.target != torch.ops.aten.adaptive_avg_pool2d.default
        ):
            raise ValueError(f"{pool_node} is not an aten adaptive_avg_pool2d operator")

        if _is_annotated([pool_node]):
            continue

        annotated_partitions.append(partition.nodes)
        input_act = pool_node.args[0]
        assert isinstance(input_act, Node)

        # only annotate input output sharing operator
        # when the output of the input node is annotated
        if (
            "quantization_annotation" not in input_act.meta
            or not input_act.meta["quantization_annotation"]._annotated
            or input_act.meta["quantization_annotation"].output_qspec is None
        ):
            input_act_qspec = get_input_act_qspec(quantization_config)
        else:
            input_act_qspec = SharedQuantizationSpec(input_act)

        # output sharing with input
        output_act_qspec = SharedQuantizationSpec((input_act, pool_node))
        pool_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map={
                input_act: input_act_qspec,
            },
            output_qspec=output_act_qspec,
            _annotated=True,
        )
    return annotated_partitions


def _annotate_add_relu(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    fused_partitions = find_sequential_partitions(
        gm, [torch.add, torch.nn.ReLU], filter_fn
    )
    annotated_partitions = []
    for fused_partition in fused_partitions:
        add_partition, relu_partition = fused_partition
        annotated_partitions.append(add_partition.nodes + relu_partition.nodes)
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
    return annotated_partitions


def _annotate_add(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    add_partitions = get_source_partitions(
        gm.graph, [operator.add, torch.add, operator.iadd], filter_fn
    )
    add_partitions = list(itertools.chain(*add_partitions.values()))
    annotated_partitions = []
    for add_partition in add_partitions:
        annotated_partitions.append(add_partition.nodes)
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
    return annotated_partitions


def _annotate_mul_relu(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    fused_partitions = find_sequential_partitions(
        gm, [torch.mul, torch.nn.ReLU], filter_fn
    )
    annotated_partitions = []
    for fused_partition in fused_partitions:
        mul_partition, relu_partition = fused_partition
        annotated_partitions.append(mul_partition.nodes + relu_partition.nodes)
        if len(relu_partition.output_nodes) > 1:
            raise ValueError("Relu partition has more than one output node")
        relu_node = relu_partition.output_nodes[0]
        if len(mul_partition.output_nodes) > 1:
            raise ValueError("mul partition has more than one output node")
        mul_node = mul_partition.output_nodes[0]

        if _is_annotated([relu_node, mul_node]):
            continue

        input_act_qspec = get_input_act_qspec(quantization_config)
        output_act_qspec = get_output_act_qspec(quantization_config)

        input_qspec_map = {}
        input_act0 = mul_node.args[0]
        if isinstance(input_act0, Node):
            input_qspec_map[input_act0] = input_act_qspec

        input_act1 = mul_node.args[1]
        if isinstance(input_act1, Node):
            input_qspec_map[input_act1] = input_act_qspec

        mul_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            _annotated=True,
        )
        relu_node.meta["quantization_annotation"] = QuantizationAnnotation(
            output_qspec=output_act_qspec,
            _annotated=True,
        )
    return annotated_partitions


def _annotate_mul(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    mul_partitions = get_source_partitions(
        gm.graph, [operator.mul, torch.mul, operator.imul], filter_fn
    )
    mul_partitions = list(itertools.chain(*mul_partitions.values()))
    annotated_partitions = []
    for mul_partition in mul_partitions:
        annotated_partitions.append(mul_partition.nodes)
        mul_node = mul_partition.output_nodes[0]
        if _is_annotated([mul_node]):
            continue

        input_act_qspec = get_input_act_qspec(quantization_config)
        output_act_qspec = get_output_act_qspec(quantization_config)

        input_qspec_map = {}
        input_act0 = mul_node.args[0]
        if isinstance(input_act0, Node):
            input_qspec_map[input_act0] = input_act_qspec

        input_act1 = mul_node.args[1]
        if isinstance(input_act1, Node):
            input_qspec_map[input_act1] = input_act_qspec

        mul_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=output_act_qspec,
            _annotated=True,
        )
    return annotated_partitions


OP_TO_ANNOTATOR = {
    "linear": _annotate_linear,
    "conv2d": _annotate_conv2d,
    "conv2d_relu": _annotate_conv2d_relu,
    "conv2d_bn": _annotate_conv2d_bn,
    "conv2d_bn_relu": _annotate_conv2d_bn_relu,
    "max_pool2d": _annotate_max_pool2d,
    "add": _annotate_add,
    "add_relu": _annotate_add_relu,
    "mul": _annotate_mul,
    "mul_relu": _annotate_mul_relu,
    "adaptive_avg_pool2d": _annotate_adaptive_avg_pool2d,
    # input output only gru
    "gru_io_only": _annotate_gru_io_only,
}


def _is_share_obs_or_fq_op(op: Callable) -> bool:
    return op in [
        torch.ops.aten.hardtanh.default,
        torch.ops.aten.hardtanh_.default,
        torch.ops.aten.mean.default,
        torch.ops.aten.mean.dim,
        torch.ops.aten.permute.default,
        torch.ops.aten.permute_copy.default,
        torch.ops.aten.squeeze.dim,
        torch.ops.aten.squeeze_copy.dim,
        # TODO: remove?
        torch.ops.aten.adaptive_avg_pool2d.default,
        torch.ops.aten.view_copy.default,
        torch.ops.aten.view.default,
        torch.ops.aten.slice_copy.Tensor,
        torch.ops.aten.flatten.using_ints,
    ]


def propagate_annotation(model: torch.fx.GraphModule) -> None:
    for n in model.graph.nodes:
        if n.op != "call_function" or not _is_share_obs_or_fq_op(n.target):
            continue

        prev_node = n.args[0]
        if not isinstance(prev_node, Node):
            continue

        quantization_annotation = prev_node.meta.get("quantization_annotation", None)
        if not quantization_annotation:
            continue

        output_qspec = quantization_annotation.output_qspec
        if not output_qspec:
            continue

        # make sure current node is not annotated
        if (
            "quantization_annotation" in n.meta
            and n.meta["quantization_annotation"]._annotated
        ):
            continue

        shared_qspec = SharedQuantizationSpec(prev_node)
        # propagate the previous output_qspec to the current node
        n.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map={
                prev_node: shared_qspec,
            },
            output_qspec=shared_qspec,
            _annotated=True,
        )
