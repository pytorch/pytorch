# mypy: allow-untyped-defs
import itertools
import typing
from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple, Optional

import torch
import torch.nn.functional as F
from torch._subclasses import FakeTensor
from torch.ao.quantization.fx.utils import get_new_attr_name_with_prefix
from torch.ao.quantization.pt2e.export_utils import _WrapperModule
from torch.ao.quantization.pt2e.utils import (
    _get_aten_graph_module_for_pattern,
    _is_conv_node,
    _is_conv_transpose_node,
)
from torch.ao.quantization.quantizer import (
    QuantizationAnnotation,
    QuantizationSpec,
    SharedQuantizationSpec,
)
from torch.ao.quantization.quantizer.utils import (
    _annotate_input_qspec_map,
    _annotate_output_qspec,
)
from torch.fx import Node
from torch.fx.passes.utils.matcher_with_name_node_map_utils import (
    SubgraphMatcherWithNameNodeMap,
)
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


# Use Annotated because list[Callable].__module__ is read-only.
OperatorPatternType = typing.Annotated[list[Callable], None]
OperatorPatternType.__module__ = (
    "torch.ao.quantization.quantizer.xnnpack_quantizer_utils"
)

AnnotatorType = Callable[
    [
        torch.fx.GraphModule,
        Optional[QuantizationConfig],
        Optional[Callable[[Node], bool]],
    ],
    Optional[list[list[Node]]],
]
OP_TO_ANNOTATOR: dict[str, AnnotatorType] = {}


def register_annotator(op: str) -> Callable[[AnnotatorType], None]:
    def decorator(annotator: AnnotatorType) -> None:
        OP_TO_ANNOTATOR[op] = annotator

    return decorator


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
    operators: list[OperatorPatternType]


def _is_annotated(nodes: list[Node]):
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


def _mark_nodes_as_annotated(nodes: list[Node]):
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
        None,
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
    assert quantization_spec.dtype == torch.float, (
        "Only float dtype for bias is supported for bias right now"
    )
    return quantization_spec


@register_annotator("linear")
def _annotate_linear(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[list[list[Node]]]:
    annotated_partitions = []
    input_act_qspec = get_input_act_qspec(quantization_config)
    output_act_qspec = get_output_act_qspec(quantization_config)
    weight_qspec = get_weight_qspec(quantization_config)
    bias_qspec = get_bias_qspec(quantization_config)
    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target != torch.ops.aten.linear.default:
            continue
        if filter_fn and not filter_fn(node):
            continue
        act_node = node.args[0]
        weight_node = node.args[1]
        bias_node = None
        if len(node.args) > 2:
            bias_node = node.args[2]

        if _is_annotated([node]) is False:  # type: ignore[list-item]
            _annotate_input_qspec_map(
                node,
                act_node,
                input_act_qspec,
            )
            _annotate_input_qspec_map(
                node,
                weight_node,
                weight_qspec,
            )
            nodes_to_mark_annotated = [node, weight_node]
            if bias_node:
                _annotate_input_qspec_map(
                    node,
                    bias_node,
                    bias_qspec,
                )
                nodes_to_mark_annotated.append(bias_node)
            _annotate_output_qspec(node, output_act_qspec)
            _mark_nodes_as_annotated(nodes_to_mark_annotated)
            annotated_partitions.append(nodes_to_mark_annotated)

    return annotated_partitions


@register_annotator("linear_relu")
def _annotate_linear_relu(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[list[list[Node]]]:
    annotated_partitions = []
    input_act_qspec = get_input_act_qspec(quantization_config)
    output_act_qspec = get_output_act_qspec(quantization_config)
    weight_qspec = get_weight_qspec(quantization_config)
    bias_qspec = get_bias_qspec(quantization_config)
    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target not in [
            torch.ops.aten.relu.default,
            torch.ops.aten.relu_.default,
        ]:
            continue
        relu_node = node
        maybe_linear_node = node.args[0]
        if (
            not isinstance(maybe_linear_node, Node)
            or maybe_linear_node.op != "call_function"
            or maybe_linear_node.target != torch.ops.aten.linear.default
        ):
            continue

        linear_node = maybe_linear_node
        if len(linear_node.users) > 1:
            # if linear node has multiple users, then it can't be fused with relu
            continue

        input_qspec_map = {}
        input_act = linear_node.args[0]
        assert isinstance(input_act, Node)
        input_qspec_map[input_act] = input_act_qspec

        weight = linear_node.args[1]
        assert isinstance(weight, Node)
        input_qspec_map[weight] = weight_qspec

        # adding weight node to the partition as well
        partition = [relu_node, linear_node, weight]
        bias = linear_node.args[2] if len(linear_node.args) > 2 else None
        if isinstance(bias, Node):
            input_qspec_map[bias] = bias_qspec
            partition.append(bias)

        if _is_annotated(partition):
            continue

        if filter_fn and any(not filter_fn(n) for n in partition):
            continue

        linear_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            _annotated=True,
        )
        relu_node.meta["quantization_annotation"] = QuantizationAnnotation(
            output_qspec=output_act_qspec,
            _annotated=True,
        )
        _mark_nodes_as_annotated(partition)
        annotated_partitions.append(partition)
    return annotated_partitions


@register_annotator("conv")
def _annotate_conv(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[list[list[Node]]]:
    annotated_partitions = []
    for n in gm.graph.nodes:
        if n.op != "call_function" or n.target not in [
            torch.ops.aten.conv1d.default,
            torch.ops.aten.conv2d.default,
        ]:
            continue
        conv_node = n

        input_qspec_map = {}
        input_act = conv_node.args[0]
        assert isinstance(input_act, Node)
        input_qspec_map[input_act] = get_input_act_qspec(quantization_config)

        weight = conv_node.args[1]
        assert isinstance(weight, Node)
        input_qspec_map[weight] = get_weight_qspec(quantization_config)

        # adding weight node to the partition as well
        partition = [conv_node, conv_node.args[1]]

        bias = conv_node.args[2] if len(conv_node.args) > 2 else None
        if isinstance(bias, Node):
            input_qspec_map[bias] = get_bias_qspec(quantization_config)
            partition.append(bias)

        if _is_annotated(partition):
            continue

        if filter_fn and any(not filter_fn(n) for n in partition):
            continue

        conv_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=get_output_act_qspec(quantization_config),
            _annotated=True,
        )
        _mark_nodes_as_annotated(partition)
        annotated_partitions.append(partition)
    return annotated_partitions


def _do_annotate_conv_relu(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
    is_conv_transpose: bool = False,
):
    annotated_partitions = []
    for n in gm.graph.nodes:
        if n.op != "call_function" or n.target not in [
            torch.ops.aten.relu.default,
            torch.ops.aten.relu_.default,
        ]:
            continue
        relu_node = n
        maybe_conv_node = n.args[0]

        is_conv_node = _is_conv_transpose_node if is_conv_transpose else _is_conv_node
        if not isinstance(maybe_conv_node, Node) or not is_conv_node(maybe_conv_node):
            continue
        conv_node = maybe_conv_node

        if len(conv_node.users) > 1:
            # relu shouldn't be fuseable to conv if there are other users
            # of convolution
            continue

        input_qspec_map = {}
        input_act = conv_node.args[0]
        assert isinstance(input_act, Node)
        input_qspec_map[input_act] = get_input_act_qspec(quantization_config)

        weight = conv_node.args[1]
        assert isinstance(weight, Node)
        input_qspec_map[weight] = get_weight_qspec(quantization_config)

        # adding weight node to the partition as well
        partition = [relu_node, conv_node, conv_node.args[1]]
        bias = conv_node.args[2] if len(conv_node.args) > 2 else None
        if isinstance(bias, Node):
            input_qspec_map[bias] = get_bias_qspec(quantization_config)
            partition.append(bias)

        # pyrefly: ignore  # bad-argument-type
        if _is_annotated(partition):
            continue

        # pyrefly: ignore  # bad-argument-type
        if filter_fn and any(not filter_fn(n) for n in partition):
            continue

        conv_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map, _annotated=True
        )
        relu_node.meta["quantization_annotation"] = QuantizationAnnotation(
            output_qspec=get_output_act_qspec(quantization_config),  # type: ignore[arg-type]
            _annotated=True,
        )
        # pyrefly: ignore  # bad-argument-type
        _mark_nodes_as_annotated(partition)
        annotated_partitions.append(partition)
    return annotated_partitions


@register_annotator("conv_relu")
def _annotate_conv_relu(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[list[list[Node]]]:
    return _do_annotate_conv_relu(
        gm, quantization_config, filter_fn, is_conv_transpose=False
    )


@register_annotator("conv_transpose_relu")
def _annotate_conv_transpose_relu(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[list[list[Node]]]:
    return _do_annotate_conv_relu(
        gm, quantization_config, filter_fn, is_conv_transpose=True
    )


@register_annotator("conv_bn")
def _annotate_conv_bn(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[list[list[Node]]]:
    """
    Find conv + batchnorm partitions
    Note: This is only used for QAT. In PTQ, batchnorm should already be fused into the conv.
    """
    return _do_annotate_conv_bn(gm, quantization_config, filter_fn, has_relu=False)


@register_annotator("conv_bn_relu")
def _annotate_conv_bn_relu(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[list[list[Node]]]:
    """
    Find conv + batchnorm + relu partitions
    Note: This is only used for QAT. In PTQ, batchnorm should already be fused into the conv.
    """
    return _do_annotate_conv_bn(gm, quantization_config, filter_fn, has_relu=True)


@register_annotator("conv_transpose_bn")
def _annotate_conv_transpose_bn(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[list[list[Node]]]:
    """
    Find conv_transpose + batchnorm partitions
    Note: This is only used for QAT. In PTQ, batchnorm should already be fused into the conv.
    """
    return _do_annotate_conv_bn(
        gm, quantization_config, filter_fn, has_relu=False, is_conv_transpose=True
    )


@register_annotator("conv_transpose_bn_relu")
def _annotate_conv_transpose_bn_relu(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[list[list[Node]]]:
    """
    Find conv_transpose + batchnorm + relu partitions
    Note: This is only used for QAT. In PTQ, batchnorm should already be fused into the conv.
    """
    return _do_annotate_conv_bn(
        gm, quantization_config, filter_fn, has_relu=True, is_conv_transpose=True
    )


def _do_annotate_conv_bn(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]],
    has_relu: bool,
    is_conv_transpose: bool = False,
) -> list[list[Node]]:
    """
    Given a function that takes in a `conv_fn` and returns a conv-bn[-relu] pattern,
    return a list of annotated partitions.

    The output of the pattern must include a dictionary from string name to node
    for the following names: "input", "conv", "weight", "bias", and "output".
    """

    # Example inputs for conv-bn1d patterns
    _conv1d_bn_example_inputs = (
        torch.randn(1, 1, 3),  # x
        torch.randn(1, 1, 1),  # conv_weight
        torch.randn(1),  # conv_bias
        torch.randn(1),  # bn_weight
        torch.randn(1),  # bn_bias
        torch.randn(1),  # bn_running_mean
        torch.randn(1),  # bn_running_var
    )

    # Example inputs for conv-bn2d patterns
    _conv2d_bn_example_inputs = (
        torch.randn(1, 1, 3, 3),  # x
        torch.randn(1, 1, 1, 1),  # conv_weight
        torch.randn(1),  # conv_bias
        torch.randn(1),  # bn_weight
        torch.randn(1),  # bn_bias
        torch.randn(1),  # bn_running_mean
        torch.randn(1),  # bn_running_var
    )

    def get_pattern(conv_fn: Callable, relu_is_inplace: bool):
        def _conv_bn(x, conv_weight, conv_bias, bn_weight, bn_bias, bn_rm, bn_rv):
            conv = conv_fn(x, conv_weight, conv_bias)
            bn = F.batch_norm(conv, bn_rm, bn_rv, bn_weight, bn_bias, training=True)
            if has_relu:
                output = F.relu_(bn) if relu_is_inplace else F.relu(bn)
            else:
                output = bn
            return output, {
                "input": x,
                "conv": conv,
                "weight": conv_weight,
                "bias": conv_bias,
                "output": output,
            }

        return _WrapperModule(_conv_bn)

    # Needed for matching, otherwise the matches gets filtered out due to unused
    # nodes returned by batch norm
    gm.graph.eliminate_dead_code()
    gm.recompile()

    matches = []
    if is_conv_transpose:
        combinations = [
            (F.conv_transpose1d, _conv1d_bn_example_inputs),
            (F.conv_transpose2d, _conv2d_bn_example_inputs),
        ]
    else:
        combinations = [
            (F.conv1d, _conv1d_bn_example_inputs),  # type: ignore[list-item]
            (F.conv2d, _conv2d_bn_example_inputs),  # type: ignore[list-item]
        ]

    # Add `is_cuda` and `relu_is_inplace` dimensions
    combinations = itertools.product(  # type: ignore[assignment]
        combinations,
        [True, False] if torch.cuda.is_available() else [False],  # is_cuda
        [True, False] if has_relu else [False],  # relu_is_inplace
    )

    # Match against all conv dimensions and cuda variants
    for (conv_fn, example_inputs), is_cuda, relu_is_inplace in combinations:  # type: ignore[misc]
        pattern = get_pattern(conv_fn, relu_is_inplace)  # type: ignore[has-type]
        pattern = _get_aten_graph_module_for_pattern(pattern, example_inputs, is_cuda)  # type: ignore[has-type]
        pattern.graph.eliminate_dead_code()
        pattern.recompile()
        matcher = SubgraphMatcherWithNameNodeMap(pattern, ignore_literals=True)
        matches.extend(matcher.match(gm.graph))

    # Annotate nodes returned in the matches
    annotated_partitions = []
    for match in matches:
        name_node_map = match.name_node_map
        input_node = name_node_map["input"]
        conv_node = name_node_map["conv"]
        weight_node = name_node_map["weight"]
        bias_node = name_node_map["bias"]
        output_node = name_node_map["output"]

        # TODO: annotate the uses of input, weight, and bias separately instead
        # of assuming they come from a single conv node. This is not possible today
        # because input may have multiple users, and we can't rely on the conv node
        # always being the first user. This was the case in models with skip
        # connections like resnet18

        # Validate conv args
        if conv_node.args[0] is not input_node:
            raise ValueError("Conv arg did not contain input node ", input_node)
        if conv_node.args[1] is not weight_node:
            raise ValueError("Conv arg did not contain weight node ", weight_node)
        if len(conv_node.args) > 2 and conv_node.args[2] is not bias_node:
            raise ValueError("Conv arg did not contain bias node ", bias_node)

        # Skip if the partition is already annotated or is filtered out by the user
        partition = [conv_node, weight_node]
        if bias_node is not None:
            partition.append(bias_node)
        if _is_annotated(partition):
            continue
        if filter_fn and any(not filter_fn(n) for n in partition):
            continue

        # Annotate conv inputs and pattern output
        input_qspec_map = {}
        input_qspec_map[input_node] = get_input_act_qspec(quantization_config)
        input_qspec_map[weight_node] = get_weight_qspec(quantization_config)
        if bias_node is not None:
            input_qspec_map[bias_node] = get_bias_qspec(quantization_config)
        conv_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            _annotated=True,
        )
        output_node.meta["quantization_annotation"] = QuantizationAnnotation(
            output_qspec=get_output_act_qspec(quantization_config),  # type: ignore[arg-type]
            _annotated=True,
        )
        _mark_nodes_as_annotated(partition)
        annotated_partitions.append(partition)
    return annotated_partitions


@register_annotator("gru_io_only")
def _annotate_gru_io_only(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[list[list[Node]]]:
    gru_partitions = get_source_partitions(gm.graph, [torch.nn.GRU], filter_fn)
    gru_partitions = list(itertools.chain.from_iterable(gru_partitions.values()))
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
        input_act = input_nodes[0]
        input_act_user = next(iter(input_act.users.keys()))
        assert isinstance(input_act, Node)
        assert isinstance(input_act_user, Node)
        input_act_user.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map={
                input_act: get_input_act_qspec(quantization_config),
            },
            _annotated=True,
        )

        hidden_state = input_nodes[1]
        hidden_state_user = next(iter(hidden_state.users.keys()))
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


@register_annotator("adaptive_avg_pool2d")
def _annotate_adaptive_avg_pool2d(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[list[list[Node]]]:
    """Always annotate adaptive_avg_pool2d op"""
    module_partitions = get_source_partitions(
        gm.graph, [torch.nn.AdaptiveAvgPool2d, F.adaptive_avg_pool2d], filter_fn
    )
    partitions = list(itertools.chain.from_iterable(module_partitions.values()))
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


def _is_input_large_scalar(node: Node, gm: torch.fx.GraphModule):
    """Check if input is a large scalar value. So that we can skip quantization for the node
    since histc op (in HistogramObserver) only works for values up to certain upper bound
    """
    if node.op == "get_attr":
        qualified_name = str(node.target)
        module_path, _, name = qualified_name.rpartition(".")
        submod = gm.get_submodule(module_path)
        tensor = getattr(submod, name)
        # torch.histc works until this upper bound
        HISTC_UPPER_BOUND = 3.4028235e15
        return tensor.numel() == 1 and abs(tensor.item()) > HISTC_UPPER_BOUND
    return False


def _is_input_non_float_tensor(node: Node):
    """Check if the input is not a float tensor, so that we can skip quantization for the node
    since observers only works with float Tensors
    """
    if "val" not in node.meta or not isinstance(node.meta["val"], FakeTensor):
        return True
    return node.meta["val"].dtype != torch.float32


@register_annotator("add_relu")
def _annotate_add_relu(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[list[list[Node]]]:
    annotated_partitions = []
    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target not in [
            torch.ops.aten.relu.default,
            torch.ops.aten.relu_.default,
        ]:
            continue
        relu_node = node
        maybe_add = node.args[0]
        if (
            not isinstance(maybe_add, Node)
            or maybe_add.op != "call_function"
            or maybe_add.target
            not in [
                torch.ops.aten.add.Tensor,
                torch.ops.aten.add_.Tensor,
            ]
        ):
            continue

        add_node = maybe_add

        if len(add_node.users) > 1:
            # add can't be fused with ReLU if the result of add is being used
            # else where in the graph
            continue

        partition = [relu_node, add_node]

        if _is_annotated(partition):
            continue

        if filter_fn and any(not filter_fn(n) for n in partition):
            continue

        input_act_qspec = get_input_act_qspec(quantization_config)
        output_act_qspec = get_output_act_qspec(quantization_config)

        input_qspec_map = {}
        input_act0 = add_node.args[0]
        if isinstance(input_act0, Node):
            if _is_input_large_scalar(input_act0, gm):
                continue
            if _is_input_non_float_tensor(input_act0):
                continue
            partition.append(input_act0)
            input_qspec_map[input_act0] = input_act_qspec

        input_act1 = add_node.args[1]
        if isinstance(input_act1, Node):
            if _is_input_large_scalar(input_act1, gm):
                continue
            if _is_input_non_float_tensor(input_act1):
                continue
            partition.append(input_act1)
            input_qspec_map[input_act1] = input_act_qspec

        add_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            _annotated=True,
        )
        relu_node.meta["quantization_annotation"] = QuantizationAnnotation(
            output_qspec=output_act_qspec,
            _annotated=True,
        )
        annotated_partitions.append(partition)
    return annotated_partitions


@register_annotator("add")
def _annotate_add(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[list[list[Node]]]:
    annotated_partitions = []
    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target not in [
            torch.ops.aten.add.Tensor,
            torch.ops.aten.add_.Tensor,
        ]:
            continue
        add_node = node
        partition = [add_node]

        if _is_annotated(partition):
            continue

        if filter_fn and any(not filter_fn(n) for n in partition):
            continue

        input_act_qspec = get_input_act_qspec(quantization_config)
        output_act_qspec = get_output_act_qspec(quantization_config)

        input_qspec_map = {}
        input_act0 = add_node.args[0]
        if isinstance(input_act0, Node):
            if _is_input_large_scalar(input_act0, gm):
                continue
            if _is_input_non_float_tensor(input_act0):
                continue
            input_qspec_map[input_act0] = input_act_qspec
            partition.append(input_act0)

        input_act1 = add_node.args[1]
        if isinstance(input_act1, Node):
            if _is_input_large_scalar(input_act1, gm):
                continue
            if _is_input_non_float_tensor(input_act1):
                continue
            input_qspec_map[input_act1] = input_act_qspec
            partition.append(input_act1)

        add_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=output_act_qspec,
            _annotated=True,
        )
        annotated_partitions.append(partition)
    return annotated_partitions


@register_annotator("mul_relu")
def _annotate_mul_relu(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[list[list[Node]]]:
    annotated_partitions = []
    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target not in [
            torch.ops.aten.relu.default,
            torch.ops.aten.relu_.default,
        ]:
            continue
        relu_node = node
        maybe_mul = node.args[0]
        if (
            not isinstance(maybe_mul, Node)
            or maybe_mul.op != "call_function"
            or maybe_mul.target
            not in [
                torch.ops.aten.mul.Tensor,
                torch.ops.aten.mul_.Tensor,
            ]
        ):
            continue

        mul_node = maybe_mul
        if len(mul_node.users) > 1:
            # mul can't be fused with ReLU if the result of mul is being used
            # else where in the graph
            continue

        partition = [relu_node, mul_node]

        if _is_annotated(partition):
            continue

        if filter_fn and any(not filter_fn(n) for n in partition):
            continue

        input_act_qspec = get_input_act_qspec(quantization_config)
        output_act_qspec = get_output_act_qspec(quantization_config)

        input_qspec_map = {}
        input_act0 = mul_node.args[0]
        if isinstance(input_act0, Node):
            if _is_input_large_scalar(input_act0, gm):
                continue
            if _is_input_non_float_tensor(input_act0):
                continue
            partition.append(input_act0)
            input_qspec_map[input_act0] = input_act_qspec

        input_act1 = mul_node.args[1]
        if isinstance(input_act1, Node):
            if _is_input_large_scalar(input_act1, gm):
                continue
            if _is_input_non_float_tensor(input_act1):
                continue
            partition.append(input_act1)
            input_qspec_map[input_act1] = input_act_qspec

        mul_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            _annotated=True,
        )
        relu_node.meta["quantization_annotation"] = QuantizationAnnotation(
            output_qspec=output_act_qspec,
            _annotated=True,
        )
        annotated_partitions.append(partition)
    return annotated_partitions


@register_annotator("mul")
def _annotate_mul(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[list[list[Node]]]:
    annotated_partitions = []
    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target not in [
            torch.ops.aten.mul.Tensor,
            torch.ops.aten.mul_.Tensor,
        ]:
            continue

        mul_node = node
        partition = [mul_node]
        if _is_annotated(partition):
            continue

        if filter_fn and any(not filter_fn(n) for n in partition):
            continue

        input_act_qspec = get_input_act_qspec(quantization_config)
        output_act_qspec = get_output_act_qspec(quantization_config)

        input_qspec_map = {}
        input_act0 = mul_node.args[0]
        if isinstance(input_act0, Node):
            if _is_input_large_scalar(input_act0, gm):
                continue
            if _is_input_non_float_tensor(input_act0):
                continue
            input_qspec_map[input_act0] = input_act_qspec
            partition.append(input_act0)

        input_act1 = mul_node.args[1]
        if isinstance(input_act1, Node):
            if _is_input_large_scalar(input_act1, gm):
                continue
            if _is_input_non_float_tensor(input_act1):
                continue
            input_qspec_map[input_act1] = input_act_qspec
            partition.append(input_act0)

        mul_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=output_act_qspec,
            _annotated=True,
        )
        annotated_partitions.append(partition)
    return annotated_partitions


# TODO: remove Optional in return type, fix annotated_partitions logic
@register_annotator("cat")
def _annotate_cat(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[list[list[Node]]]:
    cat_partitions = get_source_partitions(gm.graph, [torch.cat], filter_fn)
    cat_partitions = list(itertools.chain.from_iterable(cat_partitions.values()))
    annotated_partitions = []
    for cat_partition in cat_partitions:
        cat_node = cat_partition.output_nodes[0]
        if _is_annotated([cat_node]):
            continue

        if cat_node.target != torch.ops.aten.cat.default:
            # TODO: change this to AnnotationException
            raise Exception(  # noqa: TRY002
                f"Expected cat node: torch.ops.aten.cat.default, but found {cat_node.target}"
                " please check if you are calling the correct capture API"
            )

        annotated_partitions.append(cat_partition.nodes)

        input_act_qspec = get_input_act_qspec(quantization_config)
        inputs = cat_node.args[0]

        input_qspec_map = {}
        input_act0 = inputs[0]  # type: ignore[index]
        if isinstance(input_act0, Node):
            input_qspec_map[input_act0] = input_act_qspec

        shared_with_input0_qspec = SharedQuantizationSpec((input_act0, cat_node))  # type: ignore[arg-type]
        for input_act in inputs[1:]:  # type: ignore[index, union-attr]
            if input_act not in input_qspec_map:
                input_qspec_map[input_act] = shared_with_input0_qspec  # type: ignore[index]

        output_act_qspec = shared_with_input0_qspec

        cat_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=output_act_qspec,
            _annotated=True,
        )
    return annotated_partitions


def _is_share_obs_or_fq_op(op: Callable) -> bool:
    return op in [
        torch.ops.aten.relu.default,
        torch.ops.aten.hardtanh.default,
        torch.ops.aten.hardtanh_.default,
        torch.ops.aten.max_pool2d.default,
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


# TODO: make the list of ops customizable
def _convert_scalars_to_attrs(model: torch.fx.GraphModule) -> torch.fx.GraphModule:
    for n in model.graph.nodes:
        if n.op != "call_function" or n.target not in [
            torch.ops.aten.add.Tensor,
            torch.ops.aten.mul.Tensor,
        ]:
            continue
        args = list(n.args)
        new_args = []
        for i in range(len(args)):
            if isinstance(args[i], torch.fx.Node):
                new_args.append(args[i])
                continue
            prefix = "_tensor_constant_"
            get_new_attr_name = get_new_attr_name_with_prefix(prefix)
            tensor_constant_name = get_new_attr_name(model)
            float_tensor = torch.tensor(float(args[i]))
            model.register_buffer(tensor_constant_name, float_tensor)
            fake_mode = n.meta["val"].fake_mode
            with model.graph.inserting_before(n):
                get_attr_node = model.graph.create_node(
                    "get_attr", tensor_constant_name, (), {}
                )
                get_attr_node.meta["val"] = fake_mode.from_tensor(
                    float_tensor, static_shapes=True
                )
                new_args.append(get_attr_node)
        n.args = tuple(new_args)
    model.recompile()
    return model
