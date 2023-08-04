import torch
from torch.fx import Node
from typing import Optional, Callable
# TODO: move QuantizationConfig here
from .quantizer import (
    QuantizationConfig,
)
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions
# TODO: move these to this file
from torch.ao.quantization.quantizer.utils import (
    _annotate_input_qspec_map,
    _annotate_output_qspec,
    _is_sym_size_node,
    _node_only_used_for_sym_size,
    get_bias_qspec,
    get_input_act_qspec,
    get_output_act_qspec,
    get_weight_qspec,
)

def _mark_nodes_as_annotated(nodes: List[Node]):
    for node in nodes:
        if node is not None:
            if "quantization_annotation" not in node.meta:
                node.meta["quantization_annotation"] = QuantizationAnnotation()
            node.meta["quantization_annotation"]._annotated = True

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
                n
                for n in p.input_nodes
                if not _node_only_used_for_sym_size(n, p.nodes)
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

_OP_TO_ANNOTATOR = {
    "linear": _annotate_linear
}
