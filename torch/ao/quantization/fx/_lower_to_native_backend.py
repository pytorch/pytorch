import torch
from . import subgraph_rewriter_FORKED_DO_NOT_USE
from .graph_module import QuantizedGraphModule
from .quantized_fusion_patterns_and_replacements import get_fbgemm_patterns_and_replacements
from .match_utils import (
    is_match,
    MatchAllNode,
    calculate_module_name_to_num_node_users,
)
from .utils import _parent_name
from .quantization_types import Pattern

def _lower_ref_linear_module(model: QuantizedGraphModule) -> QuantizedGraphModule:
    # traverse the graph and find dequantize - ref quantized linear - quantize patterns and
    # and replace it with quantized linear modules
    pattern: Pattern = (  # type: ignore[assignment]
        torch.quantize_per_tensor,
        (torch.nn.quantized._reference.Linear, "dequantize"),
        MatchAllNode, MatchAllNode, MatchAllNode)
    modules = dict(model.named_modules())
    nodes = list(model.graph.nodes)
    # TODO: maybe orgnize this better (e.g. break down to more functions)
    # to make this function more readable
    module_name_to_num_node_users = \
        calculate_module_name_to_num_node_users(model.graph)
    for n in model.graph.nodes:
        if not is_match(modules, n, pattern, module_name_to_num_node_users):
            continue
        q_node = n
        linear_node = q_node.args[0]
        dq_node = linear_node.args[0]
        # get output scale/zero_point/dtype from the quantize node
        scale_node = q_node.args[1]
        zero_point_node = q_node.args[2]
        dtype = q_node.args[3]

        # this can be removed if we add support for "get_attr" in is_match
        if scale_node.op != "get_attr" or zero_point_node.op != "get_attr":
            print("Find the pattern but scale_node and zero_point node are not `get_attr`,"
                  f"got: {scale_node.format_node} {zero_point_node.format_node()}")
            continue

        # this can be removed if we add support for constants in is_match
        if dtype != torch.quint8:
            print(f"Only qint8 output for quantized linear is supported, got: {dtype}")
            continue

        # change this pattern to use torch.nn.quantized.Linear
        ref_qlinear = modules[linear_node.target]
        # initialize torch.nn.quantized.Linear with torch.nn.quantized._reference.Linear
        output_scale = getattr(model, scale_node.target)
        output_zero_point = getattr(model, zero_point_node.target)
        # TODO: we can get the class from a map in the future and make this
        # configurable by user
        qlinear = torch.nn.quantized.Linear.from_reference(
            ref_qlinear, output_scale, output_zero_point)

        # replace ref_linear with linear
        parent_name, module_name = _parent_name(linear_node.target)
        setattr(modules[parent_name], module_name, qlinear)
        # remvoe dq node:
        dq_node_input = dq_node.args[0]

        dq_node.replace_all_uses_with(dq_node_input)
        model.graph.erase_node(dq_node)

        # remove q node and args:
        q_node.replace_all_uses_with(linear_node)
        model.graph.erase_node(q_node)
        model.graph.erase_node(scale_node)
        model.graph.erase_node(zero_point_node)
    model.recompile()
    return model

def _lower_to_native_backend(model: QuantizedGraphModule) -> QuantizedGraphModule:
    """ Lower a quantized reference model (with reference quantized operator patterns)
    to the native backend in PyTorch (fbgemm/qnnpack), both backends shares the same
    operator signature so they can be lowered with the same function
    """
    model = _lower_ref_linear_module(model)
    model.recompile()
    for pattern, replacement in get_fbgemm_patterns_and_replacements():
        subgraph_rewriter_FORKED_DO_NOT_USE.replace_pattern(model, pattern, replacement)
    model.graph.lint()
    return model
