import torch
from torch.fx import subgraph_rewriter
from .graph_module import QuantizedGraphModule
from .quantized_fusion_patterns_and_replacements import get_fbgemm_patterns_and_replacements
from .match_utils import is_match
from .match_utils import MatchAllNode
from .utils import _parent_name

def _lower_ref_linear_module(model: QuantizedGraphModule) -> QuantizedGraphModule:
    # traverse the graph and find dequantize - ref quantized linear - quantize patterns and
    # and replace it with quantized linear modules
    pattern = (torch.quantize_per_tensor, (torch.nn.quantized._reference.Linear, "dequantize"), MatchAllNode, MatchAllNode, MatchAllNode)
    modules = dict(model.named_modules())
    nodes = list(model.graph.nodes)
    # TODO: maybe orgnize this better (e.g. break down to more functions)
    # to make this function more readable
    for n in model.graph.nodes:
        if is_match(modules, n, pattern):
            q_node = n
            linear_node = q_node.args[0]
            dq_node = linear_node.args[0]
            if not len(dq_node.users) == 1:
                continue
            if not len(linear_node.users) == 1:
                continue
            # get output scale/zero_point/dtype from the quantize node
            scale_node = q_node.args[1]
            zero_point_node = q_node.args[2]
            dtype = q_node.args[3]

            if scale_node.op != "get_attr" or zero_point_node.op != "get_attr":
                continue

            if dtype != torch.quint8:
                print(f"Only qint8 output for quantized linear is supported, got: {dtype}")
                continue

            # change this pattern to use torch.nn.quantized.Linear
            ref_qlinear = modules[linear_node.target]
            # initialize qlinear with ref_qlinear (https://github.com/pytorch/pytorch/blob/master/torch/nn/quantized/_reference/modules/linear.py)
            qlinear = torch.nn.quantized.Linear(ref_qlinear.in_features, ref_qlinear.out_features)
            qweight = ref_qlinear.get_quantized_weight()
            qlinear.set_weight_bias(qweight, ref_qlinear.bias)

            act_scale = getattr(model, scale_node.target)
            act_zero_point = getattr(model, zero_point_node.target)
            qlinear.scale = float(act_scale)
            qlinear.zero_point = int(act_zero_point)

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
        subgraph_rewriter.replace_pattern(model, pattern, replacement)
    model.graph.lint()
    return model
