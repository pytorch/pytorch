import torch
import torch.nn.functional as F
from torch.backends.mkldnn.quantization import functional as mqf
from torch.fx import subgraph_rewriter

functional_patterns_and_replacements = []

def register_pattern_conv_linear(op, replacement):
    def op_pattern(x, w, b, scale, zero_point):
        x = x.dequantize()
        w = w.dequantize()
        x = op(x, w, b)
        x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
        return x

    def op_replacement(x, w, b, scale, zero_point):
        return replacement(x, w, b)
    functional_patterns_and_replacements.append((op_pattern, op_replacement))

def register_pattern_conv_linear_relu(op, replacement):
    def op_pattern(x, w, b, scale, zero_point):
        x = x.dequantize()
        w = w.dequantize()
        x = op(x, w, b)
        x = F.relu(x)
        x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
        return x

    def op_replacement(x, w, b, scale, zero_point):
        return replacement(x, w, b)
    functional_patterns_and_replacements.append((op_pattern, op_replacement))

register_pattern_conv_linear(F.conv1d, mqf.conv1d_mkldnn)
register_pattern_conv_linear(F.conv2d, mqf.conv2d_mkldnn)
register_pattern_conv_linear(F.conv3d, mqf.conv3d_mkldnn)
register_pattern_conv_linear(F.linear, mqf.linear_mkldnn)
register_pattern_conv_linear_relu(F.conv1d, mqf.conv1d_relu_mkldnn)
register_pattern_conv_linear_relu(F.conv2d, mqf.conv2d_relu_mkldnn)
register_pattern_conv_linear_relu(F.conv3d, mqf.conv3d_relu_mkldnn)
register_pattern_conv_linear_relu(F.linear, mqf.linear_relu_mkldnn)

def get_all_patterns_and_replacements():
    return functional_patterns_and_replacements

# Lowering functions
def lower_functional_ops(model: torch.fx.GraphModule):
    module_dict = dict(model.named_modules())
    for pattern, replacement in get_all_patterns_and_replacements():
        subgraph_rewriter.replace_pattern(model, pattern, replacement)
    model.graph.lint()
    return model

def lower_modules(model: torch.fx.GraphModule):
    return model

# API for users
# Lower reference quantized model to MKLDNN backend
def lower_to_mkldnn_backend(model: torch.fx.GraphModule):
    model = lower_functional_ops(model)
    model = lower_modules(model)
    return model
