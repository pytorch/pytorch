import torch
from torch import nn
import torch.nn.functional as F
from torch.backends.mkldnn.quantization import functional as mqf
from torch.fx import subgraph_rewriter

# Conv patterns and replacements
# Conv1d
def functional_conv1d_pattern(x, w, b, scale, zero_point):
    x = x.dequantize()
    w = w.dequantize()
    x = F.conv1d(x, w, b)
    x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
    return x

def functional_conv1d_replacement(x, w, b, scale, zero_point):
    return mqf.conv1d_mkldnn(x, w, b)

def functional_conv1d_relu_pattern(x, w, b, scale, zero_point):
    x = x.dequantize()
    w = w.dequantize()
    x = F.conv1d(x, w, b)
    x = F.relu(x)
    x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
    return x

def functional_conv1d_relu_replacement(x, w, b, scale, zero_point):
    return mqf.conv1d_relu_mkldnn(x, w, b)

# Conv2d
def functional_conv2d_pattern(x, w, b, scale, zero_point):
    x = x.dequantize()
    w = w.dequantize()
    x = F.conv2d(x, w, b)
    x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
    return x

def functional_conv2d_replacement(x, w, b, scale, zero_point):
    return mqf.conv2d_mkldnn(x, w, b)

def functional_conv2d_relu_pattern(x, w, b, scale, zero_point):
    x = x.dequantize()
    w = w.dequantize()
    x = F.conv2d(x, w, b)
    x = F.relu(x)
    x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
    return x

def functional_conv2d_relu_replacement(x, w, b, scale, zero_point):
    return mqf.conv2d_relu_mkldnn(x, w, b)

# Conv3d
def functional_conv3d_pattern(x, w, b, scale, zero_point):
    x = x.dequantize()
    w = w.dequantize()
    x = F.conv3d(x, w, b)
    x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
    return x

def functional_conv3d_replacement(x, w, b, scale, zero_point):
    return mqf.conv3d_mkldnn(x, w, b)

def functional_conv3d_relu_pattern(x, w, b, scale, zero_point):
    x = x.dequantize()
    w = w.dequantize()
    x = F.conv3d(x, w, b)
    x = F.relu(x)
    x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
    return x

def functional_conv3d_relu_replacement(x, w, b, scale, zero_point):
    return mqf.conv3d_relu_mkldnn(x, w, b)

# Linear patterns and replacements
def functional_linear_pattern(x, w, b, scale, zero_point):
    x = x.dequantize()
    w = w.dequantize()
    x = F.linear(x, w, b)
    x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
    return x

def functional_linear_replacement(x, w, b, scale, zero_point):
    return mqf.linear_mkldnn(x, w, b)

def functional_linear_relu_pattern(x, w, b, scale, zero_point):
    x = x.dequantize()
    w = w.dequantize()
    x = F.linear(x, w, b)
    x = F.relu(x)
    x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
    return x

def functional_linear_relu_replacement(x, w, b, scale, zero_point):
    return mqf.linear_relu_mkldnn(x, w, b)

#
# Return all pattern-replacement pairs
#
def get_all_patterns_and_replacements():
    return [
        # Conv
        (functional_conv1d_pattern, functional_conv1d_replacement),
        (functional_conv1d_relu_pattern, functional_conv1d_relu_replacement),
        (functional_conv2d_pattern, functional_conv2d_replacement),
        (functional_conv2d_relu_pattern, functional_conv2d_relu_replacement),
        (functional_conv3d_pattern, functional_conv3d_replacement),
        (functional_conv3d_relu_pattern, functional_conv3d_relu_replacement),
        # Linear
        (functional_linear_pattern, functional_linear_replacement),
        (functional_linear_relu_pattern, functional_linear_relu_replacement)
    ]

#
# Lowering functions
#
def lower_functional_ops(model: torch.fx.GraphModule):
    module_dict = dict(model.named_modules())
    for pattern, replacement in get_all_patterns_and_replacements():
        subgraph_rewriter.replace_pattern(model, pattern, replacement)
    model.graph.lint()
    return model

def lower_modules(model: torch.fx.GraphModule):
    return model

#
# API for users
# Lower reference quantized model to MKLDNN backend
#
def lower_to_mkldnn_backend(model: torch.fx.GraphModule):
    model = lower_functional_ops(model)
    model = lower_modules(model)
    return model

