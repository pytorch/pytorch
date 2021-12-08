import torch
import operator

def relu_pattern(x, scale, zero_point, inplace):
    x = x.dequantize()
    x = torch.nn.functional.relu(x, inplace)
    x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
    return x

def relu_replacement(x, scale, zero_point, inplace):
    x = torch.nn.functional.relu(x, inplace)
    return x

def relu_op_pattern(x, scale, zero_point):
    x = x.dequantize()
    x = x.relu()
    x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
    return x

def relu_op_replacement(x, scale, zero_point):
    x = x.relu()
    return x

def relu_inplace_op_pattern(x, scale, zero_point):
    x = x.dequantize()
    x = x.relu_()
    x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
    return x

def relu_inplace_op_replacement(x, scale, zero_point):
    x = x.relu_()
    return x

def relu6_pattern(x, scale, zero_point, inplace=False):
    x = x.dequantize()
    x = torch.nn.functional.relu6(x, inplace=inplace)
    x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
    return x

def relu6_replacement(x, scale, zero_point, inplace=False):
    x = torch.nn.functional.relu6(x, inplace=inplace)
    return x

def hardtanh_pattern(x, scale, zero_point, min_val=-1.0, max_val=1.0, inplace=False):
    x = x.dequantize()
    x = torch.nn.functional.hardtanh(x, min_val=min_val, max_val=max_val, inplace=inplace)
    x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
    return x

def hardtanh_replacement(x, scale, zero_point, min_val=-1.0, max_val=1.0, inplace=False):
    x = torch.nn.functional.hardtanh(x, min_val=min_val, max_val=max_val, inplace=inplace)
    return x

def hardtanh_inplace_pattern(x, scale, zero_point, min_val=-1.0, max_val=1.0):
    x = x.dequantize()
    x = torch.nn.functional.hardtanh_(x, min_val=min_val, max_val=max_val)
    x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
    return x

def hardtanh_inplace_replacement(x, scale, zero_point, min_val=-1.0, max_val=1.0):
    x = torch.nn.functional.hardtanh_(x, min_val=min_val, max_val=max_val)
    return x

def adaptive_avg_pool1d_pattern(x, scale, zero_point, output_size):
    x = x.dequantize()
    x = torch.nn.functional.adaptive_avg_pool1d(x, output_size)
    x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
    return x

def adaptive_avg_pool1d_replacement(x, scale, zero_point, output_size):
    x = torch.nn.functional.adaptive_avg_pool1d(x, output_size)
    return x

def adaptive_avg_pool2d_pattern(x, scale, zero_point, output_size):
    x = x.dequantize()
    x = torch.nn.functional.adaptive_avg_pool2d(x, output_size)
    x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
    return x

def adaptive_avg_pool2d_replacement(x, scale, zero_point, output_size):
    x = torch.nn.functional.adaptive_avg_pool2d(x, output_size)
    return x

def adaptive_avg_pool3d_pattern(x, scale, zero_point, output_size):
    x = x.dequantize()
    x = torch.nn.functional.adaptive_avg_pool3d(x, output_size)
    x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
    return x

def adaptive_avg_pool3d_replacement(x, scale, zero_point, output_size):
    x = torch.nn.functional.adaptive_avg_pool3d(x, output_size)
    return x

def dropout_pattern(x, scale, zero_point, p=0.5, training=True, inplace=False):
    x = x.dequantize()
    x = torch.nn.functional.dropout(x, p=p, training=training, inplace=inplace)
    x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
    return x

def dropout_replacement(x, scale, zero_point, p=0.5, training=True, inplace=False):
    x = torch.nn.functional.dropout(x, p=p, training=training, inplace=inplace)
    return x

def interpolate_pattern(x, scale, zero_point, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None):
    x = x.dequantize()
    x = torch.nn.functional.interpolate(x, size, scale_factor, mode, align_corners, recompute_scale_factor)
    x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
    return x

def interpolate_replacement(x, scale, zero_point, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None):
    x = torch.nn.functional.interpolate(x, size, scale_factor, mode, align_corners, recompute_scale_factor)
    return x

def max_pool1d_pattern(x, scale, zero_point, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
    x = x.dequantize()
    x = torch.nn.functional.MaxPool1d(x, kernel_size, stride, padding, dilation, return_indices, ceil_mode)
    x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
    return x

def max_pool1d_replacement(x, scale, zero_point, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
    x = torch.nn.functional.MaxPool1d(x, kernel_size, stride, padding, dilation, return_indices, ceil_mode)
    return x

def max_pool2d_pattern(x, scale, zero_point, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
    x = x.dequantize()
    x = torch.nn.functional.MaxPool2d(x, kernel_size, stride, padding, dilation, return_indices, ceil_mode)
    x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
    return x

def max_pool2d_replacement(x, scale, zero_point, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
    x = torch.nn.functional.MaxPool2d(x, kernel_size, stride, padding, dilation, return_indices, ceil_mode)
    return x

def max_pool3d_pattern(x, scale, zero_point, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
    x = x.dequantize()
    x = torch.nn.functional.MaxPool3d(x, kernel_size, stride, padding, dilation, return_indices, ceil_mode)
    x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
    return x

def max_pool3d_replacement(x, scale, zero_point, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
    x = torch.nn.functional.MaxPool3d(x, kernel_size, stride, padding, dilation, return_indices, ceil_mode)
    return x

def avg_pool1d_pattern(x, scale, zero_point, kernel_size, stride, padding, ceil_mode, count_include_pad):
    x = x.dequantize()
    x = torch.avg_pool1d(x, kernel_size, stride, padding, ceil_mode, count_include_pad)
    x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
    return x

def avg_pool1d_replacement(x, scale, zero_point, kernel_size, stride, padding, ceil_mode, count_include_pad):
    x = torch.avg_pool1d(x, kernel_size, stride, padding, dilation, return_indices, ceil_mode)
    return x

def avg_pool2d_pattern(x, scale, zero_point, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override):
    x = x.dequantize()
    x = _C._nn.avg_pool2d(x, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)
    x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
    return x

def avg_pool2d_replacement(x, scale, zero_point, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override):
    x = _C._nn.avg_pool2d(x, kernel_size, stride, padding, dilation, return_indices, ceil_mode, divisor_override)
    return x

def avg_pool3d_pattern(x, scale, zero_point, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override):
    x = x.dequantize()
    x = torch._C._nn.avg_pool3d(x, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)
    x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
    return x

def avg_pool3d_replacement(x, scale, zero_point, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override):
    x = torch._C._nn.avg_pool3d(x, kernel_size, stride, padding, dilation, return_indices, ceil_mode, divisor_override)
    return x

def clamp_pattern(x, scale, zero_point, min_val, max_val):
    x = x.dequantize()
    x = torch.clamp(x, min_val, max_val)
    x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
    return x

def clamp_replacement(x, scale, zero_point, min_val, max_val):
    x = torch.clamp(x, min_val, max_val)
    return x

def clamp_op_pattern(x, scale, zero_point, min_val, max_val):
    x = x.dequantize()
    x = x.clamp(min_val, max_val)
    x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
    return x

def clamp_op_replacement(x, scale, zero_point, min_val, max_val):
    x = x.clamp(min_val, max_val)
    return x

def min_pattern(x, scale, zero_point, dim, keepdim):
    x = x.dequantize()
    x = torch.min(x, dim, keepdim)
    x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
    return x

def min_replacement(x, scale, zero_point, dim, keepdim):
    x = torch.min(x, dim, keepdim)
    return x

def max_pattern(x, scale, zero_point, dim, keepdim):
    x = x.dequantize()
    x = torch.max(x, dim, keepdim)
    x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
    return x

def max_replacement(x, scale, zero_point, dim, keepdim):
    x = torch.max(x, dim, keepdim)
    return x

def mean_pattern(x, scale, zero_point, dim, keepdim):
    x = x.dequantize()
    x = torch.mean(x, dim, keepdim)
    x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
    return x

def mean_replacement(x, scale, zero_point, dim, keepdim):
    x = torch.mean(x, dim, keepdim)
    return x

def mean_op_pattern(x, scale, zero_point, dim, keepdim):
    x = x.dequantize()
    x = x.mean(dim, keepdim)
    x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
    return x

def mean_op_replacement(x, scale, zero_point, dim, keepdim):
    x = x.mean(dim, keepdim)
    return x

def flatten_pattern(x, scale, zero_point, start_dim, end_dim):
    x = x.dequantize()
    x = torch.flatten(x, start_dim, end_dim)
    x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
    return x

def flatten_replacement(x, scale, zero_point, start_dim, end_dim):
    x = torch.flatten(x, start_dim, end_dim)
    return x

def floordiv_pattern(x, scale, zero_point, y):
    x = x.dequantize()
    # TODO quantize 2nd input
    x = operator.floordiv(x, y)
    x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
    return x

def floordiv_replacement(x, scale, zero_point, y):
    x = operator.floordiv(x, y)
    return x

#
# Match Filters
#
def second_input_is_scalar(match, pattern_graph, replacement_graph):
    """ check the node that's matched to the second input of the pattern graph
    is a scalar number
    """
    input_idx = 0
    for node in pattern_graph.nodes:
        if node.op == "placeholder":
            if input_idx == 1:
                num_node = node
            input_idx += 1
    if not isinstance(match.nodes_map[num_node], (int, float)):
        return False
    return True

# pattern and replacement for binary_op + relu module
def get_binary_op_mrelu_pttn_and_rplcmnt(binary_op, qbinary_oprelu):
    class BinaryOpReLUPattern(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = torch.nn.ReLU()

        def forward(self, x, y, scale, zero_point):
            y = y.dequantize()
            x = x.dequantize()
            x = binary_op(x, y)
            x = self.relu(x)
            x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
            return x

    class BinaryOpReLUReplacement(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = torch.nn.ReLU()

        def forward(self, x, y, scale, zero_point):
            x = qbinary_oprelu(x, y, scale, zero_point)
            return x

    class BinaryOpScalarReLU1Pattern(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = torch.nn.ReLU()

        def forward(self, x, num, scale, zero_point):
            x = x.dequantize()
            x = binary_op(x, num)
            x = self.relu(x)
            x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
            return x

    class BinaryOpScalarReLU2Pattern(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = torch.nn.ReLU()

        def forward(self, x, num, scale, zero_point):
            x = x.dequantize()
            x = binary_op(num, x)
            x = self.relu(x)
            x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
            return x

    class BinaryOpScalarReLUReplacement(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = torch.nn.ReLU()

        def forward(self, x, num, scale, zero_point):
            x = qbinary_oprelu(x, num)
            return x

    return [
        (BinaryOpReLUPattern(), BinaryOpReLUReplacement(), []),
        (BinaryOpScalarReLU1Pattern(), BinaryOpScalarReLUReplacement(), [second_input_is_scalar]),
        (BinaryOpScalarReLU2Pattern(), BinaryOpScalarReLUReplacement(), [second_input_is_scalar]),
    ]

def get_binary_op_frelu_pttn_and_rplcmnt(binary_op, qbinary_oprelu):

    def binary_op_relu_inplace_pattern(x, y, scale, zero_point):
        y = y.dequantize()
        x = x.dequantize()
        x = binary_op(x, y)
        x = torch.nn.functional.relu(x, inplace=True)
        x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
        return x

    def binary_op_relu_non_inplace_pattern(x, y, scale, zero_point):
        y = y.dequantize()
        x = x.dequantize()
        x = binary_op(x, y)
        x = torch.nn.functional.relu(x, inplace=False)
        x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
        return x

    def binary_op_relu_replacement(x, y, scale, zero_point):
        x = qbinary_oprelu(x, y, scale, zero_point)
        return x

    def binary_op_scalar_relu_1_inplace_pattern(x, num, scale, zero_point):
        x = x.dequantize()
        x = binary_op(x, num)
        x = torch.nn.functional.relu(x, inplace=True)
        x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
        return x

    def binary_op_scalar_relu_1_non_inplace_pattern(x, num, scale, zero_point):
        x = x.dequantize()
        x = binary_op(x, num)
        x = torch.nn.functional.relu(x, inplace=False)
        x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
        return x

    def binary_op_scalar_relu_2_inplace_pattern(x, num, scale, zero_point):
        x = x.dequantize()
        x = binary_op(num, x)
        x = torch.nn.functional.relu(x, inplace=True)
        x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
        return x

    def binary_op_scalar_relu_2_non_inplace_pattern(x, num, scale, zero_point):
        x = x.dequantize()
        x = binary_op(num, x)
        x = torch.nn.functional.relu(x, inplace=False)
        x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
        return x

    def binary_op_scalar_relu_replacement(x, num, scale, zero_point):
        return qbinary_oprelu(x, num)

    return [
        (binary_op_relu_inplace_pattern, binary_op_relu_replacement, []),
        (binary_op_relu_non_inplace_pattern, binary_op_relu_replacement, []),
        (binary_op_scalar_relu_1_inplace_pattern, binary_op_scalar_relu_replacement, [second_input_is_scalar]),
        (binary_op_scalar_relu_1_non_inplace_pattern, binary_op_scalar_relu_replacement, [second_input_is_scalar]),
        (binary_op_scalar_relu_2_inplace_pattern, binary_op_scalar_relu_replacement, [second_input_is_scalar]),
        (binary_op_scalar_relu_2_non_inplace_pattern, binary_op_scalar_relu_replacement, [second_input_is_scalar]),
    ]


def get_binary_op_pttn_and_rplcmnt(binary_op, qbinary_op):

    def binary_op_pattern(x, y, scale, zero_point):
        y = y.dequantize()
        x = x.dequantize()
        x = binary_op(x, y)
        x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
        return x

    def binary_op_replacement(x, y, scale, zero_point):
        x = qbinary_op(x, y, scale, zero_point)
        return x

    def binary_op_scalar_1_pattern(x, num, scale, zero_point):
        x = x.dequantize()
        x = binary_op(x, num)
        x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
        return x

    def binary_op_scalar_2_pattern(x, num, scale, zero_point):
        x = x.dequantize()
        x = binary_op(num, x)
        x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
        return x

    def binary_op_scalar_1_replacement(x, num, scale, zero_point):
        x = qbinary_op(x, num)
        return x

    def binary_op_scalar_2_replacement(x, num, scale, zero_point):
        x = qbinary_op(num, x)
        return x

    return [
        (binary_op_pattern, binary_op_replacement, []),
        (binary_op_scalar_1_pattern, binary_op_scalar_1_replacement, [second_input_is_scalar]),
        (binary_op_scalar_2_pattern, binary_op_scalar_2_replacement, [second_input_is_scalar]),
    ]

def get_binary_op_pattern_and_replacements():
    binary_ops = [operator.add, operator.mul, torch.add, torch.mul]
    binary_op_to_qbinary_op = {
        operator.add: torch.ops.quantized.add,
        operator.mul: torch.ops.quantized.mul,
        torch.add: torch.ops.quantized.add,
        torch.mul: torch.ops.quantized.mul,
    }
    binary_op_to_qbinary_oprelu = {
        operator.add: torch.ops.quantized.add_relu,
        operator.mul: torch.ops.quantized.mul_relu,
        torch.add: torch.ops.quantized.add_relu,
        torch.mul: torch.ops.quantized.mul_relu,
    }
    pattern_and_replacements = []
    for binary_op in binary_ops:
        if binary_op in binary_op_to_qbinary_oprelu:
            pattern_and_replacements.extend(get_binary_op_mrelu_pttn_and_rplcmnt(binary_op, binary_op_to_qbinary_oprelu[binary_op]))
            pattern_and_replacements.extend(get_binary_op_frelu_pttn_and_rplcmnt(binary_op, binary_op_to_qbinary_oprelu[binary_op]))

        if binary_op in binary_op_to_qbinary_op:
            pattern_and_replacements.extend(get_binary_op_pttn_and_rplcmnt(binary_op, binary_op_to_qbinary_op[binary_op]))

    return pattern_and_replacements

def _get_all_patterns_and_replacements():
    return [
        *get_binary_op_pattern_and_replacements(),
        (relu_pattern, relu_replacement, []),
        (relu_op_pattern, relu_op_replacement, []),
        (relu_inplace_op_pattern, relu_inplace_op_replacement, []),
        (relu6_pattern, relu6_replacement, []),
        (adaptive_avg_pool1d_pattern, adaptive_avg_pool1d_replacement, []),
        (adaptive_avg_pool2d_pattern, adaptive_avg_pool2d_replacement, []),
        (adaptive_avg_pool3d_pattern, adaptive_avg_pool3d_replacement, []),
        (hardtanh_pattern, hardtanh_replacement, []),
        (hardtanh_inplace_pattern, hardtanh_inplace_replacement, []),
        (dropout_pattern, dropout_replacement, []),
        (interpolate_pattern, interpolate_replacement, []),
        (max_pool1d_pattern, max_pool1d_replacement, []),
        (max_pool2d_pattern, max_pool2d_replacement, []),
        (max_pool3d_pattern, max_pool3d_replacement, []),
        (avg_pool1d_pattern, avg_pool1d_replacement, []),
        (avg_pool2d_pattern, avg_pool2d_replacement, []),
        (avg_pool3d_pattern, avg_pool3d_replacement, []),
        (clamp_pattern, clamp_replacement, []),
        (clamp_op_pattern, clamp_op_replacement, []),
        (min_pattern, min_replacement, []),
        (max_pattern, max_replacement, []),
        (mean_pattern, mean_replacement, []),
        (mean_op_pattern, mean_op_replacement, []),
        (flatten_pattern, flatten_replacement, []),
        (floordiv_pattern, floordiv_replacement, []),
    ]


# TODO: rename to include match filters
def get_fbgemm_patterns_and_replacements():
    return _get_all_patterns_and_replacements()

def get_qnnpack_patterns_and_replacements():
    return _get_all_patterns_and_replacements()
