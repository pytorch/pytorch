import torch
import operator

def relu_inplace_pattern(x, scale, zero_point):
    x = x.dequantize()
    x = torch.nn.functional.relu(x, inplace=True)
    x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
    return x

def relu_non_inplace_pattern(x, scale, zero_point):
    x = x.dequantize()
    x = torch.nn.functional.relu(x, inplace=False)
    x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
    return x

def relu_replacement(x, scale, zero_point):
    x = torch.nn.functional.relu(x)
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
        (relu_inplace_pattern, relu_replacement, []),
        (relu_non_inplace_pattern, relu_replacement, []),
    ]


# TODO: rename to include match filters
def get_fbgemm_patterns_and_replacements():
    return _get_all_patterns_and_replacements()

def get_qnnpack_patterns_and_replacements():
    return _get_all_patterns_and_replacements()
