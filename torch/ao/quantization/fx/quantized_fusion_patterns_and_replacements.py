import torch
import operator
import copy

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

# pattern and replacement for bop + relu module
def get_bop_mrelu_pr(bop, qboprelu):
    class BOpReLUPattern(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = torch.nn.ReLU()
            self.scale = torch.tensor([])
            self.zero_point = torch.tensor([])

        def forward(self, x, y):
            y = y.dequantize()
            x = x.dequantize()
            x = bop(x, y)
            x = self.relu(x)
            x = torch.quantize_per_tensor(x, self.scale, self.zero_point, torch.quint8)
            return x

    class BOpReLUReplacement(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = torch.nn.ReLU()
            self.scale = torch.tensor([1])
            self.zero_point = torch.tensor([0])

        def forward(self, x, y):
            x = qboprelu(x, y, self.scale.item(), self.zero_point.item())
            return x

    class BOpScalarReLU1Pattern(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = torch.nn.ReLU()
            self.scale = torch.tensor([])
            self.zero_point = torch.tensor([])

        def forward(self, x, num):
            x = x.dequantize()
            x = bop(x, num)
            x = self.relu(x)
            x = torch.quantize_per_tensor(x, self.scale, self.zero_point, torch.quint8)
            return x

    class BOpScalarReLU2Pattern(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = torch.nn.ReLU()
            self.scale = torch.tensor([])
            self.zero_point = torch.tensor([])

        def forward(self, x, num):
            x = x.dequantize()
            x = bop(num, x)
            x = self.relu(x)
            x = torch.quantize_per_tensor(x, self.scale, self.zero_point, torch.quint8)
            return x

    class BOpScalarReLUReplacement(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = torch.nn.ReLU()
            self.scale = torch.tensor([])
            self.zero_point = torch.tensor([])

        def forward(self, x, num):
            x = qboprelu(x, num)
            return x

    return copy.deepcopy([
        (BOpReLUPattern(), BOpReLUReplacement()),
        (BOpScalarReLU1Pattern(), BOpScalarReLUReplacement()),
        (BOpScalarReLU2Pattern(), BOpScalarReLUReplacement()),
    ])

def get_bop_frelu_pr(bop, qboprelu):

    def bop_relu_inplace_pattern(x, y, scale, zero_point):
        y = y.dequantize()
        x = x.dequantize()
        x = bop(x, y)
        x = torch.nn.functional.relu(x, inplace=True)
        x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
        return x

    def bop_relu_non_inplace_pattern(x, y, scale, zero_point):
        y = y.dequantize()
        x = x.dequantize()
        x = bop(x, y)
        x = torch.nn.functional.relu(x, inplace=False)
        x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
        return x

    def bop_relu_replacement(x, y, scale, zero_point):
        x = qboprelu(x, y, scale, zero_point)
        return x

    def bop_scalar_relu_1_inplace_pattern(x, num, scale, zero_point):
        x = x.dequantize()
        x = bop(x, num)
        x = torch.nn.functional.relu(x, inplace=True)
        x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
        return x

    def bop_scalar_relu_1_non_inplace_pattern(x, num, scale, zero_point):
        x = x.dequantize()
        x = bop(x, num)
        x = torch.nn.functional.relu(x, inplace=False)
        x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
        return x

    def bop_scalar_relu_2_inplace_pattern(x, num, scale, zero_point):
        x = x.dequantize()
        x = bop(num, x)
        x = torch.nn.functional.relu(x, inplace=True)
        x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
        return x

    def bop_scalar_relu_2_non_inplace_pattern(x, num, scale, zero_point):
        x = x.dequantize()
        x = bop(num, x)
        x = torch.nn.functional.relu(x, inplace=False)
        x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
        return x

    def bop_scalar_relu_replacement(x, num, scale, zero_point):
        return qboprelu(x, num)

    return copy.deepcopy([
        (bop_relu_inplace_pattern, bop_relu_replacement),
        (bop_relu_non_inplace_pattern, bop_relu_replacement),
        (bop_scalar_relu_1_inplace_pattern, bop_scalar_relu_replacement),
        (bop_scalar_relu_1_non_inplace_pattern, bop_scalar_relu_replacement),
        (bop_scalar_relu_2_inplace_pattern, bop_scalar_relu_replacement),
        (bop_scalar_relu_2_non_inplace_pattern, bop_scalar_relu_replacement),
    ])


def get_bop_pr(bop, qbop):

    def bop_pattern(x, y, scale, zero_point):
        y = y.dequantize()
        x = x.dequantize()
        x = bop(x, y)
        x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
        return x

    def bop_replacement(x, y, scale, zero_point):
        x = qbop(x, y, scale, zero_point)
        return x

    def bop_scalar_1_pattern(x, num, scale, zero_point):
        x = x.dequantize()
        x = bop(x, num)
        x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
        return x

    def bop_scalar_2_pattern(x, num, scale, zero_point):
        x = x.dequantize()
        x = bop(num, x)
        x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
        return x

    def bop_scalar_replacement(x, num, scale, zero_point):
        x = qbop(x, num)
        return x

    return copy.deepcopy([
        (bop_pattern, bop_replacement),
        (bop_scalar_1_pattern, bop_scalar_replacement),
        (bop_scalar_2_pattern, bop_scalar_replacement),
    ])

def get_binary_op_pattern_and_replacements():
    binary_ops = [operator.add, operator.mul, torch.add, torch.mul]
    bop_to_qbop = {
        operator.add: torch.ops.quantized.add,
        operator.mul: torch.ops.quantized.mul,
        torch.add: torch.ops.quantized.add,
        torch.mul: torch.ops.quantized.mul,
    }
    bop_to_qboprelu = {
        operator.add: torch.ops.quantized.add_relu,
        operator.mul: torch.ops.quantized.mul_relu,
        torch.add: torch.ops.quantized.add_relu,
        torch.mul: torch.ops.quantized.mul_relu,
    }
    pattern_and_replacements = []
    for bop in binary_ops:
        if bop in bop_to_qboprelu:
            pattern_and_replacements.extend(get_bop_mrelu_pr(bop, bop_to_qboprelu[bop]))
            pattern_and_replacements.extend(get_bop_frelu_pr(bop, bop_to_qboprelu[bop]))

        if bop in bop_to_qbop:
            pattern_and_replacements.extend(get_bop_pr(bop, bop_to_qbop[bop]))

    return pattern_and_replacements

def _get_all_patterns_and_replacements():
    return [
        *get_binary_op_pattern_and_replacements(),
        (relu_inplace_pattern, relu_replacement),
        (relu_non_inplace_pattern, relu_replacement),
    ]


def get_fbgemm_patterns_and_replacements():
    return _get_all_patterns_and_replacements()

def get_qnnpack_patterns_and_replacements():
    return _get_all_patterns_and_replacements()
