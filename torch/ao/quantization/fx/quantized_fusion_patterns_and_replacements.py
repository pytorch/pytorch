import torch

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

def add_pattern(x, y, scale, zero_point):
    y = y.dequantize()
    x = x.dequantize()
    x = x + y
    x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
    return x

def add_replacement(x, y, scale, zero_point):
    x = torch.ops.quantized.add(x, y, scale, zero_point)
    return x

def add_scalar_pattern(x, num, scale, zero_point):
    x = x.dequantize()
    x = x + num
    x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
    return x

def add_scalar_replacement(x, num, scale, zero_point):
    x = torch.ops.quantized.add(x, num)
    return x

def _get_all_patterns_and_replacements():
    return [
        (relu_inplace_pattern, relu_replacement),
        (relu_non_inplace_pattern, relu_replacement),
        (add_pattern, add_replacement),
        (add_scalar_pattern, add_scalar_replacement),
    ]


def get_fbgemm_patterns_and_replacements():
    return _get_all_patterns_and_replacements()

def get_qnnpack_patterns_and_replacements():
    return _get_all_patterns_and_replacements()
