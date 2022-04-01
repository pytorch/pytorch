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

def relu_method_pattern(x, scale, zero_point):
    x = x.dequantize()
    x = x.relu()
    x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
    return x

def relu_method_replacement(x, scale, zero_point):
    x = x.relu()
    return x

def relu_inplace_method_pattern(x, scale, zero_point):
    x = x.dequantize()
    x = x.relu_()
    x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
    return x

def relu_inplace_method_replacement(x, scale, zero_point):
    x = x.relu_()
    return x

def relu6_inplace_pattern(x, scale, zero_point):
    x = x.dequantize()
    x = torch.nn.functional.relu6(x, inplace=True)
    x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
    return x

def relu6_non_inplace_pattern(x, scale, zero_point):
    x = x.dequantize()
    x = torch.nn.functional.relu6(x, inplace=False)
    x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
    return x

def relu6_replacement(x, scale, zero_point):
    x = torch.nn.functional.relu6(x)
    return x


def hardtanh_pattern(x, scale, zero_point):
    x = x.dequantize()
    x = torch.nn.functional.hardtanh(x, inplace=True)
    x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
    return x

def hardtanh_non_inplace_pattern(x, scale, zero_point):
    x = x.dequantize()
    x = torch.nn.functional.hardtanh(x, inplace=False)
    x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
    return x

def hardtanh_replacement(x, scale, zero_point):
    x = torch.nn.functional.hardtanh(x)
    return x

def hardtanh_inplace_pattern(x, scale, zero_point):
    x = x.dequantize()
    x = torch.nn.functional.hardtanh_(x)
    x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
    return x

def hardtanh_inplace_replacement(x, scale, zero_point):
    x = torch.nn.functional.hardtanh_(x)
    return x

def min_pattern(x, scale, zero_point):
    x = x.dequantize()
    x = torch.min(x)
    x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
    return x

def min_replacement(x, scale, zero_point):
    x = torch.min(x)
    return x

def max_pattern(x, scale, zero_point):
    x = x.dequantize()
    x = torch.max(x)
    x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
    return x

def max_replacement(x, scale, zero_point):
    x = torch.max(x)
    return x

def mean_pattern(x, scale, zero_point):
    x = x.dequantize()
    x = torch.mean(x)
    x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
    return x

def mean_replacement(x, scale, zero_point):
    x = torch.mean(x)
    return x

def mean_method_pattern(x, scale, zero_point):
    x = x.dequantize()
    x = x.mean()
    x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
    return x

def mean_method_replacement(x, scale, zero_point):
    x = x.mean()
    return x

def flatten_pattern(x, scale, zero_point):
    x = x.dequantize()
    x = torch.flatten(x)
    x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
    return x

def flatten_replacement(x, scale, zero_point):
    x = torch.flatten(x)
    return x

def _get_all_patterns_and_replacements():
    return [
        (relu_inplace_pattern, relu_replacement),
        (relu_non_inplace_pattern, relu_replacement),
        (relu_method_pattern, relu_method_replacement),
        (relu_inplace_method_pattern, relu_inplace_method_replacement),
        (relu6_inplace_pattern, relu6_replacement),
        (relu6_non_inplace_pattern, relu6_replacement),
        (hardtanh_pattern, hardtanh_replacement),
        (hardtanh_non_inplace_pattern, hardtanh_replacement),
        (hardtanh_inplace_pattern, hardtanh_inplace_replacement),
        (mean_pattern, mean_replacement),
        (mean_method_pattern, mean_method_replacement),
    ]


def get_fbgemm_patterns_and_replacements():
    return _get_all_patterns_and_replacements()

def get_qnnpack_patterns_and_replacements():
    return _get_all_patterns_and_replacements()
