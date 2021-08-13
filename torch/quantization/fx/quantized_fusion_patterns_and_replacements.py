import torch

def relu_pattern(x, scale, zero_point):
    x = x.dequantize()
    x = torch.nn.functional.relu(x)
    x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
    return x

def relu_replacement(x, scale, zero_point):
    x = torch.nn.functional.relu(x)
    return x


def _get_all_patterns_and_replacements():
    return [
        (relu_pattern, relu_replacement)
    ]


def get_fbgemm_patterns_and_replacements():
    return _get_all_patterns_and_replacements()

def get_qnnpack_patterns_and_replacements():
    return _get_all_patterns_and_replacements()
