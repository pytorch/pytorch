import torch

module_type_list = {
    torch.nn.ReLU,
    torch.nn.ReLU6,
    torch.nn.AdaptiveAvgPool1d,
    torch.nn.AdaptiveAvgPool2d,
    torch.nn.AdaptiveAvgPool3d,
    torch.nn.AvgPool1d,
    torch.nn.AvgPool2d,
    torch.nn.AvgPool3d,
    torch.nn.MaxPool1d,
    torch.nn.MaxPool2d,
    torch.nn.MaxPool3d,
}
func_list = {
    torch.nn.functional.adaptive_avg_pool1d,
    torch.nn.functional.adaptive_avg_pool2d,
    torch.nn.functional.adaptive_avg_pool3d,
    torch.nn.functional.max_pool1d,
    torch.nn.functional.max_pool2d,
    torch.nn.functional.max_pool3d,
    torch.nn.functional.relu,
    torch.nn.functional.hardtanh,
    torch.nn.functional.hardtanh_,
}
method_list = {
    torch.mean,
    'relu',
    'relu_',
}

def is_flow_supported_ops(node, modules):
    is_supported_call_function = node.op == "call_function" and node.target in func_list
    is_supported_call_method = node.op == "call_method" and node.target in method_list
    is_supported_call_module = node.op == "call_module" and type(modules[str(node.target)]) in module_type_list
    return is_supported_call_function, is_supported_call_method, is_supported_call_module
