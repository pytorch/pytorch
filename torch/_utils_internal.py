import torch

# For binary operations (ops that take two tensors)
# This shows which dtype combinations work for the given op
supported = torch._C._get_supported_dtypes_for_binary_op('add')
print(supported)