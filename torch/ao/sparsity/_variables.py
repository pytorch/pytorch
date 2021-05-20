
from torch import ao
from torch import nn

_static_sparse_quantized_mapping = dict([
  nn.Linear: ao.nn.sparse.quantized.Linear,
])

_dynamic_sparse_quantized_mapping = dict([
  nn.Linear: ao.nn.sparse.quantized.dynamic.Linear,
])


def get_static_sparse_quantized_mapping():
  return copy.deepcopy(_static_sparse_quantized_mapping)

def get_dynamic_sparse_quantized_mapping():
  return copy.deepcopy(_dynamic_sparse_quantized_mapping)
