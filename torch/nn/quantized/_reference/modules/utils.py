# flake8: noqa: F401
r"""Quantized Reference Modules

This module is in the process of migration to
`torch/ao/nn/quantized/_reference`, and is kept here for
compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
appropriate file under the `torch/ao/nn/quantized/_reference`,
while adding an import statement here.
"""
from torch.ao.nn.quantized._reference.modules.utils import _quantize_weight
from torch.ao.nn.quantized._reference.modules.utils import _quantize_and_dequantize_weight
from torch.ao.nn.quantized._reference.modules.utils import _save_weight_qparams
from torch.ao.nn.quantized._reference.modules.utils import _get_weight_qparam_keys
from torch.ao.nn.quantized._reference.modules.utils import ReferenceQuantizedModule
