# flake8: noqa: F401
r"""Quantized Modules.

This file is in the process of migration to `torch/ao/nn/quantized`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
appropriate file under the `torch/ao/nn/quantized/modules`,
while adding an import statement here.
"""

from torch.ao.nn.quantized.modules.utils import (
    _hide_packed_params_repr,
    _ntuple_from_first,
    _pair_from_first,
    _quantize_weight,
    WeightedQuantizedModule,
)
