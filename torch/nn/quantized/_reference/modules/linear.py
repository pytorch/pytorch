# flake8: noqa: F401
r"""Quantized Reference Modules

This module is in the process of migration to
`torch/ao/nn/quantized/_reference`, and is kept here for
compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
appropriate file under the `torch/ao/nn/quantized/_reference`,
while adding an import statement here.
"""

from torch.ao.nn.quantized._reference.modules.linear import Linear
