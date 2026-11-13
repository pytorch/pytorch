# flake8: noqa: F401
r"""
This file is in the process of migration to `torch/ao/quantization`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
`torch/ao/quantization/quantize_fx.py`, while adding an import statement
here.
"""

from torch.ao.quantization.fx.graph_module import ObservedGraphModule
from torch.ao.quantization.quantize_fx import (
    _check_is_graph_module,
    _convert_fx,
    _convert_standalone_module_fx,
    _fuse_fx,
    _prepare_fx,
    _prepare_standalone_module_fx,
    _swap_ff_with_fxff,
    convert_fx,
    fuse_fx,
    prepare_fx,
    prepare_qat_fx,
    QuantizationTracer,
    Scope,
    ScopeContextManager,
)
