# flake8: noqa: F401
r"""
This file is in the process of migration to `torch/ao/quantization`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
appropriate files under `torch/ao/quantization/fx/`, while adding an import statement
here.
"""
from torch.ao.quantization.fx.quantization_patterns import (
    QuantizeHandler,
    BinaryOpQuantizeHandler,
    CatQuantizeHandler,
    ConvReluQuantizeHandler,
    LinearReLUQuantizeHandler,
    BatchNormQuantizeHandler,
    EmbeddingQuantizeHandler,
    RNNDynamicQuantizeHandler,
    DefaultNodeQuantizeHandler,
    FixedQParamsOpQuantizeHandler,
    CopyNodeQuantizeHandler,
    CustomModuleQuantizeHandler,
    GeneralTensorShapeOpQuantizeHandler,
    StandaloneModuleQuantizeHandler
)

_NAMESPACE = "torch.quantization.fx.quantization_patterns"
QuantizeHandler.__module__ = _NAMESPACE
BinaryOpQuantizeHandler.__module__ = _NAMESPACE
CatQuantizeHandler.__module__ = _NAMESPACE
ConvReluQuantizeHandler.__module__ = _NAMESPACE
LinearReLUQuantizeHandler.__module__ = _NAMESPACE
BatchNormQuantizeHandler.__module__ = _NAMESPACE
EmbeddingQuantizeHandler.__module__ = _NAMESPACE
RNNDynamicQuantizeHandler.__module__ = _NAMESPACE
DefaultNodeQuantizeHandler.__module__ = _NAMESPACE
FixedQParamsOpQuantizeHandler.__module__ = _NAMESPACE
CopyNodeQuantizeHandler.__module__ = _NAMESPACE
CustomModuleQuantizeHandler.__module__ = _NAMESPACE
GeneralTensorShapeOpQuantizeHandler.__module__ = _NAMESPACE
StandaloneModuleQuantizeHandler.__module__ = _NAMESPACE
