# flake8: noqa: F401
r"""
This file is in the process of migration to `torch/ao/quantization`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
appropriate files under `torch/ao/quantization/fx/`, while adding an import statement
here.
"""
from torch.ao.quantization.fx.quantize_handler import (
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

QuantizeHandler.__module__ = "torch.ao.quantization.fx.quantization_patterns"
BinaryOpQuantizeHandler.__module__ = "torch.ao.quantization.fx.quantization_patterns"
CatQuantizeHandler.__module__ = "torch.ao.quantization.fx.quantization_patterns"
ConvReluQuantizeHandler.__module__ = "torch.ao.quantization.fx.quantization_patterns"
LinearReLUQuantizeHandler.__module__ = "torch.ao.quantization.fx.quantization_patterns"
BatchNormQuantizeHandler.__module__ = "torch.ao.quantization.fx.quantization_patterns"
EmbeddingQuantizeHandler.__module__ = "torch.ao.quantization.fx.quantization_patterns"
RNNDynamicQuantizeHandler.__module__ = "torch.ao.quantization.fx.quantization_patterns"
DefaultNodeQuantizeHandler.__module__ = "torch.ao.quantization.fx.quantization_patterns"
FixedQParamsOpQuantizeHandler.__module__ = "torch.ao.quantization.fx.quantization_patterns"
CopyNodeQuantizeHandler.__module__ = "torch.ao.quantization.fx.quantization_patterns"
CustomModuleQuantizeHandler.__module__ = "torch.ao.quantization.fx.quantization_patterns"
GeneralTensorShapeOpQuantizeHandler.__module__ = "torch.ao.quantization.fx.quantization_patterns"
StandaloneModuleQuantizeHandler.__module__ = "torch.ao.quantization.fx.quantization_patterns"
