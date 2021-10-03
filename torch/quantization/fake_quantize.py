# flake8: noqa: F401
r"""
This file is in the process of migration to `torch/ao/quantization`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
`torch/ao/quantization/fake_quantize.py`, while adding an import statement
here.
"""

from torch.ao.quantization.fake_quantize import (
    _is_per_channel,
    _is_per_tensor,
    _is_symmetric_quant,
    FakeQuantizeBase,
    FakeQuantize,
    FixedQParamsFakeQuantize,
    FusedMovingAvgObsFakeQuantize,
    default_fake_quant,
    default_weight_fake_quant,
    default_symmetric_fixed_qparams_fake_quant,
    default_affine_fixed_qparams_fake_quant,
    default_per_channel_weight_fake_quant,
    default_histogram_fake_quant,
    default_fused_act_fake_quant,
    default_fused_wt_fake_quant,
    default_fused_per_channel_wt_fake_quant,
    _is_fake_quant_script_module,
    disable_fake_quant,
    enable_fake_quant,
    disable_observer,
    enable_observer,
)
