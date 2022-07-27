from torch.ao.quantization.qconfig import QConfig
from torch.ao.quantization.observer import (
    MinMaxObserver
)
from torch.ao.quantization.fake_quantize import (
    default_symmetric_fake_quant
)
import torch
from torch.ao.quantization.experimental.fake_quantize import APoTFakeQuantize

# uniform activation and weight, b=8 k=2
uniform_qconfig_8bit = QConfig(activation=default_symmetric_fake_quant.with_args(observer=MinMaxObserver, dtype=torch.quint8),
                               weight=default_symmetric_fake_quant.with_args(observer=MinMaxObserver, dtype=torch.qint8))

# uniform activation, APoT weight, b=8 k=2
apot_weight_qconfig_8bit = QConfig(activation=default_symmetric_fake_quant.with_args(observer=MinMaxObserver, dtype=torch.quint8),
                                   weight=APoTFakeQuantize.with_args(b=8, k=2, dtype=torch.qint8))

# APoT activation and uniform weight, b=8 k=2
apot_qconfig_8bit = QConfig(activation=APoTFakeQuantize.with_args(b=8, k=2, dtype=torch.quint8),
                            weight=APoTFakeQuantize.with_args(b=8, k=2, dtype=torch.qint8))

# uniform activation and weight, b=4 k=2
uniform_qconfig_4bit = QConfig(activation=default_symmetric_fake_quant.with_args(quant_min=0,
                                                                                 quant_max=15,
                                                                                 observer=MinMaxObserver,
                                                                                 dtype=torch.quint8),
                               weight=default_symmetric_fake_quant.with_args(observer=MinMaxObserver, dtype=torch.qint8))

# uniform activation, APoT weight, b=4 k=2
apot_weight_qconfig_4bit = QConfig(activation=default_symmetric_fake_quant.with_args(quant_min=0,
                                                                                     quant_max=15,
                                                                                     observer=MinMaxObserver,
                                                                                     dtype=torch.quint8),
                                   weight=APoTFakeQuantize.with_args(b=4, k=2, dtype=torch.qint8))

# APoT activation and uniform weight, b=4 k=2
apot_qconfig_4bit = QConfig(activation=APoTFakeQuantize.with_args(b=4, k=2, dtype=torch.quint8),
                            weight=APoTFakeQuantize.with_args(b=4, k=2, dtype=torch.qint8))
