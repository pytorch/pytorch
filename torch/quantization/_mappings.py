from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.qat as qat
import torch.nn.quantized as nnq

from ._wrappers import QuantStub
from ._wrappers import DeQuantStub

# Map for swapping float module to quantized ones.
_DEFAULT_MODULE_MAPPING = {
    nn.Linear: nnq.Linear,
    nn.ReLU: nnq.ReLU,
    nn.Conv2d: nnq.Conv2d,
    QuantStub: nnq.Quantize,
    DeQuantStub: nnq.DeQuantize,
    # QAT modules:
    qat.Linear: nnq.Linear,
    qat.Conv2d: nnq.Conv2d,
}

# Map for swapping float module to qat modules.
_DEFAULT_QAT_MODULE_MAPPING = {
    nn.Linear: nn.qat.Linear,
    nn.Conv2d: nn.qat.Conv2d,
}
