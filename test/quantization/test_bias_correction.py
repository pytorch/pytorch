import torch
from torch.testing._internal.common_quantization import QuantizationTestCase
import torch.quantization._correct_bias as _correct_bias

class TestBiasCorrection(QuantizationTestCase):
