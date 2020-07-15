import torch
import torch.nn as nn

from torch.testing._internal.common_quantization import QuantizationTestCase

import torch.quantization._equalize as _equalize

class TestEqualizeEager(QuantizationTestCase):
    def test_scaling_channels(self):
        tensor1 = nn.Conv2d(4,4,2).weight
        tensor2 = nn.Linear(4,4).weight
        output_axis = 0
        input_axis = 1

        mod_tensor1, mod_tensor2 = _equalize.scaling_channels(tensor1, tensor1, output_axis, input_axis)

        output_channel_tensor1 = _equalize.channel_range(mod_tensor1, output_axis)
        input_channel_tensor2 = _equalize.channel_range(mod_tensor2, input_axis)

        # ensuring the channels ranges of tensor1's input is the same as
        # tensor2's output
        self.assertEqual(output_channel_tensor1, input_channel_tensor2)

    def test_equalization(self):
        module1 = nn.Conv2d(3,4,2)
        module1_output_channel_axis = 0
        module2 = nn.Linear(4,4)
        module2_input_channel_axis = 1

        _equalize.cross_layer_equalization(module1, module2)

        mod_tensor1, mod_tensor2 = module1.weight, module2.weight
        output_channel_tensor1 = _equalize.channel_range(mod_tensor1, module1_output_channel_axis)
        input_channel_tensor2 = _equalize.channel_range(mod_tensor2, module2_input_channel_axis)

        self.assertEqual(output_channel_tensor1, input_channel_tensor2)
