import torch.nn as nn

from torch.testing._internal.common_quantization import QuantizationTestCase

import torch.quantization._equalize as _equalize

class TestEqualizeEager(QuantizationTestCase):
    def test_scaling_channels(self):
        tensor1 = nn.Conv2d(4, 4, 2).weight
        tensor2 = nn.Linear(4, 4).weight
        output_axis = 0
        input_axis = 1

        mod_tensor1, mod_tensor2 = _equalize.scaling_channels(tensor1, tensor1, output_axis, input_axis)

        output_channel_tensor1 = _equalize.channel_range(mod_tensor1, output_axis)
        input_channel_tensor2 = _equalize.channel_range(mod_tensor2, input_axis)

        # ensuring the channels ranges of tensor1's input is the same as
        # tensor2's output
        self.assertEqual(output_channel_tensor1, input_channel_tensor2)

    def test_cross_layer_equalization(self):
        module1 = nn.Conv2d(3, 4, 2)
        module1_output_channel_axis = 0
        module2 = nn.Linear(4, 4)
        module2_input_channel_axis = 1

        _equalize.cross_layer_equalization(module1, module2)

        mod_tensor1, mod_tensor2 = module1.weight, module2.weight
        output_channel_tensor1 = _equalize.channel_range(mod_tensor1, module1_output_channel_axis)
        input_channel_tensor2 = _equalize.channel_range(mod_tensor2, module2_input_channel_axis)

        self.assertEqual(output_channel_tensor1, input_channel_tensor2)

    def test_convergence_test(self):
        pass

    def test_equalize(self):
        class chain_module(nn.Module):
            def __init__(self):
                """
                In the constructor we instantiate two nn.Linear modules and assign them as
                member variables.
                """
                super(chain_module, self).__init__()
                self.linear1 = nn.Linear(3, 4)
                self.linear2 = nn.Linear(4, 5)
                self.linear3 = nn.Linear(5, 6)

            def forward(self, x):
                """
                In the forward function we accept a Tensor of input data and we must return
                a Tensor of output data. We can use Modules defined in the constructor as
                well as arbitrary operators on Tensors.
                """
                x = self.linear1(x)
                x = self.linear2(x)
                x = self.linear3(x)
                return x
        mod = chain_module()
        equalize(mod, [['linear1','linear2'],['linear2','linear3']], 1e-6)
