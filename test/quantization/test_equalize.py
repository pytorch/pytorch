import torch
import torch.nn as nn

from torch.testing._internal.common_quantization import QuantizationTestCase

import torch.quantization._equalize as _equalize

import copy

class TestEqualizeEager(QuantizationTestCase):
    def channels_equalized_test(self, tensor1, tensor2, output_axis, input_axis):
        output_channel_tensor1 = _equalize.channel_range(tensor1, output_axis)
        input_channel_tensor2 = _equalize.channel_range(tensor2, input_axis)

        # ensuring the channels ranges of tensor1's input is the same as
        # tensor2's output
        self.assertEqual(output_channel_tensor1, input_channel_tensor2)

    def get_module(self, model, name):
        curr = model
        name = name.split('.')
        for subname in name:
            curr = curr._modules[subname]
        return curr

    def test_cross_layer_equalization(self):
        module1 = nn.Conv2d(3, 4, 2)
        module2 = nn.Linear(4, 4)

        module1_output_channel_axis = 0
        module2_input_channel_axis = 1

        _equalize.cross_layer_equalization(module1, module2)

        mod_tensor1, mod_tensor2 = module1.weight, module2.weight

        self.channels_equalized_test(mod_tensor1, mod_tensor2, module1_output_channel_axis, module2_input_channel_axis)

    def test_converged(self):
        module1 = nn.Linear(3, 3)
        module2 = nn.Linear(3, 3)

        module1.weight = nn.parameter.Parameter(torch.ones(module1.weight.size()))
        module2.weight = nn.parameter.Parameter(torch.zeros(module1.weight.size()))

        # input is a dictionary
        dictionary_1 = {'linear1': module1}
        dictionary_2 = {'linear1': module2}
        self.assertTrue(_equalize.converged(dictionary_1, dictionary_1, 1e-6))
        self.assertFalse(_equalize.converged(dictionary_1, dictionary_2, 1e-6))

    def test_equalize(self):
        class ChainModule(nn.Module):
            def __init__(self):
                """
                In the constructor we instantiate two nn.Linear modules and assign them as
                member variables.
                """
                super(ChainModule, self).__init__()
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
        chain1 = ChainModule()
        chain2 = copy.deepcopy(chain1)

        _equalize.equalize(chain1, [['linear1', 'linear2'], ['linear2', 'linear3']], 1e-6)
        linear1 = self.get_module(chain1, 'linear1')
        linear2 = self.get_module(chain1, 'linear2')
        linear3 = self.get_module(chain1, 'linear3')

        self.channels_equalized_test(linear1.weight, linear2.weight, 0, 1)
        self.channels_equalized_test(linear2.weight, linear3.weight, 0, 1)

        input = torch.randn(20, 3)
        self.assertEqual(chain1(input), chain2(input))
