import torch
import torch.nn as nn

from torch.testing._internal.common_quantization import QuantizationTestCase

import torch.quantization._equalize as _equalize

import copy

class TestEqualizeEager(QuantizationTestCase):
    def checkChannelsEqualized(self, tensor1, tensor2, output_axis, input_axis):
        ''' Checks the channel ranges of tensor1, tensor2 are the same,
        which is an indication that equalization has been applied correctly
        '''
        output_channel_tensor1 = _equalize.channel_range(tensor1, output_axis)
        input_channel_tensor2 = _equalize.channel_range(tensor2, input_axis)

        # ensuring the channels ranges of tensor1's input is the same as
        # tensor2's output
        self.assertEqual(output_channel_tensor1, input_channel_tensor2)

    def getModule(self, model, name):
        ''' Given the name is a submodule to a model, return the submodule
        '''
        curr = model
        name = name.split('.')
        for subname in name:
            curr = curr._modules[subname]
        return curr

    def test_cross_layer_equalization(self):
        ''' applies _equalize.cross_layer_equalization on two modules and checks
        to make sure channels ranges are equivalent
        '''
        module1 = nn.Conv2d(3, 4, 2)
        module2 = nn.Linear(4, 4)

        module1_output_channel_axis = 0
        module2_input_channel_axis = 1

        _equalize.cross_layer_equalization(module1, module2)

        mod_tensor1, mod_tensor2 = module1.weight, module2.weight

        self.checkChannelsEqualized(mod_tensor1, mod_tensor2, module1_output_channel_axis, module2_input_channel_axis)

    def test_converged(self):
        ''' Sanity checks on _equalize.converged working
        identical modules should return true
        modules with high difference in weights should return false
        '''
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
        ''' First checks to see if _equalize.equalize can handle multiple
        pair modules as input
        then checks correctness of the function by ensuring the equalized
        and unequalized versions of the model yield the same output
        given the same input
        '''
        class ChainModule(nn.Module):
            def __init__(self):
                super(ChainModule, self).__init__()
                self.linear1 = nn.Linear(3, 4)
                self.linear2 = nn.Linear(4, 5)
                self.linear3 = nn.Linear(5, 6)

            def forward(self, x):
                x = self.linear1(x)
                x = self.linear2(x)
                x = self.linear3(x)
                return x
        chain1 = ChainModule()
        chain2 = copy.deepcopy(chain1)

        _equalize.equalize(chain1, [['linear1', 'linear2'], ['linear2', 'linear3']], 1e-6)
        linear1 = self.getModule(chain1, 'linear1')
        linear2 = self.getModule(chain1, 'linear2')
        linear3 = self.getModule(chain1, 'linear3')

        self.checkChannelsEqualized(linear1.weight, linear2.weight, 0, 1)
        self.checkChannelsEqualized(linear2.weight, linear3.weight, 0, 1)

        input = torch.randn(20, 3)
        self.assertEqual(chain1(input), chain2(input))
