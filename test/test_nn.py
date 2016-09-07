import math
import torch
import unittest
from copy import deepcopy

import torch.nn as nn
from torch.autograd import Variable
from common_nn import NNTestCase, ModuleTest, CriterionTest, TestBase, \
    module_tests, criterion_tests, TEST_CUDA, PRECISION

class InputVariableMixin(object):
    def _get_input(self):
        input = TestBase._get_input(self)
        def map_variables(i):
            if torch.isTensor(i):
                return Variable(i)
            else:
                return type(i)(map_variables(elem) for elem in i)
        return map_variables(input)


class NewModuleTest(InputVariableMixin, ModuleTest):
    def __init__(self, *args, **kwargs):
        super(NewModuleTest, self).__init__(*args, **kwargs)
        self.check_inplace = kwargs.get('check_inplace', False)

    def _do_test(self, test_case, module, input):
        test_case.check_jacobian(module, input, self.jacobian_input)

        if self.check_inplace:
            module_ip = self.constructor(*self.constructor_args, inplace=True)

            output = module(input)
            test_case.assertFalse(input.dirty)

            output2 = module_ip(input)
            test_case.assertTrue(input.dirty)

            test_case.assertEqual(output, output2)


class NewCriterionTest(InputVariableMixin, CriterionTest):
    pass


class TestNN(NNTestCase):

    def _forward(self, module, input):
        return module(input)

    def _backward(self, module, input, output, grad_output):
        output.backward(grad_output)
        return input.grad

    def _forward_criterion(self, criterion, input, target):
        return criterion(input, target).data[0]

    def _backward_criterion(self, criterion, input, target):
        input.grad.zero_()
        criterion(input, target).backward()
        return input.grad

    def _zero_grad_parameters(self, module):
        if hasattr(module, 'weight') and module.weight is not None:
            module.weight.grad.zero_()
        if hasattr(module, 'bias') and module.bias is not None:
            module.bias.grad.zero_()

    def _get_parameters(self, module):
        params = []
        d_params = []
        if hasattr(module, 'weight') and module.weight is not None:
            params += [module.weight.data]
            d_params += [module.weight.grad]
        if hasattr(module, 'bias') and module.bias is not None:
            params += [module.bias.data]
            d_params += [module.bias.grad]
        return params, d_params

    def test_hooks(self):
        module = nn.Sigmoid()
        input = Variable(torch.ones(5, 5))

        counter = {
            'forwards': 0,
            'backwards': 0
        }

        def fw_hook(inc, h_module, input, output):
            self.assertIsInstance(input, tuple)
            self.assertIsInstance(output, Variable)
            self.assertTrue(h_module is module)
            self.assertEqual(input[0].data, torch.ones(5, 5))
            self.assertEqual(output.data, torch.Tensor(5, 5).fill_(1 / (1 + 1 / math.e)))
            counter['forwards'] += inc

        def bw_hook(inc, h_module, grad_input, grad_output):
            self.assertIsInstance(grad_input, tuple)
            self.assertIsInstance(grad_output, tuple)
            self.assertTrue(h_module is module)
            self.assertEqual(grad_output[0], torch.ones(5, 5) * 2)
            counter['backwards'] += inc

        module.register_forward_hook('test', lambda *args: fw_hook(1, *args))

        module(input)
        module(input)
        self.assertEqual(counter['forwards'], 2)
        self.assertEqual(counter['backwards'], 0)

        module.register_backward_hook('test', lambda *args: bw_hook(1, *args))

        output = module(input)
        self.assertEqual(counter['forwards'], 3)
        self.assertEqual(counter['backwards'], 0)

        output.backward(torch.ones(5, 5) * 2)
        self.assertEqual(counter['forwards'], 3)
        self.assertEqual(counter['backwards'], 1)

        output.backward(torch.ones(5, 5) * 2)
        self.assertEqual(counter['forwards'], 3)
        self.assertEqual(counter['backwards'], 2)

        module.register_forward_hook('test2', lambda *args: fw_hook(2, *args))

        output = module(input)
        self.assertEqual(counter['forwards'], 6)
        self.assertEqual(counter['backwards'], 2)

        module.register_backward_hook('test2', lambda *args: bw_hook(2, *args))

        module(input).backward(torch.ones(5, 5) * 2)
        self.assertEqual(counter['forwards'], 9)
        self.assertEqual(counter['backwards'], 5)

        module.remove_backward_hook('test2')

        module(input).backward(torch.ones(5, 5) * 2)
        self.assertEqual(counter['forwards'], 12)
        self.assertEqual(counter['backwards'], 6)

        module.remove_forward_hook('test2')

        module(input).backward(torch.ones(5, 5) * 2)
        self.assertEqual(counter['forwards'], 13)
        self.assertEqual(counter['backwards'], 7)

    def test_volatile(self):
        module = nn.Conv2d(2, 5, ksize=3, pad=1)
        input = torch.randn(1, 2, 10, 10)
        x = Variable(input)
        y = Variable(input.clone(), volatile=True)

        output = module(x)
        self.assertFalse(output.volatile)
        self.assertTrue(output.requires_grad)
        output.backward(torch.ones(1, 5, 10, 10))

        vol_output = module(y)
        self.assertTrue(vol_output.volatile)
        self.assertFalse(vol_output.requires_grad)
        self.assertRaises(RuntimeError, lambda: vol_output.backward(torch.ones(1, 5, 10, 10)))



def add_test(test):
    test_name = test.get_name()
    cuda_test_name = test_name + '_cuda'
    if hasattr(TestNN, test_name):
        raise RuntimeError('Found two tests with the same name: ' + test_name)
    if hasattr(TestNN, cuda_test_name):
        raise RuntimeError('Found two tests with the same name: ' + cuda_test_name)
    setattr(TestNN, test_name, lambda self,test=test: test(self))
    setattr(TestNN, cuda_test_name, lambda self,test=test: test.test_cuda(self))


new_module_tests = [
    dict(
        module_name='Conv2d',
        constructor_args=(3, 4, (3, 3)),
        input_size=(2, 3, 6, 6)
    ),
    dict(
        module_name='Conv2d',
        constructor_args=(3, 4, (3, 3), (2, 2)),
        input_size=(2, 3, 6, 6),
        desc='strided'
    ),
    dict(
        module_name='Conv2d',
        constructor_args=(3, 4, (3, 3), (2, 2), (1, 1)),
        input_size=(2, 3, 6, 6),
        desc='padding'
    ),
    dict(
        module_name='MaxPool2d',
        constructor_args=((3, 3), (2, 2), (1, 1)),
        input_size=(1, 3, 7, 7)
    ),
    dict(
        module_name='AvgPool2d',
        constructor_args=((2, 2),),
        input_size=(2, 3, 6, 6),
    ),
    dict(
        module_name='AvgPool2d',
        constructor_args=((2, 2), (2, 2)),
        input_size=(2, 3, 6, 6),
        desc='stride',
    ),
    dict(
        module_name='AvgPool2d',
        constructor_args=((2, 2), (2, 2), (1, 1)),
        input_size=(2, 3, 6, 6),
        desc='stride_pad',
    ),
]

for test_params in module_tests + new_module_tests:
    test_params = deepcopy(test_params)
    # TODO: CUDA is not implemented yet
    name = test_params.pop('module_name')
    test_params['constructor'] = getattr(nn, name)
    test = NewModuleTest(**test_params)
    add_test(test)
for test_params in criterion_tests:
    test_params = deepcopy(test_params)
    name = test_params.pop('module_name')
    test_params['constructor'] = getattr(nn, name)
    test = NewCriterionTest(**test_params)
    add_test(test)


if __name__ == '__main__':
    unittest.main()
