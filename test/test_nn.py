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
    def _do_test(self, test_case, module, input):
        test_case.check_jacobian(module, input, self.jacobian_input)


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
        if hasattr(module, 'weight'):
            module.weight.grad.zero_()
        if hasattr(module, 'bias'):
            module.bias.grad.zero_()

    def _get_parameters(self, module):
        params = []
        d_params = []
        if hasattr(module, 'weight'):
            params += [module.weight.data]
            d_params += [module.weight.grad]
        if hasattr(module, 'bias'):
            params += [module.bias.data]
            d_params += [module.bias.grad]
        return params, d_params


def add_test(test):
    test_name = test.get_name()
    cuda_test_name = test_name + '_cuda'
    if hasattr(TestNN, test_name):
        raise RuntimeError('Found two tests with the same name: ' + test_name)
    if hasattr(TestNN, cuda_test_name):
        raise RuntimeError('Found two tests with the same name: ' + cuda_test_name)
    setattr(TestNN, test_name, lambda self,test=test: test(self))
    setattr(TestNN, cuda_test_name, lambda self,test=test: test.test_cuda(self))


for test_params in module_tests:
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
