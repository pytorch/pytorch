import math
import torch
import random
import unittest
from copy import deepcopy
from itertools import repeat

import torch.nn as nn
import torch.nn.parallel as dp
from torch.autograd import Variable
from common_nn import NNTestCase, ModuleTest, CriterionTest, TestBase, \
    module_tests, criterion_tests, TEST_CUDA, PRECISION
from common import freeze_rng_state


class InputVariableMixin(object):
    def _get_input(self):
        input = TestBase._get_input(self)
        def map_variables(i):
            if isinstance(i, Variable):
                return i
            elif torch.is_tensor(i):
                return Variable(i, requires_grad=True)
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

            input_version = input._version
            output = module(input)
            test_case.assertEqual(input._version, input_version)

            input_ip = deepcopy(input)
            output_ip = module_ip(input_ip)
            test_case.assertNotEqual(input_ip._version, input_version)

            test_case.assertEqual(output, output_ip)

        if type(input.data) == torch.LongTensor and TEST_CUDA:
            input = input.cuda()
            module.float().cuda()
            module(input)
            for p in module.parameters():
                test_case.assertEqual(type(p.data), torch.cuda.FloatTensor)
                test_case.assertEqual(p.get_device(), 0)

            if torch.cuda.device_count() > 1:
                input = input.cuda(1)
                module.cuda(1)
                with torch.cuda.device(1):
                    module(input)
                for p in module.parameters():
                    test_case.assertEqual(type(p.data), torch.cuda.FloatTensor)
                    test_case.assertEqual(p.get_device(), 1)
        else:
            # to float
            if type(input.data) != torch.LongTensor:
                input = input.float()
            module.float()
            module(input)
            for p in module.parameters():
                test_case.assertEqual(type(p.data), torch.FloatTensor)

            # and back to double
            if type(input.data) != torch.LongTensor:
                input = input.double()
            module.double()
            module(input)
            for p in module.parameters():
                test_case.assertEqual(type(p.data), torch.DoubleTensor)

            # TODO: Hardshrink is lacking a CUDA implementation
            if TEST_CUDA and type(module) != nn.Hardshrink:
                # to GPU0
                input = input.float().cuda()
                module.float().cuda()
                module(input)
                for p in module.parameters():
                    test_case.assertEqual(type(p.data), torch.cuda.FloatTensor)
                    test_case.assertEqual(p.get_device(), 0)

                # to CPU
                input = input.cpu()
                module.cpu()
                module(input)
                for p in module.parameters():
                    test_case.assertEqual(type(p.data), torch.FloatTensor)

                # back to GPU0
                input = input.cuda()
                module.cuda()
                module(input)
                for p in module.parameters():
                    test_case.assertEqual(type(p.data), torch.cuda.FloatTensor)
                    test_case.assertEqual(p.get_device(), 0)

                if torch.cuda.device_count() >= 2:
                    # to GPU1
                    input = input.cuda(1)
                    module.cuda(1)
                    with torch.cuda.device(1):
                        module(input)
                    for p in module.parameters():
                        test_case.assertEqual(type(p.data), torch.cuda.FloatTensor)
                        test_case.assertEqual(p.get_device(), 1)


class NewCriterionTest(InputVariableMixin, CriterionTest):
    # TODO: check that criterions don't ignore grad_output

    def _get_target(self, target):
        return Variable(target, requires_grad=False)


class TestNN(NNTestCase):

    def _forward(self, module, input):
        with freeze_rng_state():
            return module(input)

    def _backward(self, module, input, output, grad_output):
        output.backward(grad_output, retain_variables=True)
        return input.grad

    def _forward_criterion(self, criterion, input, target):
        if isinstance(input, tuple):
            args = input + (target,)
            output = criterion(*args)
        else:
            output = criterion(input, target)
        return output.data[0]

    def _backward_criterion(self, criterion, input, target):
        input_tuple = input if isinstance(input, tuple) else (input,)
        for i in input_tuple:
            i.grad.zero_()
        args = input_tuple + (target,)
        criterion(*args).backward()
        if isinstance(input, tuple):
            return tuple(map(lambda i: i.grad, input))
        else:
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
        input = Variable(torch.ones(5, 5), requires_grad=True)

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

        output.backward(torch.ones(5, 5) * 2, retain_variables=True)
        self.assertEqual(counter['forwards'], 3)
        self.assertEqual(counter['backwards'], 1)

        output.backward(torch.ones(5, 5) * 2, retain_variables=True)
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
        module = nn.Conv2d(2, 5, kernel_size=3, padding=1)
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

    def _test_dropout(self, cls, input):
        p = 0.2
        input.fill_(1-p)

        module = cls(p)
        input_var = Variable(input, requires_grad=True)
        output = module(input_var)
        self.assertLess(abs(output.data.mean() - (1-p)), 0.05)
        output.backward(input)
        self.assertLess(abs(input_var.grad.mean() - (1-p)), 0.05)

        module = cls(p, True)
        input_var = Variable(input.clone(), requires_grad=True)
        output = module(input_var + 0)
        self.assertLess(abs(output.data.mean() - (1-p)), 0.05)
        output.backward(input)
        self.assertLess(abs(input_var.grad.mean() - (1-p)), 0.05)

        # Check that these don't raise errors
        module.__repr__()
        str(module)

    def test_parameters(self):
        def num_params(module):
            return len(list(module.parameters()))
        class Net(nn.Container):
            def __init__(self):
                super(Net, self).__init__(
                    l1=l,
                    l2=l
                )
                self.param = Variable(torch.Tensor(3, 5))
        l = nn.Linear(10, 20)
        n = Net()
        s = nn.Sequential(n, n, n, n)
        self.assertEqual(num_params(l), 2)
        self.assertEqual(num_params(n), 3)
        self.assertEqual(num_params(s), 3)

    def test_modules(self):
        class Net(nn.Container):
            def __init__(self):
                super(Net, self).__init__()
                self.l1 = l
                self.l2 = l
                self.param = Variable(torch.Tensor(3, 5))
        l = nn.Linear(10, 20)
        n = Net()
        s = nn.Sequential(n, n, n, n)
        self.assertEqual(list(s.modules()), [s, n, l])

    def test_Sequential_getitem(self):
        l1 = nn.Linear(10, 20)
        l2 = nn.Linear(20, 30)
        l3 = nn.Linear(30, 40)
        l4 = nn.Linear(40, 50)
        n = nn.Sequential(l1, l2, l3, l4)
        self.assertEqual(n[0], l1)
        self.assertEqual(n[1], l2)
        self.assertEqual(n[2], l3)
        self.assertEqual(n[3], l4)

    def test_add_module(self):
        l = nn.Linear(10, 20)
        net = nn.Container(
            l=l,
            l2=l,
            empty=None,
        )
        self.assertEqual(net.l, l)
        self.assertEqual(net.l2, l)
        self.assertEqual(net.empty, None)
        net.add_module('l3', l)
        self.assertEqual(net.l3, l)
        self.assertRaises(KeyError, lambda: net.add_module('l', l))
        self.assertRaises(TypeError, lambda: net.add_module('x', 'non-module'))

    def test_type(self):
        l = nn.Linear(10, 20)
        net = nn.Container(
            l=l,
            l2=l,
            empty=None,
        )
        net.float()
        self.assertIsInstance(l.weight.data, torch.FloatTensor)
        self.assertIsInstance(l.bias.data, torch.FloatTensor)
        net.double()
        self.assertIsInstance(l.weight.data, torch.DoubleTensor)
        self.assertIsInstance(l.bias.data, torch.DoubleTensor)

    def test_non_leaf_parameters(self):
        l1 = nn.Linear(10, 10)
        l2 = nn.Linear(10, 10)
        def assign_weight():
            l2.weight = l1.weight + 2
        self.assertRaises(ValueError, assign_weight)
        # This should work though
        l2.weight = Variable(torch.randn(10, 10))

    def test_embedding_padding_idx(self):
        embedding = nn.Embedding(10, 20, padding_idx = 0)
        input = Variable(torch.LongTensor([[0,2,4,5],[4,3,0,9]]))
        output = embedding(input)
        self.assertEqual(output[0][0].sum().data[0], 0)
        self.assertEqual(output[1][2].sum().data[0], 0)

    def test_Dropout(self):
        input = torch.Tensor(1000)
        self._test_dropout(nn.Dropout, input)

    def test_Dropout2d(self):
        b = random.randint(1, 5)
        w = random.randint(1, 5)
        h = random.randint(1, 5)
        num_features = 1000
        input = torch.Tensor(num_features, b, w, h)
        self._test_dropout(nn.Dropout2d, input)

    def test_Dropout3d(self):
        b = random.randint(1, 5)
        w = random.randint(1, 5)
        h = random.randint(1, 5)
        d = random.randint(1, 2)
        num_features = 1000
        input = torch.Tensor(num_features, b, d, w, h)
        self._test_dropout(nn.Dropout3d, input)

    def _test_maxpool_indices(self, num_dim):
        def expected_indices(dim):
            if dim == 1:
                return torch.DoubleTensor([1, 3])
            lower_dim = expected_indices(dim-1)
            lower_dim = lower_dim.view(1, *lower_dim.size())
            return torch.cat((lower_dim+4, lower_dim+12), 0)

        def expected_grad(dim):
            if dim == 1:
                return torch.DoubleTensor([0, 1, 0, 1])
            lower_dim_grad = expected_grad(dim-1)
            grad = lower_dim_grad.view(1, *lower_dim_grad.size())
            zero = torch.zeros(grad.size())
            return torch.cat((zero, grad, zero, grad), 0)

        module_cls = getattr(nn, 'MaxPool{}d'.format(num_dim))
        module = module_cls(2, return_indices=True)
        numel = 4 ** num_dim
        input = torch.range(1, numel).view(1, 1, *repeat(4, num_dim))
        input_var = Variable(input, requires_grad=True)

        # Check forward
        output, indices = module(input_var)
        if num_dim != 3:
            expected_indices = expected_indices(num_dim)
            expected_output = expected_indices + 1
            self.assertEqual(indices.data.squeeze(), expected_indices)
            self.assertEqual(output.data.squeeze(), expected_output)
        self.assertTrue(output.requires_grad)
        self.assertFalse(indices.requires_grad)

        # Make sure backward works
        grad_output = torch.DoubleTensor(output.size()).fill_(1)
        output.backward(grad_output, retain_variables=True)
        expected_grad = expected_grad(num_dim)
        self.assertEqual(input_var.grad, expected_grad.view_as(input))

        # Make sure backward after changing indices will result in an error
        indices.add_(1)
        self.assertRaises(RuntimeError, lambda: output.backward(grad_output))

    def test_MaxPool1d_indices(self):
        self._test_maxpool_indices(1)

    def test_MaxPool2d_indices(self):
        self._test_maxpool_indices(2)

    def test_MaxPool3d_indices(self):
        self._test_maxpool_indices(3)

    def _test_scatter(self, x):
        if not TEST_CUDA or torch.cuda.device_count() < 2:
            raise unittest.SkipTest("Only one GPU detected")
        x = Variable(x)
        result = dp.scatter(x, (0, 1))
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], x[:2])
        self.assertEqual(result[0].get_device(), 0)
        self.assertEqual(result[1], x[2:])
        self.assertEqual(result[1].get_device(), 1)

    def test_scatter_cpu(self):
        self._test_scatter(torch.randn(4, 4))

    def test_scatter_gpu(self):
        self._test_scatter(torch.randn(4, 4))

    def _test_gather(self, output_device):
        if not TEST_CUDA or torch.cuda.device_count() < 2:
            raise unittest.SkipTest("Only one GPU detected")
        inputs = (
            Variable(torch.randn(2, 4).cuda(0)),
            Variable(torch.randn(2, 4).cuda(1))
        )
        result = dp.gather(inputs, output_device)
        self.assertEqual(result.size().tolist(), [4, 4])
        self.assertEqual(result[:2], inputs[0])
        self.assertEqual(result[2:], inputs[1])
        if output_device != -1:
            self.assertEqual(result.get_device(), output_device)
        else:
            self.assertFalse(result.is_cuda)

    def test_gather_cpu(self):
        self._test_gather(-1)

    def test_gather_gpu(self):
        self._test_gather(0)

    @unittest.skipIf(not TEST_CUDA or torch.cuda.device_count() < 2,
                     "Only one GPU detected")
    def _test_replicate(self):
        module = nn.Linear(10, 5).float().cuda()
        input = torch.randn(2, 10).float().cuda()
        expected_output = module(input).data
        replicas = dp.replicate(module, (0, 1))
        for i, replica in enumerate(replicas):
            for p in replica.parameters():
                self.assertEqual(p.get_device(), i)
            replica_input = input.cuda(i)
            self.assertEqual(replica(replica_input).data, expected_output)

    @unittest.skipIf(not TEST_CUDA or torch.cuda.device_count() < 2,
                     "Only one GPU detected")
    def test_parallel_apply(self):
        l1 = nn.Linear(10, 5).float().cuda(0)
        l2 = nn.Linear(10, 5).float().cuda(1)
        i1 = Variable(torch.randn(2, 10).float().cuda(0))
        i2 = Variable(torch.randn(2, 10).float().cuda(1))
        expected1 = l1(i1).data
        expected2 = l2(i2).data
        inputs = (i1, i2)
        modules = (l1, l2)
        expected_outputs = (expected1, expected2)
        outputs = dp.parallel_apply(modules, inputs)
        for out, expected in zip(outputs, expected_outputs):
            self.assertEqual(out.data, expected)

        inputs = (i1, Variable(i2.data.new()))
        expected_outputs = (expected1, expected2.new())

    @unittest.skipIf(not TEST_CUDA or torch.cuda.device_count() < 2,
                     "Only one GPU detected")
    def test_data_parallel(self):
        l = nn.Linear(10, 5).float().cuda()
        i = Variable(torch.randn(20, 10).float().cuda(1))
        l.cuda(1)
        expected_out = l(i).data
        l.cuda(0)
        out = dp.data_parallel(l, i, (0, 1))
        self.assertEqual(out.get_device(), 1)
        self.assertEqual(out.data, expected_out)

    def test_parameter_dict(self):
        l = nn.Linear(5, 5)
        block = nn.Container(
            conv=nn.Conv2d(3, 3, 3, bias=False)
        )
        net = nn.Container(
            linear1=l,
            linear2=l,
            block=block,
            empty=None,
        )
        param_dict = net.parameter_dict()
        self.assertEqual(len(param_dict), 5)
        self.assertIn('linear1.weight', param_dict)
        self.assertIn('linear1.bias', param_dict)
        self.assertIn('linear2.weight', param_dict)
        self.assertIn('linear2.bias', param_dict)
        self.assertIn('block.conv.weight', param_dict)
        self.assertNotIn('block.conv.bias', param_dict)
        self.assertFalse(any(map(lambda k: k.startswith('empty'), param_dict.keys())))
        for k, v in param_dict.items():
            param = net
            for component in k.split('.'):
                param = getattr(param, component)
            self.assertIs(v, param)

        l = nn.Linear(5, 5)
        param_dict = l.parameter_dict()
        self.assertEqual(len(param_dict), 2)
        self.assertIs(param_dict['weight'], l.weight)
        self.assertIs(param_dict['bias'], l.bias)

    def test_load_parameter_dict(self):
        l = nn.Linear(5, 5)
        block = nn.Container(
            conv=nn.Conv2d(3, 3, 3, bias=False)
        )
        net = nn.Container(
            linear1=l,
            linear2=l,
            block=block,
            empty=None,
        )
        param_dict = {
            'linear1.weight': Variable(torch.ones(5, 5)),
            'block.conv.bias': Variable(torch.range(1, 3)),
        }
        net.load_parameter_dict(param_dict)
        self.assertIs(net.linear1.weight, param_dict['linear1.weight'])
        self.assertIs(net.block.conv.bias, param_dict['block.conv.bias'])

    def test_ConvTranspose2d_output_size(self):
        m = nn.ConvTranspose2d(3, 4, 3, 3, 0, 2)
        i = Variable(torch.randn(2, 3, 6, 6))
        for h in range(15, 22):
            for w in range(15, 22):
                if 18 <= h <= 20 and 18 <= w <= 20:
                    size = (h, w)
                    if h == 19:
                        size = torch.LongStorage(size)
                    elif h == 2:
                        size = torch.LongStorage((2, 4) + size)
                    m(i, output_size=(h, w))
                else:
                    self.assertRaises(ValueError, lambda: m(i, (h, w)))

    def test_MaxUnpool2d_output_size(self):
        m = nn.MaxPool2d(3, stride=2, return_indices=True)
        mu = nn.MaxUnpool2d(3, stride=2)
        big_t = torch.rand(1, 1, 6, 6)
        big_t[0][0][4][4] = 100
        output_big, indices_big = m(Variable(big_t))
        self.assertRaises(RuntimeError, lambda: mu(output_big, indices_big))

        small_t = torch.rand(1, 1, 5, 5)
        for i in range(0, 4, 2):
            for j in range(0, 4, 2):
                small_t[:,:,i,j] = 100
        output_small, indices_small = m(Variable(small_t))
        for h in range(3, 10):
            for w in range(3, 10):
                if 4 <= h <= 6 and 4 <= w <= 6:
                    size = (h, w)
                    if h == 5:
                        size = torch.LongStorage(size)
                    elif h == 6:
                        size = torch.LongStorage((1, 1) + size)
                    mu(output_small, indices_small, output_size=size)
                else:
                    self.assertRaises(ValueError, lambda:
                            mu(output_small, indices_small, (h, w)))


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
        module_name='Conv1d',
        constructor_args=(4, 5, 3),
        input_size=(2, 4, 10)
    ),
    dict(
        module_name='Conv1d',
        constructor_args=(4, 5, 3),
        input_size=(2, 4, 10),
        desc='stride'
    ),
    dict(
        module_name='MaxPool1d',
        constructor_args=(4,),
        input_size=(2, 10, 4)
    ),
    dict(
        module_name='MaxPool1d',
        constructor_args=(4, 4),
        input_size=(2, 10, 4),
        desc='stride'
    ),
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
        module_name='Conv2d',
        constructor_args=(3, 2, (3, 3), (2, 2), (1, 1), (2, 2)),
        input_size=(2, 3, 8, 8),
        desc='dilated'
    ),
    dict(
        module_name='Conv2d',
        constructor_args=(3, 4, (3, 3), 1, 0, None, 1, False),
        input_size=(2, 3, 6, 6),
        desc='no_bias',
    ),
    dict(
        module_name='ConvTranspose2d',
        constructor_args=(3, 4, 3, (2, 2), 1, (1, 1)),
        input_size=(1, 3, 7, 7)
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
    dict(
        module_name='LPPool2d',
        constructor_args=(2, (2, 2), 2),
        input_size=(1, 3, 7, 7)
    ),
    dict(
        module_name='LPPool2d',
        constructor_args=(1.5, 2),
        input=torch.rand(1, 3, 7, 7),
        desc='norm'
    ),
    dict(
        module_name='ReflectionPad2d',
        constructor_args=((1, 2, 3, 4),),
        input_size=(2, 3, 8, 8)
    ),
    dict(
        module_name='ReplicationPad2d',
        constructor_args=((1, 2, 3, 4),),
        input_size=(2, 3, 4, 4)
    ),
    dict(
        module_name='Conv3d',
        constructor_args=(3, 4, 2),
        input_size=(2, 3, 3, 3, 3)
    ),
    dict(
        module_name='Conv3d',
        constructor_args=(3, 4, 2, 2),
        input_size=(2, 3, 5, 5, 5),
        desc='stride'
    ),
    dict(
        module_name='Conv3d',
        constructor_args=(3, 4, 2, 2, 1),
        input_size=(2, 3, 5, 5, 5),
        desc='stride_padding'
    ),
    dict(
        module_name='ConvTranspose3d',
        constructor_args=(2, 3, (2, 2, 2)),
        input_size=(1, 2, 4, 4, 4)
    ),
    dict(
        module_name='MaxPool3d',
        constructor_args=((2, 2, 2),),
        input_size=(2, 3, 5, 5, 5)
    ),
    dict(
        module_name='MaxPool3d',
        constructor_args=(2, (2, 2, 2)),
        input_size=(2, 3, 5, 5, 5),
        desc='stride'
    ),
    dict(
        module_name='MaxPool3d',
        constructor_args=(2, 2, (1, 1, 1)),
        input_size=(2, 3, 5, 5, 5),
        desc='stride_padding'
    ),
    dict(
        module_name='AvgPool3d',
        constructor_args=((2, 2, 2),),
        input_size=(2, 3, 4, 4, 4)
    ),
    dict(
        module_name='AvgPool3d',
        constructor_args=(2, (2, 2, 2)),
        input_size=(2, 3, 5, 5, 5),
        desc='stride'
    ),
    dict(
        module_name='ReplicationPad3d',
        constructor_args=((1, 2, 3, 4, 5, 6),),
        input_size=(2, 3, 5, 5, 5)
    ),
    dict(
        module_name='Embedding',
        constructor_args=(4, 3),
        input=Variable(
            torch.randperm(2).repeat(1, 2).long(),
            requires_grad=False
        ),
        jacobian_input=False
    ),
    dict(
        constructor=lambda: nn.FractionalMaxPool2d(2, output_ratio=0.5, _random_samples=torch.DoubleTensor(1, 3, 2).uniform_()),
        input_size=(1, 3, 5, 5),
        fullname='FractionalMaxPool2d_ratio',
        test_cuda=False),
    dict(
        constructor=lambda: nn.FractionalMaxPool2d((2, 2), output_size=(4, 4), _random_samples=torch.DoubleTensor(1, 3, 2).uniform_()),
        input_size=(1, 3, 7, 7),
        fullname='FractionalMaxPool2d_size',
        test_cuda=False
    ),
]


for test_params in module_tests + new_module_tests:
    # TODO: CUDA is not implemented yet
    if 'constructor' not in test_params:
        name = test_params.pop('module_name')
        test_params['constructor'] = getattr(nn, name)
    test = NewModuleTest(**test_params)
    add_test(test)
for test_params in criterion_tests:
    name = test_params.pop('module_name')
    test_params['constructor'] = getattr(nn, name)
    test = NewCriterionTest(**test_params)
    add_test(test)


class UnpoolingNet2d(nn.Container):

    def __init__(self):
        super(UnpoolingNet2d, self).__init__(
            pool=nn.MaxPool2d(2, return_indices=True),
            unpool=nn.MaxUnpool2d(2)
        )

    def forward(self, input):
        return self.unpool(*self.pool(input))


class UnpoolingNet3d(nn.Container):

    def __init__(self):
        super(UnpoolingNet3d, self).__init__(
            pool=nn.MaxPool3d(2, return_indices=True),
            unpool=nn.MaxUnpool3d(2)
        )

    def forward(self, input):
        return self.unpool(*self.pool(input))


add_test(NewModuleTest(
    constructor=UnpoolingNet2d,
    input_size=(1, 1, 8, 8),
    fullname='MaxUnpool2d_net'))
add_test(NewModuleTest(
    constructor=UnpoolingNet3d,
    input_size=(1, 1, 8, 8, 8),
    fullname='MaxUnpool3d_net'))


if __name__ == '__main__':
    unittest.main()
