import math
import torch
import random
import unittest
import contextlib
from copy import deepcopy
from itertools import repeat
from functools import wraps

import torch.nn as nn
import torch.nn.parallel as dp
from torch.autograd import Variable
from torch.nn import Parameter
from common_nn import NNTestCase, ModuleTest, CriterionTest, TestBase, \
    module_tests, criterion_tests, TEST_CUDA, TEST_MULTIGPU, TEST_CUDNN, PRECISION
from common import freeze_rng_state

def default_tensor_type(type):
    type_str = torch.typename(type)
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            old_type = torch.typename(torch.Tensor())
            torch.set_default_tensor_type(type_str)
            try:
                return fn(*args, **kwargs)
            finally:
                torch.set_default_tensor_type(old_type)
        return wrapper
    return decorator

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
    # # protip: uncomment this line to figure out which test is segfaulting
    # def setUp(self):
    #     print("In method", self._testMethodName)
    #     super(TestNN, self).setUp()

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
                self.param = Parameter(torch.Tensor(3, 5))
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
        net.type(torch.FloatTensor)
        self.assertIsInstance(l.weight.data, torch.FloatTensor)
        self.assertIsInstance(l.bias.data, torch.FloatTensor)
        net.type(torch.DoubleTensor)
        self.assertIsInstance(l.weight.data, torch.DoubleTensor)
        self.assertIsInstance(l.bias.data, torch.DoubleTensor)
        if TEST_CUDA:
            net.type(torch.cuda.FloatTensor)
            self.assertIsInstance(l.weight.data, torch.cuda.FloatTensor)
            self.assertIsInstance(l.bias.data, torch.cuda.FloatTensor)

    def test_non_leaf_parameters(self):
        l1 = nn.Linear(10, 10)
        l2 = nn.Linear(10, 10)
        def assign_weight():
            l2.weight = l1.weight + 2
        self.assertRaises(TypeError, assign_weight)
        # This should work though
        l2.weight = Parameter(torch.randn(10, 10))

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

    def _test_scatter(self, tensor):
        x = Variable(tensor, requires_grad=True)
        result = dp.scatter(x, (0, 1))
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], x[:2])
        self.assertEqual(result[0].get_device(), 0)
        self.assertEqual(result[1], x[2:])
        self.assertEqual(result[1].get_device(), 1)
        grad = result[0].data.clone().fill_(2)
        result[0].backward(grad)
        self.assertEqual(x.grad[:2], grad)
        self.assertEqual(x.grad[2:], grad.clone().zero_())

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_scatter_cpu(self):
        self._test_scatter(torch.randn(4, 4))

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_scatter_gpu(self):
        self._test_scatter(torch.randn(4, 4).cuda())

    def _test_gather(self, output_device):
        inputs = (
            Variable(torch.randn(2, 4).cuda(0), requires_grad=True),
            Variable(torch.randn(2, 4).cuda(1), requires_grad=True)
        )
        result = dp.gather(inputs, output_device)
        self.assertEqual(result.size(), torch.Size([4, 4]))
        self.assertEqual(result[:2], inputs[0])
        self.assertEqual(result[2:], inputs[1])
        if output_device != -1:
            self.assertEqual(result.get_device(), output_device)
        else:
            self.assertFalse(result.is_cuda)
        grad = torch.randn(4, 4)
        if output_device != -1:
            grad = grad.cuda(output_device)
        result.backward(grad)
        self.assertEqual(inputs[0].grad, grad[:2])
        self.assertEqual(inputs[1].grad, grad[2:])

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_gather_cpu(self):
        self._test_gather(-1)

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_gather_gpu(self):
        self._test_gather(0)

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_replicate(self):
        module = nn.Linear(10, 5).float().cuda()
        input = Variable(torch.randn(2, 10).float().cuda())
        expected_output = module(input).data
        replicas = dp.replicate(module, (0, 1))
        for i, replica in enumerate(replicas):
            for p in replica.parameters():
                self.assertEqual(p.get_device(), i)
            replica_input = input.cuda(i)
            self.assertEqual(replica(replica_input).data, expected_output)

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_replicate_buffers(self):
        net = nn.Container()
        net.bn = nn.BatchNorm2d(10)
        net.cuda()
        replicas = dp.replicate(net, (0, 1))
        for i, replica in enumerate(replicas):
            self.assertEqual(replica.bn.running_mean.get_device(), i, 'buffer on wrong device')
            self.assertEqual(replica.bn.running_var.get_device(), i, 'buffer on wrong device')

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
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

    def test_data_parallel_noop(self):
        l = nn.Linear(10, 5).float()
        i = Variable(torch.randn(20, 10).float())
        out = dp.data_parallel(l, i, [])
        self.assertEqual(out, l(i))
        self.assertFalse(out.is_cuda)

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_data_parallel_small_back(self):
        l = nn.Linear(10, 5).float().cuda()
        i = Variable(torch.randn(20, 10).float().cuda())
        out = dp.data_parallel(l, i, (0, 1))
        self.assertEqual(out, l(i))

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_data_parallel(self):
        l = nn.Linear(10, 5).float().cuda()
        i = Variable(torch.randn(20, 10).float().cuda(1))
        l.cuda(1)
        expected_out = l(i).data
        l.cuda(0)
        out = dp.data_parallel(l, i, (0, 1))
        self.assertEqual(out.get_device(), 0)
        self.assertEqual(out.data, expected_out)

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_data_parallel_nested_output(self):
        def fn(input):
            return [input, (input.sin(), input.cos(), [input.add(1)]), input]
        class Net(nn.Container):
            def forward(self, input):
                return fn(input)
        i = Variable(torch.randn(2, 2).float().cuda(1))
        gpus = range(torch.cuda.device_count())
        output = dp.data_parallel(Net(), i, gpus)
        self.assertEqual(output, fn(i))
        self.assertIsInstance(output[0], Variable)
        self.assertIsInstance(output[1], tuple)
        self.assertIsInstance(output[1][0], Variable)
        self.assertIsInstance(output[1][1], Variable)
        self.assertIsInstance(output[1][2], list)
        self.assertIsInstance(output[1][2][0], Variable)
        self.assertIsInstance(output[2], Variable)

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_data_parallel_nested_input(self):
        def fn(input):
            return input[1][0]
        class Net(nn.Container):
            def forward(self, input):
                return fn(input)
        i = Variable(torch.randn(20, 3).float().cuda(1))
        input = (i.cos(), (i.sin(), i), i.sin())
        gpus = range(torch.cuda.device_count())
        output = dp.data_parallel(Net(), input, gpus)
        self.assertEqual(output, fn(input))

    def test_data_parallel_module(self):
        l = nn.Linear(10, 5).float().cuda()
        i = Variable(torch.randn(20, 10).float().cuda())
        expected_out = l(i).data
        net = nn.DataParallel(l)
        out = net(i)
        self.assertEqual(out.get_device(), 0)
        self.assertEqual(out.data, expected_out)

    def test_parameter_dict(self):
        l = nn.Linear(5, 5)
        block = nn.Container(
            conv=nn.Conv2d(3, 3, 3, bias=False)
        )
        net = nn.Container(
            linear1=l,
            linear2=l,
            bn=nn.BatchNorm2d(2),
            block=block,
            empty=None,
        )
        state_dict = net.state_dict()
        self.assertEqual(len(state_dict), 9)
        self.assertIn('linear1.weight', state_dict)
        self.assertIn('linear1.bias', state_dict)
        self.assertIn('linear2.weight', state_dict)
        self.assertIn('linear2.bias', state_dict)
        self.assertIn('block.conv.weight', state_dict)
        self.assertIn('block.conv.weight', state_dict)
        self.assertNotIn('block.conv.bias', state_dict)
        self.assertIn('bn.weight', state_dict)
        self.assertIn('bn.bias', state_dict)
        self.assertIn('bn.running_var', state_dict)
        self.assertIn('bn.running_mean', state_dict)
        self.assertFalse(any(map(lambda k: k.startswith('empty'), state_dict.keys())))
        for k, v in state_dict.items():
            param = net
            for component in k.split('.'):
                param = getattr(param, component)
            self.assertIs(v, param)

        l = nn.Linear(5, 5)
        state_dict = l.state_dict()
        self.assertEqual(len(state_dict), 2)
        self.assertIs(state_dict['weight'], l.weight)
        self.assertIs(state_dict['bias'], l.bias)

    def test_load_parameter_dict(self):
        l = nn.Linear(5, 5)
        block = nn.Container(
            conv1=nn.Conv2d(3, 3, 3, bias=False),
            conv2=nn.Conv2d(3, 3, 3, bias=False),
        )
        net = nn.Container(
            linear1=l,
            linear2=l,
            bn=nn.BatchNorm2d(2),
            block=block,
            empty=None,
        )
        state_dict = {
            'linear1.weight': Parameter(torch.ones(5, 5)),
            'block.conv1.bias': Parameter(torch.range(1, 3)),
            'block.conv2.bias': None,
            'bn.running_mean': torch.randn(2),
        }
        net.load_state_dict(state_dict)
        self.assertIs(net.linear1.weight, state_dict['linear1.weight'])
        self.assertIs(net.block.conv1.bias, state_dict['block.conv1.bias'])
        self.assertIs(net.block.conv2.bias, state_dict['block.conv2.bias'])
        self.assertIs(net.bn.running_mean, state_dict['bn.running_mean'])

        state_dict = {
            'linear1.weight': torch.ones(5, 5)
        }
        self.assertRaises(TypeError, lambda: net.load_state_dict(state_dict))

    def test_parameter_assignment(self):
        l = nn.Linear(5, 5)
        def num_params():
            return len(list(l.parameters()))
        self.assertEqual(num_params(), 2)

        new_param = Parameter(torch.randn(5, 5))
        l.param_name = new_param
        self.assertEqual(num_params(), 3)
        self.assertIn(new_param, l.parameters())

        var = Variable(torch.randn(5, 5))
        l.var_name = var
        self.assertEqual(num_params(), 3)
        self.assertNotIn(var, l.parameters())

        # Make sure Variables are not saved as parameters
        l.variable_attr = Variable(torch.Tensor(5, 5))
        self.assertEqual(num_params(), 3)
        l.param_attr = Parameter(torch.Tensor(5, 5))
        self.assertEqual(num_params(), 4)

        # It shouldn't be possible to replace a parameter with a Variable
        def assign_var():
            l.param_attr = Variable(torch.Tensor(5, 5))
        self.assertRaises(TypeError, assign_var)
        # But replacing it with None should be fine
        l.param_attr = None
        self.assertEqual(num_params(), 3)

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


    def test_RNN_cell(self):
        # this is just a smoke test; these modules are implemented through
        # autograd so no Jacobian test is needed
        for module in (nn.RNNCell, nn.GRUCell):
            for bias in (True, False):
                input = Variable(torch.randn(3, 10))
                hx = Variable(torch.randn(3, 20))
                cell = module(10, 20, bias=bias)
                for i in range(6):
                    hx = cell(input, hx)

                hx.sum().backward()

    def test_LSTM_cell(self):
        # this is just a smoke test; these modules are implemented through
        # autograd so no Jacobian test is needed
        for bias in (True, False):
            input = Variable(torch.randn(3, 10))
            hx = Variable(torch.randn(3, 20))
            cx = Variable(torch.randn(3, 20))
            lstm = nn.LSTMCell(10, 20, bias=bias)
            for i in range(6):
                hx, cx = lstm(input, (hx, cx))

            (hx+cx).sum().backward()

    @unittest.skipIf(not TEST_CUDNN, "needs cudnn")
    @default_tensor_type(torch.FloatTensor)  # FIXME: just until torch.cuda.DoubleTensor.sum() implemented
    def test_RNN_cpu_vs_cudnn(self):

        def forward_backward(cuda, rnn, input_val, hx_val, weights_val):
            is_lstm = type(rnn) == nn.LSTM

            for x_layer, y_layer in zip(rnn.all_weights, weights_val):
                for x, y in zip(x_layer, y_layer):
                    x.data.copy_(y.data)

            input = Variable(input_val.clone(), requires_grad=True)
            if is_lstm:
                hx = (Variable(hx_val.clone(), requires_grad=True),
                      Variable(hx_val.add(1), requires_grad=True))
            else:
                hx = Variable(hx_val.clone(), requires_grad=True)

            if cuda:
                rnn.cuda()
                input.data = input.data.cuda()
                if is_lstm:
                    hx[0].data = hx[0].data.cuda()
                    hx[1].data = hx[1].data.cuda()
                else:
                    hx.data = hx.data.cuda()

            output, hy = rnn(input, hx)
            # FIXME this is because of a pytorch bug
            if is_lstm:
                fake_loss = 0*(hy[0] + hy[1]).sum()
            else:
                fake_loss = 0*hy.sum()

            loss = output.sum() + fake_loss
            loss.backward()

            return {'output': output.data,
                    'hy': hy[0].data if is_lstm else hy.data,
                    'weights': rnn.all_weights,
                    'grad_input': input.grad,
                    'grad_hx': hx[0].grad if is_lstm else hx.grad,
                    'cy': hy[1].data if is_lstm else None,
                    'grad_cx': hx[1].grad if is_lstm else None}

        input_size = 10
        hidden_size = 6
        num_layers = 2
        seq_length = 7
        batch = 5

        def compare_cpu_gpu(outputs_cpu, outputs_gpu):
            self.assertEqual(list(outputs_cpu.keys()), list(outputs_gpu.keys()))
            for key in outputs_cpu.keys():
                if key != 'weights':
                    self.assertEqual(outputs_cpu[key], outputs_gpu[key], prec=5e-5, message=key)

            # check grad weights separately, as nested dict
            for cpu_layer_weight, gpu_layer_weight in zip(outputs_cpu['weights'], outputs_gpu['weights']):
                for (cpu_weight, gpu_weight) in zip(cpu_layer_weight, gpu_layer_weight):
                    self.assertEqual(cpu_weight.grad, gpu_weight.grad, prec=5e-5)


        input_val = torch.randn(seq_length, batch, input_size)
        for module in (nn.RNN, nn.LSTM, nn.GRU):
            for bias in (True, False):
                for bidirectional in (False, True):
                    for dropout in (0, 1): # Because of dropout randomness, can only compare 0 and 1
                        num_directions = 2 if bidirectional else 1
                        hx_val = torch.randn(num_layers * num_directions, batch, hidden_size)

                        rnn = module(input_size,
                                     hidden_size,
                                     num_layers,
                                     bias=bias,
                                     dropout=dropout,
                                     bidirectional=bidirectional)

                        outputs_cpu = forward_backward(
                            False, rnn, input_val, hx_val, rnn.all_weights)

                        rnn_gpu = module(input_size,
                                         hidden_size,
                                         num_layers,
                                         bias=bias,
                                         dropout=dropout,
                                         bidirectional=bidirectional)

                        outputs_gpu = forward_backward(
                            True, rnn_gpu, input_val, hx_val, rnn.all_weights)

                        compare_cpu_gpu(outputs_cpu, outputs_gpu)

        for nonlinearity in ('tanh', 'relu'):
            hx_val = torch.randn(num_layers, batch, hidden_size)

            rnn = nn.rnn.RNN(input_size, hidden_size, num_layers, bias=bias, nonlinearity=nonlinearity)
            outputs_cpu = forward_backward(False, rnn, input_val, hx_val, rnn.all_weights)

            rnn_gpu = nn.rnn.RNN(input_size, hidden_size, num_layers, bias=bias, nonlinearity=nonlinearity)
            outputs_gpu = forward_backward(True, rnn_gpu, input_val, hx_val, rnn.all_weights)

            compare_cpu_gpu(outputs_cpu, outputs_gpu)

    @unittest.skipIf(not TEST_CUDNN, "needs cudnn")
    def test_RNN_dropout(self):
        # checking the assumption that cuDNN sticks dropout in between
        # RNN layers
        for p in (0, 0.276, 0.731, 1):
            for train in (True, False):
                for cuda in (True, False):
                    rnn = nn.RNN(10, 1000, 2, bias=False, dropout=p, nonlinearity='relu')
                    if cuda:
                        rnn.cuda()

                    if train:
                        rnn.train()
                    else:
                        rnn.eval()
                    rnn.weight_ih_l0.data.fill_(1)
                    rnn.weight_hh_l0.data.fill_(1)
                    rnn.weight_ih_l1.data.fill_(1)
                    rnn.weight_hh_l1.data.fill_(1)
                    input = Variable(torch.Tensor(1,1,10).fill_(1))
                    hx = Variable(torch.Tensor(2,1,1000).fill_(0))
                    if cuda:
                        input = input.cuda()
                        hx = hx.cuda()

                    output, hy = rnn(input, hx)
                    self.assertEqual(output.data.min(), output.data.max())
                    output_val = output.data[0][0][0]
                    if p == 0 or not train:
                        self.assertEqual(output_val, 10000)
                    elif p == 1:
                        self.assertEqual(output_val, 0)
                    else:
                        self.assertGreater(output_val, 8000)
                        self.assertLess(output_val, 12000)
                        denorm_mod = (output_val * (1 - p)) % 10
                        self.assertLess(min(denorm_mod, 10 - denorm_mod), 1e-2)

                    self.assertEqual(hy[0].data.min(), hy[0].data.max())
                    self.assertEqual(hy[1].data.min(), hy[1].data.max())
                    self.assertEqual(hy.data[0][0][0], 10)
                    self.assertEqual(hy.data[1][0][0], output_val)


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
        module_name='ConvTranspose2d',
        constructor_args=(3, 4, 3, (2, 2), 1, (1, 1), 1, False),
        input_size=(1, 3, 7, 7),
        desc='no_bias'
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
            torch.randperm(2).repeat(1, 2),
            requires_grad=False
        ),
        jacobian_input=False
    ),
    dict(
        constructor=lambda: nn.FractionalMaxPool2d(2, output_ratio=0.5, _random_samples=torch.DoubleTensor(1, 3, 2).uniform_()),
        input_size=(1, 3, 5, 5),
        fullname='FractionalMaxPool2d_ratio',
        test_cuda=False
    ),
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
