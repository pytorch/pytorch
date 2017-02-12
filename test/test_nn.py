import math
import torch
import random
import unittest
import contextlib
from copy import deepcopy
from itertools import repeat, product
from functools import wraps

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel as dp
from torch.autograd import Variable
from torch.nn import Parameter
from common_nn import NNTestCase, ModuleTest, CriterionTest, TestBase, \
    module_tests, criterion_tests, TEST_CUDA, TEST_MULTIGPU, TEST_CUDNN, \
    TEST_CUDNN_VERSION, PRECISION
from common import freeze_rng_state, run_tests


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
        self.cudnn = kwargs.get('cudnn', False)
        self.check_inplace = kwargs.get('check_inplace', False)

    def _do_test(self, test_case, module, input):
        test_case.check_jacobian(module, input, self.jacobian_input)

        # check if module can be printed
        module.__repr__()

        if self.check_inplace:
            module_ip = self.constructor(*self.constructor_args, inplace=True)

            input_version = input._version
            output = module(input)
            test_case.assertEqual(input._version, input_version)

            input_ip = deepcopy(input)
            input_ip_clone = input_ip.clone()
            output_ip = module_ip(input_ip_clone)
            test_case.assertNotEqual(input_ip_clone._version, input_version)
            test_case.assertEqual(output, output_ip)
            grad = output.data.clone().normal_()
            output.backward(grad)
            output_ip.backward(grad)
            test_case.assertEqual(output.grad, output_ip.grad)

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

                if self.cudnn:
                    torch.backends.cudnn.enabled = False
                    try:
                        module(input)
                        for p in module.parameters():
                            test_case.assertEqual(type(p.data), torch.cuda.FloatTensor)
                            test_case.assertEqual(p.get_device(), 0)
                    finally:
                        torch.backends.cudnn.enabled = True

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
        if input.grad is None:
            return None
        return input.grad.data

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
            i.grad.data.zero_()
        args = input_tuple + (target,)
        criterion(*args).backward()
        if isinstance(input, tuple):
            return tuple(map(lambda i: i.grad.data, input))
        else:
            return input.grad.data

    def _zero_grad_parameters(self, module):
        if hasattr(module, 'weight') and module.weight is not None:
            module.weight.grad.data.zero_()
        if hasattr(module, 'bias') and module.bias is not None:
            module.bias.grad.data.zero_()

    def _get_parameters(self, module):
        params = []
        d_params = []
        if hasattr(module, 'weight') and module.weight is not None:
            params += [module.weight.data]
            d_params += [module.weight.grad.data]
        if hasattr(module, 'bias') and module.bias is not None:
            params += [module.bias.data]
            d_params += [module.bias.grad.data]
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
            self.assertEqual(grad_output[0].data, torch.ones(5, 5) * 2)
            counter['backwards'] += inc

        test_fwd = module.register_forward_hook(lambda *args: fw_hook(1, *args))

        module(input)
        module(input)
        self.assertEqual(counter['forwards'], 2)
        self.assertEqual(counter['backwards'], 0)

        test_bwd = module.register_backward_hook(lambda *args: bw_hook(1, *args))

        output = module(input)
        self.assertEqual(counter['forwards'], 3)
        self.assertEqual(counter['backwards'], 0)

        output.backward(torch.ones(5, 5) * 2, retain_variables=True)
        self.assertEqual(counter['forwards'], 3)
        self.assertEqual(counter['backwards'], 1)

        output.backward(torch.ones(5, 5) * 2, retain_variables=True)
        self.assertEqual(counter['forwards'], 3)
        self.assertEqual(counter['backwards'], 2)

        test2_fwd = module.register_forward_hook(lambda *args: fw_hook(2, *args))

        output = module(input)
        self.assertEqual(counter['forwards'], 6)
        self.assertEqual(counter['backwards'], 2)

        test2_bwd = module.register_backward_hook(lambda *args: bw_hook(2, *args))

        module(input).backward(torch.ones(5, 5) * 2)
        self.assertEqual(counter['forwards'], 9)
        self.assertEqual(counter['backwards'], 5)

        test2_bwd.remove()

        module(input).backward(torch.ones(5, 5) * 2)
        self.assertEqual(counter['forwards'], 12)
        self.assertEqual(counter['backwards'], 6)

        test2_fwd.remove()

        module(input).backward(torch.ones(5, 5) * 2)
        self.assertEqual(counter['forwards'], 13)
        self.assertEqual(counter['backwards'], 7)

        test_fwd.remove()
        test_bwd.remove()

    def test_hook_fail(self):
        module = nn.Sigmoid()
        input = Variable(torch.randn(5, 5), requires_grad=True)

        def fw_fail1(self, input, output):
            return output

        def fw_fail2(self, input, output):
            return input

        def bw_fail1(self, grad_input, grad_output):
            return grad_input[:-1]

        def bw_fail2(self, grad_input, grad_output):
            return grad_input + (torch.randn(2, 2),)

        with module.register_forward_hook(fw_fail1):
            with self.assertRaises(RuntimeError) as err:
                module(input)
            self.assertIn("fw_fail", err.exception.args[0])
            self.assertIn("didn't return None", err.exception.args[0])

        with module.register_forward_hook(fw_fail2):
            with self.assertRaises(RuntimeError) as err:
                module(input)
            self.assertIn("fw_fail2", err.exception.args[0])
            self.assertIn("didn't return None", err.exception.args[0])

        with module.register_backward_hook(bw_fail1):
            with self.assertRaises(RuntimeError) as err:
                module(input).sum().backward()
            self.assertIn("bw_fail", err.exception.args[0])
            self.assertIn("got 0, but expected 1", err.exception.args[0])

        with module.register_backward_hook(bw_fail2):
            with self.assertRaises(RuntimeError) as err:
                module(input).sum().backward()
            self.assertIn("bw_fail2", err.exception.args[0])
            self.assertIn("got 2, but expected 1", err.exception.args[0])

    def test_hook_writeable(self):
        module = nn.Linear(5, 5)
        input = Variable(torch.randn(5, 5), requires_grad=True)

        def bw_hook(module, grad_input, grad_output):
            for grad in grad_input:
                self.assertIsInstance(grad, Variable)
            for grad in grad_output:
                self.assertIsInstance(grad, Variable)
            return tuple(gi * 2 for gi in grad_input)

        module.register_backward_hook(bw_hook)
        module(input).backward(torch.ones(5, 5))
        expected_grad = torch.ones(5, 5).mm(module.weight.data) * 2
        self.assertEqual(input.grad.data, expected_grad)

    def test_zero_grad(self):
        module = nn.Linear(5, 5)
        for p in module.parameters():
            p.requires_grad = False
        module.zero_grad()

        module.weight.requires_grad = True
        module.weight.grad.data.fill_(1)
        module.zero_grad()
        self.assertEqual(module.weight.grad.data, module.weight.data.clone().zero_())

        module.bias.requires_grad = True
        module.weight.grad.data.fill_(1)
        module.bias.grad.data.fill_(1)
        module.zero_grad()
        self.assertEqual(module.weight.grad.data, module.weight.data.clone().zero_())
        self.assertEqual(module.bias.grad.data, module.bias.data.clone().zero_())

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
        input.fill_(1 - p)

        module = cls(p)
        input_var = Variable(input, requires_grad=True)
        output = module(input_var)
        self.assertLess(abs(output.data.mean() - (1 - p)), 0.05)
        output.backward(input)
        self.assertLess(abs(input_var.grad.data.mean() - (1 - p)), 0.05)

        module = cls(p, True)
        input_var = Variable(input.clone(), requires_grad=True)
        output = module(input_var + 0)
        self.assertLess(abs(output.data.mean() - (1 - p)), 0.05)
        output.backward(input)
        self.assertLess(abs(input_var.grad.data.mean() - (1 - p)), 0.05)

        # Check that these don't raise errors
        module.__repr__()
        str(module)

    def test_parameters(self):
        def num_params(module):
            return len(list(module.parameters()))

        class Net(nn.Module):

            def __init__(self):
                super(Net, self).__init__()
                self.l1 = l
                self.l2 = l
                self.param = Parameter(torch.Tensor(3, 5))
        l = nn.Linear(10, 20)
        n = Net()
        s = nn.Sequential(n, n, n, n)
        self.assertEqual(num_params(l), 2)
        self.assertEqual(num_params(n), 3)
        self.assertEqual(num_params(s), 3)

    def test_modules(self):
        class Net(nn.Module):

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

    def test_ListModule(self):
        modules = [nn.ReLU(), nn.Linear(5, 5)]
        module_list = nn.ModuleList(modules)

        def check():
            self.assertEqual(len(module_list), len(modules))
            for m1, m2 in zip(modules, module_list):
                self.assertIs(m1, m2)
            for m1, m2 in zip(modules, module_list.children()):
                self.assertIs(m1, m2)
            for i in range(len(modules)):
                self.assertIs(module_list[i], modules[i])
        check()
        modules += [nn.Conv2d(3, 4, 3)]
        module_list += [modules[-1]]
        check()
        modules.append(nn.Tanh())
        module_list.append(modules[-1])
        check()
        next_modules = [nn.Linear(5, 5), nn.Sigmoid()]
        modules.extend(next_modules)
        module_list.extend(next_modules)
        check()
        modules[2] = nn.Conv2d(5, 3, 2)
        module_list[2] = modules[2]
        check()

        with self.assertRaises(TypeError):
            module_list += nn.ReLU()
        with self.assertRaises(TypeError):
            module_list.extend(nn.ReLU())

    def test_ParameterList(self):
        make_param = lambda: Parameter(torch.randn(10, 10))
        parameters = [make_param(), make_param()]
        param_list = nn.ParameterList(parameters)

        def check():
            self.assertEqual(len(parameters), len(param_list))
            for p1, p2 in zip(parameters, param_list):
                self.assertIs(p1, p2)
            for p1, p2 in zip(parameters, param_list.parameters()):
                self.assertIs(p1, p2)
            for i in range(len(parameters)):
                self.assertIs(parameters[i], param_list[i])
        check()
        parameters += [make_param()]
        param_list += [parameters[-1]]
        check()
        parameters.append(make_param())
        param_list.append(parameters[-1])
        check()
        next_params = [make_param(), make_param()]
        parameters.extend(next_params)
        param_list.extend(next_params)
        check()
        parameters[2] = make_param()
        param_list[2] = parameters[2]
        check()

        with self.assertRaises(TypeError):
            param_list += make_param()
        with self.assertRaises(TypeError):
            param_list.extend(make_param())

    def test_add_module(self):
        l = nn.Linear(10, 20)
        net = nn.Module()
        net.l = l
        net.l2 = l
        net.add_module('empty', None)
        self.assertEqual(net.l, l)
        self.assertEqual(net.l2, l)
        self.assertEqual(net.empty, None)
        net.add_module('l3', l)
        self.assertEqual(net.l3, l)
        self.assertRaises(KeyError, lambda: net.add_module('l', l))
        self.assertRaises(TypeError, lambda: net.add_module('x', 'non-module'))

    def test_type(self):
        l = nn.Linear(10, 20)
        net = nn.Module()
        net.l = l
        net.l2 = l
        net.add_module('empty', None)
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
        embedding = nn.Embedding(10, 20, padding_idx=0)
        input = Variable(torch.LongTensor([[0, 2, 4, 5], [4, 3, 0, 9]]))
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

    def _test_maxpool_indices(self, num_dim, type=torch.FloatTensor):
        def expected_indices(dim):
            if dim == 1:
                return torch.DoubleTensor([1, 3])
            lower_dim = expected_indices(dim - 1)
            lower_dim = lower_dim.view(1, *lower_dim.size())
            return torch.cat((lower_dim + 4, lower_dim + 12), 0)

        def expected_grad(dim):
            if dim == 1:
                return torch.DoubleTensor([0, 1, 0, 1])
            lower_dim_grad = expected_grad(dim - 1)
            grad = lower_dim_grad.view(1, *lower_dim_grad.size())
            zero = torch.zeros(grad.size())
            return torch.cat((zero, grad, zero, grad), 0)

        module_cls = getattr(nn, 'MaxPool{}d'.format(num_dim))
        module = module_cls(2, return_indices=True).type(type)
        numel = 4 ** num_dim
        input = torch.range(1, numel).view(1, 1, *repeat(4, num_dim)).type(type)
        input_var = Variable(input, requires_grad=True)

        # Check forward
        output, indices = module(input_var)
        if num_dim != 3:
            expected_indices = expected_indices(num_dim)
            expected_output = expected_indices + 1
            self.assertEqual(indices.dim(), input.dim())
            self.assertEqual(indices.data.squeeze(), expected_indices)
            self.assertEqual(output.data.squeeze(), expected_output)
        self.assertTrue(output.requires_grad)
        self.assertFalse(indices.requires_grad)

        # Make sure backward works
        grad_output = torch.ones(output.size()).type(type)
        output.backward(grad_output, retain_variables=True)
        expected_grad = expected_grad(num_dim)
        self.assertEqual(input_var.grad.data, expected_grad.view_as(input))

        # Make sure backward after changing indices will result in an error
        indices.add_(1)
        self.assertRaises(RuntimeError, lambda: output.backward(grad_output))

    def test_MaxPool1d_indices(self):
        self._test_maxpool_indices(1)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_MaxPool1d_indices_cuda(self):
        self._test_maxpool_indices(1, torch.cuda.FloatTensor)

    def test_MaxPool2d_indices(self):
        self._test_maxpool_indices(2)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_MaxPool2d_indices_cuda(self):
        self._test_maxpool_indices(2, torch.cuda.FloatTensor)

    def test_MaxPool3d_indices(self):
        self._test_maxpool_indices(3)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_MaxPool3d_indices_cuda(self):
        self._test_maxpool_indices(3, torch.cuda.FloatTensor)

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
        self.assertEqual(x.grad.data[:2], grad)
        self.assertEqual(x.grad.data[2:], grad.clone().zero_())

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
        self.assertEqual(inputs[0].grad.data, grad[:2])
        self.assertEqual(inputs[1].grad.data, grad[2:])

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
        net = nn.Module()
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

        class Net(nn.Module):

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

        class Net(nn.Module):

            def forward(self, input):
                return fn(input)
        i = Variable(torch.randn(20, 3).float().cuda(1))
        input = (i.cos(), (i.sin(), i), i.sin())
        gpus = range(torch.cuda.device_count())
        output = dp.data_parallel(Net(), input, gpus)
        self.assertEqual(output, fn(input))

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_data_parallel_module(self):
        l = nn.Linear(10, 5).float().cuda()
        i = Variable(torch.randn(20, 10).float().cuda())
        expected_out = l(i).data
        net = nn.DataParallel(l)
        out = net(i)
        self.assertEqual(out.get_device(), 0)
        self.assertEqual(out.data, expected_out)

    def test_state_dict(self):
        l = nn.Linear(5, 5)
        block = nn.Module()
        block.conv = nn.Conv2d(3, 3, 3, bias=False)
        net = nn.Module()
        net.linear1 = l
        net.linear2 = l
        net.bn = nn.BatchNorm2d(2)
        net.block = block
        net.add_module('empty', None)

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
                if isinstance(param, Parameter):
                    param = param.data
            self.assertIs(v, param)

        l = nn.Linear(5, 5)
        state_dict = l.state_dict()
        self.assertEqual(len(state_dict), 2)
        self.assertIs(state_dict['weight'], l.weight.data)
        self.assertIs(state_dict['bias'], l.bias.data)

    def test_load_state_dict(self):
        l = nn.Linear(5, 5)
        block = nn.Module()
        block.conv1 = nn.Conv2d(3, 3, 3, bias=True)
        block.conv2 = nn.Conv2d(3, 3, 3, bias=False)
        net = nn.Module()
        net.linear1 = l
        net.linear2 = l
        net.bn = nn.BatchNorm2d(2)
        net.block = block
        net.add_module('empty', None)

        state_dict = net.state_dict()
        state_dict.update({
            'linear1.weight': torch.ones(5, 5),
            'block.conv1.bias': torch.range(1, 3),
            'bn.running_mean': torch.randn(2),
        })
        net.load_state_dict(state_dict)
        self.assertEqual(net.linear1.weight.data, state_dict['linear1.weight'])
        self.assertEqual(net.block.conv1.bias.data, state_dict['block.conv1.bias'])
        self.assertEqual(net.bn.running_mean, state_dict['bn.running_mean'])

        state_dict = net.state_dict()
        state_dict.update({'extra': torch.ones(5)})
        self.assertRaises(KeyError, lambda: net.load_state_dict(state_dict))

        state_dict = net.state_dict()
        del state_dict['linear1.weight']
        self.assertRaises(KeyError, lambda: net.load_state_dict(state_dict))

    def test_parameter_assignment(self):
        l = nn.Linear(5, 5)

        def num_params():
            return len(list(l.parameters()))
        self.assertEqual(num_params(), 2)

        new_param = Parameter(torch.randn(5, 5))
        l.param_name = new_param
        self.assertEqual(num_params(), 3)
        self.assertObjectIn(new_param, l.parameters())

        var = Variable(torch.randn(5, 5))
        l.var_name = var
        self.assertEqual(num_params(), 3)
        self.assertNotIn(id(var), map(id, l.parameters()))

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

    @unittest.skipIf(not TEST_CUDA, 'CUDA not available')
    def test_Conv2d_large_workspace(self):
        # These sizes require huge cuDNN workspaces. Make sure we choose a
        # reasonable algorithm that does not run out of memory
        sizes = [
            (1, 256, 109, 175),
            (1, 256, 80, 128),
            (1, 256, 120, 192),
        ]
        dtype = torch.cuda.FloatTensor

        def run_test(benchmark):
            torch.backends.cudnn.benchmark = benchmark
            conv = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1).type(dtype)
            for size in sizes:
                x = torch.randn(size).type(dtype)
                out = conv(Variable(x, requires_grad=True))
                out.backward(torch.ones(out.size()).type(dtype))

        b = torch.backends.cudnn.benchmark
        try:
            run_test(benchmark=False)
            run_test(benchmark=True)
        finally:
            torch.backends.cudnn.benchmark = b

    def test_ConvTranspose2d_output_size(self):
        m = nn.ConvTranspose2d(3, 4, 3, 3, 0, 2)
        i = Variable(torch.randn(2, 3, 6, 6))
        for h in range(15, 22):
            for w in range(15, 22):
                if 18 <= h <= 20 and 18 <= w <= 20:
                    output = m(i, output_size=(h, w))
                    self.assertEqual(output.size()[2:], (h, w))
                else:
                    self.assertRaises(ValueError, lambda: m(i, (h, w)))

    def test_Conv2d_naive_groups(self):
        # Check that grouped convolutions matches two half convolutions
        m = nn.Conv2d(4, 4, kernel_size=3, groups=2)
        i = Variable(torch.randn(2, 4, 6, 6), requires_grad=True)
        output = m(i)
        grad_output = torch.randn(2, 4, 4, 4)
        output.backward(grad_output)

        m1 = nn.Conv2d(2, 2, kernel_size=3)
        m1.weight.data.copy_(m.weight.data[:2])
        m1.bias.data.copy_(m.bias.data[:2])
        i1 = Variable(i.data[:, :2].contiguous(), requires_grad=True)
        output1 = m1(i1)
        output1.backward(grad_output[:, :2].contiguous())

        m2 = nn.Conv2d(2, 2, kernel_size=3)
        m2.weight.data.copy_(m.weight.data[2:])
        m2.bias.data.copy_(m.bias.data[2:])
        i2 = Variable(i.data[:, 2:].contiguous(), requires_grad=True)
        output2 = m2(i2)
        output2.backward(grad_output[:, 2:].contiguous())

        self.assertEqual(output, torch.cat([output1, output2], 1))
        self.assertEqual(i.grad.data,
                         torch.cat([i1.grad.data, i2.grad.data], 1))
        self.assertEqual(m.bias.grad.data,
                         torch.cat([m1.bias.grad.data, m2.bias.grad.data], 0))
        self.assertEqual(m.weight.grad.data,
                         torch.cat([m1.weight.grad.data, m2.weight.grad.data], 0))

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
                small_t[:, :, i, j] = 100
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

    def test_container_copy(self):
        class Model(nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.linear = nn.Linear(4, 5)

            def forward(self, input):
                return self.linear(input)

        input = Variable(torch.randn(2, 4))

        model = Model()
        model_cp = deepcopy(model)
        self.assertEqual(model(input).data, model_cp(input).data)

        model_cp.linear.weight[:] = 2
        self.assertNotEqual(model(input).data, model_cp(input).data)

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

    def test_invalid_dropout_p(self):
        v = Variable(torch.ones(1))
        self.assertRaises(ValueError, lambda: nn.Dropout(-0.1))
        self.assertRaises(ValueError, lambda: nn.Dropout(1.1))
        self.assertRaises(ValueError, lambda: nn.Dropout2d(-0.1))
        self.assertRaises(ValueError, lambda: nn.Dropout2d(1.1))
        self.assertRaises(ValueError, lambda: nn.Dropout3d(-0.1))
        self.assertRaises(ValueError, lambda: nn.Dropout3d(1.1))
        self.assertRaises(ValueError, lambda: F.dropout(v, -0.1))
        self.assertRaises(ValueError, lambda: F.dropout(v, 1.1))

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

            (hx + cx).sum().backward()

    def test_rnn_initial_hidden_state(self):
        rnn_modes = ['RNN', 'GRU', 'LSTM']
        for mode in rnn_modes:
            rnn = getattr(nn, mode)(30, 20, 2)
            input = Variable(torch.randn(10, 32, 30))
            hidden = Variable(torch.Tensor(2, 32, 20).zero_())
            if mode is 'LSTM':
                hidden = (hidden, hidden)
            output1, hidden1 = rnn(input, hidden)
            output2, hidden2 = rnn(input)
            self.assertEqual(output1, output2)
            self.assertEqual(hidden1, hidden2)

    def _test_rnn_retain_variables(self, dtype):
        rnn = nn.LSTM(10, 20, num_layers=2).type(dtype)
        input = Variable(torch.randn(5, 6, 10).type(dtype), requires_grad=True)
        output = rnn(input)
        output[0].sum().backward(retain_variables=True)
        grads = [input.grad.data.clone()] + [p.grad.data.clone() for p in rnn.parameters()]
        rnn.zero_grad()
        input.grad.data.zero_()
        output[0].sum().backward(retain_variables=True)
        grads2 = [input.grad.data] + [p.grad.data for p in rnn.parameters()]
        self.assertEqual(grads, grads2)

    def test_rnn_retain_variables(self):
        self._test_rnn_retain_variables(torch.DoubleTensor)

    @unittest.skipIf(not TEST_CUDA, 'CUDA not available')
    def test_rnn_retain_variables_cuda(self):
        try:
            torch.backends.cudnn.enabled = False
            self._test_rnn_retain_variables(torch.cuda.FloatTensor)
        finally:
            torch.backends.cudnn.enabled = True
        self._test_rnn_retain_variables(torch.cuda.FloatTensor)

    def _test_RNN_cpu_vs_cudnn(self, dropout):

        def forward_backward(cuda, rnn, input_val, hx_val, grad_output, grad_hy, weights_val):
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
                grad_output = grad_output.cuda()
                grad_hy = grad_hy.cuda()

            output, hy = rnn(input, hx)

            if is_lstm:
                torch.autograd.backward([output + 0, hy[0] + 0, hy[1] + 0], [grad_output, grad_hy, grad_hy + 1])
            else:
                torch.autograd.backward([output + 0, hy + 0], [grad_output, grad_hy])

            return {'output': output.data,
                    'hy': hy[0].data if is_lstm else hy.data,
                    'weights': rnn.all_weights,
                    'grad_input': input.grad.data,
                    'grad_hx': hx[0].grad.data if is_lstm else hx.grad.data,
                    'cy': hy[1].data if is_lstm else None,
                    'grad_cx': hx[1].grad.data if is_lstm else None}

        input_size = 10
        hidden_size = 6
        num_layers = 2
        seq_length = 7
        batch = 5

        def make_noncontig(tensor):
            ndim = tensor.dim()
            return torch.stack([tensor.clone().zero_(), tensor], ndim).select(ndim, 1)

        def compare_cpu_gpu(outputs_cpu, outputs_gpu):
            self.assertEqual(list(outputs_cpu.keys()), list(outputs_gpu.keys()))
            for key in outputs_cpu.keys():
                if key != 'weights':
                    self.assertEqual(outputs_cpu[key], outputs_gpu[key], prec=5e-5, message=key)

            # check grad weights separately, as nested dict
            for cpu_layer_weight, gpu_layer_weight in zip(outputs_cpu['weights'], outputs_gpu['weights']):
                for (cpu_weight, gpu_weight) in zip(cpu_layer_weight, gpu_layer_weight):
                    self.assertEqual(cpu_weight.grad.data, gpu_weight.grad.data, prec=5e-5)

        for module in (nn.RNN, nn.LSTM, nn.GRU):
            for bias, bidirectional, batch_first, contig in product((True, False), repeat=4):
                num_directions = 2 if bidirectional else 1
                if batch_first:
                    input_val = torch.randn(batch, seq_length, input_size)
                    grad_output = torch.randn(batch, seq_length, hidden_size * num_directions)
                else:
                    input_val = torch.randn(seq_length, batch, input_size)
                    grad_output = torch.randn(seq_length, batch, hidden_size * num_directions)
                hx_val = torch.randn(num_layers * num_directions, batch, hidden_size)
                grad_hy = torch.randn(num_layers * num_directions, batch, hidden_size)

                if not contig:
                    grad_output = make_noncontig(grad_output)
                    grad_hy = make_noncontig(grad_hy)
                    input_var = make_noncontig(input_val)
                    hx_val = make_noncontig(hx_val)

                rnn = module(input_size,
                             hidden_size,
                             num_layers,
                             bias=bias,
                             dropout=dropout,
                             bidirectional=bidirectional,
                             batch_first=batch_first)

                outputs_cpu = forward_backward(
                    False, rnn, input_val, hx_val, grad_output, grad_hy, rnn.all_weights)

                rnn_gpu = module(input_size,
                                 hidden_size,
                                 num_layers,
                                 bias=bias,
                                 dropout=dropout,
                                 bidirectional=bidirectional,
                                 batch_first=batch_first)

                outputs_gpu = forward_backward(
                    True, rnn_gpu, input_val, hx_val, grad_output, grad_hy, rnn.all_weights)

                compare_cpu_gpu(outputs_cpu, outputs_gpu)

        for nonlinearity in ('tanh', 'relu'):
            hx_val = torch.randn(num_layers, batch, hidden_size)
            input_val = torch.randn(seq_length, batch, input_size)
            grad_output = torch.randn(seq_length, batch, hidden_size * num_directions)
            grad_hy = torch.randn(num_layers * num_directions, batch, hidden_size)

            rnn = nn.rnn.RNN(input_size, hidden_size, num_layers, bias=bias, nonlinearity=nonlinearity)
            outputs_cpu = forward_backward(False, rnn, input_val, hx_val, grad_output, grad_hy, rnn.all_weights)

            rnn_gpu = nn.rnn.RNN(input_size, hidden_size, num_layers, bias=bias, nonlinearity=nonlinearity)
            outputs_gpu = forward_backward(True, rnn_gpu, input_val, hx_val, grad_output, grad_hy, rnn.all_weights)

            compare_cpu_gpu(outputs_cpu, outputs_gpu)

    @unittest.skipIf(not TEST_CUDNN, "needs cudnn")
    @default_tensor_type(torch.FloatTensor)  # FIXME: just until torch.cuda.DoubleTensor.sum() implemented
    def test_RNN_cpu_vs_cudnn_no_dropout(self):
        self._test_RNN_cpu_vs_cudnn(0)

    @unittest.skipIf(not (TEST_CUDNN and TEST_CUDNN_VERSION >= 5103), "needs cudnn >= 5.1")
    @default_tensor_type(torch.FloatTensor)  # FIXME: just until torch.cuda.DoubleTensor.sum() implemented
    def test_RNN_cpu_vs_cudnn_with_dropout(self):
        # Because of dropout randomness, can only compare dropout=0 and dropout=1
        self._test_RNN_cpu_vs_cudnn(1)

    @unittest.skipIf(not (TEST_CUDNN and TEST_CUDNN_VERSION >= 5103), "needs cudnn >= 5.1")
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
                    input = Variable(torch.Tensor(1, 1, 10).fill_(1))
                    hx = Variable(torch.Tensor(2, 1, 1000).fill_(0))
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

    @unittest.skipIf(not (TEST_CUDNN and TEST_CUDNN_VERSION >= 5103), "needs cudnn >= 5.1")
    def test_RNN_dropout_state(self):
        import sys
        if sys.version_info[0] == 2:
            import cPickle as pickle
        else:
            import pickle
        for p in (0, 0.1234):
            for train in (True, False):
                for cuda in (True, False):
                    rnn = nn.RNN(100, 100, 2, bias=False, dropout=p, nonlinearity='relu')
                    if cuda:
                        rnn.cuda()

                    if train:
                        rnn.train()
                    else:
                        rnn.eval()
                    input = Variable(torch.Tensor(1, 1, 100).uniform_())
                    hx = Variable(torch.Tensor(2, 1, 100).uniform_())
                    if cuda:
                        input = input.cuda()
                        hx = hx.cuda()

                    output1, hy1 = rnn(input, hx)
                    output2, hy2 = rnn(input, hx)

                    rnn_pickle = pickle.dumps(rnn)
                    rnn2 = pickle.loads(rnn_pickle)
                    output3, hy3 = rnn2(input, hx)

                    if p == 0 or not train:
                        self.assertEqual(output1, output2)
                        self.assertEqual(output1, output3)
                        self.assertEqual(hy1, hy2)
                        self.assertEqual(hy1, hy3)
                    else:
                        self.assertNotEqual(output1, output2)
                        self.assertNotEqual(output1, output3)
                        self.assertNotEqual(hy1, hy2)
                        self.assertNotEqual(hy1, hy3)

    def _verify_pixel_shuffle(self, input, output, upscale_factor):
        for c in range(output.size(1)):
            for h in range(output.size(2)):
                for w in range(output.size(3)):
                    height_idx = h // upscale_factor
                    weight_idx = w // upscale_factor
                    channel_idx = (upscale_factor * (h % upscale_factor)) + (w % upscale_factor) + \
                                  (c * upscale_factor ** 2)
                    self.assertEqual(output[:, c, h, w], input[:, channel_idx, height_idx, weight_idx])

    def test_inplace_thnn(self):
        r = nn.ReLU(True)
        input = Variable(torch.randn(5, 5), requires_grad=True)
        output = r(input + 0)
        grad_output = torch.randn(5, 5)
        grad_output_clone = grad_output.clone()
        output.backward(grad_output)
        self.assertEqual(grad_output, grad_output_clone)

    @unittest.skipIf(not TEST_CUDA, 'CUDA not available')
    def test_noncontig_conv_grad(self):
        # FIXME: remove after adding non-contiguous grad tests for all modules
        module = nn.Conv2d(3, 5, kernel_size=3, padding=1).cuda()
        input = Variable(torch.randn(2, 3, 10, 10).cuda(), requires_grad=True)
        output = module(input)

        grad = torch.randn(2, 2, 5, 10, 10).cuda()[:, 1]
        assert not grad.is_contiguous()
        output.backward(grad, retain_variables=True)
        result = output.grad.data.clone()
        output.grad.data.zero_()

        output.backward(grad.contiguous())
        self.assertEqual(result, output.grad.data)

    def test_pixel_shuffle(self):
        batch_size = random.randint(1, 3)
        upscale_factor = random.randint(2, 5)
        channels = random.randint(1, 4) * upscale_factor ** 2
        height = random.randint(5, 10)
        width = random.randint(5, 10)

        input = Variable(torch.Tensor(batch_size, channels, height, width).uniform_(), requires_grad=True)
        ps = nn.PixelShuffle(upscale_factor)
        output = ps(input)
        self._verify_pixel_shuffle(input.data, output.data, upscale_factor)
        output.backward(output.data)
        self.assertEqual(input.data, input.grad.data)

    def test_batchnorm_eval(self):
        types = (torch.FloatTensor,)
        if TEST_CUDA:
            types += (torch.cuda.FloatTensor,)
        for tp in types:
            module = nn.BatchNorm1d(3).type(tp)
            module.eval()

            data = Variable(torch.rand(4, 3).type(tp), requires_grad=True)
            grad = torch.rand(4, 3).type(tp)

            # 1st pass
            res1 = module(data)
            res1.backward(grad)
            grad1 = data.grad.data.clone()

            # 2nd pass
            data.grad.data.zero_()

            res2 = module(data)
            res2.backward(grad)
            grad2 = data.grad.data.clone()
            self.assertEqual(res1, res2)
            self.assertEqual(grad1, grad2)


def add_test(test):
    test_name = test.get_name()
    cuda_test_name = test_name + '_cuda'
    if hasattr(TestNN, test_name):
        raise RuntimeError('Found two tests with the same name: ' + test_name)
    if hasattr(TestNN, cuda_test_name):
        raise RuntimeError('Found two tests with the same name: ' + cuda_test_name)
    setattr(TestNN, test_name, lambda self, test=test: test(self))
    setattr(TestNN, cuda_test_name, lambda self, test=test: test.test_cuda(self))


new_module_tests = [
    dict(
        module_name='BatchNorm1d',
        constructor_args=(10,),
        input_size=(4, 10),
        cudnn=True,
        desc='affine'
    ),
    dict(
        module_name='BatchNorm1d',
        constructor_args=(5,),
        input_size=(4, 5, 3),
        cudnn=True,
        desc='3d_input'
    ),
    dict(
        module_name='BatchNorm1d',
        constructor_args=(10, 1e-3, 0.3, False),
        input_size=(4, 10),
        cudnn=True,
        desc='not_affine'
    ),
    dict(
        module_name='BatchNorm2d',
        constructor_args=(3,),
        input_size=(2, 3, 6, 6),
        cudnn=True,
    ),
    dict(
        module_name='BatchNorm2d',
        constructor_args=(3, 1e-3, 0.8),
        input_size=(2, 3, 6, 6),
        cudnn=True,
        desc='momentum',
    ),
    dict(
        module_name='BatchNorm2d',
        constructor_args=(3, 1e-3, 0.8, False),
        input_size=(2, 3, 6, 6),
        cudnn=True,
        desc='no_affine',
    ),
    dict(
        module_name='BatchNorm3d',
        constructor_args=(3,),
        input_size=(2, 3, 4, 4, 4),
        cudnn=True,
    ),
    dict(
        module_name='BatchNorm3d',
        constructor_args=(3, 1e-3, 0.7),
        input_size=(2, 3, 4, 4, 4),
        cudnn=True,
        desc='momentum'
    ),
    dict(
        module_name='BatchNorm3d',
        constructor_args=(3, 1e-3, 0.7, False),
        input_size=(2, 3, 4, 4, 4),
        cudnn=True,
        desc='no_affine'
    ),
    dict(
        module_name='Conv1d',
        constructor_args=(4, 5, 3),
        input_size=(2, 4, 10),
        cudnn=True,
    ),
    dict(
        module_name='Conv1d',
        constructor_args=(4, 5, 3),
        input_size=(2, 4, 10),
        cudnn=True,
        desc='stride'
    ),
    dict(
        fullname='Conv1d_dilated',
        constructor=lambda: nn.Conv1d(4, 5, kernel_size=3, dilation=2),
        input_size=(2, 4, 10),
    ),
    dict(
        fullname='Conv1d_groups',
        constructor=lambda: nn.Conv1d(4, 6, kernel_size=3, groups=2),
        input_size=(2, 4, 6),
        cudnn=True,
    ),
    dict(
        module_name='ConvTranspose1d',
        constructor_args=(3, 4, 3, (3,), 1, (1,)),
        cudnn=True,
        input_size=(1, 3, 7)
    ),
    dict(
        module_name='ConvTranspose1d',
        constructor_args=(3, 4, 3, 2, 1, 1, 1, False),
        input_size=(1, 3, 6),
        cudnn=True,
        desc='no_bias'
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
        constructor_args=(3, 4, (3, 2)),
        input_size=(2, 3, 7, 5),
        cudnn=True,
    ),
    dict(
        module_name='Conv2d',
        constructor_args=(3, 4, (3, 3), (2, 2)),
        input_size=(2, 3, 6, 6),
        cudnn=True,
        desc='strided'
    ),
    dict(
        module_name='Conv2d',
        constructor_args=(3, 4, (3, 3), (2, 2), (1, 1)),
        input_size=(2, 3, 6, 6),
        cudnn=True,
        desc='padding'
    ),
    dict(
        module_name='Conv2d',
        constructor_args=(3, 2, (3, 3), (2, 2), (1, 1), (2, 2)),
        input_size=(2, 3, 8, 8),
        cudnn=True,
        desc='dilated'
    ),
    dict(
        module_name='Conv2d',
        constructor_args=(3, 4, (3, 2), 1, 0, 1, 1, False),
        input_size=(2, 3, 6, 5),
        cudnn=True,
        desc='no_bias',
    ),
    dict(
        fullname='Conv2d_groups',
        constructor=lambda: nn.Conv2d(4, 6, (3, 2), groups=2),
        input_size=(2, 4, 6, 5),
        cudnn=True,
    ),
    dict(
        fullname='Conv2d_groups_thnn',
        constructor=lambda: nn.Conv2d(4, 6, (3, 2), groups=2),
        input_size=(2, 4, 6, 5),
    ),
    dict(
        module_name='ConvTranspose2d',
        constructor_args=(3, 4, 3, (3, 2), 1, (1, 1)),
        cudnn=True,
        input_size=(1, 3, 7, 6)
    ),
    dict(
        module_name='ConvTranspose2d',
        constructor_args=(3, 4, 3, (2, 3), 1, (1, 1), 1, False),
        input_size=(1, 3, 6, 7),
        cudnn=True,
        desc='no_bias'
    ),
    dict(
        fullname='ConvTranspose2d_groups',
        constructor=lambda: nn.ConvTranspose2d(2, 4, (2, 3), groups=2),
        input_size=(1, 2, 4, 5),
        cudnn=True,
    ),
    dict(
        module_name='MaxPool2d',
        constructor_args=((3, 3), (2, 2), (1, 1)),
        input_size=(1, 3, 7, 7)
    ),
    dict(
        module_name='AvgPool1d',
        constructor_args=(2,),
        input_size=(2, 3, 6),
    ),
    dict(
        module_name='AvgPool1d',
        constructor_args=((2,), (2,)),
        input_size=(2, 3, 6),
        desc='stride',
    ),
    dict(
        module_name='AvgPool1d',
        constructor_args=(2, 2, 1),
        input_size=(2, 3, 6),
        desc='stride_pad',
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
        constructor_args=(3, 4, (2, 3, 4)),
        input_size=(2, 3, 3, 4, 5),
        cudnn=True,
    ),
    dict(
        module_name='Conv3d',
        constructor_args=(3, 4, 2, 2),
        input_size=(2, 3, 5, 5, 5),
        cudnn=True,
        desc='stride'
    ),
    dict(
        module_name='Conv3d',
        constructor_args=(3, 4, 2, 2, 1),
        input_size=(2, 3, 5, 5, 5),
        cudnn=True,
        desc='stride_padding'
    ),
    dict(
        fullname='Conv3d_groups',
        constructor=lambda: nn.Conv3d(4, 6, kernel_size=3, groups=2),
        input_size=(2, 4, 4, 5, 4),
        cudnn=True,
    ),
    dict(
        fullname='Conv3d_dilated',
        constructor=lambda: nn.Conv3d(3, 4, kernel_size=2, dilation=2),
        input_size=(2, 3, 5, 5, 5),
    ),
    dict(
        module_name='ConvTranspose3d',
        constructor_args=(2, 3, (2, 3, 2)),
        cudnn=True,
        input_size=(1, 2, 4, 5, 4)
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
        input=Variable(torch.randperm(2).repeat(1, 2)),
        jacobian_input=False
    ),
    dict(
        constructor=lambda: nn.Embedding(4, 3, sparse=True),
        input=Variable(torch.randperm(2).repeat(1, 2)),
        jacobian_input=False,
        fullname='Embedding_sparse',
        test_cuda=False,
    ),
    dict(
        constructor=lambda: nn.FractionalMaxPool2d(
            2, output_ratio=0.5, _random_samples=torch.DoubleTensor(1, 3, 2).uniform_()),
        input_size=(1, 3, 5, 5),
        fullname='FractionalMaxPool2d_ratio',
        test_cuda=False
    ),
    dict(
        constructor=lambda: nn.FractionalMaxPool2d((2, 2), output_size=(
            4, 4), _random_samples=torch.DoubleTensor(1, 3, 2).uniform_()),
        input_size=(1, 3, 7, 7),
        fullname='FractionalMaxPool2d_size',
        test_cuda=False
    ),
    dict(
        module_name='PixelShuffle',
        constructor_args=(3,),
        input_size=(1, 9, 4, 4),
    ),
    dict(
        module_name='UpsamplingNearest2d',
        constructor_args=(12,),
        input_size=(1, 2, 4, 4),
    ),
    dict(
        module_name='UpsamplingNearest2d',
        constructor_args=((12, 16)),
        input_size=(1, 2, 3, 4),
        desc='tuple'
    ),
    dict(
        module_name='UpsamplingNearest2d',
        constructor_args=(None, 4),
        input_size=(1, 2, 4, 4),
        desc='scale'
    ),
    dict(
        module_name='UpsamplingBilinear2d',
        constructor_args=(12,),
        input_size=(1, 2, 4, 4),
    ),
    dict(
        module_name='UpsamplingBilinear2d',
        constructor_args=((4, 6)),
        input_size=(1, 2, 2, 3),
        desc='tuple'
    ),
    dict(
        module_name='UpsamplingBilinear2d',
        constructor_args=(None, 4),
        input_size=(1, 2, 4, 4),
        desc='scale'
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


class UnpoolingNet(nn.Module):

    def __init__(self, pool, unpool):
        super(UnpoolingNet, self).__init__()
        self.pool = pool
        self.unpool = unpool

    def forward(self, input):
        return self.unpool(*self.pool(input))


add_test(NewModuleTest(
    constructor=lambda: UnpoolingNet(
        nn.MaxPool1d(2, return_indices=True),
        nn.MaxUnpool1d(2)),
    input_size=(1, 1, 4),
    fullname='MaxUnpool1d_net'))
add_test(NewModuleTest(
    constructor=lambda: UnpoolingNet(
        nn.MaxPool2d(2, return_indices=True),
        nn.MaxUnpool2d(2)),
    input_size=(1, 1, 2, 4),
    fullname='MaxUnpool2d_net'))
add_test(NewModuleTest(
    constructor=lambda: UnpoolingNet(
        nn.MaxPool3d(2, return_indices=True),
        nn.MaxUnpool3d(2)),
    input_size=(1, 1, 2, 4, 6),
    fullname='MaxUnpool3d_net'))


if __name__ == '__main__':
    run_tests()
