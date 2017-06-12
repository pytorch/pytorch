import math
import random
import string
import unittest
import itertools
import contextlib
from copy import deepcopy
from itertools import repeat, product
from functools import wraps, reduce
from operator import mul

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel as dp
import torch.nn.init as init
import torch.nn.utils.rnn as rnn_utils
import torch.legacy.nn as legacy
from torch.nn.utils import clip_grad_norm
from torch.autograd import Variable, gradcheck
from torch.nn import Parameter
from common_nn import NNTestCase, ModuleTest, CriterionTest, TestBase, \
    module_tests, criterion_tests, TEST_CUDA, TEST_MULTIGPU, TEST_CUDNN, \
    TEST_CUDNN_VERSION
from common import freeze_rng_state, run_tests, TestCase, skipIfNoLapack, TEST_SCIPY

if TEST_SCIPY:
    from scipy import stats


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
            with freeze_rng_state():
                output = module(input)
            test_case.assertEqual(input._version, input_version)

            input_ip = deepcopy(input)
            input_ip_clone = input_ip.clone()
            with freeze_rng_state():
                output_ip = module_ip(input_ip_clone)
            test_case.assertNotEqual(input_ip_clone._version, input_version)
            test_case.assertEqual(output, output_ip)
            grad = output.data.clone().normal_()
            input.grad.data.zero_()
            output.backward(grad)
            output_ip.backward(grad)
            test_case.assertEqual(input.grad, input_ip.grad)

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
        output.backward(grad_output, retain_graph=True)
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
            if i.grad is not None:
                i.grad.data.zero_()
        args = input_tuple + (target,)
        criterion(*args).backward()
        if isinstance(input, tuple):
            return tuple(map(lambda i: i.grad.data, input))
        else:
            return input.grad.data

    def _zero_grad_parameters(self, module):
        if hasattr(module, 'weight') and module.weight is not None:
            if module.weight.grad is not None:
                module.weight.grad.data.zero_()
                module.weight.grad.detach_()
        if hasattr(module, 'bias') and module.bias is not None:
            if module.bias.grad is not None:
                module.bias.grad.data.zero_()
                module.bias.grad.detach_()

    def _get_parameters(self, module):
        params = []
        d_params = []
        for p in module.parameters():
            if p.grad is None:
                p._grad = Variable(p.data.clone().zero_(), volatile=True)
            params.append(p.data)
            d_params.append(p.grad.data)
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

        test_bwd = module.register_backward_hook(
            lambda *args: bw_hook(1, *args))

        output = module(input)
        self.assertEqual(counter['forwards'], 3)
        self.assertEqual(counter['backwards'], 0)

        output.backward(torch.ones(5, 5) * 2, retain_graph=True)
        self.assertEqual(counter['forwards'], 3)
        self.assertEqual(counter['backwards'], 1)

        output.backward(torch.ones(5, 5) * 2, retain_graph=True)
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

    def test_hook_cpp(self):
        counter = [0]
        bn = nn.BatchNorm1d(5)

        def hook(module, grad_inputs, grad_outputs):
            counter[0] += 1
            self.assertEqual(len(grad_inputs), 3)
            self.assertEqual(len(grad_outputs), 1)
            self.assertEqual(module, bn)

        bn.register_backward_hook(hook)
        output = bn(Variable(torch.randn(5, 5), requires_grad=True))
        output.sum().backward()

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
        i = Variable(torch.randn(2, 5), requires_grad=True)
        module = nn.Linear(5, 5)
        for p in module.parameters():
            p.requires_grad = False
        module.zero_grad()

        module.weight.requires_grad = True
        module.zero_grad()
        self.assertIsNone(module.weight.grad)  # uninitialized grad

        module(i).sum().backward()
        self.assertIsNotNone(module.weight.grad)
        self.assertGreater(module.weight.grad.data.abs().sum(), 0)
        module.zero_grad()
        self.assertEqual(module.weight.grad.data, module.weight.data.clone().zero_())

        module.bias.requires_grad = True
        module.zero_grad()
        self.assertIsNotNone(module.weight.grad)
        self.assertIsNone(module.bias.grad)
        module(i).sum().backward()
        self.assertIsNotNone(module.weight.grad)
        self.assertIsNotNone(module.bias.grad)
        self.assertGreater(module.weight.grad.data.abs().sum(), 0)
        self.assertGreater(module.bias.grad.data.abs().sum(), 0)
        module.zero_grad()
        self.assertEqual(module.weight.grad.data, module.weight.data.clone().zero_())
        self.assertEqual(module.bias.grad.data, module.bias.data.clone().zero_())

        # non-volatile grad should be zeroed out of place
        initial = module.weight.grad = Variable(torch.ones(5, 5))
        module.zero_grad()
        self.assertIsNot(module.weight.grad, initial)
        self.assertEqual(module.weight.grad.data, torch.zeros(5, 5))

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

    def test_named_parameters(self):
        def num_params(module):
            return len(dict(module.named_parameters()))

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.l1 = l
                self.l2 = l
                self.param = Parameter(torch.Tensor(3, 5))

        l = nn.Linear(10, 20)
        n = Net()
        s = nn.Sequential(n, n, n, n)

        for name in dict(l.named_parameters()).keys():
            self.assertTrue(name in ['bias', 'weight'])

        for name in dict(n.named_parameters()).keys():
            self.assertTrue(name in ['l1.bias', 'l1.weight', 'param'])

        for name in dict(s.named_parameters()).keys():
            self.assertTrue(name in ['0.l1.bias', '0.l1.weight', '0.param'])

        self.assertEqual(num_params(l), 2)
        self.assertEqual(num_params(n), 3)
        self.assertEqual(num_params(s), 3)

    def test_children(self):
        l1 = nn.Linear(2, 2)
        l2 = nn.Linear(2, 2)
        l3 = nn.Linear(2, 2)
        l4 = nn.Linear(2, 2)
        subnet = nn.Sequential(l3, l4)
        s = nn.Sequential(l1, l2, l1, l2, subnet)
        self.assertEqual(list(s.children()), [l1, l2, subnet])

    def test_dir(self):
        linear = nn.Linear(2, 2)
        linear._test_submodule = nn.Linear(2, 2)
        linear._test_parameter = Parameter(torch.Tensor(2, 2))
        linear.register_buffer('_test_buffer', torch.Tensor(2, 2))
        keys = linear.__dir__()
        self.assertIn('_test_submodule', keys)
        self.assertIn('_test_parameter', keys)
        self.assertIn('_test_buffer', keys)

        for key in keys:
            self.assertTrue(hasattr(linear, key))

    def test_named_children(self):
        l1 = nn.Linear(2, 2)
        l2 = nn.Linear(2, 2)
        l3 = nn.Linear(2, 2)
        l4 = nn.Linear(2, 2)
        subnet = nn.Sequential(l3, l4)
        s = nn.Sequential()
        s.add_module('layer1', l1)
        s.add_module('layer2', l2)
        s.add_module('layer3', l1)
        s.add_module('layer4', l2)
        s.add_module('subnet', subnet)
        self.assertEqual(list(s.named_children()), [('layer1', l1), ('layer2', l2), ('subnet', subnet)])

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

    def test_named_modules(self):
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.l1 = l
                self.l2 = l
                self.param = Variable(torch.Tensor(3, 5))
                self.block = block
        l = nn.Linear(10, 20)
        l1 = nn.Linear(10, 20)
        l2 = nn.Linear(10, 20)
        block = nn.Sequential()
        block.add_module('linear1', l1)
        block.add_module('linear2', l2)
        n = Net()
        s = nn.Sequential(n, n, n, n)
        self.assertEqual(list(s.named_modules()), [('', s), ('0', n), ('0.l1', l),
                                                   ('0.block', block), ('0.block.linear1', l1),
                                                   ('0.block.linear2', l2)])

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
        def make_param():
            return Parameter(torch.randn(10, 10))
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

    def test_clip_grad_norm(self):
        l = nn.Linear(10, 10)
        max_norm = 2

        def compute_norm(norm_type):
            norm_type = float(norm_type)
            if norm_type != float('inf'):
                total_norm = 0
                for p in l.parameters():
                    total_norm += p.grad.data.abs().pow(norm_type).sum()
                return pow(total_norm, 1. / norm_type)
            else:
                return max(p.grad.data.abs().max() for p in l.parameters())

        def compare_scaling(grads):
            p_scale = [p.grad.data.div(g).view(-1) for p, g in zip(l.parameters(), grads)]
            scale = torch.cat(p_scale)
            self.assertEqual(scale.std(), 0)
            return scale[0]

        grads = torch.arange(1, 101).view(10, 10), torch.ones(10).div(1000)
        for norm_type in [0.5, 1.5, 2, 4, 'inf']:
            for p, g in zip(l.parameters(), grads):
                p._grad = Variable(g.clone().view_as(p.data))
            norm_before = compute_norm(norm_type)
            norm = clip_grad_norm(l.parameters(), max_norm, norm_type=norm_type)
            norm_after = compute_norm(norm_type)
            self.assertEqual(norm, norm_before)
            self.assertEqual(norm_after, max_norm)
            self.assertLessEqual(norm_after, norm_before)
            compare_scaling(grads)

        # Small gradients should be left unchanged
        grads = torch.rand(10, 10).div(10000), torch.ones(10).div(500)
        for norm_type in [0.5, 1.5, 2, 4, 'inf']:
            for p, g in zip(l.parameters(), grads):
                p.grad.data.copy_(g)
            norm_before = compute_norm(norm_type)
            norm = clip_grad_norm(l.parameters(), max_norm, norm_type=norm_type)
            norm_after = compute_norm(norm_type)
            self.assertEqual(norm, norm_before)
            self.assertEqual(norm_before, norm_after)
            self.assertLessEqual(norm_after, max_norm)
            scale = compare_scaling(grads)
            self.assertEqual(scale, 1)

    def test_embedding_padding_idx(self):
        embedding = nn.Embedding(10, 20, padding_idx=0)
        input = Variable(torch.LongTensor([[0, 2, 4, 5], [4, 3, 0, 9]]))
        output = embedding(input)
        self.assertEqual(output[0][0].sum().data[0], 0)
        self.assertEqual(output[1][2].sum().data[0], 0)

    def _test_EmbeddingBag(self, cuda):
        ## check a known test example

        es = nn.EmbeddingBag(5, 2)
        # es.weight.data.zero_()
        es.weight.data.copy_(torch.range(1, 10))
        input = Variable(torch.LongTensor([3, 1, 1, 1, 4]))
        offsets = Variable(torch.LongTensor([0, 2]))
        grad_output = torch.range(1, 4).view(2, 2).type(torch.Tensor)

        expected_output = torch.Tensor(
            [[10, 12],
             [15, 18]])
        expected_grad_weight = torch.Tensor(
            [[0, 0],
             [7, 10],
             [0, 0],
             [1, 2],
             [3, 4]])

        if cuda:
            es = es.cuda()
            input = input.cuda()
            offsets = offsets.cuda()
            grad_output = grad_output.cuda()
            expected_output = expected_output.cuda()
            expected_grad_weight = expected_grad_weight.cuda()

        output = es(input, offsets)

        output.backward(grad_output)

        self.assertEqual(output.data, expected_output)
        self.assertEqual(es.weight.grad.data, expected_grad_weight)

        ## now compare EmbeddingBag vs Embedding + Sum, for constant bag length

        N = random.randint(1, 100)
        D = random.randint(1, 100)

        es = nn.EmbeddingBag(N, D)
        e = nn.Embedding(N, D)
        e.weight.data.copy_(es.weight.data)

        B = random.randint(1, 50)
        L = random.randint(1, 50)

        input = Variable(torch.rand(B, L).mul(N).long())
        offsets = Variable(torch.range(0, B - 1).mul(L).long())
        grad_output = torch.rand(B, D).type(torch.Tensor)

        if cuda:
            es = es.cuda()
            e = e.cuda()
            input = input.cuda()
            offsets = offsets.cuda()
            grad_output = grad_output.cuda()

        output = es(input.view(-1), offsets)
        ref_output = e(input).sum(1).squeeze(1)

        self.assertEqual(output, ref_output)

        output.backward(grad_output)
        ref_output.backward(grad_output)
        self.assertEqual(es.weight.grad, e.weight.grad)

    def test_EmbeddingBag(self):
        self._test_EmbeddingBag(False)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_EmbeddingBag_cuda(self):
        self._test_EmbeddingBag(True)

    # FIXME: I don't know how to add this to the gradcheck NewModuleTest
    # framework since this module has 2 inputs. But maybe not necessary
    # since I'm comparing directly with LookupTable + Sum, which are checked


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

    def test_AlphaDropout(self):
        # generate random tensor with zero mean and unit std
        input = torch.randn(5000)

        mean = input.mean()
        std = input.std()

        for p in [0.2, 0.5, 0.8]:
            module = nn.AlphaDropout(p)
            input_var = Variable(input, requires_grad=True)
            output = module(input_var)
            # output mean should be close to input mean
            self.assertLess(abs(output.data.mean() - mean), 0.1)
            # output std should be close to input std
            self.assertLess(abs(output.data.std() - std), 0.1)
            output.backward(input)

    def _test_InstanceNorm(self, cls, input):
        b, c = input.size(0), input.size(1)
        input_var = Variable(input)

        IN = cls(c, eps=0)

        output = IN(input_var)
        out_reshaped = output.transpose(1, 0).contiguous().view(c, -1)

        mean = out_reshaped.mean(1)
        var = out_reshaped.var(1, unbiased=False)

        self.assertAlmostEqual(torch.abs(mean.data).mean(), 0, delta=1e-5)
        self.assertAlmostEqual(torch.abs(var.data).mean(), 1, delta=1e-5)

        # If momentum==1 running_mean/var should be
        # equal to mean/var of the input
        IN = cls(c, momentum=1, eps=0)

        output = IN(input_var)

        input_reshaped = input_var.transpose(1, 0).contiguous().view(c, -1)
        mean = input_reshaped.mean(1)

        input_reshaped = input_var.transpose(1, 0).contiguous().view(c, b, -1)
        var = input_reshaped.var(2, unbiased=True)[:, :]

        self.assertAlmostEqual(torch.abs(mean.data - IN.running_mean).mean(), 0, delta=1e-5)
        self.assertAlmostEqual(torch.abs(var.data.mean(1) - IN.running_var).mean(), 0, delta=1e-5)

    def test_InstanceNorm2d(self):
        b = random.randint(3, 5)
        c = random.randint(1, 5)
        w = random.randint(2, 5)
        h = random.randint(2, 5)

        input = torch.Tensor(b, c, h, w).uniform_()
        self._test_InstanceNorm(nn.InstanceNorm2d, input)

    def test_InstanceNorm1d(self):
        b = random.randint(3, 5)
        c = random.randint(1, 5)
        d = random.randint(2, 5)

        input = torch.Tensor(b, c, d).uniform_()
        self._test_InstanceNorm(nn.InstanceNorm1d, input)

    def test_InstanceNorm3d(self):
        b = random.randint(3, 5)
        c = random.randint(1, 5)
        w = random.randint(2, 5)
        h = random.randint(2, 5)
        d = random.randint(2, 5)

        input = torch.Tensor(b, c, h, w, d).uniform_()
        self._test_InstanceNorm(nn.InstanceNorm3d, input)

    def test_pad(self):
        inputs = Variable(torch.randn(1, 3, 4, 4), requires_grad=True)
        self.assertTrue(gradcheck(lambda x: F.pad(x, (1, 1, 1, 1)), (inputs,)))
        self.assertTrue(gradcheck(lambda x: F.pad(x, (-1, 1, -2, 1)), (inputs,)))
        self.assertTrue(gradcheck(lambda x: F.pad(x, (-1, 1, -2, 1), value=2), (inputs,)))
        self.assertTrue(gradcheck(lambda x: F.pad(x, (-1, 1, -2, 1), mode='replicate'), (inputs,)))
        self.assertTrue(gradcheck(lambda x: F.pad(x, (-1, 1, -2, 1), mode='reflect'), (inputs,)))

        inputs = Variable(torch.randn(1, 2, 3, 4, 4), requires_grad=True)
        self.assertTrue(gradcheck(lambda x: F.pad(x, (1, 1, 1, 1, 1, 1), mode='replicate'), (inputs,)))

    def test_normalize(self):
        inputs = Variable(torch.randn(1, 3, 4, 4), requires_grad=True)
        self.assertTrue(gradcheck(lambda x: F.normalize(x, p=1, dim=-1), (inputs,)))
        self.assertTrue(gradcheck(lambda x: F.normalize(x, p=2, dim=-2), (inputs,)))

    def _test_maxpool_indices(self, num_dim, type=torch.FloatTensor):
        def expected_indices(dim):
            if dim == 1:
                return torch.DoubleTensor([1, 3]).repeat(2, 2, 1)
            if dim == 2:
                return torch.DoubleTensor([[5, 7], [13, 15]]).repeat(2, 2, 1, 1)

        def expected_grad(dim):
            if dim == 1:
                return torch.DoubleTensor([0, 1, 0, 1]).repeat(2, 2, 1)
            grad = expected_grad(dim - 1)
            zero = torch.zeros(grad.size())
            return torch.stack((zero, grad, zero, grad), 2)

        def expected_output(dim):
            if dim == 1:
                return torch.arange(2, 17, 2).view(2, 2, 2)
            if dim == 2:
                col = torch.arange(6, 63, 8)
                return torch.stack([col, col + 2], 1).view(2, 2, 2, 2)

        module_cls = getattr(nn, 'MaxPool{}d'.format(num_dim))
        module = module_cls(2, return_indices=True).type(type)
        numel = 4 ** (num_dim + 1)
        input = torch.arange(1, numel + 1).view(2, 2, *repeat(4, num_dim)).type(type)
        input_var = Variable(input, requires_grad=True)

        # Check forward
        output, indices = module(input_var)
        if num_dim != 3:
            expected_indices = expected_indices(num_dim)
            expected_output = expected_output(num_dim)
            self.assertEqual(indices.dim(), input.dim())
            self.assertEqual(indices.data.squeeze(), expected_indices)
            self.assertEqual(output.data.squeeze(), expected_output)
        self.assertTrue(output.requires_grad)
        self.assertFalse(indices.requires_grad)

        # Make sure backward works
        grad_output = torch.ones(output.size()).type(type)
        output.backward(grad_output, retain_graph=True)
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

    def test_AdaptiveMaxPool1d_indices(self):
        self._test_maxpool_indices(1)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_AdaptiveMaxPool1d_indices_cuda(self):
        self._test_maxpool_indices(1, torch.cuda.FloatTensor)

    def test_AdaptiveMaxPool2d_indices(self):
        self._test_maxpool_indices(2)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_AdaptiveMaxPool2d_indices_cuda(self):
        self._test_maxpool_indices(2, torch.cuda.FloatTensor)

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
        inputs = ((i1,), (i2,))
        modules = (l1, l2)
        expected_outputs = (expected1, expected2)

        outputs = dp.parallel_apply(modules, inputs, None)
        for out, expected in zip(outputs, expected_outputs):
            self.assertEqual(out.data, expected)

        inputs = (i1, Variable(i2.data.new()))
        expected_outputs = (expected1, expected2.new())

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_data_parallel_multiple_input(self):
        class TestModule(nn.Module):

            def forward(self, var1, var2, float1, var3=None):
                if var3 is None:
                    return float1 * (var1 * var2)
                else:
                    return float1 * (var1 * var2 + var3)

        m = TestModule()
        var1 = Variable(torch.randn(5, 5).float(), requires_grad=True)
        var2 = Variable(torch.randn(5, 5).float(), requires_grad=True)
        var3 = Variable(torch.randn(5, 5).float(), requires_grad=False)

        float1 = torch.randn(1)[0]

        expected = m(var1, var2, float1)
        loss = expected.sum()
        loss.backward()
        gvar1_exp = var1.grad.clone()
        gvar2_exp = var2.grad.clone()

        def local_test(out):
            var1.grad.data.fill_(0.0)
            var2.grad.data.fill_(0.0)
            loss = out.sum()
            loss.backward()
            self.assertEqual(out, expected)
            self.assertEqual(gvar1_exp, var1.grad)
            self.assertEqual(gvar2_exp, var2.grad)

        out = dp.data_parallel(m, (var1, var2, float1), (0, 1))
        local_test(out)

        out = dp.data_parallel(m, (var1, var2, float1), (1, 0))
        local_test(out)

        out = dp.data_parallel(m, (var1, var2, float1), (0,))
        local_test(out)

        var1.grad.data.fill_(0.0)
        var2.grad.data.fill_(0.0)
        expected = m(var1, var2, float1, var3=var3)
        loss = expected.sum()
        loss.backward()
        gvar1_exp = var1.grad.clone()
        gvar2_exp = var2.grad.clone()

        dpm = nn.DataParallel(TestModule())
        out = dpm(var1, var2, float1, var3=var3)
        local_test(out)

        dpm = nn.DataParallel(TestModule(), device_ids=[0])
        out = dpm(var1, var2, float1, var3=var3)
        local_test(out)

        kwarg_wrap = {'var3': var3}
        out = dp.data_parallel(
            m, (var1, var2, float1), (0, 1), module_kwargs=kwarg_wrap)
        local_test(out)

        out = dp.data_parallel(
            m, (var1, var2, float1), (0,), module_kwargs=kwarg_wrap)
        local_test(out)

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

        # Check for None device_ids
        out = dp.data_parallel(l, i)

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
            def forward(self, *input):
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
            'block.conv1.bias': torch.arange(1, 4),
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

    def test_assignment(self):
        l = nn.Module()
        a = nn.Parameter(torch.randn(2))
        b = nn.Parameter(torch.randn(3))
        c = nn.Parameter(torch.randn(4))
        q = nn.Linear(4, 4)
        r = nn.Linear(5, 5)
        w = nn.Linear(6, 6)

        def test_assignments(get_list, a, b, c):
            # Check that None can be shadowed
            l.a = None
            self.assertIsNone(l.a)
            self.assertIn('a', l.__dict__)
            l.a = a
            self.assertIs(l.a, a)
            self.assertEqual(get_list(), [a])
            self.assertNotIn('a', l.__dict__)

            # Assign second object
            l.b = None
            self.assertIsNone(l.b)
            self.assertIn('b', l.__dict__)
            l.b = b
            self.assertIs(l.b, b)
            self.assertEqual(get_list(), [a, b])
            self.assertNotIn('b', l.__dict__)

            # Remove and add the object back. Order should be unchanged.
            l.a = None
            self.assertIsNone(l.a)
            self.assertEqual(get_list(), [b])
            l.a = a
            self.assertIs(l.a, a)
            self.assertEqual(get_list(), [a, b])

            # Replace object with another one. Order should be unchanged.
            l.a = c
            self.assertIs(l.a, c)
            self.assertEqual(get_list(), [c, b])

            # Remove and reassign an attribute. It should appear at the end of the list now.
            del l.a
            self.assertFalse(hasattr(l, 'a'))
            l.a = a
            self.assertIs(l.a, a)
            self.assertEqual(get_list(), [b, a])

        test_assignments(lambda: list(l.parameters()), a, b, c)
        del l.a, l.b
        self.assertEqual(list(l.parameters()), [])

        test_assignments(lambda: list(l.children()), q, r, w)
        del l.a, l.b
        self.assertEqual(list(l.children()), [])

        buf = torch.randn(10)
        l.register_buffer('buf', buf)
        self.assertIs(l.buf, buf)
        l.buf = None
        self.assertIs(l.buf, None)
        self.assertNotIn('buf', l.__dict__)  # should be stored in l._buffers
        l.buf = buf
        self.assertIn('buf', l.state_dict())
        self.assertIs(l.state_dict()['buf'], buf)

    def test_Conv2d_inconsistent_types(self):
        inputs = Variable(torch.randn(4, 1, 7, 7).float())
        weights = Variable(torch.randn(1, 1, 3, 3).double())
        # inconsistent types should raise an exception
        self.assertRaises(RuntimeError, lambda: nn.functional.conv2d(inputs, weights))
        # but it should work with the same type
        nn.functional.conv2d(inputs.float(), weights.float())

    @unittest.skipIf(not TEST_CUDA, 'CUDA not available')
    def test_Conv2d_inconsistent_types_on_GPU_without_cudnn(self):
        inputs = Variable(torch.randn(4, 1, 7, 7).float().cuda())
        weights = Variable(torch.randn(1, 1, 3, 3).double().cuda())
        bias = Variable(torch.randn(1).double().cuda())

        torch.backends.cudnn.enabled = False
        # inconsistent types should raise an exception
        self.assertRaises(RuntimeError, lambda: nn.functional.conv2d(inputs, weights))
        self.assertRaises(RuntimeError, lambda: nn.functional.conv2d(inputs, weights.float(), bias))

        # but it should work with the same type
        nn.functional.conv2d(inputs.float(), weights.float(), bias.float())

    @unittest.skipIf(not TEST_CUDA, 'CUDA not available')
    @unittest.skipIf(not TEST_CUDNN, 'CUDNN not available')
    def test_Conv2d_inconsistent_types_on_GPU_with_cudnn(self):
        inputs = Variable(torch.randn(4, 1, 7, 7).float().cuda())
        weights = Variable(torch.randn(1, 1, 3, 3).double().cuda())
        bias = Variable(torch.randn(1).double().cuda())

        torch.backends.cudnn.enabled = True
        # inconsistent types should raise an exception
        self.assertRaises(RuntimeError, lambda: nn.functional.conv2d(inputs, weights))
        self.assertRaises(RuntimeError, lambda: nn.functional.conv2d(inputs, weights.float(), bias))

        # but it should work with the same type
        nn.functional.conv2d(inputs.float(), weights.float(), bias.float())

    def test_Conv2d_missing_argument(self):
        c = nn.Conv2d(3, 3, 3)
        self.assertRaises(RuntimeError, lambda: c(None))

    def test_Conv2d_backward_twice(self):
        input = Variable(torch.randn(2, 3, 5, 5))
        c = nn.Conv2d(3, 3, 3)
        o1 = c(input)
        o1.sum().backward()
        self.assertRaisesRegex(RuntimeError, 'Specify retain_variables=True',
                               lambda: o1.sum().backward())

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

    def test_conv_modules_raise_error_on_incorrect_input_size(self):
        modules = [nn.Conv1d(3, 8, 3), nn.ConvTranspose1d(3, 8, 3),
                   nn.Conv2d(3, 8, 3), nn.ConvTranspose2d(3, 8, 3),
                   nn.Conv3d(3, 8, 3), nn.ConvTranspose3d(3, 8, 3)]

        invalid_input_dims = [(2, 4), (2, 4),
                              (3, 5), (3, 5),
                              (4, 6), (4, 6)]

        for invalid_dims, module in zip(invalid_input_dims, modules):
            for dims in invalid_dims:
                input = Variable(torch.Tensor(torch.Size((3, ) * dims)))
                self.assertRaises(ValueError, lambda: module(input))

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

    # For https://github.com/pytorch/pytorch/pull/1273
    # Almost identical to the above `test_Conv2d_naive_groups`
    def test_Conv2d_groups_nobias(self):
        m = nn.Conv2d(4, 4, kernel_size=3, groups=2, bias=False)
        i = Variable(torch.randn(2, 4, 6, 6), requires_grad=True)
        output = m(i)
        grad_output = torch.randn(2, 4, 4, 4)
        output.backward(grad_output)

        m1 = nn.Conv2d(2, 2, kernel_size=3, bias=False)
        m1.weight.data.copy_(m.weight.data[:2])
        i1 = Variable(i.data[:, :2].contiguous(), requires_grad=True)
        output1 = m1(i1)
        output1.backward(grad_output[:, :2].contiguous())

        m2 = nn.Conv2d(2, 2, kernel_size=3, bias=False)
        m2.weight.data.copy_(m.weight.data[2:])
        i2 = Variable(i.data[:, 2:].contiguous(), requires_grad=True)
        output2 = m2(i2)
        output2.backward(grad_output[:, 2:].contiguous())

        self.assertEqual(output, torch.cat([output1, output2], 1))
        self.assertEqual(i.grad.data,
                         torch.cat([i1.grad.data, i2.grad.data], 1))
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
                    self.assertRaises(ValueError, lambda: mu(output_small, indices_small, (h, w)))

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

        model_cp.linear.weight.data[:] = 2
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

    def test_pack_padded_sequence(self):
        def pad(tensor, length):
            return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])
        lengths = [10, 8, 4, 2, 2, 2, 1]
        max_length = lengths[0]
        batch_sizes = [sum(map(bool, filter(lambda x: x >= i, lengths))) for i in range(1, max_length + 1)]
        offset = 0
        padded = torch.cat([pad(i * 100 + torch.arange(1, 5 * l + 1).view(l, 1, 5), max_length)
                            for i, l in enumerate(lengths, 1)], 1)
        padded = Variable(padded, requires_grad=True)
        expected_data = [[torch.arange(1, 6) + (i + 1) * 100 + 5 * n for i in range(batch_size)]
                         for n, batch_size in enumerate(batch_sizes)]
        expected_data = list(itertools.chain.from_iterable(expected_data))
        expected_data = torch.stack(expected_data, dim=0)

        for batch_first in (True, False):
            src = padded
            if batch_first:
                src = src.transpose(0, 1)

            # check output
            packed = rnn_utils.pack_padded_sequence(src, lengths, batch_first=batch_first)
            self.assertEqual(packed.data.data, expected_data)
            self.assertEqual(packed.batch_sizes, batch_sizes)

            # test inverse
            unpacked, unpacked_len = rnn_utils.pad_packed_sequence(packed, batch_first=batch_first)
            self.assertEqual(unpacked, src)
            self.assertEqual(unpacked_len, lengths)

            # check grad
            if padded.grad is not None:
                padded.grad.data.zero_()
            grad_output = unpacked.data.clone().normal_()
            unpacked.backward(grad_output)
            if batch_first:
                grad_output.transpose_(0, 1)
            for i, l in enumerate(lengths):
                self.assertEqual(padded.grad.data[:l, i], grad_output[:l, i])
                if l < 10:
                    self.assertEqual(padded.grad.data[l:, i].abs().sum(), 0)

    def _test_variable_sequence(self, cuda):
        def pad(var, length):
            if var.size(0) == length:
                return var
            return torch.cat([var, Variable(var.data.new(length - var.size(0), *var.size()[1:]).zero_())])

        lengths = [10, 10, 6, 2, 2, 1, 1]
        max_length = lengths[0]
        x_leaf = Variable(torch.randn(max_length, len(lengths), 3), requires_grad=True)
        lstm = nn.LSTM(3, 4, bidirectional=True, num_layers=2)
        lstm2 = deepcopy(lstm)
        if cuda:
            x = x_leaf.cuda()
            lstm.cuda()
            lstm2.cuda()
        else:
            x = x_leaf

        # Compute sequences separately
        seq_outs = []
        seq_hiddens = []
        for i, l in enumerate(lengths):
            out, hid = lstm2(x[:l, i:i + 1])
            out_pad = pad(out, max_length)
            seq_outs.append(out_pad)
            seq_hiddens.append(hid)
        seq_out = torch.cat(seq_outs, 1)
        seq_hidden = tuple(torch.cat(hids, 1) for hids in zip(*seq_hiddens))

        # Use packed format
        packed = rnn_utils.pack_padded_sequence(x, lengths)
        packed_out, packed_hidden = lstm(packed)
        unpacked, unpacked_len = rnn_utils.pad_packed_sequence(packed_out)

        # Check forward
        self.assertEqual(packed_hidden, seq_hidden)
        self.assertEqual(unpacked, seq_out)
        self.assertEqual(unpacked_len, lengths)

        # Check backward
        seq_out.sum().backward()
        grad_x = x_leaf.grad.data.clone()
        x_leaf.grad.data.zero_()
        unpacked.sum().backward()

        self.assertEqual(x_leaf.grad.data, grad_x)
        for p1, p2 in zip(lstm.parameters(), lstm2.parameters()):
            self.assertEqual(p1.grad, p2.grad)

    def test_variable_sequence(self):
        self._test_variable_sequence(False)

    @unittest.skipIf(not TEST_CUDA, 'CUDA not available')
    def test_variable_sequence_cuda(self):
        self._test_variable_sequence(True)

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

    @unittest.skipIf(not TEST_CUDA, 'CUDA not available')
    def test_cuda_rnn_fused(self):
        def copy_rnn(rnn1, rnn2):
            for x_layer, y_layer in zip(rnn1.all_weights, rnn2.all_weights):
                for x, y in zip(x_layer, y_layer):
                    x.data.copy_(y.data)

        def check_rnn_grads(rnn1, rnn2):
            for x_layer, y_layer in zip(rnn1.all_weights, rnn2.all_weights):
                for x, y in zip(x_layer, y_layer):
                    self.assertEqual(x.grad, y.grad, prec=5e-5)

        input_size = 10
        hidden_size = 6
        num_layers = 2
        seq_length = 7
        batch = 6
        input_val = torch.randn(seq_length, batch, input_size)
        grad_output = torch.randn(seq_length, batch, hidden_size)
        hx_val = torch.randn(num_layers, batch, hidden_size)
        grad_hy = torch.randn(num_layers, batch, hidden_size)
        prev = torch.backends.cudnn.enabled
        try:
            torch.backends.cudnn.enabled = False
            for module in (nn.GRU, nn.LSTM):
                for bias in (True, False):
                    rnn = module(input_size, hidden_size, num_layers, bias=bias)
                    rnn_cuda = module(input_size, hidden_size, num_layers, bias=bias).cuda()
                    copy_rnn(rnn, rnn_cuda)

                    is_lstm = isinstance(rnn, nn.LSTM)
                    if is_lstm:
                        hx = (Variable(hx_val.clone(), requires_grad=True),
                              Variable(hx_val.clone().add(1), requires_grad=True))
                        hx_cuda = (Variable(hx_val.clone().cuda(), requires_grad=True),
                                   Variable(hx_val.clone().cuda().add(1), requires_grad=True))
                    else:
                        hx = Variable(hx_val.clone(), requires_grad=True)
                        hx_cuda = Variable(hx_val.clone().cuda(), requires_grad=True)

                    inp = Variable(input_val.clone(), requires_grad=True)
                    inp_cu = Variable(input_val.clone().cuda(), requires_grad=True)
                    output1, hy1 = rnn(inp, hx)
                    output2, hy2 = rnn_cuda(inp_cu, hx_cuda)
                    if is_lstm:
                        torch.autograd.backward(
                            [output1, hy1[0], hy1[1]], [grad_output, grad_hy, grad_hy + 1]
                        )
                        torch.autograd.backward(
                            [output2, hy2[0], hy2[1]],
                            [grad_output.cuda(), grad_hy.cuda(), (grad_hy + 1).cuda()]
                        )
                    else:
                        torch.autograd.backward([output1, hy1], [grad_output, grad_hy])
                        torch.autograd.backward([output2, hy2], [grad_output.cuda(), grad_hy.cuda()])

                    self.assertEqual(output1, output2)
                    self.assertEqual(hy1, hy2)

                    check_rnn_grads(rnn, rnn_cuda)
                    self.assertEqual(inp.grad.data, inp_cu.grad.data)
                    if is_lstm:
                        self.assertEqual(hx[0].grad.data, hx_cuda[0].grad.data)
                        self.assertEqual(hx[1].grad.data, hx_cuda[1].grad.data)
                    else:
                        self.assertEqual(hx.grad.data, hx_cuda.grad.data)
        finally:
            torch.backends.cudnn.enabled = prev

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
        rnns = [nn.LSTM(10, 20, num_layers=2).type(dtype),
                nn.GRU(10, 20, num_layers=2).type(dtype),
                nn.RNN(10, 20, num_layers=2).type(dtype)]
        for rnn in rnns:
            input = Variable(torch.randn(5, 6, 10).type(dtype), requires_grad=True)
            output = rnn(input)
            output[0].sum().backward(retain_graph=True)
            grads = [input.grad.data.clone()] + [p.grad.data.clone() for p in rnn.parameters()]
            for i in range(4):
                rnn.zero_grad()
                input.grad.data.zero_()
                output[0].sum().backward(retain_graph=True)
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
            is_lstm = isinstance(rnn, nn.LSTM)

            for x_layer, y_layer in zip(rnn.all_weights, weights_val):
                for x, y in zip(x_layer, y_layer):
                    x.data.copy_(y.data)

            if isinstance(input_val, rnn_utils.PackedSequence):
                input = rnn_utils.PackedSequence(
                    Variable(input_val.data, requires_grad=True), input_val.batch_sizes)
                input_var = input.data
            else:
                input = Variable(input_val.clone(), requires_grad=True)
                input_var = input
            if is_lstm:
                hx = (Variable(hx_val.clone(), requires_grad=True),
                      Variable(hx_val.add(1), requires_grad=True))
            else:
                hx = Variable(hx_val.clone(), requires_grad=True)

            if cuda:
                rnn.cuda()
                input_var.data = input_var.data.cuda()
                if is_lstm:
                    hx[0].data = hx[0].data.cuda()
                    hx[1].data = hx[1].data.cuda()
                else:
                    hx.data = hx.data.cuda()
                grad_hy = grad_hy.cuda()
                grad_output = grad_output.cuda()

            output, hy = rnn(input, hx)

            if isinstance(output, rnn_utils.PackedSequence):
                output = output.data

            if is_lstm:
                torch.autograd.backward([output, hy[0], hy[1]], [grad_output, grad_hy, grad_hy + 1])
            else:
                torch.autograd.backward([output, hy], [grad_output, grad_hy])

            return {'output': output.data,
                    'hy': hy[0].data if is_lstm else hy.data,
                    'weights': rnn.all_weights,
                    'grad_input': input_var.grad.data,
                    'grad_hx': hx[0].grad.data if is_lstm else hx.grad.data,
                    'cy': hy[1].data if is_lstm else None,
                    'grad_cx': hx[1].grad.data if is_lstm else None}

        input_size = 10
        hidden_size = 6
        num_layers = 2
        seq_length = 7
        batch = 6

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
            for bias, bidirectional, batch_first, contig, variable_len in product((True, False), repeat=5):
                num_directions = 2 if bidirectional else 1
                if batch_first:
                    input_val = torch.randn(batch, seq_length, input_size)
                    grad_output = torch.randn(batch, seq_length, hidden_size * num_directions)
                else:
                    input_val = torch.randn(seq_length, batch, input_size)
                    grad_output = torch.randn(seq_length, batch, hidden_size * num_directions)

                if not contig:
                    grad_output = make_noncontig(grad_output)
                    grad_hy = make_noncontig(grad_hy)
                    input_var = make_noncontig(input_val)
                    hx_val = make_noncontig(hx_val)

                hx_val = torch.randn(num_layers * num_directions, batch, hidden_size)
                grad_hy = torch.randn(num_layers * num_directions, batch, hidden_size)

                if variable_len:
                    batch_sizes = [7, 5, 5, 2, 1, 1]
                    input_val = rnn_utils.pack_padded_sequence(input_val, batch_sizes, batch_first=batch_first)
                    grad_output = rnn_utils.pack_padded_sequence(grad_output, batch_sizes, batch_first=batch_first).data

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
            grad_output = torch.randn(
                seq_length, batch, hidden_size * num_directions)
            grad_hy = torch.randn(
                num_layers * num_directions, batch, hidden_size)

            rnn = nn.RNN(input_size, hidden_size, num_layers, bias=bias, nonlinearity=nonlinearity)
            outputs_cpu = forward_backward(False, rnn, input_val, hx_val, grad_output, grad_hy, rnn.all_weights)

            rnn_gpu = nn.RNN(input_size, hidden_size, num_layers, bias=bias, nonlinearity=nonlinearity)
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

    @unittest.skipIf(not (TEST_CUDNN and TEST_CUDNN_VERSION >= 5103), "needs cudnn >= 5.1")
    def test_RNN_change_dropout(self):
        for train, cuda in product((True, False), repeat=2):
            rnn = nn.RNN(100, 100, 2, dropout=0, nonlinearity='relu')
            input = Variable(torch.Tensor(3, 2, 100).uniform_())
            if cuda:
                input.data = input.data.cuda()
                rnn.cuda()

            if train:
                rnn.train()
            else:
                rnn.eval()

            prev_output = None
            for p in (0, 0.5, 0, 0.7, 0.2, 1, 0.2, 0):
                rnn.dropout = p
                output1, hy1 = rnn(input)
                output2, hy2 = rnn(input)

                if p == 0 or p == 1 or not train:
                    self.assertEqual(output1, output2)
                    self.assertEqual(hy1, hy2)
                else:
                    self.assertNotEqual(output1, output2)
                    self.assertNotEqual(hy1, hy2)

                if prev_output is not None:
                    if not train:
                        self.assertEqual(output1.data, prev_output)
                        self.assertEqual(output2.data, prev_output)
                    else:
                        self.assertNotEqual(output1.data, prev_output)
                        self.assertNotEqual(output2.data, prev_output)
                prev_output = output1.data

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
        modules = [nn.ReLU, nn.ELU, nn.SELU, nn.RReLU]
        for mod in modules:
            r = mod(inplace=True)
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
        output.backward(grad, retain_graph=True)
        self.assertIsNotNone(input.grad)
        result = input.grad.data.clone()
        input.grad.data.zero_()

        output.backward(grad.contiguous())
        self.assertEqual(result, input.grad.data)

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
            if data.grad is not None:
                data.grad.data.zero_()

            res2 = module(data)
            res2.backward(grad)
            grad2 = data.grad.data.clone()
            self.assertEqual(res1, res2)
            self.assertEqual(grad1, grad2)

    def test_pairwise_distance(self):
        input1 = Variable(torch.randn(4, 4), requires_grad=True)
        input2 = Variable(torch.randn(4, 4), requires_grad=True)
        self.assertTrue(gradcheck(lambda x, y: F.pairwise_distance(x, y), (input1, input2)))

    def test_triplet_margin_loss(self):
        input1 = Variable(torch.randn(4, 4), requires_grad=True)
        input2 = Variable(torch.randn(4, 4), requires_grad=True)
        input3 = Variable(torch.randn(4, 4), requires_grad=True)
        self.assertTrue(gradcheck(lambda x1, x2, x3: F.triplet_margin_loss(
            x1, x2, x3), (input1, input2, input3)))

    def test_triplet_margin_swap_loss(self):
        input1 = Variable(torch.randn(4, 4), requires_grad=True)
        input2 = Variable(torch.randn(4, 4), requires_grad=True)
        input3 = Variable(torch.randn(4, 4), requires_grad=True)
        self.assertTrue(gradcheck(lambda x1, x2, x3: F.triplet_margin_loss(
            x1, x2, x3, swap=True), (input1, input2, input3)))

    def test_cosine_similarity(self):
        input1 = Variable(torch.randn(4, 4), requires_grad=True)
        input2 = Variable(torch.randn(4, 4), requires_grad=True)
        self.assertTrue(gradcheck(lambda x, y: F.cosine_similarity(x, y), (input1, input2)))

        input1 = Variable(torch.randn(4, 5, 6), requires_grad=True)
        input2 = Variable(torch.randn(4, 5, 6), requires_grad=True)
        self.assertTrue(gradcheck(lambda x, y: F.cosine_similarity(x, y, dim=0), (input1, input2)))
        self.assertTrue(gradcheck(lambda x, y: F.cosine_similarity(x, y, dim=-1), (input1, input2)))

    def test_upsamplingNearest2d(self):
        m = nn.Upsample(size=4, mode='nearest')
        in_t = torch.ones(1, 1, 2, 2)
        out_t = m(Variable(in_t))
        self.assertEqual(torch.ones(1, 1, 4, 4), out_t.data)

        input = Variable(torch.randn(1, 1, 2, 2), requires_grad=True)
        self.assertTrue(gradcheck(lambda x: F.upsample(x, 4, mode='nearest'), (input,)))

    def test_upsamplingBilinear2d(self):
        m = nn.Upsample(size=4, mode='bilinear')
        in_t = torch.ones(1, 1, 2, 2)
        out_t = m(Variable(in_t))
        self.assertEqual(torch.ones(1, 1, 4, 4), out_t.data)

        input = Variable(torch.randn(1, 1, 2, 2), requires_grad=True)
        self.assertTrue(gradcheck(lambda x: F.upsample(x, 4, mode='bilinear'), (input,)))

    def test_upsamplingNearest3d(self):
        m = nn.Upsample(size=4, mode='nearest')
        in_t = torch.ones(1, 1, 2, 2, 2)
        out_t = m(Variable(in_t))
        self.assertEqual(torch.ones(1, 1, 4, 4, 4), out_t.data)

        input = Variable(torch.randn(1, 1, 2, 2, 2), requires_grad=True)
        self.assertTrue(gradcheck(lambda x: F.upsample(x, 4, mode='nearest'), (input,)))

    def test_upsamplingTrilinear3d(self):
        m = nn.Upsample(size=4, mode='trilinear')
        in_t = torch.ones(1, 1, 2, 2, 2)
        out_t = m(Variable(in_t))
        self.assertEqual(torch.ones(1, 1, 4, 4, 4), out_t.data)

        input = Variable(torch.randn(1, 1, 2, 2, 2), requires_grad=True)
        self.assertTrue(gradcheck(lambda x: F.upsample(x, 4, mode='trilinear'), (input,)))

    def test_bilinear(self):
        module = nn.Bilinear(10, 10, 8)
        module_legacy = legacy.Bilinear(10, 10, 8)

        module_legacy.weight.copy_(module.weight.data)
        module_legacy.bias.copy_(module.bias.data)

        input1 = torch.randn(4, 10)
        input2 = torch.randn(4, 10)

        output = module(Variable(input1), Variable(input2))
        output_legacy = module_legacy.forward([input1, input2])

        self.assertEqual(output.data, output_legacy)

        input1_1 = Variable(input1, requires_grad=True)
        input2_1 = Variable(input2, requires_grad=True)

        module.zero_grad()
        module_legacy.zeroGradParameters()

        output = module(input1_1, input2_1)
        grad_output = torch.randn(*output.size())
        gi1_legacy, gi2_legacy = module_legacy.backward([input1, input2], grad_output)
        output.backward(grad_output)
        gi1 = input1_1.grad.data.clone()
        gi2 = input2_1.grad.data.clone()

        self.assertEqual(gi1, gi1_legacy)
        self.assertEqual(gi2, gi2_legacy)
        self.assertEqual(module.weight.grad.data, module_legacy.gradWeight)
        self.assertEqual(module.bias.grad.data, module_legacy.gradBias)

        self.assertTrue(gradcheck(lambda x1, x2: F.bilinear(x1, x2, module.weight, module.bias), (input1_1, input2_1)))


class TestNNInit(TestCase):
    def setUp(self):
        random.seed(123)
        torch.manual_seed(123)

    def _is_normal(self, tensor, mean, std):
        if isinstance(tensor, Variable):
            tensor = tensor.data
        samples = list(tensor.view(-1))
        p_value = stats.kstest(samples, 'norm', args=(mean, std)).pvalue
        return p_value > 0.0001

    def _is_uniform(self, tensor, a, b):
        if isinstance(tensor, Variable):
            tensor = tensor.data
        samples = list(tensor.view(-1))
        p_value = stats.kstest(samples, 'uniform', args=(a, (b - a))).pvalue
        return p_value > 0.0001

    def _create_random_nd_tensor(self, dims, size_min, size_max, as_variable):
        size = [random.randint(size_min, size_max) for _ in range(dims)]
        tensor = torch.zeros(size)
        if as_variable:
            tensor = Variable(tensor)
        return tensor

    def _random_float(self, a, b):
        return (b - a) * random.random() + a

    def test_calculate_gain_linear(self):
        for fn in ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose2d', 'conv_transpose2d', 'conv_transpose3d']:
            gain = init.calculate_gain(fn)
            self.assertEqual(gain, 1)

    def test_calculate_gain_nonlinear(self):
        for fn in ['sigmoid', 'tanh', 'relu', 'leaky_relu']:
            gain = init.calculate_gain(fn)
            if fn == 'sigmoid':
                self.assertEqual(gain, 1)
            elif fn == 'tanh':  # 5 / 3
                self.assertEqual(gain, 1.6666666666666667)
            elif fn == 'relu':  # sqrt(2)
                self.assertEqual(gain, 1.4142135623730951)
            elif fn == 'leaky_relu':  # sqrt(2 / 1 + slope^2))
                self.assertEqual(gain, 1.4141428569978354)

    def test_calculate_gain_leaky_relu(self):
        for param in [None, 0, 0.01, 10]:
            gain = init.calculate_gain('leaky_relu', param)
            if param is None:  # Default slope is 0.01
                self.assertEqual(gain, 1.4141428569978354)
            elif param == 0:  # No slope = same gain as normal ReLU
                self.assertEqual(gain, 1.4142135623730951)
            elif param == 0.01:
                self.assertEqual(gain, 1.4141428569978354)
            elif param == 10:
                self.assertEqual(gain, 0.14071950894605836)

    def test_calculate_gain_leaky_relu_only_accepts_numbers(self):
        for param in [True, [1], {'a': 'b'}]:
            with self.assertRaises(ValueError):
                init.calculate_gain('leaky_relu', param)

    def test_calculate_gain_only_accepts_valid_nonlinearities(self):
        for n in [2, 5, 25]:
            # Generate random strings of lengths that definitely aren't supported
            random_string = ''.join([random.choice(string.ascii_lowercase) for i in range(n)])
            with self.assertRaises(ValueError):
                init.calculate_gain(random_string)

    @unittest.skipIf(not TEST_SCIPY, "Scipy not found.")
    def test_uniform(self):
        for as_variable in [True, False]:
            for dims in [1, 2, 4]:
                input_tensor = self._create_random_nd_tensor(dims, size_min=30, size_max=50, as_variable=as_variable)
                a = self._random_float(-3, 3)
                b = a + self._random_float(1, 5)
                init.uniform(input_tensor, a=a, b=b)
                assert self._is_uniform(input_tensor, a, b)

    @unittest.skipIf(not TEST_SCIPY, "Scipy not found.")
    def test_normal(self):
        for as_variable in [True, False]:
            for dims in [1, 2, 4]:
                input_tensor = self._create_random_nd_tensor(dims, size_min=30, size_max=50, as_variable=as_variable)
                mean = self._random_float(-3, 3)
                std = self._random_float(1, 5)
                init.normal(input_tensor, mean=mean, std=std)

                assert self._is_normal(input_tensor, mean, std)

    def test_constant(self):
        for as_variable in [True, False]:
            for dims in [1, 2, 4]:
                input_tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=5, as_variable=as_variable)
                val = self._random_float(1, 10)
                init.constant(input_tensor, val)
                if as_variable:
                    input_tensor = input_tensor.data

                self.assertEqual(input_tensor, input_tensor.clone().fill_(val))

    def test_eye(self):
        for as_variable in [True, False]:
            input_tensor = self._create_random_nd_tensor(2, size_min=1, size_max=5, as_variable=as_variable)
            init.eye(input_tensor)
            if as_variable:
                input_tensor = input_tensor.data

            # Check every single element
            for i in range(input_tensor.size(0)):
                for j in range(input_tensor.size(1)):
                    if i == j:
                        assert input_tensor[i][j] == 1
                    else:
                        assert input_tensor[i][j] == 0

    def test_eye_only_works_on_2d_inputs(self):
        for as_variable in [True, False]:
            for dims in [1, 3]:
                with self.assertRaises(ValueError):
                    tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=3, as_variable=as_variable)
                    init.eye(tensor)

    def test_dirac_properties(self):
        for as_variable in [True, False]:
            for dims in [3, 4, 5]:
                input_tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=5, as_variable=as_variable)
                init.dirac(input_tensor)
                if as_variable:
                    input_tensor = input_tensor.data

                c_out, c_in = input_tensor.size(0), input_tensor.size(1)
                min_d = min(c_out, c_in)
                # Check number of nonzeros is equivalent to smallest dim
                assert torch.nonzero(input_tensor).size(0) == min_d
                # Check sum of values (can have precision issues, hence assertEqual) is also equivalent
                self.assertEqual(input_tensor.sum(), min_d)

    def test_dirac_identity(self):
        batch, in_c, out_c, size, kernel_size = 8, 3, 4, 5, 3
        # Test 1D
        input_var = Variable(torch.randn(batch, in_c, size))
        filter_var = Variable(torch.zeros(out_c, in_c, kernel_size))
        init.dirac(filter_var)
        output_var = F.conv1d(input_var, filter_var)
        input_tensor, output_tensor = input_var.data, output_var.data  # Variables do not support nonzero
        self.assertEqual(input_tensor[:, :, 1:-1], output_tensor[:, :in_c, :])  # Assert in_c outputs are preserved
        assert torch.nonzero(output_tensor[:, in_c:, :]).numel() == 0  # Assert extra outputs are 0

        # Test 2D
        input_var = Variable(torch.randn(batch, in_c, size, size))
        filter_var = Variable(torch.zeros(out_c, in_c, kernel_size, kernel_size))
        init.dirac(filter_var)
        output_var = F.conv2d(input_var, filter_var)
        input_tensor, output_tensor = input_var.data, output_var.data
        self.assertEqual(input_tensor[:, :, 1:-1, 1:-1], output_tensor[:, :in_c, :, :])
        assert torch.nonzero(output_tensor[:, in_c:, :, :]).numel() == 0

        # Test 3D
        input_var = Variable(torch.randn(batch, in_c, size, size, size))
        filter_var = Variable(torch.zeros(out_c, in_c, kernel_size, kernel_size, kernel_size))
        init.dirac(filter_var)
        output_var = F.conv3d(input_var, filter_var)
        input_tensor, output_tensor = input_var.data, output_var.data
        self.assertEqual(input_tensor[:, :, 1:-1, 1:-1, 1:-1], output_tensor[:, :in_c, :, :])
        assert torch.nonzero(output_tensor[:, in_c:, :, :, :]).numel() == 0

    def test_dirac_only_works_on_3_4_5d_inputs(self):
        for as_variable in [True, False]:
            for dims in [1, 2, 6]:
                with self.assertRaises(ValueError):
                    tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=3, as_variable=as_variable)
                    init.dirac(tensor)

    def test_xavier_uniform_errors_on_inputs_smaller_than_2d(self):
        for as_variable in [True, False]:
            for dims in [0, 1]:
                tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=1, as_variable=as_variable)
                with self.assertRaises(ValueError):
                    init.xavier_uniform(tensor)

    def test_xavier_normal_errors_on_inputs_smaller_than_2d(self):
        for as_variable in [True, False]:
            for dims in [0, 1]:
                tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=1, as_variable=as_variable)
                with self.assertRaises(ValueError):
                    init.xavier_normal(tensor)

    @unittest.skipIf(not TEST_SCIPY, "Scipy not found.")
    def test_xavier_uniform(self):
        for as_variable in [True, False]:
            for use_gain in [True, False]:
                for dims in [2, 4]:
                    input_tensor = self._create_random_nd_tensor(dims, size_min=20, size_max=25,
                                                                 as_variable=as_variable)
                    gain = 1

                    if use_gain:
                        gain = self._random_float(0.1, 2)
                        init.xavier_uniform(input_tensor, gain=gain)
                    else:
                        init.xavier_uniform(input_tensor)

                    if as_variable:
                        input_tensor = input_tensor.data

                    fan_in = input_tensor.size(1)
                    fan_out = input_tensor.size(0)
                    if input_tensor.dim() > 2:
                        fan_in *= input_tensor[0, 0].numel()
                        fan_out *= input_tensor[0, 0].numel()

                    expected_std = gain * math.sqrt(2.0 / (fan_in + fan_out))
                    bounds = expected_std * math.sqrt(3)
                    assert self._is_uniform(input_tensor, -bounds, bounds)

    @unittest.skipIf(not TEST_SCIPY, "Scipy not found.")
    def test_xavier_normal(self):
        for as_variable in [True, False]:
            for use_gain in [True, False]:
                for dims in [2, 4]:
                    input_tensor = self._create_random_nd_tensor(dims, size_min=20, size_max=25,
                                                                 as_variable=as_variable)
                    gain = 1

                    if use_gain:
                        gain = self._random_float(0.1, 2)
                        init.xavier_normal(input_tensor, gain=gain)
                    else:
                        init.xavier_normal(input_tensor)

                    if as_variable:
                        input_tensor = input_tensor.data

                    fan_in = input_tensor.size(1)
                    fan_out = input_tensor.size(0)
                    if input_tensor.dim() > 2:
                        fan_in *= input_tensor[0, 0].numel()
                        fan_out *= input_tensor[0, 0].numel()

                    expected_std = gain * math.sqrt(2.0 / (fan_in + fan_out))
                    assert self._is_normal(input_tensor, 0, expected_std)

    def test_kaiming_uniform_errors_on_inputs_smaller_than_2d(self):
        for as_variable in [True, False]:
            for dims in [0, 1]:
                with self.assertRaises(ValueError):
                    tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=1, as_variable=as_variable)
                    init.kaiming_uniform(tensor)

    def test_kaiming_normal_errors_on_inputs_smaller_than_2d(self):
        for as_variable in [True, False]:
            for dims in [0, 1]:
                with self.assertRaises(ValueError):
                    tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=1, as_variable=as_variable)
                    init.kaiming_normal(tensor)

    @unittest.skipIf(not TEST_SCIPY, "Scipy not found.")
    def test_kaiming_uniform(self):
        for as_variable in [True, False]:
            for use_a in [True, False]:
                for dims in [2, 4]:
                    for mode in ['fan_in', 'fan_out']:
                        input_tensor = self._create_random_nd_tensor(dims, size_min=20, size_max=25,
                                                                     as_variable=as_variable)
                        if use_a:
                            a = self._random_float(0.1, 2)
                            init.kaiming_uniform(input_tensor, a=a, mode=mode)
                        else:
                            a = 0
                            init.kaiming_uniform(input_tensor, mode=mode)

                        if as_variable:
                            input_tensor = input_tensor.data

                        fan_in = input_tensor.size(1)
                        fan_out = input_tensor.size(0)
                        if input_tensor.dim() > 2:
                            fan_in *= input_tensor[0, 0].numel()
                            fan_out *= input_tensor[0, 0].numel()

                        if mode == 'fan_in':
                            n = fan_in
                        else:
                            n = fan_out

                        expected_std = math.sqrt(2.0 / ((1 + a**2) * n))
                        bounds = expected_std * math.sqrt(3.0)
                        assert self._is_uniform(input_tensor, -bounds, bounds)

    @unittest.skipIf(not TEST_SCIPY, "Scipy not found.")
    def test_kaiming_normal(self):
        for as_variable in [True, False]:
            for use_a in [True, False]:
                for dims in [2, 4]:
                    for mode in ['fan_in', 'fan_out']:
                        input_tensor = self._create_random_nd_tensor(dims, size_min=20, size_max=25,
                                                                     as_variable=as_variable)
                        if use_a:
                            a = self._random_float(0.1, 2)
                            init.kaiming_normal(input_tensor, a=a, mode=mode)
                        else:
                            a = 0
                            init.kaiming_normal(input_tensor, mode=mode)

                        if as_variable:
                            input_tensor = input_tensor.data

                        fan_in = input_tensor.size(1)
                        fan_out = input_tensor.size(0)
                        if input_tensor.dim() > 2:
                            fan_in *= input_tensor[0, 0].numel()
                            fan_out *= input_tensor[0, 0].numel()

                        if mode == 'fan_in':
                            n = fan_in
                        else:
                            n = fan_out

                        expected_std = math.sqrt(2.0 / ((1 + a**2) * n))
                        assert self._is_normal(input_tensor, 0, expected_std)

    def test_sparse_only_works_on_2d_inputs(self):
        for as_variable in [True, False]:
            for dims in [1, 3]:
                with self.assertRaises(ValueError):
                    sparsity = self._random_float(0.1, 0.9)
                    tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=3, as_variable=as_variable)
                    init.sparse(tensor, sparsity)

    @unittest.skipIf(not TEST_SCIPY, "Scipy not found.")
    def test_sparse_default_std(self):
        for as_variable in [True, False]:
            for use_random_std in [True, False]:
                input_tensor = self._create_random_nd_tensor(2, size_min=30, size_max=35, as_variable=as_variable)
                rows, cols = input_tensor.size(0), input_tensor.size(1)
                sparsity = self._random_float(0.1, 0.2)

                std = 0.01  # default std
                if use_random_std:
                    std = self._random_float(0.01, 0.2)
                    init.sparse(input_tensor, sparsity=sparsity, std=std)
                else:
                    init.sparse(input_tensor, sparsity=sparsity)

                if as_variable:
                    input_tensor = input_tensor.data

                for col_idx in range(input_tensor.size(1)):
                    column = input_tensor[:, col_idx]
                    assert column[column == 0].nelement() >= math.ceil(sparsity * cols)

                assert self._is_normal(input_tensor[input_tensor != 0], 0, std)

    @skipIfNoLapack
    def test_orthogonal(self):
        for as_variable in [True, False]:
            for use_gain in [True, False]:
                for tensor_size in [[3, 4], [4, 3], [20, 2, 3, 4], [2, 3, 4, 5]]:
                    input_tensor = torch.zeros(tensor_size)
                    gain = 1.0

                    if as_variable:
                        input_tensor = Variable(input_tensor)

                    if use_gain:
                        gain = self._random_float(0.1, 2)
                        init.orthogonal(input_tensor, gain=gain)
                    else:
                        init.orthogonal(input_tensor)

                    if as_variable:
                        input_tensor = input_tensor.data

                    rows, cols = tensor_size[0], reduce(mul, tensor_size[1:])
                    flattened_tensor = input_tensor.view(rows, cols)
                    if rows > cols:
                        self.assertEqual(torch.mm(flattened_tensor.t(), flattened_tensor),
                                         torch.eye(cols) * gain ** 2, prec=1e-6)
                    else:
                        self.assertEqual(torch.mm(flattened_tensor, flattened_tensor.t()),
                                         torch.eye(rows) * gain ** 2, prec=1e-6)


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
        constructor_args=(4, 5, 3, 2),
        input_size=(2, 4, 10),
        cudnn=True,
        desc='stride'
    ),
    dict(
        module_name='Conv1d',
        constructor_args=(4, 5, 3, 1, 1),
        input_size=(2, 4, 10),
        cudnn=True,
        desc='pad1'
    ),
    dict(
        module_name='Conv1d',
        constructor_args=(4, 5, 5, 1, 2),
        input_size=(2, 4, 10),
        cudnn=True,
        desc='pad2'
    ),
    dict(
        module_name='Conv1d',
        constructor_args=(4, 4, 3, 1, 1),
        input_size=(1, 4, 1),
        cudnn=True,
        desc='pad1size1'
    ),
    dict(
        module_name='Conv1d',
        constructor_args=(4, 4, 5, 1, 2),
        input_size=(1, 4, 1),
        cudnn=True,
        desc='pad2size1'
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
    # TODO
    # dict(
    #     module_name='ConvTranspose1d',
    #     constructor_args=(3, 4, 3, 2, 1, 1, 1, True, 2),
    #     input_size=(1, 3, 6),
    #     cudnn=True,
    #     desc='dilated'
    # ),
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
    # TODO
    # dict(
    #     module_name='ConvTranspose2d',
    #     constructor_args=(3, 4, 3, (2, 3), 1, (1, 1), 1, False, (2, 2)),
    #     input_size=(1, 3, 6, 7),
    #     cudnn=True,
    #     desc='dilated'
    # ),
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
        module_name='ZeroPad2d',
        constructor_args=((1, 2, 3, 4),),
        input_size=(2, 3, 4, 4)
    ),
    dict(
        module_name='ConstantPad2d',
        constructor_args=((1, 2, 3, 4), 2),
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
        constructor_args=(3, 4, (2, 3, 4), 1, 0, 1, 1, False),
        input_size=(2, 3, 3, 4, 5),
        cudnn=True,
        desc='no_bias'
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
    # TODO
    # dict(
    #     module_name='ConvTranspose3d',
    #     constructor_args=(2, 3, (2, 3, 2), 1, 0, 0, 1, True, (2, 2, 2)),
    #     cudnn=True,
    #     input_size=(1, 2, 4, 5, 4),
    #     desc='dilated'
    # ),
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
        fullname='Embedding_sparse'
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
        module_name='Upsample',
        constructor_args=(12, None, 'nearest'),
        input_size=(1, 2, 4, 4),
        desc='nearest_2d'
    ),
    dict(
        module_name='Upsample',
        constructor_args=((12, 16), None, 'nearest'),
        input_size=(1, 2, 3, 4),
        desc='nearest_tuple_2d'
    ),
    dict(
        module_name='Upsample',
        constructor_args=(None, 4, 'nearest'),
        input_size=(1, 2, 4, 4),
        desc='nearest_scale_2d'
    ),
    dict(
        module_name='Upsample',
        constructor_args=(12, None, 'bilinear'),
        input_size=(1, 2, 4, 4),
        desc='bilinear_2d'
    ),
    dict(
        module_name='Upsample',
        constructor_args=((4, 6), None, 'bilinear'),
        input_size=(1, 2, 2, 3),
        desc='bilinear_tuple_2d'
    ),
    dict(
        module_name='Upsample',
        constructor_args=(None, 4, 'bilinear'),
        input_size=(1, 2, 4, 4),
        desc='bilinear_scale_2d'
    ),
    dict(
        module_name='Upsample',
        constructor_args=(None, (2, 2), 'bilinear'),
        input_size=(1, 2, 4, 4),
        desc='bilinear_scale_tuple_shared_2d'
    ),
    dict(
        module_name='Upsample',
        constructor_args=(None, (2, 1), 'bilinear'),
        input_size=(1, 2, 4, 4),
        desc='bilinear_scale_tuple_skewed_2d'
    ),
    dict(
        module_name='Upsample',
        constructor_args=(12, None, 'nearest'),
        input_size=(1, 2, 4, 4, 4),
        desc='nearest_3d'
    ),
    dict(
        module_name='Upsample',
        constructor_args=((12, 16, 16), None, 'nearest'),
        input_size=(1, 2, 3, 4, 4),
        desc='nearest_tuple_3d'
    ),
    dict(
        module_name='Upsample',
        constructor_args=(None, 4, 'nearest'),
        input_size=(1, 2, 4, 4, 4),
        desc='nearest_scale_3d'
    ),
    dict(
        module_name='Upsample',
        constructor_args=(12, None, 'trilinear'),
        input_size=(1, 2, 4, 4, 4),
        desc='trilinear_3d'
    ),
    dict(
        module_name='Upsample',
        constructor_args=((4, 6, 6), None, 'trilinear'),
        input_size=(1, 2, 2, 3, 3),
        desc='trilinear_tuple_3d'
    ),
    dict(
        module_name='Upsample',
        constructor_args=(None, 4, 'trilinear'),
        input_size=(1, 2, 4, 4, 4),
        desc='trilinear_scale_3d'
    ),
    dict(
        module_name='AdaptiveMaxPool1d',
        constructor_args=(3,),
        input=torch.rand(1, 3, 5)
    ),
    dict(
        module_name='AdaptiveMaxPool2d',
        constructor_args=(3,),
        input=torch.rand(1, 3, 5, 6),
        desc='single'
    ),
    dict(
        module_name='AdaptiveMaxPool2d',
        constructor_args=((3, 4),),
        input=torch.rand(1, 3, 5, 6),
        desc='tuple'
    ),
    dict(
        module_name='AdaptiveAvgPool1d',
        constructor_args=(3,),
        input=torch.rand(1, 3, 5)
    ),
    dict(
        module_name='AdaptiveAvgPool2d',
        constructor_args=(3,),
        input=torch.rand(1, 3, 5, 6),
        desc='single'
    ),
    dict(
        module_name='AdaptiveAvgPool2d',
        constructor_args=((3, 4),),
        input=torch.rand(1, 3, 5, 6),
        desc='tuple'
    ),
    dict(
        module_name='SELU',
        input_size=(3, 2, 5),
        check_inplace=True
    ),
    dict(
        module_name='GLU',
        input_size=(5, 6),
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
