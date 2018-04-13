import math
import random
import string
import unittest
import itertools
import contextlib
import warnings
import pickle
from copy import deepcopy
from itertools import repeat, product
from functools import wraps, reduce
from operator import mul
from collections import OrderedDict

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel as dp
import torch.nn.init as init
import torch.nn.utils.rnn as rnn_utils
import torch.legacy.nn as legacy
from torch.nn.utils import clip_grad_norm
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.autograd import Variable, gradcheck
from torch.autograd.gradcheck import gradgradcheck
from torch.nn import Parameter
from torch.nn.parallel._functions import Broadcast
from common_nn import NNTestCase, ModuleTest, CriterionTest, TestBase, \
    module_tests, criterion_tests, TEST_CUDA, TEST_MULTIGPU, TEST_CUDNN, \
    TEST_CUDNN_VERSION, loss_reference_fns, get_size_average, get_weight, \
    smoothl1loss_reference, kldivloss_reference
from common import freeze_rng_state, run_tests, TestCase, skipIfNoLapack, \
    TEST_SCIPY, download_file, PY3, PY34, to_gpu, get_function_arglist

if TEST_SCIPY:
    from scipy import stats

ALL_TENSORTYPES = [torch.FloatTensor,
                   torch.DoubleTensor,
                   torch.HalfTensor]

NO_HALF_TENSORTYPES = [torch.FloatTensor,
                       torch.DoubleTensor]

DOUBLE_TENSORTYPES = [torch.DoubleTensor]

type2prec = {'FloatTensor': 1e-5,
             'DoubleTensor': 1e-5,
             'HalfTensor': 1e-2}


# WARNING: If you add a new top-level test case to this file, you MUST
# update test/run_test.sh to list it, otherwise it will NOT be run in
# CI.


# Used to run the same test with different tensor types
def repeat_test_for_types(dtypes):
    def repeat_helper(f):
        def call_helper(self, *args):
            for dtype in dtypes:
                if PY34:
                    with TestCase.subTest(self, dtype=dtype):
                        f(self, *args, dtype=dtype)
                else:
                    f(self, *args, dtype=dtype)

        return call_helper
    return repeat_helper


class PackedSequenceTest(TestCase):

    _type_by_name = {
        'torch.DoubleTensor': (torch.DoubleTensor, 'double'),
        'torch.FloatTensor': (torch.FloatTensor, 'float'),
        # We leave out `'torch.HalfTensor': (torch.HalfTensor, 'half'),`
        # because of an error in `pad_packed_sequence`
        # > AttributeError: 'torch.HalfTensor' object has no attribute 'fill_'
        'torch.LongTensor': (torch.LongTensor, 'long'),
        'torch.IntTensor': (torch.IntTensor, 'int'),
        'torch.ShortTensor': (torch.ShortTensor, 'short'),
        'torch.CharTensor': (torch.CharTensor, 'char'),
        'torch.ByteTensor': (torch.ByteTensor, 'byte'),
    }

    def __init__(self, *args, **kwargs):
        super(PackedSequenceTest, self).__init__(*args, **kwargs)
        self.batch_size = 5
        self.max_length = 6

    def _ordered_sequence(self, tensor_type):
        """Create ordered list of random sequences"""
        seqs = [tensor_type(random.randint(1, self.max_length))
                for _ in range(self.batch_size)]
        seqs = [Variable(s.random_()) for s in seqs]
        ordered = sorted(seqs, key=len, reverse=True)
        return ordered

    def _padded_sequence(self, tensor_type):
        """Create Variable of random padded sequences"""
        ordered = self._ordered_sequence(tensor_type)
        lengths = list(map(len, ordered))
        padded_tensor = rnn_utils.pad_sequence(ordered)
        return padded_tensor, lengths

    def test_type_casts(self):
        """Test type casting of `PackedSequence` against type casting of tensor"""
        for _, (input_type, _) in self._type_by_name.items():
            for expected_type_str, (_, cast_str) in self._type_by_name.items():
                padded, lengths = self._padded_sequence(input_type)
                packed = rnn_utils.pack_padded_sequence(padded, lengths)
                # Apply cast to `PackedSequence` instance and unpack
                masked = getattr(packed, cast_str)()
                unpacked, lengths_out = rnn_utils.pad_packed_sequence(masked)
                self.assertEqual(unpacked.type(), expected_type_str)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_cuda_mask(self):
        tensor_type = torch.FloatTensor
        cuda_type_str = 'torch.cuda.FloatTensor'
        padded, lengths = self._padded_sequence(tensor_type)
        packed = rnn_utils.pack_padded_sequence(padded, lengths)
        self.assertFalse(packed.is_cuda)
        packed = packed.cuda()
        self.assertTrue(packed.is_cuda)
        unpacked, _ = rnn_utils.pad_packed_sequence(packed)
        self.assertEqual(unpacked.type(), cuda_type_str)

    def test_total_length(self):
        padded, lengths = self._padded_sequence(torch.FloatTensor)
        max_length = max(lengths)
        packed = rnn_utils.pack_padded_sequence(padded, lengths)
        # test ValueError if total_length < max_length
        for total_length in (-1, 0, max_length - 1):
            for batch_first in (True, False):
                def err_fn():
                    rnn_utils.pad_packed_sequence(packed, batch_first=batch_first,
                                                  total_length=total_length)
            self.assertRaisesRegex(ValueError,
                                   r'Expected total_length to be at least the '
                                   r'length of the longest sequence in input',
                                   err_fn)
        # test that pad_packed_sequence returns results of correct length
        for batch_first in (True, False):
            no_extra_pad, _ = rnn_utils.pad_packed_sequence(packed, batch_first=batch_first)
            for total_length_delta in (0, 1, 8):
                total_length = max_length + total_length_delta
                unpacked, lengths_out = rnn_utils.pad_packed_sequence(packed, batch_first=batch_first,
                                                                      total_length=total_length)
                self.assertEqual(lengths, lengths_out)
                self.assertEqual(unpacked.size(1 if batch_first else 0), total_length)
                if total_length_delta == 0:
                    ref_output = no_extra_pad
                elif batch_first:
                    extra_pad = no_extra_pad.new_zeros(self.batch_size, total_length_delta)
                    ref_output = torch.cat([no_extra_pad, extra_pad], 1)
                else:
                    extra_pad = no_extra_pad.new_zeros(total_length_delta, self.batch_size)
                    ref_output = torch.cat([no_extra_pad, extra_pad], 0)
                self.assertEqual(unpacked, ref_output)


def default_tensor_type(type):
    type_str = torch.typename(type)

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            old_type = torch.Tensor().type()
            torch.set_default_tensor_type(type_str)
            try:
                return fn(*args, **kwargs)
            finally:
                torch.set_default_tensor_type(old_type)

        return wrapper

    return decorator


def _assertGradAndGradgradChecks(test_case, apply_fn, inputs):
    # call assert function rather than returning a bool since it's nicer
    # if we get whether this failed on the gradcheck or the gradgradcheck.
    test_case.assertTrue(gradcheck(apply_fn, inputs))
    test_case.assertTrue(gradgradcheck(apply_fn, inputs))


class InputVariableMixin(object):
    def _get_input(self):
        input = TestBase._get_input(self, False)

        def map_variables(i):
            if isinstance(i, torch.Tensor):
                if i.is_floating_point():
                    i.requires_grad = True
                return i
            else:
                return type(i)(map_variables(elem) for elem in i)

        return map_variables(input)


class NewModuleTest(InputVariableMixin, ModuleTest):
    def __init__(self, *args, **kwargs):
        super(NewModuleTest, self).__init__(*args, **kwargs)
        self.cudnn = kwargs.get('cudnn', False)
        self.check_inplace = kwargs.get('check_inplace', False)
        self.check_gradgrad = kwargs.get('check_gradgrad', True)

    def _do_test(self, test_case, module, input):
        test_case.check_jacobian(module, input, self.jacobian_input)

        if self.check_gradgrad:
            # could probably unify check_jacobian above with this.
            params = tuple(x for x in module.parameters())
            _assertGradAndGradgradChecks(test_case,
                                         lambda x, *args, **kw: test_case._forward(module, x), (input,) + params)

        # check if module can be printed
        module.__repr__()

        if self.check_inplace:
            # check if the inplace variant of the module gives the same result
            # as the out-of-place

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

        if isinstance(input, torch.LongTensor) and TEST_CUDA:
            # check that cuda() moves module parameters to correct GPU device,
            # and that float() casts parameters correctly

            input = input.cuda()
            module.float().cuda()
            module(input)
            for p in module.parameters():
                test_case.assertIsInstance(p, torch.cuda.FloatTensor)
                test_case.assertEqual(p.get_device(), 0)

            if torch.cuda.device_count() > 1:
                input = input.cuda(1)
                module.cuda(1)
                with torch.cuda.device(1):
                    module(input)
                for p in module.parameters():
                    test_case.assertIsInstance(p, torch.cuda.FloatTensor)
                    test_case.assertEqual(p.get_device(), 1)
        else:
            # check that float()/double() casters work correctly

            # to float
            if not isinstance(input, torch.LongTensor):
                input = input.float()
            module.float()
            module(input)
            for p in module.parameters():
                test_case.assertIsInstance(p, torch.FloatTensor)

            # and back to double
            if not isinstance(input, torch.LongTensor):
                input = input.double()
            module.double()
            module(input)
            for p in module.parameters():
                test_case.assertIsInstance(p, torch.DoubleTensor)

            # TODO: Hardshrink is lacking a CUDA implementation
            if TEST_CUDA and self.should_test_cuda and type(module) != nn.Hardshrink:
                # check that cuda() moves module parameters to correct GPU device,
                # and that float() casts parameters correctly

                # to GPU0
                input = input.float().cuda()
                module.float().cuda()
                module(input)
                for p in module.parameters():
                    test_case.assertIsInstance(p, torch.cuda.FloatTensor)
                    test_case.assertEqual(p.get_device(), 0)

                # to CPU
                input = input.cpu()
                module.cpu()
                module(input)
                for p in module.parameters():
                    test_case.assertIsInstance(p, torch.FloatTensor)

                # back to GPU0
                input = input.cuda()
                module.cuda()
                module(input)
                for p in module.parameters():
                    test_case.assertIsInstance(p, torch.cuda.FloatTensor)
                    test_case.assertEqual(p.get_device(), 0)

                # test that forwards of module runs correctly without cuDNN
                if self.cudnn:
                    with torch.backends.cudnn.flags(enabled=False):
                        module(input)
                        for p in module.parameters():
                            test_case.assertIsInstance(p, torch.cuda.FloatTensor)
                            test_case.assertEqual(p.get_device(), 0)

                if torch.cuda.device_count() >= 2:
                    # test cross-GPU transfer works
                    # to GPU1
                    input = input.cuda(1)
                    module.cuda(1)
                    with torch.cuda.device(1):
                        module(input)
                    for p in module.parameters():
                        test_case.assertIsInstance(p, torch.cuda.FloatTensor)
                        test_case.assertEqual(p.get_device(), 1)

                # test double()
                input = input.double().cuda()
                module.double().cuda()
                module(input)
                for p in module.parameters():
                    test_case.assertIsInstance(p, torch.cuda.DoubleTensor)
                    test_case.assertEqual(p.get_device(), 0)

                # test half()
                input = input.half().cuda()
                module.half().cuda()
                module(input)
                for o in module.parameters():
                    test_case.assertIsInstance(p, torch.cuda.HalfTensor)
                    test_case.assertEqual(p.get_device(), 0)

    def _get_target(self):
        return self._get_arg('target', False)

    @property
    def constructor_args(self):
        return self._get_arg('constructor_args', False)


class NewCriterionTest(InputVariableMixin, CriterionTest):
    # TODO: check that criterions don't ignore grad_output

    def __init__(self, *args, **kwargs):
        super(NewCriterionTest, self).__init__(*args, **kwargs)
        self.check_gradgrad = kwargs.get('check_gradgrad', True)

    def _do_extra_tests(self, test_case, module, input, target):
        if not self.check_gradgrad:
            return

        test_case.assertFalse(target.requires_grad)

        params = tuple(x for x in module.parameters())
        if not isinstance(input, tuple):
            inputs = (input,) + params

            def apply_fn(input, *params):
                return module(input, target)
        else:
            inputs = input + params

            def apply_fn(input1, input2, *params):
                return module(input1, input2, target)

        # TODO: we don't pass `target` as part of inputs because we don't
        # currently compute the gradient w.r.t. target for loss functions.
        gradcheck(apply_fn, inputs)
        gradgradcheck(apply_fn, inputs)

    def test_cuda(self, test_case, dtype=None):
        def convert_dtype(obj, dtype, requires_grad=False):
            if isinstance(obj, Variable):
                return Variable(obj.data.type(dtype), requires_grad=requires_grad)
            elif torch.is_tensor(obj):
                return obj.type(dtype)
            elif isinstance(obj, tuple):
                return tuple(convert_dtype(o, dtype, requires_grad) for o in obj)
            else:
                return obj

        if not TEST_CUDA or not self.should_test_cuda:
            raise unittest.SkipTest('Excluded from CUDA tests')
        try:
            cpu_input = self._get_input()
            cpu_target = self._get_target()
            cpu_module = self.constructor(*self.constructor_args)
            gpu_module = self.constructor(*self.constructor_args)

            # Convert input, target and module parameters to dtype
            if dtype is not None:
                cpu_input = convert_dtype(cpu_input, dtype, True)
                # NLLLoss requires target to be LongTensor
                if not isinstance(cpu_target, torch.LongTensor):
                    cpu_target = convert_dtype(cpu_target, dtype)
                cpu_module.type(dtype)
                gpu_module.type(dtype)

            # GPU setup
            gpu_input = to_gpu(cpu_input)
            gpu_target = to_gpu(cpu_target)
            gpu_module.cuda()

            # torch.HalfTensor doesn't support most operations, converting back to default
            if dtype == torch.HalfTensor:
                cpu_input = self._get_input()
                cpu_target = self._get_target()
                # Loss modules with weights require consistent input/module weight types
                cpu_module = self.constructor(*self.constructor_args)

            cpu_output = test_case._forward_criterion(cpu_module, cpu_input, cpu_target)
            gpu_output = test_case._forward_criterion(gpu_module, gpu_input, gpu_target)
            # dtype can be None, so set precision in this way instead of a precision map
            test_case.assertEqual(cpu_output, gpu_output, 1e-1 if dtype == torch.HalfTensor else 4e-4)

            cpu_gradInput = test_case._backward_criterion(cpu_module, cpu_input, cpu_target)
            gpu_gradInput = test_case._backward_criterion(gpu_module, gpu_input, gpu_target)
            test_case.assertEqual(cpu_gradInput, gpu_gradInput, 1e-1 if dtype == torch.HalfTensor else 4e-4)
        except NotImplementedError:
            pass

    def _get_target(self):
        return self._get_arg('target', False)

    @property
    def constructor_args(self):
        return self._get_arg('constructor_args', False)


class TestNN(NNTestCase):
    def _forward(self, module, input):
        with freeze_rng_state():
            return module(input)

    def _backward(self, module, input, output, grad_output, create_graph=False):
        output.backward(grad_output, retain_graph=True, create_graph=create_graph)
        if input.grad is None:
            return None
        return input.grad.data

    def _forward_criterion(self, criterion, input, target):
        if isinstance(input, tuple):
            args = input + (target,)
            output = criterion(*args)
        else:
            output = criterion(input, target)
        return output.item()

    def _backward_criterion(self, criterion, input, target, gradOutput=None):
        input_tuple = input if isinstance(input, tuple) else (input,)
        for i in input_tuple:
            if i.grad is not None:
                i.grad.data.zero_()
        args = input_tuple + (target,)
        if gradOutput is None:
            gradOutput = torch.ones(())
        criterion(*args).backward(gradOutput.type_as(input_tuple[0]))
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
                p._grad = torch.zeros_like(p)
            params.append(p)
            d_params.append(p.grad)
        return params, d_params

    def test_module_backcompat(self):
        from torch.serialization import SourceChangeWarning
        path = download_file('https://download.pytorch.org/test_data/linear.pt')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', SourceChangeWarning)
            m = torch.load(path)
        input = Variable(torch.randn(2, 3).float())
        self.assertEqual(m(input).size(), (2, 5))

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

    def test_no_grad(self):
        module = nn.Conv2d(2, 5, kernel_size=3, padding=1)
        input = torch.randn(1, 2, 10, 10)
        x = Variable(input)
        y = Variable(input.clone())

        output = module(x)
        self.assertTrue(output.requires_grad)
        output.backward(torch.ones(1, 5, 10, 10))

        with torch.no_grad():
            output2 = module(y)
            self.assertFalse(output2.requires_grad)
            self.assertRaises(RuntimeError, lambda: output2.backward(torch.ones(1, 5, 10, 10)))

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

    def test_call_supports_python_dict_output(self):
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.l1 = nn.Linear(10, 20)
                self.register_backward_hook(self.hook)
                self.check_backward_hook_flag = False

            def hook(self, module, grad_out, grad_in):
                self.check_backward_hook_flag = True

            def forward(self, inputs):
                return {"output": self.l1(inputs).sum()}

        net = Net()
        model_output = net(Variable(torch.randn([5, 10])))
        model_output["output"].backward()
        self.assertTrue(net.check_backward_hook_flag)

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
        keys = dir(linear)
        self.assertIn('_test_submodule', keys)
        self.assertIn('_test_parameter', keys)
        self.assertIn('_test_buffer', keys)

        for key in keys:
            self.assertTrue(hasattr(linear, key))

    def test_repr(self):
        # no extra information or sub-modules
        empty_sequential = nn.Sequential()
        expected_repr_empty = 'Sequential()'
        self.assertEqual(repr(empty_sequential), expected_repr_empty)

        # one liner extra information
        linear = nn.Linear(1, 1)
        expected_repr_linear = 'Linear(in_features=1, out_features=1, bias=True)'
        self.assertEqual(repr(linear), expected_repr_linear)

        # sub-modules repr
        sequential = nn.Sequential(linear)
        expected_repr_sequential = 'Sequential(\n' \
            '  (0): Linear(in_features=1, out_features=1, bias=True)\n' \
            ')'
        self.assertEqual(repr(sequential), expected_repr_sequential)

    def test_dir_digit(self):
        model = nn.Sequential(nn.Linear(2, 2))
        keys = dir(model)
        self.assertNotIn('0', keys)

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

    def test_register_buffer_raises_error_if_attr_exists(self):
        m = nn.Module()
        m.attribute_name = 5
        with self.assertRaises(KeyError):
            m.register_buffer('attribute_name', torch.rand(5))

        del m.attribute_name
        m.register_parameter('attribute_name', nn.Parameter())
        with self.assertRaises(KeyError):
            m.register_buffer('attribute_name', torch.rand(5))

        del m.attribute_name
        m.add_module('attribute_name', nn.Module())
        with self.assertRaises(KeyError):
            m.register_buffer('attribute_name', torch.rand(5))

    def test_register_buffer_raises_error_if_not_tensor(self):
        m = nn.Module()
        with self.assertRaises(TypeError):
            m.register_buffer('attribute_name', 5)

    def test_register_buffer_allows_overwriting_with_same_name(self):
        m = nn.Module()
        buffer1 = torch.rand(5)
        buffer2 = buffer1 + 5
        buffer3 = None
        m.register_buffer('buffer_name', buffer1)
        self.assertEqual(m.buffer_name, buffer1)
        m.register_buffer('buffer_name', buffer2)
        self.assertEqual(m.buffer_name, buffer2)
        m.register_buffer('buffer_name', buffer3)
        self.assertEqual(m.buffer_name, buffer3)

    def test_register_parameter_raises_error_if_attr_exists(self):
        m = nn.Module()
        m.attribute_name = 5
        with self.assertRaises(KeyError):
            m.register_parameter('attribute_name', nn.Parameter())

        del m.attribute_name
        m.register_buffer('attribute_name', torch.rand(5))
        with self.assertRaises(KeyError):
            m.register_parameter('attribute_name', nn.Parameter())

        del m.attribute_name
        m.add_module('attribute_name', nn.Module())
        with self.assertRaises(KeyError):
            m.register_parameter('attribute_name', nn.Parameter())

    def test_register_parameter_allows_overwriting_with_same_name(self):
        m = nn.Module()
        param1 = nn.Parameter(torch.rand(5))
        param2 = nn.Parameter(param1.data + 5)
        param3 = None
        m.register_parameter('param_name', param1)
        self.assertEqual(m.param_name, param1)
        m.register_parameter('param_name', param2)
        self.assertEqual(m.param_name, param2)
        m.register_parameter('param_name', param3)
        self.assertEqual(m.param_name, param3)

    def test_add_module_raises_error_if_attr_exists(self):
        m = nn.Module()
        m.attribute_name = 5
        with self.assertRaises(KeyError):
            m.add_module('attribute_name', nn.Module())

        del m.attribute_name
        m.register_buffer('attribute_name', torch.rand(5))
        with self.assertRaises(KeyError):
            m.add_module('attribute_name', nn.Module())

        del m.attribute_name
        m.register_parameter('attribute_name', nn.Parameter())
        with self.assertRaises(KeyError):
            m.add_module('attribute_name', nn.Module())

    def test_Sequential_getitem(self):
        l1 = nn.Linear(10, 20)
        l2 = nn.Linear(20, 30)
        l3 = nn.Linear(30, 40)
        l4 = nn.Linear(40, 50)
        n = nn.Sequential(l1, l2, l3, l4)
        self.assertIs(n[0], l1)
        self.assertIs(n[1], l2)
        self.assertIs(n[2], l3)
        self.assertIs(n[3], l4)
        self.assertIs(n[torch.tensor(3, dtype=torch.int64)], l4)
        self.assertEqual(n[1:], nn.Sequential(l2, l3, l4))
        self.assertEqual(n[3:], nn.Sequential(l4))
        self.assertEqual(n[:-1], nn.Sequential(l1, l2, l3))
        self.assertEqual(n[:-3], nn.Sequential(l1))
        self.assertEqual(n[::-1], nn.Sequential(l4, l3, l2, l1))

    def test_Sequential_setitem(self):
        l1 = nn.Linear(10, 20)
        l2 = nn.Linear(20, 30)
        l3 = nn.Linear(30, 40)
        l4 = nn.Linear(40, 50)
        n = nn.Sequential(l1, l2, l3)
        n[0] = l4
        n[-1] = l4
        n[torch.tensor(1, dtype=torch.int16)] = l1
        self.assertIs(n[0], l4)
        self.assertIs(n[1], l1)
        self.assertIs(n[2], l4)

    def test_Sequential_setitem_named(self):
        l1 = nn.Linear(10, 20)
        l2 = nn.Linear(20, 30)
        l3 = nn.Linear(30, 40)
        l4 = nn.Linear(40, 50)
        n = nn.Sequential(OrderedDict([
            ('linear1', l1),
            ('linear2', l2),
            ('linear3', l3),
        ]))

        n[0] = l4
        n[-1] = l4
        self.assertEqual(n.linear1, l4)
        self.assertEqual(n.linear3, l4)

    def test_Sequential_delitem(self):
        l1 = nn.Linear(10, 20)
        l2 = nn.Linear(20, 30)
        l3 = nn.Linear(30, 40)
        l4 = nn.Linear(40, 50)
        n = nn.Sequential(l1, l2, l3, l4)
        del n[-1]
        self.assertEqual(n, nn.Sequential(l1, l2, l3))
        del n[1::2]
        self.assertEqual(n, nn.Sequential(l1, l3))

    def test_ModuleList(self):
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
        idx = torch.tensor(2, dtype=torch.int32)
        modules[2] = nn.Conv2d(5, 3, 2)
        module_list[idx] = modules[2]
        self.assertIs(module_list[idx], modules[2])
        check()
        self.assertEqual(module_list[1:], nn.ModuleList(modules[1:]))
        self.assertEqual(module_list[3:], nn.ModuleList(modules[3:]))
        self.assertEqual(module_list[:-1], nn.ModuleList(modules[:-1]))
        self.assertEqual(module_list[:-3], nn.ModuleList(modules[:-3]))
        self.assertEqual(module_list[::-1], nn.ModuleList(modules[::-1]))
        del module_list[-1]
        self.assertEqual(module_list, nn.ModuleList(modules[:-1]))
        del module_list[1::2]
        self.assertEqual(module_list, nn.ModuleList(modules[:-1][0::2]))

        with self.assertRaises(TypeError):
            module_list += nn.ReLU()
        with self.assertRaises(TypeError):
            module_list.extend(nn.ReLU())

        l1 = nn.Linear(1, 2)
        l2 = nn.Linear(2, 3)
        l3 = nn.Linear(3, 2)
        l4 = nn.Linear(2, 3)
        subnet = nn.Sequential(l3, l4)
        s = nn.Sequential(
            OrderedDict([
                ("layer1", l1),
                ("layer2", l2),
                ("layer3", l3),
                ("layer4", l4),
                ("subnet_layer", subnet)
            ])
        )
        modules = list(s.modules())
        module_list = nn.ModuleList()
        module_list.extend(s.modules())
        check()

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
        idx = torch.tensor(2, dtype=torch.int32)
        parameters[2] = make_param()
        param_list[idx] = parameters[2]
        self.assertIs(param_list[idx], parameters[2])
        check()
        self.assertEqual(param_list[1:], nn.ParameterList(parameters[1:]))
        self.assertEqual(param_list[3:], nn.ParameterList(parameters[3:]))
        self.assertEqual(param_list[:-1], nn.ParameterList(parameters[:-1]))
        self.assertEqual(param_list[:-3], nn.ParameterList(parameters[:-3]))
        self.assertEqual(param_list[::-1], nn.ParameterList(parameters[::-1]))

        with self.assertRaises(TypeError):
            param_list += make_param()
        with self.assertRaises(TypeError):
            param_list.extend(make_param())

        l1 = nn.Linear(1, 2)
        l2 = nn.Linear(2, 3)
        l3 = nn.Linear(3, 2)
        l4 = nn.Linear(2, 3)
        subnet = nn.Sequential(l3, l4)
        s = nn.Sequential(
            OrderedDict([
                ("layer1", l1),
                ("layer2", l2),
                ("layer3", l3),
                ("layer4", l4),
                ("subnet_layer", subnet)
            ])
        )
        parameters = list(s.parameters())
        param_list = nn.ParameterList()
        param_list.extend(s.parameters())
        check()

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
        l3 = nn.Linear(20, 10)
        net.add_module('l', l3)
        self.assertEqual(net.l, l3)
        self.assertRaises(TypeError, lambda: net.add_module('x', 'non-module'))

    def test_type(self):
        l = nn.Linear(10, 20)
        net = nn.Module()
        net.l = l
        net.l2 = l
        net.add_module('empty', None)
        net.register_buffer('indices', torch.LongTensor(1))
        net.float()
        self.assertIsInstance(l.weight.data, torch.FloatTensor)
        self.assertIsInstance(l.bias.data, torch.FloatTensor)
        self.assertIsInstance(net.indices, torch.LongTensor)
        net.double()
        self.assertIsInstance(l.weight.data, torch.DoubleTensor)
        self.assertIsInstance(l.bias.data, torch.DoubleTensor)
        self.assertIsInstance(net.indices, torch.LongTensor)
        if TEST_CUDA:
            net.float().cuda()
            self.assertIsInstance(l.weight.data, torch.cuda.FloatTensor)
            self.assertIsInstance(l.bias.data, torch.cuda.FloatTensor)
            self.assertIsInstance(net.indices, torch.cuda.LongTensor)
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

    def test_parameters_to_vector(self):
        conv1 = nn.Conv2d(3, 10, 5)
        fc1 = nn.Linear(10, 20)
        model = nn.Sequential(conv1, fc1)

        vec = parameters_to_vector(model.parameters())
        self.assertEqual(vec.size(0), 980)

    def test_vector_to_parameters(self):
        conv1 = nn.Conv2d(3, 10, 5)
        fc1 = nn.Linear(10, 20)
        model = nn.Sequential(conv1, fc1)

        vec = Variable(torch.arange(0, 980))
        vector_to_parameters(vec, model.parameters())

        sample = next(model.parameters())[0, 0, 0]
        self.assertTrue(torch.equal(sample.data, vec.data[:5]))

    def test_weight_norm(self):
        input = Variable(torch.randn(3, 5))
        m = nn.Linear(5, 7)
        expected_output = m(input)

        # add weight normalization
        m = torch.nn.utils.weight_norm(m)
        self.assertEqual(m.weight_v.size(), m.weight.size())
        self.assertEqual(m.weight_g.size(), (7, 1))
        self.assertEqual(m(input), expected_output)

        # remove weight norm
        m = torch.nn.utils.remove_weight_norm(m)
        self.assertFalse(hasattr(m, 'weight_g'))
        self.assertFalse(hasattr(m, 'weight_v'))
        self.assertEqual(m(input), expected_output)

        # test with dim=1
        m = torch.nn.utils.weight_norm(m, dim=1)
        self.assertEqual(m.weight_v.size(), m.weight.size())
        self.assertEqual(m.weight_g.size(), (1, 5))
        self.assertEqual(m(input), expected_output)

    def test_weight_norm_pickle(self):
        m = torch.nn.utils.weight_norm(nn.Linear(5, 7))
        m = pickle.loads(pickle.dumps(m))
        self.assertIsInstance(m, nn.Linear)

    def test_embedding_sparse_basic(self):
        embedding = nn.Embedding(10, 20, sparse=True)
        input = Variable(torch.LongTensor([[0, 2, 4, 5], [4, 3, 0, 9]]))
        embedding(input).sum().backward()
        self.assertTrue(embedding.weight.grad.is_sparse)
        self.assertEqual(embedding.weight.grad.shape, embedding.weight.shape)

    def test_embedding_padding_idx(self):
        embedding = nn.Embedding(10, 20, padding_idx=0)
        input = Variable(torch.LongTensor([[0, 2, 4, 5], [4, 3, 0, 9]]))
        output = embedding(input)
        self.assertEqual(output[0][0].sum(), 0)
        self.assertEqual(output[1][2].sum(), 0)

        embedding = nn.Embedding(10, 20, padding_idx=0, sparse=True)
        input = Variable(torch.LongTensor([[0, 2, 4, 5], [4, 3, 0, 9]]))
        output = embedding(input)
        self.assertEqual(output[0][0].sum(), 0)
        self.assertEqual(output[1][2].sum(), 0)

        # negative indexing check for padding_idx
        # padding_idx=-2, num_embeddings=10 ==> index 8 padded
        embedding = nn.Embedding(10, 20, padding_idx=-2)
        input = Variable(torch.LongTensor([[0, 2, 8, 5], [4, 8, 0, 9]]))
        output = embedding(input)
        self.assertEqual(output[0][2].sum(), 0)
        self.assertEqual(output[1][1].sum(), 0)

        embedding = nn.Embedding(10, 20, padding_idx=-2, sparse=True)
        input = Variable(torch.LongTensor([[0, 2, 8, 5], [4, 8, 0, 9]]))
        output = embedding(input)
        self.assertEqual(output[0][2].sum(), 0)
        self.assertEqual(output[1][1].sum(), 0)

        # out of bounds check for padding_idx
        self.assertRaises(AssertionError, nn.Embedding, num_embeddings=10, embedding_dim=20, padding_idx=25)
        self.assertRaises(AssertionError, nn.Embedding, num_embeddings=10, embedding_dim=20, padding_idx=-25)

        # test backward when input contains padding_idx
        padding_idx = 0
        embedding = nn.Embedding(5, 2, padding_idx=padding_idx)
        for n in (1, 2):
            for other_indices in ([], [1, 3], [2]):
                indices = torch.LongTensor(other_indices + [padding_idx] * n)
                pre = embedding.weight[padding_idx].clone()
                embedding(indices).sum().backward()
                after = (embedding.weight + embedding.weight.grad)[padding_idx]
                embedding.zero_grad()
                self.assertEqual(after, pre)

    def test_embedding_max_norm(self):
        embedding = nn.Embedding(22, 5, max_norm=1.0)
        input = Variable(torch.LongTensor([2, 8, 8, 6]))
        output = embedding(input)
        self.assertEqual(output[1], output[2])
        self.assertTrue(output.data.norm(p=2, dim=1).le(1).all())

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    @repeat_test_for_types(ALL_TENSORTYPES)
    def test_embedding_max_norm_cuda(self, dtype=torch.FloatTensor):
        embedding = nn.Embedding(22, 5, max_norm=1.0).type(dtype).cuda()
        # nn.Embedding only takes LongTensor as input
        input = Variable(torch.LongTensor([2, 8, 8, 6])).cuda()
        output = embedding(input)
        self.assertEqual(output[1], output[2])
        self.assertTrue(output.data.norm(p=2, dim=1).le(1).all())

    def test_embedding_from_pretrained(self):
        a = torch.Tensor([[1, 2, 3], [4, 5, 6]])
        embedding = nn.Embedding.from_pretrained(a)
        self.assertEqual(a, embedding.weight.data)

        input = Variable(torch.LongTensor([0, 1]))
        output = embedding(input)
        self.assertEqual(a, output)

    def test_embedding_functional(self):
        a = Variable(torch.LongTensor([
            [1, 3, 2],
            [0, 2, 1]
        ]))
        embeddings = Variable(torch.rand(4, 3), requires_grad=True)

        embed_old = torch.nn.Embedding(4, 3)
        embed_old.weight.data = embeddings.data
        res_old = embed_old(a)

        res_F = F.embedding(a, embeddings)
        self.assertEqual(res_old, res_F)

    def _test_gumbel_softmax_st(self, cuda, dtype=torch.FloatTensor):
        th = torch.cuda if cuda else torch
        """
        Things we might want to check:
        - if we make various draws, do we get different one-hot values?
        - is the proportion approximately in line with the softmax values?
        - with hard, is it one-hot?
        - with hard, is there still a gradient?
        """
        num_draws = 100
        K = 3
        logits = torch.FloatTensor([[0.2, 0.8, 0.1]])
        if dtype != torch.HalfTensor:
            logits = logits.type(dtype)
        logits_softmax = torch.nn.functional.softmax(Variable(logits), 1)
        y_draws = torch.zeros(num_draws, K)
        preds = torch.zeros(num_draws)

        if cuda:
            logits = logits.cuda()
            y_draws = y_draws.cuda()
            preds = preds.cuda()

        exceed_limits = 0
        for draw in range(num_draws):
            logits_var = Variable(logits, requires_grad=True)
            y_draw = torch.nn.functional.gumbel_softmax(
                logits_var,
                hard=True)
            assert y_draw.size() == logits.size()
            # check we have a gradient
            assert y_draw.requires_grad
            err = y_draw - Variable(logits.new([[0, 0.5, 0.3]]))
            loss = (err * err).sum()
            loss.backward()
            if logits_var.grad.data.std() < 0.01 or logits_var.grad.data.std() > 1.0:
                exceed_limits += 1
            y_draws[draw] = y_draw.data
            _, pred = y_draw.max(1)
            preds[draw] = pred.data[0]
        assert exceed_limits / num_draws < 0.05
        # check it's approximately one-hot
        num_ones = (y_draws == 1).int().sum()
        num_zeros = (y_draws == 0).int().sum()
        assert num_ones + num_zeros == num_draws * K
        assert num_ones == num_draws
        # check output classes approx in line with logits
        num_class_one = (preds == 1).int().sum()
        assert num_class_one < num_draws
        assert num_class_one > num_draws / 3

    def test_gumbel_softmax_st(self):
        self._test_gumbel_softmax_st(False)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    @repeat_test_for_types(ALL_TENSORTYPES)
    def test_gumbel_softmax_st_cuda(self, dtype=torch.FloatTensor):
        self._test_gumbel_softmax_st(True, dtype=dtype)

    def _test_EmbeddingBag(self, cuda, mode, sparse, dtype=torch.DoubleTensor):
        # check a known test example
        es = nn.EmbeddingBag(5, 2, mode=mode, sparse=sparse)
        es.weight.data.copy_(torch.arange(1, 11).resize_as_(es.weight.data))
        es.type(dtype)
        input = Variable(torch.LongTensor([3, 1, 1, 1, 4, 0]))
        offsets = Variable(torch.LongTensor([0, 3]))
        grad_output = torch.arange(1, 5).view(2, 2).type(dtype)

        if mode == 'sum':
            expected_output = torch.Tensor(
                [[13, 16],
                 [13, 16]])
            expected_grad_weight = torch.Tensor(
                [[3, 4],
                 [5, 8],
                 [0, 0],
                 [1, 2],
                 [3, 4]])
        else:
            expected_output = torch.Tensor(
                [[13. / 3, 16. / 3],
                 [13. / 3, 16. / 3]])
            expected_grad_weight = torch.Tensor(
                [[3. / 3, 4. / 3],
                 [1. / 3 + 1. / 3 + 3. / 3, 2. / 3 + 2. / 3 + 4. / 3],
                 [0., 0.],
                 [1. / 3, 2. / 3],
                 [3. / 3, 4. / 3]])

        if cuda:
            es = es.cuda()
            input = input.cuda()
            offsets = offsets.cuda()
            grad_output = grad_output.cuda()
            expected_output = expected_output.cuda()
            expected_grad_weight = expected_grad_weight.cuda()

        output = es(input, offsets)
        output.backward(grad_output)

        es_weight_grad = es.weight.grad.data
        if sparse:
            es_weight_grad = es.weight.grad.data.to_dense()
        self.assertEqual(output.data, expected_output)
        self.assertEqual(es_weight_grad, expected_grad_weight, type2prec[dtype.__name__])

        # check same example except as 2D (2 x 3)
        input = Variable(input.data.view(2, -1))
        es.zero_grad()
        output = es(input)
        output.backward(grad_output)

        es_weight_grad = es.weight.grad.data
        if sparse:
            es_weight_grad = es.weight.grad.data.to_dense()
        self.assertEqual(output.data, expected_output)
        self.assertEqual(es_weight_grad, expected_grad_weight, type2prec[dtype.__name__])

        # now compare EmbeddingBag vs Embedding + Sum/Mean, for constant bag length
        def _test_vs_Embedding(N, D, B, L):
            es = nn.EmbeddingBag(N, D, mode=mode, sparse=sparse).type(dtype)
            e = nn.Embedding(N, D).type(dtype)
            e.weight.data.copy_(es.weight.data)
            input = Variable(torch.rand(B, L).mul(N).long())
            offsets = Variable(torch.arange(0, B).mul(L).long())
            grad_output = torch.rand(B, D).type(dtype)

            if cuda:
                es = es.cuda()
                e = e.cuda()
                input = input.cuda()
                offsets = offsets.cuda()
                grad_output = grad_output.cuda()

            output = es(input.view(-1), offsets)
            if mode == 'sum':
                ref_output = e(input).sum(1)
            else:
                ref_output = e(input).mean(1)

            self.assertEqual(output, ref_output, type2prec[dtype.__name__])

            output.backward(grad_output)
            ref_output.backward(grad_output)
            es_weight_grad = es.weight.grad.data
            if sparse:
                es_weight_grad = es.weight.grad.data.to_dense()
            self.assertEqual(es_weight_grad, e.weight.grad, type2prec[dtype.__name__])

        N, D, B, L = random.randint(1, 100), random.randint(1, 100), random.randint(1, 50), random.randint(1, 50)
        _test_vs_Embedding(N, D, B, L)
        for p in itertools.product([1, 2], repeat=4):
            _test_vs_Embedding(*p)

        # check that giving illegal input combos raises error
        es = nn.EmbeddingBag(10, 20, mode=mode, sparse=sparse)
        input = Variable(torch.ones(3, 4))
        offset = Variable(torch.arange(0, 3))
        self.assertRaises(ValueError, lambda: es(input, offset))
        self.assertRaises(ValueError, lambda: es(input.view(-1)))
        offset[0] = 1
        self.assertRaises(ValueError, lambda: es(input.view(-1), offset))
        offset[0] = 0
        offset[-1] = 100
        self.assertRaises(ValueError, lambda: es(input.view(-1), offset))

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_pool3d_size_one_feature_dim(self):
        # Tests crazy strides for feature dim of size 1
        x = Variable(torch.randn(7, 1, 5, 3, 2).cuda())
        strange_strides = [30, 1234, 6, 2, 1]
        y = x.as_strided(x.size(), strange_strides)
        x = x.cpu().as_strided(x.size(), strange_strides)

        to_test = {
            'max_pool3d': lambda t: F.max_pool3d(t, (5, 1, 1), stride=(5, 1, 1)),
            'avg_pool3d': lambda t: F.avg_pool3d(t, (5, 1, 1), stride=(5, 1, 1)),
        }

        for test, fn in to_test.items():
            # Should not crash
            out_y = fn(y)
            out_x = fn(x)
            self.assertEqual(out_y, out_x.cuda(), test)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_AvgPool3d_backward_after_cat_dim1_cuda(self):
        # x has to have batch_size 1 to test contiguous checks
        x = Variable(torch.randn(1, 3, 4, 4, 4).cuda(), requires_grad=True)
        y = F.avg_pool3d(x, kernel_size=3, padding=1, stride=2)

        grad = torch.randn(y.size()).cuda()
        # increase the stride in dimension 0. the tensor is still contiguous because size[0] is 1
        stride = list(grad.stride())
        stride[0] = stride[0] * 2
        grad.set_(grad.storage(), 0, grad.size(), stride)
        assert grad.is_contiguous()

        y.backward(grad)

    @unittest.skipIf(not TEST_CUDNN, "needs cudnn")
    def test_contig_wrong_stride_cudnn(self):
        # x has to have batch_size 1 to test contiguous checks
        x = torch.randn(1, 16, 5, 5).cuda()
        stride = list(x.stride())
        stride[0] = 20
        # change the stride in dimension 0. the tensor is still contiguous because size[0] is 1
        x.set_(x.storage(), 0, x.size(), stride)
        self.assertTrue(x.is_contiguous())
        F.conv_transpose2d(Variable(x), Variable(torch.randn(16, 1, 1, 1)).cuda())
        F.conv2d(Variable(x), Variable(torch.randn(1, 16, 1, 1)).cuda())

    def test_embedding_bag(self):
        self._test_EmbeddingBag(False, 'sum', False)
        self._test_EmbeddingBag(False, 'mean', False)
        self._test_EmbeddingBag(False, 'sum', True)
        self._test_EmbeddingBag(False, 'mean', True)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    @repeat_test_for_types(ALL_TENSORTYPES)
    def test_embedding_bag_cuda(self, dtype=torch.FloatTensor):
        self._test_EmbeddingBag(True, 'sum', False, dtype)
        self._test_EmbeddingBag(True, 'mean', False, dtype)
        if dtype != torch.HalfTensor:
            # torch.cuda.sparse.HalfTensor is not enabled.
            self._test_EmbeddingBag(True, 'sum', True, dtype)
            self._test_EmbeddingBag(True, 'mean', True, dtype)

    def test_fractional_max_pool2d(self):
        x = Variable(torch.randn(1, 2, 7, 7), requires_grad=True)
        samples = x.new(1, 2, 2).uniform_()

        def func(x):
            return F.fractional_max_pool2d(
                x, (2, 2), output_size=(3, 3), _random_samples=samples)

        self.assertEqual(func(x).shape, (1, 2, 3, 3))
        gradcheck(func, [x])
        gradgradcheck(func, [x])

        x = Variable(torch.randn(2, 7, 7), requires_grad=True)
        samples = x.new(2, 2).uniform_()
        self.assertEqual(func(x).shape, (2, 3, 3))
        gradcheck(func, [x])
        gradgradcheck(func, [x])

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

    def _test_InstanceNorm_general(self, cls, input, type):
        # default case track_running_stats=False
        b, c = input.size(0), input.size(1)
        input_var = Variable(input.type(type), requires_grad=True)

        IN = cls(c, eps=0).type(type)

        output = IN(input_var)
        out_reshaped = output.view(b * c, -1)

        mean = out_reshaped.mean(1)
        var = out_reshaped.var(1, unbiased=False)

        self.assertAlmostEqual(torch.abs(mean.data).mean(), 0, delta=1e-5)
        self.assertAlmostEqual(torch.abs(var.data).mean(), 1, delta=1e-5)

        # check that eval mode doesn't change behavior
        grad_out = output.data.clone().normal_()
        res1 = output.data.clone()
        output.backward(grad_out)
        grad1 = input_var.grad.data.clone()

        IN.eval()
        output = IN(input_var)
        input_var.grad = None
        output.backward(grad_out)
        res2 = output.data
        grad2 = input_var.grad.data
        self.assertEqual(res1, res2)
        self.assertEqual(grad1, grad2)

        # If track_running_stats=True and momentum=1, running_mean/var should be
        # equal to mean/var of the input (with unbias correction)
        IN = cls(c, momentum=1, eps=0, track_running_stats=True).type(type)

        output = IN(input_var.type(type))

        input_reshaped = input_var.transpose(1, 0).contiguous().view(c, -1)
        mean = input_reshaped.mean(1)

        input_reshaped = input_var.transpose(1, 0).contiguous().view(c, b, -1)
        var = input_reshaped.var(2, unbiased=True)[:, :]

        self.assertAlmostEqual(torch.abs(mean.data - IN.running_mean).mean(), 0, delta=1e-5)
        self.assertAlmostEqual(torch.abs(var.data.mean(1) - IN.running_var).mean(), 0, delta=1e-5)

        # in eval mode, adding X * std to a channel in input should make the
        # corresponding channel in output have mean X
        IN.eval()
        delta = (IN.running_var.sqrt() * torch.arange(c).type(type)).view(-1, *[1 for _ in range(2, input.dim())])
        output = IN(input_var + Variable(delta))
        self.assertEqual(output.transpose(0, 1).contiguous().view(c, -1).mean(1), torch.arange(c))

    def _test_InstanceNorm_cuda_half(self, cls, input):
        # THNN
        input = Variable(input.cuda().half().random_(1, 10), requires_grad=True)
        m = cls(input.size(1), affine=True, track_running_stats=True).cuda().half()
        thnn_output = m(input)
        thnn_output.sum().backward()
        thnn_input_grad = input.grad.data.clone()
        self.assertEqual(thnn_output.type(), input.type())
        # cuDNN
        if TEST_CUDNN:
            input.grad = None
            m = m.float()
            cudnn_output = m(input)
            cudnn_output.sum().backward()
            cudnn_input_grad = input.grad.data.clone()
            self.assertEqual(cudnn_output.type(), input.type())
            self.assertAlmostEqual(cudnn_output, thnn_output, delta=1e-4)
            self.assertAlmostEqual(cudnn_input_grad, thnn_input_grad, delta=1e-3)

    def test_InstanceNorm1d_general(self):
        b = random.randint(3, 5)
        c = random.randint(3, 5)
        d = random.randint(8, 10)

        input = torch.Tensor(b, c, d).uniform_()
        self._test_InstanceNorm_general(nn.InstanceNorm1d, input, torch.FloatTensor)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_InstanceNorm1d_general_cuda(self):
        b = random.randint(3, 5)
        c = random.randint(3, 5)
        d = random.randint(8, 10)

        input = torch.Tensor(b, c, d).uniform_()
        self._test_InstanceNorm_general(nn.InstanceNorm1d, input, torch.cuda.FloatTensor)
        self._test_InstanceNorm_cuda_half(nn.InstanceNorm1d, input)

    def test_InstanceNorm2d_general(self):
        b = random.randint(3, 5)
        c = random.randint(3, 5)
        w = random.randint(3, 6)
        h = random.randint(6, 8)

        input = torch.Tensor(b, c, h, w).uniform_()
        self._test_InstanceNorm_general(nn.InstanceNorm2d, input, torch.FloatTensor)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_InstanceNorm2d_general_cuda(self):
        b = random.randint(3, 5)
        c = random.randint(3, 5)
        w = random.randint(3, 6)
        h = random.randint(6, 8)

        input = torch.Tensor(b, c, h, w).uniform_()
        self._test_InstanceNorm_general(nn.InstanceNorm2d, input, torch.cuda.FloatTensor)
        self._test_InstanceNorm_cuda_half(nn.InstanceNorm2d, input)

    def test_InstanceNorm3d_general(self):
        b = random.randint(3, 5)
        c = random.randint(3, 5)
        w = random.randint(2, 5)
        h = random.randint(2, 5)
        d = random.randint(2, 5)

        input = torch.Tensor(b, c, h, w, d).uniform_()
        self._test_InstanceNorm_general(nn.InstanceNorm3d, input, torch.FloatTensor)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_InstanceNorm3d_general_cuda(self):
        b = random.randint(3, 5)
        c = random.randint(2, 5)
        w = random.randint(2, 5)
        h = random.randint(2, 5)
        d = random.randint(2, 5)

        input = torch.Tensor(b, c, h, w, d).uniform_()
        self._test_InstanceNorm_general(nn.InstanceNorm3d, input, torch.cuda.FloatTensor)
        self._test_InstanceNorm_cuda_half(nn.InstanceNorm3d, input)

    def _test_LayerNorm_general(self, type):
        for i in range(2, 6):
            shape = torch.LongTensor(i).random_(3, 6).tolist()
            x = type(*shape).uniform_(0, 10)
            normalized_ndim = random.randint(1, i - 1)  # inclusive
            normalized_shape = shape[-normalized_ndim:]
            unnormalized_shape = shape[:-normalized_ndim]

            # test that LN normalizes to mean 0 and stddev 1
            ln = nn.LayerNorm(normalized_shape, eps=0).type(type)
            ln.weight.data.fill_(1)
            ln.bias.data.fill_(0)
            output = ln(x)
            out_reshaped = output.view(*(unnormalized_shape + [-1]))
            mean = out_reshaped.mean(-1)
            var = out_reshaped.var(-1, unbiased=False)
            self.assertAlmostEqual(torch.abs(mean.data).mean(), 0, delta=1e-5)
            self.assertAlmostEqual(torch.abs(var.data).mean(), 1, delta=1e-5)

            # test that LN applies weight and bias correctly
            scale, bias = torch.FloatTensor(2).uniform_(0.2, 2).tolist()
            ln.weight.data.fill_(scale)
            ln.bias.data.fill_(bias)
            output = ln(x)
            out_reshaped = output.view(*(unnormalized_shape + [-1]))
            mean = out_reshaped.mean(-1)
            var = out_reshaped.var(-1, unbiased=False)
            self.assertAlmostEqual(torch.abs(mean.data).mean(), bias, delta=1e-5)
            self.assertAlmostEqual(torch.abs(var.data).mean(), scale ** 2, delta=1e-5)

        bad_norm_shape_input_shape = {
            (): (),
            (2, 3): (3,),
            (2,): (1, 2, 3),
            (10,): (2, 3),
            10: (2, 3),
        }
        for norm_shape, input_shape in bad_norm_shape_input_shape.items():
            ln = nn.LayerNorm(norm_shape)
            input = type(*input_shape).uniform_(0, 10)
            self.assertRaises(RuntimeError, lambda: ln(input))

    def _test_LayerNorm_cuda_half(self):
        input = torch.zeros(2, 3, 3, 2, requires_grad=True).cuda().half().random_(1, 10)
        m = nn.LayerNorm([3, 2]).cuda().half()
        output = m(input)
        output.sum().backward()
        self.assertEqual(output.type(), input.type())

    def test_LayerNorm_general(self):
        self._test_LayerNorm_general(torch.FloatTensor)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_LayerNorm_general_cuda(self):
        self._test_LayerNorm_general(torch.cuda.FloatTensor)
        self._test_LayerNorm_cuda_half()

    def _test_GroupNorm_general(self, type):
        good_shape_g = {
            (1, 2, 3, 4): 2,
            (2, 3, 10): 3,
            (3, 1, 1, 1, 2): 1,
            (2, 6, 4, 2, 2): 3,
        }
        for shape, g in good_shape_g.items():
            x = type(*shape).uniform_(0, 10)
            b = shape[0]
            c = shape[1]

            # test that GN normalizes to mean 0 and stddev 1
            gn = nn.GroupNorm(g, c, eps=0).type(type)
            gn.weight.data.fill_(1)
            gn.bias.data.fill_(0)
            output = gn(x)
            out_reshaped = output.view(b, g, -1)
            mean = out_reshaped.mean(-1)
            var = out_reshaped.var(-1, unbiased=False)
            self.assertAlmostEqual(torch.abs(mean).mean(), 0, delta=1e-5)
            self.assertAlmostEqual(torch.abs(var).mean(), 1, delta=1e-5)

            # test that GN applies weight and bias correctly
            scale = type(c).uniform_(0.2, 2)
            bias = type(c).uniform_(0.2, 2)
            gn.weight.data.copy_(scale)
            gn.bias.data.copy_(bias)
            output = gn(x)
            out_reshaped = output.view(b, c, -1)
            out_normed = (out_reshaped - bias.view(c, 1)) / scale.view(c, 1)
            out_normed_reshaped = out_normed.view(b, g, -1)
            mean = out_normed_reshaped.mean(-1)
            var = out_normed_reshaped.var(-1, unbiased=False)
            self.assertAlmostEqual(torch.abs(mean).mean(), 0, delta=1e-5)
            self.assertAlmostEqual(torch.abs(var).mean(), 1, delta=1e-5)

        bad_shape_g = {
            (1, 2, 3, 4): 3,
            (2, 3, 10): 2,
            (3, 1, 1, 1, 2): 10,
            (2, 6, 4, 2, 2): 4,
        }
        for shape, g in bad_shape_g.items():
            gn = nn.GroupNorm(g, shape[1])
            input = type(*shape).uniform_(0, 10)
            self.assertRaises(RuntimeError, lambda: gn(input))

    def _test_GroupNorm_cuda_half(self):
        input = torch.zeros(2, 4, 3, 2, requires_grad=True).cuda().half().random_(1, 10)
        m = nn.GroupNorm(2, 4).cuda().half()
        output = m(input)
        output.sum().backward()
        self.assertEqual(output.type(), input.type())

    def test_GroupNorm_general(self):
        self._test_GroupNorm_general(torch.FloatTensor)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_GroupNorm_general_cuda(self):
        self._test_GroupNorm_general(torch.cuda.FloatTensor)
        self._test_GroupNorm_cuda_half()

    def test_pad(self):
        inputs = Variable(torch.randn(1, 3, 4, 4), requires_grad=True)
        _assertGradAndGradgradChecks(self, lambda x: F.pad(x, (1, 1, 1, 1)), (inputs,))
        _assertGradAndGradgradChecks(self, lambda x: F.pad(x, (-1, 1, -2, 1)), (inputs,))
        _assertGradAndGradgradChecks(self, lambda x: F.pad(x, (-1, 1, -2, 1), value=2), (inputs,))
        self.assertTrue(gradcheck(lambda x: F.pad(x, (-1, 1, -2, 1), mode='replicate'), (inputs,)))
        self.assertTrue(gradcheck(lambda x: F.pad(x, (-1, 1, -2, 1), mode='reflect'), (inputs,)))

        inputs = Variable(torch.randn(1, 2, 3, 4, 4), requires_grad=True)
        self.assertTrue(gradcheck(lambda x: F.pad(x, (1, 1, 1, 1, 1, 1), mode='replicate'), (inputs,)))

        # assert that relfection padding errors when pad >= input size
        expected_err_msg = r"Padding size should be less than the corresponding input dimension"
        self.assertRaisesRegex(RuntimeError, expected_err_msg,
                               lambda: F.pad(torch.randn(1, 1, 2, 3), (1, 1, 3, 0), mode='reflect'))
        self.assertRaisesRegex(RuntimeError, expected_err_msg,
                               lambda: F.pad(torch.randn(1, 1, 2), (2, 1), mode='reflect'))

    def test_pad_scalar_error(self):
        inputs = torch.tensor(0, requires_grad=True)
        self.assertRaises(AssertionError, lambda: F.pad(inputs, (1, 1)))
        self.assertRaises(AssertionError, lambda: F.pad(inputs, (1,)))

    def test_normalize(self):
        inputs = Variable(torch.randn(1, 3, 4, 4), requires_grad=True)
        self.assertTrue(gradcheck(lambda x: F.normalize(x, p=1, dim=-1), (inputs,)))
        self.assertTrue(gradcheck(lambda x: F.normalize(x, p=2, dim=-2), (inputs,)))

        inputs = torch.randn((), requires_grad=True)
        self.assertTrue(gradcheck(lambda x: F.normalize(x, p=1, dim=-1), (inputs,)))

    def _test_maxpool_indices(self, num_dim, adaptive=False, dtype=torch.FloatTensor):
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

        if adaptive:
            cls_name = 'AdaptiveMaxPool{}d'.format(num_dim)
        else:
            cls_name = 'MaxPool{}d'.format(num_dim)
        module_cls = getattr(nn, cls_name)
        module = module_cls(2, return_indices=True).type(dtype)
        numel = 4 ** (num_dim + 1)
        input = torch.arange(1, numel + 1).view(2, 2, *repeat(4, num_dim)).type(dtype)
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
        grad_output = torch.ones(output.size()).type(dtype)
        output.backward(grad_output, retain_graph=True)
        expected_grad = expected_grad(num_dim)
        self.assertEqual(input_var.grad.data, expected_grad.view_as(input))

        # Make sure backward after changing indices will result in an error
        indices.add_(1)
        self.assertRaises(RuntimeError, lambda: output.backward(grad_output))

    def test_Conv2d_naive_groups(self):
        self._test_Conv2d_naive_groups()

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    @repeat_test_for_types(ALL_TENSORTYPES)
    def test_Conv2d_naive_groups_cuda(self, dtype=torch.FloatTensor):
        dtype = getattr(torch.cuda, dtype.__name__)
        self._test_Conv2d_naive_groups(dtype)

    def test_batchnorm_eval(self):
        self._test_batchnorm_eval()

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_batchnorm_eval_cuda(self, dtype=torch.FloatTensor):
        dtype = getattr(torch.cuda, dtype.__name__)
        self._test_batchnorm_eval(dtype)

    def test_MaxPool1d_indices(self):
        self._test_maxpool_indices(1)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    @repeat_test_for_types(ALL_TENSORTYPES)
    def test_MaxPool1d_indices_cuda(self, dtype=torch.FloatTensor):
        dtype = getattr(torch.cuda, dtype.__name__)
        self._test_maxpool_indices(1, dtype=dtype)

    def test_MaxPool2d_indices(self):
        self._test_maxpool_indices(2)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    @repeat_test_for_types(ALL_TENSORTYPES)
    def test_MaxPool2d_indices_cuda(self, dtype=torch.FloatTensor):
        dtype = getattr(torch.cuda, dtype.__name__)
        self._test_maxpool_indices(2, dtype=dtype)

    def test_MaxPool3d_indices(self):
        self._test_maxpool_indices(3)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    @repeat_test_for_types(ALL_TENSORTYPES)
    def test_MaxPool3d_indices_cuda(self, dtype=torch.FloatTensor):
        dtype = getattr(torch.cuda, dtype.__name__)
        self._test_maxpool_indices(3, dtype=dtype)

    def test_AdaptiveMaxPool1d_indices(self):
        self._test_maxpool_indices(1, adaptive=True)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    @repeat_test_for_types(ALL_TENSORTYPES)
    def test_AdaptiveMaxPool1d_indices_cuda(self, dtype=torch.FloatTensor):
        dtype = getattr(torch.cuda, dtype.__name__)
        self._test_maxpool_indices(1, adaptive=True, dtype=dtype)

    def test_AdaptiveMaxPool2d_indices(self):
        self._test_maxpool_indices(2, adaptive=True)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    @repeat_test_for_types(ALL_TENSORTYPES)
    def test_AdaptiveMaxPool2d_indices_cuda(self, dtype=torch.FloatTensor):
        dtype = getattr(torch.cuda, dtype.__name__)
        self._test_maxpool_indices(2, adaptive=True, dtype=dtype)

    def test_AdaptiveMaxPool3d_indices(self):
        self._test_maxpool_indices(3, adaptive=True)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    @repeat_test_for_types(ALL_TENSORTYPES)
    def test_AdaptiveMaxPool3d_indices_cuda(self, dtype=torch.FloatTensor):
        dtype = getattr(torch.cuda, dtype.__name__)
        self._test_maxpool_indices(3, adaptive=True, dtype=dtype)

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
        _assertGradAndGradgradChecks(self, lambda y: dp.scatter(y, (0, 1)), (x,))

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
        _assertGradAndGradgradChecks(self, lambda x, y: dp.gather((x, y), output_device), inputs)

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_gather_cpu(self):
        self._test_gather(-1)

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_gather_gpu(self):
        self._test_gather(0)

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_gather_different_len_dicts(self):
        inputs = (
            {'a': Variable(torch.randn(1, 2).cuda(0), requires_grad=True)},
            {
                'b': Variable(torch.randn(1, 2).cuda(1), requires_grad=True),
                'a': Variable(torch.randn(1, 2).cuda(1), requires_grad=True)
            }
        )
        with self.assertRaises(ValueError):
            _ = dp.gather(inputs, target_device=0)

    def _test_broadcast_double_backwards(self, *tensors):
        variables = tuple(Variable(t, requires_grad=True) for t in tensors)
        _assertGradAndGradgradChecks(self, lambda *i: Broadcast.apply((0, 1), *i), variables)

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_broadcast_double_backwards_gpu(self):
        self._test_broadcast_double_backwards(torch.randn(4, 4).cuda(),
                                              torch.randn(4, 4).cuda(),
                                              torch.randn(4, 4).cuda())

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_broadcast_not_requiring_grad(self):
        variables = [
            Variable(torch.randn(1, 2).cuda(), requires_grad=True),
            Variable(torch.randn(1, 2).cuda(), requires_grad=False),
            Variable(torch.randn(1, 2).cuda(), requires_grad=False),
            Variable(torch.randn(1, 2).cuda(), requires_grad=True),
            Variable(torch.randn(1, 2).cuda(), requires_grad=True),
        ]
        broadcasted_variables = Broadcast.apply((0, 1), *variables)
        for output_idx, broadcasted_var in enumerate(broadcasted_variables):
            input_var = variables[output_idx % len(variables)]
            self.assertEqual(input_var.requires_grad, broadcasted_var.requires_grad)

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_broadcast_no_grad(self):
        x = torch.randn(1, 2, dtype=torch.float32, requires_grad=True, device='cuda')
        with torch.no_grad():
            broadcasted = Broadcast.apply((0, 1), x)
        self.assertTrue(x.requires_grad)
        for output in broadcasted:
            self.assertFalse(output.requires_grad)

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

        float1 = torch.randn(1).item()

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

    @unittest.skipIf(not TEST_MULTIGPU or not PY3, "multi-GPU not supported")
    def test_data_parallel_model_no_refcycles(self):
        # Python 2.7 will create reference cycles with the following
        # Module on multiple GPUs, but Python 3 shouldn't unless
        # there are refcycles on the PyTorch side (or the defined module)
        import gc

        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.linear = nn.Linear(1, 1)

            def forward(self, x):
                return self.linear(x)

        gc.collect()
        model = nn.DataParallel(Model().cuda())
        data = Variable(torch.randn(1).cuda())
        model(data)

        refcycles = gc.collect()
        self.assertEqual(refcycles, 0)

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_data_parallel_no_grad(self):
        test = self

        class Layer(nn.Module):
            def forward(self, x):
                test.assertFalse(torch.is_grad_enabled())
                return x

        l = Layer()
        i = Variable(torch.randn(20, 10).float().cuda())
        with torch.no_grad():
            dp.data_parallel(l, i, (0, 1))
        self.assertRaises(AssertionError, lambda: dp.data_parallel(l, i, (0, 1)))

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_data_parallel(self):
        l = nn.Linear(10, 5).float().cuda()
        i = Variable(torch.randn(20, 10).float().cuda(1))
        l.cuda(1)
        expected_out = l(i)
        loss = expected_out.sum()
        loss.backward()
        expected_grads = []
        for param in l.parameters():
            expected_grads.append(param.grad.clone())
        dev_ids_list = [(0, 1), (1, 0)]
        for dev_id in dev_ids_list:
            with torch.cuda.device(dev_id[0]):
                l.cuda()
                l.zero_grad()
                out = dp.data_parallel(l, i, dev_id)
                loss = out.sum()
                loss.backward()
                self.assertEqual(out.get_device(), dev_id[0])
                self.assertEqual(out.data, expected_out.data)
                for expected, param in zip(expected_grads, l.parameters()):
                    self.assertEqual(param.grad.data, expected.data)

        # Check for None device_ids
        l = l.cuda()
        out = dp.data_parallel(l, i)

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_data_parallel_sparse(self):
        l = nn.Embedding(10, 5, sparse=True).cuda(1)
        i = Variable(torch.LongTensor(20, 5).random_(0, 10).cuda(1))
        expected_out = l(i)
        loss = expected_out.sum()
        loss.backward()
        expected_grads = []
        for param in l.parameters():
            expected_grads.append(param.grad.clone())
        dev_ids_list = [(0, 1), (1, 0)]
        for dev_id in dev_ids_list:
            with torch.cuda.device(dev_id[0]):
                l.cuda()
                l.zero_grad()
                out = dp.data_parallel(l, i, dev_id)
                loss = out.sum()
                loss.backward()
                self.assertEqual(out.get_device(), dev_id[0])
                self.assertEqual(out.data, expected_out.data)
                for expected, param in zip(expected_grads, l.parameters()):
                    self.assertEqual(param.grad.data, expected.data)

        # Check for None device_ids
        l = l.cuda()
        out = dp.data_parallel(l, i)

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_data_parallel_nested_output(self):
        def fn(input):
            return [
                input, (input.sin(), input.cos(), [input.add(1)]), input,
                {'a': input, 'b': [input.sin()]}
            ]

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
        self.assertIsInstance(output[3], dict)
        self.assertEqual(len(output[3]), 2)
        self.assertIn('a', output[3])
        self.assertIn('b', output[3])
        self.assertIsInstance(output[3]['a'], Variable)
        self.assertIsInstance(output[3]['b'], list)
        self.assertIsInstance(output[3]['b'][0], Variable)

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
    @repeat_test_for_types(ALL_TENSORTYPES)
    def test_data_parallel_module(self, dtype=torch.FloatTensor):
        l = nn.Linear(10, 5).type(dtype).cuda()
        i = Variable(torch.randn(20, 10).type(dtype).cuda())
        expected_out = l(i).data
        net = nn.DataParallel(l)
        out = net(i)
        self.assertEqual(out.get_device(), 0)
        self.assertEqual(out.data, expected_out)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    @repeat_test_for_types(ALL_TENSORTYPES)
    def test_data_parallel_module_kwargs_only(self, dtype=torch.FloatTensor):
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.l = l

            def forward(self, input):
                return self.l(input)

        l = nn.Linear(10, 5).type(dtype).cuda()
        i = Variable(torch.randn(20, 10).type(dtype).cuda())
        expected_out = l(i).data
        n = nn.DataParallel(Net())
        out = n(input=i)
        self.assertEqual(out.get_device(), 0)
        self.assertEqual(out.data, expected_out)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    @repeat_test_for_types(ALL_TENSORTYPES)
    def test_data_parallel_module_kwargs_only_empty_list(self, dtype=torch.FloatTensor):
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.l = l

            def forward(self, input):
                return self.l(input['data'])

        l = nn.Linear(10, 5).type(dtype).cuda()
        i = Variable(torch.randn(20, 10).type(dtype).cuda())
        expected_out = l(i).data
        n = nn.DataParallel(Net())
        out = n(input={'data': i, 'unused': []})
        self.assertEqual(out.get_device(), 0)
        self.assertEqual(out.data, expected_out)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    @repeat_test_for_types(ALL_TENSORTYPES)
    def test_data_parallel_module_kwargs_only_empty_dict(self, dtype=torch.FloatTensor):
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.l = l

            def forward(self, input):
                return self.l(input['data'])

        l = nn.Linear(10, 5).type(dtype).cuda()
        i = Variable(torch.randn(20, 10).type(dtype).cuda())
        expected_out = l(i).data
        n = nn.DataParallel(Net())
        out = n(input={'data': i, 'unused': {}})
        self.assertEqual(out.get_device(), 0)
        self.assertEqual(out.data, expected_out)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    @repeat_test_for_types(ALL_TENSORTYPES)
    def test_data_parallel_module_kwargs_only_empty_tuple(self, dtype=torch.FloatTensor):
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.l = l

            def forward(self, input):
                return self.l(input['data'])

        l = nn.Linear(10, 5).type(dtype).cuda()
        i = Variable(torch.randn(20, 10).type(dtype).cuda())
        expected_out = l(i).data
        n = nn.DataParallel(Net())
        out = n(input={'data': i, 'unused': ()})
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
            self.assertEqual(v.data_ptr(), param.data_ptr())

        l = nn.Linear(5, 5)
        state_dict = l.state_dict()
        self.assertEqual(len(state_dict), 2)
        self.assertEqual(state_dict['weight'].data_ptr(), l.weight.data_ptr())
        self.assertEqual(state_dict['bias'].data_ptr(), l.bias.data_ptr())

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

        state_dict = net.state_dict()
        state_dict.update({'bn.running_mean': torch.rand(14, 4)})  # wrong size
        self.assertRaises(RuntimeError, lambda: net.load_state_dict(state_dict))

        state_dict = net.state_dict()
        old_state_dict = deepcopy(state_dict)
        state_dict = {
            'linear1.weight': torch.ones(5, 5),
            'block.conv1.bias': torch.arange(1, 4),
            'bn.running_mean': torch.randn(2),
            'nonexistent_key': torch.rand(3)
        }
        net.load_state_dict(state_dict, strict=False)
        self.assertEqual(net.linear1.weight.data, state_dict['linear1.weight'])
        self.assertEqual(net.block.conv1.bias.data, state_dict['block.conv1.bias'])
        self.assertEqual(net.bn.running_mean, state_dict['bn.running_mean'])
        new_state_dict = net.state_dict()
        del old_state_dict['linear1.weight']
        del old_state_dict['block.conv1.bias']
        del old_state_dict['bn.running_mean']
        for k, v, in old_state_dict.items():
            self.assertTrue(v.equal(new_state_dict[k]))

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

        with torch.backends.cudnn.flags(enabled=False):
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

        with torch.backends.cudnn.flags(enabled=True):
            # inconsistent types should raise an exception
            self.assertRaises(RuntimeError, lambda: nn.functional.conv2d(inputs, weights))
            self.assertRaises(RuntimeError, lambda: nn.functional.conv2d(inputs, weights.float(), bias))

            # but it should work with the same type
            nn.functional.conv2d(inputs.float(), weights.float(), bias.float())

    @unittest.skipIf(not TEST_CUDA, 'CUDA not available')
    @unittest.skipIf(not TEST_CUDNN, 'CUDNN not available')
    @repeat_test_for_types(ALL_TENSORTYPES)
    def test_Conv2d_deterministic_cudnn(self, dtype=torch.FloatTensor):
        dtype = getattr(torch.cuda, dtype.__name__)
        inputs = Variable(torch.randn(2, 3, 5, 5).type(dtype), requires_grad=True)
        with cudnn.flags(enabled=True, benchmark=True, deterministic=True):
            conv1 = torch.nn.Conv2d(3, 3, 3).type(dtype)
            conv2 = torch.nn.Conv2d(3, 3, 3).type(dtype)
            conv2.bias.data.copy_(conv1.bias.data)
            conv2.weight.data.copy_(conv1.weight.data)
            out1 = conv1(inputs)
            out2 = conv2(inputs)
            self.assertEqual(out1, out2, prec=0.0)
            y = torch.randn(out1.size()).type(dtype)
            out1.backward(y)
            out2.backward(y)
            self.assertEqual(conv1.bias.grad.data, conv2.bias.grad.data, prec=0.0)
            self.assertEqual(conv1.weight.grad.data, conv2.weight.grad.data, prec=0.0)

    def test_Conv2d_missing_argument(self):
        c = nn.Conv2d(3, 3, 3)
        self.assertRaises(TypeError, lambda: c(None))

    def test_Conv2d_backward_twice(self):
        input = Variable(torch.randn(2, 3, 5, 5))
        c = nn.Conv2d(3, 3, 3)
        o1 = c(input)
        o1.sum().backward()
        self.assertRaisesRegex(RuntimeError, 'Specify retain_graph=True',
                               lambda: o1.sum().backward())

    @unittest.skipIf(not TEST_CUDA, 'CUDA not available')
    @repeat_test_for_types(ALL_TENSORTYPES)
    def test_Conv2d_large_workspace(self, dtype=torch.FloatTensor):
        # These sizes require huge cuDNN workspaces. Make sure we choose a
        # reasonable algorithm that does not run out of memory
        sizes = [
            (1, 256, 109, 175),
            (1, 256, 80, 128),
            (1, 256, 120, 192),
        ]
        dtype = getattr(torch.cuda, dtype.__name__)

        def run_test(benchmark):
            with torch.backends.cudnn.flags(benchmark=benchmark):
                conv = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1).type(dtype)
                for size in sizes:
                    x = torch.randn(size).type(dtype)
                    out = conv(Variable(x, requires_grad=True))
                    out.backward(torch.ones(out.size()).type(dtype))

        run_test(benchmark=False)
        run_test(benchmark=True)

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
                self.assertRaises(RuntimeError, lambda: module(input))

    def test_conv_shapecheck(self):
        def test(should_raise, module, input_size):
            input = Variable(torch.Tensor(3, *input_size))
            if should_raise:
                self.assertRaises(RuntimeError, lambda: module(input))
            else:
                # just run it to ensure no exception raised.
                module(input)

        # Conv1d
        test(True, nn.Conv1d(1, 1, 3), (1, 2))
        test(True, nn.Conv1d(1, 1, 3, stride=2), (1, 2))
        test(False, nn.Conv1d(1, 1, 2), (1, 2))
        test(False, nn.Conv1d(1, 1, 2, stride=2), (1, 2))
        test(False, nn.Conv1d(1, 1, 3, stride=2, padding=1), (1, 2))

        # Conv2d
        test(True, nn.Conv2d(1, 1, (3, 3)), (1, 2, 2))
        test(False, nn.Conv2d(1, 1, (3, 3)), (1, 3, 3))
        test(False, nn.Conv2d(1, 1, (3, 3), padding=1), (1, 2, 2))

        # Conv3D
        test(True, nn.Conv3d(1, 1, (3, 3, 3)), (1, 2, 2, 2))
        test(False, nn.Conv3d(1, 1, (3, 3, 3)), (1, 3, 3, 3))
        test(False, nn.Conv3d(1, 1, (3, 3, 3), padding=1), (1, 2, 2, 2))

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

    def _test_Conv2d_naive_groups(self, dtype=torch.FloatTensor):
        # Check that grouped convolutions matches two half convolutions
        m = nn.Conv2d(4, 4, kernel_size=3, groups=2).type(dtype)
        i = Variable(torch.randn(2, 4, 6, 6).type(dtype), requires_grad=True)
        output = m(i)
        grad_output = torch.randn(2, 4, 4, 4).type(dtype)
        output.backward(grad_output)

        m1 = nn.Conv2d(2, 2, kernel_size=3).type(dtype)
        m1.weight.data.copy_(m.weight.data[:2])
        m1.bias.data.copy_(m.bias.data[:2])
        i1 = Variable(i.data[:, :2].contiguous(), requires_grad=True)
        output1 = m1(i1)
        output1.backward(grad_output[:, :2].contiguous())

        m2 = nn.Conv2d(2, 2, kernel_size=3).type(dtype)
        m2.weight.data.copy_(m.weight.data[2:])
        m2.bias.data.copy_(m.bias.data[2:])
        i2 = Variable(i.data[:, 2:].contiguous(), requires_grad=True)
        output2 = m2(i2)
        output2.backward(grad_output[:, 2:].contiguous())

        self.assertEqual(output, torch.cat([output1, output2], 1))
        self.assertEqual(i.grad.data,
                         torch.cat([i1.grad.data, i2.grad.data], 1),
                         prec=type2prec[dtype.__name__])
        self.assertEqual(m.bias.grad.data,
                         torch.cat([m1.bias.grad.data, m2.bias.grad.data], 0),
                         prec=type2prec[dtype.__name__])
        self.assertEqual(m.weight.grad.data,
                         torch.cat([m1.weight.grad.data, m2.weight.grad.data], 0),
                         prec=type2prec[dtype.__name__])

    # For https://github.com/pytorch/pytorch/pull/1273
    # Almost identical to the above `test_Conv2d_naive_groups`
    def test_Conv2d_groups_nobias(self):
        types = (torch.FloatTensor,)
        if TEST_CUDA:
            types += (torch.cuda.FloatTensor, torch.cuda.HalfTensor)
        for tp in types:
            m = nn.Conv2d(4, 4, kernel_size=3, groups=2, bias=False).type(tp)
            i = Variable(torch.randn(2, 4, 6, 6).type(tp), requires_grad=True)
            output = m(i)
            grad_output = torch.randn(2, 4, 4, 4).type(tp)
            output.backward(grad_output)

            m1 = nn.Conv2d(2, 2, kernel_size=3, bias=False).type(tp)
            m1.weight.data.copy_(m.weight.data[:2])
            i1 = Variable(i.data[:, :2].contiguous(), requires_grad=True)
            output1 = m1(i1)
            output1.backward(grad_output[:, :2].contiguous())

            m2 = nn.Conv2d(2, 2, kernel_size=3, bias=False).type(tp)
            m2.weight.data.copy_(m.weight.data[2:])
            i2 = Variable(i.data[:, 2:].contiguous(), requires_grad=True)
            output2 = m2(i2)
            output2.backward(grad_output[:, 2:].contiguous())

            self.assertEqual(output, torch.cat([output1, output2], 1))
            self.assertEqual(i.grad.data,
                             torch.cat([i1.grad.data, i2.grad.data], 1),
                             type2prec[tp.__name__])
            self.assertEqual(m.weight.grad.data,
                             torch.cat([m1.weight.grad.data, m2.weight.grad.data], 0),
                             type2prec[tp.__name__])

    # Very similar to test_Conv2d_naive_groups but with special care to handle
    # the number of groups == number of input channels
    @unittest.skipIf(not TEST_CUDA, 'CUDA not available')
    @repeat_test_for_types(ALL_TENSORTYPES)
    def test_Conv2d_depthwise_naive_groups(self, dtype=torch.FloatTensor):
        dtype = getattr(torch.cuda, dtype.__name__)
        for depth_multiplier in [1, 2]:
            m = nn.Conv2d(2, 2 * depth_multiplier, kernel_size=3, groups=2).type(dtype)
            i = Variable(torch.randn(2, 2, 6, 6).type(dtype) / 2, requires_grad=True)
            output = m(i)
            grad_output = torch.randn(2, 2 * depth_multiplier, 4, 4).type(dtype) / 2
            output.backward(grad_output)

            offset = 1 * depth_multiplier

            m1 = nn.Conv2d(1, 1 * depth_multiplier, kernel_size=3).type(dtype)
            m1.weight.data = m.weight.data[:offset].clone()
            m1.bias.data = m.bias.data[:offset].clone()
            i1 = Variable(i.data[:, :1].contiguous(), requires_grad=True)
            output1 = m1(i1)
            output1.backward(grad_output[:, :offset].contiguous())

            m2 = nn.Conv2d(1, 1 * depth_multiplier, kernel_size=3).type(dtype)
            m2.weight.data.copy_(m.weight.data[offset:])
            m2.bias.data.copy_(m.bias.data[offset:])
            i2 = Variable(i.data[:, 1:].contiguous(), requires_grad=True)
            output2 = m2(i2)
            output2.backward(grad_output[:, offset:].contiguous())

            self.assertEqual(output, torch.cat([output1, output2], 1),
                             prec=type2prec[dtype.__name__])
            self.assertEqual(i.grad.data,
                             torch.cat([i1.grad.data, i2.grad.data], 1),
                             prec=type2prec[dtype.__name__])
            self.assertEqual(m.bias.grad.data,
                             torch.cat([m1.bias.grad.data,
                                        m2.bias.grad.data], 0),
                             prec=type2prec[dtype.__name__])
            self.assertEqual(m.weight.grad.data,
                             torch.cat([m1.weight.grad.data,
                                        m2.weight.grad.data], 0),
                             prec=type2prec[dtype.__name__])

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

    def _test_loss_equal_input_target_shape(self, cast):
        # Tests losses whose inputs should have the same size.
        losses = {
            'mse_loss': lambda x, y: F.mse_loss(x, y),
            'l1_loss': lambda x, y: F.l1_loss(x, y),
            'smooth_l1_loss': lambda x, y: F.smooth_l1_loss(x, y),
            'kl_div': lambda x, y: F.kl_div(x, y),
            'poisson_nll_loss': lambda x, y: F.poisson_nll_loss(x, y),
        }

        input = Variable(cast(torch.randn(3, 5)))
        target = Variable(cast(torch.randn(5, 3)))
        for name, fn in losses.items():
            self.assertRaises(Exception, lambda: fn(input, target))

    def test_loss_equal_input_target_shape(self):
        self._test_loss_equal_input_target_shape(lambda x: x)

    def test_NLLLoss_mismatched_batch(self):
        x = torch.randn((10, 3), requires_grad=True)
        # t should have size (10,)
        t = torch.zeros((3,), dtype=torch.int64)
        with self.assertRaisesRegex(ValueError, 'Expected.*batch_size'):
            F.nll_loss(x, t)

    def test_RNN_cell_no_broadcasting(self):
        def test(cell_module, input, hx, input_size, hidden_size):
            cell = cell_module(input_size, hidden_size)
            self.assertRaises(RuntimeError, lambda: cell(input, hx))

        def test_all(hidden_size, bad_hx, good_hx, input_size, input):
            test(nn.RNNCell, input, bad_hx, input_size, hidden_size)
            test(nn.GRUCell, input, bad_hx, input_size, hidden_size)
            test(nn.LSTMCell, input, (bad_hx, good_hx), input_size, hidden_size)
            test(nn.LSTMCell, input, (good_hx, bad_hx), input_size, hidden_size)

        hidden_size = 20
        input_size = 10
        input = Variable(torch.randn(3, input_size))
        bad_hx = Variable(torch.randn(1, hidden_size))
        good_hx = Variable(torch.randn(3, hidden_size))

        # Test hidden/input batch size broadcasting
        test_all(hidden_size, bad_hx, good_hx, input_size, input)

        # Test hx's hidden_size vs module's hidden_size broadcasting
        bad_hx = Variable(torch.randn(3, 1))
        test_all(hidden_size, bad_hx, good_hx, input_size, input)

        # Test input's input_size vs module's input_size broadcasting
        bad_input = Variable(torch.randn(3, 1))
        test_all(hidden_size, good_hx, good_hx, input_size, bad_input)

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

    def test_pad_sequence(self):
        def pad(tensor, length):
            return torch.cat(
                [tensor.data, tensor.data.new(
                    length - tensor.size(0), *tensor.size()[1:]).zero_()])
        # single dimensional
        a = Variable(torch.Tensor([1, 2, 3]))
        b = Variable(torch.Tensor([4, 5]))
        c = Variable(torch.Tensor([6]))

        # batch_first = true
        expected = Variable(torch.Tensor([[1, 2, 3], [4, 5, 0], [6, 0, 0]]))
        padded = rnn_utils.pad_sequence([a, b, c], True)
        self.assertEqual(padded, expected)

        # batch_first = false
        padded = rnn_utils.pad_sequence([a, b, c])
        self.assertEqual(padded, expected.transpose(0, 1))

        # pad with non-zero value
        expected = Variable(torch.Tensor([[1, 2, 3], [4, 5, 1], [6, 1, 1]]))
        padded = rnn_utils.pad_sequence([a, b, c], True, 1)
        self.assertEqual(padded, expected)

        # more dimensional
        maxlen = 9
        for num_dim in (0, 1, 2, 3):
            sequences = []
            trailing_dims = [4] * num_dim
            for i in range(maxlen, 0, -1):
                seq_len = i * i
                sequences.append(Variable(torch.rand(seq_len, 5, *trailing_dims)))
            expected = []
            for seq in sequences:
                expected.append(pad(seq, maxlen * maxlen))
            # batch first = true
            expected = Variable(torch.stack(expected))
            padded = rnn_utils.pad_sequence(sequences, True)
            self.assertEqual(padded, expected)

            # batch first = false
            padded = rnn_utils.pad_sequence(sequences)
            self.assertEqual(padded, expected.transpose(0, 1))

        # unsorted sequences should raise exception
        self.assertRaises(
            ValueError, lambda: rnn_utils.pad_sequence([b, a, c], [2, 3, 1]))

    def test_pack_sequence(self):
        def _compatibility_test(sequences, lengths, batch_first):
            padded = rnn_utils.pad_sequence(sequences, batch_first)
            packed = rnn_utils.pack_sequence(sequences)
            unpacked = rnn_utils.pad_packed_sequence(packed, batch_first)
            self.assertEqual(padded, unpacked[0])
            pack_padded = rnn_utils.pack_padded_sequence(padded, lengths, batch_first)
            self.assertEqual(packed, pack_padded)

        # single dimensional
        a = Variable(torch.Tensor([1, 2, 3]))
        b = Variable(torch.Tensor([4, 5]))
        c = Variable(torch.Tensor([6]))
        packed = rnn_utils.pack_sequence([a, b, c])
        expected = torch.Tensor([1, 4, 6, 2, 5, 3])
        self.assertEqual(packed.batch_sizes, [3, 2, 1])
        self.assertEqual(packed.data.data, expected)

        # more dimensions
        maxlen = 9
        for num_dim in (0, 1, 2, 3):
            sequences = []
            lengths = []
            trailing_dims = [4] * num_dim
            for i in range(maxlen, 0, -1):
                seq_len = i * i
                lengths.append(seq_len)
                sequences.append(Variable(torch.rand(seq_len, 5, *trailing_dims)))

            # compatibility with other utilities
            for batch_first in (True, False):
                _compatibility_test(sequences, lengths, batch_first)

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

    def _test_variable_sequence(self, cuda, dtype=torch.FloatTensor):
        def pad(var, length):
            if var.size(0) == length:
                return var
            return torch.cat([var, Variable(var.data.new(length - var.size(0), *var.size()[1:]).zero_())])

        lengths = [10, 10, 6, 2, 2, 1, 1]
        max_length = lengths[0]
        x_leaf = Variable(torch.randn(max_length, len(lengths), 3).type(dtype), requires_grad=True)
        lstm = nn.LSTM(3, 4, bidirectional=True, num_layers=2).type(dtype)
        lstm2 = deepcopy(lstm).type(dtype)
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
            self.assertEqual(p1.grad, p2.grad, type2prec[dtype.__name__])

    def test_variable_sequence(self):
        self._test_variable_sequence(False)

    @unittest.skipIf(not TEST_CUDA, 'CUDA not available')
    @repeat_test_for_types(ALL_TENSORTYPES)
    def test_variable_sequence_cuda(self, dtype=torch.FloatTensor):
        dtype = getattr(torch.cuda, dtype.__name__)
        self._test_variable_sequence(True, dtype)

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

    @unittest.skipIf(not TEST_CUDNN, 'CUDNN not available')
    def test_cudnn_weight_format(self):
        rnns = [
            nn.LSTM(10, 20, batch_first=True),
            nn.GRU(10, 20, batch_first=True),
            nn.RNN(10, 20, batch_first=True)
        ]
        first_warn = True
        for rnn in rnns:
            rnn.cuda()
            input = Variable(torch.randn(5, 4, 10).cuda(), requires_grad=True)
            hx = Variable(torch.randn(1, 5, 20).cuda(), requires_grad=True)
            all_vars = [input, hx] + list(rnn.parameters())
            if isinstance(rnn, nn.LSTM):
                cx = Variable(torch.randn(1, 5, 20).cuda(), requires_grad=True)
                all_vars[2:2] = [cx]
                hx = (hx, cx)

            output = rnn(input, hx)
            output[0].sum().backward()
            grads = [v.grad.data.clone() for v in all_vars]
            for v in all_vars:
                v.grad.data.zero_()

            # Weights will no longer view onto the same chunk of memory
            weight = all_vars[4]
            weight_data = weight.data.clone()
            weight.data.set_(weight_data)

            for i in range(2):
                with warnings.catch_warnings(record=True) as w:
                    output_noncontig = rnn(input, hx)
                if first_warn:
                    self.assertEqual(len(w), 1)
                    self.assertIn('weights are not part of single contiguous chunk of memory', w[0].message.args[0])
                    first_warn = False
                output_noncontig[0].sum().backward()
                grads_noncontig = [v.grad.data.clone() for v in all_vars]
                for v in all_vars:
                    v.grad.data.zero_()
                self.assertEqual(output, output_noncontig)
                self.assertEqual(grads_noncontig, grads)

            # Make sure these still share storage
            weight_data[:] = 4
            self.assertEqual(weight_data, all_vars[4].data)

    @unittest.skipIf(not TEST_CUDNN, 'CUDNN not available')
    def test_cudnn_weight_tying(self):
        rnns = [
            nn.LSTM(10, 20, batch_first=True, bidirectional=True),
            nn.GRU(10, 20, batch_first=True, bidirectional=True),
            nn.RNN(10, 20, batch_first=True, bidirectional=True)
        ]
        for rnn in rnns:
            rnn.bias_ih_l0_reverse = rnn.bias_ih_l0
            rnn.cuda()
            input = Variable(torch.randn(5, 4, 10).cuda(), requires_grad=True)
            hx = Variable(torch.randn(2, 5, 20).cuda(), requires_grad=True)
            all_vars = [input, hx] + list(rnn.parameters())
            opt = torch.optim.SGD(rnn.parameters(), lr=0.1)
            opt.zero_grad()
            if isinstance(rnn, nn.LSTM):
                cx = Variable(torch.randn(2, 5, 20).cuda(), requires_grad=True)
                all_vars[2:2] = [cx]
                hx = (hx, cx)

            with warnings.catch_warnings(record=True) as w:
                output = rnn(input, hx)
            output[0].sum().backward()

            opt.step()
            with warnings.catch_warnings(record=True) as w:
                output_cuda = rnn(input, hx)
            rnn.cpu()
            hx = (hx[0].cpu(), hx[1].cpu()) if isinstance(rnn, nn.LSTM) else hx.cpu()
            output_cpu = rnn(input.cpu(), hx)
            self.assertEqual(output_cuda, output_cpu)

    @unittest.skipIf(not TEST_CUDA, 'CUDA not available')
    @repeat_test_for_types(ALL_TENSORTYPES)
    def test_cuda_rnn_fused(self, dtype=torch.FloatTensor):
        dtype = getattr(torch.cuda, dtype.__name__)

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
        input_val = torch.randn(seq_length, batch, input_size).type(dtype)
        grad_output = torch.randn(seq_length, batch, hidden_size).type(dtype)
        hx_val = torch.randn(num_layers, batch, hidden_size).type(dtype)
        grad_hy = torch.randn(num_layers, batch, hidden_size).type(dtype)
        with torch.backends.cudnn.flags(enabled=False):
            for module in (nn.GRU, nn.LSTM):
                for bias in (True, False):
                    rnn = module(input_size, hidden_size, num_layers, bias=bias).type(dtype)
                    rnn_cuda = module(input_size, hidden_size, num_layers, bias=bias).type(dtype).cuda()
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

    def test_rnn_args_check(self):
        input_size = 3
        hidden_size = 5
        num_layers = 2
        batch_size = 4
        seq_len = 6
        num_directions = 1

        def test(input_shape, hidden_shape, mode):
            for input, hidden in get_inputs(input_shape, hidden_shape, mode):
                model = getattr(nn, mode)(input_size, hidden_size, num_layers)
                self.assertRaises(RuntimeError, lambda: model(input, hidden))

        correct_input_shape = (seq_len, batch_size, input_size)
        correct_hidden_shape = (num_layers * num_directions, batch_size, hidden_size)

        def update_tuple(tup, dim, delta):
            new_tup = list(tup)
            new_tup[dim] = delta
            return tuple(new_tup)

        def get_inputs(input_shape, hidden_shape, mode):
            '''returns list( tuple(input, hidden) )
            where input, hidden are inputs to a model'''
            input = Variable(torch.randn(input_shape))
            hidden = Variable(torch.randn(hidden_shape))
            if mode is not 'LSTM':
                return [(input, hidden)]
            if hidden_shape == correct_hidden_shape:
                return [(input, (hidden, hidden))]
            good_hidden = Variable(torch.randn(correct_hidden_shape))
            return [
                (input, (hidden, good_hidden)),
                (input, (good_hidden, hidden)),
            ]

        rnn_modes = ['RNN', 'GRU', 'LSTM']
        for mode in rnn_modes:
            # Incorrect input batch size
            input_shape = update_tuple(correct_input_shape, 1, -1)
            hidden_shape = correct_hidden_shape
            test(input_shape, hidden_shape, mode)

            # Incorrect hidden batch size
            input_shape = correct_input_shape
            hidden_shape = update_tuple(correct_hidden_shape, 1, -1)
            test(input_shape, hidden_shape, mode)

            # Incorrect input size
            input_shape = update_tuple(correct_input_shape, 2, -1)
            hidden_shape = correct_hidden_shape
            test(input_shape, hidden_shape, mode)

            # Incorrect hidden size
            input_shape = correct_input_shape
            hidden_shape = update_tuple(correct_hidden_shape, 2, -1)
            test(input_shape, hidden_shape, mode)

            # Incorrect hidden[0]
            input_shape = correct_input_shape
            hidden_shape = update_tuple(correct_hidden_shape, 0, -1)
            test(input_shape, hidden_shape, mode)

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
    @repeat_test_for_types(ALL_TENSORTYPES)
    def test_rnn_retain_variables_cuda(self, dtype=torch.FloatTensor):
        dtype = getattr(torch.cuda, dtype.__name__)
        with torch.backends.cudnn.flags(enabled=False):
            self._test_rnn_retain_variables(dtype)
        self._test_rnn_retain_variables(dtype)

    def _test_RNN_cpu_vs_cudnn(self, dropout):

        def forward_backward(cuda, rnn, input_val, hx_val, grad_output, grad_hy, weights_val):
            is_lstm = isinstance(rnn, nn.LSTM)

            for x_layer, y_layer in zip(rnn.all_weights, weights_val):
                for x, y in zip(x_layer, y_layer):
                    x.data.copy_(y.data)

            if isinstance(input_val, rnn_utils.PackedSequence):
                input = rnn_utils.PackedSequence(
                    Variable(input_val.data.data, requires_grad=True), input_val.batch_sizes)
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
            for bias, bidirectional, batch_first, contig, variable_len, lens_as_variable \
                    in product((True, False), repeat=6):

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
                    lengths = [7, 5, 5, 2, 1, 1]
                    if lens_as_variable:
                        lengths = Variable(torch.LongTensor(lengths))
                    input_val = Variable(input_val)
                    grad_output = Variable(grad_output)
                    input_val = rnn_utils.pack_padded_sequence(input_val, lengths, batch_first=batch_first)
                    grad_output = rnn_utils.pack_padded_sequence(grad_output, lengths, batch_first=batch_first).data

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
                    rnn2.flatten_parameters()
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
    @repeat_test_for_types(ALL_TENSORTYPES)
    def test_noncontig_conv_grad(self, dtype=torch.FloatTensor):
        dtype = getattr(torch.cuda, dtype.__name__)
        # FIXME: remove after adding non-contiguous grad tests for all modules
        module = nn.Conv2d(3, 5, kernel_size=3, padding=1).type(dtype).cuda()
        input = Variable(torch.randn(2, 3, 10, 10).type(dtype).cuda(), requires_grad=True)
        output = module(input)

        grad = torch.randn(2, 2, 5, 10, 10).type(dtype).cuda()[:, 1]
        assert not grad.is_contiguous()
        output.backward(grad, retain_graph=True)
        self.assertIsNotNone(input.grad)
        result = input.grad.data.clone()
        input.grad.data.zero_()

        output.backward(grad.contiguous())
        self.assertEqual(result, input.grad.data, type2prec[dtype.__name__])

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

    def test_elu_inplace_view(self):
        v = Variable(torch.Tensor([1.0, -1.0, 1.0, -1.0]), requires_grad=True)

        def func(root):
            x = root.clone()
            view = x.narrow(0, 1, 2)
            res = F.elu(view, inplace=True)
            self.assertIs(res, view)
            return x

        gradcheck(func, [v])
        gradgradcheck(func, [v])

    def test_relu_inplace_view(self):
        v = Variable(torch.Tensor([1.0, -1.0, 1.0, -1.0]), requires_grad=True)

        def func(root):
            x = root.clone()
            view = x.narrow(0, 1, 2)
            res = F.relu(view, inplace=True)
            self.assertIs(res, view)
            return x

        gradcheck(func, [v])
        gradgradcheck(func, [v])

    def test_bce_with_logits_raises_if_target_and_input_are_different_size(self):
        target = Variable(torch.rand(5))
        input = Variable(torch.rand(5, 1))
        with self.assertRaises(ValueError):
            nn.BCEWithLogitsLoss()(input, target)

        target = Variable(torch.rand(5, 1))
        input = Variable(torch.rand(5))
        with self.assertRaises(ValueError):
            nn.BCEWithLogitsLoss()(input, target)

    def test_bce_with_logits_gives_same_result_as_sigmoid_and_bce_loss(self):
        sigmoid = nn.Sigmoid()

        target = Variable(torch.rand(64, 4))
        output = Variable(torch.rand(64, 4) - 0.5)

        self.assertEqual(nn.BCEWithLogitsLoss()(output, target), nn.BCELoss()(sigmoid(output), target))

        weight = torch.rand(4)
        self.assertEqual(nn.BCEWithLogitsLoss(weight)(output, target), nn.BCELoss(weight)(sigmoid(output), target))

        target = Variable(torch.FloatTensor(4, 1).fill_(0))
        output = Variable(torch.FloatTensor(4, 1).fill_(-100))

        self.assertEqual(nn.BCEWithLogitsLoss()(output, target), nn.BCELoss()(sigmoid(output), target))

        self.assertEqual(nn.BCEWithLogitsLoss(reduce=False)(output, target),
                         nn.BCELoss(reduce=False)(sigmoid(output), target))

        weight = torch.FloatTensor(1).uniform_()
        self.assertEqual(nn.BCEWithLogitsLoss(weight)(output, target), nn.BCELoss(weight)(sigmoid(output), target))

    def test_bce_with_logits_has_correct_grad_at_zero(self):
        output = Variable(torch.zeros(3, 1), requires_grad=True)
        target = Variable(torch.zeros(3, 1))
        nn.BCEWithLogitsLoss(size_average=False)(output, target).backward()
        expected_grad = Variable(torch.Tensor(3, 1).fill_(0.5))
        self.assertEqual(output.grad, expected_grad)

    def test_bce_with_logits_broadcasts_weights(self):
        target = Variable(torch.rand(16, 4))
        output = Variable(torch.rand(16, 4) - 0.5)

        weight = torch.rand(4)
        out1 = nn.BCEWithLogitsLoss(weight)(output, target)

        weight = weight.expand(16, 4).contiguous()
        out2 = nn.BCEWithLogitsLoss(weight)(output, target)

        self.assertEqual(out1, out2)

        weight = torch.rand(16, 1)
        out1 = nn.BCEWithLogitsLoss(weight)(output, target)

        weight = weight.expand(16, 4).contiguous()
        out2 = nn.BCEWithLogitsLoss(weight)(output, target)

        self.assertEqual(out1, out2)

    def test_bce_loss_broadcasts_weights(self):
        sigmoid = nn.Sigmoid()
        target = Variable(torch.rand(16, 4))
        output = Variable(torch.rand(16, 4) - 0.5)

        weight = torch.rand(4)
        out1 = nn.BCELoss(weight)(sigmoid(output), target)

        weight = weight.expand(16, 4).contiguous()
        out2 = nn.BCELoss(weight)(sigmoid(output), target)

        self.assertEqual(out1, out2)

        weight = torch.rand(16, 1)
        out1 = nn.BCELoss(weight)(sigmoid(output), target)

        weight = weight.expand(16, 4).contiguous()
        out2 = nn.BCELoss(weight)(sigmoid(output), target)

        self.assertEqual(out1, out2)

    def test_elu_inplace_gradgrad(self):
        v = Variable(torch.randn(8), requires_grad=True)

        def func(root):
            x = root.clone()
            return F.elu(x, inplace=True)

        gradcheck(func, [v])
        gradgradcheck(func, [v])

    def test_hardtanh_inplace_gradgrad(self):
        v = Variable(torch.randn(8), requires_grad=True)

        def func(root):
            x = root.clone()
            return F.hardtanh(x, inplace=True)

        gradcheck(func, [v])
        gradgradcheck(func, [v])

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_batchnorm_cudnn_half(self):
        # THNN
        input = Variable(torch.rand(2, 3, 2, 2).half().cuda().random_(1, 10), requires_grad=True)
        m = nn.BatchNorm2d(3).half().cuda()
        thnn_output = m(input)
        thnn_output.sum().backward()
        thnn_input_grad = input.grad.data.clone()
        self.assertEqual(thnn_output.type(), input.type())
        # cuDNN
        if TEST_CUDNN:
            input.grad = None
            m = m.float()
            cudnn_output = m(input)
            cudnn_output.sum().backward()
            cudnn_input_grad = input.grad.data.clone()
            self.assertEqual(cudnn_output.type(), input.type())
            self.assertEqual(cudnn_output, thnn_output)
            self.assertAlmostEqual(cudnn_input_grad, thnn_input_grad, delta=1e-3)

    def _test_batchnorm_update_stats(self, test_type=torch.FloatTensor):
        module = nn.BatchNorm1d(3).type(test_type)

        data = Variable(torch.rand(4, 3).type(test_type))

        # training pass
        old_running_mean = module.running_mean.clone()
        old_running_var = module.running_var.clone()
        module(data)
        self.assertNotEqual(old_running_mean, module.running_mean)
        self.assertNotEqual(old_running_var, module.running_var)

        # eval pass
        module.eval()
        old_running_mean = module.running_mean.clone()
        old_running_var = module.running_var.clone()
        module(data)
        self.assertEqual(old_running_mean, module.running_mean)
        self.assertEqual(old_running_var, module.running_var)

    def test_batchnorm_update_stats(self):
        self._test_batchnorm_update_stats()

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_batchnorm_update_stats_cuda(self):
        self._test_batchnorm_update_stats(torch.cuda.FloatTensor)

    def test_batchnorm_raises_error_if_running_mean_is_not_same_size_as_input(self):
        input = Variable(torch.rand(2, 10))
        running_var = torch.rand(10)
        wrong_sizes = [9, 11]
        for size in wrong_sizes:
            with self.assertRaises(RuntimeError):
                F.batch_norm(input, torch.rand(size), running_var)

    def test_batchnorm_raises_error_if_running_var_is_not_same_size_as_input(self):
        input = Variable(torch.rand(2, 10))
        running_mean = torch.rand(10)
        wrong_sizes = [9, 11]
        for size in wrong_sizes:
            with self.assertRaises(RuntimeError):
                F.batch_norm(input, running_mean, torch.rand(size))

    def test_batchnorm_raises_error_if_weight_is_not_same_size_as_input(self):
        input = Variable(torch.rand(2, 10))
        running_mean = torch.rand(10)
        running_var = torch.rand(10)
        wrong_sizes = [9, 11]
        for size in wrong_sizes:
            with self.assertRaises(RuntimeError):
                F.batch_norm(input, running_mean, running_var, weight=Parameter(torch.rand(size)))

    def test_batchnorm_raises_error_if_bias_is_not_same_size_as_input(self):
        input = Variable(torch.rand(2, 10))
        running_mean = torch.rand(10)
        running_var = torch.rand(10)
        wrong_sizes = [9, 11]
        for size in wrong_sizes:
            with self.assertRaises(RuntimeError):
                F.batch_norm(input, running_mean, running_var, bias=Parameter(torch.rand(size)))

    def _test_batchnorm_eval(self, dtype=torch.FloatTensor):
        module = nn.BatchNorm1d(3).type(dtype)
        module.eval()

        data = Variable(torch.rand(4, 3).type(dtype), requires_grad=True)
        grad = torch.rand(4, 3).type(dtype)

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

        # track_running_stats=False
        module = nn.BatchNorm1d(3, track_running_stats=False).type(dtype)

        data = Variable(torch.rand(4, 3).type(dtype), requires_grad=True)
        grad = torch.rand(4, 3).type(dtype)

        # 1st pass
        res1 = module(data)
        res1.backward(grad)
        grad1 = data.grad.data.clone()

        # set eval
        module.eval()

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

    def test_cosine_embedding_loss_no_reduce(self):
        input1 = Variable(torch.randn(15, 10), requires_grad=True)
        input2 = Variable(torch.randn(15, 10), requires_grad=True)
        target = Variable(torch.randn(15).sign())
        self.assertTrue(gradcheck(lambda x, y, z: F.cosine_embedding_loss(
            x, y, z, reduce=False), (input1, input2, target)))
        self.assertEqual(F.cosine_embedding_loss(input1, input2, target, reduce=False),
                         loss_reference_fns['CosineEmbeddingLoss'](input1, input2, target, reduce=False))

    def test_cosine_embedding_loss_margin_no_reduce(self):
        input1 = Variable(torch.randn(15, 10), requires_grad=True)
        input2 = Variable(torch.randn(15, 10), requires_grad=True)
        target = Variable(torch.randn(15).sign())
        self.assertTrue(gradcheck(lambda x, y, z: F.cosine_embedding_loss(
            x, y, z, margin=0.5, reduce=False), (input1, input2, target)))
        self.assertEqual(F.cosine_embedding_loss(input1, input2, target, margin=0.5, reduce=False),
                         loss_reference_fns['CosineEmbeddingLoss'](input1, input2, target, margin=0.5, reduce=False))

    def test_margin_ranking_loss_no_reduce(self):
        input1 = Variable(torch.randn(15).mul(10), requires_grad=True)
        input2 = Variable(torch.randn(15).mul(10), requires_grad=True)
        target = Variable(torch.randn(15).sign())
        self.assertTrue(gradcheck(lambda x, y, z: F.margin_ranking_loss(
            x, y, z, reduce=False), (input1, input2, target)))
        self.assertEqual(F.margin_ranking_loss(input1, input2, target, reduce=False),
                         loss_reference_fns['MarginRankingLoss'](input1, input2, target, reduce=False))

    def test_margin_ranking_loss_margin_no_reduce(self):
        input1 = Variable(torch.randn(15).mul(10), requires_grad=True)
        input2 = Variable(torch.randn(15).mul(10), requires_grad=True)
        target = Variable(torch.randn(15).sign())
        self.assertTrue(gradcheck(lambda x, y, z: F.margin_ranking_loss(
            x, y, z, margin=0.5, reduce=False), (input1, input2, target)))
        self.assertEqual(F.margin_ranking_loss(input1, input2, target, margin=0.5, reduce=False),
                         loss_reference_fns['MarginRankingLoss'](input1, input2, target, margin=0.5, reduce=False))

    def test_triplet_margin_loss(self):
        input1 = Variable(torch.randn(5, 10), requires_grad=True)
        input2 = Variable(torch.randn(5, 10), requires_grad=True)
        input3 = Variable(torch.randn(5, 10), requires_grad=True)
        self.assertTrue(gradcheck(lambda x1, x2, x3: F.triplet_margin_loss(
            x1, x2, x3), (input1, input2, input3)))
        self.assertEqual(F.triplet_margin_loss(input1, input2, input3),
                         loss_reference_fns['TripletMarginLoss'](input1, input2, input3))

    def test_triplet_margin_loss_swap(self):
        input1 = Variable(torch.randn(5, 10), requires_grad=True)
        input2 = Variable(torch.randn(5, 10), requires_grad=True)
        input3 = Variable(torch.randn(5, 10), requires_grad=True)
        self.assertTrue(gradcheck(lambda x1, x2, x3: F.triplet_margin_loss(
            x1, x2, x3, swap=True), (input1, input2, input3)))
        self.assertEqual(F.triplet_margin_loss(input1, input2, input3, swap=True),
                         loss_reference_fns['TripletMarginLoss'](input1, input2, input3, swap=True))

    def test_triplet_margin_loss_no_reduce(self):
        input1 = Variable(torch.randn(5, 10), requires_grad=True)
        input2 = Variable(torch.randn(5, 10), requires_grad=True)
        input3 = Variable(torch.randn(5, 10), requires_grad=True)
        self.assertTrue(gradcheck(lambda x1, x2, x3: F.triplet_margin_loss(
            x1, x2, x3, reduce=False), (input1, input2, input3)))
        self.assertEqual(F.triplet_margin_loss(input1, input2, input3, reduce=False),
                         loss_reference_fns['TripletMarginLoss'](input1, input2, input3, reduce=False))

    def test_triplet_margin_loss_swap_no_reduce(self):
        input1 = Variable(torch.randn(5, 10), requires_grad=True)
        input2 = Variable(torch.randn(5, 10), requires_grad=True)
        input3 = Variable(torch.randn(5, 10), requires_grad=True)
        self.assertTrue(gradcheck(lambda x1, x2, x3: F.triplet_margin_loss(
            x1, x2, x3, swap=True, reduce=False), (input1, input2, input3)))
        self.assertEqual(F.triplet_margin_loss(input1, input2, input3, swap=True, reduce=False),
                         loss_reference_fns['TripletMarginLoss'](input1, input2, input3, swap=True, reduce=False))

    def test_cosine_similarity(self):
        input1 = Variable(torch.randn(4, 4), requires_grad=True)
        input2 = Variable(torch.randn(4, 4), requires_grad=True)
        self.assertTrue(gradcheck(lambda x, y: F.cosine_similarity(x, y), (input1, input2)))

        input1 = Variable(torch.randn(4, 5, 6), requires_grad=True)
        input2 = Variable(torch.randn(4, 5, 6), requires_grad=True)
        self.assertTrue(gradcheck(lambda x, y: F.cosine_similarity(x, y, dim=0), (input1, input2)))
        self.assertTrue(gradcheck(lambda x, y: F.cosine_similarity(x, y, dim=-1), (input1, input2)))

        input1 = torch.randn((), requires_grad=True)
        input2 = torch.randn((), requires_grad=True)
        self.assertTrue(gradcheck(lambda x, y: F.cosine_similarity(x, y, dim=0), (input1, input2)))
        self.assertTrue(gradcheck(lambda x, y: F.cosine_similarity(x, y, dim=-1), (input1, input2)))

        # Check cosine_similarity input/output shapes
        input_size = (1, 3, 2, 1)
        expected_size = (1, 2, 1)
        input1 = Variable(torch.randn(input_size), requires_grad=True)
        input2 = Variable(torch.randn(input_size), requires_grad=True)
        self.assertEqual(F.cosine_similarity(input1, input2, dim=1).size(), expected_size)

    def test_grid_sample(self):
        def test_cpu_against_cuda(N, C, H, W, padding_mode):
            def test_shape(N, C, IH, IW, H, W, padding_mode):

                input_cpu = Variable(torch.randn(C, N, IH, IW).transpose(0, 1), requires_grad=True)
                grid_cpu = Variable(torch.randn(H, N, W, 2).transpose(0, 1), requires_grad=True)
                out_cpu = F.grid_sample(input_cpu, grid_cpu, padding_mode=padding_mode)
                self.assertTrue(out_cpu.size() == torch.Size([N, C, H, W]))

                input_cuda = Variable(input_cpu.data.transpose(0, 1).cuda().transpose(0, 1), requires_grad=True)
                grid_cuda = Variable(grid_cpu.data.transpose(0, 1).cuda().transpose(0, 1), requires_grad=True)
                out_cuda = F.grid_sample(input_cuda, grid_cuda, padding_mode=padding_mode)
                self.assertEqual(out_cpu, out_cuda)

                gradients = out_cpu.data.new(out_cpu.size()).normal_()
                out_cpu.backward(gradients)
                out_cuda.backward(gradients.cuda())
                self.assertEqual(input_cpu.grad, input_cuda.grad)
                self.assertEqual(grid_cpu.grad, grid_cuda.grad, prec=5e-5)

                # check that zero-dimensional input strides don't error out
                base_input = torch.randn(C, IH, IW)
                input_cpu = Variable(base_input.expand(input_cuda.size()), requires_grad=True)
                grid_cpu = Variable(torch.randn(N, H, W, 2), requires_grad=True)
                out_cpu = F.grid_sample(input_cpu, grid_cpu, padding_mode=padding_mode)

                input_cuda = Variable(base_input.cuda().expand(input_cuda.size()), requires_grad=True)
                grid_cuda = Variable(grid_cpu.data.cuda(), requires_grad=True)
                out_cuda = F.grid_sample(input_cuda, grid_cuda, padding_mode=padding_mode)
                self.assertEqual(out_cpu, out_cuda)

            # test same size output
            test_shape(N, C, H, W, H, W, padding_mode)

            # test larger output
            N = random.randint(1, 8)
            C = random.randint(1, 8)
            IH = random.randint(1, 8)
            IW = random.randint(1, 8)
            H = random.randint(IH + 1, 12)
            W = random.randint(IW + 1, 12)
            test_shape(N, C, IH, IW, H, W, padding_mode)

            # test smaller output
            N = random.randint(1, 8)
            C = random.randint(1, 8)
            IH = random.randint(1, 8)
            IW = random.randint(1, 8)
            H = random.randint(1, IH)
            W = random.randint(1, IW)
            test_shape(N, C, IH, IW, H, W, padding_mode)

        # test known input on CPU
        for padding_mode in ['zeros', 'border']:

            input = Variable(torch.arange(1, 11).view(1, 1, 2, 5))
            grid = Variable(torch.Tensor(
                [[-0.9, -1.4, 0, 0.2, 1],
                 [-1, -0.333, 0, 0.5, 1],
                 [-1, -0.5, 0, 0.3333, 1],
                 [-1, -0.2, 0, 1.1, 0.5]]).view(1, 2, 5, 2))
            output = F.grid_sample(input, grid, padding_mode=padding_mode)

            if padding_mode == 'zeros':
                groundtruth = torch.Tensor(
                    [[0.9600, 6.0000000000, 5.0000, 4.8340, 9.0000],
                     [2.2500, 6.333250045, 5.0000, 5.1000, 7.0000]]).view(1, 1, 2, 5)
            else:
                groundtruth = torch.Tensor(
                    [[1.2000, 6.0000000000, 5.0000, 4.8340, 9.0000],
                     [2.2500, 6.333250045, 5.0000, 5.1000, 8.7500]]).view(1, 1, 2, 5)

            self.assertEqual(output.data, groundtruth)

            # do gradcheck
            N = random.randint(1, 8)
            C = random.randint(1, 8)
            H = random.randint(1, 8)
            W = random.randint(1, 8)
            input = Variable(torch.randn(N, C, H, W), requires_grad=True)
            grid = Variable(torch.randn(N, H, W, 2), requires_grad=True)
            self.assertTrue(gradcheck(
                lambda inp, grid: F.grid_sample(inp, grid, padding_mode=padding_mode),
                (input, grid)))

            # test CUDA against CPU
            if TEST_CUDA:
                test_cpu_against_cuda(N, C, H, W, padding_mode)

    def test_grid_sample_3d(self):
        def test_cpu_against_cuda(N, C, D, H, W, padding_mode):
            def test_shape(N, C, ID, IH, IW, D, H, W, padding_mode):

                input_cpu = Variable(torch.randn(C, N, ID, IH, IW).transpose(0, 1), requires_grad=True)
                grid_cpu = Variable(torch.randn(D, N, H, W, 3).transpose(0, 1), requires_grad=True)
                out_cpu = F.grid_sample(input_cpu, grid_cpu, padding_mode=padding_mode)
                self.assertTrue(out_cpu.size() == torch.Size([N, C, D, H, W]))

                input_cuda = Variable(input_cpu.data.transpose(0, 1).cuda().transpose(0, 1), requires_grad=True)
                grid_cuda = Variable(grid_cpu.data.transpose(0, 1).cuda().transpose(0, 1), requires_grad=True)
                out_cuda = F.grid_sample(input_cuda, grid_cuda, padding_mode=padding_mode)
                self.assertEqual(out_cpu, out_cuda)

                gradients = out_cpu.data.new(out_cpu.size()).normal_()
                out_cpu.backward(gradients)
                out_cuda.backward(gradients.cuda())
                self.assertEqual(input_cpu.grad, input_cuda.grad)
                self.assertEqual(grid_cpu.grad, grid_cuda.grad, prec=5e-5)

                # check that zero-dimensional input strides don't error out
                base_input = torch.randn(C, ID, IH, IW)
                input_cpu = Variable(base_input.expand(input_cuda.size()), requires_grad=True)
                grid_cpu = Variable(torch.randn(N, D, H, W, 3), requires_grad=True)
                out_cpu = F.grid_sample(input_cpu, grid_cpu, padding_mode=padding_mode)

                input_cuda = Variable(base_input.cuda().expand(input_cuda.size()), requires_grad=True)
                grid_cuda = Variable(grid_cpu.data.cuda(), requires_grad=True)
                out_cuda = F.grid_sample(input_cuda, grid_cuda, padding_mode=padding_mode)
                self.assertEqual(out_cpu, out_cuda)

            # test same size output
            test_shape(N, C, D, H, W, D, H, W, padding_mode)

            # test larger output
            N = random.randint(1, 8)
            C = random.randint(1, 8)
            ID = random.randint(1, 8)
            IH = random.randint(1, 8)
            IW = random.randint(1, 8)
            D = random.randint(ID + 1, 12)
            H = random.randint(IH + 1, 12)
            W = random.randint(IW + 1, 12)
            test_shape(N, C, ID, IH, IW, D, H, W, padding_mode)

            # test smaller output
            N = random.randint(1, 8)
            C = random.randint(1, 8)
            ID = random.randint(1, 8)
            IH = random.randint(1, 8)
            IW = random.randint(1, 8)
            D = random.randint(1, ID)
            H = random.randint(1, IH)
            W = random.randint(1, IW)
            test_shape(N, C, ID, IH, IW, D, H, W, padding_mode)

        # test known input on CPU
        for padding_mode in ['zeros', 'border']:
            # do gradcheck
            N = random.randint(1, 8)
            C = random.randint(1, 8)
            D = random.randint(1, 8)
            H = random.randint(1, 8)
            W = random.randint(1, 8)
            input = Variable(torch.randn(N, C, D, H, W), requires_grad=True)
            grid = Variable(torch.randn(N, D, H, W, 3), requires_grad=True)
            self.assertTrue(gradcheck(
                lambda inp, grid: F.grid_sample(inp, grid, padding_mode=padding_mode),
                (input, grid)))

            # test CUDA against CPU
            if TEST_CUDA:
                test_cpu_against_cuda(N, C, D, H, W, padding_mode)

    def test_affine_grid(self):
        # test known input on CPU
        input = Variable(torch.arange(1, 7).view(1, 2, 3))
        output = F.affine_grid(input, torch.Size([1, 1, 2, 2]))
        groundtruth = torch.Tensor(
            [[[0, -3], [2, 5]], [[4, 7], [6, 15]]]).view(1, 2, 2, 2)
        self.assertEqual(output.data, groundtruth)

        # do gradcheck
        N = random.randint(1, 8)
        C = random.randint(1, 8)
        H = random.randint(1, 8)
        W = random.randint(1, 8)
        sz = torch.Size([N, C, H, W])
        inp = Variable(torch.randn(N, 2, 3), requires_grad=True)
        self.assertTrue(gradcheck(lambda inp: F.affine_grid(inp, sz), (inp,)))

        # test CPU against CUDA
        if TEST_CUDNN:
            input_cpu = Variable(torch.randn(N, 2, 3), requires_grad=True)
            out_cpu = F.affine_grid(input_cpu, sz)
            gradients = torch.randn(out_cpu.size())
            out_cpu.backward(gradients)
            input_gpu = Variable(input_cpu.data.cuda(), requires_grad=True)
            out_cuda = F.affine_grid(input_gpu, sz)
            out_cuda.backward(gradients.cuda())
            self.assertEqual(out_cpu, out_cuda)
            self.assertEqual(input_cpu.grad, input_gpu.grad)

    def test_upsamplingNearest1d(self):
        m = nn.Upsample(size=4, mode='nearest')
        in_t = torch.ones(1, 1, 2)
        out_t = m(Variable(in_t))
        self.assertEqual(torch.ones(1, 1, 4), out_t.data)

        input = Variable(torch.randn(1, 1, 2), requires_grad=True)
        gradcheck(lambda x: F.upsample(x, 4, mode='nearest'), [input])

    def test_upsamplingLinear1d(self):
        m = nn.Upsample(size=4, mode='linear')
        in_t = torch.ones(1, 1, 2)
        out_t = m(Variable(in_t))
        self.assertEqual(torch.ones(1, 1, 4), out_t.data)

        input = Variable(torch.randn(1, 1, 2), requires_grad=True)
        gradcheck(lambda x: F.upsample(x, 4, mode='linear'), (input,))

    def test_upsamplingLinear1d_spatial_invariance(self):
        m = nn.Upsample(scale_factor=3, mode='linear', align_corners=False)
        in_t_9 = torch.zeros(1, 1, 9)
        in_t_9[:, :, :4].normal_()
        out_t_9 = m(in_t_9)
        out_t_5 = m(in_t_9[:, :, :5])
        self.assertEqual(out_t_9[:, :, :15], out_t_5)

    def test_upsamplingNearest2d(self):
        m = nn.Upsample(size=4, mode='nearest')
        in_t = torch.ones(1, 1, 2, 2)
        out_t = m(Variable(in_t))
        self.assertEqual(torch.ones(1, 1, 4, 4), out_t.data)

        input = Variable(torch.randn(1, 1, 2, 2), requires_grad=True)
        self.assertEqual(
            F.upsample(input, 4, mode='nearest'),
            F.upsample(input, scale_factor=2, mode='nearest'))
        gradcheck(lambda x: F.upsample(x, 4, mode='nearest'), [input])
        gradgradcheck(lambda x: F.upsample(x, 4, mode='nearest'), [input])

    def test_upsamplingBilinear2d(self):
        m = nn.Upsample(size=4, mode='bilinear')
        in_t = torch.ones(1, 1, 2, 2)
        out_t = m(Variable(in_t))
        self.assertEqual(torch.ones(1, 1, 4, 4), out_t.data)

        input = Variable(torch.randn(1, 1, 2, 2), requires_grad=True)
        gradcheck(lambda x: F.upsample(x, 4, mode='bilinear'), [input])

    def test_upsamplingBilinear2d_spatial_invariance(self):
        m = nn.Upsample(scale_factor=3, mode='bilinear', align_corners=False)
        in_t_9 = torch.zeros(1, 1, 9, 9)
        in_t_9[:, :, :4, :4].normal_()
        out_t_9 = m(in_t_9)
        out_t_5 = m(in_t_9[:, :, :5, :5])
        self.assertEqual(out_t_9[:, :, :15, :15], out_t_5)

    def test_upsamplingNearest3d(self):
        m = nn.Upsample(size=4, mode='nearest')
        in_t = torch.ones(1, 1, 2, 2, 2)
        out_t = m(Variable(in_t))
        self.assertEqual(torch.ones(1, 1, 4, 4, 4), out_t.data)

        input = Variable(torch.randn(1, 1, 2, 2, 2), requires_grad=True)
        gradcheck(lambda x: F.upsample(x, 4, mode='nearest'), [input])

    def test_upsamplingTrilinear3d(self):
        m = nn.Upsample(size=4, mode='trilinear')
        in_t = torch.ones(1, 1, 2, 2, 2)
        out_t = m(Variable(in_t))
        self.assertEqual(torch.ones(1, 1, 4, 4, 4), out_t.data)

        input = Variable(torch.randn(1, 1, 2, 2, 2), requires_grad=True)
        self.assertEqual(
            F.upsample(input, (4, 4, 4), mode='trilinear'),
            F.upsample(input, scale_factor=2, mode='trilinear'))
        gradcheck(lambda x: F.upsample(x, 4, mode='trilinear'), [input])
        gradgradcheck(lambda x: F.upsample(x, 4, mode='trilinear'), [input])

    def test_upsamplingTrilinear3d_spatial_invariance(self):
        m = nn.Upsample(scale_factor=3, mode='trilinear', align_corners=False)
        in_t_9 = torch.zeros(1, 1, 9, 9, 9)
        in_t_9[:, :, :4, :4, :4].normal_()
        out_t_9 = m(in_t_9)
        out_t_5 = m(in_t_9[:, :, :5, :5, :5])
        self.assertEqual(out_t_9[:, :, :15, :15, :15], out_t_5)

    def test_linear_broadcasting(self):
        m = nn.Linear(5, 8)
        inp = Variable(torch.randn(2, 3, 5))
        expected = m(inp.view(6, 5)).view(2, 3, 8)
        self.assertEqual(expected, m(inp))

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

        _assertGradAndGradgradChecks(self, lambda x1, x2: F.bilinear(x1, x2, module.weight, module.bias),
                                     (input1_1, input2_1))

    def test_bilinear_no_bias(self):
        module = nn.Bilinear(10, 10, 8)
        module_no_bias = nn.Bilinear(10, 10, 8, False)

        module.bias.data.zero_()
        module.weight.data.copy_(module_no_bias.weight)

        input1 = torch.randn(4, 10, requires_grad=True)
        input2 = torch.randn(4, 10, requires_grad=True)
        grad_output = torch.randn(4, 8)

        def run(net):
            input1.grad = input2.grad = None
            output = net(input1, input2)
            output.backward(grad_output)

            return output.data, input1.grad.data, input2.grad.data

        out, g1, g2 = run(module)
        out_nb, g1_nb, g2_nb = run(module_no_bias)

        self.assertEqual(out, out_nb)
        self.assertEqual(g1, g1_nb)
        self.assertEqual(g2, g2_nb)

        _assertGradAndGradgradChecks(self,
                                     lambda x1, x2: F.bilinear(x1, x2, module_no_bias.weight, module_no_bias.bias),
                                     (input1, input2))

    def test_bilinear_broadcasting(self):
        m = nn.Bilinear(5, 6, 8)
        input1 = torch.randn(2, 3, 5)
        input2 = torch.randn(2, 3, 6)
        expected = m(input1.view(6, 5), input2.view(6, 6)).view(2, 3, 8)
        self.assertEqual(expected, m(input1, input2))

    def test_conv_tbc(self):
        inp = Variable(torch.randn(9, 4, 5), requires_grad=True)
        weight = Variable(torch.randn(3, 5, 6), requires_grad=True)
        bias = Variable(torch.randn(6), requires_grad=True)

        gradcheck(lambda i, w, b, pad: F.conv_tbc(i, w, b, pad), (inp, weight, bias, 3))

    def run_conv_double_back_test(self, kern, stride, padding, chan_in, chan_out, batch_size,
                                  inp_size, dilation, no_weight, groups=1, use_cuda=False,
                                  use_bias=True, dtype=torch.DoubleTensor):
        tensor = torch.Tensor(1)
        if use_cuda:
            tensor = tensor.cuda()

        x = Variable(tensor.new(batch_size, chan_in, inp_size, inp_size).type(dtype), requires_grad=True)
        x.data.normal_()
        weight = Variable(tensor.new(chan_out, chan_in // groups, kern, kern).type(dtype), requires_grad=not no_weight)
        weight.data.normal_()
        if use_bias:
            bias = Variable(tensor.new(chan_out).type(dtype), requires_grad=True)
            bias.data.normal_()
        else:
            bias = None

        def func(*inputs):
            if use_bias:
                lx, lweight, lbias = inputs
            else:
                lx, lweight = inputs
                lbias = None
            # We disable cudnn during forward to avoid finite difference imprecision issues
            with cudnn.flags(enabled=False):
                out = F.conv2d(lx, lweight, lbias, stride, padding, dilation, groups)
            return out

        if use_bias:
            inputs = x, weight, bias
        else:
            inputs = x, weight

        dummy_out = func(*inputs)
        grad_y = Variable(tensor.new(dummy_out.size()).type(dtype), requires_grad=True)
        grad_y.data.normal_()

        return gradgradcheck(func, inputs, (grad_y,))

    def test_conv_double_backward(self):
        batch_size = 2
        for kern, inp_size, dilations in [(3, 6, [1, 2]), (3, 7, [1]), (4, 9, [1])]:
            for stride, padding, chan_in, chan_out, dilation in \
                    product([1, 2], [0, 1, 2], [2], [3], dilations):
                for no_weight in (True, False):
                    result = self.run_conv_double_back_test(kern, stride,
                                                            padding, chan_in, chan_out,
                                                            batch_size, inp_size, dilation,
                                                            no_weight)
                    self.assertTrue(result,
                                    "Conv double backward test failed with parameters:" +
                                    "\nkern: " + str(kern) +
                                    "\nstride: " + str(stride) +
                                    "\npadding: " + str(padding) +
                                    "\nchan_in: " + str(chan_in) +
                                    "\nchan_out: " + str(chan_out) +
                                    "\nbatch_size: " + str(batch_size) +
                                    "\ninp_size: " + str(inp_size) +
                                    "\ndilation: " + str(dilation))

    def test_conv_double_backward_no_bias(self):
        kern = 3
        stride = 2
        chan_in, chan_out = 2, 4
        batch_size = 2
        inp_size = 5
        padding = 1
        dilation = 1
        no_weight = False
        use_bias = True
        result = self.run_conv_double_back_test(kern, stride,
                                                padding, chan_in, chan_out,
                                                batch_size, inp_size, dilation,
                                                no_weight, use_bias=use_bias)
        self.assertTrue(result,
                        "Conv double backward test failed with parameters:" +
                        "\nkern: " + str(kern) +
                        "\nstride: " + str(stride) +
                        "\npadding: " + str(padding) +
                        "\nchan_in: " + str(chan_in) +
                        "\nchan_out: " + str(chan_out) +
                        "\nbatch_size: " + str(batch_size) +
                        "\ninp_size: " + str(inp_size) +
                        "\ndilation: " + str(dilation))

    def test_conv_double_backward_groups(self):
        kern = 3
        stride = 1
        padding = 2
        chan_in, chan_out = 2, 4
        batch_size = 2
        inp_size = 6
        dilation = 1
        no_weight = False
        groups = 2
        result = self.run_conv_double_back_test(kern, stride,
                                                padding, chan_in * groups, chan_out * groups,
                                                batch_size, inp_size, dilation,
                                                no_weight, groups=groups)
        self.assertTrue(result,
                        "Conv double backward test failed with parameters:" +
                        "\nkern: " + str(kern) +
                        "\nstride: " + str(stride) +
                        "\npadding: " + str(padding) +
                        "\nchan_in: " + str(chan_in) +
                        "\nchan_out: " + str(chan_out) +
                        "\nbatch_size: " + str(batch_size) +
                        "\ninp_size: " + str(inp_size) +
                        "\ndilation: " + str(dilation) +
                        "\ngroups: " + str(groups))

    def test_conv_double_backward_stride(self):
        batch_size = 2

        # Cannot provide ggW when stride is > 1
        for kern, inp_size, dilations in [(3, 5, [1, 2]), (3, 7, [1])]:
            for stride, padding, chan_in, chan_out, dilation in product([2], [0, 1], [1], [2], dilations):
                no_weight = False
                self.run_conv_double_back_test(kern, stride,
                                               padding, chan_in, chan_out,
                                               batch_size, inp_size, dilation,
                                               no_weight)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_cudnn_noncontiguous_weight(self):
        # Noncontiguous weights must be contiguous() before being
        # passed to cuDNN
        input = Variable(torch.cuda.DoubleTensor([1, 1, 1]).view(1, 1, 3))
        weights1 = Variable(torch.cuda.DoubleTensor([1]).expand(1, 1, 2))
        weights2 = Variable(torch.cuda.DoubleTensor([1]).expand(1, 1, 2)).contiguous()
        self.assertEqual(F.conv1d(input, weights1, bias=None, stride=2, dilation=2),
                         F.conv1d(input, weights2, bias=None, stride=2, dilation=2))

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    @repeat_test_for_types(DOUBLE_TENSORTYPES)
    def test_conv_double_backward_cuda(self, dtype=torch.FloatTensor):
        # Double backward only runs with DoubleTensor due to precison reason
        dtype = getattr(torch.cuda, dtype.__name__)
        batch_size = 1
        for kern, inp_size, dilations in [(3, 5, [1, 2]), (4, 9, [1])]:
            for stride, padding, chan_in, chan_out, dilation in product([1], [2], [2], [3], dilations):
                no_weight = stride == 2
                result = self.run_conv_double_back_test(kern, stride,
                                                        padding, chan_in, chan_out,
                                                        batch_size, inp_size, dilation,
                                                        no_weight, use_cuda=True, dtype=dtype)
                self.assertTrue(result,
                                "Conv double backward test failed with parameters:" +
                                "\nkern: " + str(kern) +
                                "\nstride: " + str(stride) +
                                "\npadding: " + str(padding) +
                                "\nchan_in: " + str(chan_in) +
                                "\nchan_out: " + str(chan_out) +
                                "\nbatch_size: " + str(batch_size) +
                                "\ninp_size: " + str(inp_size) +
                                "\ndilation: " + str(dilation))

    def run_grad_conv_test(self, func_forward, func_backward, dim=1, gradient='input'):
        for kern, inp_size in [(3, 6), (3, 7), (4, 9)]:
            for batch, stride, padding, chan_in, chan_out, dilation in \
                    product([1, 2], [1, 2], [0, 1, 2], [2], [3], [1]):

                input_shape = [batch, chan_in]
                weight_shape = [chan_out, chan_in]
                for _ in range(dim):
                    input_shape.append(inp_size)
                    weight_shape.append(kern)

                input = torch.randn(input_shape, requires_grad=True)
                weight = torch.randn(weight_shape, requires_grad=True)
                output = func_forward(input, weight, stride=stride, padding=padding, dilation=dilation)

                gradient_o = torch.randn(output.shape)
                gradient_w = torch.autograd.grad(output, input if (gradient == 'input') else weight, gradient_o)

                self.assertAlmostEqual(gradient_w[0],
                                       func_backward(
                                           input_shape if (gradient == 'input') else input,
                                           weight_shape if (gradient == 'weight') else weight,
                                           gradient_o,
                                           stride=stride,
                                           padding=padding,
                                           dilation=dilation))

    def test_grad_conv1d_input(self):
        self.run_grad_conv_test(F.conv1d, F.grad.conv1d_input, 1, 'input')

    def test_grad_conv1d_weight(self):
        self.run_grad_conv_test(F.conv1d, F.grad.conv1d_weight, 1, 'weight')

    def test_grad_conv2d_input(self):
        self.run_grad_conv_test(F.conv2d, F.grad.conv2d_input, 2, 'input')

    def test_grad_conv2d_weight(self):
        self.run_grad_conv_test(F.conv2d, F.grad.conv2d_weight, 2, 'weight')

    def test_grad_conv3d_input(self):
        self.run_grad_conv_test(F.conv3d, F.grad.conv3d_input, 3, 'input')

    def test_grad_conv3d_weight(self):
        self.run_grad_conv_test(F.conv3d, F.grad.conv3d_weight, 3, 'weight')


class TestNNInit(TestCase):
    def setUp(self):
        super(TestNNInit, self).setUp()
        random.seed(123)

    def _is_normal(self, tensor, mean, std):
        samples = tensor.view(-1).tolist()
        p_value = stats.kstest(samples, 'norm', args=(mean, std))[1]
        return p_value > 0.0001

    def _is_uniform(self, tensor, a, b):
        samples = tensor.view(-1).tolist()
        p_value = stats.kstest(samples, 'uniform', args=(a, (b - a)))[1]
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
                init.uniform_(input_tensor, a=a, b=b)
                assert self._is_uniform(input_tensor, a, b)

    @unittest.skipIf(not TEST_SCIPY, "Scipy not found.")
    def test_normal(self):
        for as_variable in [True, False]:
            for dims in [1, 2, 4]:
                input_tensor = self._create_random_nd_tensor(dims, size_min=30, size_max=50, as_variable=as_variable)
                mean = self._random_float(-3, 3)
                std = self._random_float(1, 5)
                init.normal_(input_tensor, mean=mean, std=std)

                assert self._is_normal(input_tensor, mean, std)

    def test_constant(self):
        for as_variable in [True, False]:
            for dims in [1, 2, 4]:
                input_tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=5, as_variable=as_variable)
                val = self._random_float(1, 10)
                init.constant_(input_tensor, val)
                if as_variable:
                    input_tensor = input_tensor.data

                self.assertEqual(input_tensor, input_tensor.clone().fill_(val))

    def test_eye(self):
        for as_variable in [True, False]:
            input_tensor = self._create_random_nd_tensor(2, size_min=1, size_max=5, as_variable=as_variable)
            init.eye_(input_tensor)
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
                    init.eye_(tensor)

    def test_dirac_properties(self):
        for as_variable in [True, False]:
            for dims in [3, 4, 5]:
                input_tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=5, as_variable=as_variable)
                init.dirac_(input_tensor)
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
        init.dirac_(filter_var)
        output_var = F.conv1d(input_var, filter_var)
        input_tensor, output_tensor = input_var.data, output_var.data  # Variables do not support nonzero
        self.assertEqual(input_tensor[:, :, 1:-1], output_tensor[:, :in_c, :])  # Assert in_c outputs are preserved
        assert torch.nonzero(output_tensor[:, in_c:, :]).numel() == 0  # Assert extra outputs are 0

        # Test 2D
        input_var = Variable(torch.randn(batch, in_c, size, size))
        filter_var = Variable(torch.zeros(out_c, in_c, kernel_size, kernel_size))
        init.dirac_(filter_var)
        output_var = F.conv2d(input_var, filter_var)
        input_tensor, output_tensor = input_var.data, output_var.data
        self.assertEqual(input_tensor[:, :, 1:-1, 1:-1], output_tensor[:, :in_c, :, :])
        assert torch.nonzero(output_tensor[:, in_c:, :, :]).numel() == 0

        # Test 3D
        input_var = Variable(torch.randn(batch, in_c, size, size, size))
        filter_var = Variable(torch.zeros(out_c, in_c, kernel_size, kernel_size, kernel_size))
        init.dirac_(filter_var)
        output_var = F.conv3d(input_var, filter_var)
        input_tensor, output_tensor = input_var.data, output_var.data
        self.assertEqual(input_tensor[:, :, 1:-1, 1:-1, 1:-1], output_tensor[:, :in_c, :, :])
        assert torch.nonzero(output_tensor[:, in_c:, :, :, :]).numel() == 0

    def test_dirac_only_works_on_3_4_5d_inputs(self):
        for as_variable in [True, False]:
            for dims in [1, 2, 6]:
                with self.assertRaises(ValueError):
                    tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=3, as_variable=as_variable)
                    init.dirac_(tensor)

    def test_xavier_uniform_errors_on_inputs_smaller_than_2d(self):
        for as_variable in [True, False]:
            for dims in [0, 1]:
                tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=1, as_variable=as_variable)
                with self.assertRaises(ValueError):
                    init.xavier_uniform_(tensor)

    def test_xavier_normal_errors_on_inputs_smaller_than_2d(self):
        for as_variable in [True, False]:
            for dims in [0, 1]:
                tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=1, as_variable=as_variable)
                with self.assertRaises(ValueError):
                    init.xavier_normal_(tensor)

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
                        init.xavier_uniform_(input_tensor, gain=gain)
                    else:
                        init.xavier_uniform_(input_tensor)

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
                        init.xavier_normal_(input_tensor, gain=gain)
                    else:
                        init.xavier_normal_(input_tensor)

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
                    init.kaiming_uniform_(tensor)

    def test_kaiming_normal_errors_on_inputs_smaller_than_2d(self):
        for as_variable in [True, False]:
            for dims in [0, 1]:
                with self.assertRaises(ValueError):
                    tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=1, as_variable=as_variable)
                    init.kaiming_normal_(tensor)

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
                            init.kaiming_uniform_(input_tensor, a=a, mode=mode)
                        else:
                            a = 0
                            init.kaiming_uniform_(input_tensor, mode=mode)

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
                            init.kaiming_normal_(input_tensor, a=a, mode=mode)
                        else:
                            a = 0
                            init.kaiming_normal_(input_tensor, mode=mode)

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
                    init.sparse_(tensor, sparsity)

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
                    init.sparse_(input_tensor, sparsity=sparsity, std=std)
                else:
                    init.sparse_(input_tensor, sparsity=sparsity)

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
                        init.orthogonal_(input_tensor, gain=gain)
                    else:
                        init.orthogonal_(input_tensor)

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

    def test_deprecation(self):
        x = torch.randn(3, 3)

        def fn():
            init.normal(x)
        self.assertWarnsRegex(fn, 'deprecated', 'methods not suffixed with underscore should be deprecated')


# Generates rand tensor with non-equal values. This ensures that duplicate
# values won't be causing test failure for modules like MaxPooling.
# size should be small, otherwise randperm fails / long overflows.
def _rand_tensor_non_equal(*size):
    total = reduce(mul, size, 1)
    return torch.randperm(total).view(*size).double()


def add_test(test):
    test_name = test.get_name()
    cuda_test_name = test_name + '_cuda'
    if hasattr(TestNN, test_name):
        raise RuntimeError('Found two tests with the same name: ' + test_name)
    if hasattr(TestNN, cuda_test_name):
        raise RuntimeError('Found two tests with the same name: ' + cuda_test_name)
    setattr(TestNN, test_name, lambda self, test=test: test(self))
    # Hardshrink is not implemented in CUDA, so we must not test it.
    if not test_name.startswith("test_Hardshrink"):
        # With dtype enable, it's good enough to test against three floating types
        if 'dtype' in get_function_arglist(test.test_cuda):
            setattr(TestNN, cuda_test_name + '_float', lambda self,
                    test=test: test.test_cuda(self, dtype=torch.FloatTensor))
            setattr(TestNN, cuda_test_name + '_double', lambda self,
                    test=test: test.test_cuda(self, dtype=torch.DoubleTensor))
            setattr(TestNN, cuda_test_name + '_half', lambda self,
                    test=test: test.test_cuda(self, dtype=torch.HalfTensor))
        else:
            setattr(TestNN, cuda_test_name, lambda self, test=test: test.test_cuda(self))


def wrap_functional(fn, **kwargs):
    class FunctionalModule(nn.Module):
        def forward(self, *args):
            return fn(*args, **kwargs)
    return FunctionalModule


new_criterion_tests = [
    dict(
        module_name='BCEWithLogitsLoss',
        input_fn=lambda: torch.rand(15, 10).clamp_(1e-2, 1 - 1e-2),
        target_fn=lambda: torch.randn(15, 10).gt(0).double()
    ),
    dict(
        module_name='BCEWithLogitsLoss',
        constructor_args=(torch.rand(10),),
        input_fn=lambda: torch.rand(15, 10).clamp_(1e-2, 1 - 1e-2),
        target_fn=lambda: torch.randn(15, 10).gt(0).double(),
        desc='weights'
    ),
    dict(
        module_name='BCEWithLogitsLoss',
        constructor_args=(torch.rand(()),),
        input_fn=lambda: torch.rand(()).clamp_(1e-2, 1 - 1e-2),
        target_fn=lambda: torch.randn(()).gt(0).double(),
        desc='scalar_weights'
    ),
    dict(
        module_name='NLLLoss',
        input_size=(2, 3, 5, 5),
        target_fn=lambda: torch.rand(2, 5, 5).mul(3).floor().long(),
        reference_fn=lambda i, t, m:
            loss_reference_fns['NLLLossNd'](i, t, size_average=get_size_average(m)),
        check_no_size_average=True,
        desc='2d'
    ),
    dict(
        module_name='NLLLoss',
        constructor_args_fn=lambda: (torch.rand(3),),
        input_size=(2, 3, 5, 5),
        target=torch.rand(2, 5, 5).mul(3).floor().long(),
        reference_fn=lambda i, t, m:
            loss_reference_fns['NLLLossNd'](i, t, weight=get_weight(m)),
        desc='2d_weights',
    ),
    dict(
        module_name='NLLLoss',
        constructor_args=(None, True, 1),
        input_size=(2, 3, 5, 5),
        target_fn=lambda: torch.rand(2, 5, 5).mul(3).floor().long(),
        reference_fn=lambda i, t, m:
            loss_reference_fns['NLLLossNd'](i, t, ignore_index=1),
        desc='2d_ignore_index',
    ),
    dict(
        module_name='NLLLoss',
        input_size=(2, 3, 5, 5, 2, 2),
        target_fn=lambda: torch.rand(2, 5, 5, 2, 2).mul(3).floor().long(),
        reference_fn=lambda i, t, m:
            loss_reference_fns['NLLLossNd'](i, t, size_average=get_size_average(m)),
        check_no_size_average=True,
        desc='higher_dim'
    ),
    dict(
        module_name='NLLLoss',
        input_size=(2, 3, 5),
        target_fn=lambda: torch.rand(2, 5).mul(3).floor().long(),
        reference_fn=lambda i, t, m:
            loss_reference_fns['NLLLossNd'](i, t, size_average=get_size_average(m)),
        check_no_size_average=True,
        desc='dim_is_3'
    ),
    dict(
        module_name='PoissonNLLLoss',
        input_size=(2, 3, 4, 5),
        target_fn=lambda: torch.randn(2, 3, 4, 5).floor_().abs_(),
        desc='no_full_loss',  # without sterling approx
    ),
    dict(
        module_name='PoissonNLLLoss',
        constructor_args=(False, True, True),
        input_fn=lambda: torch.randn(2, 3, 4, 5).abs_().add_(0.001),
        target_fn=lambda: torch.randn(2, 3, 4, 5).floor_().abs_(),
        desc='full_loss',  # with sterling approx
    ),
    dict(
        module_name='L1Loss',
        input_size=(),
        target_size=(),
        reference_fn=lambda i, t, _: 1. / i.numel() * (i - t).abs().sum(),
        desc='scalar',
    ),
    dict(
        module_name='KLDivLoss',
        input_fn=lambda: torch.rand(()).log(),
        target_fn=lambda: torch.rand(()),
        reference_fn=lambda i, t, m:
            kldivloss_reference(i, t, get_size_average(m), reduce=True),
        check_no_size_average=True,
        desc='scalar',
    ),
    dict(
        module_name='MSELoss',
        input_size=(),
        target_size=(),
        reference_fn=lambda i, t, m: (i - t).abs().pow(2).sum() / (i.numel() if get_size_average(m) else 1),
        check_no_size_average=True,
        desc='scalar'
    ),
    dict(
        module_name='BCELoss',
        constructor_args_fn=lambda: (torch.rand(()),),
        input_fn=lambda: torch.rand(()).clamp_(1e-2, 1 - 1e-2),
        target_fn=lambda: torch.rand(()).gt(0).double(),
        reference_fn=lambda i, t, m: -((t * i.log() + (1 - t) * (1 - i).log()) * get_weight(m)).sum() /
            (i.numel() if get_size_average(m) else 1),
        desc='scalar_weights',
        check_gradgrad=False,
    ),
    dict(
        module_name='HingeEmbeddingLoss',
        constructor_args=(0.5,),
        input_size=(),
        target_fn=lambda: torch.randn(()).gt(0).double().mul_(2).sub(1),
        desc='scalar_margin',
        check_no_size_average=True,
    ),
    dict(
        module_name='SmoothL1Loss',
        input_size=(),
        target_size=(),
        check_no_size_average=True,
        reference_fn=lambda i, t, m:
            smoothl1loss_reference(i, t, size_average=get_size_average(m)),
        desc='scalar',
    ),
    dict(
        module_name='MultiLabelSoftMarginLoss',
        constructor_args=(torch.rand(10),),
        input_fn=lambda: torch.randn(5, 10),
        target_fn=lambda: torch.rand(5, 10).mul(2).floor(),
        reference_fn=lambda i, t, m: -((t * i.sigmoid().log() + (1 - t) * (-i).sigmoid().log()) * get_weight(m)).sum() /
            (i.numel() if get_size_average(m) else 1),
        desc='weights',
        check_no_size_average=True,
        check_gradgrad=False,
    ),
]


def poissonnllloss_no_reduce_test():
    t = Variable(torch.randn(10, 10))
    return dict(
        fullname='PoissonNLLLLoss_no_reduce',
        constructor=wrap_functional(
            lambda i: F.poisson_nll_loss(i, t.type_as(i), reduce=False)),
        input_fn=lambda: torch.rand(10, 10),
        pickle=False)


def bceloss_no_reduce_test():
    t = Variable(torch.randn(15, 10).gt(0).double())
    return dict(
        fullname='BCELoss_no_reduce',
        constructor=wrap_functional(
            lambda i: F.binary_cross_entropy(i, t.type_as(i), reduce=False)),
        input_fn=lambda: torch.rand(15, 10).clamp_(2.8e-2, 1 - 2.8e-2),
        reference_fn=lambda i, m: -(t * i.log() + (1 - t) * (1 - i).log()),
        check_gradgrad=False,
        pickle=False)


def bceloss_no_reduce_scalar_test():
    t = torch.randn(()).gt(0).double()
    return dict(
        fullname='BCELoss_no_reduce_scalar',
        constructor=wrap_functional(
            lambda i: F.binary_cross_entropy(i, t.type_as(i), reduce=False)),
        input_fn=lambda: torch.rand(()).clamp_(2.8e-2, 1 - 2.8e-2),
        reference_fn=lambda i, m: -(t * i.log() + (1 - t) * (1 - i).log()),
        check_gradgrad=False,
        pickle=False)


def bceloss_weights_no_reduce_test():
    t = Variable(torch.randn(15, 10).gt(0).double())
    weights = Variable(torch.rand(10))
    return dict(
        fullname='BCELoss_weights_no_reduce',
        constructor=wrap_functional(
            lambda i: F.binary_cross_entropy(i, t.type_as(i),
                                             weight=weights.type_as(i), reduce=False)),
        input_fn=lambda: torch.rand(15, 10).clamp_(2.8e-2, 1 - 2.8e-2),
        reference_fn=lambda i, m: -(t * i.log() + (1 - t) * (1 - i).log()) * weights,
        check_gradgrad=False,
        pickle=False)


def bceloss_weights_no_reduce_scalar_test():
    t = torch.randn(()).double()
    weights = torch.rand(())
    return dict(
        fullname='BCELoss_weights_no_reduce_scalar',
        constructor=wrap_functional(
            lambda i: F.binary_cross_entropy(i, t.type_as(i),
                                             weight=weights.type_as(i), reduce=False)),
        input_fn=lambda: torch.rand(()).clamp_(2.8e-2, 1 - 2.8e-2),
        reference_fn=lambda i, m: -(t * i.log() + (1 - t) * (1 - i).log()) * weights,
        check_gradgrad=False,
        pickle=False)


def bce_with_logistic_no_reduce_test():
    t = Variable(torch.randn(15, 10).gt(0).double())
    sigmoid = nn.Sigmoid()
    return dict(
        fullname='BCEWithLogitsLoss_no_reduce',
        constructor=wrap_functional(
            lambda i: F.binary_cross_entropy_with_logits(i, t.type_as(i), reduce=False)),
        input_fn=lambda: torch.rand(15, 10).clamp_(2.8e-2, 1 - 2.8e-2),
        reference_fn=lambda i, m: -(t * sigmoid(i).log() + (1 - t) * (1 - sigmoid(i)).log()),
        check_gradgrad=False,
        pickle=False)


def bce_with_logistic_no_reduce_scalar_test():
    t = torch.randn(()).gt(0).double()
    sigmoid = nn.Sigmoid()
    return dict(
        fullname='BCEWithLogitsLoss_no_reduce_scalar',
        constructor=wrap_functional(
            lambda i: F.binary_cross_entropy_with_logits(i, t.type_as(i), reduce=False)),
        input_fn=lambda: torch.rand(()).clamp_(2.8e-2, 1 - 2.8e-2),
        reference_fn=lambda i, m: -(t * sigmoid(i).log() + (1 - t) * (1 - sigmoid(i)).log()),
        check_gradgrad=False,
        pickle=False)


def kldivloss_no_reduce_test():
    t = Variable(torch.randn(10, 10))
    return dict(
        fullname='KLDivLoss_no_reduce',
        constructor=wrap_functional(
            lambda i: F.kl_div(i, t.type_as(i), reduce=False)),
        input_fn=lambda: torch.rand(10, 10).log(),
        reference_fn=lambda i, _:
            loss_reference_fns['KLDivLoss'](i, t.type_as(i), reduce=False),
        pickle=False)


def kldivloss_no_reduce_scalar_test():
    t = torch.randn(())
    return dict(
        fullname='KLDivLoss_no_reduce_scalar',
        constructor=wrap_functional(
            lambda i: F.kl_div(i, t.type_as(i), reduce=False)),
        input_fn=lambda: torch.rand(()).log(),
        reference_fn=lambda i, _:
            loss_reference_fns['KLDivLoss'](i, t.type_as(i), reduce=False),
        pickle=False)


def l1loss_no_reduce_test():
    t = Variable(torch.randn(2, 3, 4))
    return dict(
        fullname='L1Loss_no_reduce',
        constructor=wrap_functional(
            lambda i: F.l1_loss(i, t.type_as(i), reduce=False)),
        input_fn=lambda: torch.randn(2, 3, 4),
        reference_fn=lambda i, m: (i - t.type_as(i)).abs(),
        pickle=False)


def l1loss_no_reduce_scalar_test():
    t = torch.randn(())
    return dict(
        fullname='L1Loss_no_reduce_scalar',
        constructor=wrap_functional(
            lambda i: F.l1_loss(i, t.type_as(i), reduce=False)),
        input_fn=lambda: torch.randn(()),
        reference_fn=lambda i, m: (i - t.type_as(i)).abs(),
        pickle=False)


def mseloss_no_reduce_test():
    input_size = (2, 3, 4, 5)
    target = Variable(torch.randn(*input_size))
    return dict(
        fullname='MSELoss_no_reduce',
        constructor=wrap_functional(
            lambda i: F.mse_loss(i, target.type_as(i), reduce=False)),
        input_size=input_size,
        reference_fn=lambda i, m: (i - target).pow(2),
        pickle=False)


def mseloss_no_reduce_scalar_test():
    input_size = ()
    target = torch.randn(input_size)
    return dict(
        fullname='MSELoss_no_reduce_scalar',
        constructor=wrap_functional(
            lambda i: F.mse_loss(i, target.type_as(i), reduce=False)),
        input_size=input_size,
        reference_fn=lambda i, m: (i - target).pow(2),
        pickle=False)


def nllloss_no_reduce_test():
    t = Variable(torch.Tensor(15).uniform_().mul(10).floor().long())
    kwargs = {'reduce': False}
    return dict(
        fullname='NLLLoss_no_reduce',
        constructor=wrap_functional(
            lambda i: F.nll_loss(i, t.type_as(i).long(), **kwargs)),
        input_fn=lambda: torch.rand(15, 10).log(),
        reference_fn=lambda i, _:
            loss_reference_fns['NLLLoss'](i, t.type_as(i).long(), **kwargs),
        pickle=False)


def nllloss_no_reduce_ignore_index_test():
    t = Variable(torch.Tensor(15).uniform_().mul(10).floor().long())
    kwargs = {'ignore_index': 2, 'reduce': False}
    return dict(
        fullname='NLLLoss_no_reduce_ignore_index',
        constructor=wrap_functional(
            lambda i: F.nll_loss(i, t.type_as(i).long(), **kwargs)),
        input_fn=lambda: torch.rand(15, 10).log(),
        reference_fn=lambda i, _:
            loss_reference_fns['NLLLoss'](i, t.type_as(i).long(), **kwargs),
        pickle=False)


def nllloss_no_reduce_weights_test():
    t = Variable(torch.Tensor(15).uniform_().mul(10).floor().long())
    weight = Variable(torch.rand(10))

    def kwargs(i):
        return {'weight': weight.type_as(i), 'reduce': False}

    return dict(
        fullname='NLLLoss_no_reduce_weights',
        constructor=wrap_functional(
            lambda i: F.nll_loss(i, t.type_as(i).long(), **kwargs(i))),
        input_fn=lambda: torch.rand(15, 10).add(1e-2).log(),
        reference_fn=lambda i, _:
            loss_reference_fns['NLLLoss'](i, t.type_as(i).long(), **kwargs(i)),
        pickle=False)


def nllloss_no_reduce_weights_ignore_index_test():
    t = Variable(torch.Tensor(15).uniform_().mul(10).floor().long())
    weight = Variable(torch.rand(10))

    def kwargs(i):
        return {'weight': weight.type_as(i), 'reduce': False,
                'ignore_index': 2}

    return dict(
        fullname='NLLLoss_no_reduce_weights_ignore_index',
        constructor=wrap_functional(
            lambda i: F.nll_loss(i, t.type_as(i).long(), **kwargs(i.data))),
        input_fn=lambda: torch.rand(15, 10).add(1e-2).log(),
        reference_fn=lambda i, _:
            loss_reference_fns['NLLLoss'](i, t.type_as(i).long(), **kwargs(i)),
        pickle=False)


def nllloss_no_reduce_weights_ignore_index_neg_test():
    t = Variable(torch.Tensor(15).uniform_().mul(10).floor().long())
    weight = Variable(torch.rand(10))

    def kwargs(i):
        return {'weight': weight.type_as(i), 'reduce': False,
                'ignore_index': -1}

    return dict(
        fullname='NLLLoss_no_reduce_weights_ignore_index_neg',
        constructor=wrap_functional(
            lambda i: F.nll_loss(i, t.type_as(i).long(), **kwargs(i))),
        input=torch.rand(15, 10).add(1e-2).log(),
        reference_fn=lambda i, _:
            loss_reference_fns['NLLLoss'](i, t.type_as(i).long(), **kwargs(i)),
        pickle=False)


def nllloss2d_no_reduce_test():
    t = Variable(torch.rand(2, 5, 5).mul(3).floor().long())
    kwargs = {'reduce': False}
    return dict(
        fullname='NLLLoss2d_no_reduce',
        constructor=wrap_functional(
            lambda i: F.nll_loss(i, t.type_as(i).long(), **kwargs)),
        input_fn=lambda: torch.rand(2, 3, 5, 5).log(),
        reference_fn=lambda i, _:
            loss_reference_fns['NLLLossNd'](i, t.type_as(i).long(), **kwargs),
        pickle=False)


def nllloss2d_no_reduce_ignore_index_test():
    t = Variable(torch.rand(2, 5, 5).mul(3).floor().long())
    kwargs = {'ignore_index': 1, 'reduce': False}
    return dict(
        fullname='NLLLoss2d_no_reduce_ignore_index',
        constructor=wrap_functional(
            lambda i: F.nll_loss(i, t.type_as(i).long(), **kwargs)),
        input_fn=lambda: torch.rand(2, 3, 5, 5).log(),
        reference_fn=lambda i, _:
            loss_reference_fns['NLLLossNd'](i, t.type_as(i).long(), **kwargs),
        pickle=False)


def nllloss2d_no_reduce_weights_test():
    t = Variable(torch.rand(2, 5, 5).mul(3).floor().long())
    weight = torch.rand(3)

    def kwargs(i):
        return {'weight': weight.type_as(i), 'reduce': False}

    return dict(
        fullname='NLLLoss2d_no_reduce_weights',
        constructor=wrap_functional(
            lambda i: F.nll_loss(i, t.type_as(i).long(), **kwargs(i))),
        input_fn=lambda: torch.rand(2, 3, 5, 5).log(),
        reference_fn=lambda i, _:
            loss_reference_fns['NLLLossNd'](i, t.type_as(i).long(), **kwargs(i)),
        pickle=False)


def nlllossNd_no_reduce_test():
    t = Variable(torch.rand(2, 5, 5, 2, 2).mul(3).floor().long())
    kwargs = {'reduce': False}
    return dict(
        fullname='NLLLossNd_no_reduce',
        constructor=wrap_functional(
            lambda i: F.nll_loss(i, t.type_as(i).long(), **kwargs)),
        input_fn=lambda: torch.rand(2, 3, 5, 5, 2, 2).log(),
        reference_fn=lambda i, _:
            loss_reference_fns['NLLLossNd'](i, t.type_as(i).long(), **kwargs),
        pickle=False)


def nlllossNd_no_reduce_ignore_index_test():
    t = Variable(torch.rand(2, 5, 5, 2, 2).mul(3).floor().long())
    kwargs = {'ignore_index': 1, 'reduce': False}
    return dict(
        fullname='NLLLossNd_no_reduce_ignore_index',
        constructor=wrap_functional(
            lambda i: F.nll_loss(i, t.type_as(i).long(), **kwargs)),
        input_fn=lambda: torch.rand(2, 3, 5, 5, 2, 2).log(),
        reference_fn=lambda i, _:
            loss_reference_fns['NLLLossNd'](i, t.type_as(i).long(), **kwargs),
        pickle=False)


def nlllossNd_no_reduce_weights_test():
    t = Variable(torch.rand(2, 5, 5, 2, 2).mul(3).floor().long())
    weight = Variable(torch.rand(3))

    def kwargs(i):
        return {'weight': weight.type_as(i), 'reduce': False}

    return dict(
        fullname='NLLLossNd_no_reduce_weights',
        constructor=wrap_functional(
            lambda i: F.nll_loss(i, t.type_as(i).long(), **kwargs(i))),
        input_fn=lambda: torch.rand(2, 3, 5, 5, 2, 2).log(),
        reference_fn=lambda i, _:
            loss_reference_fns['NLLLossNd'](i, t.type_as(i).long(), **kwargs(i)),
        pickle=False)


def smoothl1loss_no_reduce_test():
    t = Variable(torch.randn(2, 3, 4))
    return dict(
        fullname='SmoothL1Loss_no_reduce',
        constructor=wrap_functional(
            lambda i: F.smooth_l1_loss(i, t.type_as(i), reduce=False)),
        input_fn=lambda: torch.randn(2, 3, 4),
        reference_fn=lambda i, _:
            loss_reference_fns['SmoothL1Loss'](i, t.type_as(i), reduce=False),
        pickle=False)


def smoothl1loss_no_reduce_scalar_test():
    t = torch.randn(())
    return dict(
        fullname='SmoothL1Loss_no_reduce_scalar',
        constructor=wrap_functional(
            lambda i: F.smooth_l1_loss(i, t.type_as(i), reduce=False)),
        input_fn=lambda: torch.randn(()),
        reference_fn=lambda i, _:
            loss_reference_fns['SmoothL1Loss'](i, t.type_as(i), reduce=False),
        pickle=False)


def multilabelmarginloss_1d_no_reduce_test():
    t = Variable(torch.rand(10).mul(10).floor().long())
    return dict(
        fullname='MultiLabelMarginLoss_1d_no_reduce',
        constructor=wrap_functional(
            lambda i: F.multilabel_margin_loss(i, t.type_as(i).long(), reduce=False)),
        input_fn=lambda: torch.randn(10),
        reference_fn=lambda i, _:
            loss_reference_fns['MultiLabelMarginLoss'](i, t.data.type_as(i).long(), reduce=False),
        check_no_size_average=True,
        check_gradgrad=False,
        pickle=False)


def multilabelmarginloss_index_neg_test():
    t = Variable(torch.clamp(torch.rand(5, 10).add(-.5).mul(20).floor().long(), min=-1))
    return dict(
        fullname='MultiLabelMarginLoss_index_neg',
        constructor=wrap_functional(
            lambda i: F.multilabel_margin_loss(i, t.type_as(i).long(), reduce=False)),
        input_fn=lambda: torch.randn(5, 10),
        reference_fn=lambda i, _:
            loss_reference_fns['MultiLabelMarginLoss'](i, t.data.type_as(i).long(), reduce=False),
        check_no_size_average=True,
        check_gradgrad=False,
        pickle=False)


def multilabelmarginloss_no_reduce_test():
    t = Variable(torch.rand(5, 10).mul(10).floor().long())
    return dict(
        fullname='MultiLabelMarginLoss_no_reduce',
        constructor=wrap_functional(
            lambda i: F.multilabel_margin_loss(i, t.type_as(i).long(), reduce=False)),
        input_fn=lambda: torch.randn(5, 10),
        reference_fn=lambda i, _:
            loss_reference_fns['MultiLabelMarginLoss'](i, t.data.type_as(i).long(), reduce=False),
        check_no_size_average=True,
        check_gradgrad=False,
        pickle=False)


def hingeembeddingloss_no_reduce_test():
    t = Variable(torch.randn(10).gt(0).double().mul_(2).sub(1))
    return dict(
        fullname='HingeEmbeddingLoss_no_reduce',
        constructor=wrap_functional(
            lambda i: F.hinge_embedding_loss(i, t.type_as(i), reduce=False)),
        input_fn=lambda: torch.randn(10),
        reference_fn=lambda i, _:
            loss_reference_fns['HingeEmbeddingLoss'](i, t.type_as(i), reduce=False),
        check_no_size_average=True,
        pickle=False)


def hingeembeddingloss_margin_no_reduce_test():
    t = Variable(torch.randn(10).gt(0).double().mul_(2).sub(1))
    return dict(
        fullname='HingeEmbeddingLoss_margin_no_reduce',
        constructor=wrap_functional(
            lambda i: F.hinge_embedding_loss(i, t.type_as(i), margin=0.5, reduce=False)),
        input_fn=lambda: torch.randn(10),
        reference_fn=lambda i, _:
            loss_reference_fns['HingeEmbeddingLoss'](i, t.type_as(i), margin=0.5, reduce=False),
        check_no_size_average=True,
        pickle=False)


def softmarginloss_no_reduce_test():
    t = Variable(torch.randn(5, 5))
    return dict(
        fullname='SoftMarginLoss_no_reduce',
        constructor=wrap_functional(
            lambda i: F.soft_margin_loss(i, t.type_as(i), reduce=False)),
        input_fn=lambda: torch.randn(5, 5),
        reference_fn=lambda i, _:
            loss_reference_fns['SoftMarginLoss'](i, t.type_as(i), reduce=False),
        check_no_size_average=True,
        pickle=False)


def multilabelsoftmarginloss_no_reduce_test():
    t = Variable(torch.rand(5, 10).mul(2).floor())
    return dict(
        fullname='MultiLabelSoftMarginLoss_no_reduce',
        constructor=wrap_functional(
            lambda i: F.multilabel_soft_margin_loss(i, t.type_as(i), reduce=False)),
        input_fn=lambda: torch.randn(5, 10),
        reference_fn=lambda i, m: (-(t * i.sigmoid().log() + (1 - t) * (-i).sigmoid().log()) /
                                   (i.numel() if get_size_average(m) else 1)),
        check_no_size_average=True,
        check_gradgrad=False,
        pickle=False)


def multilabelsoftmarginloss_weights_no_reduce_test():
    t = Variable(torch.rand(5, 10).mul(2).floor())
    weights = Variable(torch.rand(10))
    return dict(
        fullname='MultiLabelSoftMarginLoss_weights_no_reduce',
        constructor=wrap_functional(
            lambda i: F.multilabel_soft_margin_loss(i, t.type_as(i),
                                                    weight=weights.type_as(i), reduce=False)),
        input_fn=lambda: torch.randn(5, 10),
        reference_fn=lambda i, m: (-((t * i.sigmoid().log() + (1 - t) * (-i).sigmoid().log()) * weights) /
                                   (i.numel() if get_size_average(m) else 1)),
        check_no_size_average=True,
        check_gradgrad=False,
        pickle=False)


def multimarginloss_no_reduce_test():
    t = Variable(torch.rand(5).mul(8).floor().long())
    return dict(
        fullname='MultiMarginLoss_no_reduce',
        constructor=wrap_functional(
            lambda i: F.multi_margin_loss(i, t.type_as(i).long(), reduce=False)),
        input_fn=lambda: torch.randn(5, 10),
        reference_fn=lambda i, _:
            loss_reference_fns['MultiMarginLoss'](i, t.data.type_as(i).long(), reduce=False),
        check_no_size_average=True,
        check_gradgrad=False,
        pickle=False)


def multimarginloss_1d_no_reduce_test():
    t = Variable(torch.rand(1).mul(8).floor().long())
    return dict(
        fullname='MultiMarginLoss_1d_no_reduce',
        constructor=wrap_functional(
            lambda i: F.multi_margin_loss(i, t.type_as(i).long(), reduce=False)),
        input_fn=lambda: torch.randn(10),
        reference_fn=lambda i, _:
            loss_reference_fns['MultiMarginLoss'](i, t.data.type_as(i).long(), reduce=False),
        check_no_size_average=True,
        check_gradgrad=False,
        pickle=False)


def multimarginloss_p_no_reduce_test():
    t = Variable(torch.rand(5).mul(8).floor().long())
    return dict(
        fullname='MultiMarginLoss_p_no_reduce',
        constructor=wrap_functional(
            lambda i: F.multi_margin_loss(i, t.type_as(i).long(), p=2, reduce=False)),
        input_fn=lambda: torch.randn(5, 10).clamp_(1e-2, 1 - 1e-2),
        reference_fn=lambda i, _:
            loss_reference_fns['MultiMarginLoss'](i, t.data.type_as(i).long(), p=2, reduce=False),
        check_no_size_average=True,
        check_gradgrad=False,
        pickle=False)


def multimarginloss_margin_no_reduce_test():
    t = Variable(torch.rand(5).mul(8).floor().long())
    return dict(
        fullname='MultiMarginLoss_margin_no_reduce',
        constructor=wrap_functional(
            lambda i: F.multi_margin_loss(i, t.type_as(i).long(), margin=0.5, reduce=False)),
        input_fn=lambda: torch.randn(5, 10),
        reference_fn=lambda i, _:
            loss_reference_fns['MultiMarginLoss'](i, t.data.type_as(i).long(),
                                                  margin=0.5, reduce=False),
        check_no_size_average=True,
        check_gradgrad=False,
        pickle=False)


def multimarginloss_weights_no_reduce_test():
    t = Variable(torch.rand(5).mul(8).floor().long())
    weights = Variable(torch.rand(10))
    return dict(
        fullname='MultiMarginLoss_weights_no_reduce',
        constructor=wrap_functional(
            lambda i: F.multi_margin_loss(i, t.type_as(i).long(), weight=weights.type_as(i),
                                          reduce=False)),
        input_fn=lambda: torch.randn(5, 10),
        reference_fn=lambda i, _:
            loss_reference_fns['MultiMarginLoss'](i, t.data.type_as(i).long(),
                                                  weight=weights, reduce=False),
        check_no_size_average=True,
        check_gradgrad=False,
        pickle=False)


new_module_tests = [
    poissonnllloss_no_reduce_test(),
    bceloss_no_reduce_test(),
    bceloss_weights_no_reduce_test(),
    bce_with_logistic_no_reduce_test(),
    bceloss_no_reduce_scalar_test(),
    bceloss_weights_no_reduce_scalar_test(),
    bce_with_logistic_no_reduce_scalar_test(),
    kldivloss_no_reduce_test(),
    kldivloss_no_reduce_scalar_test(),
    l1loss_no_reduce_test(),
    l1loss_no_reduce_scalar_test(),
    mseloss_no_reduce_test(),
    mseloss_no_reduce_scalar_test(),
    nllloss_no_reduce_test(),
    nllloss_no_reduce_ignore_index_test(),
    nllloss_no_reduce_weights_test(),
    nllloss_no_reduce_weights_ignore_index_test(),
    nllloss_no_reduce_weights_ignore_index_neg_test(),
    nllloss2d_no_reduce_test(),
    nllloss2d_no_reduce_weights_test(),
    nllloss2d_no_reduce_ignore_index_test(),
    nlllossNd_no_reduce_test(),
    nlllossNd_no_reduce_weights_test(),
    nlllossNd_no_reduce_ignore_index_test(),
    smoothl1loss_no_reduce_test(),
    smoothl1loss_no_reduce_scalar_test(),
    multilabelmarginloss_1d_no_reduce_test(),
    multilabelmarginloss_index_neg_test(),
    multilabelmarginloss_no_reduce_test(),
    hingeembeddingloss_no_reduce_test(),
    hingeembeddingloss_margin_no_reduce_test(),
    softmarginloss_no_reduce_test(),
    multilabelsoftmarginloss_no_reduce_test(),
    multilabelsoftmarginloss_weights_no_reduce_test(),
    multimarginloss_no_reduce_test(),
    multimarginloss_1d_no_reduce_test(),
    multimarginloss_p_no_reduce_test(),
    multimarginloss_margin_no_reduce_test(),
    multimarginloss_weights_no_reduce_test(),
    dict(
        module_name='BatchNorm1d',
        constructor_args=(10,),
        input_size=(4, 10),
        cudnn=True,
        check_eval=True,
        desc='affine',
    ),
    dict(
        module_name='BatchNorm1d',
        constructor_args=(5,),
        input_size=(4, 5, 3),
        cudnn=True,
        check_eval=True,
        desc='3d_input',
    ),
    dict(
        module_name='BatchNorm1d',
        constructor_args=(10, 1e-3, 0.3, False),
        input_size=(4, 10),
        cudnn=True,
        check_eval=True,
        desc='not_affine',
    ),
    dict(
        module_name='BatchNorm1d',
        constructor_args=(10, 1e-3, 0.3, True, False),
        input_size=(4, 10),
        cudnn=True,
        check_eval=True,
        desc='not_tracking_stats',
    ),
    dict(
        module_name='BatchNorm1d',
        constructor_args=(5, 1e-3, 0.3, False),
        input_size=(4, 5, 3),
        cudnn=True,
        check_eval=True,
        desc='3d_input_not_affine',
    ),
    dict(
        module_name='BatchNorm2d',
        constructor_args=(3,),
        input_size=(2, 3, 6, 6),
        cudnn=True,
        check_eval=True,
    ),
    dict(
        module_name='BatchNorm2d',
        constructor_args=(3, 1e-3, 0.8),
        input_size=(2, 3, 6, 6),
        cudnn=True,
        check_eval=True,
        desc='momentum',
    ),
    dict(
        module_name='BatchNorm2d',
        constructor_args=(3, 1e-3, 0.8, False),
        input_size=(2, 3, 6, 6),
        cudnn=True,
        check_eval=True,
        desc='not_affine',
    ),
    dict(
        module_name='BatchNorm2d',
        constructor_args=(3, 1e-3, 0.8, True, False),
        input_size=(2, 3, 6, 6),
        cudnn=True,
        check_eval=True,
        desc='not_tracking_stats',
    ),
    dict(
        module_name='BatchNorm3d',
        constructor_args=(3,),
        input_size=(2, 3, 4, 4, 4),
        cudnn=True,
        check_eval=True,
    ),
    dict(
        module_name='BatchNorm3d',
        constructor_args=(3, 1e-3, 0.7),
        input_size=(2, 3, 4, 4, 4),
        cudnn=True,
        check_eval=True,
        desc='momentum',
    ),
    dict(
        module_name='BatchNorm3d',
        constructor_args=(3, 1e-3, 0.7, False),
        input_size=(2, 3, 4, 4, 4),
        cudnn=True,
        check_eval=True,
        desc='not_affine',
    ),
    dict(
        module_name='BatchNorm3d',
        constructor_args=(3, 1e-3, 0.7, True, False),
        input_size=(2, 3, 4, 4, 4),
        cudnn=True,
        check_eval=True,
        desc='not_tracking_stats',
    ),
    dict(
        module_name='InstanceNorm1d',
        constructor_args=(3, 1e-3, 0.3),
        input_size=(4, 3, 15),
        cudnn=True,
        check_eval=True,
    ),
    dict(
        module_name='InstanceNorm1d',
        constructor_args=(3, 1e-3, 0.3, False, True),
        input_size=(4, 3, 15),
        cudnn=True,
        check_eval=True,
        desc='tracking_stats',
    ),
    dict(
        module_name='InstanceNorm2d',
        constructor_args=(3, 1e-3, 0.3),
        input_size=(2, 3, 6, 6),
        cudnn=True,
        check_eval=True,
    ),
    dict(
        module_name='InstanceNorm2d',
        constructor_args=(3, 1e-3, 0.3, False, True),
        input_size=(2, 3, 6, 6),
        cudnn=True,
        check_eval=True,
        desc='tracking_stats',
    ),
    dict(
        module_name='InstanceNorm3d',
        constructor_args=(3, 1e-3, 0.3),
        input_size=(2, 3, 4, 4, 4),
        cudnn=True,
        check_eval=True,
    ),
    dict(
        module_name='InstanceNorm3d',
        constructor_args=(3, 1e-3, 0.3, False, True),
        input_size=(2, 3, 4, 4, 4),
        cudnn=True,
        check_eval=True,
        desc='tracking_stats',
    ),
    dict(
        module_name='LayerNorm',
        constructor_args=([5], 1e-3),
        input_size=(4, 5, 5),
        cudnn=True,
        check_eval=True,
        desc='1d_elementwise_affine',
    ),
    dict(
        module_name='LayerNorm',
        constructor_args=([5], 1e-3, False),
        input_size=(4, 5, 5),
        cudnn=True,
        check_eval=True,
        desc='1d_no_elementwise_affine',
    ),
    dict(
        module_name='LayerNorm',
        constructor_args=([2, 2, 5], 1e-3),
        input_size=(4, 2, 2, 5),
        cudnn=True,
        check_eval=True,
        desc='3d_elementwise_affine',
    ),
    dict(
        module_name='LayerNorm',
        constructor_args=([2, 2, 5], 1e-3, False),
        input_size=(4, 2, 2, 5),
        cudnn=True,
        check_eval=True,
        desc='3d_no_elementwise_affine',
    ),
    dict(
        module_name='GroupNorm',
        constructor_args=(3, 6, 1e-3),
        input_size=(4, 6, 5),
        cudnn=True,
        check_eval=True,
        desc='1d_affine',
    ),
    dict(
        module_name='GroupNorm',
        constructor_args=(5, 5, 1e-3, False),
        input_size=(4, 5, 5),
        cudnn=True,
        check_eval=True,
        desc='1d_no_affine_IN',  # this setting is equivalent with InstanceNorm
    ),
    dict(
        module_name='GroupNorm',
        constructor_args=(1, 5, 1e-3, False),
        input_size=(4, 5, 5),
        cudnn=True,
        check_eval=True,
        desc='1d_no_affine_LN',  # this setting is equivalent with LayerNorm
    ),
    dict(
        module_name='GroupNorm',
        constructor_args=(3, 6, 1e-3),
        input_size=(4, 6, 2, 3),
        cudnn=True,
        check_eval=True,
        desc='2d_affine',
    ),
    dict(
        module_name='GroupNorm',
        constructor_args=(3, 3, 1e-3, False),
        input_size=(4, 3, 2, 3),
        cudnn=True,
        check_eval=True,
        desc='2d_no_affine_IN',  # this setting is equivalent with InstanceNorm
    ),
    dict(
        module_name='GroupNorm',
        constructor_args=(1, 3, 1e-3, False),
        input_size=(4, 3, 2, 3),
        cudnn=True,
        check_eval=True,
        desc='2d_no_affine_LN',  # this setting is equivalent with LayerNorm
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
        desc='stride',
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
        fullname='ConvTranspose1d',
        constructor=lambda: nn.ConvTranspose1d(3, 4, kernel_size=3, stride=(3,), padding=1, output_padding=(1,)),
        cudnn=True,
        input_size=(1, 3, 7),
    ),
    dict(
        module_name='ConvTranspose1d',
        constructor_args=(3, 4, 3, 2, 1, 1, 1, False),
        input_size=(1, 3, 6),
        cudnn=True,
        desc='no_bias',
    ),
    dict(
        module_name='ConvTranspose1d',
        constructor_args=(3, 4, 3, 2, 1, 1, 1, True, 2),
        input_size=(1, 3, 6),
        cudnn=True,
        desc='dilated',
    ),
    dict(
        fullname='ConvTranspose1d_groups',
        constructor=lambda: nn.ConvTranspose1d(4, 6, 3, stride=(3,), padding=1, output_padding=(1,), groups=2),
        cudnn=True,
        input_size=(2, 4, 7),
    ),
    dict(
        module_name='MaxPool1d',
        constructor_args=(4,),
        input_size=(2, 10, 4),
    ),
    dict(
        module_name='MaxPool1d',
        constructor_args=(4, 4),
        input_size=(2, 10, 4),
        desc='stride',
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
        desc='strided',
    ),
    dict(
        module_name='Conv2d',
        constructor_args=(3, 4, (3, 3), (2, 2), (1, 1)),
        input_size=(2, 3, 6, 6),
        cudnn=True,
        desc='padding',
    ),
    dict(
        module_name='Conv2d',
        constructor_args=(3, 2, (3, 3), (2, 2), (1, 1), (2, 2)),
        input_size=(2, 3, 8, 8),
        cudnn=True,
        desc='dilated',
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
        input_size=(1, 3, 7, 6),
    ),
    dict(
        module_name='ConvTranspose2d',
        constructor_args=(3, 4, 3, (2, 3), 1, (1, 1), 1, False, (2, 2)),
        input_size=(1, 3, 6, 7),
        cudnn=True,
        desc='dilated',
    ),
    dict(
        module_name='ConvTranspose2d',
        constructor_args=(3, 4, 3, (2, 3), 1, (1, 1), 1, False),
        input_size=(1, 3, 6, 7),
        cudnn=True,
        desc='no_bias',
    ),
    dict(
        fullname='ConvTranspose2d_groups',
        constructor=lambda: nn.ConvTranspose2d(2, 4, (2, 3), groups=2),
        input_size=(1, 2, 4, 5),
        cudnn=True,
    ),
    dict(
        fullname='Conv2d_depthwise',
        constructor=lambda: nn.Conv2d(4, 4, (3, 3), groups=4),
        input_size=(2, 4, 6, 6),
    ),
    dict(
        fullname='Conv2d_depthwise_with_multiplier',
        constructor=lambda: nn.Conv2d(4, 8, (3, 3), groups=4),
        input_size=(2, 4, 6, 6),
    ),
    dict(
        fullname='Conv2d_depthwise_strided',
        constructor=lambda: nn.Conv2d(4, 4, (3, 3), stride=(2, 2), groups=4),
        input_size=(2, 4, 6, 6),
    ),
    dict(
        fullname='Conv2d_depthwise_padded',
        constructor=lambda: nn.Conv2d(4, 4, (3, 3), padding=(1, 1), groups=4),
        input_size=(2, 4, 6, 6),
    ),
    dict(
        fullname='Conv2d_depthwise_dilated',
        constructor=lambda: nn.Conv2d(4, 4, (2, 2), dilation=(2, 2), groups=4),
        input_size=(2, 4, 5, 5),
    ),
    dict(
        module_name='MaxPool2d',
        constructor_args=((3, 3), (2, 2), (1, 1)),
        input_size=(1, 3, 7, 7),
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
        input_size=(1, 3, 7, 7),
    ),
    dict(
        module_name='LPPool2d',
        constructor_args=(1.5, 2),
        input_fn=lambda: torch.rand(1, 3, 7, 7),
        desc='norm',
    ),
    dict(
        module_name='LPPool1d',
        constructor_args=(1.5, 2),
        input_fn=lambda: torch.rand(1, 3, 7),
        desc='norm',
    ),
    dict(
        module_name='LPPool1d',
        constructor_args=(2, 2, 3),
        input_size=(1, 3, 7),
    ),
    dict(
        module_name='LocalResponseNorm',
        constructor_args=(3, ),
        input_size=(1, 5, 7),
        desc='1d'
    ),
    dict(
        module_name='LocalResponseNorm',
        constructor_args=(2, ),
        input_size=(1, 5, 7, 7),
        desc='2d_uneven_pad'
    ),
    dict(
        module_name='LocalResponseNorm',
        constructor_args=(1, 1, 0.5, 2),
        input_size=(1, 5, 7, 7, 7),
        desc='3d_custom_params'
    ),
    dict(
        module_name='ReflectionPad1d',
        constructor_args=((1, 2),),
        input_size=(2, 3, 8),
    ),
    dict(
        module_name='ReflectionPad2d',
        constructor_args=((1, 2, 3, 4),),
        input_size=(2, 3, 8, 8),
    ),
    dict(
        module_name='ReplicationPad1d',
        constructor_args=((1, 2),),
        input_size=(2, 3, 4),
    ),
    dict(
        module_name='ReplicationPad2d',
        constructor_args=((1, 2, 3, 4),),
        input_size=(2, 3, 4, 4),
    ),
    dict(
        module_name='ZeroPad2d',
        constructor_args=((1, 2, 3, 4),),
        input_size=(2, 3, 4, 4)
    ),
    dict(
        module_name='ZeroPad2d',
        constructor_args=((-1, -1, -1, -2),),
        input_size=(2, 3, 4, 4),
        desc='negative_dims'
    ),
    dict(
        module_name='ConstantPad1d',
        constructor_args=((1, 2), 2),
        input_size=(2, 3, 4)
    ),
    dict(
        module_name='ConstantPad2d',
        constructor_args=((1, 2, 3, 4), 2),
        input_size=(2, 3, 4, 4)
    ),
    dict(
        module_name='ConstantPad3d',
        constructor_args=((1, 2, 3, 4, 1, 0), 2),
        input_size=(2, 3, 4, 4, 5)
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
        desc='no_bias',
    ),
    dict(
        module_name='Conv3d',
        constructor_args=(3, 4, 2, 2),
        input_size=(2, 3, 5, 5, 5),
        cudnn=True,
        desc='stride',
    ),
    dict(
        module_name='Conv3d',
        constructor_args=(3, 4, 2, 2, 1),
        input_size=(2, 3, 5, 5, 5),
        cudnn=True,
        desc='stride_padding',
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
        fullname='Conv3d_dilated_strided',
        constructor=lambda: nn.Conv3d(3, 4, kernel_size=2, dilation=2, stride=2),
        input_size=(2, 3, 5, 5, 5),
    ),
    dict(
        module_name='ConvTranspose3d',
        constructor_args=(2, 3, (2, 3, 2)),
        cudnn=True,
        input_size=(1, 2, 4, 5, 4),
    ),
    dict(
        module_name='ConvTranspose3d',
        constructor_args=(2, 3, (2, 3, 2), 1, 0, 0, 1, True, (2, 2, 2)),
        cudnn=True,
        input_size=(1, 2, 4, 5, 4),
        desc='dilated',
    ),
    dict(
        module_name='MaxPool3d',
        constructor_args=((2, 2, 2),),
        input_size=(2, 3, 5, 5, 5),
    ),
    dict(
        module_name='MaxPool3d',
        constructor_args=(2, (2, 2, 2)),
        input_size=(2, 3, 5, 5, 5),
        desc='stride',
    ),
    dict(
        module_name='MaxPool3d',
        constructor_args=(2, 2, (1, 1, 1)),
        input_size=(2, 3, 5, 5, 5),
        desc='stride_padding',
    ),
    dict(
        module_name='AvgPool3d',
        constructor_args=((2, 2, 2),),
        input_size=(2, 3, 4, 4, 4),
    ),
    dict(
        module_name='AvgPool3d',
        constructor_args=(2, (2, 2, 2)),
        input_size=(2, 3, 5, 5, 5),
        desc='stride',
    ),
    dict(
        module_name='AvgPool3d',
        constructor_args=(2, 2, (1, 1, 1)),
        input_size=(2, 3, 5, 5, 5),
        desc='stride_pad',
    ),
    dict(
        module_name='AvgPool3d',
        constructor_args=(4, 2, (1, 2, 1)),
        input_size=(2, 3, 5, 5, 5),
        desc='stride_pad_gpu_fixedkw_output',
    ),
    dict(
        module_name='AvgPool3d',
        constructor_args=((2, 4, 8), 1, (1, 1, 2)),
        input_size=(2, 3, 2, 4, 8),
        desc='stride_pad_gpu_general_output',
    ),
    dict(
        module_name='AvgPool3d',
        constructor_args=(3, 1, 0),
        input_size=(2, 3, 4, 4, 4),
        desc='stride1_pad0_gpu_input',
    ),
    dict(
        module_name='AvgPool3d',
        constructor_args=(2, 2, (1, 1, 1)),
        input_size=(2, 3, 4, 4, 4),
        desc='stride_pad_gpu_input_nooverlap',
    ),
    dict(
        module_name='ReplicationPad3d',
        constructor_args=((1, 2, 3, 4, 5, 6),),
        input_size=(2, 3, 5, 5, 5),
    ),
    dict(
        module_name='Embedding',
        constructor_args=(4, 3),
        input_fn=lambda: Variable(torch.randperm(2).repeat(1, 2)),
        jacobian_input=False,
        check_gradgrad=False,
    ),
    dict(
        module_name='EmbeddingBag',
        constructor_args=(4, 3),
        input_fn=lambda: Variable(torch.randperm(2).repeat(1, 2)),
        jacobian_input=False,
        check_gradgrad=False,
    ),
    dict(
        fullname='EmbeddingBag_sparse',
        constructor=lambda: nn.EmbeddingBag(4, 3, sparse=True),
        input_fn=lambda: Variable(torch.randperm(2).repeat(1, 2)),
        jacobian_input=False,
        check_gradgrad=False,
    ),
    dict(
        constructor=lambda: nn.Embedding(4, 3, sparse=True),
        input_fn=lambda: Variable(torch.randperm(2).repeat(1, 2)),
        jacobian_input=False,
        fullname='Embedding_sparse',
        check_gradgrad=False,
    ),
    dict(
        constructor=lambda: nn.FractionalMaxPool2d(
            2, output_ratio=0.5, _random_samples=torch.DoubleTensor(1, 3, 2).uniform_()),
        input_size=(1, 3, 5, 5),
        fullname='FractionalMaxPool2d_ratio',
    ),
    dict(
        constructor=lambda: nn.FractionalMaxPool2d((2, 2), output_size=(
            4, 4), _random_samples=torch.DoubleTensor(1, 3, 2).uniform_()),
        input_size=(1, 3, 7, 7),
        fullname='FractionalMaxPool2d_size',
        test_cuda=False,
    ),
    dict(
        module_name='PixelShuffle',
        constructor_args=(3,),
        input_size=(1, 9, 4, 4),
    ),
    dict(
        module_name='Upsample',
        constructor_args=(12, None, 'nearest'),
        input_size=(1, 2, 4),
        desc='nearest_1d',
    ),
    dict(
        module_name='Upsample',
        constructor_args=((12, ), None, 'nearest'),
        input_size=(1, 2, 3),
        desc='nearest_tuple_1d',
    ),
    dict(
        module_name='Upsample',
        constructor_args=(None, 4, 'nearest'),
        input_size=(1, 2, 4),
        desc='nearest_scale_1d',
    ),
    dict(
        module_name='Upsample',
        constructor_args=(12, None, 'linear', False),
        input_size=(1, 2, 4),
        desc='linear_1d',
    ),
    dict(
        module_name='Upsample',
        constructor_args=((4, ), None, 'linear', False),
        input_size=(1, 2, 3),
        desc='linear_tuple_1d',
    ),
    dict(
        module_name='Upsample',
        constructor_args=(None, 4, 'linear', False),
        input_size=(1, 2, 4),
        desc='linear_scale_1d',
    ),
    dict(
        module_name='Upsample',
        constructor_args=(12, None, 'linear', True),
        input_size=(1, 2, 4),
        desc='linear_1d_align_corners',
    ),
    dict(
        module_name='Upsample',
        constructor_args=(None, 4, 'linear', True),
        input_size=(1, 2, 4),
        desc='linear_scale_1d_align_corners',
    ),
    dict(
        module_name='Upsample',
        constructor_args=(12, None, 'nearest'),
        input_size=(1, 2, 4, 4),
        desc='nearest_2d',
    ),
    dict(
        module_name='Upsample',
        constructor_args=((12, 16), None, 'nearest'),
        input_size=(1, 2, 3, 4),
        desc='nearest_tuple_2d',
    ),
    dict(
        module_name='Upsample',
        constructor_args=(None, 4, 'nearest'),
        input_size=(1, 2, 4, 4),
        desc='nearest_scale_2d',
    ),
    dict(
        module_name='Upsample',
        constructor_args=(12, None, 'bilinear', False),
        input_size=(1, 2, 4, 4),
        desc='bilinear_2d',
    ),
    dict(
        module_name='Upsample',
        constructor_args=((4, 6), None, 'bilinear', False),
        input_size=(1, 2, 2, 3),
        desc='bilinear_tuple_2d',
    ),
    dict(
        module_name='Upsample',
        constructor_args=(None, 4, 'bilinear', False),
        input_size=(1, 2, 4, 4),
        desc='bilinear_scale_2d',
    ),
    dict(
        module_name='Upsample',
        constructor_args=(None, (2, 2), 'bilinear', False),
        input_size=(1, 2, 4, 4),
        desc='bilinear_scale_tuple_shared_2d',
    ),
    dict(
        module_name='Upsample',
        constructor_args=(None, (2, 1), 'bilinear', False),
        input_size=(1, 2, 4, 4),
        desc='bilinear_scale_tuple_skewed_2d',
    ),
    dict(
        module_name='Upsample',
        constructor_args=((4, 6), None, 'bilinear', True),
        input_size=(1, 2, 4, 4),
        desc='bilinear_tuple_2d_align_corners',
    ),
    dict(
        module_name='Upsample',
        constructor_args=(None, (2, 1), 'bilinear', True),
        input_size=(1, 2, 4, 4),
        desc='bilinear_scale_tuple_skewed_2d_align_corners',
    ),
    dict(
        module_name='Upsample',
        constructor_args=(12, None, 'nearest'),
        input_size=(1, 2, 4, 4, 4),
        desc='nearest_3d',
    ),
    dict(
        module_name='Upsample',
        constructor_args=((12, 16, 16), None, 'nearest'),
        input_size=(1, 2, 3, 4, 4),
        desc='nearest_tuple_3d',
    ),
    dict(
        module_name='Upsample',
        constructor_args=(None, 4, 'nearest'),
        input_size=(1, 2, 4, 4, 4),
        desc='nearest_scale_3d',
    ),
    dict(
        module_name='Upsample',
        constructor_args=(12, None, 'trilinear', False),
        input_size=(1, 2, 4, 4, 4),
        desc='trilinear_3d',
    ),
    dict(
        module_name='Upsample',
        constructor_args=((4, 6, 6), None, 'trilinear', False),
        input_size=(1, 2, 2, 3, 3),
        desc='trilinear_tuple_3d',
    ),
    dict(
        module_name='Upsample',
        constructor_args=(None, 3, 'trilinear', False),
        input_size=(1, 2, 3, 4, 4),
        desc='trilinear_scale_3d',
        # See https://github.com/pytorch/pytorch/issues/5006
        precision=3e-4,
    ),
    dict(
        module_name='Upsample',
        constructor_args=((4, 6, 6), None, 'trilinear', True),
        input_size=(1, 2, 2, 3, 3),
        desc='trilinear_tuple_3d_align_corners',
    ),
    dict(
        module_name='Upsample',
        constructor_args=(None, 3, 'trilinear', True),
        input_size=(1, 2, 3, 4, 4),
        desc='trilinear_scale_3d_align_corners',
        # See https://github.com/pytorch/pytorch/issues/5006
        precision=3e-4,
    ),
    dict(
        module_name='AdaptiveMaxPool1d',
        constructor_args=(3,),
        input_fn=lambda: _rand_tensor_non_equal(1, 3, 5),
    ),
    dict(
        module_name='AdaptiveMaxPool2d',
        constructor_args=(3,),
        input_fn=lambda: _rand_tensor_non_equal(1, 3, 5, 6),
        desc='single',
    ),
    dict(
        module_name='AdaptiveMaxPool2d',
        constructor_args=((3, 4),),
        input_fn=lambda: _rand_tensor_non_equal(1, 3, 5, 6),
        desc='tuple',
    ),
    dict(
        module_name='AdaptiveMaxPool3d',
        constructor_args=(3,),
        input_fn=lambda: _rand_tensor_non_equal(2, 3, 5, 6, 7),
        desc='single',
    ),
    dict(
        module_name='AdaptiveMaxPool3d',
        constructor_args=((3, 4, 5),),
        input_fn=lambda: _rand_tensor_non_equal(2, 3, 5, 6, 7),
        desc='tuple',
    ),
    dict(
        module_name='AdaptiveMaxPool3d',
        constructor_args=(3,),
        input_fn=lambda: _rand_tensor_non_equal(2, 3, 12, 9, 3),
        desc='single_nonatomic',
    ),
    dict(
        module_name='AdaptiveMaxPool3d',
        constructor_args=((3, 4, 5),),
        input_fn=lambda: _rand_tensor_non_equal(2, 3, 6, 4, 10),
        desc='tuple_nonatomic',
    ),
    dict(
        module_name='AdaptiveAvgPool1d',
        constructor_args=(3,),
        input_fn=lambda: torch.rand(1, 3, 5),
    ),
    dict(
        module_name='AdaptiveAvgPool2d',
        constructor_args=(3,),
        input_fn=lambda: torch.rand(1, 3, 5, 6),
        desc='single',
    ),
    dict(
        module_name='AdaptiveAvgPool2d',
        constructor_args=((3, 4),),
        input_fn=lambda: torch.rand(1, 3, 5, 6),
        desc='tuple',
    ),
    dict(
        module_name='AdaptiveAvgPool3d',
        constructor_args=(3,),
        input_fn=lambda: torch.rand(2, 3, 5, 2, 7),
        desc='single',
    ),
    dict(
        module_name='AdaptiveAvgPool3d',
        constructor_args=((3, 4, 5),),
        input_fn=lambda: torch.rand(2, 3, 5, 3, 7),
        desc='tuple',
    ),
    dict(
        module_name='SELU',
        input_size=(3, 2, 5),
        check_inplace=True
    ),
    dict(
        module_name='SELU',
        input_size=(),
        check_inplace=True,
        desc='scalar'
    ),
    dict(
        module_name='GLU',
        input_size=(5, 6),
    ),
    dict(
        module_name='GLU',
        constructor_args=(1,),
        input_size=(5, 6, 7),
        desc='dim'
    ),
    dict(
        constructor=wrap_functional(F.softmax, dim=-1),
        input_size=(2, 128),  # trigger the last-dim algo in CUDA
        fullname='softmax_lastdim',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.softmax, dim=1),
        input_size=(2, 128, 2, 2),  # trigger special case of spatial CUDA algo
        fullname='softmax_spatial_special',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.softmax, dim=1),
        input_size=(2, 2, 4, 4),  # regular spatial algorithm
        fullname='softmax_spatial',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.softmax, dim=0),
        input_size=(2, 3, 4, 5),
        fullname='softmax_functional_dim0',
        test_cuda=False,
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.softmax, dim=3),
        input_size=(2, 3, 4, 5),
        fullname='softmax_functional_dim3',
        test_cuda=False,
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.softmax, dim=-1),
        input_size=(),
        fullname='softmax_functional_scalar',
        test_cuda=False,
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.log_softmax, dim=-1),
        input_size=(2, 128),  # trigger the last-dim algo in CUDA
        fullname='log_softmax_lastdim',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.log_softmax, dim=1),
        input_size=(2, 128, 2, 2),  # trigger special case of spatial CUDA algo
        fullname='log_softmax_spatial_special',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.log_softmax, dim=1),
        input_size=(2, 2, 4, 4),  # regular spatial algorithm
        fullname='log_softmax_spatial',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.log_softmax, dim=0),
        input_size=(2, 3, 4, 5),
        fullname='log_softmax_dim0',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.log_softmax, dim=3),
        input_size=(2, 3, 4, 5),
        fullname='log_softmax_dim3',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.log_softmax, dim=0),
        input_size=(),
        fullname='log_softmax_scalar',
        pickle=False,
    ),
    dict(
        fullname='Unfold',
        constructor=lambda: nn.Unfold((2, 2), (1, 1), (0, 0), (1, 1)),
        input_size=(2, 4, 3, 3),
        check_gradgrad=False,
        test_cuda=True,
    ),
    dict(
        fullname='Fold',
        constructor=lambda: nn.Fold((3, 3), (2, 2), (1, 1), (0, 0), (1, 1)),
        input_size=(2, 16, 4),
        check_gradgrad=False,
        test_cuda=True,
    ),
    dict(
        fullname='Unfold_int_input',
        constructor=lambda: nn.Unfold(2, 1, 0, 1),
        input_size=(2, 4, 3, 3),
        check_gradgrad=False,
        test_cuda=True,
    ),
    dict(
        fullname='Fold_int_input',
        constructor=lambda: nn.Fold(3, 2, 1, 0, 1),
        input_size=(2, 16, 4),
        check_gradgrad=False,
        test_cuda=True,
    ),
    dict(
        module_name='Threshold',
        constructor_args=(2, 1),
        input_size=(),
        check_inplace=True,
        desc='threshold_value_scalar'
    ),

    dict(
        module_name='ReLU',
        input_size=(),
        check_inplace=True,
        desc='scalar'
    ),
    dict(
        module_name='ReLU6',
        input_size=(),
        check_inplace=True,
        desc='scalar'
    ),
    dict(
        module_name='RReLU',
        constructor_args=(0.1, 0.9),
        input_size=(),
        desc='with_up_down_scalar',
        test_cuda=False,
    ),
    dict(
        module_name='Hardtanh',
        input_size=(),
        reference_fn=lambda i, _: i.clamp(-1, 1),
        desc='scalar'
    ),
    dict(
        module_name='Sigmoid',
        input_size=(),
        desc='scalar',
    ),
    dict(
        module_name='Tanh',
        input_size=(),
        desc='scalar',
    ),
    dict(
        module_name='Softmax',
        constructor_args=(0,),
        input_size=(),
        reference_fn=lambda i, _: torch.exp(i).div(torch.exp(i).sum(0, True)),
        desc='scalar',
    ),
    dict(
        module_name='LogSoftmax',
        constructor_args=(0,),
        input_size=(),
        reference_fn=lambda i, _: torch.exp(i).div_(torch.exp(i).sum(0, False)).log_(),
        desc='multiparam_scalar',
    ),
    dict(
        module_name='ELU',
        constructor_args=(2.,),
        input_size=(),
        desc='scalar',
    ),
    dict(
        module_name='Hardshrink',
        constructor_args=(2.,),
        input_size=(),
        desc='scalar',
    ),
    dict(
        module_name='LeakyReLU',
        constructor_args=(0.5,),
        input_size=(),
        check_inplace=True,
        desc='with_negval_scalar'
    ),
    dict(
        module_name='LogSigmoid',
        input_size=(),
        reference_fn=lambda i, _: i.sigmoid().log(),
        desc='scalar'
    ),
    dict(
        module_name='Softplus',
        constructor_args=(2, -100),
        input_size=(),
        reference_fn=(lambda i, _: ((i * 2) > -100).type_as(i) * i +
                                   ((i * 2) <= -100).type_as(i) * 1. / 2. * torch.log(1 + torch.exp(2 * i))),
        desc='beta_threshold_scalar',
    ),
    dict(
        module_name='Softshrink',
        constructor_args=(1,),
        input_size=(),
        desc='lambda_scalar',
    ),
    dict(
        module_name='PReLU',
        input_size=(),
        reference_fn=lambda i, p: torch.clamp(i, min=0) + torch.clamp(i, max=0) * p[0][0],
        desc='scalar',
    ),
    dict(
        module_name='Softsign',
        input_size=(),
        reference_fn=lambda i, _: i.div(1 + torch.abs(i)),
        desc='scalar',
    ),
    dict(
        module_name='Softmin',
        constructor_args=(0,),
        input_size=(),
        desc='scalar',
    ),
    dict(
        module_name='Tanhshrink',
        input_size=(),
        desc='scalar',
    ),
]


for test_params in module_tests + new_module_tests:
    # TODO: CUDA is not implemented yet
    if 'constructor' not in test_params:
        name = test_params.pop('module_name')
        test_params['constructor'] = getattr(nn, name)
    test = NewModuleTest(**test_params)
    add_test(test)
    if 'check_eval' in test_params:
        # create a new test that is identical but that sets module.training to False
        desc = test_params.get('desc', None)
        test_params['desc'] = 'eval' if desc is None else desc + '_eval'

        def gen_eval_constructor(constructor):
            def eval_constructor(*args, **kwargs):
                cons = constructor(*args, **kwargs)
                cons.training = False
                return cons
            eval_constructor.__name__ = constructor.__name__
            return eval_constructor

        test_params['constructor'] = gen_eval_constructor(test_params['constructor'])
        test = NewModuleTest(**test_params)
        add_test(test)

for test_params in criterion_tests + new_criterion_tests:
    name = test_params.pop('module_name')
    test_params['constructor'] = getattr(nn, name)
    test = NewCriterionTest(**test_params)
    add_test(test)
    if 'check_no_size_average' in test_params:
        desc = test_params.get('desc', None)
        test_params['desc'] = 'no_size_average' if desc is None else desc + '_no_size_average'

        def gen_no_size_average_constructor(constructor):
            def no_size_average_constructor(*args, **kwargs):
                cons = constructor(*args, size_average=False, **kwargs)
                return cons
            no_size_average_constructor.__name__ = constructor.__name__
            return no_size_average_constructor

        test_params['constructor'] = gen_no_size_average_constructor(test_params['constructor'])
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
    fullname='MaxUnpool1d_net',))
add_test(NewModuleTest(
    constructor=lambda: UnpoolingNet(
        nn.MaxPool2d(2, return_indices=True),
        nn.MaxUnpool2d(2)),
    input_size=(1, 1, 2, 4),
    fullname='MaxUnpool2d_net',))
add_test(NewModuleTest(
    constructor=lambda: UnpoolingNet(
        nn.MaxPool3d(2, return_indices=True),
        nn.MaxUnpool3d(2)),
    input_size=(1, 1, 2, 4, 6),
    fullname='MaxUnpool3d_net',
    check_gradgrad=False,))

if __name__ == '__main__':
    run_tests()
