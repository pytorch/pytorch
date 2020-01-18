
import math
import sys
import random
import string
import unittest
try:
    import unittest.mock as mock
except ImportError:
    # isn't available in py2
    pass
import itertools
import warnings
import pickle
import contextlib
from copy import deepcopy
from itertools import repeat, product
from functools import reduce
from operator import mul
from collections import OrderedDict

import torch

# TODO: remove this global setting
# NN tests use double as the default dtype
torch.set_default_dtype(torch.double)

from torch._six import inf, nan
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.nn.utils.rnn as rnn_utils
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
import torch.nn.utils.prune as prune
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.autograd import gradcheck
from torch.autograd.gradcheck import gradgradcheck
from torch.nn import Parameter
from torch.nn.parallel._functions import Broadcast
from torch.testing._internal.common_utils import freeze_rng_state, run_tests, TestCase, skipIfNoLapack, skipIfRocm, \
    TEST_NUMPY, TEST_SCIPY, TEST_WITH_ROCM, download_file, PY3, to_gpu, \
    get_function_arglist, load_tests, repeat_test_for_types, ALL_TENSORTYPES, \
    ALL_TENSORTYPES2, TemporaryFileName
from torch.testing._internal.common_cuda import TEST_CUDA, TEST_MULTIGPU, TEST_CUDNN, TEST_CUDNN_VERSION
from torch.testing._internal.common_nn import NNTestCase, ModuleTest, CriterionTest, TestBase, \
    module_tests, criterion_tests, new_criterion_tests, loss_reference_fns, \
    ctcloss_reference, new_module_tests
from torch.testing._internal.common_device_type import instantiate_device_type_tests, dtypes, \
    dtypesIfCUDA, skipCUDAIfNoCudnn, skipCUDAIfCudnnVersionLessThan, onlyCUDA, \
    skipCUDAIfRocm, skipCUDAIf

from torch.nn import MultiheadAttention

from hypothesis import given
import torch.testing._internal.hypothesis_utils as hu
from torch.testing._internal.common_utils import _assertGradAndGradgradChecks
from torch.testing._internal.common_utils import dtype2prec

# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

TEST_LARGE_TENSOR = TEST_CUDA
if TEST_CUDA:
    TEST_LARGE_TENSOR = torch.cuda.get_device_properties(0).total_memory >= 12e9

if TEST_SCIPY:
    from scipy import stats
    import scipy.ndimage

if TEST_NUMPY:
    import numpy as np

NO_HALF_TENSORTYPES = [torch.float,
                       torch.double]

DOUBLE_TENSORTYPES = [torch.double]


# WARNING: If you add a new top-level test case to this file, you MUST
# update test/run_test.py to list it, otherwise it will NOT be run in
# CI.


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
        seqs = [s.random_(-128, 128) for s in seqs]
        ordered = sorted(seqs, key=len, reverse=True)
        return ordered

    def _padded_sequence(self, tensor_type):
        """Create Tensor of random padded sequences"""
        ordered = self._ordered_sequence(tensor_type)
        lengths = list(map(len, ordered))
        padded_tensor = rnn_utils.pad_sequence(ordered)
        return padded_tensor, lengths

    def test_type_casts(self):
        """Test type casting of `PackedSequence` against type casting of tensor"""
        for _, (input_type, _) in self._type_by_name.items():
            for expected_type_str, (_, cast_str) in self._type_by_name.items():
                for enforce_sorted in [True, False]:
                    padded, lengths = self._padded_sequence(input_type)
                    packed = rnn_utils.pack_padded_sequence(
                        padded, lengths, enforce_sorted=enforce_sorted)
                    # Apply cast to `PackedSequence` instance and unpack
                    masked = getattr(packed, cast_str)()
                    unpacked, lengths_out = rnn_utils.pad_packed_sequence(masked)
                    self.assertEqual(unpacked.type(), expected_type_str)

    def test_wrong_order(self):
        a = torch.ones(25, 300)
        b = torch.ones(22, 300)
        b_a = rnn_utils.pad_sequence([b, a])
        self.assertRaises(
            RuntimeError,
            lambda: rnn_utils.pack_padded_sequence(b_a, [22, 25], enforce_sorted=True))

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

    def test_to(self):
        for enforce_sorted in (True, False):
            padded, lengths = self._padded_sequence(torch.IntTensor)
            a = rnn_utils.pack_padded_sequence(
                padded, lengths, enforce_sorted=enforce_sorted).cpu()

            self.assertIs(a, a.to('cpu'))
            self.assertIs(a, a.cpu())
            self.assertIs(a, a.to('cpu', dtype=torch.int32))
            self.assertEqual(a.long(), a.to(torch.int64))

            if torch.cuda.is_available():
                for cuda in ['cuda', 'cuda:0' if torch.cuda.device_count() == 1 else 'cuda:1']:
                    b = a.cuda(device=cuda)
                    self.assertIs(b, b.to(cuda))
                    self.assertIs(b, b.cuda())
                    self.assertEqual(a, b.to('cpu'))
                    self.assertEqual(b, a.to(cuda))
                    self.assertEqual(a, b.to('cpu', dtype=torch.int32))
                    self.assertIs(b, b.to(dtype=torch.int32))
                    self.assertEqual(b.long(), b.to(dtype=torch.int64))

    def test_to_memory_format(self):
        m = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, bias=True)
        m = m.to(memory_format=torch.channels_last)
        for param in m.parameters():
            if param.dim() == 4:
                self.assertTrue(param.is_contiguous(memory_format=torch.channels_last))


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
        self.skip_double = kwargs.get('skip_double', False)

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

            if TEST_CUDA and self.should_test_cuda:
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

                if not self.skip_double:
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
                for p in module.parameters():
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
        self.check_half = kwargs.get('check_half', True)
        self.convert_target = kwargs.get('convert_target', True)

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

    def test_cuda(self, test_case, dtype=None, extra_args=None):
        def convert_dtype(obj, dtype, requires_grad=False):
            if isinstance(obj, torch.Tensor):
                return obj.detach().to(dtype=dtype).requires_grad_(requires_grad)
            elif isinstance(obj, torch.Tensor):
                return obj.to(dtype)
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
                if not isinstance(cpu_target, torch.LongTensor) and self.convert_target:
                    cpu_target = convert_dtype(cpu_target, dtype)
                cpu_module.type(dtype)
                gpu_module.type(dtype)

            # GPU setup
            gpu_input = to_gpu(cpu_input)
            gpu_target = to_gpu(cpu_target)
            gpu_module.cuda()

            # torch.HalfTensor doesn't support most operations, converting back to default
            if dtype == torch.half:
                cpu_input = self._get_input()
                cpu_target = self._get_target()
                # Loss modules with weights require consistent input/module weight types
                cpu_module = self.constructor(*self.constructor_args)

            cpu_output = test_case._forward_criterion(cpu_module, cpu_input, cpu_target, extra_args=extra_args)
            gpu_output = test_case._forward_criterion(gpu_module, gpu_input, gpu_target, extra_args=extra_args)
            # dtype can be None, so set precision in this way instead of a precision map
            test_case.assertEqual(cpu_output, gpu_output, 1e-1 if dtype == torch.half else 4e-4)

            cpu_gradInput = test_case._backward_criterion(cpu_module, cpu_input, cpu_target, extra_args=extra_args)
            gpu_gradInput = test_case._backward_criterion(gpu_module, gpu_input, gpu_target, extra_args=extra_args)
            test_case.assertEqual(cpu_gradInput, gpu_gradInput, 1e-1 if dtype == torch.half else 4e-4)
        except NotImplementedError:
            pass

    def _get_target(self):
        return self._get_arg('target', False)

    @property
    def constructor_args(self):
        return self._get_arg('constructor_args', False)

    @property
    def extra_args(self):
        return self._get_arg('extra_args', False)

class TestAvgPool(TestCase):
    def _sum_pool2d(self, x, kernel_size):
        windows = torch.nn.functional.unfold(x, kernel_size=kernel_size, stride=kernel_size)
        return torch.sum(windows, dim=1)

    def _sum_pool3d(self, x, kernel_size):
        # Because unfold does not support 3D sliding window we will split tensor to multiple tensors and calculate sum
        h = kernel_size[0]
        splited_x = [t.sum(0) for t in x.split(h) if t.size(0) == h]
        # sum_pool2d assumes tensor in (1, 1, n, m) view, so unsqueeze two times
        splited_x = [self._sum_pool2d(t.unsqueeze(0).unsqueeze(0), kernel_size[1:]) for t in splited_x]
        joined_x = torch.cat(splited_x)
        return joined_x.view(1, joined_x.numel())

    def _avg_pool2d(self, x, kernel_size):
        size = reduce((lambda x, y: x * y), kernel_size)
        return self._sum_pool2d(x, kernel_size) / size

    def _avg_pool3d(self, x, kernel_size):
        size = reduce((lambda x, y: x * y), kernel_size)
        return self._sum_pool3d(x, kernel_size) / size

    def test_doubletensor_avg_pool2d(self):
        n, m = 5, 8
        input = torch.rand(1, 1, n, m)
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                actual = torch.nn.functional.avg_pool2d(input[0], (i, j))
                actual = actual.view(1, actual.numel())
                expected = self._avg_pool2d(input, (i, j))
                self.assertTrue(torch.allclose(actual, expected, rtol=0, atol=1e-5))

    def test_avg_pool2d_with_zero_divisor(self):
        self.assertRaisesRegex(RuntimeError, "divisor must be not zero",
                               lambda: torch.nn.functional.avg_pool2d(torch.zeros(3, 3, 3), (2, 2), divisor_override=0))

    def test_doubletensor_avg_pool2d_with_divisor(self):
        n, m = 3, 3
        input = torch.rand(1, 1, n, m)
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                for divisor in [1, 7, i * j]:
                    actual = torch.nn.functional.avg_pool2d(input[0], (i, j), divisor_override=divisor)
                    actual = actual.view(1, actual.numel())
                    expected = self._sum_pool2d(input, (i, j)) / divisor
                    self.assertTrue(torch.allclose(actual, expected, rtol=0, atol=1e-5))

    def test_doubletensor_avg_pool3d(self):
        h, w, d = 5, 6, 7
        input = torch.rand(h, w, d)
        for i in range(1, h + 1):
            for j in range(1, w + 1):
                for k in range(1, d + 1):
                    actual = torch.nn.functional.avg_pool3d(input.unsqueeze(0), (i, j, k))
                    actual = actual.view(1, actual.numel())
                    expected = self._avg_pool3d(input, (i, j, k))
                    self.assertTrue(torch.allclose(actual, expected, rtol=0, atol=1e-5))

    def test_doubletensor_avg_pool3d_with_divisor(self):
        h, w, d = 6, 5, 7
        input = torch.rand(h, w, d)
        for i in range(1, h + 1):
            for j in range(1, w + 1):
                for k in range(1, d + 1):
                    for divisor in [1, 7, i * j]:
                        actual = torch.nn.functional.avg_pool3d(input.unsqueeze(0), (i, j, k), divisor_override=divisor)
                        actual = actual.view(1, actual.numel())
                        expected = self._sum_pool3d(input, (i, j, k)) / divisor
                        self.assertTrue(torch.allclose(actual, expected, rtol=0, atol=1e-5))

    def test_avg_pool3d_with_zero_divisor(self):
        self.assertRaisesRegex(RuntimeError, "divisor must be not zero",
                               lambda: torch.nn.functional.avg_pool3d(torch.zeros(3, 3, 3, 3), (2, 2, 2), divisor_override=0))

class TestNN(NNTestCase):
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True

    def _forward(self, module, input):
        with freeze_rng_state():
            return module(input)

    def _backward(self, module, input, output, grad_output, create_graph=False):
        output.backward(grad_output, retain_graph=True, create_graph=create_graph)
        if input.grad is None:
            return None
        return input.grad.data

    def _forward_criterion(self, criterion, input, target, extra_args=None):
        if extra_args is None:
            extra_args = tuple()
        if isinstance(input, tuple):
            args = input + (target,) + extra_args
            output = criterion(*args)
        else:
            output = criterion(input, target, *extra_args)
        return output

    def _backward_criterion(self, criterion, input, target, gradOutput=None, extra_args=None):
        if extra_args is None:
            extra_args = tuple()
        input_tuple = input if isinstance(input, tuple) else (input,)
        for i in input_tuple:
            if i.grad is not None:
                i.grad.data.zero_()
        args = input_tuple + (target,) + extra_args
        if gradOutput is None:
            gradOutput = torch.ones(())
        criterion(*args).backward(gradOutput.type_as(input_tuple[0]))
        if isinstance(input, tuple):
            return tuple(map(lambda i: i.grad.data, input))
        else:
            return input.grad.data

    def _zero_grad_parameters(self, module):
        for p in module.parameters():
            if p.grad is not None:
                with torch.no_grad():
                    p.grad.zero_()
                p.grad.detach_()

    def _get_parameters(self, module):
        params = []
        d_params = []
        for p in module.parameters():
            params.append(p)
            d_params.append(p.grad)
        return params, d_params

    def _create_basic_net(self):
        class Layer(nn.Module):
            def __init__(self):
                super(Layer, self).__init__()
                self.layer_dummy_param = Parameter(torch.Tensor(3, 5))
                self.register_buffer('layer_dummy_buf', torch.zeros(1, 3, 3, 7))

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.l1 = Layer()
                self.dummy_param = Parameter(torch.Tensor(3, 5))
                self.register_buffer('dummy_buf', torch.zeros(7, 3, 3, 1))

        l = Layer()
        n = Net()
        s = nn.Sequential(n, n)

        return l, n, s

    @contextlib.contextmanager
    def _compatible_subtest(self, **kwargs):
        # Added for subtest compatibility with Python 2
        if PY3:
            with self.subTest(**kwargs):
                yield
        else:
            yield

    def test_requires_grad_(self):
        m = self._create_basic_net()[-1]
        assert len(list(m.buffers())) > 0, 'invalid test'
        assert all(not b.requires_grad for b in m.buffers()) > 0, 'invalid test'
        assert len(list(m.parameters())) > 0, 'invalid test'
        assert all(p.requires_grad for p in m.parameters()) > 0, 'invalid test'
        for requires_grad in (False, True):
            self.assertIs(m.requires_grad_(requires_grad), m)
            for p in m.parameters():
                self.assertEqual(p.requires_grad, requires_grad)
            for b in m.buffers():
                self.assertFalse(b.requires_grad)

    def test_module_backcompat(self):
        from torch.serialization import SourceChangeWarning
        path = download_file('https://download.pytorch.org/test_data/linear.pt')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', SourceChangeWarning)
            m = torch.load(path)
        input = torch.randn(2, 3, dtype=torch.float)
        self.assertEqual(m(input).size(), (2, 5))

    def test_conv_backcompat(self):
        from torch.serialization import SourceChangeWarning
        # This file was generated by running on PyTorch 1.0.1 on Python 2:
        #
        #     import torch
        #     from torch import nn
        #     m = nn.Conv2d(1, 1, 1)
        #     torch.save(m, 'legacy_conv2d.pt')
        #
        # NB: This Pickle also contains some Unicode data!
        path = download_file('https://download.pytorch.org/test_data/legacy_conv2d.pt')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', SourceChangeWarning)
            if sys.version_info[0] == 2:
                m = torch.load(path)
            else:
                m = torch.load(path, encoding='utf-8')
        input = torch.randn((1, 1, 1, 1), dtype=torch.float)
        self.assertEqual(m(input).size(), (1, 1, 1, 1))

    def test_share_memory(self):
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.p = nn.Parameter(torch.eye(5))
                self.par = nn.ParameterList()
                self.par.append(nn.Parameter(torch.randn(10)))

            def forward(self, inp):
                # NB: dead code
                return inp.clone()

        net = Net()
        for p in net.parameters():
            self.assertFalse(p.storage().is_shared())
        for b in net.buffers():
            self.assertFalse(b.storage().is_shared())
        net.share_memory()
        for p in net.parameters():
            self.assertTrue(p.storage().is_shared())
        for b in net.buffers():
            self.assertTrue(b.storage().is_shared())

    def test_hooks(self):
        module = nn.Sigmoid()
        input = torch.ones(5, 5, requires_grad=True)

        counter = {
            'forwards': 0,
            'backwards': 0
        }

        def fw_hook(inc, h_module, input, output):
            self.assertIsInstance(input, tuple)
            self.assertTrue(isinstance(output, torch.Tensor))
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
        output = bn(torch.randn(5, 5, requires_grad=True))
        output.sum().backward()

    def test_hook_fail(self):
        module = nn.Sigmoid()
        input = torch.randn(5, 5, requires_grad=True)

        def bw_fail1(self, grad_input, grad_output):
            return grad_input[:-1]

        def bw_fail2(self, grad_input, grad_output):
            return grad_input + (torch.randn(2, 2),)

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
        input = torch.randn(5, 5, requires_grad=True)

        def bw_hook(module, grad_input, grad_output):
            for grad in grad_input:
                self.assertTrue(isinstance(grad, torch.Tensor))
            for grad in grad_output:
                self.assertTrue(isinstance(grad, torch.Tensor))
            return tuple(gi * 2 for gi in grad_input)

        module.register_backward_hook(bw_hook)
        module(input).backward(torch.ones(5, 5))
        expected_grad = torch.ones(5, 5).mm(module.weight.data) * 2
        self.assertEqual(input.grad.data, expected_grad)

    def test_hook_mutations(self):
        module = nn.Linear(5, 5)
        input = torch.randn(5, 5, requires_grad=True)

        def forward_pre_hook(m, input):
            return torch.nn.functional.relu(input[0])

        def forward_hook(m, input, output):
            return -output

        module.register_forward_pre_hook(forward_pre_hook)
        module.register_forward_hook(forward_hook)
        output = module(input)
        expected_res = -torch.nn.functional.linear(torch.nn.functional.relu(input), module.weight, module.bias)
        self.assertEqual(output, expected_res)
        output.backward(torch.ones(5, 5) * 2, retain_graph=True)
        mask = (input > 0).double()
        expected_grad = -torch.ones(5, 5).mm(module.weight.data) * 2 * mask
        self.assertEqual(input.grad, expected_grad)



    def test_to(self):
        m = nn.Linear(3, 5)
        self.assertIs(m, m.to('cpu'))
        self.assertIs(m, m.to('cpu', dtype=torch.float32))
        self.assertEqual(m.double(), m.to(torch.float64))
        self.assertRaises(RuntimeError, lambda: m.to('cpu', copy=True))

        if torch.cuda.is_available():
            for cuda in ['cuda', 'cuda:0' if torch.cuda.device_count() == 1 else 'cuda:1']:
                m2 = m.cuda(device=cuda)
                self.assertIs(m2, m2.to(cuda))
                self.assertEqual(m, m2.to('cpu'))
                self.assertEqual(m2, m.to(cuda))
                self.assertIs(m2, m2.to(dtype=torch.float32))
                self.assertEqual(m2.double(), m2.to(dtype=torch.float64))

    def test_zero_grad(self):
        i = torch.randn(2, 5, requires_grad=True)
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
        for dtype in [torch.bfloat16, torch.float, torch.double]:
            module = nn.Conv2d(2, 5, kernel_size=3, padding=1).to(dtype)
            input = torch.randn(1, 2, 10, 10).to(dtype)
            x = input
            y = input.clone()

            output = module(x)
            self.assertTrue(output.requires_grad)
            output.backward(torch.ones(1, 5, 10, 10))

            with torch.no_grad():
                output2 = module(y)
                self.assertFalse(output2.requires_grad)
                self.assertRaises(RuntimeError, lambda: output2.backward(torch.ones(1, 5, 10, 10)))

    def test_invalid_conv1d(self):
        for dtype in [torch.bfloat16, torch.float, torch.double]:
            module = nn.Conv1d(in_channels=3, out_channels=33, kernel_size=10, stride=1, bias=True).to(dtype)
            input = torch.randn(1, 3, 4).to(dtype)
            with self.assertRaisesRegex(RuntimeError,
                                        r'Calculated padded input size per channel: \(4\). ' +
                                        r'Kernel size: \(10\). Kernel size can\'t be greater than actual input size'):
                module(input)

            # Negative stride check
            module = nn.Conv1d(in_channels=3, out_channels=6, kernel_size=3, stride=-1, bias=True).to(dtype)
            input = torch.randn(1, 3, 4).to(dtype)
            with self.assertRaisesRegex(RuntimeError, 'non-positive stride is not supported'):
                module(input)

    def test_mismatch_shape_conv2d(self):
        x = torch.randn(1, 10, 1, 28, 28)
        w = torch.randn(6, 1, 5, 5)

        with self.assertRaisesRegex(RuntimeError,
                                    r'Expected 4-dimensional input for 4-dimensional weight 6 1 5 5,' +
                                    r' but got 5-dimensional input of size \[1, 10, 1, 28, 28\] instead'):

            F.conv2d(x, w)

    def test_invalid_conv2d(self):
        for dtype in [torch.bfloat16, torch.float, torch.double]:
            module = torch.nn.Conv2d(1, 1, kernel_size=3, dilation=2, stride=2).to(dtype)
            input = torch.empty(1, 1, 4, 4).to(dtype)
            self.assertRaises(RuntimeError, lambda: module(input))

            module = nn.Conv2d(in_channels=3, out_channels=33, kernel_size=10, stride=1, bias=True)
            input = torch.randn(1, 3, 1, 1)
            with self.assertRaisesRegex(RuntimeError,
                                        r'Calculated padded input size per channel: \(1 x 1\). ' +
                                        r'Kernel size: \(10 x 10\). Kernel size can\'t be greater than actual input size'):
                module(input)

            # Negative stride check
            module = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=4, stride=-1, bias=True).to(dtype)
            input = torch.randn(1, 3, 4, 4).to(dtype)
            with self.assertRaisesRegex(RuntimeError, 'non-positive stride is not supported'):
                module(input)

            # Zero stride check
            module = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=4, stride=0, bias=True).to(dtype)
            input = torch.randn(1, 3, 4, 4).to(dtype)
            with self.assertRaisesRegex(RuntimeError, 'non-positive stride is not supported'):
                module(input)

    def test_invalid_conv3d(self):
        for dtype in [torch.bfloat16, torch.float, torch.double]:
            module = torch.nn.Conv3d(1, 1, kernel_size=3, dilation=2, stride=2).to(dtype)
            input = torch.empty(1, 1, 4, 4, 4).to(dtype)
            self.assertRaises(RuntimeError, lambda: module(input))

            # Negative stride check
            module = torch.nn.Conv3d(1, 1, kernel_size=3, stride=-2)
            input = torch.empty(1, 1, 4, 4, 4)
            with self.assertRaisesRegex(RuntimeError, 'non-positive stride is not supported'):
                module(input)

    def _test_alpha_dropout(self, cls, input):
        mean = input.mean()
        std = input.std()

        for p in [0.2, 0.5, 0.8]:
            module = cls(p)
            input_var = input.detach().clone().requires_grad_()
            output = module(input_var)
            # output mean should be close to input mean
            self.assertLess(abs(output.data.mean() - mean), 0.1)
            # output std should be close to input std
            self.assertLess(abs(output.data.std() - std), 0.1)
            output.backward(input)

    def test_parameters_and_named_parameters(self):
        def names(named_parameters):
            return [k for k, _ in named_parameters]

        l, n, s = self._create_basic_net()

        self.assertEqual(len(list(l.parameters())), 1)
        self.assertEqual(
            names(l.named_parameters()),
            ['layer_dummy_param'])

        self.assertEqual(len(list(n.parameters())), 2)
        self.assertEqual(
            names(n.named_parameters()),
            ['dummy_param', 'l1.layer_dummy_param'])

        self.assertEqual(len(list(n.parameters(recurse=False))), 1)
        self.assertEqual(
            names(n.named_parameters(recurse=False)),
            ['dummy_param'])

        self.assertEqual(len(list(s.parameters())), 2)
        self.assertEqual(
            names(s.named_parameters()),
            ['0.dummy_param', '0.l1.layer_dummy_param'])

    def test_buffers_and_named_buffers(self):
        def names(named_buffers):
            return [k for k, _ in named_buffers]

        l, n, s = self._create_basic_net()

        self.assertEqual(len(list(l.buffers())), 1)
        self.assertEqual(
            names(l.named_buffers()),
            ['layer_dummy_buf'])

        self.assertEqual(len(list(n.buffers())), 2)
        self.assertEqual(
            names(n.named_buffers()),
            ['dummy_buf', 'l1.layer_dummy_buf'])

        self.assertEqual(len(list(n.buffers(recurse=False))), 1)
        self.assertEqual(
            names(n.named_buffers(recurse=False)),
            ['dummy_buf'])

        self.assertEqual(len(list(s.buffers())), 2)
        self.assertEqual(
            names(s.named_buffers()),
            ['0.dummy_buf', '0.l1.layer_dummy_buf'])

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
        model_output = net(torch.randn([5, 10]))
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
        with self.assertRaises(KeyError):
            s.add_module('', l1)
        with self.assertRaises(KeyError):
            s.add_module('name.with.dot', l1)
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
                self.param = torch.empty(3, 5)

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
                self.param = torch.empty(3, 5)
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

    def test_register_buffer_raises_error_if_name_is_not_string(self):
        m = nn.Module()
        expected_error = 'buffer name should be a string. Got '
        with self.assertRaisesRegex(TypeError, expected_error + 'int'):
            m.register_buffer(1, torch.rand(5))
        with self.assertRaisesRegex(TypeError, expected_error + 'NoneType'):
            m.register_buffer(None, torch.rand(5))

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

    def test_register_parameter_raises_error_if_name_is_not_string(self):
        m = nn.Module()
        expected_error = 'parameter name should be a string. Got '
        with self.assertRaisesRegex(TypeError, expected_error + 'int'):
            m.register_parameter(1, nn.Parameter())
        with self.assertRaisesRegex(TypeError, expected_error + 'NoneType'):
            m.register_parameter(None, nn.Parameter())

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
        modules.insert(1, nn.Linear(3, 2))
        module_list.insert(1, modules[1])
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
        modules[-1] = nn.Conv2d(5, 2, 1)
        module_list[-1] = modules[-1]
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

    def test_ModuleDict(self):
        modules = OrderedDict([
            ('act', nn.ReLU()),
            ('conv', nn.Conv2d(10, 10, 5)),
            ('fc', nn.Linear(5, 5)),
        ])

        module_dict = nn.ModuleDict(modules)

        def check():
            self.assertEqual(len(module_dict), len(modules))
            for k1, m2 in zip(modules, module_dict.children()):
                self.assertIs(modules[k1], m2)
            for k1, k2 in zip(modules, module_dict):
                self.assertIs(modules[k1], module_dict[k2])
            for k in module_dict:
                self.assertIs(module_dict[k], modules[k])
            for k in module_dict.keys():
                self.assertIs(module_dict[k], modules[k])
            for k, v in module_dict.items():
                self.assertIs(modules[k], v)
            for k1, m2 in zip(modules, module_dict.values()):
                self.assertIs(modules[k1], m2)
            for k in modules.keys():
                self.assertTrue(k in module_dict)
        check()

        modules['conv'] = nn.Conv2d(3, 4, 3)
        module_dict['conv'] = modules['conv']
        check()

        next_modules = [
            ('fc2', nn.Linear(5, 5)),
            ('act', nn.Sigmoid()),
        ]
        modules.update(next_modules)
        module_dict.update(next_modules)
        check()

        next_modules = OrderedDict([
            ('fc3', nn.Linear(5, 5)),
            ('act2', nn.Sigmoid()),
        ])
        modules.update(next_modules)
        module_dict.update(next_modules)
        check()

        next_modules = {
            'fc4': nn.Linear(5, 5),
            'act3': nn.Sigmoid()
        }
        modules.update(sorted(next_modules.items()))
        module_dict.update(next_modules)
        check()

        del module_dict['fc']
        del modules['fc']
        check()

        with self.assertRaises(TypeError):
            module_dict.update(nn.ReLU())

        with self.assertRaises(TypeError):
            module_dict.update([nn.ReLU()])

        with self.assertRaises(ValueError):
            module_dict.update([[nn.ReLU()]])

        with self.assertRaises(TypeError):
            module_dict[1] = nn.ReLU()

        s = nn.Sequential(modules)
        module_dict = nn.ModuleDict(s.named_children())
        check()

        c = module_dict.pop('conv')
        self.assertIs(c, modules['conv'])
        modules.pop('conv')
        check()

        module_dict.clear()
        self.assertEqual(len(module_dict), 0)
        modules.clear()
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
        parameters[-1] = make_param()
        param_list[-1] = parameters[-1]
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

    def test_ParameterDict(self):
        parameters = OrderedDict([
            ('p1', Parameter(torch.randn(10, 10))),
            ('p2', Parameter(torch.randn(10, 10))),
            ('p3', Parameter(torch.randn(10, 10))),
        ])

        parameter_dict = nn.ParameterDict(parameters)

        def check():
            self.assertEqual(len(parameter_dict), len(parameters))
            for k1, m2 in zip(parameters, parameter_dict.parameters()):
                self.assertIs(parameters[k1], m2)
            for k1, k2 in zip(parameters, parameter_dict):
                self.assertIs(parameters[k1], parameter_dict[k2])
            for k in parameter_dict:
                self.assertIs(parameter_dict[k], parameters[k])
            for k in parameter_dict.keys():
                self.assertIs(parameter_dict[k], parameters[k])
            for k, v in parameter_dict.items():
                self.assertIs(v, parameters[k])
            for k1, m2 in zip(parameters, parameter_dict.values()):
                self.assertIs(parameters[k1], m2)
            for k in parameters.keys():
                self.assertTrue(k in parameter_dict)

        check()

        parameters['p4'] = Parameter(torch.randn(10, 10))
        parameter_dict['p4'] = parameters['p4']
        check()

        next_parameters = [
            ('p5', Parameter(torch.randn(10, 10))),
            ('p2', Parameter(torch.randn(10, 10))),
        ]
        parameters.update(next_parameters)
        parameter_dict.update(next_parameters)
        check()

        next_parameters = OrderedDict([
            ('p6', Parameter(torch.randn(10, 10))),
            ('p5', Parameter(torch.randn(10, 10))),
        ])
        parameters.update(next_parameters)
        parameter_dict.update(next_parameters)
        check()

        next_parameters = {
            'p8': Parameter(torch.randn(10, 10)),
            'p7': Parameter(torch.randn(10, 10))
        }
        parameters.update(sorted(next_parameters.items()))
        parameter_dict.update(next_parameters)
        check()

        del parameter_dict['p3']
        del parameters['p3']
        check()

        with self.assertRaises(TypeError):
            parameter_dict.update(1)

        with self.assertRaises(TypeError):
            parameter_dict.update([1])

        with self.assertRaises(ValueError):
            parameter_dict.update(Parameter(torch.randn(10, 10)))

        with self.assertRaises(TypeError):
            parameter_dict[1] = Parameter(torch.randn(10, 10))

        p_pop = parameter_dict.pop('p4')
        self.assertIs(p_pop, parameters['p4'])
        parameters.pop('p4')
        check()

        parameter_dict.clear()
        self.assertEqual(len(parameter_dict), 0)
        parameters.clear()
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
        self.assertRaisesRegex(TypeError, 'module name should be a string. Got int',
                               lambda: net.add_module(1, l))
        self.assertRaisesRegex(TypeError, 'module name should be a string. Got NoneType',
                               lambda: net.add_module(None, l))

    def test_module_to_argparse(self):
        net = nn.Sequential(nn.Linear(3, 3))
        cpu = torch.device('cpu')
        with self.assertRaises(TypeError):
            net.to(cpu, True)
        with self.assertRaises(TypeError):
            net.to(torch.long)
        with self.assertRaises(TypeError):
            net.to(None, True)
        with self.assertRaises(TypeError):
            net.to(cpu, torch.long, True)
        with self.assertRaises(TypeError):
            net.to(cpu, dtype=torch.long, non_blocking=True)
        with self.assertRaises(TypeError):
            net.to([])
        with self.assertRaises(TypeError):
            net.to({}, non_blocking=True)
        with self.assertRaises(TypeError):
            net.to(torch.tensor(3, dtype=torch.long), non_blocking=True)
        with self.assertRaises(TypeError):
            net.to(cpu, torch.tensor(3, dtype=torch.long), non_blocking=True)

    def test_RNN_nonlinearity(self):
        rnn = torch.nn.RNN(1, 10)
        self.assertEqual(rnn.nonlinearity, 'tanh')

        rnn = torch.nn.RNN(1, 10, nonlinearity='relu')
        self.assertEqual(rnn.nonlinearity, 'relu')

        with self.assertRaisesRegex(ValueError, 'Unknown nonlinearity'):
            rnn = torch.nn.RNN(1, 10, nonlinearity='garbage')

    def test_module_apply_inplace_op(self):
        def add_one_inplace(t):
            return t.add_(1.0)

        # Test that applying an in-place operation to a module would bump
        # the module's parameters' version counter.
        m = nn.Linear(20, 10)
        pvm = m.weight.mul(m.weight)
        m_weight_version_saved = m.weight._version
        m = m._apply(add_one_inplace)
        self.assertGreater(m.weight._version, m_weight_version_saved)
        with self.assertRaisesRegex(RuntimeError, "modified by an inplace operation"):
            pvm.backward(torch.randn(10, 20))

        # Test that applying an in-place operation to a module would bump
        # the module's parameters' gradients' version counter.
        m = nn.Linear(20, 10)
        m.weight.grad = torch.randn(10, 20).requires_grad_()
        pgm = m.weight.grad.mul(m.weight.grad)
        m_weight_grad_version_saved = m.weight.grad._version
        m = m._apply(add_one_inplace)
        self.assertGreater(m.weight.grad._version, m_weight_grad_version_saved)
        with self.assertRaisesRegex(RuntimeError, "modified by an inplace operation"):
            pgm.backward(torch.randn(10, 20))

    def test_overwrite_module_params_on_conversion(self):
        # Test that if the conversion function passed to `module._apply()`
        # changes the TensorImpl type of `module`'s parameters, the `module`'s
        # parameters are always overwritten, regardless of the value of
        # `torch.__future__.get_overwrite_module_params_on_conversion()`.
        m = nn.Linear(20, 10)
        m.weight.grad = torch.randn(10, 20)
        weight_ref = m.weight
        weight_grad_ref = m.weight.grad
        m = m._apply(lambda t: torch.sparse_coo_tensor(torch.zeros([2, 1]), torch.ones([1]), torch.Size([10, 20])))
        self.assertNotEqual(weight_ref.layout, m.weight.layout)
        self.assertNotEqual(weight_grad_ref.layout, m.weight.grad.layout)

        # Test that under the current default settings
        # (`torch.__future__.get_overwrite_module_params_on_conversion() == False`),
        # a view to a module's parameters is not pointing to the same storage as
        # its base variable after converting the module to a different dtype.
        m = nn.Linear(20, 10).float()
        mw = m.weight[:]
        m.double()
        mw[0][0] = 5
        self.assertTrue(mw[0][0].dtype == torch.float)
        self.assertTrue(mw._base[0][0].dtype == torch.double)

        try:
            torch.__future__.set_overwrite_module_params_on_conversion(True)

            # Test that if `torch.__future__.get_overwrite_module_params_on_conversion() == True`,
            # a view to a module's parameters is still pointing to the same storage as
            # its base variable after converting the module to a different dtype.
            m = nn.Linear(20, 10).float()
            mw = m.weight[:]
            m.double()
            mw[0][0] = 5
            self.assertTrue(mw[0][0] == mw._base[0][0])

            # Test that if `torch.__future__.get_overwrite_module_params_on_conversion() == True`,
            # `float_module.double()` doesn't preserve previous references to
            # `float_module`'s parameters or gradients.
            m = nn.Linear(20, 10).float()
            m.weight.grad = torch.randn(10, 20).float()
            weight_ref = m.weight
            weight_grad_ref = m.weight.grad
            m.double()
            self.assertNotEqual(weight_ref.dtype, m.weight.dtype)
            self.assertNotEqual(weight_grad_ref.dtype, m.weight.grad.dtype)

            def add_one_inplace(t):
                return t.add_(1.0)

            # Test that if `torch.__future__.get_overwrite_module_params_on_conversion() == True`,
            # applying an in-place operation to a module would bump the module's
            # original parameters' version counter.
            m = nn.Linear(20, 10)
            pvm = m.weight.mul(m.weight)
            weight_ref = m.weight
            m_weight_version_saved = weight_ref._version
            m = m._apply(add_one_inplace)
            # Test that the in-place operation bumps the original parameter's version counter
            self.assertGreater(weight_ref._version, m_weight_version_saved)
            with self.assertRaisesRegex(RuntimeError, "modified by an inplace operation"):
                pvm.backward(torch.randn(10, 20))

            # Test that if `torch.__future__.get_overwrite_module_params_on_conversion() == True`,
            # applying an in-place operation to a module would bump the module's
            # original parameters' gradients' version counter.
            m = nn.Linear(20, 10)
            m.weight.grad = torch.randn(10, 20).requires_grad_()
            pgm = m.weight.grad.mul(m.weight.grad)
            weight_grad_ref = m.weight.grad
            m_weight_grad_version_saved = weight_grad_ref._version
            m = m._apply(add_one_inplace)
            self.assertGreater(weight_grad_ref._version, m_weight_grad_version_saved)
            with self.assertRaisesRegex(RuntimeError, "modified by an inplace operation"):
                pgm.backward(torch.randn(10, 20))

            # Test that if `torch.__future__.get_overwrite_module_params_on_conversion() == True`,
            # applying an out-of-place operation to a module doesn't bump
            # the module's original parameters' version counter.
            m = nn.Linear(20, 10)
            weight_ref = m.weight
            m_weight_version_saved = weight_ref._version
            m = m._apply(lambda t: torch.randn(t.shape))
            self.assertEqual(weight_ref._version, m_weight_version_saved)

            # Test that if `torch.__future__.get_overwrite_module_params_on_conversion() == True`,
            # applying an out-of-place operation to a module doesn't bump
            # the module's original parameters' gradients' version counter.
            m = nn.Linear(20, 10)
            m.weight.grad = torch.randn(10, 20).requires_grad_()
            weight_grad_ref = m.weight.grad
            m_weight_grad_version_saved = weight_grad_ref._version
            m = m._apply(lambda t: torch.randn(t.shape))
            self.assertEqual(weight_grad_ref._version, m_weight_grad_version_saved)
        finally:
            torch.__future__.set_overwrite_module_params_on_conversion(False)

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
        net.to(torch.half)
        self.assertIsInstance(l.weight.data, torch.HalfTensor)
        self.assertIsInstance(l.bias.data, torch.HalfTensor)
        self.assertIsInstance(net.indices, torch.LongTensor)
        if TEST_CUDA:
            net.float().cuda()
            self.assertIsInstance(l.weight.data, torch.cuda.FloatTensor)
            self.assertIsInstance(l.bias.data, torch.cuda.FloatTensor)
            self.assertIsInstance(net.indices, torch.cuda.LongTensor)
            net.cpu()
            self.assertIsInstance(l.weight.data, torch.FloatTensor)
            self.assertIsInstance(l.bias.data, torch.FloatTensor)
            self.assertIsInstance(net.indices, torch.LongTensor)
            net.to("cuda", torch.double, True)
            self.assertIsInstance(l.weight.data, torch.cuda.DoubleTensor)
            self.assertIsInstance(l.bias.data, torch.cuda.DoubleTensor)
            self.assertIsInstance(net.indices, torch.cuda.LongTensor)
            net.to(torch.empty(1, device="cuda:0", dtype=torch.half))
            self.assertIsInstance(l.weight.data, torch.cuda.HalfTensor)
            self.assertIsInstance(l.bias.data, torch.cuda.HalfTensor)
            self.assertIsInstance(net.indices, torch.cuda.LongTensor)
        net.to(torch.device("cpu"), non_blocking=True)
        self.assertIsInstance(l.weight.data, torch.HalfTensor)
        self.assertIsInstance(l.bias.data, torch.HalfTensor)
        self.assertIsInstance(net.indices, torch.LongTensor)
        net.type(torch.FloatTensor)
        self.assertIsInstance(l.weight.data, torch.FloatTensor)
        self.assertIsInstance(l.bias.data, torch.FloatTensor)
        net.to(torch.DoubleTensor(1))
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
            if norm_type != inf:
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

        grads = torch.arange(1., 101).view(10, 10), torch.ones(10).div(1000)
        for norm_type in [0.5, 1.5, 2, 4, 'inf']:
            for p, g in zip(l.parameters(), grads):
                p._grad = g.clone().view_as(p.data)
            norm_before = compute_norm(norm_type)
            norm = clip_grad_norm_(l.parameters(), max_norm, norm_type=norm_type)
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
            norm = clip_grad_norm_(l.parameters(), max_norm, norm_type=norm_type)
            norm_after = compute_norm(norm_type)
            self.assertEqual(norm, norm_before)
            self.assertEqual(norm_before, norm_after)
            self.assertLessEqual(norm_after, max_norm)
            scale = compare_scaling(grads)
            self.assertEqual(scale, 1)

        # Should accept a single Tensor as input
        p1, p2 = torch.randn(10, 10), torch.randn(10, 10)
        g = torch.arange(1., 101).view(10, 10)
        p1._grad = g.clone()
        p2._grad = g.clone()
        for norm_type in [0.5, 1.5, 2, 4, 'inf']:
            clip_grad_norm_(p1, max_norm, norm_type=norm_type)
            clip_grad_norm_([p2], max_norm, norm_type=norm_type)
            self.assertEqual(p1.grad, p2.grad)

    def test_clip_grad_value(self):
        l = nn.Linear(10, 10)
        clip_value = 2.5

        grad_w, grad_b = torch.arange(-50., 50).view(10, 10).div_(5), torch.ones(10).mul_(2)
        for grad_list in [[grad_w, grad_b], [grad_w, None]]:
            for p, g in zip(l.parameters(), grad_list):
                p._grad = g.clone().view_as(p.data) if g is not None else g

            clip_grad_value_(l.parameters(), clip_value)
            for p in filter(lambda p: p.grad is not None, l.parameters()):
                self.assertLessEqual(p.grad.data.max(), clip_value)
                self.assertGreaterEqual(p.grad.data.min(), -clip_value)

        # Should accept a single Tensor as input
        p1, p2 = torch.randn(10, 10), torch.randn(10, 10)
        g = torch.arange(-50., 50).view(10, 10).div_(5)
        p1._grad = g.clone()
        p2._grad = g.clone()
        clip_grad_value_(p1, clip_value)
        clip_grad_value_([p2], clip_value)
        self.assertEqual(p1.grad, p2.grad)

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

        vec = torch.arange(0., 980)
        vector_to_parameters(vec, model.parameters())

        sample = next(model.parameters())[0, 0, 0]
        self.assertTrue(torch.equal(sample.data, vec.data[:5]))

    # torch/nn/utils/prune.py
    @unittest.skipIf(not TEST_NUMPY, "numpy not found")
    def test_validate_pruning_amount_init(self):
        r"""Test the first util function that validates the pruning
        amount requested by the user the moment the pruning method
        is initialized. This test checks that the expected errors are
        raised whenever the amount is invalid.
        The original function runs basic type checking + value range checks.
        It doesn't check the validity of the pruning amount with
        respect to the size of the tensor to prune. That's left to
        `_validate_pruning_amount`, tested below.
        """
        # neither float not int should raise TypeError
        with self.assertRaises(TypeError):
            prune._validate_pruning_amount_init(amount="I'm a string")

        # float not in [0, 1] should raise ValueError
        with self.assertRaises(ValueError):
            prune._validate_pruning_amount_init(amount=1.1)
        with self.assertRaises(ValueError):
            prune._validate_pruning_amount_init(amount=20.)

        # negative int should raise ValueError
        with self.assertRaises(ValueError):
            prune._validate_pruning_amount_init(amount=-10)

        # all these should pass without errors because they're valid amounts
        prune._validate_pruning_amount_init(amount=0.34)
        prune._validate_pruning_amount_init(amount=1500)
        prune._validate_pruning_amount_init(amount=0)
        prune._validate_pruning_amount_init(amount=0.)
        prune._validate_pruning_amount_init(amount=1)
        prune._validate_pruning_amount_init(amount=1.)
        self.assertTrue(True)

    @unittest.skipIf(not TEST_NUMPY, "numpy not found")
    def test_validate_pruning_amount(self):
        r"""Tests the second util function that validates the pruning
        amount requested by the user, this time with respect to the size
        of the tensor to prune. The rationale is that if the pruning amount,
        converted to absolute value of units to prune, is larger than
        the number of units in the tensor, then we expect the util function
        to raise a value error.
        """
        # if amount is int and amount > tensor_size, raise ValueError
        with self.assertRaises(ValueError):
            prune._validate_pruning_amount(amount=20, tensor_size=19)

        # amount is a float so this should not raise an error
        prune._validate_pruning_amount(amount=0.3, tensor_size=0)

        # this is okay
        prune._validate_pruning_amount(amount=19, tensor_size=20)
        prune._validate_pruning_amount(amount=0, tensor_size=0)
        prune._validate_pruning_amount(amount=1, tensor_size=1)
        self.assertTrue(True)

    @unittest.skipIf(not TEST_NUMPY, "numpy not found")
    def test_compute_nparams_to_prune(self):
        r"""Test that requested pruning `amount` gets translated into the
        correct absolute number of units to prune.
        """
        self.assertEqual(
            prune._compute_nparams_toprune(amount=0, tensor_size=15),
            0
        )
        self.assertEqual(
            prune._compute_nparams_toprune(amount=10, tensor_size=15),
            10
        )
        # if 1 is int, means 1 unit
        self.assertEqual(
            prune._compute_nparams_toprune(amount=1, tensor_size=15),
            1
        )
        # if 1. is float, means 100% of units
        self.assertEqual(
            prune._compute_nparams_toprune(amount=1., tensor_size=15),
            15
        )
        self.assertEqual(
            prune._compute_nparams_toprune(amount=0.4, tensor_size=17),
            7
        )

    def test_random_pruning_sizes(self):
        r"""Test that the new parameters and buffers created by the pruning
        method have the same size as the input tensor to prune. These, in
        fact, correspond to the pruned version of the tensor itself, its
        mask, and its original copy, so the size must match.
        """
        # fixturize test
        # TODO: add other modules
        modules = [nn.Linear(5, 7), nn.Conv3d(2, 2, 2)]
        names = ['weight', 'bias']

        for m in modules:
            for name in names:
                with self._compatible_subtest(m=m, name=name):
                    original_tensor = getattr(m, name)

                    prune.random_unstructured(m, name=name, amount=0.1)
                    # mask has the same size as tensor being pruned
                    self.assertEqual(
                        original_tensor.size(),
                        getattr(m, name + '_mask').size()
                    )
                    # 'orig' tensor has the same size as the original tensor
                    self.assertEqual(
                        original_tensor.size(),
                        getattr(m, name + '_orig').size()
                    )
                    # new tensor has the same size as the original tensor
                    self.assertEqual(
                        original_tensor.size(),
                        getattr(m, name).size()
                    )

    def test_random_pruning_orig(self):
        r"""Test that original tensor is correctly stored in 'orig'
        after pruning is applied. Important to make sure we don't
        lose info about the original unpruned parameter.
        """
        # fixturize test
        # TODO: add other modules
        modules = [nn.Linear(5, 7), nn.Conv3d(2, 2, 2)]
        names = ['weight', 'bias']

        for m in modules:
            for name in names:
                with self._compatible_subtest(m=m, name=name):

                    # tensor prior to pruning
                    original_tensor = getattr(m, name)
                    prune.random_unstructured(m, name=name, amount=0.1)
                    self.assertEqual(
                        original_tensor,
                        getattr(m, name + '_orig')
                    )

    def test_random_pruning_new_weight(self):
        r"""Test that module.name now contains a pruned version of
        the original tensor obtained from multiplying it by the mask.
        """
        # fixturize test
        # TODO: add other modules
        modules = [nn.Linear(5, 7), nn.Conv3d(2, 2, 2)]
        names = ['weight', 'bias']

        for m in modules:
            for name in names:
                with self._compatible_subtest(m=m, name=name):
                    # tensor prior to pruning
                    original_tensor = getattr(m, name)
                    prune.random_unstructured(m, name=name, amount=0.1)
                    # weight = weight_orig * weight_mask
                    self.assertEqual(
                        getattr(m, name),
                        getattr(m, name + '_orig')
                        * getattr(m, name + '_mask').to(
                            dtype=original_tensor.dtype
                        ),
                    )

    def test_identity_pruning(self):
        r"""Test that a mask of 1s does not change forward or backward.
        """
        input_ = torch.ones(1, 5)
        m = nn.Linear(5, 2)
        y_prepruning = m(input_)  # output prior to pruning

        # compute grad pre-pruning and check it's equal to all ones
        y_prepruning.sum().backward()
        old_grad_weight = m.weight.grad.clone()  # don't grab pointer!
        self.assertEqual(old_grad_weight, torch.ones_like(m.weight))
        old_grad_bias = m.bias.grad.clone()
        self.assertEqual(old_grad_bias, torch.ones_like(m.bias))

        # remove grads
        m.zero_grad()

        # force the mask to be made of all 1s
        prune.identity(m, name="weight")

        # with mask of 1s, output should be identical to no mask
        y_postpruning = m(input_)
        self.assertEqual(y_prepruning, y_postpruning)

        # with mask of 1s, grad should be identical to no mask
        y_postpruning.sum().backward()
        self.assertEqual(old_grad_weight, m.weight_orig.grad)
        self.assertEqual(old_grad_bias, m.bias.grad)

        # calling forward twice in a row shouldn't change output
        y1 = m(input_)
        y2 = m(input_)
        self.assertEqual(y1, y2)

    @unittest.skipIf(not PY3, "mock is not available in Python 2")
    def test_random_pruning_0perc(self):
        r"""Test that a mask of 1s does not change forward or backward.
        """
        input_ = torch.ones(1, 5)
        m = nn.Linear(5, 2)
        y_prepruning = m(input_)  # output prior to pruning

        # compute grad pre-pruning and check it's equal to all ones
        y_prepruning.sum().backward()
        old_grad_weight = m.weight.grad.clone()  # don't grab pointer!
        self.assertEqual(old_grad_weight, torch.ones_like(m.weight))
        old_grad_bias = m.bias.grad.clone()
        self.assertEqual(old_grad_bias, torch.ones_like(m.bias))

        # remove grads
        m.zero_grad()

        # force the mask to be made of all 1s
        with mock.patch(
            "torch.nn.utils.prune.RandomUnstructured.compute_mask"
        ) as compute_mask:
            compute_mask.return_value = torch.ones_like(m.weight)
            prune.random_unstructured(m, name='weight', amount=0.9)  # amount won't count

        # with mask of 1s, output should be identical to no mask
        y_postpruning = m(input_)
        self.assertEqual(y_prepruning, y_postpruning)

        # with mask of 1s, grad should be identical to no mask
        y_postpruning.sum().backward()
        self.assertEqual(old_grad_weight, m.weight_orig.grad)
        self.assertEqual(old_grad_bias, m.bias.grad)

        # calling forward twice in a row shouldn't change output
        y1 = m(input_)
        y2 = m(input_)
        self.assertEqual(y1, y2)

    @unittest.skipIf(not PY3, "mock is not available in Python 2")
    def test_random_pruning(self):
        input_ = torch.ones(1, 5)
        m = nn.Linear(5, 2)

        # define custom mask to assign with mock
        mask = torch.ones_like(m.weight)
        mask[1, 0] = 0
        mask[0, 3] = 0

        # check grad is zero for masked weights
        with mock.patch(
            "torch.nn.utils.prune.RandomUnstructured.compute_mask"
        ) as compute_mask:
            compute_mask.return_value = mask
            prune.random_unstructured(m, name='weight', amount=0.9)

        y_postpruning = m(input_)
        y_postpruning.sum().backward()
        # weight_orig is the parameter, so it's the tensor that will accumulate the grad
        self.assertEqual(m.weight_orig.grad, mask)  # all 1s, except for masked units
        self.assertEqual(m.bias.grad, torch.ones_like(m.bias))

        # make sure that weight_orig update doesn't modify [1, 0] and [0, 3]
        old_weight_orig = m.weight_orig.clone()
        # update weights
        learning_rate = 1.
        for p in m.parameters():
            p.data.sub_(p.grad.data * learning_rate)
        # since these are pruned, they should not be updated
        self.assertEqual(old_weight_orig[1, 0], m.weight_orig[1, 0])
        self.assertEqual(old_weight_orig[0, 3], m.weight_orig[0, 3])

    @unittest.skipIf(not PY3, "mock is not available in Python 2")
    def test_random_pruning_forward(self):
        r"""check forward with mask (by hand).
        """
        input_ = torch.ones(1, 5)
        m = nn.Linear(5, 2)

        # define custom mask to assign with mock
        mask = torch.zeros_like(m.weight)
        mask[1, 0] = 1
        mask[0, 3] = 1

        with mock.patch(
            "torch.nn.utils.prune.RandomUnstructured.compute_mask"
        ) as compute_mask:
            compute_mask.return_value = mask
            prune.random_unstructured(m, name='weight', amount=0.9)

        yhat = m(input_)
        self.assertEqual(yhat[0, 0], m.weight_orig[0, 3] + m.bias[0])
        self.assertEqual(yhat[0, 1], m.weight_orig[1, 0] + m.bias[1])

    @unittest.skipIf(not PY3, "mock is not available in Python 2")
    def test_remove_pruning_forward(self):
        r"""Remove pruning and check forward is unchanged from previous
        pruned state.
        """
        input_ = torch.ones(1, 5)
        m = nn.Linear(5, 2)

        # define custom mask to assign with mock
        mask = torch.ones_like(m.weight)
        mask[1, 0] = 0
        mask[0, 3] = 0

        # check grad is zero for masked weights
        with mock.patch(
            "torch.nn.utils.prune.RandomUnstructured.compute_mask"
        ) as compute_mask:
            compute_mask.return_value = mask
            prune.random_unstructured(m, name='weight', amount=0.9)

        y_postpruning = m(input_)

        prune.remove(m, 'weight')

        y_postremoval = m(input_)
        self.assertEqual(y_postpruning, y_postremoval)

    def test_pruning_id_consistency(self):
        r"""Test that pruning doesn't change the id of the parameters, which
        would otherwise introduce issues with pre-existing optimizers that
        point to old parameters.
        """
        m = nn.Linear(5, 2, bias=False)

        tensor_id = id(list(m.parameters())[0])

        prune.random_unstructured(m, name="weight", amount=0.9)
        self.assertEqual(tensor_id, id(list(m.parameters())[0]))

        prune.remove(m, "weight")
        self.assertEqual(tensor_id, id(list(m.parameters())[0]))

    def test_random_pruning_pickle(self):
        modules = [nn.Linear(5, 7), nn.Conv3d(2, 2, 2)]
        names = ['weight', 'bias']

        for m in modules:
            for name in names:
                with self._compatible_subtest(m=m, name=name):
                    prune.random_unstructured(m, name=name, amount=0.1)
                    m_new = pickle.loads(pickle.dumps(m))
                    self.assertIsInstance(m_new, type(m))

    def test_multiple_pruning_calls(self):
        # if you call pruning twice, the hook becomes a PruningContainer
        m = nn.Conv3d(2, 2, 2)
        prune.l1_unstructured(m, name='weight', amount=0.1)
        weight_mask0 = m.weight_mask  # save it for later sanity check

        # prune again
        prune.ln_structured(m, name='weight', amount=0.3, n=2, dim=0)
        hook = next(iter(m._forward_pre_hooks.values()))
        self.assertIsInstance(
            hook,
            torch.nn.utils.prune.PruningContainer
        )
        # check that container._tensor_name is correctly set no matter how
        # many pruning methods are in the container
        self.assertEqual(hook._tensor_name, 'weight')

        # check that the pruning container has the right length
        # equal to the number of pruning iters
        self.assertEqual(len(hook), 2)  # m.weight has been pruned twice

        # check that the entries of the pruning container are of the expected
        # type and in the expected order
        self.assertIsInstance(hook[0], torch.nn.utils.prune.L1Unstructured)
        self.assertIsInstance(hook[1], torch.nn.utils.prune.LnStructured)

        # check that all entries that are 0 in the 1st mask are 0 in the
        # 2nd mask too
        self.assertTrue(torch.all(m.weight_mask[weight_mask0 == 0] == 0))

        # prune again
        prune.ln_structured(m, name='weight', amount=0.1, n=float('inf'), dim=1)
        # check that container._tensor_name is correctly set no matter how
        # many pruning methods are in the container
        hook = next(iter(m._forward_pre_hooks.values()))
        self.assertEqual(hook._tensor_name, 'weight')

    def test_pruning_container(self):
        # create an empty container
        container = prune.PruningContainer()
        container._tensor_name = 'test'
        self.assertEqual(len(container), 0)

        p = prune.L1Unstructured(amount=2)
        p._tensor_name = 'test'

        # test adding a pruning method to a container
        container.add_pruning_method(p)

        # test error raised if tensor name is different
        q = prune.L1Unstructured(amount=2)
        q._tensor_name = 'another_test'
        with self.assertRaises(ValueError):
            container.add_pruning_method(q)

        # test that adding a non-pruning method object to a pruning container
        # raises a TypeError
        with self.assertRaises(TypeError):
            container.add_pruning_method(10)
        with self.assertRaises(TypeError):
            container.add_pruning_method('ugh')

    def test_pruning_container_compute_mask(self):
        r"""Test `compute_mask` of pruning container with a known `t` and
        `default_mask`. Indirectly checks that Ln structured pruning is
        acting on the right axis.
        """
        # create an empty container
        container = prune.PruningContainer()
        container._tensor_name = 'test'

        # 1) test unstructured pruning
        # create a new pruning method
        p = prune.L1Unstructured(amount=2)
        p._tensor_name = 'test'
        # add the pruning method to the container
        container.add_pruning_method(p)

        # create tensor to be pruned
        t = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]).to(dtype=torch.float32)
        # create prior mask by hand
        default_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 1]])
        # since we are pruning the two lowest magnitude units, the outcome of
        # the calculation should be this:
        expected_mask = torch.tensor([[0, 0, 1, 0], [1, 1, 0, 1]])
        computed_mask = container.compute_mask(t, default_mask)
        self.assertEqual(expected_mask, computed_mask)

        # 2) test structured pruning
        q = prune.LnStructured(amount=1, n=2, dim=0)
        q._tensor_name = 'test'
        container.add_pruning_method(q)
        # since we are pruning the lowest magnitude one of the two rows, the
        # outcome of the calculation should be this:
        expected_mask = torch.tensor([[0, 0, 0, 0], [1, 1, 0, 1]])
        computed_mask = container.compute_mask(t, default_mask)
        self.assertEqual(expected_mask, computed_mask)

        # 2) test structured pruning, along another axis
        r = prune.LnStructured(amount=1, n=2, dim=1)
        r._tensor_name = 'test'
        container.add_pruning_method(r)
        # since we are pruning the lowest magnitude of the four columns, the
        # outcome of the calculation should be this:
        expected_mask = torch.tensor([[0, 1, 1, 0], [0, 1, 0, 1]])
        computed_mask = container.compute_mask(t, default_mask)
        self.assertEqual(expected_mask, computed_mask)


    def test_l1_unstructured_pruning(self):
        r"""Test that l1 unstructured pruning actually removes the lowest
        entries by l1 norm (by hand). It also checks that applying l1
        unstructured pruning more than once respects the previous mask.
        """
        m = nn.Linear(4, 2)
        # modify its weight matrix by hand
        m.weight = torch.nn.Parameter(
            torch.tensor(
                [[1, 2, 3, 4], [-4, -3, -2, -1]], dtype=torch.float32
            )
        )

        prune.l1_unstructured(m, 'weight', amount=2)
        expected_weight = torch.tensor([[0, 2, 3, 4], [-4, -3, -2, 0]])
        self.assertEqual(expected_weight, m.weight)

        # check that pruning again removes the next two smallest entries
        prune.l1_unstructured(m, 'weight', amount=2)
        expected_weight = torch.tensor([[0, 0, 3, 4], [-4, -3, 0, 0]])
        self.assertEqual(expected_weight, m.weight)

    def test_unstructured_pruning_same_magnitude(self):
        r"""Since it may happen that the tensor to prune has entries with the
        same exact magnitude, it is important to check that pruning happens
        consistenly based on the bottom % of weights, and not by threshold,
        which would instead kill off *all* units with magnitude = threshold.
        """
        AMOUNT = 0.2
        p = prune.L1Unstructured(amount=AMOUNT)
        # create a random tensors with entries in {-2, 0, 2}
        t = 2 * torch.randint(low=-1, high=2, size=(10, 7))
        nparams_toprune = prune._compute_nparams_toprune(AMOUNT, t.nelement())

        computed_mask = p.compute_mask(t, default_mask=torch.ones_like(t))
        nparams_pruned = torch.sum(computed_mask == 0)
        self.assertEqual(nparams_toprune, nparams_pruned)

    def test_random_structured_pruning_amount(self):

        AMOUNT = 0.6
        AXIS = 2
        p = prune.RandomStructured(amount=AMOUNT, dim=AXIS)
        t = 2 * torch.randint(low=-1, high=2, size=(5, 4, 2)).to(
            dtype=torch.float32
        )
        nparams_toprune = prune._compute_nparams_toprune(AMOUNT, t.shape[AXIS])

        computed_mask = p.compute_mask(t, default_mask=torch.ones_like(t))
        # check that 1 column is fully prune, the others are left untouched
        remaining_axes = [_ for _ in range(len(t.shape)) if _ != AXIS]
        per_column_sums = sorted(
            torch.sum(computed_mask == 0, axis=remaining_axes)
        )
        assert per_column_sums == [0, 20]


    def test_ln_structured_pruning(self):
        r"""Check Ln structured pruning by hand.
        """
        m = nn.Conv2d(3, 1, 2)
        m.weight.data = torch.Tensor(
            [[[[1., 2.], [1., 2.5]],
             [[0.5, 1.], [0.1, 0.1]],
             [[-3., -5.], [0.1, -1.]]]]
        )
        # expected effect of pruning 1 of the 3 channels by L2-norm
        expected_mask_axis1 = torch.ones_like(m.weight)
        expected_mask_axis1[:, 1] = 0.

        prune.ln_structured(m, 'weight', amount=1, n=2, dim=1)
        self.assertEqual(expected_mask_axis1, m.weight_mask)

        # expected effect of pruning 1 of the 2 columns along axis -1 by L1-norm
        expected_mask_axis3 = expected_mask_axis1
        expected_mask_axis3[:, :, :, 0] = 0.

        prune.ln_structured(m, 'weight', amount=1, n=1, dim=-1)
        self.assertEqual(expected_mask_axis3, m.weight_mask)


    def test_remove_pruning(self):
        r"""`prune.remove` removes the hook and the reparametrization
        and makes the pruning final in the original parameter.
        """
        modules = [nn.Linear(5, 7), nn.Conv3d(2, 2, 2)]
        names = ['weight', 'bias']

        for m in modules:
            for name in names:
                with self._compatible_subtest(m=m, name=name):
                    # first prune
                    prune.random_unstructured(m, name, amount=0.5)
                    self.assertIn(name + "_orig", dict(m.named_parameters()))
                    self.assertIn(name + "_mask", dict(m.named_buffers()))
                    self.assertNotIn(name, dict(m.named_parameters()))
                    self.assertTrue(hasattr(m, name))
                    pruned_t = getattr(m, name)

                    # then remove pruning
                    prune.remove(m, name)
                    self.assertIn(name, dict(m.named_parameters()))
                    self.assertNotIn(name + "_orig", dict(m.named_parameters()))
                    self.assertNotIn(name + "_mask", dict(m.named_buffers()))
                    final_t = getattr(m, name)

                    self.assertEqual(pruned_t, final_t)

    def test_remove_pruning_exception(self):
        r"""Removing from an unpruned tensor throws an assertion error
        """
        modules = [nn.Linear(5, 7), nn.Conv3d(2, 2, 2)]
        names = ['weight', 'bias']

        for m in modules:
            for name in names:
                with self._compatible_subtest(m=m, name=name):
                    # check that the module isn't pruned
                    self.assertFalse(prune.is_pruned(m))
                    # since it isn't pruned, pruning can't be removed from it
                    with self.assertRaises(ValueError):
                        prune.remove(m, name)


    def test_global_pruning(self):
        r"""Test that global l1 unstructured pruning over 2 parameters removes
        the `amount=4` smallest global weights across the 2 parameters.
        """
        m = nn.Linear(4, 2)
        n = nn.Linear(3, 1)
        # modify the weight matrices by hand
        m.weight = torch.nn.Parameter(
            torch.tensor([[1, 2, 3, 4], [-4, -3, -2, -1]]).to(
                dtype=torch.float32)
        )
        n.weight = torch.nn.Parameter(
            torch.tensor([[0, 0.1, -2]]).to(
                dtype=torch.float32)
        )

        params_to_prune = (
            (m, 'weight'),
            (n, 'weight'),
        )

        # prune the 4 smallest weights globally by L1 magnitude
        prune.global_unstructured(
            params_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=4
        )

        expected_mweight = torch.tensor([[0, 2, 3, 4], [-4, -3, -2, 0]])
        self.assertEqual(expected_mweight, m.weight)

        expected_nweight = torch.tensor([[0, 0, -2]]).to(dtype=n.weight.dtype)
        self.assertEqual(expected_nweight, n.weight)


    def test_custom_from_mask_pruning(self):
        r"""Test that the CustomFromMask is capable of receiving
        as input at instantiation time a custom mask, and combining it with
        the previous default mask to generate the correct final mask.
        """
        # new mask
        mask = torch.tensor([[0, 1, 1, 0], [0, 0, 1, 1]])
        # old mask
        default_mask = torch.tensor([[0, 0, 0, 0], [1, 1, 1, 1]])

        # some tensor (not actually used)
        t = torch.rand_like(mask.to(dtype=torch.float32))

        p = prune.CustomFromMask(mask=mask)

        computed_mask = p.compute_mask(t, default_mask)
        expected_mask = torch.tensor([[0, 0, 0, 0], [0, 0, 1, 1]]).to(
            dtype=t.dtype
        )

        self.assertEqual(computed_mask, expected_mask)

    @unittest.skipIf(not PY3, "mock is not available in Python 2")
    def test_pruning_rollback(self):
        r"""Test that if something fails when the we try to compute the mask,
        then the model isn't left in some intermediate half-pruned state.
        The try/except statement in `apply` should handle rolling back
        to the previous state before pruning began.
        """
        modules = [nn.Linear(5, 7), nn.Conv3d(2, 2, 2)]
        names = ['weight', 'bias']

        for m in modules:
            for name in names:
                with self._compatible_subtest(m=m, name=name):

                    with mock.patch(
                        "torch.nn.utils.prune.L1Unstructured.compute_mask"
                    ) as compute_mask:
                        compute_mask.side_effect = Exception('HA!')
                        with self.assertRaises(Exception):
                            prune.l1_unstructured(m, name=name, amount=0.9)

                        self.assertTrue(
                            name in dict(m.named_parameters())
                        )
                        self.assertFalse(
                            name + '_mask' in dict(m.named_buffers())
                        )
                        self.assertFalse(
                            name + '_orig' in dict(m.named_parameters())
                        )

    def test_pruning_serialization_model(self):
        # create a model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 1),
        )
        # check that everything looks normal before pruning
        self.assertNotIn('0.weight_orig', model.state_dict())
        self.assertNotIn('0.weight_mask', model.state_dict())
        self.assertIn('0.weight', model.state_dict())

        # prune one of its parameters
        prune.l1_unstructured(module=model[0], name='weight', amount=0.9)

        # check that the original weight and the new mask are present
        self.assertIn('0.weight_orig', model.state_dict())
        self.assertIn('0.weight_mask', model.state_dict())
        self.assertNotIn('0.weight', model.state_dict())
        self.assertTrue(hasattr(model[0], 'weight'))

        pruned_weight = model[0].weight

        with TemporaryFileName() as fname:
            torch.save(model, fname)
            new_model = torch.load(fname)

        # check that the original weight and the new mask are present
        self.assertIn('0.weight_orig', new_model.state_dict())
        self.assertIn('0.weight_mask', new_model.state_dict())
        self.assertNotIn('0.weight', new_model.state_dict())
        self.assertTrue(hasattr(new_model[0], 'weight'))

        self.assertEqual(pruned_weight, new_model[0].weight)


    def test_pruning_serialization_state_dict(self):
        # create a model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 1),
        )
        # check that everything looks normal before pruning
        self.assertNotIn('0.weight_orig', model.state_dict())
        self.assertNotIn('0.weight_mask', model.state_dict())
        self.assertIn('0.weight', model.state_dict())

        # prune one of its parameters
        prune.l1_unstructured(module=model[0], name='weight', amount=0.9)

        # check that the original weight and the new mask are present
        self.assertIn('0.weight_orig', model.state_dict())
        self.assertIn('0.weight_mask', model.state_dict())
        self.assertNotIn('0.weight', model.state_dict())
        self.assertTrue(hasattr(model[0], 'weight'))

        pruned_weight = model[0].weight

        # make pruning permanent and restore parameter names as in base
        # architecture
        prune.remove(module=model[0], name='weight')

        # check that the original weight and the new mask are no longer present
        self.assertNotIn('0.weight_orig', model.state_dict())
        self.assertNotIn('0.weight_mask', model.state_dict())
        self.assertIn('0.weight', model.state_dict())

        # save the state dict of model and reload it into new_model
        new_model = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 1),
        )
        with TemporaryFileName() as fname:
            torch.save(model.state_dict(), fname)
            new_model.load_state_dict(torch.load(fname))

        # check that the original weight and the new mask are not present in
        # new_model either.
        self.assertNotIn('0.weight_orig', new_model.state_dict())
        self.assertNotIn('0.weight_mask', new_model.state_dict())
        self.assertIn('0.weight', new_model.state_dict())

        self.assertEqual(pruned_weight, new_model[0].weight)


    def test_prune(self):
        # create a new pruning method
        p = prune.L1Unstructured(amount=2)
        # create tensor to be pruned
        t = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]).to(dtype=torch.float32)
        # create prior mask by hand
        default_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 1]])
        # since we are pruning the two lowest magnitude units, the outcome of
        # the calculation should be this:
        expected_mask = torch.tensor([[0, 0, 1, 0], [1, 1, 0, 1]])
        pruned_tensor = p.prune(t, default_mask)
        self.assertEqual(t * expected_mask, pruned_tensor)


    def test_weight_norm(self):
        input = torch.randn(3, 5)
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

        # test with dim=None
        m = nn.Linear(5, 7)
        expected_output = m(input)
        m = torch.nn.utils.weight_norm(m, dim=None)
        self.assertEqual(m(input), expected_output)

        with self.assertRaisesRegex(RuntimeError, 'register two weight_norm hooks'):
            m = torch.nn.utils.weight_norm(m)
            m = torch.nn.utils.weight_norm(m)

    def test_weight_norm_pickle(self):
        m = torch.nn.utils.weight_norm(nn.Linear(5, 7))
        m = pickle.loads(pickle.dumps(m))
        self.assertIsInstance(m, nn.Linear)

    def test_spectral_norm(self):
        input = torch.randn(3, 5)
        m = nn.Linear(5, 7)
        m = torch.nn.utils.spectral_norm(m)

        self.assertEqual(m.weight_u.size(), torch.Size([m.weight.size(0)]))
        # weight_orig should be trainable
        self.assertTrue(hasattr(m, 'weight_orig'))
        self.assertTrue('weight_orig' in m._parameters)
        # weight_u should be just a reused buffer
        self.assertTrue(hasattr(m, 'weight_u'))
        self.assertTrue('weight_u' in m._buffers)
        self.assertTrue('weight_v' in m._buffers)
        # weight should be a plain attribute, not counted as a buffer or a param
        self.assertFalse('weight' in m._buffers)
        self.assertFalse('weight' in m._parameters)
        # it should also be sharing storage as `weight_orig`
        self.assertEqual(m.weight_orig.storage(), m.weight.storage())
        self.assertEqual(m.weight_orig.size(), m.weight.size())
        self.assertEqual(m.weight_orig.stride(), m.weight.stride())

        m = torch.nn.utils.remove_spectral_norm(m)
        self.assertFalse(hasattr(m, 'weight_orig'))
        self.assertFalse(hasattr(m, 'weight_u'))
        # weight should be converted back as a parameter
        self.assertTrue(hasattr(m, 'weight'))
        self.assertTrue('weight' in m._parameters)

        with self.assertRaisesRegex(RuntimeError, 'register two spectral_norm hooks'):
            m = torch.nn.utils.spectral_norm(m)
            m = torch.nn.utils.spectral_norm(m)

        # test correctness in training/eval modes and cpu/multi-gpu settings
        for apply_dp in (True, False):
            if apply_dp:
                if not TEST_MULTIGPU:
                    continue
                device = torch.device('cuda:0')

                def maybe_wrap(m):
                    return torch.nn.DataParallel(m, [0, 1])
            else:
                device = torch.device('cpu')

                def maybe_wrap(m):
                    return m

            for requires_grad in (True, False):
                m = nn.Linear(3, 4).to(device)
                m.weight.requires_grad_(requires_grad)
                m = torch.nn.utils.spectral_norm(m)
                wrapped_m = maybe_wrap(m)
                self.assertTrue(hasattr(m, 'weight_u'))
                u0 = m.weight_u.clone()
                v0 = m.weight_v.clone()

                # TEST TRAINING BEHAVIOR

                # assert that u and v are updated
                input = torch.randn(2, 3, device=device)
                out = wrapped_m(input)
                self.assertNotEqual(u0, m.weight_u)
                self.assertNotEqual(v0, m.weight_v)

                # assert that backprop reaches weight_orig
                # can't use gradcheck because the function changes as we
                # activate through it in training mode
                if requires_grad:
                    torch.autograd.grad(out.sum(), m.weight_orig)

                # test backward works with multiple forwards
                # it uses training mode so we need to reset `u` and `v` vectors
                # to same value at beginning for finite difference test to pass
                saved_u = m.weight_u.clone()
                saved_v = m.weight_v.clone()

                def fn(input):
                    m.weight_u.data.copy_(saved_u)
                    m.weight_v.data.copy_(saved_v)
                    out0 = wrapped_m(input)
                    out1 = wrapped_m(input)
                    return out0 + out1

                torch.autograd.gradcheck(fn, (input.clone().requires_grad_(),))

                # test removing
                pre_remove_out = wrapped_m(input)
                m = torch.nn.utils.remove_spectral_norm(m)
                self.assertEqual(wrapped_m(input), pre_remove_out)

                m = torch.nn.utils.spectral_norm(m)
                for _ in range(3):
                    pre_remove_out = wrapped_m(input)
                m = torch.nn.utils.remove_spectral_norm(m)
                self.assertEqual(wrapped_m(input), pre_remove_out)

                # TEST EVAL BEHAVIOR

                m = torch.nn.utils.spectral_norm(m)
                wrapped_m(input)
                last_train_out = wrapped_m(input)
                last_train_u = m.weight_u.clone()
                last_train_v = m.weight_v.clone()
                wrapped_m.zero_grad()
                wrapped_m.eval()

                eval_out0 = wrapped_m(input)
                # assert eval gives same result as last training iteration
                self.assertEqual(eval_out0, last_train_out)
                # assert doing more iteartion in eval don't change things
                self.assertEqual(eval_out0, wrapped_m(input))
                self.assertEqual(last_train_u, m.weight_u)
                self.assertEqual(last_train_v, m.weight_v)

                # FIXME: the code below is flaky when executed with DataParallel
                # see https://github.com/pytorch/pytorch/issues/13818
                if apply_dp:
                    continue

                # test backward works with multiple forwards in mixed training
                # and eval modes
                # it uses training mode so we need to reset `u` and `v` vectors
                # to same value at beginning for finite difference test to pass
                saved_u = m.weight_u.clone()
                saved_v = m.weight_v.clone()

                def fn(input):
                    m.weight_u.data.copy_(saved_u)
                    m.weight_v.data.copy_(saved_v)
                    wrapped_m.train()
                    out0 = wrapped_m(input)
                    wrapped_m.eval()
                    out1 = wrapped_m(input)
                    wrapped_m.train()
                    out2 = wrapped_m(input)
                    wrapped_m.eval()
                    out3 = wrapped_m(input)
                    return out0 + out1 + out2 + out3

                torch.autograd.gradcheck(fn, (input.clone().requires_grad_(),))

                # assert that backprop reaches weight_orig in eval
                if requires_grad:
                    def fn(weight):
                        return wrapped_m(input)

                    torch.autograd.gradcheck(fn, (m.weight_orig,))

    def test_spectral_norm_load_state_dict(self):
        inp = torch.randn(2, 3)
        for activate_times in (0, 3):
            # Test backward compatibility
            # At version None -> 1: weight becomes not a buffer and v vector becomes a buffer
            m = nn.Linear(3, 5)
            snm = torch.nn.utils.spectral_norm(m)
            snm.train()
            for _ in range(activate_times):
                snm(inp)

            version_latest_ref_state_dict = deepcopy(snm.state_dict())
            self.assertEqual({'weight_orig', 'bias', 'weight_u', 'weight_v'}, set(version_latest_ref_state_dict.keys()))

            # test that non-strict loading works
            non_strict_state_dict = deepcopy(version_latest_ref_state_dict)
            non_strict_state_dict['nonsense'] = 'nonsense'
            with self.assertRaisesRegex(RuntimeError, r'Unexpected key\(s\) in state_dict: "nonsense"'):
                snm.load_state_dict(non_strict_state_dict, strict=True)
            snm.load_state_dict(non_strict_state_dict, strict=False)
            del non_strict_state_dict['weight_orig']
            snm.load_state_dict(non_strict_state_dict, strict=False)
            del non_strict_state_dict['weight_u']
            snm.load_state_dict(non_strict_state_dict, strict=False)
            del non_strict_state_dict['weight_v']
            snm.load_state_dict(non_strict_state_dict, strict=False)
            non_strict_state_dict['weight'] = snm.weight.detach().clone()  # set W as a buffer
            snm.load_state_dict(non_strict_state_dict, strict=False)
            del non_strict_state_dict._metadata['']['spectral_norm']       # remove metadata info
            snm.load_state_dict(non_strict_state_dict, strict=False)
            del non_strict_state_dict['weight']                            # remove W buffer
            snm.load_state_dict(non_strict_state_dict, strict=False)
            del non_strict_state_dict['bias']
            snm.load_state_dict(non_strict_state_dict, strict=False)

            # craft a version None state_dict
            version_none_state_dict = deepcopy(version_latest_ref_state_dict)
            self.assertIn('spectral_norm', version_none_state_dict._metadata[''])
            del version_none_state_dict._metadata['']['spectral_norm']       # remove metadata info
            del version_none_state_dict['weight_v']                          # remove v vector
            version_none_state_dict['weight'] = snm.weight.detach().clone()  # set W as a buffer

            # normal state_dict
            for version_latest_with_metadata in [True, False]:
                version_latest_state_dict = deepcopy(version_latest_ref_state_dict)

                if not version_latest_with_metadata:
                    # We want to still load a user-crafted state_dict, one without metadata
                    del version_latest_state_dict._metadata['']['spectral_norm']

                # test that re-wrapping does not matter
                m = torch.nn.utils.remove_spectral_norm(snm)
                snm = torch.nn.utils.spectral_norm(m)

                snm.load_state_dict(version_latest_ref_state_dict)
                with torch.no_grad():
                    snm.eval()
                    out0_eval = snm(inp)
                    snm.train()
                    out1_train = snm(inp)
                    out2_train = snm(inp)
                    snm.eval()
                    out3_eval = snm(inp)

                # test that re-wrapping does not matter
                m = torch.nn.utils.remove_spectral_norm(snm)
                snm = torch.nn.utils.spectral_norm(m)

                snm.load_state_dict(version_none_state_dict)
                if activate_times > 0:
                    # since in loading version None state dict, we assume that the
                    # values in the state dict have gone through at lease one
                    # forward, we only test for equivalence when activate_times > 0.
                    with torch.no_grad():
                        snm.eval()
                        self.assertEqual(out0_eval, snm(inp))
                        snm.train()
                        self.assertEqual(out1_train, snm(inp))
                        self.assertEqual(out2_train, snm(inp))
                        snm.eval()
                        self.assertEqual(out3_eval, snm(inp))

                # test that re-wrapping does not matter
                m = torch.nn.utils.remove_spectral_norm(snm)
                snm = torch.nn.utils.spectral_norm(m)

                # Test normal loading
                snm.load_state_dict(version_latest_state_dict)
                with torch.no_grad():
                    snm.eval()
                    self.assertEqual(out0_eval, snm(inp))
                    snm.train()
                    self.assertEqual(out1_train, snm(inp))
                    self.assertEqual(out2_train, snm(inp))
                    snm.eval()
                    self.assertEqual(out3_eval, snm(inp))

    def test_spectral_norm_dim(self):
        inp = torch.randn(2, 3, 10, 12)
        m = nn.ConvTranspose2d(3, 4, (5, 6))
        m = torch.nn.utils.spectral_norm(m)
        # this should not run into incompatible shapes
        x = m(inp)
        # check that u refers to the same dimension
        self.assertEqual(m.weight_u.shape, m.weight_orig[0, :, 0, 0].shape)

    def test_spectral_norm_forward(self):
        input = torch.randn(3, 5)
        m = nn.Linear(5, 7)
        m = torch.nn.utils.spectral_norm(m)
        # naive forward
        _weight, _bias, _u = m.weight_orig, m.bias, m.weight_u
        _weight_mat = _weight.view(_weight.size(0), -1)
        _v = torch.mv(_weight_mat.t(), _u)
        _v = F.normalize(_v, dim=0, eps=1e-12)
        _u = torch.mv(_weight_mat, _v)
        _u = F.normalize(_u, dim=0, eps=1e-12)
        _weight.data /= torch.dot(_u, torch.matmul(_weight_mat, _v))
        out_hat = torch.nn.functional.linear(input, _weight, _bias)
        expect_out = m(input)
        self.assertAlmostEqual(expect_out, out_hat)

    def test_spectral_norm_pickle(self):
        m = torch.nn.utils.spectral_norm(nn.Linear(5, 7))
        m = pickle.loads(pickle.dumps(m))
        self.assertIsInstance(m, nn.Linear)

    def test_threshold_int(self):
        x = torch.tensor([-3, -2, -1, 0, 1, 2, 3])
        expected = torch.tensor([99, 99, 99, 99, 1, 2, 3])
        self.assertEqual(F.threshold(x, 0, 99), expected)

    def test_embedding_sparse_basic(self):
        embedding = nn.Embedding(10, 20, sparse=True)
        input = torch.tensor([[0, 2, 4, 5], [4, 3, 0, 9]], dtype=torch.long)
        embedding(input).sum().backward()
        self.assertTrue(embedding.weight.grad.is_sparse)
        self.assertEqual(embedding.weight.grad.shape, embedding.weight.shape)

    def test_embedding_sparse_empty_tensor(self):
        embedding = nn.Embedding(0, 0, sparse=True)
        input = torch.tensor([], dtype=torch.int64)
        embedding(input).sum().backward()
        self.assertTrue(embedding.weight.grad.is_sparse)
        self.assertEqual(embedding.weight.grad.shape, embedding.weight.shape)

        embedding = nn.Embedding(10, 0, sparse=True)
        input = torch.LongTensor([[0, 2, 4, 5], [4, 3, 0, 9]])
        embedding(input).sum().backward()
        self.assertTrue(embedding.weight.grad.is_sparse)
        self.assertEqual(embedding.weight.grad.shape, embedding.weight.shape)

    def test_move_sparse_half_embedding(self):
        embedding = nn.Embedding(10, 3, sparse=True)
        self.assertEqual(embedding.weight.device.type, 'cpu')
        self.assertEqual(embedding.weight.dtype, torch.float64)
        embedding.to(torch.float16)
        self.assertEqual(embedding.weight.dtype, torch.float16)
        self.assertEqual(embedding.embedding_dim, 3)
        self.assertEqual(embedding.num_embeddings, 10)

        if torch.cuda.is_available():
            embedding.to('cuda')
            self.assertEqual(embedding.weight.device.type, 'cuda')
            embedding.to('cpu')
            self.assertEqual(embedding.weight.device.type, 'cpu')

    def test_embedding_max_norm(self):
        embedding = nn.Embedding(22, 5, max_norm=1.0)
        input = torch.tensor([2, 8, 8, 6], dtype=torch.long)
        output = embedding(input)
        self.assertEqual(output[1], output[2])
        self.assertTrue(output.data.norm(p=2, dim=1).le(1).all())

    def test_embedding_from_pretrained(self):
        a = torch.Tensor([[1, 2, 3], [4, 5, 6]])
        embedding = nn.Embedding.from_pretrained(a)
        self.assertEqual(a, embedding.weight.data)

        input = torch.LongTensor([0, 1])
        output = embedding(input)
        self.assertEqual(a, output)

    def test_embedding_from_pretrained_options(self):
        a = torch.Tensor([[1, 2, 3], [4, 5, 6]])
        opts = {
            "max_norm": 2.,
            "norm_type": .5,
            "scale_grad_by_freq": False,
            "sparse": True
        }
        embedding = nn.Embedding.from_pretrained(a, **opts)
        input = torch.LongTensor([0, 1])
        output = embedding(input)
        # test output and that weight matrix was renormalized
        self.assertEqual(a, output)
        self.assertTrue(a.ne(torch.arange(1, 7, dtype=a.dtype).view(2, 3)).all())
        self.assertTrue(output.data.norm(p=opts["norm_type"], dim=1).le(opts["max_norm"]).all())

    def test_embedding_functional(self):
        a = torch.tensor([
            [1, 3, 2],
            [0, 2, 1]
        ], dtype=torch.long)
        embeddings = torch.rand(4, 3, requires_grad=True)

        embed_old = torch.nn.Embedding(4, 3)
        embed_old.weight.data = embeddings.data
        res_old = embed_old(a)

        res_F = F.embedding(a, embeddings)
        self.assertEqual(res_old, res_F)

    @unittest.skipUnless('fbgemm' in torch.backends.quantized.supported_engines,
                         'Linear_FP16_weight requires FBGEMM. FBGEMM is only optimized for CPUs'
                         ' with instruction set support avx2 or newer.')
    def test_fb_fc_packed(self):
        X = np.random.rand(16, 16).astype(np.float32) - 0.5
        W = np.random.rand(16, 16).astype(np.float32) - 0.5
        b = np.random.rand(16).astype(np.float32) - 0.5

        def fc_op(X, W, b):
            return np.dot(X, W.T) + b

        x_tensor = torch.tensor(X)
        w_tensor = torch.tensor(W)
        b_tensor = torch.tensor(b)
        packed_w_tensor = torch.fbgemm_pack_gemm_matrix_fp16(w_tensor)
        actual_output = torch.fbgemm_linear_fp16_weight(x_tensor, packed_w_tensor, b_tensor)
        expected_output = fc_op(X, W, b)
        torch.testing.assert_allclose(expected_output, actual_output.cpu(), atol=1e-3, rtol=1e-3)

    def test_embeddingbag_from_pretrained(self):
        a = torch.Tensor([[1, 2, 3], [4, 5, 6]])
        embeddingbag = nn.EmbeddingBag.from_pretrained(a)
        self.assertEqual(a, embeddingbag.weight.data)

        input = torch.LongTensor([[0, 1]])
        output = embeddingbag(input)
        self.assertEqual(a.mean(0, keepdim=True), output)

    def test_embeddingbag_from_pretrained_options(self):
        a = torch.Tensor([[1, 2, 3], [4, 5, 6]])
        opts = {
            "max_norm": 2.,
            "norm_type": .5,
            "scale_grad_by_freq": False,
            "mode": "max",
            "sparse": False
        }
        embeddingbag = nn.EmbeddingBag.from_pretrained(a, **opts)

        input = torch.LongTensor([[0, 1]])
        output = embeddingbag(input)
        self.assertEqual(a.max(0, keepdim=True)[0], output)
        self.assertTrue(a.ne(torch.arange(1, 7, dtype=a.dtype).view(2, 3)).all())
        self.assertTrue(a.norm(p=opts["norm_type"], dim=1).le(opts["max_norm"]).all())

    def test_fractional_max_pool2d(self):
        x = torch.randn(1, 2, 7, 7, requires_grad=True)
        samples = x.new(1, 2, 2).uniform_()

        def func(x):
            return F.fractional_max_pool2d(
                x, (2, 2), output_size=(3, 3), _random_samples=samples)

        self.assertEqual(func(x).shape, (1, 2, 3, 3))
        gradcheck(func, [x])
        gradgradcheck(func, [x])

        x = torch.randn(2, 7, 7, requires_grad=True)
        samples = x.new(2, 2).uniform_()
        self.assertEqual(func(x).shape, (2, 3, 3))
        gradcheck(func, [x])
        gradgradcheck(func, [x])

    def test_AlphaDropout(self):
        # generate random tensor with zero mean and unit std
        input = torch.randn(5000)
        self._test_alpha_dropout(nn.AlphaDropout, input)

    def test_FeatureAlphaDropout(self):
        b = random.randint(1, 5)
        w = random.randint(1, 5)
        h = random.randint(1, 5)
        d = random.randint(1, 2)
        num_features = 1000
        input = torch.randn(num_features, b, d, w, h)
        self._test_alpha_dropout(nn.FeatureAlphaDropout, input)

    def test_pad(self):
        inputs = torch.randn(1, 3, 4, 4, requires_grad=True)
        _assertGradAndGradgradChecks(self, lambda x: F.pad(x, (1, 1, 1, 1)), (inputs,))
        _assertGradAndGradgradChecks(self, lambda x: F.pad(x, (-1, 1, -2, 1)), (inputs,))
        _assertGradAndGradgradChecks(self, lambda x: F.pad(x, (-1, 1, -2, 1), value=2), (inputs,))
        self.assertTrue(gradcheck(lambda x: F.pad(x, (-1, 1, -2, 1), mode='replicate'), (inputs,)))
        self.assertTrue(gradcheck(lambda x: F.pad(x, (-1, 1, -2, 1), mode='reflect'), (inputs,)))

        inputs = torch.randn(1, 2, 3, 4, 4, requires_grad=True)
        self.assertTrue(gradcheck(lambda x: F.pad(x, (1, 1, 1, 1, 1, 1), mode='replicate'), (inputs,)))

        # assert that relfection padding errors when pad >= input size
        expected_err_msg = r"Padding size should be less than the corresponding input dimension"
        self.assertRaisesRegex(RuntimeError, expected_err_msg,
                               lambda: F.pad(torch.randn(1, 1, 2, 3), (1, 1, 3, 0), mode='reflect'))
        self.assertRaisesRegex(RuntimeError, expected_err_msg,
                               lambda: F.pad(torch.randn(1, 1, 2), (2, 1), mode='reflect'))

    def test_pad_scalar_error(self):
        inputs = torch.tensor(0., requires_grad=True)
        self.assertRaises(AssertionError, lambda: F.pad(inputs, (1, 1)))
        self.assertRaises(AssertionError, lambda: F.pad(inputs, (1,)))

    @unittest.skipIf(not TEST_NUMPY, "numpy not found")
    def test_multihead_attention(self):
        def _scaled_dot_attn_ref(Q, K, V, dims, unseen_mask=None, key_padding_mask=None):
            """ Numpy-based reference implementation of scaled dot attention
            for testing"""

            QKT = _batchmatmul(
                Q,
                np.transpose(K, axes=[0, 1, 3, 2])
                / np.sqrt(dims[3], dtype=np.float32),  # divide by sqrt(d_head)
            )
            b1, b2, s1, s2 = QKT.shape
            if unseen_mask is not None or key_padding_mask is not None:
                # assert s1 == s2
                for i in range(b1):
                    for j in range(b2):
                        for m in range(s1):
                            for n in range(s2):
                                if unseen_mask is not None and unseen_mask[m][n] == 0:
                                    QKT[i, j, m, n] = -np.inf
                                if key_padding_mask is not None and key_padding_mask[i][n]:
                                    QKT[i, j, m, n] = -np.inf

            reference = _softmax(QKT)
            ref_attn_weight = reference
            ref_attn_weight = np.sum(ref_attn_weight, axis=1) / b2
            reference = _batchmatmul(reference, V)
            return reference, ref_attn_weight

        def _batchmatmul(a, b):  # batchmatmul over 4 dim matrix
            """ Numpy-based batch matrix multiply over 4 dim matrix"""
            assert a.shape[0] == b.shape[0]
            assert a.shape[1] == b.shape[1]
            retval = np.zeros(
                (a.shape[0], a.shape[1], a.shape[2], b.shape[3]), dtype=np.float32
            )
            for i in range(a.shape[0]):
                for j in range(a.shape[1]):
                    retval[i, j, :, :] = np.matmul(a[i, j, :, :], b[i, j, :, :])
            return retval

        def _softmax(x):  # softmax over 4 dim matrix
            """ Numpy-based reference softmax over 4 dim matrix"""
            np.seterr(invalid='ignore')
            output = np.zeros(x.shape, dtype=np.float64)
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    for k in range(x.shape[2]):
                        x_curr = x[i, j, k, :]
                        e_x = np.exp(x_curr - np.amax(x_curr))
                        output[i, j, k, :] = e_x / np.sum(e_x)
            return output

        def _split_heads_ref(X, dims, nheads, d_head):
            X_split = np.reshape(X, dims[:2] + [nheads, d_head])
            X_split_transposed = np.transpose(X_split, [0, 2, 1, 3])
            reference = np.reshape(X_split_transposed, [dims[0], nheads, dims[1], d_head])
            return reference

        def _combine_heads_ref(X, dims, nheads, d_head):
            X_transposed = np.transpose(X, [0, 2, 1, 3])
            reference = np.reshape(X_transposed, dims[:2] + [nheads * d_head])
            return reference

        def _fc(X, X_weight, X_bias):
            X_fc_b = X_bias.detach().numpy()
            X_fc_w = X_weight.detach().numpy()
            return np.matmul(X, np.transpose(X_fc_w)) + X_fc_b

        def _create_src_lengths_mask(batch_size, src_lengths):
            """
            Generate boolean mask to prevent attention beyond the end of source
            Inputs:
              batch_size : int
              src_lengths : [batch_size] of sentence lengths
            Outputs:
              [batch_size, max_src_len]
            """
            max_srclen = src_lengths.max()
            src_indices = torch.arange(0, max_srclen).unsqueeze(0).type_as(src_lengths)
            src_indices = src_indices.expand(batch_size, max_srclen)
            src_lengths = src_lengths.unsqueeze(dim=1).expand(batch_size, max_srclen)
            # returns [batch_size, max_seq_len]
            return (src_indices < src_lengths).int().detach()

        def _multihead_attn_test_helper(add_key_padding_mask=False, add_bias_kv=False, add_zero_attn=False,
                                        saved_kv=False, same_embed_dim=False):
            for _ in range(100):
                batch_sz, seq_len = [random.randint(2, 10) for r in range(2)]
                d_head = random.randint(3, 10)
                nheads = random.randint(3, 10)
                d_model = d_head * nheads
                if same_embed_dim:
                    kv_dim = d_model
                else:
                    kv_dim = random.randint(5, 20)
                dims = [batch_sz, seq_len, kv_dim]

                saved_k = None
                saved_k_tensor = None
                saved_v = None
                saved_v_tensor = None
                if saved_kv:
                    saved_k = np.random.rand(batch_sz * nheads, seq_len, d_head)
                    saved_k_tensor = torch.from_numpy(saved_k).to(torch.get_default_dtype())
                    saved_v = np.random.rand(batch_sz * nheads, seq_len, d_head)
                    saved_v_tensor = torch.from_numpy(saved_v).to(torch.get_default_dtype())

                key_padding_mask = None
                key_padding_mask_tensor = None
                if add_key_padding_mask:
                    seq_mask = np.random.randint(0, 2, (1, seq_len))
                    key_padding_mask = (np.repeat(seq_mask, batch_sz, axis=0) == 1)
                    key_padding_mask_tensor = torch.from_numpy(key_padding_mask)

                decoder_state = np.random.rand(batch_sz, d_model)
                K = np.random.rand(*dims)
                V = K
                Q = np.expand_dims(decoder_state, 1)
                attn_mask = np.random.randint(0 , 2, size=(1, seq_len))
                attn_mask_tensor = torch.from_numpy(attn_mask).float()
                attn_mask_tensor.masked_fill_(attn_mask_tensor == 0, float('-inf'))
                attn_mask_tensor.masked_fill_(attn_mask_tensor > 0, float('0.0'))
                attn_mask_tensor = attn_mask_tensor.double()

                decoder_state_tensor = torch.from_numpy(decoder_state).to(torch.get_default_dtype())
                source_hid_tensor = torch.from_numpy(K).to(torch.get_default_dtype()).transpose(0, 1)

                multihead_attn_module = MultiheadAttention(d_model, nheads,
                                                           add_bias_kv=add_bias_kv,
                                                           add_zero_attn=add_zero_attn,
                                                           kdim=kv_dim, vdim=kv_dim)

                if add_bias_kv:
                    bias_k = multihead_attn_module.bias_k.detach().numpy()
                    bias_v = multihead_attn_module.bias_v.detach().numpy()
                else:
                    bias_k = None
                    bias_v = None

                _Q = decoder_state_tensor.unsqueeze(1).transpose(0, 1)
                _V = source_hid_tensor
                _K = source_hid_tensor

                if multihead_attn_module._qkv_same_embed_dim:
                    result, result_weight = torch.nn.functional.multi_head_attention_forward(
                        _Q, _K, _V,
                        d_model, nheads,
                        multihead_attn_module.in_proj_weight, multihead_attn_module.in_proj_bias,
                        multihead_attn_module.bias_k, multihead_attn_module.bias_v,
                        multihead_attn_module.add_zero_attn, multihead_attn_module.dropout,
                        multihead_attn_module.out_proj.weight, multihead_attn_module.out_proj.bias,
                        multihead_attn_module.training, key_padding_mask_tensor, True, attn_mask_tensor,
                        static_k=saved_k_tensor, static_v=saved_v_tensor)
                else:
                    result, result_weight = torch.nn.functional.multi_head_attention_forward(
                        _Q, _K, _V,
                        d_model, nheads,
                        None, multihead_attn_module.in_proj_bias,
                        multihead_attn_module.bias_k, multihead_attn_module.bias_v,
                        multihead_attn_module.add_zero_attn, multihead_attn_module.dropout,
                        multihead_attn_module.out_proj.weight, multihead_attn_module.out_proj.bias,
                        multihead_attn_module.training, key_padding_mask_tensor, True, attn_mask_tensor,
                        True, multihead_attn_module.q_proj_weight,
                        multihead_attn_module.k_proj_weight, multihead_attn_module.v_proj_weight,
                        static_k=saved_k_tensor, static_v=saved_v_tensor)

                result = result.squeeze(0).detach().numpy()

                if multihead_attn_module._qkv_same_embed_dim:
                    q_proj_weight = multihead_attn_module.in_proj_weight[:d_model]
                    k_proj_weight = multihead_attn_module.in_proj_weight[d_model:(d_model * 2)]
                    v_proj_weight = multihead_attn_module.in_proj_weight[(d_model * 2):]
                else:
                    q_proj_weight = multihead_attn_module.q_proj_weight
                    k_proj_weight = multihead_attn_module.k_proj_weight
                    v_proj_weight = multihead_attn_module.v_proj_weight

                Q_fc = _fc(Q, q_proj_weight, multihead_attn_module.in_proj_bias[:d_model])
                K_fc = _fc(K, k_proj_weight, multihead_attn_module.in_proj_bias[d_model:(d_model * 2)])
                V_fc = _fc(V, v_proj_weight, multihead_attn_module.in_proj_bias[(d_model * 2):])

                if add_bias_kv:
                    K_fc = np.concatenate((K_fc, np.repeat(bias_k, K_fc.shape[0], axis=0)), axis=1)
                    V_fc = np.concatenate((V_fc, np.repeat(bias_v, V_fc.shape[0], axis=0)), axis=1)
                    if attn_mask is not None:
                        attn_mask = np.concatenate((attn_mask, np.ones([1, 1])), axis=1)
                    if key_padding_mask is not None:
                        key_padding_mask = np.concatenate((key_padding_mask, np.full((batch_sz, 1), False, dtype=bool)), axis=1)
                    dims[1] += 1
                Q_split = _split_heads_ref(
                    Q_fc, [batch_sz, 1, d_model], nheads, d_head
                )

                if saved_k is not None:
                    K_split = np.reshape(saved_k, [dims[0], nheads, dims[1], d_head])
                else:
                    K_split = _split_heads_ref(K_fc, dims, nheads, d_head)

                if saved_v is not None:
                    V_split = np.reshape(saved_v, [dims[0], nheads, dims[1], d_head])
                else:
                    V_split = _split_heads_ref(V_fc, dims, nheads, d_head)

                if add_zero_attn:
                    dims[1] += 1
                    K_split = np.concatenate((K_split, np.zeros([K_split.shape[0], K_split.shape[1], 1, K_split.shape[3]])), axis=2)
                    V_split = np.concatenate((V_split, np.zeros([V_split.shape[0], V_split.shape[1], 1, V_split.shape[3]])), axis=2)

                    if attn_mask is not None:
                        attn_mask = np.concatenate((attn_mask, np.ones([1, 1])), axis=1)

                    if key_padding_mask is not None:
                        key_padding_mask = np.concatenate((key_padding_mask, np.full((batch_sz, 1), False, dtype=bool)), axis=1)
                attn_heads, ref_attn_weight = _scaled_dot_attn_ref(
                    Q=Q_split,
                    K=K_split,
                    V=V_split,
                    dims=Q_split.shape,
                    unseen_mask=attn_mask,
                    key_padding_mask=key_padding_mask
                )
                combined_attn_heads = _combine_heads_ref(
                    X=attn_heads, dims=[batch_sz, 1], nheads=nheads, d_head=d_head
                )

                reference = _fc(combined_attn_heads, multihead_attn_module.out_proj.weight, multihead_attn_module.out_proj.bias)
                reference = np.squeeze(reference, axis=1)

                # result = reference
                self.assertEqual(tuple(result.shape), (batch_sz, d_model))
                np.testing.assert_allclose(result, reference, atol=1e-5)

                # result_weight = ref_attn_weight
                result_weight = result_weight.detach().numpy()
                self.assertEqual(tuple(result_weight.shape), tuple(ref_attn_weight.shape))
                np.testing.assert_allclose(result_weight, ref_attn_weight, atol=1e-5)

        def test_multihead_attn_add_bias_kv():
            _multihead_attn_test_helper(add_bias_kv=True)

        def test_multihead_attn_add_zero_attn():
            _multihead_attn_test_helper(add_zero_attn=True)

        def test_multihead_attn_no_masking():
            _multihead_attn_test_helper()

        def test_multihead_attn_key_padding_mask():
            _multihead_attn_test_helper(add_key_padding_mask=True)

        def test_multihead_attn_saved_kv():
            _multihead_attn_test_helper(saved_kv=True)

        def test_multihead_attn_add_bias_kv_zero_attn():
            _multihead_attn_test_helper(add_key_padding_mask=True, add_bias_kv=True,
                                        add_zero_attn=True)

        def test_multihead_attn_all_arguments1():
            _multihead_attn_test_helper(add_key_padding_mask=True, add_zero_attn=True, saved_kv=True)

        def test_multihead_attn_all_arguments2():
            _multihead_attn_test_helper(add_key_padding_mask=True, add_bias_kv=True,
                                        add_zero_attn=True, saved_kv=True)

        def test_multihead_attn_all_arguments3():
            _multihead_attn_test_helper(add_key_padding_mask=True, add_zero_attn=True,
                                        saved_kv=True, same_embed_dim=True)

        test_multihead_attn_add_zero_attn()  # Test MultiheadAttention with add_zero_attn
        test_multihead_attn_add_bias_kv()  # Test MultiheadAttention with add_bias_kv
        test_multihead_attn_no_masking()   # Test MultiheadAttention without masking
        test_multihead_attn_key_padding_mask()  # Test MultiheadAttention with src lengths
        test_multihead_attn_saved_kv()  # Test MultiheadAttention with static kv.
        test_multihead_attn_add_bias_kv_zero_attn()  # Test MultiheadAttention with bias_kv and zero_attn.
        test_multihead_attn_all_arguments1()  # Test MultiheadAttention with all the argument.
        with self.assertRaisesRegex(AssertionError, "bias cannot be added to static key."):
            test_multihead_attn_all_arguments2()  # Test MultiheadAttention with all the argument.
        test_multihead_attn_all_arguments3()  # Test MultiheadAttention with all the argument.

    def test_normalize(self):
        inputs = torch.randn(1, 3, 4, 4, requires_grad=True)
        self.assertTrue(gradcheck(lambda x: F.normalize(x, p=1, dim=-1), (inputs,)))
        self.assertTrue(gradcheck(lambda x: F.normalize(x, p=2, dim=-2), (inputs,)))

        inputs = torch.randn((), requires_grad=True)
        self.assertTrue(gradcheck(lambda x: F.normalize(x, p=1, dim=-1), (inputs,)))

    def test_adaptive_pooling_input_size(self):
        for numel in (2, 3):
            for pool_type in ('Max', 'Avg'):
                cls_name = 'Adaptive{}Pool{}d'.format(pool_type, numel)
                module_cls = getattr(nn, cls_name)
                output_size = (2,) * numel
                module = module_cls(output_size)

                input = torch.randn(output_size)
                self.assertRaises(ValueError, lambda: module(input))

    def test_adaptive_pooling_size_none(self):
        for numel in (2, 3):
            for pool_type in ('Max', 'Avg'):
                cls_name = 'Adaptive{}Pool{}d'.format(pool_type, numel)
                module_cls = getattr(nn, cls_name)
                output_size = (2,) * (numel - 1) + (None,)
                module = module_cls(output_size)

                input = torch.randn((4,) * (numel + 1))
                output = module(input)
                self.assertEqual(output.size(), (4,) + (2,) * (numel - 1) + (4,))

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_adaptive_pooling_avg_nhwc(self):
        input = torch.randint(1, 10, (4, 8, 8, 8), dtype=torch.float32, device="cuda")
        input = input.contiguous(memory_format=torch.channels_last).requires_grad_()
        grad = torch.randint(1, 10, (4, 8, 7, 7), dtype=torch.float32, device="cuda")
        pool = torch.nn.AdaptiveAvgPool2d((7, 7)).cuda()

        ref_input = input.detach().clone().contiguous().requires_grad_(True)
        ref_grad = grad.detach().clone().contiguous()
        ref_pool = torch.nn.AdaptiveAvgPool2d((7, 7)).cuda()

        out = pool(input)
        out.backward(grad)
        ref_out = ref_pool(ref_input)
        ref_out.backward(ref_grad)

        self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))
        self.assertTrue(ref_out.is_contiguous())
        self.assertEqual(out, ref_out)
        self.assertEqual(input.grad, ref_input.grad)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_adaptive_pooling_avg_nhwc_non_contiguous(self):
        input = torch.randint(1, 10, (4, 8, 8, 8), dtype=torch.float32, device="cuda")
        input = input.contiguous(memory_format=torch.channels_last)
        input = input[:, ::2, :, :].requires_grad_()
        grad = torch.randint(1, 10, (4, 8, 7, 7), dtype=torch.float32, device="cuda")
        grad = grad[:, ::2, :, :]
        pool = torch.nn.AdaptiveAvgPool2d((7, 7)).cuda()

        ref_input = input.detach().clone().contiguous().requires_grad_(True)
        ref_grad = grad.detach().clone().contiguous()
        ref_pool = torch.nn.AdaptiveAvgPool2d((7, 7)).cuda()

        out = pool(input)
        out.backward(grad)
        ref_out = ref_pool(ref_input)
        ref_out.backward(ref_grad)

        self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))
        self.assertTrue(ref_out.is_contiguous())
        self.assertEqual(out, ref_out)
        self.assertEqual(input.grad, ref_input.grad)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    @unittest.skipIf(not TEST_LARGE_TENSOR, "not enough memory")
    def test_adaptive_pooling_avg_nhwc_launch_config_backward(self):
        input = torch.randint(1, 10, (1, 32, 2 ** 17 + 1, 32), dtype=torch.float32, device="cuda")
        input = input.contiguous(memory_format=torch.channels_last).requires_grad_()
        grad = torch.randint(1, 10, (1, 32, 10, 32), dtype=torch.float32, device="cuda")

        pool = torch.nn.AdaptiveAvgPool2d((10, 32)).cuda()

        ref_input = input.detach().clone().contiguous().requires_grad_(True)
        ref_grad = grad.detach().clone().contiguous()
        ref_pool = torch.nn.AdaptiveAvgPool2d((10, 32)).cuda()

        out = pool(input)
        out.backward(grad)
        ref_out = ref_pool(ref_input)
        ref_out.backward(ref_grad)

        self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))
        self.assertTrue(ref_out.is_contiguous())
        self.assertEqual(out, ref_out)
        self.assertEqual(input.grad, ref_input.grad)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    @unittest.skipIf(not TEST_LARGE_TENSOR, "not enough memory")
    def test_adaptive_pooling_avg_nhwc_launch_config_forward(self):
        input = torch.randint(1, 10, (1, 32, 16, 16), dtype=torch.float32, device="cuda")
        input = input.contiguous(memory_format=torch.channels_last).requires_grad_()
        pool = torch.nn.AdaptiveAvgPool2d((2 ** 17 + 1, 32)).cuda()

        ref_input = input.detach().clone().contiguous().requires_grad_(True)
        ref_pool = torch.nn.AdaptiveAvgPool2d((2 ** 17 + 1, 32)).cuda()

        out = pool(input)
        ref_out = ref_pool(ref_input)

        self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))
        self.assertTrue(ref_out.is_contiguous())
        self.assertEqual(out, ref_out)

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_broadcast_double_backwards_gpu(self):
        tensors = (torch.randn(4, 4, device='cuda', requires_grad=True),
                   torch.randn(4, 4, device='cuda', requires_grad=True),
                   torch.randn(4, 4, device='cuda', requires_grad=True))
        _assertGradAndGradgradChecks(self, lambda *i: Broadcast.apply((0, 1), *i), tensors)

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_broadcast_not_requiring_grad(self):
        variables = [
            torch.randn(1, 2, device='cuda', requires_grad=True),
            torch.randn(1, 2, device='cuda', requires_grad=False),
            torch.randn(1, 2, device='cuda', requires_grad=False),
            torch.randn(1, 2, device='cuda', requires_grad=True),
            torch.randn(1, 2, device='cuda', requires_grad=True),
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
        self.assertEqual(len(state_dict), 10)
        self.assertEqual(len(state_dict._metadata), 6)
        self.assertIn('', state_dict._metadata)
        self.assertIn('linear1', state_dict._metadata)
        self.assertIn('linear1.weight', state_dict)
        self.assertIn('linear1.bias', state_dict)
        self.assertIn('linear2', state_dict._metadata)
        self.assertIn('linear2.weight', state_dict)
        self.assertIn('linear2.bias', state_dict)
        self.assertIn('block', state_dict._metadata)
        self.assertIn('block.conv', state_dict._metadata)
        self.assertIn('block.conv.weight', state_dict)
        self.assertIn('block.conv.weight', state_dict)
        self.assertNotIn('block.conv.bias', state_dict)
        self.assertIn('bn', state_dict._metadata)
        self.assertIn('bn.weight', state_dict)
        self.assertIn('bn.bias', state_dict)
        self.assertIn('bn.running_var', state_dict)
        self.assertIn('bn.running_mean', state_dict)
        self.assertIn('bn.num_batches_tracked', state_dict)
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
        self.assertEqual(len(state_dict._metadata), 1)
        self.assertIn('', state_dict._metadata)
        self.assertTrue(state_dict._metadata['']['version'] >= 0)
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
        incompatible_keys = net.load_state_dict(state_dict)
        self.assertEqual(len(incompatible_keys.missing_keys), 0)
        self.assertEqual(len(incompatible_keys.unexpected_keys), 0)
        self.assertNotIn('Incompatible', str(incompatible_keys))
        self.assertNotIn('Incompatible', repr(incompatible_keys))
        self.assertEqual(net.linear1.weight.data, state_dict['linear1.weight'])
        self.assertEqual(net.block.conv1.bias.data, state_dict['block.conv1.bias'])
        self.assertEqual(net.bn.running_mean, state_dict['bn.running_mean'])

        state_dict = net.state_dict()
        state_dict.update({'extra': torch.ones(5)})
        self.assertRaises(RuntimeError, lambda: net.load_state_dict(state_dict))
        incompatible_keys = net.load_state_dict(state_dict, strict=False)
        self.assertEqual(len(incompatible_keys.missing_keys), 0)
        self.assertEqual(len(incompatible_keys.unexpected_keys), 1)
        self.assertIn('extra', incompatible_keys.unexpected_keys)
        self.assertIn('Incompatible', str(incompatible_keys))
        self.assertIn('Incompatible', repr(incompatible_keys))

        state_dict = net.state_dict()
        state_dict.update({'extra.param': torch.ones(5)})
        self.assertRaises(RuntimeError, lambda: net.load_state_dict(state_dict))
        incompatible_keys = net.load_state_dict(state_dict, strict=False)
        self.assertEqual(len(incompatible_keys.missing_keys), 0)
        self.assertEqual(len(incompatible_keys.unexpected_keys), 1)
        self.assertIn('extra.param', incompatible_keys.unexpected_keys)

        state_dict = net.state_dict()
        del state_dict['linear1.weight']
        self.assertRaises(RuntimeError, lambda: net.load_state_dict(state_dict))
        incompatible_keys = net.load_state_dict(state_dict, strict=False)
        self.assertEqual(len(incompatible_keys.missing_keys), 1)
        self.assertEqual(len(incompatible_keys.unexpected_keys), 0)
        self.assertIn('linear1.weight', incompatible_keys.missing_keys)
        state_dict.update({'extra.param': torch.ones(5)})
        self.assertRaises(RuntimeError, lambda: net.load_state_dict(state_dict))
        incompatible_keys = net.load_state_dict(state_dict, strict=False)
        self.assertEqual(len(incompatible_keys.missing_keys), 1)
        self.assertEqual(len(incompatible_keys.unexpected_keys), 1)
        self.assertIn('linear1.weight', incompatible_keys.missing_keys)
        self.assertIn('extra.param', incompatible_keys.unexpected_keys)

        state_dict = net.state_dict()
        state_dict.update({'bn.running_mean': torch.rand(14, 4)})  # wrong size
        self.assertRaises(RuntimeError, lambda: net.load_state_dict(state_dict))
        self.assertRaises(RuntimeError, lambda: net.load_state_dict(state_dict, strict=False))

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

    def test_load_state_dict_BC(self):
        # BatchNormNd
        # Added num_batches_tracked buffer at version 2. For state dict with
        # earlier versions or no versions, it should provide default value of 0.
        bn = nn.BatchNorm2d(3)
        state_dict = bn.state_dict()
        del state_dict['num_batches_tracked']
        state_dict._metadata['']['version'] = 1  # version 1
        bn.load_state_dict(state_dict)
        self.assertEqual(bn.num_batches_tracked.dtype, torch.long)
        self.assertEqual(bn.num_batches_tracked.item(), 0)
        del state_dict._metadata['']['version']  # no version
        bn.load_state_dict(state_dict)
        self.assertEqual(bn.num_batches_tracked.dtype, torch.long)
        self.assertEqual(bn.num_batches_tracked.item(), 0)

    @unittest.skipIf(not PY3, 'Python 2.7 generates cyclic trash')
    def test_load_state_dict_ref_cycle(self):
        # load_state_dict shouldn't cause a reference cycle involving Tensors
        import gc

        m = torch.nn.LSTM(16, 16, bidirectional=True)

        gc.collect()
        m.load_state_dict(deepcopy(m).state_dict())
        refcycles = gc.collect()

        self.assertEqual(refcycles, 0)

    def test_load_state_dict_custom(self):

        class CustomState(nn.Module):
            def __init__(self):
                super(CustomState, self).__init__()
                self.param = torch.nn.Parameter(torch.ones(1))
                self.sub = torch.nn.Linear(5, 5)

            def _save_to_state_dict(self, destination, prefix, keep_vars):
                destination[prefix + "serialized"] = self.param.data + 1

            def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs):
                # skip some of the error handling
                self.param.data.copy_(state_dict[prefix + "serialized"] - 1)

        # use sequential to verify nesting
        m = nn.Sequential(CustomState())
        m[0].param[0] = 10
        m[0].sub.weight[0, 0] = 555
        state_dict = m.state_dict()
        self.assertEqual(state_dict["0.serialized"].item(), 11)
        self.assertIn("0.sub.weight", state_dict)
        self.assertNotIn("0.param", state_dict)
        del m
        mm = nn.Sequential(CustomState())
        self.assertEqual(mm[0].param[0].item(), 1)
        mm.load_state_dict(state_dict)
        self.assertEqual(mm[0].param[0].item(), 10)
        self.assertEqual(mm[0].sub.weight[0, 0].item(), 555)

    def test_parameter_assignment(self):
        l = nn.Linear(5, 5)

        def num_params():
            return len(list(l.parameters()))

        self.assertEqual(num_params(), 2)

        new_param = Parameter(torch.randn(5, 5))
        l.param_name = new_param
        self.assertEqual(num_params(), 3)
        self.assertObjectIn(new_param, l.parameters())

        var = torch.randn(5, 5)
        l.var_name = var
        self.assertEqual(num_params(), 3)
        self.assertNotIn(id(var), map(id, l.parameters()))

        # Make sure Variables are not saved as parameters
        l.variable_attr = torch.empty(5, 5)
        self.assertEqual(num_params(), 3)
        l.param_attr = Parameter(torch.empty(5, 5))
        self.assertEqual(num_params(), 4)

        # It shouldn't be possible to replace a parameter with a Variable
        def assign_var():
            l.param_attr = torch.empty(5, 5)

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
        self.assertEqual(l.state_dict()['buf'], buf)

    def test_Conv2d_inconsistent_types(self):
        inputs = torch.randn(4, 1, 7, 7, dtype=torch.float)
        weights = torch.randn(1, 1, 3, 3, dtype=torch.double)
        # inconsistent types should raise an exception
        self.assertRaises(RuntimeError, lambda: nn.functional.conv2d(inputs, weights))
        # but it should work with the same type
        nn.functional.conv2d(inputs.float(), weights.float())

    @unittest.skipIf(not TEST_CUDA, 'CUDA not available')
    def test_Conv2d_inconsistent_types_on_GPU_without_cudnn(self):
        inputs = torch.randn(4, 1, 7, 7, dtype=torch.float, device="cuda")
        weights = torch.randn(1, 1, 3, 3, dtype=torch.double, device="cuda")
        bias = torch.randn(1, dtype=torch.double, device="cuda")

        with torch.backends.cudnn.flags(enabled=False):
            # inconsistent types should raise an exception
            self.assertRaises(RuntimeError, lambda: nn.functional.conv2d(inputs, weights))
            self.assertRaises(RuntimeError, lambda: nn.functional.conv2d(inputs, weights.float(), bias))

            # but it should work with the same type
            nn.functional.conv2d(inputs.float(), weights.float(), bias.float())

    @unittest.skipIf(not TEST_CUDA, 'CUDA not available')
    @unittest.skipIf(not TEST_CUDNN, 'CUDNN not available')
    def test_cudnn_non_contiguous(self):
        x = torch.randn(192, 16, 50).cuda()
        x = x.permute(0, 2, 1).contiguous().permute(0, 2, 1)
        m = torch.nn.Conv1d(
            in_channels=16,
            out_channels=32,
            kernel_size=2,
            bias=True).cuda()
        result = m(x)

    @unittest.skipIf(not TEST_CUDA, 'CUDA not available')
    @unittest.skipIf(not TEST_CUDNN, 'CUDNN not available')
    def test_Conv2d_inconsistent_types_on_GPU_with_cudnn(self):
        inputs = torch.randn(4, 1, 7, 7, dtype=torch.float, device="cuda")
        weights = torch.randn(1, 1, 3, 3, dtype=torch.double, device="cuda")
        bias = torch.randn(1, dtype=torch.double, device="cuda")

        with torch.backends.cudnn.flags(enabled=True):
            # inconsistent types should raise an exception
            self.assertRaises(RuntimeError, lambda: nn.functional.conv2d(inputs, weights))
            self.assertRaises(RuntimeError, lambda: nn.functional.conv2d(inputs, weights.float(), bias))

            # but it should work with the same type
            nn.functional.conv2d(inputs.float(), weights.float(), bias.float())

    @unittest.skipIf(not TEST_CUDA, 'CUDA not available')
    @unittest.skipIf(not TEST_CUDNN, 'CUDNN not available')
    @repeat_test_for_types(ALL_TENSORTYPES2)
    def test_Conv2d_deterministic_cudnn(self, dtype=torch.float):
        inputs = torch.randn(2, 3, 5, 5, device="cuda", dtype=dtype, requires_grad=True)
        with cudnn.flags(enabled=True, benchmark=True, deterministic=True):
            conv1 = torch.nn.Conv2d(3, 3, 3).to("cuda", dtype)
            conv2 = torch.nn.Conv2d(3, 3, 3).to("cuda", dtype)
            conv2.bias.data.copy_(conv1.bias.data)
            conv2.weight.data.copy_(conv1.weight.data)
            out1 = conv1(inputs)
            out2 = conv2(inputs)
            self.assertEqual(out1, out2, prec=0.0)
            y = torch.randn(out1.size(), device="cuda", dtype=dtype)
            out1.backward(y)
            out2.backward(y)
            self.assertEqual(conv1.bias.grad.data, conv2.bias.grad.data, prec=0.0)
            self.assertEqual(conv1.weight.grad.data, conv2.weight.grad.data, prec=0.0)

    def test_Conv2d_missing_argument(self):
        c = nn.Conv2d(3, 3, 3)
        self.assertRaises(TypeError, lambda: c(None))

    def test_Conv2d_backward_twice(self):
        input = torch.randn(2, 3, 5, 5)
        c = nn.Conv2d(3, 3, 3)
        o1 = c(input)
        o1.sum().backward()
        self.assertRaisesRegex(RuntimeError, 'Specify retain_graph=True',
                               lambda: o1.sum().backward())

    @unittest.skipIf(not TEST_CUDA, 'CUDA not available')
    @repeat_test_for_types(ALL_TENSORTYPES2)
    def test_Conv2d_large_workspace(self, dtype=torch.float):
        # These sizes require huge cuDNN workspaces. Make sure we choose a
        # reasonable algorithm that does not run out of memory
        sizes = [
            (1, 256, 109, 175),
            (1, 256, 80, 128),
            (1, 256, 120, 192),
        ]

        def run_test(benchmark):
            with torch.backends.cudnn.flags(benchmark=benchmark):
                conv = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1).to("cuda", dtype)
                for size in sizes:
                    x = torch.randn(size, device="cuda", dtype=dtype)
                    out = conv(x.detach().clone().requires_grad_())
                    out.backward(torch.ones_like(out))

        run_test(benchmark=False)
        run_test(benchmark=True)

    def test_conv_modules_raise_error_on_incorrect_input_size(self):
        for dtype in [torch.bfloat16, torch.double, torch.float]:
            modules = [nn.Conv1d(3, 8, 3).to(dtype), nn.ConvTranspose1d(3, 8, 3).to(dtype),
                       nn.Conv2d(3, 8, 3).to(dtype), nn.ConvTranspose2d(3, 8, 3).to(dtype),
                       nn.Conv3d(3, 8, 3).to(dtype), nn.ConvTranspose3d(3, 8, 3).to(dtype)]

            invalid_input_dims = [(2, 4), (2, 4),
                                  (3, 5), (3, 5),
                                  (4, 6), (4, 6)]

            for invalid_dims, module in zip(invalid_input_dims, modules):
                for dims in invalid_dims:
                    input = torch.empty(torch.Size((3, ) * dims))
                    self.assertRaises(RuntimeError, lambda: module(input))

    def test_conv_shapecheck(self):
        def test(should_raise, module, input_size, dtype):
            input = torch.empty(3, *input_size).to(dtype)
            if should_raise:
                self.assertRaises(RuntimeError, lambda: module(input))
            else:
                # just run it to ensure no exception raised.
                module(input)

        for dtype in [torch.bfloat16, torch.float, torch.double]:
            # Conv1d
            test(True, nn.Conv1d(1, 1, 3).to(dtype), (1, 2), dtype)
            test(True, nn.Conv1d(1, 1, 3, stride=2).to(dtype), (1, 2), dtype)
            test(False, nn.Conv1d(1, 1, 2).to(dtype), (1, 2), dtype)
            test(False, nn.Conv1d(1, 1, 2, stride=2).to(dtype), (1, 2), dtype)
            test(False, nn.Conv1d(1, 1, 3, stride=2, padding=1).to(dtype), (1, 2), dtype)

            # Conv2d
            test(True, nn.Conv2d(1, 1, (3, 3)).to(dtype), (1, 2, 2), dtype)
            test(False, nn.Conv2d(1, 1, (3, 3)).to(dtype), (1, 3, 3), dtype)
            test(False, nn.Conv2d(1, 1, (3, 3), padding=1).to(dtype), (1, 2, 2), dtype)

            # Conv3D
            test(True, nn.Conv3d(1, 1, (3, 3, 3)).to(dtype), (1, 2, 2, 2), dtype)
            test(False, nn.Conv3d(1, 1, (3, 3, 3)).to(dtype), (1, 3, 3, 3), dtype)
            test(False, nn.Conv3d(1, 1, (3, 3, 3), padding=1).to(dtype), (1, 2, 2, 2), dtype)

    def test_ConvTranspose2d_output_size(self):
        m = nn.ConvTranspose2d(3, 4, 3, 3, 0, 2)
        i = torch.randn(2, 3, 6, 6)
        for h in range(15, 22):
            for w in range(15, 22):
                if 18 <= h <= 20 and 18 <= w <= 20:
                    output = m(i, output_size=(h, w))
                    self.assertEqual(output.size()[2:], (h, w))
                else:
                    self.assertRaises(ValueError, lambda: m(i, (h, w)))

    def test_ConvTranspose3d_correct_output_size(self):
        # Check that ConvTranspose3d can take a 5d output_size.
        m = nn.ConvTranspose3d(2, 2, 2)
        i = torch.rand(1, 2, 1, 1, 1)
        out = m(i, output_size=(1, 2, 2, 2, 2))

    @unittest.skipIf(not TEST_CUDA, 'CUDA not available')
    def test_ConvTranspose2d_half_cublas_gemm(self):
        with torch.backends.cudnn.flags(enabled=False):
            inputs = torch.randn(1, 1, 16, 16, device='cuda', dtype=torch.half)
            deconv = nn.ConvTranspose2d(
                1, 1, 3, stride=2, padding=1, output_padding=1).cuda().half()
            output = deconv(inputs)
            output.mean().backward()

    # For https://github.com/pytorch/pytorch/pull/1273
    # Almost identical to the above `test_Conv2d_naive_groups`
    def test_Conv2d_groups_nobias(self):
        dev_dtypes = [("cpu", torch.float)]
        if TEST_CUDA:
            dev_dtypes += [("cuda", torch.float), ("cuda", torch.half)]
        if TEST_WITH_ROCM:
            dev_dtypes += [("cuda", torch.bfloat16)]
        for device, dtype in dev_dtypes:
            m = nn.Conv2d(4, 4, kernel_size=3, groups=2, bias=False).to(device, dtype)
            i = torch.randn(2, 4, 6, 6, device=device, dtype=dtype, requires_grad=True)
            output = m(i)
            grad_output = torch.randn(2, 4, 4, 4, device=device, dtype=dtype)
            output.backward(grad_output)

            m1 = nn.Conv2d(2, 2, kernel_size=3, bias=False).to(device, dtype)
            m1.weight.data.copy_(m.weight.data[:2])
            i1 = i.data[:, :2].contiguous().requires_grad_(True)
            output1 = m1(i1)
            output1.backward(grad_output[:, :2].contiguous())

            m2 = nn.Conv2d(2, 2, kernel_size=3, bias=False).to(device, dtype)
            m2.weight.data.copy_(m.weight.data[2:])
            i2 = i.data[:, 2:].contiguous().requires_grad_(True)
            output2 = m2(i2)
            output2.backward(grad_output[:, 2:].contiguous())

            self.assertEqual(output, torch.cat([output1, output2], 1))
            self.assertEqual(i.grad.data,
                             torch.cat([i1.grad.data, i2.grad.data], 1),
                             dtype2prec[dtype])
            self.assertEqual(m.weight.grad.data,
                             torch.cat([m1.weight.grad.data, m2.weight.grad.data], 0),
                             1e-1 if dtype == torch.half else dtype2prec[dtype])

    # Almost identical to the above `test_Conv2d_naive_groups`
    # Covering special case when group > 1, input-channel / group < 16 and output-channel is multiple of 16
    # See also https://github.com/pytorch/pytorch/pull/18463#issuecomment-476563686
    # and https://github.com/pytorch/pytorch/pull/18463#issuecomment-477001024
    def test_Conv2d_groups_nobias_v2(self):
        torch.manual_seed(123)
        dev_dtypes = [("cpu", torch.float)]
        if TEST_CUDA:
            dev_dtypes += [("cuda", torch.float), ("cuda", torch.half)]
        if TEST_WITH_ROCM:
            dev_dtypes += [("cuda", torch.bfloat16)]
        for device, dtype in dev_dtypes:
            m = nn.Conv2d(4, 16, kernel_size=3, groups=2, bias=False).to(device, dtype)
            i = torch.randn(2, 4, 6, 6, device=device, dtype=dtype, requires_grad=True)
            output = m(i)
            grad_output = torch.randn(2, 16, 4, 4, device=device, dtype=dtype)
            output.backward(grad_output)

            m1 = nn.Conv2d(2, 8, kernel_size=3, bias=False).to(device, dtype)
            m1.weight.data.copy_(m.weight.data[:8])
            i1 = i.data[:, :2].contiguous().requires_grad_(True)
            output1 = m1(i1)
            output1.backward(grad_output[:, :8].contiguous())

            m2 = nn.Conv2d(2, 8, kernel_size=3, bias=False).to(device, dtype)
            m2.weight.data.copy_(m.weight.data[8:])
            i2 = i.data[:, 2:].contiguous().requires_grad_(True)
            output2 = m2(i2)
            output2.backward(grad_output[:, 8:].contiguous())

            self.assertEqual(output, torch.cat([output1, output2], 1))
            self.assertEqual(i.grad.data,
                             torch.cat([i1.grad.data, i2.grad.data], 1),
                             dtype2prec[dtype])
            self.assertEqual(m.weight.grad.data,
                             torch.cat([m1.weight.grad.data, m2.weight.grad.data], 0),
                             1e-1 if dtype == torch.half else dtype2prec[dtype])

    # Very similar to test_Conv2d_naive_groups but with special care to handle
    # the number of groups == number of input channels
    @unittest.skipIf(not TEST_CUDA, 'CUDA not available')
    @repeat_test_for_types(ALL_TENSORTYPES)
    def test_Conv2d_depthwise_naive_groups_cuda(self, dtype=torch.float):
        for depth_multiplier in [1, 2]:
            m = nn.Conv2d(2, 2 * depth_multiplier, kernel_size=3, groups=2).to("cuda", dtype)
            i = torch.randn(2, 2, 6, 6, device="cuda", dtype=dtype).div_(2).requires_grad_()
            output = m(i)
            grad_output = torch.randn(2, 2 * depth_multiplier, 4, 4, device="cuda", dtype=dtype) / 2
            output.backward(grad_output)

            offset = 1 * depth_multiplier

            m1 = nn.Conv2d(1, 1 * depth_multiplier, kernel_size=3).to("cuda", dtype)
            m1.weight.data = m.weight.data[:offset].clone()
            m1.bias.data = m.bias.data[:offset].clone()
            i1 = i.detach()[:, :1].clone().requires_grad_()
            output1 = m1(i1)
            output1.backward(grad_output[:, :offset].contiguous())

            m2 = nn.Conv2d(1, 1 * depth_multiplier, kernel_size=3).to("cuda", dtype)
            m2.weight.data.copy_(m.weight.data[offset:])
            m2.bias.data.copy_(m.bias.data[offset:])
            i2 = i.detach()[:, 1:].clone().requires_grad_()
            output2 = m2(i2)
            output2.backward(grad_output[:, offset:].contiguous())

            self.assertEqual(output, torch.cat([output1, output2], 1),
                             prec=dtype2prec[dtype])
            self.assertEqual(i.grad.data,
                             torch.cat([i1.grad.data, i2.grad.data], 1),
                             prec=dtype2prec[dtype])
            self.assertEqual(m.bias.grad.data,
                             torch.cat([m1.bias.grad.data,
                                        m2.bias.grad.data], 0),
                             prec=dtype2prec[dtype])
            self.assertEqual(m.weight.grad.data,
                             torch.cat([m1.weight.grad.data,
                                        m2.weight.grad.data], 0),
                             prec=dtype2prec[dtype])

    def test_MaxUnpool2d_output_size(self):
        m = nn.MaxPool2d(3, stride=2, return_indices=True)
        mu = nn.MaxUnpool2d(3, stride=2)
        big_t = torch.rand(1, 1, 6, 6)
        big_t[0][0][4][4] = 100
        output_big, indices_big = m(big_t)
        self.assertRaises(RuntimeError, lambda: mu(output_big, indices_big))

        small_t = torch.rand(1, 1, 5, 5)
        for i in range(0, 4, 2):
            for j in range(0, 4, 2):
                small_t[:, :, i, j] = 100
        output_small, indices_small = m(small_t)
        for h in range(3, 10):
            for w in range(3, 10):
                if 4 <= h <= 6 and 4 <= w <= 6:
                    size = (h, w)
                    if h == 6:
                        size = (1, 1) + size

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

        input = torch.randn(2, 4)

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
                input = torch.randn(3, 10)
                hx = torch.randn(3, 20)
                cell = module(10, 20, bias=bias)
                for _ in range(6):
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

        input = cast(torch.randn(3, 5))
        target = cast(torch.randn(5, 3))
        for _name, fn in losses.items():
            self.assertRaises(Exception, lambda: fn(input, target))

    def test_loss_equal_input_target_shape(self):
        self._test_loss_equal_input_target_shape(lambda x: x)

    def test_mse_loss_size_warning(self):
        i = torch.randn((10, 1), requires_grad=True)
        t = torch.randn((10,))
        with warnings.catch_warnings(record=True) as w:
            # Ensure warnings are being shown
            warnings.simplefilter("always")
            # Trigger Warning
            F.mse_loss(i, t)
            # Check warning occurs
            self.assertEqual(len(w), 1)
            self.assertIn('Please ensure they have the same size.', str(w[0]))

    def test_nll_loss_mismatched_batch(self):
        x = torch.randn((10, 3), requires_grad=True)
        # t should have size (10,)
        t = torch.zeros((3,), dtype=torch.int64)
        with self.assertRaisesRegex(ValueError, 'Expected.*batch_size'):
            F.nll_loss(x, t)

    def test_nll_loss_out_of_bounds_ignore_index(self):
        x = torch.randn(6, 3, requires_grad=True)
        t = torch.tensor([0, 1, 255, 0, 1, 2], dtype=torch.int64)
        for reduction in ['mean', 'none']:
            F.nll_loss(x, t, ignore_index=255, reduction=reduction).sum().backward()

    def test_poisson_nll_loss_reduction_modes(self):
        input = torch.tensor([0.5, 1.5, 2.5])
        target = torch.tensor([1., 2., 3.])
        component_wise_loss = torch.exp(input) - target * input
        self.assertEqual(component_wise_loss,
                         F.poisson_nll_loss(input, target, reduction='none'))
        self.assertEqual(torch.sum(component_wise_loss),
                         F.poisson_nll_loss(input, target, reduction='sum'))
        self.assertEqual(torch.mean(component_wise_loss),
                         F.poisson_nll_loss(input, target, reduction='mean'))
        with self.assertRaisesRegex(ValueError, 'is not valid'):
            F.poisson_nll_loss(input, target, reduction='total')

    def test_KLDivLoss_batch_mean(self):
        input_shape = (2, 5)
        log_prob1 = F.log_softmax(torch.randn(input_shape), 1)
        prob2 = F.softmax(torch.randn(input_shape), 1)

        loss = nn.KLDivLoss(reduction='batchmean')
        l = loss(log_prob1, prob2)

        loss_none_reduce = nn.KLDivLoss(reduction='sum')(log_prob1, prob2)
        expected = loss_none_reduce / input_shape[0]

        self.assertEqual(l, expected)

    def test_CTCLoss_typechecks(self):
        target_lengths = torch.tensor([30, 25, 20])
        input_lengths = torch.tensor([50, 50, 50])
        targets = torch.randint(1, 15, (sum(target_lengths),), dtype=torch.int)
        log_probs = torch.randn(50, 3, 15, dtype=torch.float).log_softmax(2)
        with self.assertRaises(RuntimeError):
            _input_lengths = input_lengths.to(dtype=torch.float)
            torch.nn.functional.ctc_loss(log_probs, targets, _input_lengths, target_lengths)
        with self.assertRaises(RuntimeError):
            target_lengths = target_lengths.to(dtype=torch.float)
            torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths)

    @unittest.skipIf(not TEST_CUDA, 'CUDA not available')
    def test_CTCLoss_lengthchecks_cuda(self):
        target_lengths = [30, 25, 20]
        input_lengths = [50, 50, 50]
        targets = torch.randint(1, 15, (3, 29), dtype=torch.long, device='cuda')
        log_probs = torch.randn(50, 3, 15, dtype=torch.float, device='cuda').log_softmax(2)
        with self.assertRaises(RuntimeError):
            torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths)

    def test_CTCLoss_lengthchecks_cpu(self):
        target_lengths = [30, 25, 20]
        input_lengths = [50, 50, 50]
        targets = torch.randint(1, 15, (3, 29), dtype=torch.int)
        log_probs = torch.randn(50, 3, 15, dtype=torch.float).log_softmax(2)
        with self.assertRaises(RuntimeError):
            torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths)

    @unittest.skipIf(not TEST_CUDA, 'CUDA not available')
    def test_CTCLoss_long_targets(self):
        input_length = 4000
        vocab_size = 3
        batch_size = 4
        target_length = 1200

        log_probs = torch.randn(input_length, batch_size, vocab_size).log_softmax(2).requires_grad_()
        targets = torch.randint(low=1, high=vocab_size - 1, size=(batch_size, target_length), dtype=torch.long)
        input_lengths = batch_size * [input_length]
        target_lengths = batch_size * [target_length]

        res_cpu = torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths,
                                               reduction='sum', zero_infinity=True)
        grad_out = torch.randn_like(res_cpu)
        grad_cpu, = torch.autograd.grad(res_cpu, log_probs, grad_out)

        with torch.backends.cudnn.flags(enabled=False):
            res_gpu = torch.nn.functional.ctc_loss(log_probs.cuda(), targets.cuda(), input_lengths, target_lengths,
                                                   reduction='sum', zero_infinity=True)
            grad_gpu, = torch.autograd.grad(res_gpu, log_probs, grad_out.cuda())
        self.assertAlmostEqual(res_cpu, res_gpu, delta=1e-4)
        self.assertAlmostEqual(grad_cpu, grad_gpu, delta=1e-4)

    @unittest.skipIf(not TEST_CUDA, 'CUDA not available')
    def test_CTCLoss_zero_infinity(self):
        target_lengths = [60, 25, 20]
        input_lengths = [50, 50, 50]
        targets = torch.randint(1, 15, (sum(target_lengths),), dtype=torch.int, device='cuda')
        log_probs = torch.randn(50, 3, 15, dtype=torch.float, device='cuda').log_softmax(2).requires_grad_()
        res = torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths,
                                           reduction='sum', zero_infinity=True)
        with torch.backends.cudnn.flags(enabled=False):
            res2 = torch.nn.functional.ctc_loss(log_probs, targets.cuda().long(), input_lengths, target_lengths,
                                                reduction='sum', zero_infinity=True)
        res_cpu = torch.nn.functional.ctc_loss(log_probs.cpu(), targets.cpu(), input_lengths, target_lengths,
                                               reduction='sum', zero_infinity=True)

        self.assertAlmostEqual(res2, res, delta=1e-4)
        self.assertAlmostEqual(res_cpu, res.cpu(), delta=1e-4)
        g1, = torch.autograd.grad(res, log_probs)
        g2, = torch.autograd.grad(res2, log_probs)
        g3, = torch.autograd.grad(res_cpu, log_probs)
        self.assertAlmostEqual(g2, g3, delta=1e-4)
        self.assertAlmostEqual(g1, g2, delta=1e-4)
        self.assertTrue((g1 == g1).all().item())  # check that we don't have NaN

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
        input = torch.randn(3, input_size)
        bad_hx = torch.randn(1, hidden_size)
        good_hx = torch.randn(3, hidden_size)

        # Test hidden/input batch size broadcasting
        test_all(hidden_size, bad_hx, good_hx, input_size, input)

        # Test hx's hidden_size vs module's hidden_size broadcasting
        bad_hx = torch.randn(3, 1)
        test_all(hidden_size, bad_hx, good_hx, input_size, input)

        # Test input's input_size vs module's input_size broadcasting
        bad_input = torch.randn(3, 1)
        test_all(hidden_size, good_hx, good_hx, input_size, bad_input)

    def test_invalid_dropout_p(self):
        v = torch.ones(1)
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
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([4, 5])
        c = torch.tensor([6])

        # batch_first = true
        expected = torch.tensor([[4, 5, 0], [1, 2, 3], [6, 0, 0]])
        padded = rnn_utils.pad_sequence([b, a, c], True)
        self.assertEqual(padded, expected)

        # batch_first = false
        padded = rnn_utils.pad_sequence([b, a, c])
        self.assertEqual(padded, expected.transpose(0, 1))

        # pad with non-zero value
        expected = torch.tensor([[4, 5, 1], [1, 2, 3], [6, 1, 1]])
        padded = rnn_utils.pad_sequence([b, a, c], True, 1)
        self.assertEqual(padded, expected)

        # Test pad sorted sequence
        expected = torch.tensor([[1, 2, 3], [4, 5, 0], [6, 0, 0]])
        padded = rnn_utils.pad_sequence([a, b, c], True)
        self.assertEqual(padded, expected)

        # more dimensions
        maxlen = 9
        for num_dim in (0, 1, 2, 3):
            sequences = []
            trailing_dims = [4] * num_dim
            for i in range(1, maxlen + 1):
                seq_len = i * i
                sequences.append(torch.rand(seq_len, 5, *trailing_dims))
            random.shuffle(sequences)
            expected = []
            for seq in sequences:
                expected.append(pad(seq, maxlen * maxlen))
            # batch first = true
            expected = torch.stack(expected)
            padded = rnn_utils.pad_sequence(sequences, True)
            self.assertEqual(padded, expected)

            # batch first = false
            padded = rnn_utils.pad_sequence(sequences)
            self.assertEqual(padded, expected.transpose(0, 1))

    def test_pack_sequence(self):
        def _compatibility_test(sequences, lengths, batch_first, enforce_sorted=False):
            padded = rnn_utils.pad_sequence(sequences, batch_first)
            packed = rnn_utils.pack_sequence(sequences, enforce_sorted)
            unpacked = rnn_utils.pad_packed_sequence(packed, batch_first)
            self.assertEqual(padded, unpacked[0])
            pack_padded = rnn_utils.pack_padded_sequence(
                padded, lengths, batch_first, enforce_sorted)
            self.assertEqual(packed, pack_padded)

        # single dimensional
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([4, 5])
        c = torch.tensor([6])
        packed = rnn_utils.pack_sequence([a, b, c], enforce_sorted=False)
        expected = torch.tensor([1, 4, 6, 2, 5, 3])
        self.assertEqual(packed.batch_sizes, [3, 2, 1])
        self.assertEqual(packed.data.data, expected)
        self.assertEqual(packed.sorted_indices, [0, 1, 2])
        self.assertEqual(packed.unsorted_indices, [0, 1, 2])

        packed_unsorted = rnn_utils.pack_sequence([b, c, a], enforce_sorted=False)
        self.assertEqual(packed_unsorted.batch_sizes, [3, 2, 1])
        self.assertEqual(packed_unsorted.data.data, expected)
        self.assertEqual(packed_unsorted.sorted_indices, [2, 0, 1])
        self.assertEqual(packed_unsorted.unsorted_indices, [1, 2, 0])

        # single dimensional, enforce_sorted = True
        packed_enforce_sorted = rnn_utils.pack_sequence([a, b, c], enforce_sorted=True)
        self.assertEqual(packed_enforce_sorted.batch_sizes, [3, 2, 1])
        self.assertEqual(packed_enforce_sorted.data.data, expected)
        self.assertTrue(packed_enforce_sorted.sorted_indices is None)
        self.assertTrue(packed_enforce_sorted.unsorted_indices is None)

        with self.assertRaisesRegex(RuntimeError, 'must be sorted in decreasing order'):
            rnn_utils.pack_sequence([b, c, a], enforce_sorted=True)

        with self.assertRaisesRegex(RuntimeError, 'You can pass `enforce_sorted=False`'):
            rnn_utils.pack_sequence([b, c, a], enforce_sorted=True)

        # more dimensions
        maxlen = 9
        for num_dim in (0, 1, 2, 3):
            sequences = []
            lengths = []
            trailing_dims = [4] * num_dim
            for i in range(maxlen, 0, -1):
                seq_len = i * i
                lengths.append(seq_len)
                sequences.append(torch.rand(seq_len, 5, *trailing_dims))
            unsorted_sequences = [s.clone() for s in sequences]
            random.shuffle(unsorted_sequences)
            unsorted_sequences_lengths = [t.size(0) for t in unsorted_sequences]

            # compatibility with other utilities
            for batch_first in (True, False):
                for enforce_sorted in (True, False):
                    _compatibility_test(sequences, lengths, batch_first, enforce_sorted)
                _compatibility_test(unsorted_sequences, unsorted_sequences_lengths,
                                    batch_first)

    def test_pack_padded_sequence(self):
        def generate_test_case(sorted_lengths, should_shuffle):
            def pad(tensor, length):
                return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

            max_length = sorted_lengths[0]
            batch_sizes = [sum(map(bool, filter(lambda x: x >= i, sorted_lengths)))
                           for i in range(1, max_length + 1)]
            offset = 0
            padded = torch.cat([pad(i * 100 + torch.arange(1., 5 * l + 1).view(l, 1, 5), max_length)
                                for i, l in enumerate(sorted_lengths, 1)], 1)
            expected_data = [[torch.arange(1., 6) + (i + 1) * 100 + 5 * n for i in range(batch_size)]
                             for n, batch_size in enumerate(batch_sizes)]
            expected_data = list(itertools.chain.from_iterable(expected_data))
            expected_data = torch.stack(expected_data, dim=0)

            if should_shuffle:
                # Shuffle the padded sequence to create an unsorted sequence
                permutation = list(range(len(sorted_lengths)))
                random.shuffle(permutation)

                unsorted_indices = torch.tensor(permutation)
                padded = padded.index_select(1, unsorted_indices)
                lengths = torch.tensor(sorted_lengths).index_select(0, unsorted_indices)
            else:
                unsorted_indices = None
                lengths = sorted_lengths

            return padded.requires_grad_(), lengths, expected_data, batch_sizes, unsorted_indices

        test_cases = [
            # sorted_lengths, should_shuffle
            [[10, 8, 4, 2, 2, 2, 1], False],
            [[11, 10, 8, 6, 4, 3, 1], False],
            [[11, 10, 8, 6, 4, 3, 1], True],
        ]

        for test_case, batch_first in itertools.product(test_cases, (True, False)):
            sorted_lengths, should_shuffle = test_case
            padded, lengths, expected_data, batch_sizes, unsorted_indices = generate_test_case(
                sorted_lengths, should_shuffle)

            src = padded
            if batch_first:
                src = src.transpose(0, 1)

            # check output
            packed = rnn_utils.pack_padded_sequence(src, lengths, batch_first=batch_first,
                                                    enforce_sorted=not should_shuffle)
            self.assertEqual(packed.data.data, expected_data)
            self.assertEqual(packed.batch_sizes, batch_sizes)
            self.assertEqual(packed.unsorted_indices, unsorted_indices)

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

        # test error messages
        with self.assertRaisesRegex(RuntimeError, 'You can pass `enforce_sorted=False`'):
            packed = rnn_utils.pack_padded_sequence(torch.randn(3, 3), [1, 3, 2])
        with self.assertRaisesRegex(RuntimeError, 'empty tensor'):
            packed = rnn_utils.pack_padded_sequence(torch.randn(0, 0), [])

    def test_LSTM_cell(self):
        # this is just a smoke test; these modules are implemented through
        # autograd so no Jacobian test is needed
        for bias in (True, False):
            input = torch.randn(3, 10)
            hx = torch.randn(3, 20)
            cx = torch.randn(3, 20)
            lstm = nn.LSTMCell(10, 20, bias=bias)
            for _ in range(6):
                hx, cx = lstm(input, (hx, cx))

            (hx + cx).sum().backward()

    @unittest.skipIf(not TEST_CUDA, 'CUDA not available')
    def test_pack_sequence_batch_sizes_throw(self):
        with self.assertRaisesRegex(ValueError, r"batch_sizes should always be on CPU"):
            m = nn.LSTM(3, 4, bidirectional=True, num_layers=2).to('cuda')
            a = torch.rand(5, 3, device='cuda')
            b = torch.tensor([1, 1, 1, 1, 1], device='cuda')
            input = nn.utils.rnn.PackedSequence(a, b)

    def test_Transformer_cell(self):
        # this is just a smoke test; these modules are implemented through
        # autograd so no Jacobian test is needed
        d_model = 512
        nhead = 16
        num_encoder_layers = 4
        num_decoder_layers = 3
        dim_feedforward = 256
        dropout = 0.3
        bsz = 8
        seq_length = 35
        tgt_length = 15

        transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers,
                                     dim_feedforward, dropout)
        src = torch.randn(seq_length, bsz, d_model)
        src_mask = transformer.generate_square_subsequent_mask(seq_length).double()
        tgt = torch.randn(tgt_length, bsz, d_model)
        tgt_mask = transformer.generate_square_subsequent_mask(tgt_length).double()
        memory_mask = torch.randn(tgt_length, seq_length).double()
        src_key_padding_mask = torch.rand(bsz, seq_length) >= 0.5
        tgt_key_padding_mask = torch.rand(bsz, tgt_length) >= 0.5
        memory_key_padding_mask = torch.rand(bsz, seq_length) >= 0.5

        output = transformer(src, tgt,
                             src_mask=src_mask,
                             tgt_mask=tgt_mask,
                             memory_mask=memory_mask,
                             src_key_padding_mask=src_key_padding_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask,
                             memory_key_padding_mask=memory_key_padding_mask)
        output.sum().backward()

    def test_transformerencoderlayer(self):
        # this is a deterministic test for TransformerEncoderLayer
        d_model = 4
        nhead = 2
        dim_feedforward = 16
        dropout = 0.0
        bsz = 2

        model = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)

        # set constant weights of the model
        for idx, p in enumerate(model.parameters()):
            x = p.data
            sz = x.view(-1).size(0)
            shape = x.shape
            x = torch.cos(torch.arange(0, sz).float().view(shape))
            p.data.copy_(x)

        # deterministic input
        encoder_input = torch.Tensor([[[20, 30, 40, 50]]])
        result = model(encoder_input)
        ref_output = torch.Tensor([[[2.258703, 0.127985, -0.697881, 0.170862]]])
        result = result.detach().numpy()
        ref_output = ref_output.detach().numpy()
        self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
        np.testing.assert_allclose(result, ref_output, atol=1e-5)
        # 0 values are NOT masked. This shouldn't mask anything.
        mask = torch.Tensor([[0]]) == 1
        result = model(encoder_input, src_key_padding_mask=mask)
        result = result.detach().numpy()
        self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
        np.testing.assert_allclose(result, ref_output, atol=1e-5)
        # 1 values are masked. Since there is only 1 input embedding this
        # will result in nan.
        mask = torch.Tensor([[1]]) == 1
        result = model(encoder_input, src_key_padding_mask=mask)
        result = result.detach().numpy()
        self.assertTrue(np.isnan(result).all())

        # deterministic input
        encoder_input = torch.Tensor([[[1, 2, 3, 4]],
                                      [[5, 6, 7, 8]]])
        result = model(encoder_input)
        ref_output = torch.Tensor([[[2.272644, 0.119035, -0.691669, 0.153486]],
                                   [[2.272644, 0.119035, -0.691669, 0.153486]]])
        result = result.detach().numpy()
        ref_output = ref_output.detach().numpy()
        self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
        np.testing.assert_allclose(result, ref_output, atol=1e-5)
        # all 0 which is no masking
        mask = torch.Tensor([[0, 0]]) == 1
        result = model(encoder_input, src_key_padding_mask=mask)
        result = result.detach().numpy()
        self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
        np.testing.assert_allclose(result, ref_output, atol=1e-5)
        mask = torch.Tensor([[1, 0]]) == 1
        result = model(encoder_input, src_key_padding_mask=mask)
        ref_output = torch.Tensor([[[2.301516, 0.092249, -0.679101, 0.103088]],
                                   [[2.301516, 0.092249, -0.679101, 0.103088]]])
        result = result.detach().numpy()
        ref_output = ref_output.detach().numpy()
        self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
        np.testing.assert_allclose(result, ref_output, atol=1e-5)

        # deterministic input
        encoder_input = torch.Tensor([[[0.7462, 0.6653, 0.5679, 0.4891],
                                       [0.5387, 0.1655, 0.3565, 0.0471]],
                                      [[0.8335, 0.2799, 0.5031, 0.2947],
                                       [0.1402, 0.0318, 0.7636, 0.1346]],
                                      [[0.6333, 0.9344, 0.1376, 0.9938],
                                       [0.8924, 0.2872, 0.6692, 0.2944]],
                                      [[0.9897, 0.6915, 0.3154, 0.1733],
                                       [0.8645, 0.3513, 0.3064, 0.0767]],
                                      [[0.8117, 0.2366, 0.4838, 0.7881],
                                       [0.3718, 0.4945, 0.9511, 0.0864]]])
        result = model(encoder_input)
        ref_output = torch.Tensor([[[2.428589, 0.020835, -0.602055, -0.085249],
                                    [2.427987, 0.021213, -0.602496, -0.084103]],
                                   [[2.424689, 0.019155, -0.604793, -0.085672],
                                    [2.413863, 0.022211, -0.612486, -0.072490]],
                                   [[2.433774, 0.021598, -0.598343, -0.087548],
                                    [2.425104, 0.019748, -0.604515, -0.084839]],
                                   [[2.436185, 0.022682, -0.596625, -0.087261],
                                    [2.433556, 0.021891, -0.598509, -0.086832]],
                                   [[2.416246, 0.017512, -0.610712, -0.082961],
                                    [2.422901, 0.024187, -0.606178, -0.074929]]])
        result = result.detach().numpy()
        ref_output = ref_output.detach().numpy()
        self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
        np.testing.assert_allclose(result, ref_output, atol=1e-5)
        # all 0
        mask = torch.zeros([2, 5]) == 1
        result = model(encoder_input, src_key_padding_mask=mask)
        result = result.detach().numpy()
        self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
        np.testing.assert_allclose(result, ref_output, atol=1e-5)
        mask[0, 1] = 1
        mask[1, 3] = 1
        mask[1, 4] = 1
        result = model(encoder_input, src_key_padding_mask=mask)
        ref_output = torch.Tensor([[[2.429026, 0.020793, -0.601741, -0.085642],
                                    [2.428811, 0.021445, -0.601912, -0.084252]],
                                   [[2.425009, 0.019155, -0.604566, -0.085899],
                                    [2.415408, 0.02249 , -0.611415, -0.073]],
                                   [[2.434199, 0.021682, -0.598039, -0.087699],
                                    [2.42598, 0.019941, -0.603896, -0.085091]],
                                   [[2.436457, 0.022736, -0.59643 , -0.08736],
                                    [2.434021, 0.022093, -0.598179, -0.08679]],
                                   [[2.416531, 0.017498, -0.610513, -0.083181],
                                    [2.4242, 0.024653, -0.605266, -0.074959]]])
        result = result.detach().numpy()
        ref_output = ref_output.detach().numpy()
        self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
        np.testing.assert_allclose(result, ref_output, atol=1e-5)

    def test_transformerencoderlayer_gelu(self):
        # this is a deterministic test for TransformerEncoderLayer with gelu activation
        d_model = 4
        nhead = 2
        dim_feedforward = 16
        dropout = 0.0
        bsz = 2
        activation = "gelu"

        model = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)

        # set constant weights of the model
        for idx, p in enumerate(model.parameters()):
            x = p.data
            sz = x.view(-1).size(0)
            shape = x.shape
            x = torch.cos(torch.arange(0, sz).float().view(shape))
            p.data.copy_(x)

        # deterministic input
        encoder_input = torch.Tensor([[[20, 30, 40, 50]]])
        result = model(encoder_input)
        ref_output = torch.Tensor([[[2.249815, 0.131006, -0.702199, 0.177868]]])
        self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
        torch.testing.assert_allclose(result, ref_output)

        # deterministic input
        encoder_input = torch.Tensor([[[1, 2, 3, 4]],
                                      [[5, 6, 7, 8]]])
        result = model(encoder_input)
        ref_output = torch.Tensor([[[2.264103, 0.121417, -0.696012, 0.159724]],
                                   [[2.264103, 0.121417, -0.696012, 0.159724]]])
        self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
        torch.testing.assert_allclose(result, ref_output)

        # deterministic input
        encoder_input = torch.Tensor([[[0.7462, 0.6653, 0.5679, 0.4891],
                                       [0.5387, 0.1655, 0.3565, 0.0471]],
                                      [[0.8335, 0.2799, 0.5031, 0.2947],
                                       [0.1402, 0.0318, 0.7636, 0.1346]],
                                      [[0.6333, 0.9344, 0.1376, 0.9938],
                                       [0.8924, 0.2872, 0.6692, 0.2944]],
                                      [[0.9897, 0.6915, 0.3154, 0.1733],
                                       [0.8645, 0.3513, 0.3064, 0.0767]],
                                      [[0.8117, 0.2366, 0.4838, 0.7881],
                                       [0.3718, 0.4945, 0.9511, 0.0864]]])
        result = model(encoder_input)
        ref_output = torch.Tensor([[[2.42163188, 0.03227153, -0.60714219, -0.05908082],
                                    [2.42151276, 0.03302179, -0.60722523, -0.05762651]],
                                   [[2.41926761, 0.02974034, -0.60879519, -0.0621269],
                                    [2.41626395, 0.03539356, -0.61087842, -0.04978623]],
                                   [[2.42382808, 0.03218872, -0.6055963, -0.06073591],
                                    [2.41983477, 0.03085259, -0.60840145, -0.06046414]],
                                   [[2.42500749, 0.03328855, -0.60476388, -0.0595334],
                                    [2.4237977, 0.03290575, -0.60561789, -0.05940082]],
                                   [[2.41383916, 0.02686345, -0.61256377, -0.06380707],
                                    [2.42000277, 0.03800944, -0.60824798, -0.04754947]]])
        self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
        torch.testing.assert_allclose(result, ref_output)

    def test_transformerdecoderlayer(self):
        # this is a deterministic test for TransformerDecoderLayer
        d_model = 4
        nhead = 2
        dim_feedforward = 16
        dropout = 0.0
        bsz = 2
        seq_length = 5
        tgt_length = 3

        model = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)

        # set constant weights of the model
        for idx, p in enumerate(model.parameters()):
            x = p.data
            sz = x.view(-1).size(0)
            shape = x.shape
            x = torch.cos(torch.arange(0, sz).float().view(shape))
            p.data.copy_(x)

        # deterministic input
        decoder_input = torch.Tensor([[[20, 30, 40, 50]]])
        memory_input = torch.Tensor([[[60, 70, 80, 90]]])
        result = model(decoder_input, memory_input)
        ref_output = torch.Tensor([[[2.314351, 0.094805, -0.671322, 0.101977]]])
        result = result.detach().numpy()
        ref_output = ref_output.detach().numpy()
        self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
        np.testing.assert_allclose(result, ref_output, atol=1e-5)

        # deterministic input
        decoder_input = torch.Tensor([[[9, 10, 11, 12]],
                                     [[11, 12, 13, 14]]])
        memory_input = torch.Tensor([[[1, 2, 3, 4]]])
        result = model(decoder_input, memory_input)
        result = result.detach().numpy()
        ref_output = torch.Tensor([[[2.422245, 0.051716, -0.606338, -0.024756]],
                                   [[2.422245, 0.051716, -0.606338, -0.024756]]])
        ref_output = ref_output.detach().numpy()
        self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
        np.testing.assert_allclose(result, ref_output, atol=1e-5)

        # deterministic input
        decoder_input = torch.Tensor([[[1, 2, 3, 4]],
                                      [[5, 6, 7, 8]]])
        memory_input = torch.Tensor([[[9, 10, 11, 12]],
                                     [[11, 12, 13, 14]]])
        result = model(decoder_input, memory_input)
        ref_output = torch.Tensor([[[2.343536, 0.085561, -0.654954, 0.074991]],
                                   [[2.343536, 0.085561, -0.654954, 0.074991]]])
        result = result.detach().numpy()
        ref_output = ref_output.detach().numpy()
        self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
        np.testing.assert_allclose(result, ref_output, atol=1e-5)

        # deterministic input
        decoder_input = torch.Tensor([[[0.4517, 0.6793, 0.5313, 0.0034],
                                       [0.2678, 0.3677, 0.4459, 0.7166]],
                                      [[0.8100, 0.3716, 0.4096, 0.1976],
                                       [0.6958, 0.8844, 0.6081, 0.8315]],
                                      [[0.0494, 0.9343, 0.5955, 0.3830],
                                       [0.5404, 0.3464, 0.9378, 0.6200]]])
        memory_input = torch.Tensor([[[0.7462, 0.6653, 0.5679, 0.4891],
                                      [0.5387, 0.1655, 0.3565, 0.0471]],
                                     [[0.8335, 0.2799, 0.5031, 0.2947],
                                      [0.1402, 0.0318, 0.7636, 0.1346]],
                                     [[0.6333, 0.9344, 0.1376, 0.9938],
                                      [0.8924, 0.2872, 0.6692, 0.2944]],
                                     [[0.9897, 0.6915, 0.3154, 0.1733],
                                      [0.8645, 0.3513, 0.3064, 0.0767]],
                                     [[0.8117, 0.2366, 0.4838, 0.7881],
                                      [0.3718, 0.4945, 0.9511, 0.0864]]])
        result = model(decoder_input, memory_input)
        ref_output = torch.Tensor([[[2.430065, 0.027862, -0.601136, -0.073096],
                                    [2.431935, 0.028907, -0.599809, -0.072488]],
                                   [[2.428457, 0.027053, -0.602275, -0.073462],
                                    [2.431970, 0.029387, -0.599789, -0.071621]],
                                   [[2.431934, 0.028196, -0.599802, -0.073809],
                                    [2.432306, 0.028858, -0.599542, -0.072846]]])
        result = result.detach().numpy()
        ref_output = ref_output.detach().numpy()
        self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
        np.testing.assert_allclose(result, ref_output, atol=1e-5)

        # key_padding_mask
        key_padding_mask = torch.zeros(2, 3) == 1
        result = model(decoder_input, memory_input, tgt_key_padding_mask=key_padding_mask)
        ref_output = torch.Tensor([[[2.430065, 0.027862, -0.601136, -0.073096],
                                    [2.431935, 0.028907, -0.599809, -0.072488]],
                                   [[2.428457, 0.027053, -0.602275, -0.073462],
                                    [2.431970, 0.029387, -0.599789, -0.071621]],
                                   [[2.431934, 0.028196, -0.599802, -0.073809],
                                    [2.432306, 0.028858, -0.599542, -0.072846]]])
        result = result.detach().numpy()
        ref_output = ref_output.detach().numpy()
        self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
        np.testing.assert_allclose(result, ref_output, atol=1e-5)

        # key_padding_mask
        key_padding_mask[0, 2] = 1
        key_padding_mask[1, 1] = 1
        key_padding_mask[1, 2] = 1
        result = model(decoder_input, memory_input, tgt_key_padding_mask=key_padding_mask)
        ref_output = torch.Tensor([[[2.430025, 0.027643, -0.601164, -0.073476],
                                    [2.4323, 0.029375, -0.599553, -0.071881]],
                                   [[2.428523, 0.026838, -0.602226, -0.07391],
                                    [2.432634, 0.029842, -0.599318, -0.071253]],
                                   [[2.432278, 0.028152, -0.599555, -0.074139],
                                    [2.432659, 0.029244, -0.599294, -0.072382]]])
        result = result.detach().numpy()
        ref_output = ref_output.detach().numpy()
        self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
        np.testing.assert_allclose(result, ref_output, atol=1e-5)

        # memory_key_padding_mask
        key_padding_mask = torch.zeros(2, 5) == 1
        result = model(decoder_input, memory_input, memory_key_padding_mask=key_padding_mask)
        ref_output = torch.Tensor([[[2.430065, 0.027862, -0.601136, -0.073096],
                                    [2.431935, 0.028907, -0.599809, -0.072488]],
                                   [[2.428457, 0.027053, -0.602275, -0.073462],
                                    [2.431970, 0.029387, -0.599789, -0.071621]],
                                   [[2.431934, 0.028196, -0.599802, -0.073809],
                                    [2.432306, 0.028858, -0.599542, -0.072846]]])
        result = result.detach().numpy()
        ref_output = ref_output.detach().numpy()
        self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
        np.testing.assert_allclose(result, ref_output, atol=1e-5)

        # memory_key_padding_mask
        key_padding_mask[0, 4] = 1
        key_padding_mask[1, 3] = 1
        key_padding_mask[1, 4] = 1
        result = model(decoder_input, memory_input, memory_key_padding_mask=key_padding_mask)
        ref_output = torch.Tensor([[[2.429757, 0.027358, -0.601351, -0.073816],
                                    [2.432692, 0.028583, -0.599263, -0.073634]],
                                   [[2.428247, 0.02662, -0.602419, -0.074123],
                                    [2.432657, 0.029055, -0.599293, -0.072732]],
                                   [[2.431515, 0.027687, -0.600096, -0.074459],
                                    [2.433075, 0.028543, -0.598987, -0.073985]]])
        result = result.detach().numpy()
        ref_output = ref_output.detach().numpy()
        self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
        np.testing.assert_allclose(result, ref_output, atol=1e-5)

    def test_transformerdecoderlayer_gelu(self):
        # this is a deterministic test for TransformerDecoderLayer with gelu activation
        d_model = 4
        nhead = 2
        dim_feedforward = 16
        dropout = 0.0
        bsz = 2
        seq_length = 5
        tgt_length = 3
        activation = "gelu"

        model = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)

        # set constant weights of the model
        for idx, p in enumerate(model.parameters()):
            x = p.data
            sz = x.view(-1).size(0)
            shape = x.shape
            x = torch.cos(torch.arange(0, sz).float().view(shape))
            p.data.copy_(x)

        # deterministic input
        decoder_input = torch.Tensor([[[20, 30, 40, 50]]])
        memory_input = torch.Tensor([[[60, 70, 80, 90]]])
        result = model(decoder_input, memory_input)
        ref_output = torch.Tensor([[[2.306435, 0.095946, -0.675796, 0.10687]]])
        self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
        torch.testing.assert_allclose(result, ref_output)

        # deterministic input
        decoder_input = torch.Tensor([[[9, 10, 11, 12]],
                                     [[11, 12, 13, 14]]])
        memory_input = torch.Tensor([[[1, 2, 3, 4]]])
        result = model(decoder_input, memory_input)
        ref_output = torch.Tensor([[[2.415448, 0.054389, -0.610932, -0.0156613]],
                                   [[2.415448, 0.054389, -0.610932, -0.0156613]]])
        self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
        torch.testing.assert_allclose(result, ref_output)

        # deterministic input
        decoder_input = torch.Tensor([[[1, 2, 3, 4]],
                                      [[5, 6, 7, 8]]])
        memory_input = torch.Tensor([[[9, 10, 11, 12]],
                                     [[11, 12, 13, 14]]])
        result = model(decoder_input, memory_input)
        ref_output = torch.Tensor([[[2.338531, 0.087709, -0.65776, 0.080646]],
                                   [[2.338531, 0.087709, -0.65776, 0.080646]]])
        self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
        torch.testing.assert_allclose(result, ref_output)

        # deterministic input
        decoder_input = torch.Tensor([[[0.4517, 0.6793, 0.5313, 0.0034],
                                       [0.2678, 0.3677, 0.4459, 0.7166]],
                                      [[0.8100, 0.3716, 0.4096, 0.1976],
                                       [0.6958, 0.8844, 0.6081, 0.8315]],
                                      [[0.0494, 0.9343, 0.5955, 0.3830],
                                       [0.5404, 0.3464, 0.9378, 0.6200]]])
        memory_input = torch.Tensor([[[0.7462, 0.6653, 0.5679, 0.4891],
                                      [0.5387, 0.1655, 0.3565, 0.0471]],
                                     [[0.8335, 0.2799, 0.5031, 0.2947],
                                      [0.1402, 0.0318, 0.7636, 0.1346]],
                                     [[0.6333, 0.9344, 0.1376, 0.9938],
                                      [0.8924, 0.2872, 0.6692, 0.2944]],
                                     [[0.9897, 0.6915, 0.3154, 0.1733],
                                      [0.8645, 0.3513, 0.3064, 0.0767]],
                                     [[0.8117, 0.2366, 0.4838, 0.7881],
                                      [0.3718, 0.4945, 0.9511, 0.0864]]])
        result = model(decoder_input, memory_input)
        ref_output = torch.Tensor([[[2.42049104, 0.03443088, -0.60793706, -0.05436271],
                                    [2.42210631, 0.03546578, -0.60679895, -0.05357488]],
                                   [[2.41907674, 0.0336104, -0.60892977, -0.05490462],
                                    [2.42216881, 0.03586554, -0.6067524, -0.05289126]],
                                   [[2.42205716, 0.03488046, -0.60683681, -0.05460596],
                                    [2.42240309, 0.0354595, -0.60659063, -0.05378816]]])
        self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
        torch.testing.assert_allclose(result, ref_output)

    @unittest.skipIf(not (TEST_CUDNN and TEST_MULTIGPU), 'CUDNN or multi-gpu not available')
    def test_cudnn_rnn_dropout_states_device(self):
        rnn = nn.RNN(10, 20, num_layers=2, dropout=.5)
        device = 1
        input = torch.randn(5, 4, 10).cuda(device)
        rnn.cuda(device)
        hx = torch.randn(2, 4, 20).cuda(device)
        output = rnn(input, hx)

    @unittest.skipIf(not TEST_CUDNN, 'CUDNN not available')
    @skipIfRocm
    def test_cudnn_weight_format(self):
        rnns = [
            nn.LSTM(10, 20, batch_first=True),
            nn.GRU(10, 20, batch_first=True),
            nn.RNN(10, 20, batch_first=True)
        ]
        first_warn = True
        for rnn in rnns:
            rnn.cuda()
            input = torch.randn(5, 4, 10, requires_grad=True, device="cuda")
            hx = torch.randn(1, 5, 20, requires_grad=True, device="cuda")
            all_vars = [input, hx] + list(rnn.parameters())
            if isinstance(rnn, nn.LSTM):
                cx = torch.randn(1, 5, 20, requires_grad=True, device="cuda")
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
            with torch.no_grad():
                weight.set_(weight_data)

            for _ in range(2):
                with warnings.catch_warnings(record=True) as w:
                    output_noncontig = rnn(input, hx)
                if first_warn:
                    self.assertEqual(len(w), 1)
                    self.assertIn('weights are not part of single contiguous chunk of memory', w[0].message.args[0])
                    first_warn = False
                    warnings.resetwarnings()
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
            input = torch.randn(5, 4, 10, requires_grad=True, device="cuda")
            hx = torch.randn(2, 5, 20, requires_grad=True, device="cuda")
            all_vars = [input, hx] + list(rnn.parameters())
            opt = torch.optim.SGD(rnn.parameters(), lr=0.1)
            opt.zero_grad()
            if isinstance(rnn, nn.LSTM):
                cx = torch.randn(2, 5, 20, requires_grad=True, device="cuda")
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
    @repeat_test_for_types(NO_HALF_TENSORTYPES)
    def test_cuda_rnn_fused(self, dtype=torch.float):

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
        input_val = torch.randn(seq_length, batch, input_size, dtype=dtype)
        grad_output = torch.randn(seq_length, batch, hidden_size, dtype=dtype)
        hx_val = torch.randn(num_layers, batch, hidden_size, dtype=dtype)
        grad_hy = torch.randn(num_layers, batch, hidden_size, dtype=dtype)
        with torch.backends.cudnn.flags(enabled=False):
            for module in (nn.GRU, nn.LSTM):
                for bias in (True, False):
                    rnn = module(input_size, hidden_size, num_layers, bias=bias).to(dtype)
                    rnn_cuda = module(input_size, hidden_size, num_layers, bias=bias).to("cuda", dtype)
                    copy_rnn(rnn, rnn_cuda)

                    is_lstm = isinstance(rnn, nn.LSTM)
                    if is_lstm:
                        hx = (hx_val.clone().requires_grad_(True),
                              hx_val.clone().add(1).requires_grad_(True))
                        hx_cuda = (hx_val.clone().cuda().requires_grad_(True),
                                   hx_val.clone().cuda().add(1).requires_grad_(True))
                    else:
                        hx = hx_val.clone().requires_grad_(True)
                        hx_cuda = hx_val.clone().cuda().requires_grad_(True)

                    inp = input_val.clone().requires_grad_(True)
                    inp_cu = input_val.clone().cuda().requires_grad_(True)
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

    def test_transformer_args_check(self):
        model_name = 'Transformer'
        d_model = 128
        nhead = 4
        num_encoder_layers = 2
        num_decoder_layers = 3
        dim_feedforward = 65
        dropout = 0.3
        bsz = 3
        seq_len = 35
        tgt_len = 15
        activations = ["relu", "gelu"]

        wrong_bsz = 7
        wrong_d_model = 63
        wrong_nhead = 5
        wrong_activation = "abc"

        def test(encoder_input_shape, decoder_input_shape,
                 src_mask_len=None, tgt_mask_len=None, memory_mask_size=None,
                 src_key_padding_mask_size=None, tgt_key_padding_mask_size=None,
                 memory_key_padding_mask_size=None):
            encoder_input = torch.randn(encoder_input_shape)
            decoder_input = torch.randn(decoder_input_shape)
            model = getattr(nn, model_name)(d_model, nhead, num_encoder_layers,
                                            num_decoder_layers, dim_feedforward, dropout)

            if src_mask_len is not None:
                src_mask = model.generate_square_subsequent_mask(src_mask_len)
            else:
                src_mask = None

            if tgt_mask_len is not None:
                tgt_mask = model.generate_square_subsequent_mask(tgt_mask_len)
            else:
                tgt_mask = None

            if memory_mask_size is not None:
                memory_task = torch.rand(memory_mask_size)
            else:
                memory_task = None

            if src_key_padding_mask_size is not None:
                src_key_padding_mask = torch.rand(src_key_padding_mask_size) >= 0.5
            else:
                src_key_padding_mask = None

            if tgt_key_padding_mask_size is not None:
                tgt_key_padding_mask = torch.rand(tgt_key_padding_mask_size) >= 0.5
            else:
                tgt_key_padding_mask = None

            if memory_key_padding_mask_size is not None:
                memory_key_padding_mask = torch.rand(memory_key_padding_mask_size) >= 0.5
            else:
                memory_key_padding_mask = None

            with self.assertRaises(RuntimeError):
                model(encoder_input, decoder_input,
                      src_mask=src_mask,
                      tgt_mask=tgt_mask,
                      memory_mask=memory_task,
                      src_key_padding_mask=src_key_padding_mask,
                      tgt_key_padding_mask=tgt_key_padding_mask,
                      memory_key_padding_mask=memory_key_padding_mask)


        correct_encoder_input_shape = (seq_len, bsz, d_model)
        correct_decoder_input_shape = (tgt_len, bsz, d_model)

        def update_shape(shape, dim, new_dim_size):
            new_shape = list(shape)
            new_shape[dim] = new_dim_size
            return tuple(new_shape)

        # Incorrect encoder_input batch size
        encoder_input_shape = update_shape(correct_encoder_input_shape, 1, wrong_bsz)
        decoder_input_shape = correct_decoder_input_shape
        test(encoder_input_shape, decoder_input_shape)

        # Incorrect decoder_input batch size
        encoder_input_shape = correct_encoder_input_shape
        decoder_input_shape = update_shape(correct_decoder_input_shape, 1, wrong_bsz)
        test(encoder_input_shape, decoder_input_shape)

        # Incorrect encoder_input input size
        encoder_input_shape = update_shape(correct_encoder_input_shape, 2, wrong_d_model)
        decoder_input_shape = correct_decoder_input_shape
        test(encoder_input_shape, decoder_input_shape)

        # Incorrect decoder_input input size
        encoder_input_shape = correct_encoder_input_shape
        decoder_input_shape = update_shape(correct_decoder_input_shape, 2, wrong_d_model)
        test(encoder_input_shape, decoder_input_shape)

        # Incorrect nhead
        encoder_input_shape = correct_encoder_input_shape
        decoder_input_shape = correct_decoder_input_shape
        with self.assertRaises(AssertionError):
            model = getattr(nn, model_name)(d_model, wrong_nhead, num_encoder_layers,
                                            num_decoder_layers, dim_feedforward, dropout)

        # Incorrect src_mask
        encoder_input_shape = correct_encoder_input_shape
        decoder_input_shape = correct_decoder_input_shape
        wrong_src_mask_size = seq_len + 1
        test(encoder_input_shape, decoder_input_shape, src_mask_len=wrong_src_mask_size)

        # Incorrect tgt_mask
        encoder_input_shape = correct_encoder_input_shape
        decoder_input_shape = correct_decoder_input_shape
        wrong_tgt_mask_size = tgt_len + 1
        test(encoder_input_shape, decoder_input_shape, tgt_mask_len=wrong_tgt_mask_size)

        # Incorrect memory_mask
        encoder_input_shape = correct_encoder_input_shape
        decoder_input_shape = correct_decoder_input_shape
        wrong_tgt_mask_size = tgt_len + 1
        test(encoder_input_shape, decoder_input_shape,
             memory_mask_size=(wrong_tgt_mask_size, wrong_src_mask_size))

        # Incorrect src_key_padding_mask
        encoder_input_shape = correct_encoder_input_shape
        decoder_input_shape = correct_decoder_input_shape
        with self.assertRaises(AssertionError):
            test(encoder_input_shape, decoder_input_shape,
                 src_key_padding_mask_size=(wrong_bsz, wrong_src_mask_size))

        # Incorrect tgt_key_padding_mask
        encoder_input_shape = correct_encoder_input_shape
        decoder_input_shape = correct_decoder_input_shape
        with self.assertRaises(AssertionError):
            test(encoder_input_shape, decoder_input_shape,
                 tgt_key_padding_mask_size=(wrong_bsz, wrong_tgt_mask_size))

        # Incorrect memory_key_padding_mask
        encoder_input_shape = correct_encoder_input_shape
        decoder_input_shape = correct_decoder_input_shape
        with self.assertRaises(AssertionError):
            test(encoder_input_shape, decoder_input_shape,
                 memory_key_padding_mask_size=(wrong_bsz, wrong_src_mask_size))

        # Correct activations
        for activation in activations:
            model = getattr(nn, model_name)(d_model, nhead, num_encoder_layers, num_decoder_layers,
                                            dim_feedforward, dropout, activation)
        # Incorrect activation
        with self.assertRaises(RuntimeError):
            model = getattr(nn, model_name)(d_model, nhead, num_encoder_layers, num_decoder_layers,
                                            dim_feedforward, dropout, wrong_activation)

    def test_transformer_layer_args_check(self):
        model_names = ['TransformerEncoderLayer', 'TransformerDecoderLayer']
        d_model = 128
        nhead = 4
        dim_feedforward = 65
        dropout = 0.3
        bsz = 3
        seq_len = 35
        tgt_len = 15
        activations = ["relu", "gelu"]

        wrong_activation = "abc"

        encoder_input_shape = (seq_len, bsz, d_model)
        decoder_input_shape = (tgt_len, bsz, d_model)

        encoder_input = torch.randn(encoder_input_shape)
        decoder_input = torch.randn(decoder_input_shape)

        for model_name in model_names:
            for activation in activations:
                model = getattr(nn, model_name)(d_model, nhead, dim_feedforward,
                                                dropout, activation)
        # Incorrect activation
        for model_name in model_names:
            with self.assertRaises(RuntimeError):
                model = getattr(nn, model_name)(d_model, nhead, dim_feedforward,
                                                dropout, wrong_activation)

    def test_rnn_args_check(self):
        input_size = 3
        hidden_size = 5
        num_layers = 2
        batch_size = 4
        seq_len = 6
        num_directions = 1
        bad_size = 7  # prime number so that no size can divide it.

        def test(input_shape, hidden_shape, mode):
            for input, hidden in get_inputs(input_shape, hidden_shape, mode):
                model = getattr(nn, mode)(input_size, hidden_size, num_layers)
                self.assertRaises(RuntimeError, lambda: model(input, hidden))

        correct_input_shape = (seq_len, batch_size, input_size)
        correct_hidden_shape = (num_layers * num_directions, batch_size, hidden_size)

        def update_shape(shape, dim, new_dim_size):
            new_shape = list(shape)
            new_shape[dim] = new_dim_size
            return tuple(new_shape)

        def get_inputs(input_shape, hidden_shape, mode):
            '''returns list( tuple(input, hidden) )
            where input, hidden are inputs to a model'''
            input = torch.randn(input_shape)
            hidden = torch.randn(hidden_shape)
            if mode != 'LSTM':
                return [(input, hidden)]
            if hidden_shape == correct_hidden_shape:
                return [(input, (hidden, hidden))]
            good_hidden = torch.randn(correct_hidden_shape)
            return [
                (input, (hidden, good_hidden)),
                (input, (good_hidden, hidden)),
            ]

        rnn_modes = ['RNN', 'GRU', 'LSTM']
        for mode in rnn_modes:
            # Incorrect input batch size
            input_shape = update_shape(correct_input_shape, 1, bad_size)
            hidden_shape = correct_hidden_shape
            test(input_shape, hidden_shape, mode)

            # Incorrect hidden batch size
            input_shape = correct_input_shape
            hidden_shape = update_shape(correct_hidden_shape, 1, bad_size)
            test(input_shape, hidden_shape, mode)

            # Incorrect input size
            input_shape = update_shape(correct_input_shape, 2, bad_size)
            hidden_shape = correct_hidden_shape
            test(input_shape, hidden_shape, mode)

            # Incorrect hidden size
            input_shape = correct_input_shape
            hidden_shape = update_shape(correct_hidden_shape, 2, bad_size)
            test(input_shape, hidden_shape, mode)

            # Incorrect hidden[0]
            input_shape = correct_input_shape
            hidden_shape = update_shape(correct_hidden_shape, 0, bad_size)
            test(input_shape, hidden_shape, mode)

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_rnn_check_device(self):
        input_size = 3
        hidden_size = 5
        num_layers = 2
        batch_size = 4
        seq_len = 6
        num_directions = 1

        correct_input_shape = (seq_len, batch_size, input_size)
        correct_hidden_shape = (num_layers * num_directions, batch_size, hidden_size)
        rnn_modes = ['RNN', 'GRU', 'LSTM']

        for mode in rnn_modes:
            model = getattr(nn, mode)(input_size, hidden_size, num_layers)
            input = torch.randn(correct_input_shape)
            hidden = torch.randn(correct_hidden_shape)

            # input and weights are not at the same device
            with self.assertRaisesRegex(RuntimeError,
                                        "Input and parameter tensors are not at the same device"):
                model(input.to('cuda:0'))

            # input and hiddens are not at the same device
            with self.assertRaisesRegex(RuntimeError,
                                        r"Input and hidden tensors are not at the same device"):
                if mode == 'LSTM':
                    model(input, (hidden.to('cuda:0'), hidden.to('cuda:0')))
                else:
                    model(input, (hidden.to('cuda:0')))

            # hidden tensors are not at the same CUDA device
            if mode == 'LSTM':
                with self.assertRaisesRegex(RuntimeError,
                                            "Input and hidden tensors are not at the same device"):
                    model(input.to('cuda:0'), (hidden.to('cuda:0'), hidden.to('cuda:1')))

    def test_rnn_initial_hidden_state(self):
        rnn_modes = ['RNN', 'GRU', 'LSTM']
        for mode in rnn_modes:
            rnn = getattr(nn, mode)(30, 20, 2)
            input = torch.randn(10, 32, 30)
            hidden = torch.zeros(2, 32, 20)

            if mode == 'LSTM':
                hidden = (hidden, hidden)
            output1, hidden1 = rnn(input, hidden)
            output2, hidden2 = rnn(input)
            self.assertEqual(output1, output2)
            self.assertEqual(hidden1, hidden2)

    def _test_RNN_cpu_vs_cudnn(self, dropout, dtype=torch.double):

        def forward_backward(cuda, rnn, input_val, hx_val, grad_output, grad_hy, weights_val):
            is_lstm = isinstance(rnn, nn.LSTM)

            for x_layer, y_layer in zip(rnn.all_weights, weights_val):
                for x, y in zip(x_layer, y_layer):
                    x.data.copy_(y.data)

            if isinstance(input_val, rnn_utils.PackedSequence):
                input = rnn_utils.PackedSequence(
                    input_val.data.data.requires_grad_(True), input_val.batch_sizes)
                input_var = input.data
            else:
                input = input_val.clone().requires_grad_(True)
                input_var = input
            if is_lstm:
                hx = (hx_val.clone().requires_grad_(True),
                      hx_val.add(1).requires_grad_(True))
            else:
                hx = hx_val.clone().requires_grad_(True)

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
            for bias, bidirectional, batch_first, contig, variable_len, lens_as_tensor \
                    in product((True, False), repeat=6):

                num_directions = 2 if bidirectional else 1
                if batch_first:
                    input_val = torch.randn(batch, seq_length, input_size, dtype=dtype)
                    grad_output = torch.randn(batch, seq_length, hidden_size * num_directions, dtype=dtype)
                else:
                    input_val = torch.randn(seq_length, batch, input_size, dtype=dtype)
                    grad_output = torch.randn(seq_length, batch, hidden_size * num_directions, dtype=dtype)

                if not contig:
                    grad_output = make_noncontig(grad_output)
                    grad_hy = make_noncontig(grad_hy)
                    input_var = make_noncontig(input_val)
                    hx_val = make_noncontig(hx_val)

                hx_val = torch.randn(num_layers * num_directions, batch, hidden_size, dtype=dtype)
                grad_hy = torch.randn(num_layers * num_directions, batch, hidden_size, dtype=dtype)

                if variable_len:
                    lengths = [7, 5, 5, 2, 1, 1]
                    if lens_as_tensor:
                        lengths = torch.tensor(lengths, dtype=torch.long)
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
            hx_val = torch.randn(num_layers, batch, hidden_size, dtype=dtype)
            input_val = torch.randn(seq_length, batch, input_size, dtype=dtype)
            grad_output = torch.randn(
                seq_length, batch, hidden_size * num_directions, dtype=dtype)
            grad_hy = torch.randn(
                num_layers * num_directions, batch, hidden_size, dtype=dtype)

            rnn = nn.RNN(input_size, hidden_size, num_layers, bias=bias, nonlinearity=nonlinearity)
            outputs_cpu = forward_backward(False, rnn, input_val, hx_val, grad_output, grad_hy, rnn.all_weights)

            rnn_gpu = nn.RNN(input_size, hidden_size, num_layers, bias=bias, nonlinearity=nonlinearity)
            outputs_gpu = forward_backward(True, rnn_gpu, input_val, hx_val, grad_output, grad_hy, rnn.all_weights)

            compare_cpu_gpu(outputs_cpu, outputs_gpu)

    @unittest.skipIf(not TEST_CUDNN, "needs cudnn")
    def test_RNN_cpu_vs_cudnn_no_dropout(self):
        self._test_RNN_cpu_vs_cudnn(0)

    @unittest.skipIf(not (TEST_CUDNN and (TEST_CUDNN_VERSION if TEST_CUDNN_VERSION else 0) >= 5103), "needs cudnn >= 5.1")
    def test_RNN_cpu_vs_cudnn_with_dropout(self):
        # Because of dropout randomness, can only compare dropout=0 and dropout=1
        self._test_RNN_cpu_vs_cudnn(1)

    @unittest.skipIf(not TEST_CUDNN, "needs cudnn")
    def test_RNN_cudnn_weight_norm(self):
        input_size = 10
        hidden_size = 6
        num_layers = 2
        seq_length = 7
        batch = 6
        m = nn.LSTM(input_size, hidden_size, num_layers).cuda()
        input = torch.randn(seq_length, batch, input_size).cuda()
        expected_output = m(input)
        # add weight normalization
        name = 'weight_hh_l0'
        m = torch.nn.utils.weight_norm(m, name=name)
        # otherwise, subsequent warnings will be hidden, and further tests rely on them
        warnings.simplefilter("always")
        self.assertEqual(m(input), expected_output)

        # remove weight norm
        m = torch.nn.utils.remove_weight_norm(m, name=name)
        self.assertEqual(m(input), expected_output)

    @unittest.skipIf(not (TEST_CUDNN and (TEST_CUDNN_VERSION if TEST_CUDNN_VERSION else 0) >= 5103), "needs cudnn >= 5.1")
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
                    input = torch.ones(1, 1, 10)
                    hx = torch.zeros(2, 1, 1000)
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

    @unittest.skipIf(not (TEST_CUDNN and (TEST_CUDNN_VERSION if TEST_CUDNN_VERSION else 0) >= 5103), "needs cudnn >= 5.1")
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
                    input = torch.rand(1, 1, 100)
                    hx = torch.rand(2, 1, 100)
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

    @unittest.skipIf(not (TEST_CUDNN and (TEST_CUDNN_VERSION if TEST_CUDNN_VERSION else 0) >= 5103), "needs cudnn >= 5.1")
    def test_RNN_change_dropout(self):
        for train, cuda in product((True, False), repeat=2):
            rnn = nn.RNN(100, 100, 2, dropout=0, nonlinearity='relu')
            input = torch.rand(3, 2, 100)
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
        modules = [nn.ReLU, nn.ELU, nn.SELU, nn.CELU, nn.RReLU]
        for mod in modules:
            r = mod(inplace=True)
            input = torch.randn(5, 5, requires_grad=True)
            output = r(input + 0)
            grad_output = torch.randn(5, 5)
            grad_output_clone = grad_output.clone()
            output.backward(grad_output)
            self.assertEqual(grad_output, grad_output_clone)

    @unittest.skipIf(not TEST_CUDA, 'CUDA not available')
    @repeat_test_for_types(ALL_TENSORTYPES2)
    def test_noncontig_conv_grad_cuda(self, dtype=torch.float):
        # FIXME: remove after adding non-contiguous grad tests for all modules
        module = nn.Conv2d(3, 5, kernel_size=3, padding=1).to("cuda", dtype)
        input = torch.randn(2, 3, 10, 10, dtype=dtype, device="cuda", requires_grad=True)
        output = module(input)

        grad = torch.randn(2, 2, 5, 10, 10, dtype=dtype, device="cuda")[:, 1]
        assert not grad.is_contiguous()
        output.backward(grad, retain_graph=True)
        self.assertIsNotNone(input.grad)
        result = input.grad.data.clone()
        input.grad.data.zero_()

        output.backward(grad.contiguous())
        self.assertEqual(result, input.grad.data, dtype2prec[dtype])

    def test_pixel_shuffle(self):
        batch_size = random.randint(1, 3)
        upscale_factor = random.randint(2, 5)
        channels = random.randint(1, 4) * upscale_factor ** 2
        height = random.randint(5, 10)
        width = random.randint(5, 10)

        input = torch.rand(batch_size, channels, height, width, requires_grad=True)
        ps = nn.PixelShuffle(upscale_factor)
        output = ps(input)
        self._verify_pixel_shuffle(input.data, output.data, upscale_factor)
        output.backward(output.data)
        self.assertEqual(input.data, input.grad.data)

    def test_elu_inplace_view(self):
        v = torch.tensor([1.0, -1.0, 1.0, -1.0], requires_grad=True)

        def func(root):
            x = root.clone()
            view = x.narrow(0, 1, 2)
            res = F.elu(view, inplace=True)
            self.assertIs(res, view)
            return x

        gradcheck(func, [v])
        gradgradcheck(func, [v])

    def test_relu_inplace_view(self):
        v = torch.tensor([1.0, -1.0, 1.0, -1.0], requires_grad=True)

        def func(root):
            x = root.clone()
            view = x.narrow(0, 1, 2)
            res = F.relu(view, inplace=True)
            self.assertIs(res, view)
            return x

        gradcheck(func, [v])
        gradgradcheck(func, [v])

    @unittest.skipIf(not TEST_CUDA, 'CUDA not available')
    def test_PReLU_backward_requires_grad_false(self):
        m = nn.PReLU().to('cuda')
        x = torch.randn(2, 3, 4, 5, requires_grad=False, device='cuda')
        y = m(x)
        y.mean().backward()
        self.assertEqual(x.grad, None)

    @unittest.skipIf(
        not TEST_NUMPY or not TEST_SCIPY, "Numpy or Scipy not found")
    def test_gelu(self):
        def _test_gelu(n, m, dtype, contiguous):
            def _gelu_ref(X):
                return X * stats.norm.cdf(X)

            if contiguous:
                X = torch.rand(n, m, dtype=dtype, requires_grad=True)
            else:
                X = torch.rand(n, m, dtype=dtype, requires_grad=True)[:, ::2]
            res = F.gelu(X)
            ref = _gelu_ref(X.detach().numpy())
            self.assertEqual(res, ref)
            gradcheck(F.gelu, [X], eps=1e-4)

            if TEST_CUDA:
                X_cuda = X.cuda()
                res_cuda = F.gelu(X_cuda)
                self.assertEqual(res_cuda.cpu(), ref)
                gradcheck(F.gelu, [X_cuda], eps=1e-4)

        for n in range(1, 10):
            for m in range(1, 10):
                _test_gelu(n, m, torch.float32, True)
                _test_gelu(n, m, torch.float32, False)
                _test_gelu(n, m, torch.float64, True)
                _test_gelu(n, m, torch.float64, False)


    def test_bce_loss_always_nonnegative(self):
        target = torch.ones(5)
        input = torch.ones(5)
        self.assertEqual((nn.BCELoss()(input, target) < 0).sum(), 0)

        target = torch.zeros(5)
        input = torch.zeros(5)
        self.assertEqual((nn.BCELoss()(input, target) < 0).sum(), 0)

    def test_bce_with_logits_raises_if_target_and_input_are_different_size(self):
        target = torch.rand(5)
        input = torch.rand(5, 1)
        with self.assertRaises(ValueError):
            nn.BCEWithLogitsLoss()(input, target)

        target = torch.rand(5, 1)
        input = torch.rand(5)
        with self.assertRaises(ValueError):
            nn.BCEWithLogitsLoss()(input, target)

    def test_bce_with_logits_gives_same_result_as_sigmoid_and_bce_loss(self):
        sigmoid = nn.Sigmoid()

        target = torch.rand(64, 4)
        output = torch.rand(64, 4) - 0.5

        self.assertEqual(nn.BCEWithLogitsLoss()(output, target), nn.BCELoss()(sigmoid(output), target))

        weight = torch.rand(4)
        self.assertEqual(nn.BCEWithLogitsLoss(weight)(output, target), nn.BCELoss(weight)(sigmoid(output), target))

        target = torch.zeros(4, 1, dtype=torch.float)
        output = torch.empty(4, 1, dtype=torch.float).fill_(-100)

        self.assertEqual(nn.BCEWithLogitsLoss()(output, target), nn.BCELoss()(sigmoid(output), target))

        self.assertEqual(nn.BCEWithLogitsLoss(reduction='none')(output, target),
                         nn.BCELoss(reduction='none')(sigmoid(output), target))

        weight = torch.rand(1, dtype=torch.float)
        self.assertEqual(nn.BCEWithLogitsLoss(weight)(output, target), nn.BCELoss(weight)(sigmoid(output), target))

    def test_bce_with_logits_has_correct_grad_at_zero(self):
        output = torch.zeros(3, 1, requires_grad=True)
        target = torch.zeros(3, 1)
        nn.BCEWithLogitsLoss(reduction='sum')(output, target).backward()
        expected_grad = torch.empty(3, 1).fill_(0.5)
        self.assertEqual(output.grad, expected_grad)

    def test_bce_with_logits_broadcasts_weights(self):
        target = torch.rand(16, 4)
        output = torch.rand(16, 4) - 0.5

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

    def test_bce_with_logits_ones_in_pos_weights_are_the_same_as_none(self):
        target = torch.rand(64, 4)
        output = torch.rand(64, 4) - 0.5
        pos_weight = torch.ones(64, 4)

        self.assertEqual(nn.BCEWithLogitsLoss()(output, target),
                         nn.BCEWithLogitsLoss(pos_weight=pos_weight)(output, target))

    def test_bce_with_logits_broadcasts_pos_weights(self):
        target = torch.rand(64, 4)
        output = torch.rand(64, 4) - 0.5
        pos_weight = torch.rand(4)
        out1 = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(output, target)

        pos_weight1 = pos_weight.expand(1, 4)
        out2 = nn.BCEWithLogitsLoss(pos_weight=pos_weight1)(output, target)

        pos_weight2 = pos_weight.expand(64, 4)
        out3 = nn.BCEWithLogitsLoss(pos_weight=pos_weight2)(output, target)

        self.assertEqual(out1, out2)
        self.assertEqual(out1, out3)

    def test_bce_with_logits_with_pos_weight_has_correct_grad_at_zero(self):
        output = torch.zeros(3, 1, requires_grad=True)
        target = torch.zeros(3, 1)
        pos_weight = torch.ones(3, 1)
        nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='sum')(output, target).backward()
        expected_grad = torch.empty(3, 1).fill_(0.5)
        grad = output.grad
        self.assertEqual(grad, expected_grad)

    def test_bce_with_logits_stability(self):
        output = torch.tensor([0., -120.])
        target = torch.tensor([0., 1.])
        pos_weight = torch.tensor([1., 1.])

        out1 = nn.BCEWithLogitsLoss()(output, target)
        self.assertTrue(torch.isfinite(out1).all().item())

        out2 = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(output, target)
        self.assertTrue(torch.isfinite(out2).all().item())

    def test_bce_loss_broadcasts_weights(self):
        sigmoid = nn.Sigmoid()
        target = torch.rand(16, 4)
        output = torch.rand(16, 4) - 0.5

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
        v = torch.randn(8, requires_grad=True)

        def func(root):
            x = root.clone()
            return F.elu(x, inplace=True)

        gradcheck(func, [v])
        gradgradcheck(func, [v])

    def test_hardtanh_inplace_gradgrad(self):
        v = torch.randn(8, requires_grad=True)

        def func(root):
            x = root.clone()
            return F.hardtanh(x, inplace=True)

        gradcheck(func, [v])
        gradgradcheck(func, [v])

    # test hardtanh backward froo large tensor
    def test_hardtanh_backward(self):
        x = torch.randn(128, 10000, requires_grad=True)
        grad = torch.randn(128, 10000)
        z = torch.zeros(128, 10000)
        y = F.hardtanh(x)
        y.backward(grad)
        # ref backward path for hardtanh
        mask = (x > -1) & (x < 1)
        x_grad_ref = torch.where(mask, grad, z)
        self.assertEqual(x.grad, x_grad_ref)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    @unittest.skipIf(not TEST_CUDNN, "needs cudnn")
    @skipIfRocm
    def test_batchnorm_cudnn_nhwc(self):
        input = torch.randint(1, 10, (4, 8, 2, 2), dtype=torch.float32, device="cuda", requires_grad=True)
        input = input.contiguous(memory_format=torch.channels_last)
        input.retain_grad()
        grad = torch.randint(1, 10, (4, 8, 2, 2), dtype=torch.float32, device="cuda")
        grad = grad.contiguous(memory_format=torch.channels_last)
        bn = nn.BatchNorm2d(8).cuda().float()
        bn.weight.data.uniform_()
        bn.bias.data.uniform_()

        ref_input = input.detach().clone().contiguous().requires_grad_(True)
        ref_grad = grad.detach().clone().contiguous()
        ref_bn = nn.BatchNorm2d(8).cuda().float()
        ref_bn.load_state_dict(bn.state_dict())

        out = bn(input)
        out.backward(grad)
        ref_out = ref_bn(ref_input)
        ref_out.backward(ref_grad)

        self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))
        self.assertTrue(ref_out.is_contiguous())
        self.assertEqual(out, ref_out)
        self.assertEqual(bn.weight.grad, ref_bn.weight.grad)
        self.assertEqual(bn.bias.grad, ref_bn.bias.grad)
        self.assertEqual(input.grad, ref_input.grad)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_batchnorm_cudnn_half(self):
        # THNN
        input = torch.randint(1, 10, (2, 3, 2, 2), dtype=torch.half, device="cuda", requires_grad=True)
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

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_batchnorm_nonaffine_cuda_half_input(self):
        input = torch.randn(16, 3, 24, 24, dtype=torch.half, device="cuda")
        m = nn.BatchNorm2d(3, affine=False).cuda().float()  # keep running stats in FP32
        output = m(input)
        self.assertEqual(output.type(), input.type())
        m.eval()
        output = m(input)
        self.assertEqual(output.type(), input.type())

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    @repeat_test_for_types([torch.float, torch.half])
    def test_batchnorm_large_batch(self, dtype=torch.float):
        bn = nn.BatchNorm1d(1).to('cuda', dtype)
        data = torch.rand(131072, 1, device="cuda", dtype=dtype)
        out = bn(data).sum().backward()

    def test_batchnorm_raises_error_if_running_mean_is_not_same_size_as_input(self):
        input = torch.rand(2, 10)
        running_var = torch.rand(10)
        wrong_sizes = [9, 11]
        for size in wrong_sizes:
            with self.assertRaises(RuntimeError):
                F.batch_norm(input, torch.rand(size), running_var)

    def test_batchnorm_raises_error_if_running_var_is_not_same_size_as_input(self):
        input = torch.rand(2, 10)
        running_mean = torch.rand(10)
        wrong_sizes = [9, 11]
        for size in wrong_sizes:
            with self.assertRaises(RuntimeError):
                F.batch_norm(input, running_mean, torch.rand(size))

    def test_batchnorm_raises_error_if_weight_is_not_same_size_as_input(self):
        input = torch.rand(2, 10)
        running_mean = torch.rand(10)
        running_var = torch.rand(10)
        wrong_sizes = [9, 11]
        for size in wrong_sizes:
            with self.assertRaises(RuntimeError):
                F.batch_norm(input, running_mean, running_var, weight=Parameter(torch.rand(size)))

    def test_batchnorm_raises_error_if_bias_is_not_same_size_as_input(self):
        input = torch.rand(2, 10)
        running_mean = torch.rand(10)
        running_var = torch.rand(10)
        wrong_sizes = [9, 11]
        for size in wrong_sizes:
            with self.assertRaises(RuntimeError):
                F.batch_norm(input, running_mean, running_var, bias=Parameter(torch.rand(size)))

    def test_pairwise_distance(self):
        input1 = torch.randn(4, 4, requires_grad=True)
        input2 = torch.randn(4, 4, requires_grad=True)
        self.assertTrue(gradcheck(lambda x, y: F.pairwise_distance(x, y), (input1, input2)))

    @skipIfRocm
    def test_pdist(self):
        for device, trans in itertools.product(device_(), [False, True]):
            inp = torch.randn(4, 5, dtype=torch.double, device=device, requires_grad=True)
            if trans:
                inp = inp.transpose(0, 1)
            for p in [0, 1, 2, 0.5, 1.5, 2.5, float('inf')]:
                self.assertTrue(gradcheck(lambda x: F.pdist(x, p), (inp,)))

    def test_pdist_zeros(self):
        """Test that grad is still valid when dist is 0"""
        for device in device_():
            inp = torch.randn(1, 3, dtype=torch.double, device=device, requires_grad=True).repeat([2, 1])
            for p in [0, 1, 2, 0.5, 1.5, 2.5, float('inf')]:
                self.assertTrue(gradcheck(lambda x: F.pdist(x, p), (inp,)))

    def test_pdist_empty_row(self):
        for device in device_():
            inp = torch.randn(1, 3, dtype=torch.double, device=device, requires_grad=True)
            self.assertTrue(gradcheck(F.pdist, (inp,)))

    def test_pdist_empty_col(self):
        for device in device_():
            inp = torch.randn(4, 0, dtype=torch.double, device=device, requires_grad=True)
            self.assertTrue(gradcheck(F.pdist, (inp,)))

    @unittest.expectedFailure
    def test_pdist_cpu_gradgrad_unimplemented(self):
        inp = torch.randn(4, 5, requires_grad=True)
        gradgradcheck(F.pdist, (inp,))

    @skipIfRocm
    @unittest.expectedFailure
    def test_pdist_cuda_gradgrad_unimplemented(self):
        inp = torch.randn(4, 5, device='cuda', requires_grad=True)
        gradgradcheck(F.pdist, (inp,))

    def test_cosine_embedding_loss_with_diff_type(self):
        for device in device_():
            input1 = torch.tensor([[2, 3, 4], [6, 2, 4]], dtype=torch.double, device=device)
            input2 = torch.tensor([[2, 3, 5], [3, 2, 1]], dtype=torch.double, device=device)
            target = torch.tensor([1, -1], dtype=torch.int, device=device)
            expected = torch.nn.functional.cosine_embedding_loss(input1, input2, target)
            for dt1 in torch.testing.get_all_math_dtypes(device):
                for dt2 in torch.testing.get_all_math_dtypes(device):
                    for dt3 in torch.testing.get_all_math_dtypes(device):
                        # dt3 is used as dtype for target = [1, -1], so let's skip unsigned type
                        if dt3 == torch.uint8:
                            continue
                        input1 = input1.to(dt1)
                        input2 = input2.to(dt2)
                        target = target.to(dt3)
                        result = torch.nn.functional.cosine_embedding_loss(input1, input2, target)
                        self.assertEqual(result.item(), expected.item(), 0.001)

    def test_kl_div_with_diff_type(self):
        for device in device_():
            input = torch.tensor([[2, 3, 5], [3, 2, 1]], dtype=torch.double, device=device)
            target = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.double, device=device)
            expected = torch.nn.functional.kl_div(input, target)
            for input_dtype in torch.testing.get_all_math_dtypes(device):
                for target_dtype in [torch.float32, torch.float64, torch.float16]:
                    if (torch.device(device).type == 'cpu' and target_dtype == torch.float16):
                        continue
                    input = input.to(input_dtype)
                    target = target.to(target_dtype)
                    result = torch.nn.functional.kl_div(input, target)
                    self.assertEqual(result.item(), expected.item(), 0.001)

    def test_cosine_embedding_loss_no_reduce(self):
        input1 = torch.randn(15, 10, requires_grad=True)
        input2 = torch.randn(15, 10, requires_grad=True)
        target = torch.randn(15).sign()
        self.assertTrue(gradcheck(lambda x, y, z: F.cosine_embedding_loss(
            x, y, z, reduction='none'), (input1, input2, target)))
        self.assertEqual(F.cosine_embedding_loss(input1, input2, target, reduction='none'),
                         loss_reference_fns['CosineEmbeddingLoss'](input1, input2, target, reduction='none'))

    def test_cosine_embedding_loss_margin_no_reduce(self):
        input1 = torch.randn(15, 10, requires_grad=True)
        input2 = torch.randn(15, 10, requires_grad=True)
        target = torch.randn(15).sign()
        self.assertTrue(gradcheck(lambda x, y, z: F.cosine_embedding_loss(
            x, y, z, margin=0.5, reduction='none'), (input1, input2, target)))
        self.assertEqual(F.cosine_embedding_loss(input1, input2, target, margin=0.5, reduction='none'),
                         loss_reference_fns['CosineEmbeddingLoss'](input1, input2, target,
                                                                   margin=0.5, reduction='none'))

    def test_margin_ranking_loss_no_reduce(self):
        input1 = torch.randn(15).mul_(10).requires_grad_()
        input2 = torch.randn(15).mul_(10).requires_grad_()
        target = torch.randn(15).sign()
        self.assertTrue(gradcheck(lambda x, y, z: F.margin_ranking_loss(
            x, y, z, reduction='none'), (input1, input2, target)))
        self.assertEqual(F.margin_ranking_loss(input1, input2, target, reduction='none'),
                         loss_reference_fns['MarginRankingLoss'](input1, input2, target, reduction='none'))

    def test_margin_ranking_loss_margin_no_reduce(self):
        input1 = torch.randn(15).mul_(10).requires_grad_()
        input2 = torch.randn(15).mul_(10).requires_grad_()
        target = torch.randn(15).sign()
        self.assertTrue(gradcheck(lambda x, y, z: F.margin_ranking_loss(
            x, y, z, margin=0.5, reduction='none'), (input1, input2, target)))
        self.assertEqual(F.margin_ranking_loss(input1, input2, target, margin=0.5, reduction='none'),
                         loss_reference_fns['MarginRankingLoss'](input1, input2, target, margin=0.5, reduction='none'))

    def test_triplet_margin_loss(self):
        input1 = torch.randn(5, 10, requires_grad=True)
        input2 = torch.randn(5, 10, requires_grad=True)
        input3 = torch.randn(5, 10, requires_grad=True)
        self.assertTrue(gradcheck(lambda x1, x2, x3: F.triplet_margin_loss(
            x1, x2, x3), (input1, input2, input3)))
        self.assertEqual(F.triplet_margin_loss(input1, input2, input3),
                         loss_reference_fns['TripletMarginLoss'](input1, input2, input3))

    def test_triplet_margin_loss_swap(self):
        input1 = torch.randn(5, 10, requires_grad=True)
        input2 = torch.randn(5, 10, requires_grad=True)
        input3 = torch.randn(5, 10, requires_grad=True)
        self.assertTrue(gradcheck(lambda x1, x2, x3: F.triplet_margin_loss(
            x1, x2, x3, swap=True), (input1, input2, input3)))
        self.assertEqual(F.triplet_margin_loss(input1, input2, input3, swap=True),
                         loss_reference_fns['TripletMarginLoss'](input1, input2, input3, swap=True))

    def test_triplet_margin_loss_no_reduce(self):
        input1 = torch.randn(5, 10, requires_grad=True)
        input2 = torch.randn(5, 10, requires_grad=True)
        input3 = torch.randn(5, 10, requires_grad=True)
        self.assertTrue(gradcheck(lambda x1, x2, x3: F.triplet_margin_loss(
            x1, x2, x3, reduction='none'), (input1, input2, input3)))
        self.assertEqual(F.triplet_margin_loss(input1, input2, input3, reduction='none'),
                         loss_reference_fns['TripletMarginLoss'](input1, input2, input3, reduction='none'))

    def test_triplet_margin_loss_swap_no_reduce(self):
        input1 = torch.randn(5, 10, requires_grad=True)
        input2 = torch.randn(5, 10, requires_grad=True)
        input3 = torch.randn(5, 10, requires_grad=True)
        self.assertTrue(gradcheck(lambda x1, x2, x3: F.triplet_margin_loss(
            x1, x2, x3, swap=True, reduction='none'), (input1, input2, input3)))
        self.assertEqual(F.triplet_margin_loss(input1, input2, input3, swap=True, reduction='none'),
                         loss_reference_fns['TripletMarginLoss'](input1, input2, input3, swap=True, reduction='none'))

    def test_pointwise_loss_target_grad_none_reduction(self):
        i = torch.randn(5, 10)
        t = torch.randn(5, 10, requires_grad=True)
        self.assertEqual(F.mse_loss(i, t, reduction='none').size(), t.size())
        self.assertEqual(F.l1_loss(i, t, reduction='none').size(), t.size())

    def test_pointwise_loss_broadcast(self):
        losses = {
            'mse_loss': lambda x, y, r: F.mse_loss(x, y, reduction=r),
            'l1_loss': lambda x, y, r: F.l1_loss(x, y, reduction=r),
            'smooth_l1_loss': lambda x, y, r: F.smooth_l1_loss(x, y, reduction=r),
        }

        input = torch.randn(2, 1, requires_grad=True)
        for _name, fn in losses.items():
            for requires_grad in [True, False]:
                # When target.requires_grad=True, its impl is in Python, while the other is in TH.
                target = torch.randn(2, 10, requires_grad=requires_grad)
                for reduction in ['none', 'mean', 'sum']:
                    l = fn(input, target, reduction)
                    if reduction == 'none':
                        self.assertEqual(l.size(), target.size())
                    self.assertTrue(gradcheck(fn, (input, target, reduction)))

    # https://github.com/pytorch/pytorch/issues/27692 reports
    # that l1_loss get a wrong result for big batch size
    def test_l1_loss_correct(self):
        for N in range(1, 50, 10):
            input = torch.rand(N, 3, 1024, 1024)
            self.assertEqual(
                torch.nn.L1Loss()(input, torch.zeros_like(input)),
                input.abs().mean())

    def test_cosine_similarity(self):
        input1 = torch.randn(4, 4, requires_grad=True)
        input2 = torch.randn(4, 4, requires_grad=True)
        self.assertTrue(gradcheck(lambda x, y: F.cosine_similarity(x, y), (input1, input2)))

        input1 = torch.randn(4, 5, 6, requires_grad=True)
        input2 = torch.randn(4, 5, 6, requires_grad=True)
        self.assertTrue(gradcheck(lambda x, y: F.cosine_similarity(x, y, dim=0), (input1, input2)))
        self.assertTrue(gradcheck(lambda x, y: F.cosine_similarity(x, y, dim=-1), (input1, input2)))

        input1 = torch.randn((), requires_grad=True)
        input2 = torch.randn((), requires_grad=True)
        self.assertTrue(gradcheck(lambda x, y: F.cosine_similarity(x, y, dim=0), (input1, input2)))
        self.assertTrue(gradcheck(lambda x, y: F.cosine_similarity(x, y, dim=-1), (input1, input2)))

        # Check cosine_similarity input/output shapes
        input_size = (1, 3, 2, 1)
        expected_size = (1, 2, 1)
        input1 = torch.randn(input_size, requires_grad=True)
        input2 = torch.randn(input_size, requires_grad=True)
        self.assertEqual(F.cosine_similarity(input1, input2, dim=1).size(), expected_size)

        # Check numerical precision, issue #18057
        vv1 = torch.tensor(list([float(i) for i in range(84)])).unsqueeze(0)
        vv2 = torch.tensor(list([float(i) for i in range(84)])).unsqueeze(0)
        out = F.cosine_similarity(vv1, vv2)
        self.assertLessEqual(out, 1.0)

        # Check dividing by 0.
        input1 = torch.randn(10).requires_grad_()
        input2 = torch.zeros_like(input1).requires_grad_()
        torch.cosine_similarity(input1, input2, 0).sum().backward()
        self.assertEqual(input1.grad, torch.zeros_like(input1))
        self.assertEqual(input2.grad, input1 * 1e8)

    def test_grid_sample_error_checking(self):
        input = torch.empty(1, 1, 2, 2)
        grid = torch.empty(1, 1, 1, 2)

        # assert no error
        F.grid_sample(input, grid, align_corners=False)

        with self.assertRaisesRegex(ValueError, "but got: 'garbage'"):
            F.grid_sample(input, grid, mode='garbage', align_corners=False)

        with self.assertRaisesRegex(ValueError, "but got: 'garbage'"):
            F.grid_sample(input, grid, padding_mode='garbage', align_corners=False)

        with self.assertRaisesRegex(RuntimeError, "expected input and grid to have same dtype"):
            F.grid_sample(input.float(), grid.double(), align_corners=False)

        with self.assertRaisesRegex(RuntimeError, "expected 4D or 5D input"):
            F.grid_sample(input[0], grid, align_corners=False)

        with self.assertRaisesRegex(RuntimeError, "grid with same number of dimensions"):
            F.grid_sample(input, torch.empty(1, 1, 1, 1, 3), align_corners=False)

        with self.assertRaisesRegex(RuntimeError, "expected grid and input to have same batch size"):
            F.grid_sample(input, torch.empty(2, 1, 1, 2), align_corners=False)

        with self.assertRaisesRegex(RuntimeError, "expected grid to have size 2 in last dimension"):
            F.grid_sample(input, torch.empty(1, 1, 1, 3), align_corners=False)

        with self.assertRaisesRegex(RuntimeError, "expected input to have non-empty spatial dimensions"):
            F.grid_sample(torch.empty(1, 1, 0, 2), grid, align_corners=False)

        if TEST_CUDA:
            with self.assertRaisesRegex(RuntimeError, "expected input and grid to be on same device"):
                F.grid_sample(input.cuda(), grid, align_corners=False)

    def test_affine_grid_error_checking(self):
        # 2D affine
        theta = torch.empty(1, 2, 3, dtype=torch.double)
        size = torch.Size([1, 1, 2, 2])

        # assert no error
        F.affine_grid(theta, size, align_corners=False)

        # check for warning for empty span along dimension
        with warnings.catch_warnings(record=True) as w:
            # Ensure warnings are being shown
            warnings.simplefilter("always")
            # Should not trigger warning
            F.affine_grid(theta, torch.Size([1, 1, 2, 1]), align_corners=False)
            # Check no warning occurs
            self.assertNotIn('See the documentation of affine_grid for details.', ' '.join(map(str, w)))
            # Should trigger warning
            F.affine_grid(theta, torch.Size([1, 1, 2, 1]), align_corners=True)
            # Check warning occurs
            self.assertIn('See the documentation of affine_grid for details.', ' '.join(map(str, w)))

        with self.assertRaisesRegex(ValueError, "Expected theta to have floating point type"):
            F.affine_grid(theta.int(), size, align_corners=False)

        with self.assertRaisesRegex(ValueError, "Expected a batch of 2D affine matrices of shape Nx2x3"):
            F.affine_grid(theta[0], size, align_corners=False)

        with self.assertRaisesRegex(ValueError, "Expected a batch of 2D affine matrices of shape Nx2x3"):
            F.affine_grid(theta.unsqueeze(0), size, align_corners=False)

        with self.assertRaisesRegex(ValueError, "Expected a batch of 2D affine matrices of shape Nx2x3"):
            F.affine_grid(theta.repeat(1, 2, 1), size, align_corners=False)

        with self.assertRaisesRegex(ValueError, "Expected a batch of 2D affine matrices of shape Nx2x3"):
            F.affine_grid(theta.repeat(1, 1, 2), size, align_corners=False)

        # 3D affine
        theta = torch.empty(1, 3, 4, dtype=torch.double)
        size = torch.Size([1, 1, 2, 2, 2])

        # assert no error
        F.affine_grid(theta, size, align_corners=False)

        # check for warning for empty span along dimension
        with warnings.catch_warnings(record=True) as w:
            # Ensure warnings are being shown
            warnings.simplefilter("always")
            # Should not trigger warning
            F.affine_grid(theta, torch.Size([1, 1, 3, 2, 1]), align_corners=False)
            # Check no warning occurs
            self.assertNotIn('See the documentation of affine_grid for details.', ' '.join(map(str, w)))
            # Should trigger warning
            F.affine_grid(theta, torch.Size([1, 1, 3, 2, 1]), align_corners=True)
            # Check warning occurs
            self.assertIn('See the documentation of affine_grid for details.', ' '.join(map(str, w)))

        with self.assertRaisesRegex(ValueError, "Expected a batch of 3D affine matrices of shape Nx3x4"):
            F.affine_grid(theta[0], size, align_corners=False)

        with self.assertRaisesRegex(ValueError, "Expected a batch of 3D affine matrices of shape Nx3x4"):
            F.affine_grid(theta.unsqueeze(0), size, align_corners=False)

        with self.assertRaisesRegex(ValueError, "Expected a batch of 3D affine matrices of shape Nx3x4"):
            F.affine_grid(theta.repeat(1, 2, 1), size, align_corners=False)

        with self.assertRaisesRegex(ValueError, "Expected a batch of 3D affine matrices of shape Nx3x4"):
            F.affine_grid(theta.repeat(1, 1, 2), size, align_corners=False)

        with self.assertRaisesRegex(NotImplementedError, "affine_grid only supports 4D and 5D sizes"):
            F.affine_grid(theta, torch.Size([1, 2, 2]), align_corners=False)

        with self.assertRaisesRegex(NotImplementedError, "affine_grid only supports 4D and 5D sizes"):
            F.affine_grid(theta, torch.Size([1, 1, 2, 2, 2, 2]), align_corners=False)

    def test_grid_sample(self):
        def test(N, C, H, W, mode, padding_mode, align_corners):
            def test_shape(N, C, IH, IW, H, W, mode, padding_mode, align_corners):
                for grid_dim_contig_order in [(0, 1, 2, 3), (0, 3, 1, 2), (3, 0, 1, 2), (0, 2, 1, 3)]:
                    # grid_dim_contig_order specifies the dimension order that can
                    # make grid to be contiguous.
                    # i.e., grid.permute(grid_dim_contig_order) is contiguous.
                    # e.g., with grid_dim_contig_order=[0, 3, 1, 2], grid should be
                    #       initialized with contiguous tensor of shape [N, 2, H, W]
                    #       and permuted to [N, H, W, 2] afterwards.
                    grid_shape = [N, H, W, 2]
                    grid_init_shape = [grid_shape[d] for d in grid_dim_contig_order]
                    grid_fwd_permute = [None, None, None, None]
                    for i, d in enumerate(grid_dim_contig_order):
                        grid_fwd_permute[d] = i

                    def get_grid(device='cpu', data=None):
                        if data is not None:
                            assert list(data.shape) == grid_shape
                            data = data.permute(grid_dim_contig_order).to(device)
                        else:
                            data = torch.randn(grid_init_shape, device=device)
                        grid = data.permute(grid_fwd_permute)
                        assert grid.permute(grid_dim_contig_order).is_contiguous()
                        return grid

                    input_cpu = torch.randn(C, N, IH, IW).transpose(0, 1).requires_grad_()
                    grid_cpu = get_grid().requires_grad_()
                    out_cpu = F.grid_sample(input_cpu, grid_cpu, mode=mode, padding_mode=padding_mode,
                                            align_corners=align_corners)
                    self.assertTrue(out_cpu.size() == torch.Size([N, C, H, W]))

                    gradients = torch.randn_like(out_cpu)
                    out_cpu.backward(gradients)

                    if TEST_CUDA:
                        input_cuda = input_cpu.detach().transpose(0, 1).cuda().transpose(0, 1).requires_grad_()
                        grid_cuda = get_grid('cuda', grid_cpu.detach()).requires_grad_()
                        out_cuda = F.grid_sample(input_cuda, grid_cuda, mode=mode, padding_mode=padding_mode,
                                                 align_corners=align_corners)
                        self.assertEqual(out_cpu, out_cuda)

                        out_cuda.backward(gradients.cuda())
                        self.assertEqual(input_cpu.grad, input_cuda.grad)
                        self.assertEqual(grid_cpu.grad, grid_cuda.grad, prec=5e-5)

                        # check that zero-dimensional input strides don't error out
                        base_input = torch.randn(N, C, 1, IW)
                        input_cpu = base_input.expand_as(input_cuda).requires_grad_()
                        out_cpu = F.grid_sample(input_cpu, grid_cpu, mode=mode, padding_mode=padding_mode,
                                                align_corners=align_corners)

                        input_cuda = base_input.cuda().expand_as(input_cuda).requires_grad_()
                        out_cuda = F.grid_sample(input_cuda, grid_cuda, mode=mode, padding_mode=padding_mode,
                                                 align_corners=align_corners)
                        self.assertEqual(out_cpu, out_cuda)

            # test same size output
            test_shape(N, C, H, W, H, W, mode, padding_mode, align_corners)

            # test larger output
            N = random.randint(2, 8)
            C = random.randint(2, 8)
            IH = random.randint(2, 8)
            IW = random.randint(2, 8)
            H = random.randint(IH + 1, 12)
            W = random.randint(IW + 1, 12)
            test_shape(N, C, IH, IW, H, W, mode, padding_mode, align_corners)

            # test smaller output
            N = random.randint(2, 8)
            C = random.randint(2, 8)
            IH = random.randint(2, 8)
            IW = random.randint(2, 8)
            H = random.randint(2, IH)
            W = random.randint(2, IW)
            test_shape(N, C, IH, IW, H, W, mode, padding_mode, align_corners)

            # test 1x1 inpput
            N = random.randint(2, 8)
            C = random.randint(2, 8)
            IH = 1
            IW = 1
            H = random.randint(2, 5)
            W = random.randint(2, 5)
            test_shape(N, C, IH, IW, H, W, mode, padding_mode, align_corners)

            # testing empty grid
            N = random.randint(2, 8)
            C = random.randint(2, 8)
            IH = random.randint(2, 8)
            IW = random.randint(2, 8)
            W = random.randint(3, IW + 2)
            test_shape(N, C, IH, IW, 0, W, mode, padding_mode, align_corners)

            # testing empty channel
            N = random.randint(2, 8)
            IH = random.randint(2, 8)
            IW = random.randint(2, 8)
            H = random.randint(3, IH + 2)
            W = random.randint(3, IW + 2)
            test_shape(N, 0, IH, IW, H, W, mode, padding_mode, align_corners)

            # testing empty batch
            C = random.randint(2, 8)
            IH = random.randint(2, 8)
            IW = random.randint(2, 8)
            H = random.randint(3, IH + 2)
            W = random.randint(3, IW + 2)
            test_shape(0, C, IH, IW, H, W, mode, padding_mode, align_corners)

        for mode in ('bilinear', 'nearest'):
            for padding_mode in ('zeros', 'border', 'reflection'):
                for align_corners in (True, False):
                    # test known input on CPU
                    input = torch.arange(1., 11).view(1, 1, 2, 5)
                    grid = torch.tensor(
                        [[[-0.9, -4.1], [0, 0.2000], [1, -1], [-0.333, 1e-10], [0.5, 1.0]],
                         [[-1.0, -0.5], [0, 0.3333], [1, -1], [-0.200, 1e-10], [1.5, 0.5]]]).view(1, 2, 5, 2)
                    if mode == 'bilinear':
                        if padding_mode == 'zeros':
                            if align_corners:
                                groundtruth = torch.tensor(
                                    [[0.0000, 6.0000000000, 5.0000, 4.8340, 9.0000],
                                     [2.2500, 6.3332500450, 5.0000, 5.1000, 0.0000]]).view(1, 1, 2, 5)
                            else:
                                groundtruth = torch.tensor(
                                    [[0.0000, 6.5000000000, 1.2500, 4.6675000191, 4.6250],
                                     [0.5000, 7.1665000916, 1.2500, 5.0000000000, 0.0000]]).view(1, 1, 2, 5)
                        elif padding_mode == 'border':
                            if align_corners:
                                groundtruth = torch.tensor(
                                    [[1.2000, 6.0000000000, 5.0000, 4.8340, 9.0000],
                                     [2.2500, 6.3332500450, 5.0000, 5.1000, 8.7500]]).view(1, 1, 2, 5)
                            else:
                                groundtruth = torch.tensor(
                                    [[1.0000, 6.5000000000, 5.0000, 4.6675000191, 9.2500],
                                     [1.0000, 7.1665000916, 5.0000, 5.0000000000, 10.0000]]).view(1, 1, 2, 5)
                        elif padding_mode == 'reflection':
                            if align_corners:
                                groundtruth = torch.tensor(
                                    [[3.4500, 6.0000000000, 5.0000, 4.8340, 9.0000],
                                     [2.2500, 6.3332500450, 5.0000, 5.1000, 7.7500]]).view(1, 1, 2, 5)
                            else:
                                groundtruth = torch.tensor(
                                    [[3.0000004768, 6.5000000000, 5.0000, 4.6675000191, 9.2500],
                                     [1.0000000000, 7.1665000916, 5.0000, 5.0000000000, 9.2500]]).view(1, 1, 2, 5)
                        else:
                            raise AssertionError("missing groundtruth test for padding mode '{}'".format(padding_mode))
                    elif mode == 'nearest':
                        if padding_mode == 'zeros':
                            if align_corners:
                                groundtruth = torch.tensor(
                                    [[0., 8., 5., 7., 9.],
                                     [1., 8., 5., 8., 0.]]).view(1, 1, 2, 5)
                            else:
                                groundtruth = torch.tensor(
                                    [[0., 8., 5., 7., 0.],
                                     [1., 8., 5., 8., 0.]]).view(1, 1, 2, 5)
                        elif padding_mode == 'border':
                            if align_corners:
                                groundtruth = torch.tensor(
                                    [[1., 8., 5., 7., 9.],
                                     [1., 8., 5., 8., 10.]]).view(1, 1, 2, 5)
                            else:
                                groundtruth = torch.tensor(
                                    [[1., 8., 5., 7., 9.],
                                     [1., 8., 5., 8., 10.]]).view(1, 1, 2, 5)
                        elif padding_mode == 'reflection':
                            if align_corners:
                                groundtruth = torch.tensor(
                                    [[1., 8., 5., 7., 9.],
                                     [1., 8., 5., 8., 9.]]).view(1, 1, 2, 5)
                            else:
                                groundtruth = torch.tensor(
                                    [[1., 8., 5., 7., 9.],
                                     [1., 8., 5., 8., 9.]]).view(1, 1, 2, 5)
                        else:
                            raise AssertionError("missing groundtruth test for padding mode '{}'".format(padding_mode))
                    else:
                        raise AssertionError("missing groundtruth test for interpolation mode '{}'".format(mode))
                    output = F.grid_sample(input, grid, mode=mode, padding_mode=padding_mode,
                                           align_corners=align_corners)
                    self.assertEqual(output, groundtruth,
                                     "groundtruth comparison failed for mode={}, "
                                     "padding_mode={}".format(mode, padding_mode))

                    # do gradcheck
                    N = random.randint(2, 8)
                    C = random.randint(2, 6)
                    H = random.randint(2, 8)
                    W = random.randint(2, 8)
                    input = torch.randn(N, C, H, W, requires_grad=True)
                    grid = torch.randn(N, H, W, 2, requires_grad=True)
                    self.assertTrue(gradcheck(
                        lambda inp, grid: F.grid_sample(inp, grid, mode=mode, padding_mode=padding_mode,
                                                        align_corners=align_corners),
                        (input, grid)))

                    test(N, C, H, W, mode, padding_mode, align_corners=align_corners)
                    if TEST_CUDNN:
                        with cudnn.flags(enabled=False):
                            test(N, C, H, W, mode, padding_mode, align_corners=align_corners)

    def test_grid_sample_3d(self):
        def test(N, C, D, H, W, mode, padding_mode, align_corners):
            def test_shape(N, C, ID, IH, IW, D, H, W, mode, padding_mode, align_corners):
                input_cpu = torch.randn(C, N, ID, IH, IW).transpose(0, 1).requires_grad_()
                grid_cpu = torch.randn(D, N, H, W, 3).transpose(0, 1).requires_grad_()
                out_cpu = F.grid_sample(input_cpu, grid_cpu, mode=mode, padding_mode=padding_mode,
                                        align_corners=align_corners)
                self.assertTrue(out_cpu.size() == torch.Size([N, C, D, H, W]))

                gradients = torch.randn_like(out_cpu)
                out_cpu.backward(gradients)

                if TEST_CUDA:
                    input_cuda = input_cpu.detach().transpose(0, 1).cuda().transpose(0, 1).requires_grad_()
                    grid_cuda = grid_cpu.detach().transpose(0, 1).cuda().transpose(0, 1).requires_grad_()
                    out_cuda = F.grid_sample(input_cuda, grid_cuda, mode=mode, padding_mode=padding_mode,
                                             align_corners=align_corners)
                    self.assertEqual(out_cpu, out_cuda)

                    out_cuda.backward(gradients.cuda())
                    self.assertEqual(input_cpu.grad, input_cuda.grad)
                    self.assertEqual(grid_cpu.grad, grid_cuda.grad, prec=5e-5)

                    # check that zero-dimensional input strides don't error out
                    base_input = torch.randn(N, C, 1, IH, IW)
                    input_cpu = base_input.expand_as(input_cuda).requires_grad_()
                    grid_cpu = torch.randn(N, D, H, W, 3, requires_grad=True)
                    out_cpu = F.grid_sample(input_cpu, grid_cpu, mode=mode, padding_mode=padding_mode,
                                            align_corners=align_corners)

                    input_cuda = base_input.cuda().expand_as(input_cuda).requires_grad_()
                    grid_cuda = grid_cpu.detach().cuda().requires_grad_()
                    out_cuda = F.grid_sample(input_cuda, grid_cuda, mode=mode, padding_mode=padding_mode,
                                             align_corners=align_corners)
                    self.assertEqual(out_cpu, out_cuda)

            # test same size output
            test_shape(N, C, D, H, W, D, H, W, mode, padding_mode, align_corners)

            # test larger output
            N = random.randint(2, 7)
            C = random.randint(2, 5)
            ID = random.randint(2, 7)
            IH = random.randint(2, 7)
            IW = random.randint(2, 7)
            D = random.randint(ID + 1, 10)
            H = random.randint(IH + 1, 10)
            W = random.randint(IW + 1, 10)
            test_shape(N, C, ID, IH, IW, D, H, W, mode, padding_mode, align_corners)

            # test smaller output
            N = random.randint(2, 7)
            C = random.randint(2, 5)
            ID = random.randint(2, 7)
            IH = random.randint(2, 7)
            IW = random.randint(2, 7)
            D = random.randint(2, ID)
            H = random.randint(2, IH)
            W = random.randint(2, IW)
            test_shape(N, C, ID, IH, IW, D, H, W, mode, padding_mode, align_corners)

            # test 1x1 inpput
            N = random.randint(2, 7)
            C = random.randint(2, 7)
            ID = 1
            IH = 1
            IW = 1
            H = random.randint(2, 5)
            W = random.randint(2, 5)
            test_shape(N, C, ID, IH, IW, D, H, W, mode, padding_mode, align_corners)

            # testing empty grid
            N = random.randint(2, 7)
            C = random.randint(2, 5)
            ID = random.randint(2, 7)
            IH = random.randint(2, 7)
            IW = random.randint(2, 7)
            D = random.randint(3, ID + 2)
            W = random.randint(3, IW + 2)
            test_shape(N, C, ID, IH, IW, D, 0, W, mode, padding_mode, align_corners)

            # testing empty channel
            N = random.randint(2, 7)
            ID = random.randint(2, 5)
            IH = random.randint(2, 7)
            IW = random.randint(2, 7)
            D = random.randint(3, ID + 2)
            H = random.randint(3, IH + 2)
            W = random.randint(3, IW + 2)
            test_shape(N, 0, ID, IH, IW, D, H, W, mode, padding_mode, align_corners)

            # testing empty batch
            C = random.randint(2, 5)
            ID = random.randint(2, 7)
            IH = random.randint(2, 7)
            IW = random.randint(2, 7)
            D = random.randint(3, ID + 2)
            H = random.randint(3, IH + 2)
            W = random.randint(3, IW + 2)
            test_shape(0, C, ID, IH, IW, D, H, W, mode, padding_mode, align_corners)

        for mode in ('bilinear', 'nearest'):
            for padding_mode in ('zeros', 'border', 'reflection'):
                for align_corners in (True, False):
                    # do gradcheck
                    N = random.randint(2, 5)
                    C = random.randint(2, 4)
                    D = random.randint(2, 5)
                    H = random.randint(2, 5)
                    W = random.randint(2, 5)
                    input = torch.randn(N, C, D, H, W, requires_grad=True)
                    grid = torch.randn(N, D, H, W, 3, requires_grad=True)
                    self.assertTrue(gradcheck(
                        lambda inp, grid: F.grid_sample(inp, grid, mode=mode, padding_mode=padding_mode,
                                                        align_corners=align_corners),
                        (input, grid)))

                    test(N, C, D, H, W, mode, padding_mode, align_corners)

    def test_affine_grid(self):
        # test known input on CPU
        input = torch.arange(1., 7).view(1, 2, 3)
        output = F.affine_grid(input, torch.Size([1, 1, 2, 2]), align_corners=True)
        groundtruth = torch.Tensor(
            [[[0, -3], [2, 5]], [[4, 7], [6, 15]]]).view(1, 2, 2, 2)
        self.assertEqual(output, groundtruth)
        output = F.affine_grid(input, torch.Size([1, 1, 2, 2]), align_corners=False)
        groundtruth = torch.Tensor(
            [[[1.5, 1.5], [2.5, 5.5]], [[3.5, 6.5], [4.5, 10.5]]]).view(1, 2, 2, 2)
        self.assertEqual(output, groundtruth)

        for align_corners in (True, False):
            # do gradcheck
            N = random.randint(1, 8)
            C = random.randint(1, 8)
            H = random.randint(1, 8)
            W = random.randint(1, 8)
            sz = torch.Size([N, C, H, W])
            inp = torch.randn(N, 2, 3, requires_grad=True)
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")  # python2 requires this so other tests can trigger
                self.assertTrue(gradcheck(
                    lambda inp: F.affine_grid(inp, sz, align_corners=align_corners),
                    (inp,)))

        # test CPU against CUDA
        if TEST_CUDA:
            N = random.randint(1, 8)
            C = random.randint(1, 8)
            H = random.randint(1, 8)
            W = random.randint(1, 8)
            sz = torch.Size([N, C, H, W])
            for align_corners in (True, False):
                input_cpu = torch.randn(N, 2, 3, requires_grad=True)
                with warnings.catch_warnings(record=True):
                    warnings.simplefilter("always")  # python2 requires this so other tests can trigger
                    out_cpu = F.affine_grid(input_cpu, sz, align_corners=align_corners)
                gradients = torch.randn(out_cpu.size())
                out_cpu.backward(gradients)
                input_gpu = input_cpu.detach().cuda().requires_grad_()
                with warnings.catch_warnings(record=True):
                    warnings.simplefilter("always")  # python2 requires this so other tests can trigger
                    out_cuda = F.affine_grid(input_gpu, sz, align_corners=align_corners)
                out_cuda.backward(gradients.cuda())
                self.assertEqual(out_cpu, out_cuda)
                self.assertEqual(input_cpu.grad, input_gpu.grad)

    def test_affine_grid_3d(self):
        # test known input on CPU
        input = torch.arange(1., 13).view(1, 3, 4)
        output = F.affine_grid(input, torch.Size([1, 1, 2, 2, 2]), align_corners=True)
        groundtruth = torch.Tensor(
            [[[[[-2, -10, -18], [0, 0, 0]], [[2, 2, 2], [4, 12, 20]]],
              [[[4, 4, 4], [6, 14, 22]], [[8, 16, 24], [10, 26, 42]]]]]).view(1, 2, 2, 2, 3)
        self.assertEqual(output, groundtruth)
        output = F.affine_grid(input, torch.Size([1, 1, 2, 2, 2]), align_corners=False)
        groundtruth = torch.Tensor(
            [[[[[1, -1, -3], [2, 4, 6]], [[3, 5, 7], [4, 10, 16]]],
              [[[4, 6, 8], [5, 11, 17]], [[6, 12, 18], [7, 17, 27]]]]]).view(1, 2, 2, 2, 3)
        self.assertEqual(output, groundtruth)

        for align_corners in (True, False):
            # do gradcheck
            N = random.randint(1, 8)
            C = random.randint(1, 8)
            D = random.randint(1, 8)
            H = random.randint(1, 8)
            W = random.randint(1, 8)
            sz = torch.Size([N, C, D, H, W])
            inp = torch.randn(N, 3, 4, requires_grad=True)
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")  # python2 requires this so other tests can trigger
                self.assertTrue(gradcheck(
                    lambda inp: F.affine_grid(inp, sz, align_corners=align_corners),
                    (inp,)))

        # test CPU against CUDA
        if TEST_CUDA:
            N = random.randint(1, 8)
            C = random.randint(1, 8)
            D = random.randint(1, 8)
            H = random.randint(1, 8)
            W = random.randint(1, 8)
            sz = torch.Size([N, C, D, H, W])
            for align_corners in (True, False):
                input_cpu = torch.randn(N, 3, 4, requires_grad=True)
                with warnings.catch_warnings(record=True):
                    warnings.simplefilter("always")  # python2 requires this so other tests can trigger
                    out_cpu = F.affine_grid(input_cpu, sz, align_corners=align_corners)
                gradients = torch.randn(out_cpu.size())
                out_cpu.backward(gradients)
                input_gpu = input_cpu.detach().cuda().requires_grad_()
                with warnings.catch_warnings(record=True):
                    warnings.simplefilter("always")  # python2 requires this so other tests can trigger
                    out_cuda = F.affine_grid(input_gpu, sz, align_corners=align_corners)
                out_cuda.backward(gradients.cuda())
                self.assertEqual(out_cpu, out_cuda)
                self.assertEqual(input_cpu.grad, input_gpu.grad)

    @unittest.skipIf((not TEST_NUMPY) or (not TEST_SCIPY) or (scipy.__version__ < '1.0.0'),
                     "Scipy v1.0 and/or numpy not found")
    def test_affine_2d_rotate0(self):
        # scipy before 1.0.0 do not support homogeneous coordinate
        # scipy.ndimage.affine_transform, so we need to skip.
        for device in device_():
            input_size = [1, 1, 3, 3]
            input_ary = np.array(np.random.random(input_size), dtype=np.float32)
            output_size = [1, 1, 5, 5]
            angle_rad = 0.

            transform_tensor, transform_ary, offset = \
                _buildEquivalentAffineTransforms2d(device, input_size, output_size, angle_rad)

            scipy_ary = scipy.ndimage.affine_transform(
                input_ary[0, 0],
                transform_ary,
                offset=offset,
                output_shape=output_size[2:],
                order=1,
                mode='nearest',
                prefilter=False)

            affine_tensor = torch.nn.functional.affine_grid(
                transform_tensor,
                torch.Size(output_size),
                align_corners=True
            )

            gridsample_ary = torch.nn.functional.grid_sample(
                torch.tensor(input_ary, device=device).to(device),
                affine_tensor,
                padding_mode='border',
                align_corners=True
            ).to('cpu').numpy()

            assert np.abs(scipy_ary.mean() - gridsample_ary.mean()) < 1e-6
            assert np.abs(scipy_ary - gridsample_ary).max() < 1e-6

    @unittest.skipIf((not TEST_NUMPY) or (not TEST_SCIPY) or (scipy.__version__ < '1.0.0'),
                     "Scipy v1.0 and/or numpy not found")
    def test_affine_2d_rotate90(self):
        # scipy before 1.0.0 do not support homogeneous coordinate
        # scipy.ndimage.affine_transform, so we need to skip.
        for device, input_size2dsq, output_size2dsq in \
                itertools.product(device_(), input_size2dsq_(), output_size2dsq_()):
            input_size = input_size2dsq
            input_ary = np.array(np.random.random(input_size), dtype=np.float32)
            output_size = output_size2dsq
            angle_rad = 0.25 * math.pi * 2

            transform_tensor, transform_ary, offset = \
                _buildEquivalentAffineTransforms2d(device, input_size, output_size, angle_rad)

            scipy_ary = scipy.ndimage.affine_transform(
                input_ary[0, 0],
                transform_ary,
                offset=offset,
                output_shape=output_size[2:],
                order=1,
                mode='nearest',
                prefilter=True)

            if input_size2dsq == output_size2dsq:
                assert np.abs(scipy_ary.mean() - input_ary.mean()) < 1e-6
            assert np.abs(scipy_ary[0, 0] - input_ary[0, 0, 0, -1]).max() < 1e-6
            assert np.abs(scipy_ary[0, -1] - input_ary[0, 0, -1, -1]).max() < 1e-6
            assert np.abs(scipy_ary[-1, -1] - input_ary[0, 0, -1, 0]).max() < 1e-6
            assert np.abs(scipy_ary[-1, 0] - input_ary[0, 0, 0, 0]).max() < 1e-6

            affine_tensor = torch.nn.functional.affine_grid(
                transform_tensor,
                torch.Size(output_size),
                align_corners=True
            )

            gridsample_ary = torch.nn.functional.grid_sample(
                torch.tensor(input_ary, device=device).to(device),
                affine_tensor,
                padding_mode='border',
                align_corners=True
            ).to('cpu').numpy()

            assert np.abs(scipy_ary.mean() - gridsample_ary.mean()) < 1e-6
            assert np.abs(scipy_ary - gridsample_ary).max() < 1e-6

    @unittest.skipIf((not TEST_NUMPY) or (not TEST_SCIPY) or (scipy.__version__ < '1.0.0'),
                     "Scipy v1.0 and/or numpy not found")
    def test_affine_2d_rotate45(self):
        # scipy before 1.0.0 do not support homogeneous coordinate
        # scipy.ndimage.affine_transform, so we need to skip.
        for device in device_():
            input_size = [1, 1, 3, 3]
            input_ary = np.array(np.zeros(input_size), dtype=np.float32)
            input_ary[0, 0, 0, :] = 0.5
            input_ary[0, 0, 2, 2] = 1.0
            output_size = [1, 1, 3, 3]
            angle_rad = 0.125 * math.pi * 2

            transform_tensor, transform_ary, offset = \
                _buildEquivalentAffineTransforms2d(device, input_size, output_size, angle_rad)

            scipy_ary = scipy.ndimage.affine_transform(
                input_ary[0, 0],
                transform_ary,
                offset=offset,
                output_shape=output_size[2:],
                order=1,
                mode='nearest',
                prefilter=False)

            affine_tensor = torch.nn.functional.affine_grid(
                transform_tensor,
                torch.Size(output_size),
                align_corners=True
            )

            gridsample_ary = torch.nn.functional.grid_sample(
                torch.tensor(input_ary, device=device).to(device),
                affine_tensor,
                padding_mode='border',
                align_corners=True
            ).to('cpu').numpy()

            assert np.abs(scipy_ary - gridsample_ary).max() < 1e-6

    @unittest.skipIf((not TEST_NUMPY) or (not TEST_SCIPY) or (scipy.__version__ < '1.0.0'),
                     "Scipy v1.0 and/or numpy not found")
    def test_affine_2d_rotateRandom(self):
        # scipy before 1.0.0 do not support homogeneous coordinate
        # scipy.ndimage.affine_transform, so we need to skip.
        for device, angle_rad, input_size2d, output_size2d in \
                itertools.product(device_(), angle_rad_(), input_size2d_(), output_size2d_()):

            input_size = input_size2d
            input_ary = np.array(np.random.random(input_size), dtype=np.float32).round(3)
            output_size = output_size2d

            input_ary[0, 0, 0, 0] = 2
            input_ary[0, 0, 0, -1] = 4
            input_ary[0, 0, -1, 0] = 6
            input_ary[0, 0, -1, -1] = 8

            transform_tensor, transform_ary, grid_ary = \
                _buildEquivalentAffineTransforms2d(device, input_size, output_size, angle_rad)

            scipy_ary = scipy.ndimage.affine_transform(
                input_ary[0, 0],
                transform_ary,
                output_shape=output_size[2:],
                order=1,
                mode='nearest',
                prefilter=False)

            affine_tensor = torch.nn.functional.affine_grid(
                transform_tensor,
                torch.Size(output_size),
                align_corners=True
            )

            gridsample_ary = torch.nn.functional.grid_sample(
                torch.tensor(input_ary, device=device).to(device),
                affine_tensor,
                padding_mode='border',
                align_corners=True
            ).to('cpu').numpy()

            affine_tensor = affine_tensor.to('cpu')

            for r in range(affine_tensor.size(1)):
                for c in range(affine_tensor.size(2)):
                    grid_out = np.dot(grid_ary, [r, c, 1])
                    assert np.allclose(affine_tensor[0, r, c], grid_out[:2], atol=1e-5)

            assert np.abs(scipy_ary - gridsample_ary).max() < 1e-5

    @unittest.skipIf((not TEST_NUMPY) or (not TEST_SCIPY) or (scipy.__version__ < '1.0.0'),
                     "Scipy v1.0 and/or numpy not found")
    def test_affine_3d_rotateRandom(self):
        # scipy before 1.0.0 do not support homogeneous coordinate
        # scipy.ndimage.affine_transform, so we need to skip.
        for device, angle_rad, axis_vector, input_size3d, output_size3d in \
                itertools.product(device_(), angle_rad_(), axis_vector_(), input_size3d_(), output_size3d_()):
            input_size = input_size3d
            input_ary = np.array(np.random.random(input_size), dtype=np.float32)
            output_size = output_size3d

            input_ary[0, 0, 0, 0, 0] = 2
            input_ary[0, 0, 0, 0, -1] = 3
            input_ary[0, 0, 0, -1, 0] = 4
            input_ary[0, 0, 0, -1, -1] = 5
            input_ary[0, 0, -1, 0, 0] = 6
            input_ary[0, 0, -1, 0, -1] = 7
            input_ary[0, 0, -1, -1, 0] = 8
            input_ary[0, 0, -1, -1, -1] = 9

            transform_tensor, transform_ary, grid_ary = \
                _buildEquivalentAffineTransforms3d(device, input_size, output_size, angle_rad, axis_vector)

            scipy_ary = scipy.ndimage.affine_transform(
                input_ary[0, 0],
                transform_ary,
                output_shape=output_size[2:],
                order=1,
                mode='nearest',
                prefilter=False)

            affine_tensor = torch.nn.functional.affine_grid(
                transform_tensor,
                torch.Size(output_size),
                align_corners=True
            )

            gridsample_ary = torch.nn.functional.grid_sample(
                torch.tensor(input_ary, device=device).to(device),
                affine_tensor,
                padding_mode='border',
                align_corners=True
            ).to('cpu').numpy()

            affine_tensor = affine_tensor.to('cpu')

            for i in range(affine_tensor.size(1)):
                for r in range(affine_tensor.size(2)):
                    for c in range(affine_tensor.size(3)):
                        grid_out = np.dot(grid_ary, [i, r, c, 1])
                        assert np.allclose(affine_tensor[0, i, r, c], grid_out[:3], atol=1e-5)

            assert np.abs(scipy_ary - gridsample_ary).max() < 1e-5

    def test_upsamplingNearest1d(self):
        m = nn.Upsample(size=4, mode='nearest')
        in_t = torch.ones(1, 1, 2)
        with warnings.catch_warnings(record=True) as w:
            out_t = m(in_t)
        self.assertEqual(torch.ones(1, 1, 4), out_t.data)

        input = torch.randn(1, 1, 2, requires_grad=True)
        gradcheck(lambda x: F.interpolate(x, 4, mode='nearest'), [input])

    def test_upsamplingLinear1d(self):
        for align_corners in [True, False]:
            kwargs = dict(mode='linear', align_corners=align_corners)

            # test float scale factor up & downsampling
            for scale_factor in [0.5, 1.5, 2]:
                m = nn.Upsample(scale_factor=scale_factor, **kwargs)
                in_t = torch.ones(1, 1, 2)
                out_size = int(math.floor(in_t.shape[-1] * scale_factor))
                with warnings.catch_warnings(record=True) as w:
                    out_t = m(in_t)
                self.assertEqual(torch.ones(1, 1, out_size), out_t.data)

                input = torch.randn(1, 1, 2, requires_grad=True)
                gradcheck(lambda x: F.interpolate(x, out_size, **kwargs), (input,))

    def test_upsamplingLinear1d_spatial_invariance(self):
        m = nn.Upsample(scale_factor=3, mode='linear', align_corners=False)
        in_t_9 = torch.zeros(1, 1, 9)
        in_t_9[:, :, :4].normal_()
        with warnings.catch_warnings(record=True) as w:
            out_t_9 = m(in_t_9)
            out_t_5 = m(in_t_9[:, :, :5])
        self.assertEqual(out_t_9[:, :, :15], out_t_5)

    def test_upsamplingNearest2d(self):
        m = nn.Upsample(size=4, mode='nearest')
        in_t = torch.ones(1, 1, 2, 2)
        with warnings.catch_warnings(record=True) as w:
            out_t = m(in_t)
        self.assertEqual(torch.ones(1, 1, 4, 4), out_t.data)

        input = torch.randn(1, 1, 2, 2, requires_grad=True)
        self.assertEqual(
            F.interpolate(input, 4, mode='nearest'),
            F.interpolate(input, scale_factor=2, mode='nearest'))
        gradcheck(lambda x: F.interpolate(x, 4, mode='nearest'), [input])
        gradgradcheck(lambda x: F.interpolate(x, 4, mode='nearest'), [input])

    def test_upsamplingBilinear2d(self):
        for align_corners in [True, False]:
            kwargs = dict(mode='bilinear', align_corners=align_corners)

            # test float scale factor up & downsampling
            for scale_factor in [0.5, 1.5, 2]:
                m = nn.Upsample(scale_factor=scale_factor, **kwargs)
                in_t = torch.ones(1, 1, 2, 2)
                out_size = int(math.floor(in_t.shape[-1] * scale_factor))
                with warnings.catch_warnings(record=True) as w:
                    out_t = m(in_t)
                self.assertEqual(torch.ones(1, 1, out_size, out_size), out_t.data)

                input = torch.randn(1, 1, 2, 2, requires_grad=True)
                gradcheck(lambda x: F.interpolate(x, out_size, **kwargs), [input])

    def test_upsamplingBicubic2d(self):
        # test output against known input: align_corners=False result must match opencv
        in_t = torch.arange(8).view(1, 2, 2, 2).type(torch.FloatTensor)
        expected_out_t = torch.Tensor(
            [[[[-0.31641, 0.01562, 0.56250, 0.89453],
              [0.34766, 0.67969, 1.22656, 1.55859],
              [1.44141, 1.77344, 2.32031, 2.65234],
              [2.10547, 2.43750, 2.98438, 3.31641]],

             [[3.68359, 4.01562, 4.56250, 4.89453],
              [4.34766, 4.67969, 5.22656, 5.55859],
              [5.44141, 5.77344, 6.32031, 6.65234],
              [6.10547, 6.43750, 6.98438, 7.31641]]]])
        out_t = F.interpolate(in_t, scale_factor=2, mode='bicubic', align_corners=False)
        torch.set_printoptions(precision=5)
        self.assertEqual(out_t, expected_out_t)

        device_list = ['cpu']
        if TEST_CUDA:
            device_list.append('cuda')

        for align_corners in [True, False]:
            kwargs = dict(mode='bicubic', align_corners=align_corners)
            # test float scale factor up & downsampling
            for device in device_list:
                for scale_factor in [0.5, 1.5, 2]:
                    in_t = torch.ones(2, 2, 2, 2).to(device)
                    out_t = F.interpolate(in_t, scale_factor=scale_factor, **kwargs)
                    out_size = int(math.floor(in_t.shape[-1] * scale_factor))
                    self.assertEqual(torch.ones(2, 2, out_size, out_size), out_t.data)

                    input = torch.randn(2, 2, 2, 2, requires_grad=True)
                    gradcheck(lambda x: F.interpolate(x, out_size, **kwargs), [input])

    def test_upsampling_not_recompute_scale_factor(self):
        # test output against known input: result must match opencv
        in_t = torch.arange(8).view(1, 2, 2, 2).type(torch.FloatTensor)
        expected_out_t = torch.Tensor(
            [[[[-0.32725, -0.08843, 0.37933, 0.79744],
              [0.15039, 0.38921, 0.85697, 1.27508],
              [1.08591, 1.32473, 1.79249, 2.21060],
              [1.92213, 2.16095, 2.62871, 3.04682]],

             [[3.67275, 3.91157, 4.37933, 4.79744],
              [4.15039, 4.38921, 4.85697, 5.27508],
              [5.08591, 5.32473, 5.79249, 6.21060],
              [5.92213, 6.16095, 6.62871, 7.04682]]]])
        out_t = F.interpolate(in_t, scale_factor=2.3, mode='bicubic', align_corners=False, recompute_scale_factor=False)
        torch.set_printoptions(precision=5)
        self.assertEqual(out_t, expected_out_t)

        device_list = ['cpu']
        if TEST_CUDA:
            device_list.append('cuda')

        for align_corners in [True, False]:
            kwargs = dict(mode='bicubic', align_corners=align_corners)
            # test float scale factor up & downsampling
            for device in device_list:
                for scale_factor in [0.6, 1.6, 2.3]:
                    in_t = torch.ones(2, 2, 2, 2).to(device)
                    out_t = F.interpolate(in_t, scale_factor=scale_factor, **kwargs)
                    out_size = int(math.floor(in_t.shape[-1] * scale_factor))
                    self.assertEqual(torch.ones(2, 2, out_size, out_size), out_t.data)

                    input = torch.randn(2, 2, 2, 2, requires_grad=True)
                    gradcheck(lambda x: F.interpolate(x, out_size, **kwargs), [input])

    def test_upsamplingBilinear2d_spatial_invariance(self):
        m = nn.Upsample(scale_factor=3, mode='bilinear', align_corners=False)
        in_t_9 = torch.zeros(1, 1, 9, 9)
        in_t_9[:, :, :4, :4].normal_()
        with warnings.catch_warnings(record=True) as w:
            out_t_9 = m(in_t_9)
            out_t_5 = m(in_t_9[:, :, :5, :5])
        self.assertEqual(out_t_9[:, :, :15, :15], out_t_5)

    def test_upsamplingNearest3d(self):
        m = nn.Upsample(size=4, mode='nearest')
        in_t = torch.ones(1, 1, 2, 2, 2)
        with warnings.catch_warnings(record=True) as w:
            out_t = m(in_t)
        self.assertEqual(torch.ones(1, 1, 4, 4, 4), out_t.data)

        input = torch.randn(1, 1, 2, 2, 2, requires_grad=True)
        gradcheck(lambda x: F.interpolate(x, 4, mode='nearest'), [input])

    def test_upsamplingTrilinear3d(self):
        for align_corners in [True, False]:
            kwargs = dict(mode='trilinear', align_corners=align_corners)

            # test float scale factor up & downsampling
            for scale_factor in [0.5, 1.5, 2]:
                m = nn.Upsample(scale_factor=scale_factor, **kwargs)
                in_t = torch.ones(1, 1, 2, 2, 2)
                out_size = int(math.floor(in_t.shape[-1] * scale_factor))
                with warnings.catch_warnings(record=True) as w:
                    out_t = m(in_t)
                self.assertEqual(torch.ones(1, 1, out_size, out_size, out_size), out_t.data)

                input = torch.randn(1, 1, 2, 2, 2, requires_grad=True)
                self.assertEqual(
                    F.interpolate(input, (out_size, out_size, out_size), **kwargs),
                    F.interpolate(input, scale_factor=scale_factor, **kwargs))
                gradcheck(lambda x: F.interpolate(x, out_size, **kwargs), [input])
                gradgradcheck(lambda x: F.interpolate(x, out_size, **kwargs), [input])

    def test_upsamplingTrilinear3d_spatial_invariance(self):
        m = nn.Upsample(scale_factor=3, mode='trilinear', align_corners=False)
        in_t_9 = torch.zeros(1, 1, 9, 9, 9)
        in_t_9[:, :, :4, :4, :4].normal_()
        with warnings.catch_warnings(record=True) as w:
            out_t_9 = m(in_t_9)
            out_t_5 = m(in_t_9[:, :, :5, :5, :5])
        self.assertEqual(out_t_9[:, :, :15, :15, :15], out_t_5)

    def test_interpolate(self):
        def _test_interpolate_helper(in_t, scale_factor, layer):
            out_size = int(math.floor(in_t.shape[-1] * scale_factor))
            dim = len(in_t.shape) - 2
            out_shape = [1, 1] + [out_size] * dim
            with warnings.catch_warnings(record=True) as w:
                out_t = m(in_t)
            self.assertEqual(torch.ones(out_shape), out_t)

            self.assertEqual(
                F.interpolate(in_t, (out_size,) * dim, **kwargs),
                F.interpolate(in_t, scale_factor=scale_factor, **kwargs))
            gradcheck(lambda x: F.interpolate(x, out_size, **kwargs), [in_t])
            gradgradcheck(lambda x: F.interpolate(x, out_size, **kwargs), [in_t])

        def _make_input(dim):
            size = [1, 1]
            size += [2] * dim
            return torch.ones(size, requires_grad=True)

        device_list = ['cpu']
        if TEST_CUDA:
            device_list.append('cuda')

        for device in device_list:
            for scale_factor in [0.5, 1.5, 2]:
                for mode in ['nearest', 'area']:
                    kwargs = dict(mode=mode)
                    m = nn.Upsample(scale_factor=scale_factor, **kwargs).to(device)
                    for input in [_make_input(1), _make_input(2), _make_input(3)]:
                        _test_interpolate_helper(input, scale_factor, m)

                for align_corners in [True, False]:
                    kwargs = dict(mode='linear', align_corners=align_corners)
                    m = nn.Upsample(scale_factor=scale_factor, **kwargs).to(device)
                    _test_interpolate_helper(_make_input(1), scale_factor, m)

                    kwargs = dict(mode='bilinear', align_corners=align_corners)
                    m = nn.Upsample(scale_factor=scale_factor, **kwargs).to(device)
                    _test_interpolate_helper(_make_input(2), scale_factor, m)

                    kwargs = dict(mode='bicubic', align_corners=align_corners)

                    def m(t):
                        return F.interpolate(t, scale_factor=scale_factor, **kwargs).to(device)
                    _test_interpolate_helper(_make_input(2), scale_factor, m)

                    kwargs = dict(mode='trilinear', align_corners=align_corners)
                    m = nn.Upsample(scale_factor=scale_factor, **kwargs).to(device)
                    _test_interpolate_helper(_make_input(3), scale_factor, m)

    def test_linear_broadcasting(self):
        m = nn.Linear(5, 8)
        inp = torch.randn(2, 3, 5)
        expected = m(inp.view(6, 5)).view(2, 3, 8)
        self.assertEqual(expected, m(inp))

    def test_bilinear(self):
        module = nn.Bilinear(10, 10, 8)
        input1 = torch.randn(4, 10, requires_grad=True)
        input2 = torch.randn(4, 10, requires_grad=True)
        grad_output = torch.randn(4, 8)

        res = module(input1, input2)
        expected = (torch.einsum("bi,kij,bj->bk", input1, module.weight, input2) +
                    module.bias)
        self.assertEqual(res, expected)
        grads = torch.autograd.grad(res, [module.weight, module.bias, input1, input2], grad_output)
        grads_expected = torch.autograd.grad(expected, [module.weight, module.bias, input1, input2], grad_output)
        for g, ge in zip(grads, grads_expected):
            self.assertEqual(g, ge)

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
        inp = torch.randn(9, 4, 5, requires_grad=True)
        weight = torch.randn(3, 5, 6, requires_grad=True)
        bias = torch.randn(6, requires_grad=True)

        gradcheck(lambda i, w, b, pad: F.conv_tbc(i, w, b, pad), (inp, weight, bias, 3))

    def run_conv_double_back_test(self, kern, stride, padding, chan_in, chan_out, batch_size,
                                  inp_size, dilation, no_weight, groups=1, use_cuda=False,
                                  use_bias=True, dtype=torch.double):
        if use_cuda:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        x = torch.randn(batch_size, chan_in, inp_size, inp_size, device=device,
                        dtype=dtype, requires_grad=True)
        weight = torch.randn(chan_out, chan_in // groups, kern, kern, device=device,
                             dtype=dtype, requires_grad=not no_weight)
        if use_bias:
            bias = torch.randn(chan_out, device=device, dtype=dtype, requires_grad=True)
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
        grad_y = torch.randn_like(dummy_out, device=device, dtype=dtype, requires_grad=True)

        # Issue #15353: test mkldnn double backward, don't run gradgradcheck due
        # to imprecision issues
        if dtype == torch.float:
            g, = torch.autograd.grad(dummy_out.sum(), x, create_graph=True)
            return g.requires_grad

        return gradgradcheck(func, inputs, (grad_y,))

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    @unittest.skipIf(not TEST_CUDNN, "needs cudnn")
    @skipIfRocm
    def test_conv_cudnn_nhwc(self):
        input = torch.randint(1, 10, (2, 8, 4, 4), dtype=torch.float32, device="cuda", requires_grad=True)
        input = input.contiguous(memory_format=torch.channels_last)
        input.retain_grad()
        grad = torch.rand(2, 4, 2, 2, dtype=torch.float32, device="cuda")
        grad = grad.contiguous(memory_format=torch.channels_last)
        conv = nn.Conv2d(8, 4, 3).cuda().float()
        conv.weight.data = conv.weight.contiguous(memory_format=torch.channels_last)

        ref_input = input.detach().clone().contiguous().requires_grad_(True)
        ref_grad = grad.detach().clone().contiguous()
        ref_conv = nn.Conv2d(8, 4, 3).cuda().float()
        # load_state_dict will restore the stride & memory_layout on ref_conv.weight.
        ref_conv.load_state_dict(conv.state_dict())

        out = conv(input)
        out.backward(grad)
        ref_out = ref_conv(ref_input)
        ref_out.backward(ref_grad)

        self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))
        self.assertTrue(ref_out.is_contiguous())
        self.assertEqual(out, ref_out)
        self.assertEqual(conv.weight.grad, ref_conv.weight.grad)
        self.assertEqual(conv.bias.grad, ref_conv.bias.grad)
        self.assertEqual(input.grad, ref_input.grad)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    @unittest.skipIf(not TEST_CUDNN, "needs cudnn")
    @skipIfRocm
    def test_grouped_conv_cudnn_nhwc_support(self):
        # in order to catch the hols in grouped convolution in nhwc support for earlier cudnn version
        input = torch.randn((16, 16, 8, 8), dtype=torch.float16, device="cuda").to(memory_format=torch.channels_last)
        weight = torch.randn((8, 4, 3, 3), dtype=torch.float16, device="cuda").to(memory_format=torch.channels_last)
        out = torch.cudnn_convolution(input, weight, None, (1, 1), (1, 1), (1, 1), 4, False, False)
        input = torch.randn((16, 8, 8, 8), dtype=torch.float16, device="cuda").to(memory_format=torch.channels_last)
        out = torch.cudnn_convolution_transpose(input, weight, None, (1, 1), (0, 0), (1, 1), (1, 1), 4, False, False)

    @unittest.expectedFailure
    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    @unittest.skipIf(not TEST_CUDNN, "needs cudnn")
    @skipIfRocm
    def test_conv_cudnn_memory_layout_dominance(self):
        # desired behavior here is to have the memory_layout of conv.weight to
        # dominante the layout of output.
        # which is not the same as current behavior, we'll fix this in
        # following up PRs and remove the `expectedFailure` tag
        input = torch.randint(1, 10, (2, 8, 4, 4), dtype=torch.float32, device="cuda", requires_grad=True)
        conv = nn.Conv2d(8, 4, 3).cuda().float()

        out = conv(input)
        self.assertTrue(out.is_contiguous())

        input = input.contiguous(memory_format=torch.channels_last)
        out = conv(input)
        self.assertTrue(out.is_contiguous())

        conv.weight.data = conv.weight.contiguous(memory_format=torch.channels_last)
        out = conv(input)
        self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))

        input = input.contiguous()
        out = conv(input)
        self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))

    def test_conv_double_backward(self):
        batch_size = 2
        for kern, inp_size, dilations in [(3, 6, [1, 2]), (3, 7, [1]), (4, 9, [1])]:
            for stride, padding, chan_in, chan_out, dilation in \
                    product([1, 2], [0, 1, 2], [2], [3], dilations):
                for no_weight in (True, False):
                    for dtype in (torch.float, torch.double):
                        result = self.run_conv_double_back_test(kern, stride,
                                                                padding, chan_in, chan_out,
                                                                batch_size, inp_size, dilation,
                                                                no_weight, dtype=dtype)
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
                                        "\ndtype: " + str(dtype))

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
        input = torch.tensor([1, 1, 1], dtype=torch.double, device="cuda").view(1, 1, 3)
        weights1 = torch.tensor([1], dtype=torch.double, device="cuda").expand(1, 1, 2)
        weights2 = torch.tensor([1], dtype=torch.double, device="cuda").expand(1, 1, 2).contiguous()
        self.assertEqual(F.conv1d(input, weights1, bias=None, stride=2, dilation=2),
                         F.conv1d(input, weights2, bias=None, stride=2, dilation=2))

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    @repeat_test_for_types(DOUBLE_TENSORTYPES)
    def test_conv_double_backward_cuda(self, dtype=torch.double):
        # Double backward only runs with DoubleTensor due to precison reason
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

                for has_bias in [True, False]:
                    input_shape = [batch, chan_in]
                    weight_shape = [chan_out, chan_in]
                    for _ in range(dim):
                        input_shape.append(inp_size)
                        weight_shape.append(kern)

                    input = torch.randn(input_shape, requires_grad=True)
                    weight = torch.randn(weight_shape, requires_grad=True)
                    if has_bias:
                        bias = torch.randn([chan_out], requires_grad=True)
                    output = func_forward(input, weight, stride=stride, padding=padding, dilation=dilation, bias=bias)

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

    @unittest.skipIf(not torch._nnpack_available(), "NNPACK unavailable")
    def test_nnpack_conv(self):
        for kern, inp_size in [(3, 6), (3, 7), (4, 9)]:
            for batch, stride, padding, chan_in, chan_out in \
                    product([1, 2, 3, 4], [1, 2], [0, 1, 2], [2], [3]):

                for has_bias in [True, False]:
                    input_shape = [batch, chan_in]
                    weight_shape = [chan_out, chan_in]
                    for _ in range(2):
                        input_shape.append(inp_size)
                        weight_shape.append(kern)

                    input = torch.randn(input_shape, requires_grad=True, dtype=torch.float)
                    weight = torch.randn(weight_shape, requires_grad=True, dtype=torch.float)
                    if has_bias:
                        bias = torch.randn([chan_out], requires_grad=True, dtype=torch.float)
                    output = torch._nnpack_spatial_convolution(input, weight, stride=stride, padding=padding, bias=bias)
                    output_expected = torch.nn.functional.conv2d(input, weight, stride=stride, padding=padding, bias=bias)
                    self.assertAlmostEqual(output, output_expected, delta=3e-4)

                    gradient_o = torch.randn(output.shape, dtype=torch.float)

                    grads = torch.autograd.grad(output, [input, weight], gradient_o)
                    grads_expected = torch.autograd.grad(output_expected, [input, weight], gradient_o)
                    for gr, gr_expected in zip(grads, grads_expected):
                        self.assertAlmostEqual(gr, gr_expected, delta=3e-4)

    def test_fold_invalid_arg(self):
        # input wrong dimension

        fold = nn.Fold(output_size=(4, 5), kernel_size=(2, 3))
        with self.assertRaisesRegex(NotImplementedError, r"Only 3D input Tensors are supported"):
            fold(torch.randn(1, 5))

        # input.size(1) not divisible by \prod(kernel_size)

        fold = nn.Fold(output_size=(4, 5), kernel_size=(2, 3))
        with self.assertRaisesRegex(RuntimeError, r"be divisible by the product of kernel_size"):
            fold(torch.randn(1, 5, 9))

        with self.assertRaisesRegex(RuntimeError, r"be divisible by the product of kernel_size"):
            fold(torch.randn(1, 19, 9))

        # input.size(2) not matching the total number of sliding blocks

        with self.assertRaisesRegex(RuntimeError, r"match the calculated number of sliding blocks"):
            fold = nn.Fold(output_size=(4, 5), kernel_size=(2, 3))
            fold(torch.randn(1, 6, 10))

        with self.assertRaisesRegex(RuntimeError, r"match the calculated number of sliding blocks"):
            fold = nn.Fold(output_size=(4, 5), kernel_size=(2, 3), stride=(2, 2))
            fold(torch.randn(1, 6, 5))

        with self.assertRaisesRegex(RuntimeError, r"match the calculated number of sliding blocks"):
            fold = nn.Fold(output_size=(4, 5), kernel_size=(2, 3), stride=(2, 2), dilation=(1, 2), padding=(2, 0))
            fold(torch.randn(1, 6, 5))  # should be 4 * 1 = 4 sliding blocks

    def test_unfold_invalid_arg(self):
        # input wrong dimension

        unfold = nn.Unfold(kernel_size=(2, 3))
        with self.assertRaisesRegex(NotImplementedError, r"Only 4D input Tensors are supported"):
            unfold(torch.randn(1, 5, 2))

        # calculated output shape is too small

        with self.assertRaisesRegex(RuntimeError, r"too small \(non-positive\)"):
            unfold = nn.Unfold(kernel_size=(2, 3))
            unfold(torch.randn(1, 2, 2, 2))

        with self.assertRaisesRegex(RuntimeError, r"too small \(non-positive\)"):
            unfold = nn.Unfold(kernel_size=(5, 3), padding=(1, 1))
            unfold(torch.randn(1, 2, 2, 3))

        with self.assertRaisesRegex(RuntimeError, r"too small \(non-positive\)"):
            unfold = nn.Unfold(kernel_size=(1, 3), padding=(1, 1), dilation=(1, 2))
            unfold(torch.randn(1, 2, 2, 2))

    def test_conv_padding_mode(self):
        with self.assertRaisesRegex(ValueError, "padding_mode must be one of"):
            nn.Conv2d(3, 3, 3, padding_mode="xyz")

        with self.assertRaisesRegex(ValueError, "padding_mode must be one of"):
            nn.Conv2d(3, 3, 3, padding_mode=3)

        with self.assertRaisesRegex(ValueError, "Only \"zeros\" "):
            nn.ConvTranspose2d(3, 3, 3, padding_mode="reflect")

    def test_softmin(self):
        x = torch.randn(2, 16)
        self.assertEqual(F.softmin(x, 1), F.softmax(-x, 1))
        self.assertEqual(F.softmin(x, 0), F.softmax(-x, 0))

    @repeat_test_for_types([torch.float, torch.bfloat16])
    def test_log_softmax(self, dtype=torch.float):
        x_small = torch.ones(1, 2, dtype=dtype)
        x_big = x_small + 1e16
        self.assertEqual(F.log_softmax(x_small, -1), F.log_softmax(x_big, -1))

    def test_log_softmax_cpu(self, dtype=torch.bfloat16):
        inputf = torch.rand(32, 100, device="cpu", dtype=torch.float, requires_grad=True)
        input = inputf.to(dtype).detach().requires_grad_(True)
        outf = F.log_softmax(inputf, dim=-1)
        out = F.log_softmax(input, dim=-1)
        self.assertEqual(out.dtype, dtype)
        self.assertEqual(out, outf, prec=0.1)

        out.sum().backward()
        outf.sum().backward()
        self.assertEqual(input.grad.dtype, dtype)
        self.assertEqual(input.grad, inputf.grad.to(dtype), prec=0.1)

    def test_adaptive_log_softmax(self):
        # args validation
        with self.assertRaises(ValueError):
            _ = nn.AdaptiveLogSoftmaxWithLoss(16, 20, [5, 15, 15], div_value=2.)

        with self.assertRaises(ValueError):
            _ = nn.AdaptiveLogSoftmaxWithLoss(16, 20, [5, 15, 10], div_value=2.)

        with self.assertRaises(ValueError):
            _ = nn.AdaptiveLogSoftmaxWithLoss(16, 20, [5, 10, 25], div_value=2.)

        with self.assertRaisesRegex(ValueError, "cutoffs should be a sequence of unique,"):
            _ = nn.AdaptiveLogSoftmaxWithLoss(16, 20, [5, 10, 20], div_value=2.)

        # not raise
        _ = nn.AdaptiveLogSoftmaxWithLoss(16, 20, [5, 10, 19], div_value=2.)

        # input shapes
        with self.assertRaisesRegex(RuntimeError, r"Input and target should have the same size"):
            asfm = nn.AdaptiveLogSoftmaxWithLoss(16, 20, [5, 10, 15], div_value=2.)
            x = torch.randn(2, 16)
            y = torch.tensor([0, 5, 10])
            asfm(x, y)

        # out-of-bound targets
        with self.assertRaisesRegex(RuntimeError, r"Target values should be in"):
            asfm = nn.AdaptiveLogSoftmaxWithLoss(16, 20, [5, 10, 15], div_value=2.)
            x = torch.randn(2, 16)
            y = torch.tensor([0, 20])
            asfm(x, y)

        # cluster sizes
        asfm = nn.AdaptiveLogSoftmaxWithLoss(16, 20, [5, 10, 15], div_value=2.)
        x = torch.randn(2, 16)
        y = torch.tensor([0, 17])

        self.assertEqual(asfm.head.weight.size(), (5 + 3, 16))   # 5 targets in head, 3 clusters, dimensionality 16
        self.assertEqual(asfm.tail[0][1].weight.size(), (5, 8))  # 5 targets in this cluster, dimensionality 8
        self.assertEqual(asfm.tail[1][1].weight.size(), (5, 4))
        self.assertEqual(asfm.tail[2][1].weight.size(), (5, 2))

        self.assertEqual(asfm(x, y).output.size(), (2, ))

        # log_probs actually returns log_proba
        asfm = nn.AdaptiveLogSoftmaxWithLoss(8, 4, [2], div_value=2.)
        x = torch.randn(4, 8)
        logprob_out = asfm.log_prob(x)

        self.assertEqual(torch.exp(logprob_out).data.sum(1), torch.ones(4))

        # forward returns the same thing as log_probs
        for v in [0, 1, 2, 3]:
            y = torch.full((4,), v, dtype=torch.long)
            out, loss = asfm(x, y)

            self.assertEqual(out, logprob_out.gather(1, y.unsqueeze(1)).squeeze())
            self.assertEqual(loss, F.nll_loss(logprob_out, y))

        # predict
        x = torch.randn(64, 8).abs_()

        # argmax in shortlist
        asfm = nn.AdaptiveLogSoftmaxWithLoss(8, 10, [4, 8], div_value=2., head_bias=True)
        asfm.head.weight.data.abs_()
        asfm.head.bias.data.abs_()
        asfm.head.weight.data[asfm.shortlist_size:, :].zero_()

        out = asfm.predict(x)
        self.assertEqual(out, asfm.log_prob(x).argmax(dim=1))

        # argmax outside of shortlist
        asfm = nn.AdaptiveLogSoftmaxWithLoss(8, 10, [4, 8], div_value=2., head_bias=True)
        asfm.head.weight.data.abs_()
        asfm.head.bias.data.abs_()
        asfm.head.weight.data[:asfm.shortlist_size, :].zero_()

        out = asfm.predict(x)
        self.assertEqual(out, asfm.log_prob(x).argmax(dim=1))

        # half of the argmax in shortlist, half in clusters
        asfm = nn.AdaptiveLogSoftmaxWithLoss(8, 10, [4, 8], div_value=2., head_bias=True)
        asfm.head.weight.data.abs_()
        asfm.head.bias.data.abs_()

        x[:32, :asfm.shortlist_size].zero_()
        x[32:, asfm.shortlist_size:].zero_()

        asfm.head.weight.data[:asfm.shortlist_size, asfm.shortlist_size:].zero_()
        asfm.head.weight.data[asfm.shortlist_size:, :asfm.shortlist_size].zero_()

        out = asfm.predict(x)
        self.assertEqual(out, asfm.log_prob(x).argmax(dim=1))

    def test_cross_entropy_loss(self, dtype=torch.bfloat16):
        loss_cpu = nn.CrossEntropyLoss().cpu()
        inputf = torch.randn(15, 10, device="cpu", dtype=torch.float, requires_grad=True)
        input = inputf.to(dtype).detach().requires_grad_(True)
        target = torch.empty(15, dtype=torch.long).random_(10)

        outf = loss_cpu(inputf, target)
        out = loss_cpu(input, target)
        self.assertEqual(out.dtype, dtype)
        self.assertEqual(out, outf, prec=1e-1)

        outf.backward()
        out.backward()
        self.assertEqual(input.grad.dtype, dtype)
        self.assertEqual(input.grad, inputf.grad, prec=1e-1)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_convert_sync_batchnorm(self):
        module = torch.nn.Sequential(
            torch.nn.BatchNorm1d(100),
            torch.nn.InstanceNorm1d(100)
        ).cuda()
        sync_bn_module = torch.nn.SyncBatchNorm.convert_sync_batchnorm(module)
        children = list(sync_bn_module.children())
        self.assertEqual(children[0].__class__, torch.nn.SyncBatchNorm)
        self.assertEqual(children[1].__class__, torch.nn.InstanceNorm1d)


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

    def _create_random_nd_tensor(self, dims, size_min, size_max):
        size = [random.randint(size_min, size_max) for _ in range(dims)]
        tensor = torch.zeros(size)
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
        for dims in [1, 2, 4]:
            input_tensor = self._create_random_nd_tensor(dims, size_min=30, size_max=50)
            a = self._random_float(-3, 3)
            b = a + self._random_float(1, 5)
            init.uniform_(input_tensor, a=a, b=b)
            assert self._is_uniform(input_tensor, a, b)

    @unittest.skipIf(not TEST_SCIPY, "Scipy not found.")
    def test_normal(self):
        for dims in [1, 2, 4]:
            input_tensor = self._create_random_nd_tensor(dims, size_min=30, size_max=50)
            mean = self._random_float(-3, 3)
            std = self._random_float(1, 5)
            init.normal_(input_tensor, mean=mean, std=std)

            assert self._is_normal(input_tensor, mean, std)

    def test_constant(self):
        for dims in [1, 2, 4]:
            input_tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=5)
            val = self._random_float(1, 10)
            init.constant_(input_tensor, val)

            self.assertEqual(input_tensor, input_tensor.clone().fill_(val))

    def test_ones_and_zeros(self):
        for init_fn_, val in zip([init.ones_, init.zeros_], [1, 0]):
            for dims in [1, 2, 4]:
                input_tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=5)
                init_fn_(input_tensor)

                self.assertEqual(input_tensor, input_tensor.clone().fill_(val))

    def test_eye(self):
        input_tensor = self._create_random_nd_tensor(2, size_min=1, size_max=5)
        init.eye_(input_tensor)

        # Check every single element
        for i in range(input_tensor.size(0)):
            for j in range(input_tensor.size(1)):
                if i == j:
                    assert input_tensor[i][j] == 1
                else:
                    assert input_tensor[i][j] == 0

    def test_eye_only_works_on_2d_inputs(self):
        for dims in [1, 3]:
            with self.assertRaises(ValueError):
                tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=3)
                init.eye_(tensor)

    def test_max_unpool(self):
        # Test 1D
        output, indices = F.max_pool1d(torch.randn([1, 1, 4]), 2, stride=2, return_indices=True)
        self.assertEqual(F.max_unpool1d(output, indices, 2), F.max_unpool1d(output, indices, 2, stride=2))

        # Test list / tuple passed as argument to max_unpool1d
        input = torch.randn([1, 1, 5])
        output, indices = F.max_pool1d(input, 2, stride=2, return_indices=True)
        self.assertEqual(F.max_unpool1d(output, indices, 2, stride=2, output_size=input.shape),
                         F.max_unpool1d(output, indices, 2, stride=2, output_size=input.size()))

        # Test 2D
        output, indices = F.max_pool2d(torch.randn([1, 1, 4, 4]), 2, stride=2, return_indices=True)
        self.assertEqual(F.max_unpool2d(output, indices, 2), F.max_unpool2d(output, indices, 2, stride=2))

        # Test 3D
        output, indices = F.max_pool3d(torch.randn([4, 4, 4, 4, 4]), 2, stride=2, return_indices=True)
        self.assertEqual(F.max_unpool3d(output, indices, 2), F.max_unpool3d(output, indices, 2, stride=2))

    def test_dirac_properties(self):
        for dims in [3, 4, 5]:
            input_tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=5)
            init.dirac_(input_tensor)

            c_out, c_in = input_tensor.size(0), input_tensor.size(1)
            min_d = min(c_out, c_in)
            # Check number of nonzeros is equivalent to smallest dim
            assert torch.nonzero(input_tensor).size(0) == min_d
            # Check sum of values (can have precision issues, hence assertEqual) is also equivalent
            self.assertEqual(input_tensor.sum(), min_d)

    def test_dirac_identity(self):
        batch, in_c, out_c, size, kernel_size = 8, 3, 4, 5, 3
        # Test 1D
        input_var = torch.randn(batch, in_c, size)
        filter_var = torch.zeros(out_c, in_c, kernel_size)
        init.dirac_(filter_var)
        output_var = F.conv1d(input_var, filter_var)
        input_tensor, output_tensor = input_var.data, output_var.data  # Variables do not support nonzero
        self.assertEqual(input_tensor[:, :, 1:-1], output_tensor[:, :in_c, :])  # Assert in_c outputs are preserved
        assert torch.nonzero(output_tensor[:, in_c:, :]).numel() == 0  # Assert extra outputs are 0

        # Test 2D
        input_var = torch.randn(batch, in_c, size, size)
        filter_var = torch.zeros(out_c, in_c, kernel_size, kernel_size)
        init.dirac_(filter_var)
        output_var = F.conv2d(input_var, filter_var)
        input_tensor, output_tensor = input_var.data, output_var.data
        self.assertEqual(input_tensor[:, :, 1:-1, 1:-1], output_tensor[:, :in_c, :, :])
        assert torch.nonzero(output_tensor[:, in_c:, :, :]).numel() == 0

        # Test 3D
        input_var = torch.randn(batch, in_c, size, size, size)
        filter_var = torch.zeros(out_c, in_c, kernel_size, kernel_size, kernel_size)
        init.dirac_(filter_var)
        output_var = F.conv3d(input_var, filter_var)
        input_tensor, output_tensor = input_var.data, output_var.data
        self.assertEqual(input_tensor[:, :, 1:-1, 1:-1, 1:-1], output_tensor[:, :in_c, :, :])
        assert torch.nonzero(output_tensor[:, in_c:, :, :, :]).numel() == 0

    def test_dirac_only_works_on_3_4_5d_inputs(self):
        for dims in [1, 2, 6]:
            with self.assertRaises(ValueError):
                tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=3)
                init.dirac_(tensor)

    def test_xavier_uniform_errors_on_inputs_smaller_than_2d(self):
        for dims in [0, 1]:
            tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=1)
            with self.assertRaises(ValueError):
                init.xavier_uniform_(tensor)

    def test_xavier_normal_errors_on_inputs_smaller_than_2d(self):
        for dims in [0, 1]:
            tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=1)
            with self.assertRaises(ValueError):
                init.xavier_normal_(tensor)

    @unittest.skipIf(not TEST_SCIPY, "Scipy not found.")
    def test_xavier_uniform(self):
        for use_gain in [True, False]:
            for dims in [2, 4]:
                input_tensor = self._create_random_nd_tensor(dims, size_min=20, size_max=25)
                gain = 1

                if use_gain:
                    gain = self._random_float(0.1, 2)
                    init.xavier_uniform_(input_tensor, gain=gain)
                else:
                    init.xavier_uniform_(input_tensor)

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
        for use_gain in [True, False]:
            for dims in [2, 4]:
                input_tensor = self._create_random_nd_tensor(dims, size_min=20, size_max=25)
                gain = 1

                if use_gain:
                    gain = self._random_float(0.1, 2)
                    init.xavier_normal_(input_tensor, gain=gain)
                else:
                    init.xavier_normal_(input_tensor)

                fan_in = input_tensor.size(1)
                fan_out = input_tensor.size(0)
                if input_tensor.dim() > 2:
                    fan_in *= input_tensor[0, 0].numel()
                    fan_out *= input_tensor[0, 0].numel()

                expected_std = gain * math.sqrt(2.0 / (fan_in + fan_out))
                assert self._is_normal(input_tensor, 0, expected_std)

    def test_kaiming_uniform_errors_on_inputs_smaller_than_2d(self):
        for dims in [0, 1]:
            with self.assertRaises(ValueError):
                tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=1)
                init.kaiming_uniform_(tensor)

    def test_kaiming_normal_errors_on_inputs_smaller_than_2d(self):
        for dims in [0, 1]:
            with self.assertRaises(ValueError):
                tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=1)
                init.kaiming_normal_(tensor)

    @unittest.skipIf(not TEST_SCIPY, "Scipy not found.")
    def test_kaiming_uniform(self):
        for use_a in [True, False]:
            for dims in [2, 4]:
                for mode in ['fan_in', 'fan_out']:
                    input_tensor = self._create_random_nd_tensor(dims, size_min=20, size_max=25)
                    if use_a:
                        a = self._random_float(0.1, 2)
                        init.kaiming_uniform_(input_tensor, a=a, mode=mode)
                    else:
                        a = 0
                        init.kaiming_uniform_(input_tensor, mode=mode)

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
        for use_a in [True, False]:
            for dims in [2, 4]:
                for mode in ['fan_in', 'fan_out']:
                    input_tensor = self._create_random_nd_tensor(dims, size_min=20, size_max=25)
                    if use_a:
                        a = self._random_float(0.1, 2)
                        init.kaiming_normal_(input_tensor, a=a, mode=mode)
                    else:
                        a = 0
                        init.kaiming_normal_(input_tensor, mode=mode)

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
        for dims in [1, 3]:
            with self.assertRaises(ValueError):
                sparsity = self._random_float(0.1, 0.9)
                tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=3)
                init.sparse_(tensor, sparsity)

    @unittest.skipIf(not TEST_SCIPY, "Scipy not found.")
    def test_sparse_default_std(self):
        for use_random_std in [True, False]:
            input_tensor = self._create_random_nd_tensor(2, size_min=30, size_max=35)
            rows, cols = input_tensor.size(0), input_tensor.size(1)
            sparsity = self._random_float(0.1, 0.2)

            std = 0.01  # default std
            if use_random_std:
                std = self._random_float(0.01, 0.2)
                init.sparse_(input_tensor, sparsity=sparsity, std=std)
            else:
                init.sparse_(input_tensor, sparsity=sparsity)

            for col_idx in range(input_tensor.size(1)):
                column = input_tensor[:, col_idx]
                assert column[column == 0].nelement() >= math.ceil(sparsity * rows)

            assert self._is_normal(input_tensor[input_tensor != 0], 0, std)

    @skipIfNoLapack
    def test_orthogonal(self):
        for use_gain in [True, False]:
            for tensor_size in [[3, 4], [4, 3], [20, 2, 3, 4], [2, 3, 4, 5]]:
                input_tensor = torch.zeros(tensor_size)
                gain = 1.0

                if use_gain:
                    gain = self._random_float(0.1, 2)
                    init.orthogonal_(input_tensor, gain=gain)
                else:
                    init.orthogonal_(input_tensor)

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

class TestFusionEval(TestCase):
    @given(X=hu.tensor(shapes=((5, 3, 5, 5),)),
           running_mean=hu.tensor(shapes=(6,)),
           running_var=hu.tensor(shapes=(6,)))
    def test_fuse_module_eval_numerics(self, X, running_mean, running_var):
        inputs, _ = X

        iC, oC = inputs.shape[1], len(running_mean[0])
        inputs = torch.from_numpy(inputs).to(torch.double)
        kernel_size = (3, 3)

        conv_ref = torch.nn.Conv2d(iC, oC, bias=True, kernel_size=kernel_size)
        bn_ref = torch.nn.BatchNorm2d(oC)
        bn_ref.running_mean = torch.from_numpy(running_mean[0]).to(torch.double)
        bn_ref.running_var = torch.from_numpy(running_var[0]).to(torch.double)

        conv_ref.eval()
        bn_ref.eval()

        Y_ref = bn_ref(conv_ref(inputs))
        conv_bn_fused = torch.nn.utils.fusion.fuse_conv_bn_eval(conv_ref,
                                                                bn_ref)
        Y_hat = conv_bn_fused(inputs)

        self.assertEqual(Y_ref, Y_hat, message="Conv+BN fusion results are off")


def add_test(test, decorator=None):
    def add(test_name, fn):
        if hasattr(TestNN, test_name):
            raise RuntimeError('Found two tests with the same name: ' + test_name)
        if decorator is not None:
            fn = decorator(fn)
        setattr(TestNN, test_name, fn)

    test_name = test.get_name()
    add(test_name, lambda self, test=test: test(self))
    cuda_test_name = test_name + '_cuda'
    # With dtype enable, it's good enough to test against three floating types
    kwargs = {}
    if 'extra_args' in get_function_arglist(test.test_cuda):
        kwargs['extra_args'] = test.extra_args

    if 'dtype' in get_function_arglist(test.test_cuda):
        add(cuda_test_name + '_float', lambda self,
            test=test, kwargs=kwargs: test.test_cuda(self, dtype=torch.float, **kwargs))
        add(cuda_test_name + '_double', lambda self,
            test=test, kwargs=kwargs: test.test_cuda(self, dtype=torch.double, **kwargs))

        def test_half(self, test=test, kwargs=kwargs):
            test.test_cuda(self, dtype=torch.half, **kwargs)
        if getattr(test, 'check_half', True):
            add(cuda_test_name + '_half', test_half)
    else:
        add(cuda_test_name, lambda self, test=test, kwargs=kwargs: test.test_cuda(self, **kwargs))

for test_params in module_tests + new_module_tests:
    # TODO: CUDA is not implemented yet
    if 'constructor' not in test_params:
        name = test_params.pop('module_name')
        test_params['constructor'] = getattr(nn, name)
    decorator = test_params.pop('decorator', None)
    test = NewModuleTest(**test_params)
    add_test(test, decorator)
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
        add_test(test, decorator)
    if 'check_with_long_tensor' in test_params:
        fullname = test_params.get('fullname', None)
        if fullname:
            test_params['fullname'] = fullname + '_with_long_tensor'
        else:
            desc = test_params.get('desc', None)
            test_params['desc'] = 'with_long_tensor' if desc is None else desc + '_with_long_tensor'

        def double_equivalent_of_long_tensor(size):
            return torch.randint(-1000, 1000, size=size).double()

        def apply_to_cons(t):
            if t.is_floating_point():
                if isinstance(t, Parameter):
                    return Parameter(double_equivalent_of_long_tensor(t.size()))
                elif isinstance(t, torch.Tensor):
                    return double_equivalent_of_long_tensor(t.size())
            else:
                return t

        def gen_long_tensor_constructor(constructor):
            def long_tensor_constructor(*args, **kwargs):
                cons = constructor(*args, **kwargs)
                cons._apply(apply_to_cons)
                return cons
            long_tensor_constructor.__name__ = constructor.__name__
            return long_tensor_constructor

        def gen_long_tensor_input(input_size):
            def input_func():
                return double_equivalent_of_long_tensor(input_size)
            return input_func

        def reference_fn(i, p, m):
            m._apply(lambda t: t.long())
            input = i.long()
            out = m.forward(input)
            return out

        test_params['constructor'] = gen_long_tensor_constructor(test_params['constructor'])
        test_params['input_fn'] = gen_long_tensor_input(test_params['input_size'])
        test_params['reference_fn'] = reference_fn
        test_params['check_forward_only'] = True
        # Currently we don't support conv2d/conv3d for LongTensor in CUDA
        test_params['test_cuda'] = False
        test = NewModuleTest(**test_params)

        add_test(test, decorator)

for test_params in criterion_tests + new_criterion_tests:
    name = test_params.pop('module_name')
    test_params['constructor'] = getattr(nn, name)
    test = NewCriterionTest(**test_params)
    decorator = test_params.pop('decorator', None)
    add_test(test, decorator)
    if 'check_sum_reduction' in test_params:
        desc = test_params.get('desc', None)
        test_params['desc'] = 'sum_reduction' if desc is None else desc + '_sum_reduction'

        def gen_sum_reduction_constructor(constructor):
            def sum_reduction_constructor(*args, **kwargs):
                cons = constructor(*args, reduction='sum', **kwargs)
                return cons
            sum_reduction_constructor.__name__ = constructor.__name__
            return sum_reduction_constructor

        test_params['constructor'] = gen_sum_reduction_constructor(test_params['constructor'])
        test = NewCriterionTest(**test_params)
        add_test(test, decorator)


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


class _AdaptiveLogSoftmaxWithLoss(nn.AdaptiveLogSoftmaxWithLoss):
    def __call__(self, input):
        t = torch.tensor([0, 1, 4, 8]).to(input.device)
        return nn.AdaptiveLogSoftmaxWithLoss.__call__(self, input, t).output

add_test(NewModuleTest(
    constructor=lambda: _AdaptiveLogSoftmaxWithLoss(16, 10, [2, 6]),
    input_size=(4, 16),
    fullname='AdaptiveLogSoftmax'))


# The following are helpers for TestNN.test_affine_*
if torch.cuda.is_available():
    def device_():
        return ['cpu', 'cuda']
else:
    def device_():
        return ['cpu']


def angle_rad_():
    return [r * math.pi * 2 for r in [0.0, 0.5, 0.25, 0.125, random.random()]]


def axis_vector_():
    t = (random.random(), random.random(), random.random())
    l = sum(x ** 2 for x in t) ** 0.5

    return [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), tuple(x / l for x in t)]


def input_size2d_():
    return [[1, 1, 3, 5], [1, 1, 3, 3], [1, 1, 4, 4], [1, 1, 3, 4]]


def output_size2d_():
    return [[1, 1, 5, 3], [1, 1, 3, 5], [1, 1, 4, 3], [1, 1, 5, 5], [1, 1, 6, 6]]


def input_size2dsq_():
    return [[1, 1, 2, 2], [1, 1, 3, 3], [1, 1, 4, 4], [1, 1, 6, 6]]


def output_size2dsq_():
    return [[1, 1, 2, 2], [1, 1, 3, 3], [1, 1, 4, 4], [1, 1, 5, 5], [1, 1, 6, 6]]


def input_size3d_():
    return [[1, 1, 2, 2, 2], [1, 1, 2, 3, 4], [1, 1, 3, 3, 3], [1, 1, 4, 4, 4], [1, 1, 3, 4, 5]]


def input_size3dsq_():
    return [[1, 1, 2, 2, 2], [1, 1, 3, 3, 3], [1, 1, 4, 4, 4], [1, 1, 6, 6, 6]]


def output_size3dsq_():
    return [[1, 1, 2, 2, 2], [1, 1, 3, 3, 3], [1, 1, 4, 4, 4], [1, 1, 5, 5, 5], [1, 1, 6, 6, 6]]


def output_size3d_():
    return [[1, 1, 2, 2, 2], [1, 1, 3, 3, 3], [1, 1, 3, 4, 5], [1, 1, 4, 3, 2], [1, 1, 5, 5, 5], [1, 1, 6, 6, 6]]


def _buildEquivalentAffineTransforms2d(device, input_size, output_size, angle_rad):
    input_center = [(x - 1) / 2.0 for x in input_size]
    output_center = [(x - 1) / 2.0 for x in output_size]

    s = math.sin(angle_rad)
    c = math.cos(angle_rad)

    intrans_ary = np.array([
        [1, 0, input_center[2]],
        [0, 1, input_center[3]],
        [0, 0, 1],
    ], dtype=np.float64)

    inscale_ary = np.array([
        [input_center[2], 0, 0],
        [0, input_center[3], 0],
        [0, 0, 1],
    ], dtype=np.float64)

    rotation_ary = np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1],
    ], dtype=np.float64)

    outscale_ary = np.array([
        [1.0 / output_center[2], 0, 0],
        [0, 1.0 / output_center[3], 0],
        [0, 0, 1],
    ], dtype=np.float64)

    outtrans_ary = np.array([
        [1, 0, -output_center[2]],
        [0, 1, -output_center[3]],
        [0, 0, 1],
    ], dtype=np.float64)

    reorder_ary = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
    ], dtype=np.float64)

    transform_ary = np.dot(np.dot(np.dot(np.dot(
        intrans_ary,
        inscale_ary),
        rotation_ary.T),
        outscale_ary),
        outtrans_ary)
    grid_ary = np.dot(np.dot(np.dot(reorder_ary, rotation_ary.T), outscale_ary), outtrans_ary)

    transform_tensor = torch.from_numpy((rotation_ary)).to(device, torch.float32)
    transform_tensor = transform_tensor[:2].unsqueeze(0)

    return transform_tensor, transform_ary, grid_ary


def _buildEquivalentAffineTransforms3d(device, input_size, output_size, angle_rad, axis_vector):
    input_center = [(x - 1) / 2.0 for x in input_size]
    output_center = [(x - 1) / 2.0 for x in output_size]

    s = math.sin(angle_rad)
    c = math.cos(angle_rad)
    c1 = 1 - c

    intrans_ary = np.array([
        [1, 0, 0, input_center[2]],
        [0, 1, 0, input_center[3]],
        [0, 0, 1, input_center[4]],
        [0, 0, 0, 1],
    ], dtype=np.float64)

    inscale_ary = np.array([
        [input_center[2], 0, 0, 0],
        [0, input_center[3], 0, 0],
        [0, 0, input_center[4], 0],
        [0, 0, 0, 1],
    ], dtype=np.float64)

    l, m, n = axis_vector
    scipyRotation_ary = np.array([
        [l * l * c1 + c, m * l * c1 - n * s, n * l * c1 + m * s, 0],
        [l * m * c1 + n * s, m * m * c1 + c, n * m * c1 - l * s, 0],
        [l * n * c1 - m * s, m * n * c1 + l * s, n * n * c1 + c, 0],
        [0, 0, 0, 1],
    ], dtype=np.float64)

    z, y, x = axis_vector
    torchRotation_ary = np.array([
        [x * x * c1 + c, y * x * c1 - z * s, z * x * c1 + y * s, 0],
        [x * y * c1 + z * s, y * y * c1 + c, z * y * c1 - x * s, 0],
        [x * z * c1 - y * s, y * z * c1 + x * s, z * z * c1 + c, 0],
        [0, 0, 0, 1],
    ], dtype=np.float64)

    outscale_ary = np.array([
        [1.0 / output_center[2], 0, 0, 0],
        [0, 1.0 / output_center[3], 0, 0],
        [0, 0, 1.0 / output_center[4], 0],
        [0, 0, 0, 1],
    ], dtype=np.float64)

    outtrans_ary = np.array([
        [1, 0, 0, -output_center[2]],
        [0, 1, 0, -output_center[3]],
        [0, 0, 1, -output_center[4]],
        [0, 0, 0, 1],
    ], dtype=np.float64)

    reorder_ary = np.array([
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
    ], dtype=np.float64)

    transform_ary = np.dot(np.dot(np.dot(np.dot(
        intrans_ary,
        inscale_ary),
        np.linalg.inv(scipyRotation_ary)),
        outscale_ary),
        outtrans_ary)
    grid_ary = np.dot(np.dot(np.dot(reorder_ary, np.linalg.inv(scipyRotation_ary)), outscale_ary), outtrans_ary)

    transform_tensor = torch.from_numpy((torchRotation_ary)).to(device, torch.float32)
    transform_tensor = transform_tensor[:3].unsqueeze(0)

    return transform_tensor, transform_ary, grid_ary
# end TestNN.test_affine_* helpers


class TestNNDeviceType(NNTestCase):
    def _test_dropout(self, cls, device, input):
        p = 0.2
        input = input.to(device).fill_(1 - p)

        module = cls(p)
        input_var = input.clone().requires_grad_()
        output = module(input_var)
        self.assertLess(abs(output.data.mean() - (1 - p)), 0.05)
        output.backward(input)
        self.assertLess(abs(input_var.grad.data.mean() - (1 - p)), 0.05)

        module = cls(p, True)
        input_var = input.clone().requires_grad_()
        output = module(input_var + 0)
        self.assertLess(abs(output.data.mean() - (1 - p)), 0.05)
        output.backward(input)
        self.assertLess(abs(input_var.grad.data.mean() - (1 - p)), 0.05)

        # check eval mode doesn't change anything
        for inplace in [True, False]:
            module = cls(p, inplace).eval()
            self.assertEqual(input, module(input))

        # Check that these don't raise errors
        module.__repr__()
        str(module)

    def _test_InstanceNorm_general(self, cls, input, device, dtype=torch.float):
        # default case track_running_stats=False
        b, c = input.size(0), input.size(1)
        input_var = input.to(device=device, dtype=dtype).requires_grad_()

        IN = cls(c, eps=0).to(device, dtype)

        output = IN(input_var)
        out_reshaped = output.view(b * c, -1)

        mean = out_reshaped.mean(1)
        var = out_reshaped.var(1, unbiased=False)

        self.assertAlmostEqual(torch.abs(mean.data).mean(), 0, delta=1e-5)
        self.assertAlmostEqual(torch.abs(var.data).mean(), 1, delta=1e-5)

        # check that eval mode doesn't change behavior
        grad_out = torch.randn_like(output)
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
        IN = cls(c, momentum=1, eps=0, track_running_stats=True).to(device, dtype)

        output = IN(input_var)

        input_reshaped = input_var.transpose(1, 0).reshape(c, -1)
        mean = input_reshaped.mean(1)

        input_reshaped = input_var.transpose(1, 0).reshape(c, b, -1)
        var = input_reshaped.var(2, unbiased=True)[:, :]

        self.assertAlmostEqual(torch.abs(mean.data - IN.running_mean).mean(), 0, delta=1e-5)
        self.assertAlmostEqual(torch.abs(var.data.mean(1) - IN.running_var).mean(), 0, delta=1e-5)

        # in eval mode, adding X * std to a channel in input should make the
        # corresponding channel in output have mean X
        IN.eval()
        delta = IN.running_var.sqrt() * torch.arange(c, device=device, dtype=dtype)
        delta = delta.view(-1, *[1 for _ in range(2, input.dim())])
        output = IN(input_var + delta)
        self.assertEqual(output.transpose(0, 1).reshape(c, -1).mean(1), torch.arange(c))

    def _test_InstanceNorm_cuda_half(self, cls, input, device):
        # THNN
        input = input.to(device=device, dtype=torch.half).random_(1, 10).requires_grad_(True)
        m = cls(input.size(1), affine=True, track_running_stats=True).to(device, torch.half)
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

    def _test_LayerNorm_general(self, device, dtype=torch.float):
        for i in range(2, 6):
            shape = torch.randint(3, 6, (i,), dtype=torch.long).tolist()
            x = torch.empty(*shape, device=device, dtype=dtype).uniform_(0, 10)
            normalized_ndim = random.randint(1, i - 1)  # inclusive
            normalized_shape = shape[-normalized_ndim:]
            unnormalized_shape = shape[:-normalized_ndim]

            # test that LN normalizes to mean 0 and stddev 1
            ln = nn.LayerNorm(normalized_shape, eps=0).to(device, dtype)
            ln.weight.data.fill_(1)
            ln.bias.data.fill_(0)
            output = ln(x)
            out_reshaped = output.view(*(unnormalized_shape + [-1]))
            mean = out_reshaped.mean(-1)
            var = out_reshaped.var(-1, unbiased=False)
            self.assertAlmostEqual(torch.abs(mean.data).mean(), 0, delta=1e-5)
            self.assertAlmostEqual(torch.abs(var.data).mean(), 1, delta=1e-5)

            # test that LN applies weight and bias correctly
            scale, bias = torch.empty(2).uniform_(0.2, 2).tolist()
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
            input = torch.empty(input_shape, device=device, dtype=dtype).uniform_(0, 10)
            self.assertRaises(RuntimeError, lambda: ln(input))

    def _test_LayerNorm_cuda_half(self, device):
        input = torch.empty(2, 3, 3, 2, device=device, dtype=torch.half).random_(1, 10).requires_grad_(True)
        m = nn.LayerNorm([3, 2]).to(device, torch.half)
        output = m(input)
        output.sum().backward()
        self.assertEqual(output.type(), input.type())

    def _test_GroupNorm_general(self, device, dtype=torch.float):
        good_shape_g = {
            (1, 2, 3, 4): 2,
            (2, 3, 10): 3,
            (3, 1, 1, 1, 2): 1,
            (2, 6, 4, 2, 2): 3,
        }
        for shape, g in good_shape_g.items():
            x = torch.empty(*shape, device=device, dtype=dtype).uniform_(0, 10)
            b = shape[0]
            c = shape[1]

            # test that GN normalizes to mean 0 and stddev 1
            gn = nn.GroupNorm(g, c, eps=0).to(device, dtype)
            gn.weight.data.fill_(1)
            gn.bias.data.fill_(0)
            output = gn(x)
            out_reshaped = output.view(b, g, -1)
            mean = out_reshaped.mean(-1)
            var = out_reshaped.var(-1, unbiased=False)
            self.assertAlmostEqual(torch.abs(mean).mean(), 0, delta=1e-5)
            self.assertAlmostEqual(torch.abs(var).mean(), 1, delta=1e-5)

            # test that GN applies weight and bias correctly
            scale = torch.empty(c, device=device, dtype=dtype).uniform_(0.2, 2)
            bias = torch.empty(c, device=device, dtype=dtype).uniform_(0.2, 2)
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
            input = torch.empty(*shape, device=device, dtype=dtype).uniform_(0, 10)
            self.assertRaises(RuntimeError, lambda: gn(input))

    def _test_GroupNorm_cuda_half(self):
        input = torch.zeros(2, 4, 3, 2, requires_grad=True).cuda().half().random_(1, 10)
        m = nn.GroupNorm(2, 4).to("cuda", torch.half)
        output = m(input)
        output.sum().backward()
        self.assertEqual(output.type(), input.type())

    def _test_module_empty_input(self, module, inp):
        inp.requires_grad_(True)
        out = module(inp)
        gO = torch.rand_like(out)
        out.backward(gO)
        self.assertEqual(out.size(), inp.size())
        for p in module.parameters():
            if p.requires_grad and p.grad is not None:
                self.assertEqual(p.grad, torch.zeros_like(p.grad))

    def test_Dropout(self, device):
        input = torch.Tensor(1000)
        self._test_dropout(nn.Dropout, device, input)

    def test_Dropout2d(self, device):
        b = random.randint(1, 5)
        w = random.randint(1, 5)
        h = random.randint(1, 5)
        num_features = 1000
        input = torch.Tensor(num_features, b, w, h)
        self._test_dropout(nn.Dropout2d, device, input)

    def test_Dropout3d(self, device):
        b = random.randint(1, 5)
        w = random.randint(1, 5)
        h = random.randint(1, 5)
        d = random.randint(1, 2)
        num_features = 1000
        input = torch.Tensor(num_features, b, d, w, h)
        self._test_dropout(nn.Dropout3d, device, input)

    def test_InstanceNorm1d_general(self, device):
        b = random.randint(3, 5)
        c = random.randint(3, 5)
        d = random.randint(8, 10)

        input = torch.rand(b, c, d)
        self._test_InstanceNorm_general(nn.InstanceNorm1d, input, device)

        if self.device_type == 'cuda':
            self._test_InstanceNorm_cuda_half(nn.InstanceNorm1d, input, device)

    def test_InstanceNorm2d_general(self, device):
        b = random.randint(3, 5)
        c = random.randint(3, 5)
        w = random.randint(3, 6)
        h = random.randint(6, 8)

        input = torch.rand(b, c, h, w)
        self._test_InstanceNorm_general(nn.InstanceNorm2d, input, device)

        if self.device_type == 'cuda':
            self._test_InstanceNorm_cuda_half(nn.InstanceNorm2d, input, device)

    def test_InstanceNorm3d_general(self, device):
        b = random.randint(3, 5)
        c = random.randint(3, 5)
        w = random.randint(2, 5)
        h = random.randint(2, 5)
        d = random.randint(2, 5)

        input = torch.rand(b, c, h, w, d)
        self._test_InstanceNorm_general(nn.InstanceNorm3d, input, device)

        if self.device_type == 'cuda':
            self._test_InstanceNorm_cuda_half(nn.InstanceNorm3d, input, device)

    def test_LayerNorm_general(self, device):
        self._test_LayerNorm_general(device)

        if self.device_type == 'cuda':
            self._test_LayerNorm_cuda_half(device)

    def test_GroupNorm_general(self, device):
        self._test_GroupNorm_general(device)

        if self.device_type == 'cuda':
            self._test_GroupNorm_cuda_half()

    def test_BatchNorm_empty(self, device):
        mod = torch.nn.BatchNorm2d(3).to(device)
        inp = torch.randn(0, 3, 2, 2, device=device)
        self._test_module_empty_input(mod, inp)
        if self.device_type == 'cuda' and self.has_cudnn():
            with torch.backends.cudnn.flags(enabled=False):
                self._test_module_empty_input(mod, inp)

    def test_one_hot(self, device):
        with self.assertRaises(RuntimeError):
            torch.nn.functional.one_hot(torch.tensor([3, 4, -1, 0], device=device), -1)

        with self.assertRaises(RuntimeError):
            torch.nn.functional.one_hot(torch.tensor([3, 4, 1, 0], device=device), 3)

        t = torch.nn.functional.one_hot(torch.tensor([3, 4, 1, 0], device=device))
        expected = torch.tensor([[0, 0, 0, 1, 0],
                                 [0, 0, 0, 0, 1],
                                 [0, 1, 0, 0, 0],
                                 [1, 0, 0, 0, 0]], device=device)
        self.assertEqual(t, expected)

        t = torch.nn.functional.one_hot(torch.tensor([3, 4, 1, 0], device=device), -1)
        expected = torch.tensor([[0, 0, 0, 1, 0],
                                 [0, 0, 0, 0, 1],
                                 [0, 1, 0, 0, 0],
                                 [1, 0, 0, 0, 0]], device=device)
        self.assertEqual(t, expected)

        t = torch.nn.functional.one_hot(torch.tensor([3, 4, 1, 0], device=device), 6)
        expected = torch.tensor([[0, 0, 0, 1, 0, 0],
                                 [0, 0, 0, 0, 1, 0],
                                 [0, 1, 0, 0, 0, 0],
                                 [1, 0, 0, 0, 0, 0]], device=device)
        self.assertEqual(t, expected)

        t = torch.nn.functional.one_hot(torch.tensor([[3, 4], [1, 0]], device=device))
        expected = torch.tensor([[[0, 0, 0, 1, 0],
                                  [0, 0, 0, 0, 1]],
                                 [[0, 1, 0, 0, 0],
                                  [1, 0, 0, 0, 0]]], device=device)
        self.assertEqual(t, expected)

        t = torch.nn.functional.one_hot(torch.tensor(4, device=device))
        expected = torch.tensor([0, 0, 0, 0, 1], device=device)
        self.assertEqual(t, expected)

        t = torch.nn.functional.one_hot(torch.empty([4, 0], dtype=torch.long, device=device), 100)
        expected = torch.empty([4, 0, 100])
        self.assertEqual(t, expected)

        with self.assertRaises(RuntimeError):
            torch.nn.functional.one_hot(torch.empty([4, 0], dtype=torch.long, device=device))

        with self.assertRaises(RuntimeError):
            torch.nn.functional.one_hot(torch.tensor([3, 4, 1, 0], device=device), -2)

    def test_nn_scalars(self, device):
        # One off tests to ensure scalars from nn.yaml are properly applied
        def verify_scalars(input, output):
            if input.dim() == 0:
                self.assertEqual((), output.shape)
            else:
                self.assertNotEqual((), output.shape)
            output.sum().backward()
            self.assertEqual(input.shape, input.grad.shape)

        for input_shape in [(5, 6), ()]:
            for module in [torch.nn.ELU, torch.nn.Hardtanh, torch.nn.LeakyReLU, torch.nn.LogSigmoid,
                           torch.nn.RReLU, torch.nn.Softshrink, torch.nn.Softplus, torch.nn.Sigmoid,
                           torch.nn.Tanh]:
                input = torch.randn(input_shape, device=device, requires_grad=True)
                m = module()
                output = m(input)
                verify_scalars(input, output)

    def test_nn_scalars_reductions(self, device):
        # One off tests to ensure scalars from nn.yaml are properly applied
        def verify_reduction_scalars(input, reduction, output):
            if reduction != 'none' or input.dim() == 0:
                self.assertEqual((), output.shape)
            else:
                self.assertNotEqual((), output.shape)
            output.sum().backward()
            self.assertEqual(input.shape, input.grad.shape)

        for input_shape in [(5, 6), ()]:
            for reduction in ['none', 'mean', 'sum']:
                for module in [torch.nn.BCELoss, torch.nn.L1Loss, torch.nn.MSELoss,
                               torch.nn.SmoothL1Loss, torch.nn.SoftMarginLoss]:
                    input = torch.randn(input_shape, device=device, requires_grad=True)
                    target = torch.empty(input_shape, device=device).random_(2)
                    sigmoid = nn.Sigmoid()

                    input = torch.randn(input_shape, device=device, requires_grad=True)
                    m = module(reduction=reduction)
                    output = m(sigmoid(input), target)
                    verify_reduction_scalars(input, reduction, output)

    # We don't want to make propagating NaN a hard requirement on ops, but for
    # these easy ones, we should make them do so.
    def test_nonlinearity_propagate_nan(self, device):
        def test(nonlinearity, *args, **kwargs):
            x = torch.tensor([nan], device=device)
            fn = getattr(F, nonlinearity)
            try:
                self.assertTrue(math.isnan(fn(x, *args, **kwargs).item()))
            except Exception as e:
                if 'not implemented' not in str(e):
                    raise

        test('relu')
        test('relu', inplace=True)
        test('relu6')
        test('elu')
        test('selu')
        test('celu')
        test('rrelu')
        test('rrelu', inplace=True)
        test('hardtanh')
        test('tanh')
        test('sigmoid')
        test('logsigmoid')
        test('hardshrink')
        test('tanhshrink')
        test('softsign')
        test('softmin', 0)
        test('softmax', 0)
        test('log_softmax', 0)
        test('leaky_relu', 0.2)
        test('threshold', 3, 2)
        test('threshold', 3, 2, inplace=True)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_max_pool2d_nhwc(self, device):
        input = torch.randint(1, 10, (4, 8, 8, 8), dtype=torch.float32, device="cuda")
        input = input.contiguous(memory_format=torch.channels_last).requires_grad_()
        grad = torch.randint(1, 10, (4, 8, 1, 1), dtype=torch.float32, device="cuda")
        pool = torch.nn.MaxPool2d((7, 7)).cuda()

        ref_input = input.detach().clone().contiguous().requires_grad_(True)
        ref_grad = grad.detach().clone().contiguous()
        ref_pool = torch.nn.MaxPool2d((7, 7)).cuda()

        out = pool(input)
        out.backward(grad)
        ref_out = ref_pool(ref_input)
        ref_out.backward(ref_grad)

        self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))
        self.assertTrue(ref_out.is_contiguous())
        self.assertTrue(torch.allclose(out, ref_out))
        self.assertTrue(torch.allclose(input.grad, ref_input.grad))

    def test_embedding_dense_grad(self, device):
        embd = nn.Embedding(20, 20).to(device)
        weight = embd.weight

        def fn_wrapper(device):
            def fn(weight):
                inp = torch.tensor([[0, 1, 1, 2], [3, 5, 7, 11]], dtype=torch.long).to(device)
                return torch.nn.functional.embedding(inp, weight)
            return fn

        fn = fn_wrapper(device)
        _assertGradAndGradgradChecks(self, fn, (weight, ))

    @dtypesIfCUDA(torch.float16, torch.float64)
    @dtypes(torch.float64)
    def test_embedding_backward(self, device, dtype):
        embedding = nn.Embedding(10, 3, sparse=True)
        tensor = torch.tensor([[7, 1, 3]])
        ones = torch.tensor(1.).expand(3, 3)
        tensorTwice = tensor.repeat(1, 2)
        onesTwice = torch.cat((ones, ones))

        embedding = embedding.to(dtype=dtype).to(device)
        tensor = tensor.to(device)
        ones = ones.to(device)
        tensorTwice = tensorTwice.to(device)
        onesTwice = onesTwice.to(device)

        embedding.zero_grad()
        embedding(tensor[0]).sum().backward()
        self.assertEqual(embedding.weight.grad._indices(), tensor)
        self.assertEqual(embedding.weight.grad._values(), ones)

        embedding.zero_grad()
        embedding(tensor[0]).sum().backward()
        embedding(tensor[0]).sum().backward()
        self.assertEqual(embedding.weight.grad._indices(), tensorTwice)
        self.assertEqual(embedding.weight.grad._values(), onesTwice)

        embedding.zero_grad()
        embedding(tensor[0]).sum().backward()
        tensor[0, 0] = 8
        embedding(tensor[0]).sum().backward()
        tensorTwice[0, 3] = 8
        self.assertEqual(embedding.weight.grad._indices(), tensorTwice)
        self.assertEqual(embedding.weight.grad._values(), onesTwice)

    def test_embedding_padding_idx(self, device):
        embedding = nn.Embedding(10, 20, padding_idx=0).to(device)
        input = torch.tensor([[0, 2, 4, 5], [4, 3, 0, 9]], dtype=torch.long).to(device)
        output = embedding(input)
        self.assertEqual(output[0][0].sum(), 0)
        self.assertEqual(output[1][2].sum(), 0)

        embedding = nn.Embedding(10, 20, padding_idx=0, sparse=True).to(device)
        input = torch.tensor([[0, 2, 4, 5], [4, 3, 0, 9]], dtype=torch.long).to(device)
        output = embedding(input)
        self.assertEqual(output[0][0].sum(), 0)
        self.assertEqual(output[1][2].sum(), 0)

        # negative indexing check for padding_idx
        # padding_idx=-2, num_embeddings=10 ==> index 8 padded
        embedding = nn.Embedding(10, 20, padding_idx=-2).to(device)
        input = torch.tensor([[0, 2, 8, 5], [4, 8, 0, 9]], dtype=torch.long).to(device)
        output = embedding(input)
        self.assertEqual(output[0][2].sum(), 0)
        self.assertEqual(output[1][1].sum(), 0)

        embedding = nn.Embedding(10, 20, padding_idx=-2, sparse=True).to(device)
        input = torch.tensor([[0, 2, 8, 5], [4, 8, 0, 9]], dtype=torch.long).to(device)
        output = embedding(input)
        self.assertEqual(output[0][2].sum(), 0)
        self.assertEqual(output[1][1].sum(), 0)

        # out of bounds check for padding_idx
        self.assertRaises(AssertionError, nn.Embedding, num_embeddings=10, embedding_dim=20, padding_idx=25)
        self.assertRaises(AssertionError, nn.Embedding, num_embeddings=10, embedding_dim=20, padding_idx=-25)

        # test backward when input contains padding_idx
        padding_idx = 0
        embedding = nn.Embedding(5, 2, padding_idx=padding_idx).to(device)
        for n in (1, 2, 1000):  # Need large N to trigger all the methods we have implemented
            for other_indices in ([], [1, 3], [2]):
                indices = torch.tensor(other_indices + [padding_idx] * n, dtype=torch.long).to(device)
                pre = embedding.weight[padding_idx].clone()
                embedding(indices).sum().backward()
                after = (embedding.weight + embedding.weight.grad)[padding_idx]
                embedding.zero_grad()
                self.assertEqual(after, pre)

                # test double backward
                emb_sum = embedding(indices).sum()
                emb_grad = torch.autograd.grad(outputs=emb_sum, inputs=list(embedding.parameters()), retain_graph=True)
                scalar = emb_grad[0].sum() + emb_sum
                scalar.backward()
                after = (embedding.weight + embedding.weight.grad)[padding_idx]
                embedding.zero_grad()
                self.assertEqual(after, pre)

    @dtypesIfCUDA(torch.half, torch.float)
    @dtypes(torch.float)
    def test_softmax_backward(self, device, dtype):
        sizes = [(0, 10), (32, 20), (10, 0)]
        for fn in [F.softmax, F.log_softmax]:
            for size in sizes:
                input = torch.rand(size, device=device, dtype=dtype, requires_grad=True)
                for dim in [0, 1]:
                    output = fn(input, dtype=torch.float, dim=dim).sum()
                    grad_input, = torch.autograd.grad(output, input, create_graph=True)
                    grad_input.sum().backward()

    @skipIfRocm
    @unittest.skipIf(not TEST_LARGE_TENSOR, "not enough memory to run test")
    def test_conv_large_nosplit(self, device):
        # Here we just test the convolution correctly route to the fallback implementation
        # that is, it does not crash. The correctness of fallback implementation should be
        # covered in other tests
        dtype = torch.half if self.device_type == 'cuda' else torch.float
        conv1 = nn.Conv2d(2, 2, 8, 8).to(device).to(dtype)
        input_large = torch.randn(1, 2, 1024, 1024 * 1024, dtype=dtype, device=device)
        conv1(input_large)
        conv2 = torch.nn.Conv2d(1, 1024, 1, 1).to(device).to(dtype)
        input_large = torch.randn(1, 1, 2048, 1024 , dtype=dtype, device=device)
        conv2(input_large)

    def test_conv_noncontig_weights(self, device):
        for dim in (1, 2, 3):
            for grouped in (False, True):
                nc = 3
                groups = 3 if grouped else 1
                w = torch.randn([3] * dim, device=device)
                w = w.expand([nc, int(nc / groups)] + list(w.shape))
                w = w.detach().requires_grad_()
                x = torch.randn([1, nc] + ([5] * dim), device=device, requires_grad=True)
                y = getattr(F, 'conv{}d'.format(dim))(x, w, groups=groups)
                y.sum().backward()
                y = getattr(F, 'conv_transpose{}d'.format(dim))(x, w, groups=groups)
                y.sum().backward()

    def test_conv_noncontig_weights_and_bias(self, device):
        # need floats to exercise https://github.com/pytorch/pytorch/issues/16018
        for bias in [True, False]:
            conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                              bias=bias).to(device, torch.float)

            input_nc = torch.randn((1, 3, 224, 224, 2), device=device, dtype=torch.float)[:, :, :, :, 1]
            input_c = input_nc.contiguous()

            weight_nc = torch.randn((64, 3, 7, 7, 2), device=device, dtype=torch.float)[:, :, :, :, 1]
            conv1.weight = nn.Parameter(weight_nc)
            weight_c = conv1.weight.contiguous()

            if bias:
                bias_nc = torch.randn((64, 2), device=device, dtype=torch.float)[:, 1]
                conv1.bias = nn.Parameter(bias_nc)
                bias_c = conv1.bias.contiguous()

            out1 = conv1(input_nc)
            conv1.weight = nn.Parameter(weight_c)
            if bias:
                conv1.bias = nn.Parameter(bias_c)
            out2 = conv1(input_c)
            self.assertEqual(out1, out2)

    @unittest.skipIf(not TEST_LARGE_TENSOR, "not enough memory to run test")
    @unittest.skipIf(sys.platform == "win32", "See https://github.com/pytorch/pytorch/issues/31650")
    def test_conv_transposed_large(self, device):
        dtype = torch.half if self.device_type == 'cuda' else torch.float
        conv = nn.ConvTranspose2d(1, 1, 1, 1, bias=False).to(device).to(dtype)
        input_large = torch.randn(4096, 1, 512, 1024, dtype=dtype, device=device)
        # forward
        ret = conv(input_large)
        maxdiff0 = (ret.narrow(0, 0, 1024) - conv(input_large.narrow(0, 0, 1024))).abs_().max().item()
        maxdiff1 = (ret.narrow(0, 1024, 1024) - conv(input_large.narrow(0, 1024, 1024))).abs_().max().item()
        maxdiff2 = (ret.narrow(0, 2048, 1024) - conv(input_large.narrow(0, 2048, 1024))).abs_().max().item()
        maxdiff3 = (ret.narrow(0, 3072, 1024) - conv(input_large.narrow(0, 3072, 1024))).abs_().max().item()
        self.assertEqual(maxdiff0, 0)
        self.assertEqual(maxdiff1, 0)
        self.assertEqual(maxdiff2, 0)
        self.assertEqual(maxdiff3, 0)

    @skipIfRocm
    @unittest.skipIf(not TEST_LARGE_TENSOR, "not enough memory to run test")
    def test_conv_large(self, device):
        dtype = torch.half if self.device_type == 'cuda' else torch.float
        conv = nn.Conv2d(2, 2, 8, 8, bias=False).to(device).to(dtype)
        input_large = torch.randn(4097, 2, 512, 512, dtype=dtype, device=device)
        # forward
        ret = conv(input_large)
        self.assertEqual(ret[:2048], conv(input_large[:2048]))
        self.assertEqual(ret[2048:4096], conv(input_large[2048:4096]))
        self.assertEqual(ret[4096:], conv(input_large[4096:]))

        # backward
        conv.zero_grad()
        # When computing the backward, we are using the `max(dim=1)`` to create
        # some sparsity. Without this sparsity, the rounding error would be
        # too large (as large as 1e-5) to satisfy the creterion (1e-6) of `assertEqual`
        ret.view(4097, -1).max(dim=1).values.sum().backward()
        del ret
        grad1 = conv.weight.grad.detach().clone()
        conv.zero_grad()
        conv(input_large[:2048]).view(2048, -1).max(dim=1).values.sum().backward()
        conv(input_large[2048:4096]).view(2048, -1).max(dim=1).values.sum().backward()
        conv(input_large[4096:]).view(1, -1).max(dim=1).values.sum().backward()
        grad2 = conv.weight.grad.detach().clone()
        # gradients are at the order of hundreds, we need to scale it to
        # the order of one so that we can compare
        scale = 1 / grad1.abs().mean()
        grad1 = grad1 * scale
        grad2 = grad2 * scale
        self.assertEqual(grad1, grad2)

    def _test_gumbel_softmax_st_shapes(self, device, dtype, shape, dim, count_expected):
        logits = torch.randn(shape, dtype=torch.float, device=device)
        logits = logits.to(dtype)

        y_draw = F.gumbel_softmax(logits, hard=True, dim=dim)

        # All values positive
        self.assertGreaterEqual(y_draw.min(), 0)
        # Shape unchanged
        self.assertTrue(y_draw.shape == logits.shape)
        # One choice per draw
        self.assertEqual(y_draw.sum(), count_expected, prec=torch.finfo(y_draw.dtype).eps)

    def _test_gumbel_softmax_straight_through(self, device, dtype):
        num_draws = 100

        logits = torch.tensor([[0.2, 0.8, 0.1]], device=device)
        logits = logits.reshape([1, 3])
        logits = logits.to(dtype).requires_grad_()
        probs = logits.softmax(dim=-1)

        counts = torch.zeros_like(logits)
        for _ in range(num_draws):
            y_draw = F.gumbel_softmax(logits, hard=True)
            counts = counts + y_draw

        # All values positive
        self.assertGreaterEqual(y_draw.min(), 0)
        # Each experiment should result in 1 draw.
        self.assertEqual(counts.sum(), num_draws, prec=torch.finfo(counts.dtype).eps)

        # check results is asymptotically as expected.
        expected = probs * num_draws
        # ~z is approximately N(0,1) for unbiased count
        z = (counts - expected) / (expected * (1 - probs)).sqrt()
        # A (lazy) approximate 99% two-sided test:
        # occurs with prob alpha~>=0.01 if unbiased
        self.assertLess(z.abs().max().item(), 2.58)

    def _test_gumbel_softmax_grad(self, device, dtype):
        # "hard" and "not hard" should propagate same gradient.
        logits_soft = torch.zeros(10, 10, dtype=dtype, device=device, requires_grad=True)
        logits_hard = torch.zeros(10, 10, dtype=dtype, device=device, requires_grad=True)

        seed = torch.random.get_rng_state()
        y_soft = F.gumbel_softmax(logits_soft, hard=False)
        torch.random.set_rng_state(seed)
        y_hard = F.gumbel_softmax(logits_hard, hard=True)

        y_soft.sum().backward()
        y_hard.sum().backward()

        # 2eps = 1x addition + 1x subtraction.
        tol = 2 * torch.finfo(dtype).eps
        self.assertAlmostEqual(logits_soft.grad, logits_hard.grad, delta=tol)

    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    @dtypes(torch.float, torch.double)
    def test_gumbel_softmax(self, device, dtype):
        self._test_gumbel_softmax_st_shapes(device, dtype, shape=[5], dim=0, count_expected=1)
        self._test_gumbel_softmax_st_shapes(device, dtype, shape=[5], dim=-1, count_expected=1)
        self._test_gumbel_softmax_st_shapes(device, dtype, shape=[5, 4], dim=1, count_expected=5)
        self._test_gumbel_softmax_st_shapes(device, dtype, shape=[5, 4, 3], dim=1, count_expected=5 * 3)
        self._test_gumbel_softmax_st_shapes(device, dtype, shape=[5, 4, 3], dim=-1, count_expected=5 * 4)
        self._test_gumbel_softmax_straight_through(device, dtype)
        self._test_gumbel_softmax_grad(device, dtype)

    def _test_rnn_retain_variables(self, device, dtype):
        rnns = [nn.LSTM(10, 20, num_layers=2).to(device, dtype),
                nn.GRU(10, 20, num_layers=2).to(device, dtype),
                nn.RNN(10, 20, num_layers=2).to(device, dtype)]
        for rnn in rnns:
            input = torch.randn(5, 6, 10, device=device, dtype=dtype, requires_grad=True)
            output = rnn(input)
            output[0].sum().backward(retain_graph=True)
            grads = [input.grad.data.clone()] + [p.grad.data.clone() for p in rnn.parameters()]
            for _ in range(4):
                rnn.zero_grad()
                input.grad.data.zero_()
                output[0].sum().backward(retain_graph=True)
                grads2 = [input.grad.data] + [p.grad.data for p in rnn.parameters()]
                self.assertEqual(grads, grads2)

    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    @dtypes(torch.double)
    def test_rnn_retain_variables(self, device, dtype):
        self._test_rnn_retain_variables(device, dtype)

        if self.device_type == 'cuda' and self.has_cudnn():
            with torch.backends.cudnn.flags(enabled=False):
                self._test_rnn_retain_variables(device, dtype)

    @onlyCUDA
    def test_upsamplingNearest1d_launch_config(self, device):
        m = nn.Upsample(scale_factor=2)
        inp = torch.rand(2**25, 1, 1, device=device)
        out = m(inp)
        inp_ref = inp.cpu()
        out_ref = m(inp_ref)
        self.assertEqual(out_ref, out)

    @onlyCUDA
    def test_upsamplingNearest2d_launch_config(self, device):
        m = nn.Upsample(scale_factor=2)
        inp = torch.rand(2**25, 1, 1, 1, device=device)
        out = m(inp)
        inp_ref = inp.cpu()
        out_ref = m(inp_ref)
        self.assertEqual(out_ref, out)

    @onlyCUDA
    def test_upsamplingNearest3d_launch_config(self, device):
        m = nn.Upsample(scale_factor=2)
        inp = torch.rand(2**25, 1, 1, 1, 1, device=device)
        out = m(inp)
        inp_ref = inp.cpu()
        out_ref = m(inp_ref)
        self.assertEqual(out_ref, out)

    @unittest.expectedFailure
    @skipIfRocm
    @onlyCUDA
    def test_upsamplingNearest2d_launch_fail(self, device):
        m = nn.Upsample(scale_factor=2)
        # launch grid_y == 2**16 (larger than maximum y-dimension limit 65535)
        inp = torch.rand(1, 1, 2**15, 2**8, device=device)
        out = m(inp)

    @onlyCUDA
    @skipCUDAIfCudnnVersionLessThan(7600)
    def test_CTCLoss_cudnn(self, device):
        target_lengths = [30, 25, 20]
        input_lengths = [50, 50, 50]
        targets = torch.randint(1, 15, (sum(target_lengths),), dtype=torch.int)
        log_probs = torch.randn(50, 3, 15, dtype=torch.float, device=device).log_softmax(2)
        res = torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        expected = ctcloss_reference(log_probs, targets.cuda(), input_lengths, target_lengths).float()
        with torch.backends.cudnn.flags(enabled=False):
            res2 = torch.nn.functional.ctc_loss(log_probs, targets.cuda().long(), input_lengths, target_lengths)
        self.assertEqual(res, expected)
        self.assertEqual(res2, res)

    @onlyCUDA
    @skipCUDAIfNoCudnn
    def test_contig_wrong_stride_cudnn(self, device):
        # x has to have batch_size 1 to test contiguous checks
        x = torch.randn(1, 16, 5, 5, device=device)
        stride = list(x.stride())
        stride[0] = 20
        # change the stride in dimension 0. the tensor is still contiguous because size[0] is 1
        x.set_(x.storage(), 0, x.size(), stride)
        self.assertTrue(x.is_contiguous())
        F.conv_transpose2d(x, torch.randn(16, 1, 1, 1, device=device))
        F.conv2d(x, torch.randn(1, 16, 1, 1, device=device))

    def _ordered_sequence(self, tensor_type):
        """Create ordered list of random sequences"""
        seqs = [tensor_type(random.randint(1, 6))
                for _ in range(5)]
        seqs = [s.random_(-128, 128) for s in seqs]
        ordered = sorted(seqs, key=len, reverse=True)
        return ordered

    def _padded_sequence(self, tensor_type):
        """Create Tensor of random padded sequences"""
        ordered = self._ordered_sequence(tensor_type)
        lengths = list(map(len, ordered))
        padded_tensor = rnn_utils.pad_sequence(ordered)
        return padded_tensor, lengths

    @onlyCUDA
    def test_device_mask(self, device):
        for enforce_sorted in [True, False]:
            tensor_type = torch.FloatTensor
            cuda_type_str = 'torch.cuda.FloatTensor'
            padded, lengths = self._padded_sequence(tensor_type)
            packed = rnn_utils.pack_padded_sequence(
                padded, lengths, enforce_sorted=enforce_sorted)
            self.assertFalse(packed.is_cuda)
            packed = packed.to(device)
            self.assertTrue(packed.is_cuda)
            unpacked, _ = rnn_utils.pad_packed_sequence(packed)
            self.assertEqual(unpacked.type(), cuda_type_str)

    @onlyCUDA
    def test_overwrite_module_params_on_conversion_cpu_device(self, device):
        # Test that under the current default settings
        # (`torch.__future__.get_overwrite_module_params_on_conversion() == False`),
        # a view to a module's parameters is not pointing to the same storage as
        # its base variable after converting the module to a different device.
        m = nn.Linear(20, 10)
        mw = m.weight[:]
        m.to(device)
        with torch.no_grad():
            # Without using `torch.no_grad()`, this will leak CUDA memory.
            # (Issue is filed at https://github.com/pytorch/pytorch/issues/21875)
            mw[0][0] = 5
            self.assertTrue(mw[0][0].device.type == "cpu")
            self.assertTrue(mw._base[0][0].device.type == "cuda")

        try:
            torch.__future__.set_overwrite_module_params_on_conversion(True)

            # Test that if `torch.__future__.get_overwrite_module_params_on_conversion() == True`,
            # a view to a module's parameters is still pointing to the same storage as
            # its base variable after converting the module to a different device.
            m = nn.Linear(20, 10)
            mw = m.weight[:]
            m.to(device)
            mw[0][0] = 5
            self.assertTrue(mw[0][0] == mw._base[0][0])

            # Test that if `torch.__future__.get_overwrite_module_params_on_conversion() == True`,
            # `cpu_module.to("cuda")` doesn't preserve previous references to
            # `cpu_module`'s parameters or gradients.
            m = nn.Linear(20, 10)
            m.weight.grad = torch.randn(10, 20)
            weight_ref = m.weight
            weight_grad_ref = m.weight.grad
            m.to(device)
            self.assertNotEqual(weight_ref.device, m.weight.device)
            self.assertNotEqual(weight_grad_ref.device, m.weight.grad.device)
        finally:
            torch.__future__.set_overwrite_module_params_on_conversion(False)

    @onlyCUDA
    @dtypes(torch.half, torch.float, torch.double)
    def test_embedding_max_norm_device(self, device, dtype):
        embedding = nn.Embedding(22, 5, max_norm=1.0).to(device, dtype=dtype)
        # nn.Embedding only takes LongTensor as input
        input = torch.tensor([2, 8, 8, 6], device=device, dtype=torch.long)
        output = embedding(input)
        self.assertEqual(output[1], output[2])
        self.assertTrue(output.data.norm(p=2, dim=1).le(1).all())

    # test is flaky on ROCm CI
    @onlyCUDA
    @skipCUDAIfRocm
    @dtypes(torch.half, torch.float)
    def test_softmax(self, device, dtype):
        input = torch.rand(32, 100, device=device, dtype=dtype, requires_grad=True)
        inputf = input.to(torch.float).detach().requires_grad_(True)
        out = F.softmax(input, dim=-1, dtype=torch.float)
        outf = F.softmax(inputf, dim=-1)
        # should be bitwise equal
        self.assertEqual(out, outf, prec=0)
        gO = torch.empty_like(outf).uniform_()
        out.backward(gO)
        outf.backward(gO)
        # should be bitwise equal
        self.assertEqual(input.grad, inputf.grad.to(dtype), prec=0)

    @onlyCUDA
    def test_pool3d_size_one_feature_dim(self, device):
        # Tests crazy strides for feature dim of size 1
        x = torch.randn(7, 1, 5, 3, 2, device=device)
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
            self.assertEqual(out_y, out_x.to(device), test)

    @onlyCUDA
    def test_AvgPool3d_backward_after_cat_dim1_device(self, device):
        # x has to have batch_size 1 to test contiguous checks
        x = torch.randn(1, 3, 4, 4, 4, device=device, requires_grad=True)
        y = F.avg_pool3d(x, kernel_size=3, padding=1, stride=2)

        grad = torch.randn(y.size(), device=device)
        # increase the stride in dimension 0. the tensor is still contiguous because size[0] is 1
        stride = list(grad.stride())
        stride[0] = stride[0] * 2
        grad.set_(grad.storage(), 0, grad.size(), stride)
        assert grad.is_contiguous()

        y.backward(grad)

    def test_embedding_bag_empty_input(self, device):
        m = 4
        n = 3
        x = torch.tensor([], device=device, dtype=torch.long)
        for sparse in [True, False]:
            Embed = torch.nn.EmbeddingBag(m, n, sparse=sparse)
            Embed.to(device)

            output = Embed(input=x, offsets=torch.tensor([0], device=device, dtype=torch.long))
            self.assertEqual(output, torch.zeros_like(output))

            output = Embed(input=x, offsets=torch.tensor([0, 0], device=device, dtype=torch.long))
            self.assertEqual(output, torch.zeros_like(output))

    def test_EmbeddingBag_per_sample_weights_failures(self, device):
        # Failure 1: mismatched embeddings / per_sample_weights dtype
        es = nn.EmbeddingBag(5, 2, mode='sum').to(dtype=torch.float, device=device)
        input = torch.tensor([3, 1, 1, 1, 4, 0], dtype=torch.long, device=device)
        offsets = torch.tensor([0, 0, 3, 3, 6], dtype=torch.long, device=device)
        per_sample_weights = torch.randn_like(input, dtype=torch.double, device=device)
        if device == 'cpu':
            with self.assertRaisesRegex(RuntimeError, 'have the same type as'):
                es(input, offsets, per_sample_weights)
        else:
            with self.assertRaisesRegex(RuntimeError, 'expected scalar type'):
                es(input, offsets, per_sample_weights)

        # Failure 2.1: input/per_sample_weights have different sizes (1d input)
        input = torch.tensor([3, 1, 1, 1, 4, 0], dtype=torch.long, device=device)
        offsets = torch.tensor([0, 0, 3, 3, 6], dtype=torch.long, device=device)
        per_sample_weights = torch.randn(5, dtype=torch.float, device=device)
        with self.assertRaisesRegex(ValueError, 'same shape as the input'):
            es(input, offsets, per_sample_weights)

        # Failure 2.2: input/per_sample_weights have different sizes (2d input)
        input = torch.randint(5, (7, 3), dtype=torch.long, device=device)
        offsets = None
        per_sample_weights = torch.randn(7 * 3, dtype=torch.float, device=device)
        with self.assertRaisesRegex(ValueError, 'same shape as the input'):
            es(input, offsets, per_sample_weights)

        # Failure 3: Unsupported per_sample_weights and mode=('max', 'mean')
        for unsupported_mode in ('max', 'mean'):
            es = nn.EmbeddingBag(5, 2, mode=unsupported_mode).to(
                dtype=torch.float, device=device)
            input = torch.randint(5, (7, 3), dtype=torch.long, device=device)
            offsets = None
            per_sample_weights = torch.randn(7, 3, dtype=torch.float, device=device)
            with self.assertRaisesRegex(NotImplementedError,
                                        "only supported for mode='sum'"):
                es(input, offsets, per_sample_weights)

    def _embedding_bag_reference_impl(self, input, weight, offsets=None, mode='sum',
                                      per_sample_weights=None):
        assert mode == 'sum'
        assert offsets is not None
        if per_sample_weights is None:
            per_sample_weights = torch.ones(input.size())
        assert input.numel() == per_sample_weights.numel()

        bags = []
        embeddings = weight.index_select(0, input) * per_sample_weights.unsqueeze(1)
        for index, offset in enumerate(offsets):
            if index + 1 < len(offsets):
                next_offset = offsets[index + 1]
            else:
                next_offset = len(input)
            length = next_offset - offset
            bags.append(embeddings.narrow(0, offset, length).sum(0))
        return torch.stack(bags)

    def test_EmbeddingBag_per_sample_weights_and_offsets(self, device):
        def test_per_sample_weights(mode, dtype, trainable_scale):
            es = nn.EmbeddingBag(5, 2, mode=mode).to(dtype=dtype, device=device)
            es.weight.data.copy_(
                torch.arange(1, 11, device=device, dtype=dtype).view_as(es.weight))
            input = torch.tensor([3, 1, 1, 1, 4, 0], device=device, dtype=torch.long)
            offsets = torch.tensor([0, 0, 3, 3, 6], device=device, dtype=torch.long)
            per_sample_weights = torch.randn_like(input, dtype=dtype) \
                                      .requires_grad_(trainable_scale)
            ref_per_sample_weights = \
                per_sample_weights.detach().requires_grad_(trainable_scale)
            reference_weights = es.weight.detach().requires_grad_()

            expected = self._embedding_bag_reference_impl(
                input, reference_weights, offsets, mode, ref_per_sample_weights)
            result = es(input, offsets, per_sample_weights)
            self.assertEqual(result, expected, prec=dtype2prec[dtype])

            grad = torch.randn_like(expected)
            result.backward(grad)
            expected.backward(grad)
            self.assertEqual(es.weight.grad, reference_weights.grad,
                             dtype2prec[dtype])
            if trainable_scale:
                self.assertEqual(per_sample_weights.grad, ref_per_sample_weights.grad,
                                 prec=dtype2prec[dtype])

        if device == 'cuda':
            dtypes = (torch.float, torch.double, torch.half)
        else:
            dtypes = (torch.float, torch.double)
        modes = ('sum',)
        trainable_scale = (True, False)
        for dtype, mode, trainable in itertools.product(dtypes, modes, trainable_scale):
            test_per_sample_weights(mode, dtype, trainable)

    def _test_EmbeddingBag_vs_Embedding(self, N, D, B, L, max_norm=None,
                                        mode='mean',
                                        device='cpu',
                                        dtype=torch.float,
                                        test_per_sample_weights=False,
                                        trainable_per_sample_weights=False,
                                        sparse=False,
                                        test_backward=True,
                                        backward_prec=None):
        es = nn.EmbeddingBag(N, D, mode=mode, sparse=sparse, max_norm=max_norm).to(device, dtype)
        e = nn.Embedding(N, D, max_norm=max_norm).to(device, dtype)
        e.weight.data.copy_(es.weight)
        input = torch.randint(N, (B, L), device=device, dtype=torch.long)
        offsets = torch.arange(0, B, device=device, dtype=torch.long).mul_(L)
        grad_output = torch.rand(B, D, device=device, dtype=dtype)

        if test_per_sample_weights:
            # To prevent large gradients, weights should sum to 1 for each bag
            per_sample_weights = \
                torch.randn(B, L, device=device, dtype=dtype).softmax(dim=-1)
            per_sample_weights_reference = \
                per_sample_weights.clone().requires_grad_(trainable_per_sample_weights)
            per_sample_weights.requires_grad_(trainable_per_sample_weights)
            output = es(input.view(-1), offsets, per_sample_weights.view(-1))
        else:
            output = es(input.view(-1), offsets)
            per_sample_weights = None
            per_sample_weights_reference = None

        if mode == 'sum':
            if test_per_sample_weights:
                ref_output = (e(input) * per_sample_weights_reference.unsqueeze(-1)).sum(1)
            else:
                ref_output = e(input).sum(1)
        elif mode == 'mean':
            assert not test_per_sample_weights
            ref_output = e(input).mean(1)
        elif mode == 'max':
            assert not test_per_sample_weights
            ref_output = e(input).max(1)[0]

        self.assertEqual(output, ref_output, dtype2prec[dtype])

        if not test_backward:
            return

        output.backward(grad_output)
        ref_output.backward(grad_output)
        es_weight_grad = es.weight.grad.data
        if sparse:
            es_weight_grad = es.weight.grad.data.to_dense()

        # We have more floating point error here because we are dealing with larger numbers
        if backward_prec is None:
            needed_prec = dtype2prec[dtype] * 2
        else:
            needed_prec = backward_prec

        self.assertEqual(es_weight_grad, e.weight.grad, needed_prec)

        if test_per_sample_weights and trainable_per_sample_weights:
            self.assertEqual(per_sample_weights.grad, per_sample_weights_reference.grad,
                             dtype2prec[dtype])

    @skipCUDAIf(True, "Temporarily disabled. See t54369166")
    def test_EmbeddingBag_per_sample_weights_and_no_offsets(self, device):
        def run_tests(dtype, mode, sparse, trainable_per_sample_weights):
            kwargs = dict(test_per_sample_weights=True, device=device,
                          mode=mode, dtype=dtype, sparse=sparse,
                          trainable_per_sample_weights=trainable_per_sample_weights)

            # Simple case
            self._test_EmbeddingBag_vs_Embedding(2, 3, 5, 7, **kwargs)

            # B * L > 1000
            self._test_EmbeddingBag_vs_Embedding(2, 5, 53, 23, **kwargs)

            # Large num_embedding
            self._test_EmbeddingBag_vs_Embedding(101, 5, 3, 7, **kwargs)

            # Large embedding_dim
            self._test_EmbeddingBag_vs_Embedding(2, 101, 3, 7, **kwargs)

        dtypes = (torch.float, torch.double)
        modes = ('sum',)
        sparsity = (True, False)
        trainable_scale = (True, False)
        for dtype, mode, sparse, trainable_per_sample_weights in \
                itertools.product(dtypes, modes, sparsity, trainable_scale):
            run_tests(dtype, mode, sparse, trainable_per_sample_weights)

        # Test CUDA Dense on half precision
        if device == 'cuda':
            dtypes = (torch.half,)
            modes = ('sum',)
            sparsity = (False,)
            trainable_scale = (True, False)
            for dtype, mode, sparse, trainable_per_sample_weights in \
                    itertools.product(dtypes, modes, sparsity, trainable_scale):
                run_tests(dtype, mode, sparse, trainable_per_sample_weights)

    def _test_EmbeddingBag(self, device, mode, sparse, dtype=torch.double, test_backward=True):
        # check a known test example
        es = nn.EmbeddingBag(5, 2, mode=mode, sparse=sparse).to(device, dtype)
        es.weight.data.copy_(torch.arange(1, 11, device=device, dtype=dtype).view_as(es.weight))
        input = torch.tensor([3, 1, 1, 1, 4, 0], device=device, dtype=torch.long)
        offsets = torch.tensor([0, 0, 3, 3, 6], device=device, dtype=torch.long)

        grad_output = torch.tensor(
            [1, 2,
             3, 4], device=device, dtype=dtype).view(2, 2)
        grad_output_with_empty = torch.tensor(
            [99, 99,
             1, 2,
             99, 99,
             3, 4,
             99, 99], device=device, dtype=dtype).view(5, 2)

        if mode == "sum" or mode == "mean":
            denominator = 1 if mode == "sum" else 3
            expected_output = torch.tensor(
                [[13, 16],
                 [13, 16]], device=device, dtype=dtype) / denominator

            expected_output_with_empty = torch.tensor(
                [[0, 0],
                 [13, 16],
                 [0, 0],
                 [13, 16],
                 [0, 0]], device=device, dtype=dtype) / denominator

            expected_grad_weight = torch.tensor(
                [[3, 4],
                 [5, 8],
                 [0, 0],
                 [1, 2],
                 [3, 4]], device=device, dtype=dtype) / denominator
        elif mode == "max":
            expected_output = torch.tensor(
                [[7, 8],
                 [9, 10]], device=device, dtype=dtype)

            expected_output_with_empty = torch.tensor(
                [[0, 0],
                 [7, 8],
                 [0, 0],
                 [9, 10],
                 [0, 0]], device=device, dtype=dtype)

            expected_grad_weight = torch.tensor(
                [[0, 0],
                 [0, 0],
                 [0, 0],
                 [1, 2],
                 [3, 4]], device=device, dtype=dtype)

        output = es(input, offsets)
        output.backward(grad_output_with_empty)

        es_weight_grad = es.weight.grad.data
        if sparse:
            es_weight_grad = es.weight.grad.to_dense()
        self.assertEqual(output, expected_output_with_empty)
        self.assertEqual(es_weight_grad, expected_grad_weight, dtype2prec[dtype])

        # check same example except as 2D (2 x 3)
        input = input.view(2, -1)
        es.zero_grad()
        output = es(input)
        output.backward(grad_output)

        es_weight_grad = es.weight.grad
        if sparse:
            es_weight_grad = es.weight.grad.to_dense()
        self.assertEqual(output, expected_output)
        self.assertEqual(es_weight_grad, expected_grad_weight, dtype2prec[dtype])

        # test all empty bags
        es.zero_grad()
        inputs = torch.tensor([], dtype=torch.long, device=device)
        offsets = torch.tensor([0, 0, 0, 0], device=device)
        es(inputs, offsets).sum().backward()
        dense_grad = es.weight.grad
        if dense_grad.is_sparse:
            dense_grad = dense_grad.to_dense()
        self.assertEqual(dense_grad, torch.zeros_like(es.weight))

        # now compare EmbeddingBag vs Embedding + Sum/Mean, for constant bag length
        N, D, B, L = random.randint(1, 100), random.randint(1, 100), random.randint(1, 50), random.randint(1, 50)
        kwargs = dict(mode=mode, sparse=sparse, device=device, dtype=dtype, test_backward=test_backward)
        self._test_EmbeddingBag_vs_Embedding(N, D, B, L, **kwargs)
        for max_norm in (None, 3):
            for p in itertools.product([1, 2], repeat=4):
                self._test_EmbeddingBag_vs_Embedding(*p, max_norm=max_norm, **kwargs)

        # check that giving illegal input combos raises error
        es = nn.EmbeddingBag(10, 20, mode=mode, sparse=sparse)
        input = torch.ones(3, 4)
        offset = torch.arange(0, 3)
        self.assertRaises(ValueError, lambda: es(input, offset))
        self.assertRaises(ValueError, lambda: es(input.view(-1)))
        offset[0] = 1
        self.assertRaises(ValueError, lambda: es(input.view(-1), offset))
        offset[0] = 0
        offset[-1] = 100
        self.assertRaises(ValueError, lambda: es(input.view(-1), offset))

    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    @dtypes(torch.float, torch.double)
    def test_embedding_bag_device(self, device, dtype):
        self._test_EmbeddingBag(device, 'sum', False, dtype)
        self._test_EmbeddingBag(device, 'mean', False, dtype)
        self._test_EmbeddingBag(device, 'max', False, dtype)

        test_backward = False
        if self.device_type == 'cuda':
            # see 'todo' in test_embedding_bag.
            test_backward = dtype is not torch.float16
        elif self.device_type == 'cpu':
            # TODO: figure out why precision on sparse embeddings isn't the
            # same as for dense.
            test_backward = dtype is not torch.float

        self._test_EmbeddingBag(device, 'sum', True, dtype, test_backward=test_backward)
        self._test_EmbeddingBag(device, 'mean', True, dtype, test_backward=test_backward)

    @onlyCUDA
    @dtypes(torch.half, torch.float, torch.double)
    def test_multihead_attention_dtype(self, device, dtype):
        embed_dim = 128
        num_heads = 8
        sl = 10
        bs = 8
        model = nn.MultiheadAttention(embed_dim, num_heads).cuda().to(dtype)
        q = torch.randn(sl, bs, embed_dim, device=device, dtype=dtype)
        k = torch.randn(sl, bs, embed_dim, device=device, dtype=dtype)
        v = torch.randn(sl, bs, embed_dim, device=device, dtype=dtype)
        out = model(q, k, v)
        self.assertEqual(q.size(), out[0].size())
        self.assertEqual(dtype, out[0].dtype)

    @dtypesIfCUDA(*ALL_TENSORTYPES2)
    @dtypes(torch.float)
    def test_Conv2d_naive_groups(self, device, dtype):
        # Check that grouped convolutions matches two half convolutions
        m = nn.Conv2d(4, 4, kernel_size=3, groups=2).to(device, dtype)
        i = torch.randn(2, 4, 6, 6, device=device, dtype=dtype, requires_grad=True)
        output = m(i)
        grad_output = torch.randn(2, 4, 4, 4, device=device, dtype=dtype)
        output.backward(grad_output)

        m1 = nn.Conv2d(2, 2, kernel_size=3).to(device, dtype)
        m1.weight.data.copy_(m.weight.data[:2])
        m1.bias.data.copy_(m.bias.data[:2])
        i1 = i.data[:, :2].contiguous().requires_grad_(True)
        output1 = m1(i1)
        output1.backward(grad_output[:, :2].contiguous())

        m2 = nn.Conv2d(2, 2, kernel_size=3).to(device, dtype)
        m2.weight.data.copy_(m.weight.data[2:])
        m2.bias.data.copy_(m.bias.data[2:])
        i2 = i.data[:, 2:].contiguous().requires_grad_(True)
        output2 = m2(i2)
        output2.backward(grad_output[:, 2:].contiguous())

        self.assertEqual(output, torch.cat([output1, output2], 1))
        self.assertEqual(i.grad.data,
                         torch.cat([i1.grad.data, i2.grad.data], 1),
                         prec=dtype2prec[dtype])
        self.assertEqual(m.bias.grad.data,
                         torch.cat([m1.bias.grad.data, m2.bias.grad.data], 0),
                         prec=dtype2prec[dtype])
        self.assertEqual(m.weight.grad.data,
                         torch.cat([m1.weight.grad.data, m2.weight.grad.data], 0),
                         prec=dtype2prec[dtype])

    def _test_batchnorm_grad(self, device, dtype=torch.double):
        bs, n_feat, size_feat = 4, 5, 6
        input = torch.arange(bs * n_feat * size_feat, device=device,
                             requires_grad=True, dtype=dtype).view(bs, n_feat, size_feat)
        weight = torch.arange(1, n_feat + 1, device=device, requires_grad=True, dtype=dtype)
        bias = torch.arange(n_feat, device=device, requires_grad=True, dtype=dtype)
        running_mean = 1 - torch.arange(n_feat, device=device, dtype=dtype)
        running_var = 2 * torch.arange(n_feat, device=device, dtype=dtype)
        for training in [False, True]:
            _assertGradAndGradgradChecks(self, F.batch_norm, (input, running_mean, running_var, weight, bias,
                                                              training, 0.1, 0.0001))

    def test_batchnorm_grad(self, device):
        self._test_batchnorm_grad(device)

        if self.device_type == 'cuda' and self.has_cudnn():
            with torch.backends.cudnn.flags(enabled=False):
                self._test_batchnorm_grad(device)

    def _test_batchnorm_eval(self, device, dtype=torch.float):
        module = nn.BatchNorm1d(3).to(device, dtype)
        module.eval()

        data = torch.rand(4, 3, device=device, dtype=dtype, requires_grad=True)
        grad = torch.rand(4, 3, device=device, dtype=dtype)

        # 1st pass
        res1 = module(data)
        res1.backward(grad)
        grad1 = data.grad.clone()

        # 2nd pass
        if data.grad is not None:
            data.grad.data.zero_()

        res2 = module(data)
        res2.backward(grad)
        grad2 = data.grad.clone()
        self.assertEqual(res1, res2)
        self.assertEqual(grad1, grad2)

        # track_running_stats=False
        module = nn.BatchNorm1d(3, track_running_stats=False).to(device, dtype)

        data = torch.rand(4, 3, device=device, dtype=dtype, requires_grad=True)
        grad = torch.rand(4, 3, device=device, dtype=dtype)

        # 1st pass
        res1 = module(data)
        res1.backward(grad)
        grad1 = data.grad.clone()

        # set eval
        module.eval()

        # 2nd pass
        if data.grad is not None:
            data.grad.data.zero_()

        res2 = module(data)
        res2.backward(grad)
        grad2 = data.grad.clone()
        self.assertEqual(res1, res2)
        self.assertEqual(grad1, grad2)

    def test_batchnorm_eval(self, device):
        self._test_batchnorm_eval(device)

        if self.device_type == 'cuda' and self.has_cudnn():
            with torch.backends.cudnn.flags(enabled=False):
                self._test_batchnorm_eval(device)

    def _test_batchnorm_simple_average(self, device, dtype):
        module = nn.BatchNorm1d(3, momentum=None).to(dtype=dtype, device=device)
        zeros = torch.zeros(3, dtype=dtype, device=device)
        ones = torch.ones(3, dtype=dtype, device=device)
        self.assertEqual(module.running_mean, zeros)
        self.assertEqual(module.running_var, ones)

        data1 = torch.rand(4, 3, dtype=dtype, device=device)
        data2 = torch.rand(4, 3, dtype=dtype, device=device)

        # 1st pass
        res1 = module(data1)
        running_mean1 = module.running_mean.clone()
        running_var1 = module.running_var.clone()
        self.assertNotEqual(running_mean1, zeros)
        self.assertNotEqual(running_var1, ones)

        # reset stats
        module.reset_running_stats()
        self.assertEqual(module.running_mean, zeros)
        self.assertEqual(module.running_var, ones)

        # 2nd pass
        res2 = module(data2)
        running_mean2 = module.running_mean.clone()
        running_var2 = module.running_var.clone()
        self.assertNotEqual(running_mean2, zeros)
        self.assertNotEqual(running_var2, ones)

        # reset stats
        module.reset_running_stats()
        self.assertEqual(module.running_mean, zeros)
        self.assertEqual(module.running_var, ones)

        # 3rd (combined) pass
        res3 = module(data1)
        res4 = module(data2)
        self.assertEqual(res3, res1)
        self.assertEqual(res4, res2)
        self.assertAlmostEqual(module.running_mean, (running_mean1 + running_mean2) / 2)
        self.assertAlmostEqual(module.running_var, (running_var1 + running_var2) / 2)

    @dtypes(torch.float)
    def test_batchnorm_simple_average(self, device, dtype):
        self._test_batchnorm_simple_average(device, dtype)

        if self.device_type == 'cuda' and self.has_cudnn():
            with torch.backends.cudnn.flags(enabled=False):
                self._test_batchnorm_simple_average(device, dtype)

    def _test_maxpool_indices(self, num_dim, adaptive=False, device="cpu", dtype=torch.float):
        def expected_indices(dim):
            if dim == 1:
                return torch.tensor([1, 3], dtype=torch.double).repeat(2, 2, 1)
            if dim == 2:
                return torch.tensor([[5, 7], [13, 15]], dtype=torch.double).repeat(2, 2, 1, 1)

        def expected_grad(dim):
            if dim == 1:
                return torch.tensor([0, 1, 0, 1], dtype=torch.double).repeat(2, 2, 1)
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
        module = module_cls(2, return_indices=True).to(device, dtype=dtype)
        numel = 4 ** (num_dim + 1)
        input = torch.arange(1, numel + 1).view(2, 2, *repeat(4, num_dim)).to(device, dtype=dtype)
        input_var = input.clone().detach().requires_grad_()

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
        grad_output = torch.ones(output.size(), device=device, dtype=dtype)
        output.backward(grad_output, retain_graph=True)
        expected_grad = expected_grad(num_dim)
        self.assertEqual(input_var.grad.data, expected_grad.view_as(input))

        # Make sure backward after changing indices will result in an error
        indices.add_(1)
        self.assertRaises(RuntimeError, lambda: output.backward(grad_output))

        # Make sure -Infinity is handled correctly
        t = torch.tensor([[[float("-inf")]]])
        m = nn.MaxPool1d(kernel_size=1, return_indices=True)
        output, indices = m(t)
        self.assertEqual(output[0, 0, 0], float("-inf"), allow_inf=True)
        self.assertEqual(indices[0, 0, 0], 0)

        t = torch.tensor([[[float("-inf")]]])
        m = nn.MaxPool2d(kernel_size=1, return_indices=True)
        output, indices = m(t)
        self.assertEqual(output[0, 0, 0], float("-inf"), allow_inf=True)
        self.assertEqual(indices[0, 0, 0], 0)

        t = torch.tensor([[[[float("-inf")]]]])
        m = nn.MaxPool3d(kernel_size=1, return_indices=True)
        output, indices = m(t)
        self.assertEqual(output[0, 0, 0, 0], float("-inf"), allow_inf=True)
        self.assertEqual(indices[0, 0, 0, 0], 0)

    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    @dtypes(torch.float)
    def test_MaxPool1d_indices(self, device, dtype):
        self._test_maxpool_indices(1, device=device, dtype=dtype)

    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    @dtypes(torch.float)
    def test_MaxPool2d_indices(self, device, dtype):
        self._test_maxpool_indices(2, device=device, dtype=dtype)

    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    @dtypes(torch.float)
    def test_MaxPool3d_indices(self, device, dtype):
        self._test_maxpool_indices(3, device=device, dtype=dtype)

    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    @dtypes(torch.float)
    def test_AdaptiveMaxPool1d_indices(self, device, dtype):
        self._test_maxpool_indices(1, adaptive=True, device=device, dtype=dtype)

    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    @dtypes(torch.float)
    def test_AdaptiveMaxPool2d_indices(self, device, dtype):
        self._test_maxpool_indices(2, adaptive=True, device=device, dtype=dtype)

    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    @dtypes(torch.float)
    def test_AdaptiveMaxPool3d_indices(self, device, dtype):
        self._test_maxpool_indices(3, adaptive=True, device=device, dtype=dtype)

    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    @dtypes(torch.float)
    def test_max_pool_nan(self, device, dtype):
        for adaptive in ['', 'adaptive_']:
            for num_dim in [1, 2, 3]:
                fn_name = '{}max_pool{}d'.format(adaptive, num_dim)
                fn = getattr(F, fn_name)
                x = torch.full([1, 1] + num_dim * [3], nan)
                res = fn(x, 1 if adaptive else 3)
                self.assertTrue(math.isnan(res.item()))

    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    @dtypes(torch.float)
    def test_pool_large_size(self, device, dtype):
        for op in ('max', 'avg'):
            for num_dim in [1, 2, 3]:
                fn_name = '{}_pool{}d'.format(op, num_dim)
                fn = getattr(F, fn_name)
                # 16777217 is the smallest integer not expressible in float32
                x = torch.ones([1, 1, 16777217] + (num_dim - 1) * [1],
                               device=device, dtype=dtype)
                res = fn(x, 1, stride=1, padding=0)
                # check if the output shape was still computed correctly
                self.assertEqual(x.shape[2], res.shape[2])

    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    @dtypes(torch.float)
    def test_pool_invalid_size(self, device, dtype):
        for op in ('max', 'avg'):
            for num_dim in [1, 2, 3]:
                fn_name = '{}_pool{}d'.format(op, num_dim)
                fn = getattr(F, fn_name)
                # use a configuration that gives zero outputs only
                # when doing a correct floor division by the stride
                x = torch.ones([1, 1] + num_dim * [4],
                               device=device, dtype=dtype)
                with self.assertRaisesRegex(RuntimeError, r"too small|smaller than"):
                    try:
                        res = fn(x, 3, stride=2, padding=0, dilation=2)
                    except TypeError:
                        # some implementations do not support dilation
                        res = fn(x, 6, stride=2, padding=0)

    def test_CTCLoss_empty_target(self, device):
        target_lengths = [0, 0, 0]
        input_lengths = [50, 50, 50]
        targets = torch.randint(1, 15, (0,), dtype=torch.long, device=device)
        log_probs = torch.randn(50, 3, 15, dtype=torch.double, device=device).log_softmax(2)
        loss = torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths, reduction='none')
        self.assertTrue((loss >= 0).all().item())
        self.assertAlmostEqual(-log_probs.sum(0)[:, 0], loss)

        target_lengths = [0, 9, 0]
        input_lengths = [50, 50, 50]
        targets = torch.randint(1, 15, (9,), dtype=torch.long, device=device)
        log_probs = torch.randn(50, 3, 15, dtype=torch.double, device=device).log_softmax(2)
        loss = torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths, reduction='none')
        self.assertTrue((loss >= 0).all().item())
        self.assertAlmostEqual(-log_probs.sum(0)[[0, 2], 0], loss[[0, 2]])

    def test_empty_dropout(self, device):
        x = torch.Tensor([]).to(device)
        out = torch.nn.functional.dropout(x)
        self.assertEqual(out.size(), x.size())

    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    @dtypes(torch.float)
    def test_variable_sequence(self, device, dtype):
        def pad(var, length):
            if var.size(0) == length:
                return var
            return torch.cat([var, var.new_zeros(length - var.size(0), *var.size()[1:])])

        def maybe_index_tuple(maybe_tuple_of_tensors, index):
            if maybe_tuple_of_tensors is None:
                return None
            return tuple(maybe_tuple_of_tensors[j][:, index:index + 1, :].contiguous()
                         for j in range(2))

        def check_lengths(lengths, enforce_sorted, use_default_hiddens):
            input_size = 3
            hidden_size = 4
            num_layers = 2
            bidirectional = True

            max_length = max(lengths)
            x_leaf = torch.randn(max_length, len(lengths), input_size, device=device,
                                 dtype=dtype, requires_grad=True)
            num_directions = 2 if bidirectional else 1
            lstm = nn.LSTM(input_size, hidden_size, bidirectional=bidirectional,
                           num_layers=num_layers).to(device, dtype)
            lstm2 = deepcopy(lstm).to(device, dtype)
            x = x_leaf

            hidden0 = None
            if not use_default_hiddens:
                hidden0 = tuple(torch.randn(num_directions * num_layers, len(lengths), hidden_size,
                                            device=device, dtype=dtype)
                                for _ in range(2))

            # Compute sequences separately
            seq_outs = []
            seq_hiddens = []
            for i, l in enumerate(lengths):
                hidden_i = maybe_index_tuple(hidden0, i)
                out, hid = lstm2(x[:l, i:i + 1], hidden_i)
                out_pad = pad(out, max_length)
                seq_outs.append(out_pad)
                seq_hiddens.append(hid)
            seq_out = torch.cat(seq_outs, 1)
            seq_hidden = tuple(torch.cat(hids, 1) for hids in zip(*seq_hiddens))

            # Use packed format
            packed = rnn_utils.pack_padded_sequence(x, lengths, enforce_sorted=enforce_sorted)
            packed_out, packed_hidden = lstm(packed, hidden0)
            unpacked, unpacked_len = rnn_utils.pad_packed_sequence(packed_out)

            # Check forward
            prec = dtype2prec[dtype]
            self.assertEqual(packed_hidden, seq_hidden, prec)
            self.assertEqual(unpacked, seq_out, prec)
            self.assertEqual(unpacked_len, lengths, prec)

            # Check backward
            seq_out.sum().backward()
            grad_x = x_leaf.grad.data.clone()
            x_leaf.grad.data.zero_()
            unpacked.sum().backward()

            self.assertEqual(x_leaf.grad, grad_x, dtype2prec[dtype])
            for p1, p2 in zip(lstm.parameters(), lstm2.parameters()):
                prec = dtype2prec[dtype]
                if dtype == torch.float16:
                    prec = 2e-2
                self.assertEqual(p1.grad, p2.grad, prec)

        tests = [
            # enforce_sorted, lengths
            [True, [5]],
            [False, [5]],
            [True, [10, 10, 6, 2, 2, 1, 1]],
            [False, [10, 10, 6, 2, 2, 1, 1]],
            [False, [2, 1, 3, 2, 10, 5, 3]],
        ]

        for enforce_sorted, seq_lens, in tests:
            for use_default_hiddens in (True, False):
                check_lengths(seq_lens, enforce_sorted, use_default_hiddens)

    def _test_batchnorm_update_stats(self, device, dtype=torch.float):
        module = nn.BatchNorm1d(3).to(device, dtype)

        data = torch.rand(4, 3, device=device, dtype=dtype)

        # training pass
        old_running_mean = module.running_mean.clone()
        old_running_var = module.running_var.clone()
        old_num_batches_tracked = module.num_batches_tracked.clone()
        module(data)
        self.assertNotEqual(old_running_mean, module.running_mean)
        self.assertNotEqual(old_running_var, module.running_var)
        self.assertEqual(old_num_batches_tracked + 1, module.num_batches_tracked)

        # eval pass
        module.eval()
        old_running_mean = module.running_mean.clone()
        old_running_var = module.running_var.clone()
        old_num_batches_tracked = module.num_batches_tracked.clone()
        module(data)
        self.assertEqual(old_running_mean, module.running_mean)
        self.assertEqual(old_running_var, module.running_var)
        self.assertEqual(old_num_batches_tracked, module.num_batches_tracked)

    def test_batchnorm_update_stats(self, device):
        self._test_batchnorm_update_stats(device)

        if self.device_type == 'cuda' and self.has_cudnn():
            with torch.backends.cudnn.flags(enabled=False):
                self._test_batchnorm_update_stats(device)

    def test_multi_margin_loss_errors(self, device):
        self.assertRaises(RuntimeError,
                          lambda: nn.functional.multi_margin_loss(torch.randn(5, device=device),
                                                                  torch.zeros(3, device=device)))

instantiate_device_type_tests(TestNNDeviceType, globals())

if __name__ == '__main__':
    run_tests()
