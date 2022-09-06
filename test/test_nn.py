# Owner(s): ["module: nn"]

import contextlib
import math
import random
import string
import unittest
import io
import unittest.mock as mock
import itertools
import warnings
import pickle
from copy import deepcopy
from itertools import product
from functools import reduce, partial
from operator import mul
from collections import OrderedDict
from tempfile import NamedTemporaryFile
import weakref
import gc

import torch

# TODO: remove this global setting
# NN tests use double as the default dtype
torch.set_default_dtype(torch.double)

from torch._six import inf, nan
import torch.autograd.forward_ad as fwAD
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.nn.utils.rnn as rnn_utils
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
import torch.nn.utils.parametrize as parametrize
import torch.nn.utils.prune as prune
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.nn import Parameter
from torch.nn.parallel._functions import Broadcast
from torch.testing._internal.common_dtype import integral_types, get_all_math_dtypes
from torch.testing._internal.common_utils import freeze_rng_state, run_tests, TestCase, skipIfNoLapack, skipIfRocm, \
    TEST_NUMPY, TEST_SCIPY, TEST_WITH_CROSSREF, TEST_WITH_ROCM, \
    download_file, get_function_arglist, load_tests, skipIfMps,\
    TemporaryFileName, TEST_WITH_UBSAN, IS_PPC, \
    parametrize as parametrize_test, subtest, instantiate_parametrized_tests, IS_WINDOWS, \
    skipIfTorchDynamo
from torch.testing._internal.common_cuda import TEST_CUDA, TEST_MULTIGPU, TEST_CUDNN, TEST_CUDNN_VERSION
from torch.testing._internal.common_nn import NNTestCase, NewModuleTest, CriterionTest, \
    module_tests, criterion_tests, loss_reference_fns, \
    ctcloss_reference, new_module_tests, single_batch_reference_fn, _test_bfloat16_ops, _test_module_empty_input
from torch.testing._internal.common_device_type import instantiate_device_type_tests, dtypes, \
    dtypesIfCUDA, precisionOverride, skipCUDAIfCudnnVersionLessThan, onlyCUDA, onlyCPU, \
    skipCUDAIfRocm, skipCUDAIf, skipCUDAIfNotRocm, \
    onlyNativeDeviceTypes, deviceCountAtLeast, largeTensorTest, expectedFailureMeta, skipMeta, get_all_device_types
from torch.nn import MultiheadAttention

from hypothesis import given
import torch.testing._internal.hypothesis_utils as hu
from torch.testing._internal.common_utils import _assertGradAndGradgradChecks, gradcheck, gradgradcheck, \
    GRADCHECK_NONDET_TOL
from torch.testing._internal.common_utils import dtype2prec_DONTUSE
from torch.testing._internal.common_cuda import tf32_on_and_off, tf32_is_not_fp32, tf32_off, tf32_on
from torch.types import _TensorOrTensors


AMPERE_OR_ROCM = TEST_WITH_ROCM or tf32_is_not_fp32()

# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

if TEST_SCIPY:
    from scipy import stats
    import scipy.signal
    import scipy.ndimage

if TEST_NUMPY:
    import numpy as np


# WARNING: If you add a new top-level test case to this file, you MUST
# update test/run_test.py to list it, otherwise it will NOT be run in
# CI.

class TestNN(NNTestCase):
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True

    def _forward(self, module, input: _TensorOrTensors):
        with freeze_rng_state():
            if isinstance(input, tuple):
                return module(*input)
            else:
                return module(input)

    def _backward(self, module, input: _TensorOrTensors, output, grad_output, create_graph=False):
        output.backward(grad_output, retain_graph=True, create_graph=create_graph)
        if isinstance(input, tuple):
            return tuple(i.grad.data if i.grad is not None else None for i in input)
        else:
            return input.grad.data if input.grad is not None else None

    def _forward_criterion(self, criterion, input, target, extra_args=None):
        if extra_args is None:
            extra_args = tuple()
        if isinstance(input, tuple):
            args = input + (target,) + extra_args
            output = criterion(*args)
        else:
            output = criterion(input, target, *extra_args)
        return output

    def _backward_criterion(self, criterion, input, output, target, gradOutput=None, extra_args=None):
        if extra_args is None:
            extra_args = tuple()
        input_tuple = input if isinstance(input, tuple) else (input,)
        output_tuple = output if isinstance(output, tuple) else (output,)
        for i in input_tuple:
            if i.grad is not None:
                i.grad.data.zero_()
        args = input_tuple + (target,) + extra_args
        if gradOutput is None:
            gradOutput = torch.ones(())
        criterion(*args).backward(gradOutput.to(output_tuple[0]))
        if isinstance(input, tuple):
            return tuple(i.grad.data for i in input)
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
                self.layer_dummy_param = Parameter(torch.empty(3, 5))
                self.register_buffer('layer_dummy_buf', torch.zeros(1, 3, 3, 7))

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.l1 = Layer()
                self.dummy_param = Parameter(torch.empty(3, 5))
                self.register_buffer('dummy_buf', torch.zeros(7, 3, 3, 1))

        l = Layer()
        n = Net()
        s = nn.Sequential(n, n)

        return l, n, s

    def test_parse_to(self):
        # Test for buggy use of THPMemoryFormat_New
        self.assertEqual(
            repr(torch._C._nn._parse_to(memory_format=torch.contiguous_format)[3]),
            "torch.contiguous_format"
        )

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

    def _test_hooks(self, backward_register_fn):
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
            self.assertEqual(input[0], torch.ones(5, 5))
            self.assertEqual(output, torch.empty(5, 5).fill_(1 / (1 + 1 / math.e)))
            counter['forwards'] += inc

        def bw_hook(inc, h_module, grad_input, grad_output):
            self.assertIsInstance(grad_input, tuple)
            self.assertIsInstance(grad_output, tuple)
            self.assertTrue(h_module is module)
            self.assertEqual(grad_output[0], torch.ones(5, 5) * 2)
            counter['backwards'] += inc

        # backward_pre_hook expects callback with only `module` and `grad_output`
        # as arguments.
        def bw_pre_hook(inc, h_module, grad_output):
            self.assertIsInstance(grad_output, tuple)
            self.assertTrue(h_module is module)
            self.assertEqual(grad_output[0], torch.ones(5, 5) * 2)
            counter['backwards'] += inc

        test_fwd = module.register_forward_hook(lambda *args: fw_hook(1, *args))

        module(input)
        module(input)
        self.assertEqual(counter['forwards'], 2)
        self.assertEqual(counter['backwards'], 0)

        bw_hook_fn = bw_pre_hook if backward_register_fn == 'register_full_backward_pre_hook' else bw_hook
        test_bwd = getattr(module, backward_register_fn)(
            lambda *args: bw_hook_fn(1, *args))

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

        test2_bwd = getattr(module, backward_register_fn)(lambda *args: bw_hook_fn(2, *args))

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

    @skipIfTorchDynamo("TorchDynamo does not work well with hooks")
    def test_hooks(self):
        self._test_hooks("register_backward_hook")
        self._test_hooks("register_full_backward_hook")
        self._test_hooks("register_full_backward_pre_hook")

    def test_hook_cpp(self):
        bn = nn.BatchNorm1d(5)

        def hook(module, grad_inputs, grad_outputs):
            self.assertEqual(len(grad_inputs), 1)
            self.assertEqual(len(grad_outputs), 1)
            self.assertEqual(module, bn)

        bn.register_full_backward_hook(hook)
        output = bn(torch.randn(5, 5, requires_grad=True))
        output.sum().backward()

    @skipIfTorchDynamo("TorchDynamo does not work well with hooks")
    def test_backward_hooks_interaction(self):
        # Test to make sure that the grad_outputs
        # updated by full_backward_pre_hook are received by
        # the full_backward_hook
        module = torch.nn.Sigmoid()

        cnt = {'backward_cnt': 0}

        def bw_pre_hook(m, grad_output):
            cnt['backward_cnt'] += 1
            return (grad_output[0] * 0.5, )

        def bw_hook(m, grad_in, grad_output):
            self.assertEqual(torch.full_like(grad_output[0], 0.5), grad_output[0])
            cnt['backward_cnt'] += 1
            return grad_output

        module.register_full_backward_pre_hook(bw_pre_hook)
        module.register_full_backward_hook(bw_hook)

        t = torch.ones(1, 2, requires_grad=True)
        module(t).sum().backward()
        self.assertEqual(cnt['backward_cnt'], 2)

    def test_hook_invalid_outputs(self):
        module = nn.Sigmoid()
        input = torch.randn(5, 5, requires_grad=True)

        def bw_fail1(self, grad_input, grad_output):
            return grad_input[:-1]

        def bw_fail2(self, grad_input, grad_output):
            return grad_input + (torch.randn(2, 2),)

        with module.register_backward_hook(bw_fail1):
            with self.assertRaisesRegex(RuntimeError, 'got 0, but expected 1'):
                module(input).sum().backward()

        with module.register_backward_hook(bw_fail2):
            with self.assertRaisesRegex(RuntimeError, 'got 2, but expected 1'):
                module(input).sum().backward()

        def bw_pre_fail1(self, grad_output):
            return ()

        def bw_pre_fail2(self, grad_output):
            return grad_output + (torch.randn(2, 2),)

        with module.register_full_backward_pre_hook(bw_pre_fail1):
            with self.assertRaisesRegex(RuntimeError, 'got 0, but expected 1'):
                module(input).sum().backward()

        with module.register_full_backward_pre_hook(bw_pre_fail2):
            with self.assertRaisesRegex(RuntimeError, 'got 2, but expected 1'):
                module(input).sum().backward()

    def test_hook_requires_grad(self):
        test_self = self

        class MyModule(nn.Module):
            def forward(self, arg1, arg2, arg3):
                test_self.assertTrue(arg1.requires_grad)
                test_self.assertFalse(arg2.requires_grad)
                test_self.assertTrue(arg3.requires_grad)
                return arg1.sum() + arg2.sum() + arg3.sum()

        inp = torch.rand(2, requires_grad=True)
        mod = MyModule()

        mod(inp, inp.detach(), inp)
        # Ensure that requires grad is properly propagated
        mod.register_full_backward_hook(lambda mod, gI, gO: None)
        mod(inp, inp.detach(), inp)

    @skipIfTorchDynamo("TorchDynamo does not work well with hooks")
    def test_hook_no_requires_grad(self):
        mod = nn.Linear(2, 3)

        inp = torch.rand(1, 2)

        return_val = "None"
        hook_called = [0]

        def hook(mod, grad_input, grad_output):
            hook_called[0] += 1
            for gI in grad_input:
                self.assertIsNone(gI)
            for gO in grad_output:
                self.assertEqual(gO.size(), (1, 3))

            if return_val == "grad_input":
                return grad_input
            elif return_val == "invalid":
                # If the inputs were requiring gradients, this would be
                # a valid return
                return inp
            elif return_val == "None":
                return None
            else:
                raise RuntimeError("Invalid return_val string")

        mod.register_full_backward_hook(hook)

        # This should run and trigger the hook properly
        mod(inp).sum().backward()
        self.assertEqual(hook_called[0], 1)

        return_val = "grad_input"

        mod(inp).sum().backward()
        self.assertEqual(hook_called[0], 2)

        return_val = "invalid"
        with self.assertRaisesRegex(RuntimeError, "where no input requires gradient"):
            mod(inp).sum().backward()

    def test_hook_last_arg_requires_grad(self):
        mod = nn.L1Loss()
        inp = torch.rand(1, requires_grad=True)
        mod.register_full_backward_hook(lambda m, gI, gO: None)

        try:
            mod(inp.detach(), inp)
        except Exception as ex:
            self.fail("Unexpected exception: %s" % ex)

    def test_hook_extra_input(self):
        class MyModule(nn.Module):
            def forward(self, non_tensor, tensor):
                return tensor.clone(), non_tensor

        inp = torch.rand(2, requires_grad=True)
        mod = MyModule()

        def hook(mod, grad_input, grad_output):
            self.assertIsNone(grad_input[0])
            self.assertIsInstance(grad_input[1], torch.Tensor)

            self.assertIsInstance(grad_output[0], torch.Tensor)
            self.assertIsNone(grad_output[1])

        mod.register_full_backward_hook(hook)
        out, _ = mod(True, inp)
        out.sum().backward()

    def test_hook_inplace(self):
        class MyModule(nn.Module):
            def forward(self, inp, do_inplace):
                self.inp = inp
                if do_inplace:
                    inp += 1
                return inp.clone()

        hook_called = [0]

        def hook(mod, grad_input, grad_output):
            hook_called[0] += 1

        def hook_pre(mod, grad_output):
            hook_called[0] += 1

        inp = torch.rand(10, requires_grad=True)
        mod = MyModule()
        for hook_fn, register_fn in [(hook, mod.register_full_backward_hook),
                                     (hook_pre, mod.register_full_backward_pre_hook)]:
            hook_called[0] = 0
            with register_fn(hook_fn):
                # No inplace should work
                mod(inp, False).sum().backward()
                self.assertEqual(hook_called[0], 1)

                # Input inplace error should throw an error
                with self.assertRaisesRegex(RuntimeError, "Output 0 of BackwardHookFunctionBackward is "
                                            "a view and is being modified inplace."):
                    mod(inp.clone(), True)

                # Input inplace error should throw an error if we try to re-use the view after they have
                # been modified
                local_inp = inp.clone()
                out = mod(local_inp, False)
                local_inp[0] *= 1
                with self.assertRaisesRegex(RuntimeError, "Output 0 of BackwardHookFunctionBackward is "
                                            "a view and its base or another view"):
                    # Any operation involving the view will fail here
                    mod.inp + 2

                # Output inplace error should throw an error
                out = mod(inp, False)
                with self.assertRaisesRegex(RuntimeError, "BackwardHookFunctionBackward is a view "
                                            "and is being modified inplace."):
                    out += 1

    def test_hook_non_full_warning(self):
        def noop(*args):
            pass

        a = torch.rand(2, requires_grad=True)
        b = torch.rand(2, requires_grad=True)

        # Check invalid input container
        class MyModule(nn.Module):
            def forward(self, l):
                return l[0].clone(), l[1].clone()

        m = MyModule()
        m.register_backward_hook(noop)

        with self.assertWarnsRegex(UserWarning, "does not take as input a single Tensor or a tuple of Tensors"):
            m([a, b])

        # Check invalid output container
        class MyModule(nn.Module):
            def forward(self, a, b):
                return [a.clone(), b.clone()]

        m = MyModule()
        m.register_backward_hook(noop)

        with self.assertWarnsRegex(UserWarning, "does not return a single Tensor or a tuple of Tensors"):
            m(a, b)

        # Check invalid output from different Nodes
        class MyModule(nn.Module):
            def forward(self, a, b):
                return a.clone(), b.clone()

        m = MyModule()
        m.register_backward_hook(noop)

        with self.assertWarnsRegex(UserWarning, "outputs are generated by different autograd Nodes"):
            m(a, b)

        # Check invalid forward with multiple Nodes
        class MyModule(nn.Module):
            def forward(self, a):
                return a.clone().clone()

        m = MyModule()
        m.register_backward_hook(noop)

        with self.assertWarnsRegex(UserWarning, "the forward contains multiple autograd Nodes"):
            m(a)

    def test_hook_backward_size(self):
        # Make module with multiple operations in forward
        # And different size for input and outputs
        class MyModule(nn.Module):
            def forward(self, arg1, arg2):
                tmp = arg1.sum() * arg2
                tmp = tmp + arg2.sum() * arg1.sum()
                tmp = tmp.sum().view(1)
                tmp = tmp.expand(8).contiguous()
                return tmp

        module = MyModule()
        inp1 = torch.randn(5, 5, requires_grad=True)
        inp2 = torch.randn(10, 10, requires_grad=True)

        def bw_hook(module, grad_input, grad_output):
            self.assertEqual(len(grad_input), 2)
            self.assertEqual(grad_input[0].size(), torch.Size([5, 5]))
            self.assertEqual(grad_input[1].size(), torch.Size([10, 10]))
            self.assertEqual(len(grad_output), 1)
            self.assertEqual(grad_output[0].size(), torch.Size([8]))

        with module.register_full_backward_hook(bw_hook):
            module(inp1, inp2).sum().backward()

    @skipIfTorchDynamo("TorchDynamo does not work well with hooks")
    def test_hook_backward_writeable(self):
        module = nn.Sigmoid()
        input = torch.randn(5, 5, requires_grad=True)
        sig_x = torch.nn.functional.sigmoid(input)

        def bw_hook(module, grad_input, grad_output):
            for grad in grad_input:
                self.assertTrue(isinstance(grad, torch.Tensor))
            for grad in grad_output:
                self.assertTrue(isinstance(grad, torch.Tensor))
            return tuple(gi * 2 for gi in grad_input)

        module.register_backward_hook(bw_hook)
        module(input).backward(torch.ones(5, 5))
        expected_grad = sig_x * (1 - sig_x) * 2
        self.assertEqual(input.grad, expected_grad)

    @skipIfTorchDynamo("TorchDynamo does not work well with hooks")
    def test_hook_forward_preforward_writable(self):
        module = nn.Sigmoid()
        input = torch.randn(5, 5, requires_grad=True)
        sig_x = torch.nn.functional.sigmoid(input)

        def forward_pre_hook(m, input):
            return torch.nn.functional.relu(input[0])

        def forward_hook(m, input, output):
            return -output

        module.register_forward_pre_hook(forward_pre_hook)
        module.register_forward_hook(forward_hook)
        output = module(input)
        expected_res = -torch.nn.functional.sigmoid(torch.nn.functional.relu(input))
        self.assertEqual(output, expected_res)
        output.backward(torch.ones(5, 5) * 2, retain_graph=True)
        mask = (input > 0).double()
        expected_grad = -sig_x * (1 - sig_x) * 2 * mask
        self.assertEqual(input.grad, expected_grad)

    def test_hook_buffer_registration(self):
        for return_buffer in (True, False):
            def buffer_registration_hook(module, name, buffer):
                buffer.registered = True
                if return_buffer:
                    return buffer
            handle = torch.nn.modules.module.register_module_buffer_registration_hook(
                buffer_registration_hook
            )
            try:
                l, n, s = self._create_basic_net()
                for b in s.buffers():
                    self.assertTrue(getattr(b, "registered", False))
            finally:
                handle.remove()

    def test_hook_submodule_registration(self):
        for return_submodule in (True, False):
            def module_registration_hook(module, name, submodule):
                module.registered = True
                submodule.registered = True
                if return_submodule:
                    return submodule
            handle = torch.nn.modules.module.register_module_module_registration_hook(
                module_registration_hook
            )
            try:
                l, n, s = self._create_basic_net()
                for m in s.modules():
                    self.assertTrue(getattr(m, "registered", False))
            finally:
                handle.remove()

    def test_hook_parameter_registration(self):
        for return_parameter in (True, False):
            def parameter_registration_hook(module, name, parameter):
                parameter.registered = True
                if return_parameter:
                    return parameter
            handle = torch.nn.modules.module.register_module_parameter_registration_hook(
                parameter_registration_hook
            )
            try:
                l, n, s = self._create_basic_net()
                for p in s.parameters():
                    self.assertTrue(getattr(p, "registered", False))
            finally:
                handle.remove()

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

        # Force set to None.
        module.zero_grad(set_to_none=True)
        self.assertIsNone(module.weight.grad)


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

        # test remove_duplicate
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buffer1", torch.empty(3, 5))
                self.register_buffer("buffer2", self.buffer1)

        m = M()
        self.assertEqual(names(m.named_buffers()),
                         ["buffer1"])
        self.assertEqual(names(m.named_buffers(remove_duplicate=False)),
                         ["buffer1", "buffer2"])

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

    def test_train_errors_for_invalid_mode(self):
        class SubclassNet(nn.Module):
            def __init__(self):
                super(SubclassNet, self).__init__()
                self.l1 = nn.Linear(2, 2)

            def forward(self, inputs):
                return self.l1(inputs)

        subclass_net = SubclassNet()
        sequential_net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))

        error_modes = ["invalid_str", torch.device('cpu')]
        modules_to_check = [subclass_net, sequential_net]

        for error_mode, module in itertools.product(error_modes, modules_to_check):
            with self.assertRaises(ValueError):
                module.train(error_mode)

    def test_dir(self):
        linear = nn.Linear(2, 2)
        linear._test_submodule = nn.Linear(2, 2)
        linear._test_parameter = Parameter(torch.empty(2, 2))
        linear.register_buffer('_test_buffer', torch.empty(2, 2))
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
        s = nn.Sequential(n, n)
        self.assertEqual(list(s.named_modules()), [('', s), ('0', n), ('0.l1', l),
                                                   ('0.block', block), ('0.block.linear1', l1),
                                                   ('0.block.linear2', l2)])
        # test the option to not remove duplicate module instances
        self.assertEqual(list(s.named_modules(remove_duplicate=False)), [
            ('', s), ('0', n), ('0.l1', l), ('0.l2', l),
            ('0.block', block), ('0.block.linear1', l1),
            ('0.block.linear2', l2),
            ('1', n), ('1.l1', l), ('1.l2', l),
            ('1.block', block), ('1.block.linear1', l1),
            ('1.block.linear2', l2)])

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

    def test_get_buffer(self):
        m = nn.Module()
        buffer1 = torch.randn(2, 3)
        buffer2 = torch.randn(4, 5)
        m.register_buffer('foo', buffer1)
        m.register_buffer('bar', buffer2)
        self.assertEqual(buffer1, m.get_buffer('foo'))
        self.assertEqual(buffer2, m.get_buffer('bar'))

    def test_get_buffer_from_submodules(self):
        class MyModule(nn.Module):
            def __init__(self, foo, bar):
                super().__init__()
                self.sub = Sub(foo, bar)

        class Sub(nn.Module):
            def __init__(self, foo, bar):
                super().__init__()
                self.register_buffer('foo', foo)
                self.subsub = SubSub(bar)

        class SubSub(nn.Module):
            def __init__(self, bar):
                super().__init__()
                self.register_buffer('bar', bar)

        foo = torch.randn(2, 3)
        bar = torch.randn(4, 5)
        m = MyModule(foo, bar)
        self.assertEqual(foo, m.get_buffer('sub.foo'))
        self.assertEqual(bar, m.get_buffer('sub.subsub.bar'))

    def test_buffer_not_persistent(self):
        m = nn.Module()
        m.register_buffer('buf', torch.rand(5), persistent=False)
        self.assertTrue(len(list(m.buffers())) == 1)
        self.assertTrue(len(m.state_dict()) == 0)

    def test_buffer_not_persistent_del(self):
        m = nn.Module()
        m.register_buffer('buf', torch.rand(5), persistent=False)
        del m.buf
        self.assertTrue(len(list(m.buffers())) == 0)

    def test_buffer_not_persistent_overwrite(self):
        m = nn.Module()
        m.register_buffer('buf', torch.rand(5), persistent=False)
        m.register_buffer('buf', torch.rand(5))

        # can we overwrite a non-persistent buffer with a persistent one?
        self.assertTrue(len(list(m.buffers())) == 1)
        self.assertTrue(len(m.state_dict()) == 1)

        # can we overwrite a persistent buffer with a non-persistent one?
        m.register_buffer('buf', torch.rand(5), persistent=False)
        self.assertTrue(len(list(m.buffers())) == 1)
        self.assertTrue(len(m.state_dict()) == 0)

    def test_buffer_not_persistent_assign(self):
        m = nn.Module()
        m.register_buffer('buf', torch.rand(5), persistent=False)

        # Assigning None removes the buffer but if we then assign a new Tensor
        # to the same property, it should still be marked as a buffer.
        m.buf = None
        self.assertTrue(len(list(m.buffers())) == 0)
        self.assertTrue(len(m.state_dict()) == 0)
        m.buf = torch.rand(5)
        self.assertTrue(len(list(m.buffers())) == 1)
        self.assertTrue(len(m.state_dict()) == 0)

        # Assigning a Parameter removes the buffer.
        m.buf = nn.Parameter(torch.rand(5))
        self.assertTrue(len(list(m.buffers())) == 0)
        self.assertTrue(len(m.state_dict()) == 1)

    @unittest.skipIf(not TEST_NUMPY, "numpy not found")
    def test_load_state_dict_invalid(self):
        m = torch.nn.Linear(2, 2, bias=False)

        state_dict = {'weight': np.random.randn(2, 2)}
        with self.assertRaisesRegex(RuntimeError,
                                    "expected torch.Tensor or Tensor-like object from checkpoint but received"):
            m.load_state_dict(state_dict)

        state_dict = {'weight': ((1., 1.), (2., 2.))}
        with self.assertRaisesRegex(RuntimeError,
                                    "expected torch.Tensor or Tensor-like object from checkpoint but received"):
            m.load_state_dict(state_dict)

    def test_load_state_dict_type(self):
        m = nn.Module()

        with self.assertRaisesRegex(TypeError,
                                    "Expected state_dict to be dict-like, got"):
            m.load_state_dict("")
        with self.assertRaisesRegex(TypeError,
                                    "Expected state_dict to be dict-like, got"):
            m.load_state_dict(2)

    def test_buffer_not_persistent_load(self):
        m = nn.Module()
        m.register_buffer('buf', torch.rand(5), persistent=False)
        m.load_state_dict({})

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
        methods_to_test = ['add_module', 'register_module']
        for fn in methods_to_test:
            m = nn.Module()
            m.attribute_name = 5
            with self.assertRaises(KeyError):
                getattr(m, fn)('attribute_name', nn.Module())

            del m.attribute_name
            m.register_buffer('attribute_name', torch.rand(5))
            with self.assertRaises(KeyError):
                getattr(m, fn)('attribute_name', nn.Module())

            del m.attribute_name
            m.register_parameter('attribute_name', nn.Parameter())
            with self.assertRaises(KeyError):
                getattr(m, fn)('attribute_name', nn.Module())

    @unittest.expectedFailure
    def test_getattr_with_property(self):
        class Model(nn.Module):
            @property
            def some_property(self):
                return self.something_that_doesnt_exist

        model = Model()

        with self.assertRaisesRegex(
                AttributeError,
                r"'Model' object has no attribute 'something_that_doesnt_exist'"):
            model.some_property

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

    def test_Sequential_add(self):
        l1 = nn.Linear(1, 2)
        l2 = nn.Linear(2, 3)
        l3 = nn.Linear(3, 4)
        l4 = nn.Linear(4, 5)
        n = nn.Sequential(l1, l2)
        other = nn.Sequential(l3, l4)
        self.assertEqual(n + other, nn.Sequential(l1, l2, l3, l4))

    def test_Sequential_iadd(self):
        l1 = nn.Linear(10, 20)
        l2 = nn.Linear(20, 30)
        l3 = nn.Linear(30, 40)
        l4 = nn.Linear(40, 50)
        n = nn.Sequential(l1, l2, l3)
        n2 = nn.Sequential(l4)
        n += n2
        n2 += n
        self.assertEqual(n, nn.Sequential(l1, l2, l3, l4))
        self.assertEqual(n2, nn.Sequential(l4, l1, l2, l3, l4))

    def test_Sequential_mul(self):
        l1 = nn.Linear(10, 20)
        l2 = nn.Linear(20, 30)
        l3 = nn.Linear(30, 40)
        l4 = nn.Linear(40, 50)
        n = nn.Sequential(l1, l2, l3, l4)
        n2 = n * 2
        self.assertEqual(n2, nn.Sequential(l1, l2, l3, l4, l1, l2, l3, l4))

    def test_Sequential_rmul(self):
        l1 = nn.Linear(10, 20)
        l2 = nn.Linear(20, 30)
        l3 = nn.Linear(30, 40)
        l4 = nn.Linear(40, 50)
        n = nn.Sequential(l1, l2, l3, l4)
        n2 = 2 * n
        self.assertEqual(n2, nn.Sequential(l1, l2, l3, l4, l1, l2, l3, l4))

    def test_Sequential_imul(self):
        l1 = nn.Linear(10, 20)
        l2 = nn.Linear(20, 30)
        l3 = nn.Linear(30, 40)
        l4 = nn.Linear(40, 50)
        n = nn.Sequential(l1, l2, l3, l4)
        n *= 2
        self.assertEqual(n, nn.Sequential(l1, l2, l3, l4, l1, l2, l3, l4))
        n *= 2
        self.assertEqual(
            n,
            nn.Sequential(l1, l2, l3, l4, l1, l2, l3, l4, l1, l2, l3, l4, l1, l2, l3, l4)
        )

    def test_Sequential_append(self):
        l1 = nn.Linear(10, 20)
        l2 = nn.Linear(20, 30)
        l3 = nn.Linear(30, 40)
        l4 = nn.Linear(40, 50)
        n = nn.Sequential(l1, l2, l3)
        n2 = n.append(l4)
        self.assertEqual(n, nn.Sequential(l1, l2, l3, l4))
        self.assertEqual(n2, nn.Sequential(l1, l2, l3, l4))
        self.assertEqual(nn.Sequential(l1).append(l2).append(l4), nn.Sequential(l1, l2, l4))

    def test_Sequential_pop(self):
        l1 = nn.Linear(1, 2)
        l2 = nn.Linear(2, 3)
        l3 = nn.Linear(3, 4)
        l4 = nn.Linear(4, 5)
        n1 = nn.Sequential(l1, l2, l3, l4)
        self.assertEqual(l4, n1.pop(3))
        n2 = nn.Sequential(l1, l2, l3)
        self.assertEqual(n1, n2)
        # check order of the index
        for k, mod in zip(range(len(n1)), n1):
            self.assertIs(n1[k], mod)

    def test_Sequential_insert(self):
        l1 = nn.Linear(1, 2)
        l2 = nn.Linear(2, 3)
        l3 = nn.Linear(3, 4)

        n1 = nn.Sequential(l1, l2, l3)
        module_1 = nn.Linear(4, 5)
        n2 = nn.Sequential(l1, module_1, l2, l3)
        self.assertEqual(n1.insert(1, module_1), n2)

        # test for negative support
        n3 = nn.Sequential(l1, l2, l3)
        module_2 = nn.Linear(5, 6)
        n4 = nn.Sequential(l1, module_2, l2, l3)
        self.assertEqual(n3.insert(-2, module_2), n4)

    def test_Sequential_insert_fail_case(self):
        l1 = nn.Linear(1, 2)
        l2 = nn.Linear(2, 3)
        l3 = nn.Linear(3, 4)

        module = nn.Linear(5, 6)

        # test for error case
        n1 = nn.Sequential(l1, l2, l3)
        with self.assertRaises(IndexError):
            n1.insert(-5, module)

        with self.assertRaises(AssertionError):
            n1.insert(1, [nn.Linear(6, 7)])

    def test_Sequential_extend(self):
        l1 = nn.Linear(10, 20)
        l2 = nn.Linear(20, 30)
        l3 = nn.Linear(30, 40)
        l4 = nn.Linear(40, 50)
        n1 = nn.Sequential(l1, l2)
        n2 = nn.Sequential(l3, l4)
        n3 = nn.Sequential(l1, l2)
        for l in n2:
            n1.append(l)
        n3.extend(n2)
        self.assertEqual(n3, n1)

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
        modules = modules + [nn.Conv2d(3, 4, 3, bias=False), nn.GELU()]
        module_list = module_list + nn.ModuleList(modules[-2:])
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

        modules = [nn.ReLU(), nn.Linear(5, 5), nn.Conv2d(3, 4, 3)]
        module_list = nn.ModuleList(modules)
        self.assertEqual(modules.pop(1), module_list.pop(1))
        self.assertEqual(modules, module_list)
        # check order of the index
        for k, mod in zip(range(len(module_list)), module_list):
            self.assertIs(module_list[k], mod)

        # verify the right exception is thrown when trying to "forward" through a ModuleList
        self.assertRaises(NotImplementedError, module_list)
        self.assertRaises(NotImplementedError, module_list, torch.rand(1, 3))

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
        modules.update(next_modules.items())
        module_dict.update(next_modules)
        check()

        next_modules = nn.ModuleDict([
            ('fc5', nn.Linear(5, 5)),
            ('act4', nn.Sigmoid()),
        ])
        modules.update(next_modules)
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

        # verify the right exception is thrown when trying to "forward" through a ModuleDict
        self.assertRaises(NotImplementedError, module_dict)
        self.assertRaises(NotImplementedError, module_dict, torch.rand(1, 3))

    def test_ParameterList(self):
        def make_param():
            return Parameter(torch.randn(2, 2))
        parameters = [make_param(), make_param()]
        param_list = nn.ParameterList(parameters)

        def check():
            self.assertEqual(len(parameters), len(param_list))
            for p1, p2 in zip(parameters, param_list):
                self.assertIs(p1, p2)
            for p1, p2 in zip(filter(lambda x: isinstance(x, Parameter), parameters), param_list.parameters()):
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

        param_list.append(torch.rand(2, 2))
        self.assertIsInstance(param_list[-1], Parameter)
        parameters.append(param_list[-1])

        param_list.extend([torch.rand(2, 2), "foo"])
        self.assertIsInstance(param_list[-2], Parameter)
        self.assertIsInstance(param_list[-1], str)
        parameters.extend(param_list[-2:])

        param_list += ["bar", torch.rand(2, 2)]
        self.assertIsInstance(param_list[-2], str)
        self.assertIsInstance(param_list[-1], Parameter)
        parameters += param_list[-2:]
        check()

    def test_ParameterList_meta(self):
        p = torch.nn.Parameter(torch.empty(1, device='meta'))
        self.assertExpectedInline(str(p), """\
Parameter containing:
tensor(..., device='meta', size=(1,), requires_grad=True)""")
        pl = torch.nn.ParameterList([p])
        self.assertExpectedInline(str(pl), """ParameterList(  (0): Parameter containing: [torch.float64 of size 1])""")

    def test_ParameterList_replication(self):
        # The actual replication code from DP cannot be used on CPU so doing it manually here
        def make_param():
            return Parameter(torch.randn(2, 2))
        parameters = [make_param(), make_param()]
        param_list = nn.ParameterList(parameters)

        new_param_list = param_list._replicate_for_data_parallel()

        for n, p in param_list.named_parameters():
            # Do a view here so that we can check the base later
            setattr(new_param_list, n, p.view_as(p))

        for p, p2 in zip(param_list, new_param_list):
            self.assertEqual(p, p2)
            self.assertIsNotNone(p2.grad_fn)
            self.assertIs(p2._base, p)

    def test_ParameterDict(self):
        parameters = OrderedDict([
            ('p1', Parameter(torch.randn(10, 10))),
            ('p2', Parameter(torch.randn(10, 10))),
            ('p3', Parameter(torch.randn(10, 10))),
        ])

        parameter_dict = nn.ParameterDict(parameters)

        def check():
            self.assertEqual(len(parameter_dict), len(parameters))
            for i, (k1, (k2, m2)) in enumerate(zip(parameters, parameter_dict.named_parameters())):
                self.assertEqual(k1, k2)
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

        next_parameters = nn.ParameterDict([
            ('p10', Parameter(torch.randn(10, 10))),
            ('p9', Parameter(torch.randn(10, 10))),
        ])
        parameters.update(next_parameters)
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

        p_pop = parameter_dict.pop('p4')
        self.assertIs(p_pop, parameters['p4'])
        parameters.pop('p4')
        check()

        # Check reverse works
        forward = list(iter(parameter_dict))
        backward = list(reversed(parameter_dict))
        self.assertEqual(len(forward), len(backward))
        n = len(forward)
        for i in range(n):
            self.assertIs(forward[i], backward[n - i - 1])
        check()

        # Check copy works
        copy = parameter_dict.copy()

        # Check all keys are present and have shallow copied values
        for key in parameter_dict:
            self.assertTrue(key in copy)
            self.assertEqual(parameter_dict[key], copy[key])
            self.assertIs(parameter_dict[key], copy[key])
        check()

        parameter_dict["p20"] = Parameter(torch.randn(10, 10))
        copy["p21"] = Parameter(torch.randn(9, 10))

        self.assertTrue("p20" in parameter_dict)
        self.assertFalse("p20" in copy)
        self.assertFalse("p21" in parameter_dict)
        self.assertTrue("p21" in copy)
        parameter_dict.pop("p20")
        check()

        p = Parameter(torch.randn(10, 10))
        parameter_dict['p12'] = p
        p_popitem = parameter_dict.popitem()
        self.assertEqual(p_popitem[0], 'p12')
        self.assertIs(p_popitem[1], p)
        check()

        # Unit test for set_default
        # 1. Ensure parameter is correctly inserted when
        #    the key is not present in `ParameterDict`
        assert 'p11' not in parameter_dict
        assert 'p11' not in parameters
        parameters['p11'] = Parameter(torch.randn(10, 10))
        p_setdefault = parameter_dict.setdefault('p11', parameters['p11'])
        self.assertIs(p_setdefault, parameters['p11'])
        self.assertIs(p_setdefault, parameter_dict['p11'])
        check()
        # 2. Ensure parameter is NOT inserted when the
        #    key is already present in `ParameterDict`
        p = Parameter(torch.randn(10, 10))
        self.assertFalse(parameter_dict.setdefault('p11', p) is p)
        check()
        # 3. Ensure `None` is inserted when the key is not
        #    present in `Parameter` and parameter is not specified
        self.assertIs(parameter_dict.setdefault('p26'), None)
        del parameter_dict['p26']
        check()

        parameters2 = OrderedDict([
            ('p13', Parameter(torch.randn(10, 10))),
            ('p2', Parameter(torch.randn(10, 10))),
            ('p3', Parameter(torch.randn(10, 10))),
        ])
        parameter_dict2 = nn.ParameterDict(parameters2)
        parameters.update(parameters2)
        parameter_dict |= parameter_dict2
        check()

        parameters2 = OrderedDict()
        parameter_dict2 = nn.ParameterDict(parameters2)
        parameters.update(parameters2)
        parameter_dict |= parameter_dict2
        check()

        parameters2 = OrderedDict([
            ('p14', Parameter(torch.randn(10, 10))),
            ('p15', Parameter(torch.randn(10, 10))),
            ('p13', Parameter(torch.randn(10, 10))),
        ])
        parameter_dict2 = nn.ParameterDict(parameters2)
        parameters.update(parameters2)
        parameter_dict |= parameter_dict2
        check()

        # Check __or__ and __ror__ works
        parameters2 = OrderedDict([
            ('p20', Parameter(torch.randn(10, 10))),
            ('p21', Parameter(torch.randn(10, 10))),
            ('p22', Parameter(torch.randn(10, 10))),
        ])
        parameter_dict2 = nn.ParameterDict(parameters2)
        parameters.update(parameters2)
        parameter_dict = parameter_dict | parameter_dict2
        check()

        parameters2 = OrderedDict([
            ('p23', Parameter(torch.randn(10, 10))),
            ('p24', Parameter(torch.randn(10, 10))),
            ('p25', Parameter(torch.randn(10, 10))),
        ])
        parameter_dict2 = nn.ParameterDict(parameters2)
        parameters2.update(parameters)
        parameters = parameters2
        parameter_dict = parameter_dict2 | parameter_dict
        check()

        parameters['p17'] = Parameter(torch.randn(10, 10))
        parameter_dict['p17'] = parameters['p17']
        self.assertIs(parameters['p17'], parameter_dict.get('p17'))
        temp_param = Parameter(torch.randn(10, 10))
        self.assertIs(parameters['p17'], parameter_dict.get('p17', temp_param))
        self.assertIs(None, parameter_dict.get('p18'))
        self.assertIs(temp_param, parameter_dict.get('p18', temp_param))
        check()

        parameter_dict.clear()
        self.assertEqual(len(parameter_dict), 0)
        parameters.clear()
        check()

        parameter_dict2 = parameter_dict.fromkeys(['p19', 'p20'])
        self.assertEqual({'p19': None, 'p20': None}, parameter_dict2)
        check()

        parameter_dict2 = parameter_dict.fromkeys(['p19', 'p20'], temp_param)
        self.assertEqual({'p19': temp_param, 'p20': temp_param}, parameter_dict2)
        check()

        parameter_dict['p21'] = torch.rand(2, 2)
        self.assertIsInstance(parameter_dict['p21'], Parameter)
        parameters['p21'] = parameter_dict['p21']

        parameter_dict.update({'p22': torch.rand(2, 2), 'foo': 'bar'})
        self.assertIsInstance(parameter_dict['p22'], Parameter)
        self.assertIsInstance(parameter_dict['foo'], str)
        parameters['p22'] = parameter_dict['p22']
        parameters['foo'] = parameter_dict['foo']

    def test_ParameterDict_replication(self):
        # The actual replication code from DP cannot be used on CPU so doing it manually here
        def make_param():
            return Parameter(torch.randn(2, 2))
        parameters = {"foo": make_param(), "bar": make_param()}
        param_dict = nn.ParameterDict(parameters)

        new_param_dict = param_dict._replicate_for_data_parallel()

        for n, p in param_dict.named_parameters():
            # Do a view here so that we can check the base later
            setattr(new_param_dict, n, p.view_as(p))

        for (k, p), (k2, p2) in zip(param_dict.items(), new_param_dict.items()):
            self.assertEqual(k, k2)
            self.assertEqual(p, p2)
            self.assertIsNotNone(p2.grad_fn)
            self.assertIs(p2._base, p)

        self.assertEqual(param_dict["foo"], new_param_dict["foo"])

    def test_add_module(self):
        methods_to_test = ['add_module', 'register_module']
        for fn in methods_to_test:
            l = nn.Linear(10, 20)
            net = nn.Module()
            net.l = l
            net.l2 = l
            getattr(net, fn)('empty', None)
            self.assertEqual(net.l, l)
            self.assertEqual(net.l2, l)
            self.assertEqual(net.empty, None)
            getattr(net, fn)('l3', l)
            self.assertEqual(net.l3, l)
            l3 = nn.Linear(20, 10)
            getattr(net, fn)('l', l3)
            self.assertEqual(net.l, l3)
            self.assertRaises(TypeError, lambda: getattr(net, fn)('x', 'non-module'))
            self.assertRaisesRegex(TypeError, 'module name should be a string. Got int',
                                   lambda: getattr(net, fn)(1, l))
            self.assertRaisesRegex(TypeError, 'module name should be a string. Got NoneType',
                                   lambda: getattr(net, fn)(None, l))

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
        with torch.no_grad():
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
            with torch.no_grad():
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
        net.to(torch.float)
        self.assertIsInstance(l.weight.data, torch.FloatTensor)
        self.assertIsInstance(l.bias.data, torch.FloatTensor)
        net.to(torch.DoubleTensor(1))
        self.assertIsInstance(l.weight.data, torch.DoubleTensor)
        self.assertIsInstance(l.bias.data, torch.DoubleTensor)
        if TEST_CUDA:
            net.to(device='cuda', dtype=torch.float)
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

    # FIXME: Rewrite this test using functions not depending on LAPACK
    #        and remove the `@skipIfNoLapack` (see #70995)
    # torch/nn/utils/parametrize
    @skipIfNoLapack
    def test_register_and_remove_parametrization(self):
        r"""Test that it is possible to add a few parametrizations
        on a parameter or a buffer and that removing them restores the initial state
        It also tests that backpropagating through them works as expected
        """
        # Define a couple matrix parametrizations
        class Skew(nn.Module):
            def forward(self, X):
                X = X.tril(-1)
                return X - X.T

        class Orthogonal(nn.Module):
            def forward(self, X):
                # Cayley map
                # If X is skew-symmetric it returns an orthogonal matrix
                Id = torch.eye(X.size(0), device=X.device)
                # We call contiguous because solve returns a tensor with strides that are Fortran-contiguous
                # and autograd raises a performance warning.
                # This happens when we remove the parametrization with leave_parametrized=True,
                # which does a set_ with a non-contiguous tensor while the gradient is contiguous
                return torch.linalg.solve(Id + X, Id - X).contiguous()

        class Resize(nn.Module):
            def forward(self, X):
                return X[[0]]

        class NoResize(nn.Module):
            def forward(self, X):
                return X

        # Define a couple vector parametrizations
        class FirstZero(nn.Module):
            def forward(self, x):
                return torch.cat([x.new_zeros(1), x[1:]])

        class LastZero(nn.Module):
            def forward(self, x):
                return torch.cat([x[:-1], x.new_zeros(1)])

        model = nn.Linear(8, 8)
        initial_weight_id = id(model.weight)
        initial_bias_id = id(model.bias)
        initial_model = deepcopy(model)

        # Test unsafe flag
        with self.assertRaisesRegex(ValueError, "Registering a parametrization may not change the shape of the tensor"):
            parametrize.register_parametrization(model, "weight", Resize())  # default unsafe = False
            model(torch.ones(8, 8))

        # One parametrization with unsafe=True
        parametrize.register_parametrization(model, "weight", Resize(), unsafe=True)
        self.assertTrue(hasattr(model, "parametrizations"))
        self.assertTrue(parametrize.is_parametrized(model))
        self.assertTrue(parametrize.is_parametrized(model, "weight"))
        self.assertFalse(parametrize.is_parametrized(model, "bias"))
        self.assertNotIn("weight", model._parameters)
        A = model.weight
        self.assertTrue(A.shape[0] == 1)
        parametrize.remove_parametrizations(model, "weight", leave_parametrized=False)
        self.assertFalse(hasattr(model, "parametrizations"))
        self.assertEqual(model.weight, initial_model.weight)
        self.assertEqual(id(model.weight), initial_weight_id)
        self.assertEqual(model.__class__, nn.Linear)

        # Two parametrizations with unsafe=True
        parametrize.register_parametrization(model, "weight", Resize(), unsafe=True)
        parametrize.register_parametrization(model, "weight", NoResize(), unsafe=False)
        self.assertTrue(hasattr(model, "parametrizations"))
        self.assertTrue(parametrize.is_parametrized(model))
        self.assertTrue(parametrize.is_parametrized(model, "weight"))
        self.assertFalse(parametrize.is_parametrized(model, "bias"))
        self.assertNotIn("weight", model._parameters)
        A = model.weight
        self.assertTrue(A.shape[0] == 1)
        parametrize.remove_parametrizations(model, "weight", leave_parametrized=False)
        self.assertFalse(hasattr(model, "parametrizations"))
        self.assertEqual(model.weight, initial_model.weight)
        self.assertEqual(id(model.weight), initial_weight_id)
        self.assertEqual(model.__class__, nn.Linear)

        # Test unsafe flag doesn't change expected behavior
        parametrize.register_parametrization(model, "weight", Skew(), unsafe=True)
        self.assertTrue(hasattr(model, "parametrizations"))
        self.assertTrue(parametrize.is_parametrized(model))
        self.assertTrue(parametrize.is_parametrized(model, "weight"))
        self.assertFalse(parametrize.is_parametrized(model, "bias"))
        self.assertNotIn("weight", model._parameters)
        # Result should be skew-symmetric
        A = model.weight
        self.assertEqual(A, -A.T)
        # Remove and check consistency
        parametrize.remove_parametrizations(model, "weight", leave_parametrized=False)
        self.assertFalse(hasattr(model, "parametrizations"))
        self.assertEqual(model.weight, initial_model.weight)
        self.assertEqual(id(model.weight), initial_weight_id)
        self.assertEqual(model.__class__, nn.Linear)

        # Test one parametrization
        parametrize.register_parametrization(model, "weight", Skew())
        self.assertTrue(hasattr(model, "parametrizations"))
        self.assertTrue(parametrize.is_parametrized(model))
        self.assertTrue(parametrize.is_parametrized(model, "weight"))
        self.assertFalse(parametrize.is_parametrized(model, "bias"))
        self.assertNotIn("weight", model._parameters)
        # Result should be skew-symmetric
        A = model.weight
        self.assertEqual(A, -A.T)
        # Remove and check consistency
        parametrize.remove_parametrizations(model, "weight", leave_parametrized=False)
        self.assertFalse(hasattr(model, "parametrizations"))
        self.assertEqual(model.weight, initial_model.weight)
        self.assertEqual(id(model.weight), initial_weight_id)
        self.assertEqual(model.__class__, nn.Linear)

        # Test two parametrizations at the same time and removing them
        parametrize.register_parametrization(model, "weight", Skew())
        parametrize.register_parametrization(model, "weight", Orthogonal())
        # Result should be orthogonal
        X = model.weight
        Id = torch.eye(X.size(0), device=X.device)
        self.assertEqual(X.T @ X, Id)
        # Structure tests
        self.assertTrue(hasattr(model, "parametrizations"))
        self.assertTrue(parametrize.is_parametrized(model))
        self.assertTrue(parametrize.is_parametrized(model, "weight"))
        self.assertFalse(parametrize.is_parametrized(model, "bias"))
        self.assertIn("weight", model.parametrizations)
        self.assertNotIn("weight", model._parameters)
        # Remove
        parametrize.remove_parametrizations(model, "weight", leave_parametrized=False)
        self.assertEqual(model.weight, initial_model.weight)
        self.assertEqual(id(model.weight), initial_weight_id)
        self.assertFalse(hasattr(model, "parametrizations"))
        self.assertEqual(model.__class__, nn.Linear)

        # Add everything
        parametrize.register_parametrization(model, "weight", Skew())
        parametrize.register_parametrization(model, "weight", Orthogonal())
        parametrize.register_parametrization(model, "bias", FirstZero())
        parametrize.register_parametrization(model, "bias", LastZero())

        # Basic tests
        self.assertTrue(parametrize.is_parametrized(model))
        self.assertTrue(parametrize.is_parametrized(model, "weight"))
        self.assertTrue(parametrize.is_parametrized(model, "bias"))
        self.assertEqual(model.bias[0].item(), 0.)
        self.assertEqual(model.bias[-1].item(), 0.)
        self.assertEqual(len(list(model.parameters())), 2)  # Nothing weird has happpened
        # Should not throw

        sgd = torch.optim.SGD(model.parameters(), lr=0.01)

        weight_copy = model.weight.clone()
        bias_copy = model.bias.clone()
        sgd.zero_grad()
        (model.weight.T @ model.bias).sum().backward()
        sgd.step()
        self.assertNotEqual(model.weight, weight_copy)
        self.assertNotEqual(model.bias, bias_copy)

        # Remove first parametrization.
        # Check that the model is still parametrized and so is the second parameter
        parametrize.remove_parametrizations(model, "weight", leave_parametrized=False)
        self.assertTrue(parametrize.is_parametrized(model))             # Still parametrized
        self.assertFalse(parametrize.is_parametrized(model, "weight"))  # Parametrization removed
        self.assertTrue(parametrize.is_parametrized(model, "bias"))     # Still parametrized
        self.assertEqual(model.bias[0].item(), 0.)                      # Still parametrized
        self.assertEqual(model.bias[-1].item(), 0.)                     # Still parametrized
        self.assertNotEqual(model.weight, initial_model.weight)         # Has been updated
        self.assertEqual(id(model.weight), initial_weight_id)           # Keeps the same id
        self.assertEqual(len(list(model.parameters())), 2)              # Nothing weird has happened
        # Should not throw
        weight_copy = model.weight.clone()
        bias_copy = model.bias.clone()
        sgd.zero_grad()
        (model.weight.T @ model.bias).sum().backward()
        sgd.step()
        self.assertNotEqual(model.weight, weight_copy)
        self.assertNotEqual(model.bias, bias_copy)

        # Remove the second parametrization.
        # Check that the module is not parametrized
        parametrize.remove_parametrizations(model, "bias", leave_parametrized=False)
        self.assertFalse(parametrize.is_parametrized(model))  # Not parametrized
        self.assertNotEqual(model.bias, initial_model.bias)   # Has been updated
        self.assertNotEqual(model.bias[0].item(), 0.)         # Not parametrized
        self.assertNotEqual(model.bias[-1].item(), 0.)        # Not parametrized
        self.assertEqual(id(model.bias), initial_bias_id)     # Keeps the same id
        self.assertFalse(hasattr(model, "parametrizations"))  # Not parametrized the module
        self.assertEqual(model.__class__, nn.Linear)          # Resores the previous class
        self.assertEqual(len(list(model.parameters())), 2)    # Nothing weird has happeed

        # Should not throw things are updated
        weight_copy = model.weight.clone()
        bias_copy = model.bias.clone()
        sgd.zero_grad()
        (model.weight.T @ model.bias).sum().backward()
        sgd.step()
        self.assertNotEqual(model.weight, weight_copy)
        self.assertNotEqual(model.bias, bias_copy)

        # Test leave_parametrized=True
        for _ in range(2):
            parametrize.register_parametrization(model, "weight", Skew())
            parametrize.register_parametrization(model, "weight", Orthogonal())
            parametrize.remove_parametrizations(model, "weight", leave_parametrized=True)
            # We didn't change the dtype nor had multiple inputs, so the id should be the same
            self.assertEqual(id(model.weight), initial_weight_id)
            self.assertEqual(id(model.bias), initial_bias_id)

            # Should not throw. Things are updated
            weight_copy = model.weight.clone()
            bias_copy = model.bias.clone()
            sgd.zero_grad()
            (model.weight.T @ model.bias).sum().backward()
            sgd.step()
            self.assertNotEqual(model.weight, weight_copy)
            self.assertNotEqual(model.bias, bias_copy)

    def test_register_and_remove_nested_parametrization(self):
        r"""Test that it is possible to nest the parametrizations
        meaning that the original param is parametrized again
        """
        class Skew(nn.Module):
            def forward(self, X):
                X = X.tril(-1)
                return X - X.T

        model = nn.Linear(8, 8)
        # Add top level parametrization
        parametrize.register_parametrization(model, "weight", Skew())
        self.assertTrue(hasattr(model, "parametrizations"))
        self.assertTrue(parametrize.is_parametrized(model))
        self.assertTrue(parametrize.is_parametrized(model, "weight"))
        self.assertFalse(parametrize.is_parametrized(model, "bias"))
        self.assertNotIn("weight", model._parameters)
        # Result should be skew-symmetric
        A = model.weight
        self.assertEqual(A, -A.T)

        # Add nested parametrization
        param_mod = model.parametrizations.weight
        self.assertFalse(hasattr(param_mod, "parametrizations"))
        self.assertFalse(parametrize.is_parametrized(param_mod))
        self.assertFalse(parametrize.is_parametrized(param_mod, "original"))

        parametrize.register_parametrization(param_mod, "original", Skew())
        self.assertTrue(hasattr(param_mod, "parametrizations"))
        self.assertTrue(parametrize.is_parametrized(param_mod))
        self.assertTrue(parametrize.is_parametrized(param_mod, "original"))
        self.assertNotIn("original", param_mod._parameters)
        # Result should be skew-symmetric
        A = param_mod.original
        self.assertEqual(A, -A.T)

        # Remove nested param and check consistency
        parametrize.remove_parametrizations(param_mod, "original", leave_parametrized=False)
        self.assertFalse(hasattr(param_mod, "parametrizations"))
        self.assertEqual(param_mod.__class__, parametrize.ParametrizationList)

        # Remove top level and check consistency
        parametrize.remove_parametrizations(model, "weight", leave_parametrized=False)
        self.assertFalse(hasattr(model, "parametrizations"))
        self.assertEqual(model.__class__, nn.Linear)

    def test_register_and_remove_buffer_parametrization(self):
        r"""Test that it is possible to add and remove parametrizations on buffers"""
        # Define a couple vector parametrizations
        class FirstZero(nn.Module):
            def forward(self, x):
                return torch.cat([x.new_zeros(1), x[1:]])

        class LastZero(nn.Module):
            def forward(self, x):
                return torch.cat([x[:-1], x.new_zeros(1)])

        model = nn.Linear(8, 8)

        # Instantiate parametrizations on buffers. It should work as expected
        delattr(model, "bias")
        model.register_buffer("bias", torch.ones(8))
        parametrize.register_parametrization(model, "bias", FirstZero())
        parametrize.register_parametrization(model, "bias", LastZero())
        self.assertTrue(parametrize.is_parametrized(model))
        self.assertTrue(parametrize.is_parametrized(model, "bias"))
        self.assertEqual(model.bias[0].item(), 0.)
        self.assertEqual(model.bias[-1].item(), 0.)
        self.assertTrue((model.bias[1:-1] == torch.ones(6)).all())
        self.assertEqual(len(list(model.parameters())), 1)

        # Remove parametrizations on buffers. It should work as expected
        parametrize.remove_parametrizations(model, "bias", leave_parametrized=True)
        self.assertFalse(parametrize.is_parametrized(model))
        self.assertFalse(parametrize.is_parametrized(model, "bias"))
        self.assertEqual(model.bias[0].item(), 0.)
        self.assertEqual(model.bias[-1].item(), 0.)
        self.assertTrue((model.bias[1:-1] == torch.ones(6)).all())
        self.assertEqual(len(list(model.parameters())), 1)

    # FIXME: Rewrite this test using functions not depending on LAPACK
    #        and remove the `@skipIfNoLapack` (see #70995)
    @skipIfNoLapack
    def test_serialization_parametrization(self):
        r"""Test that it is possible to serialize a parametrized model via state_dict"""
        # A stateful parametrization
        class Orthogonal(nn.Module):
            def __init__(self, n):
                super().__init__()
                self.register_buffer("id", torch.eye(n))
                self.register_buffer("B", torch.empty(n, n))
                init.orthogonal_(self.B)

            def forward(self, X):
                A = X.triu(1)
                A = A - A.T
                return self.B @ torch.linalg.solve(self.id + A, self.id - A)

        def get_model():
            model = torch.nn.Sequential(
                torch.nn.Linear(5, 5),
                torch.nn.ReLU(),
                torch.nn.Linear(5, 1),
            )

            parametrize.register_parametrization(model[0], "weight", Orthogonal(5))
            return model

        model = get_model()

        prev_weight = model[0].weight
        prev_B = model[0].parametrizations.weight[0].B

        new_model = get_model()
        with TemporaryFileName() as fname:
            torch.save(model.state_dict(), fname)
            new_model.load_state_dict(torch.load(fname))

        # Integrity tests
        self.assertTrue(parametrize.is_parametrized(new_model[0], "weight"))
        self.assertEqual(prev_weight, new_model[0].weight)
        self.assertEqual(prev_B, new_model[0].parametrizations.weight[0].B)

        # Trying to save the whole parametrized model raises
        with self.assertRaisesRegex(RuntimeError, "state_dict"):
            with TemporaryFileName() as fname:
                torch.save(model, fname)

    # FIXME: Rewrite this test using functions not depending on LAPACK
    #        and remove the `@skipIfNoLapack` (see #70995)
    @skipIfNoLapack
    def test_initialization_parametrization(self):
        r"""Test that it is possible to initialize a parametrization when it
            implements a `right_inverse` method
        """
        class Skew(nn.Module):
            def forward(self, X):
                A = X.triu(1)
                return A - A.T

            def is_skew(self, A):
                return torch.allclose(A, -A.T, atol=1e-6)

            def right_inverse(self, X):
                if not self.is_skew(X):
                    raise ValueError("The matrix is not skew-symmetric.")
                return X.triu(1)

        # Implements a Cayley map where right_inverse is not quite the inverse of forward
        class Orthogonal(nn.Module):
            def __init__(self, n):
                super().__init__()
                self.register_buffer("B", torch.eye(n))

            def forward(self, X):
                Id = torch.eye(X.size(0))
                return self.B @ torch.linalg.solve(Id + X, Id - X)

            def is_orthogonal(self, X):
                Id = torch.eye(X.size(0))
                return torch.allclose(X.T @ X, Id, atol=1e-4)

            def right_inverse(self, X):
                if not self.is_orthogonal(X):
                    raise ValueError("The input is not orthogonal.")
                # cayley(0) == Id, so B @ cayley(0) == B
                self.B = X
                return torch.zeros_like(X)

        N = 5
        model = nn.Linear(N, N)
        # Register the skew-symmetric constraint. The result is now skew-symmetric
        skew = Skew()
        # Make the weight skew-symmetric before registering the parametrization
        with torch.no_grad():
            model.weight.set_(skew(model.weight))
        parametrize.register_parametrization(model, "weight", skew)
        X = torch.rand(N, N)
        # X is not skew-symmetric, so it throws an error
        with self.assertRaises(ValueError):
            model.weight = X
        # Make X skew-symmetric
        X = X - X.T
        model.weight = X
        self.assertEqual(model.parametrizations.weight.original, X.triu(1))
        self.assertEqual(model.weight, X)

        # Having several parametrizations registered should work in the same way
        parametrize.register_parametrization(model, "weight", Orthogonal(N))
        # Register now the Cayley map. The result is now orthogonal
        X = torch.rand(N, N)
        # X is not orthogonal, so it throws an error
        with self.assertRaises(ValueError):
            model.weight = X
        init.orthogonal_(X)
        model.weight = X
        self.assertEqual(model.weight, X)
        self.assertEqual(model.parametrizations.weight.original, torch.zeros_like(X))

    def test_errors_unparametrized_tensor_parametrization(self):
        # Test errors when registering a parametrization on an unparametrized tensor
        module = nn.Linear(3, 4)
        weight_init = module.weight.clone()

        class Identity(nn.Module):
            def forward(self, x):
                return x

        # Register a parametrization on a non-existing parameter throws
        with self.assertRaisesRegex(ValueError, "does not have a parameter"):
            parametrize.register_parametrization(module, "foo", Identity())
        self.assertFalse(parametrize.is_parametrized(module))

        # Removing parametrizations from an unparametrized tensor throws
        with self.assertRaisesRegex(ValueError, "does not have a parametrization"):
            parametrize.remove_parametrizations(module, "bias")
        self.assertFalse(parametrize.is_parametrized(module))

        # A correct parametrization with several outputs
        class Sum(nn.Module):
            def forward(self, x, y):
                return x + y

            def right_inverse(self, z):
                return z, torch.zeros_like(z)

        parametrize.register_parametrization(module, "weight", Sum())
        # Cannot remove a parametrization with several outputs with `leave_parametrized=False`
        with self.assertRaisesRegex(ValueError, "leave_parametrized=False"):
            parametrize.remove_parametrizations(module, "weight", leave_parametrized=False)
        parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)

        # A parametrization with an incorrect number of outputs
        class WrongNumberParams(nn.Module):
            def forward(self, x, y, z):
                return x + y + z

            def right_inverse(self, w):
                return w, torch.zeros_like(w)

        # Makes param(*param.right_inverse(X)) fail
        with self.assertRaisesRegex(TypeError, "positional argument"):
            parametrize.register_parametrization(module, "weight", WrongNumberParams())
        self.assertFalse(parametrize.is_parametrized(module))

        # A parametrization with a right_inverse that does not return a Tensor or Sequence[Tensor]
        class WrongRightInverse(Identity):
            def right_inverse(self, z):
                return None

        # right_inverse should return a Tensor or a Sequence[Tensor]
        with self.assertRaisesRegex(ValueError, "Tensor or a Sequence of"):
            parametrize.register_parametrization(module, "weight", WrongRightInverse())
        self.assertFalse(parametrize.is_parametrized(module))

        # If it's a sequence, it must to be a sequence of tensors
        class WrongRightInverseSequence(nn.Module):
            def forward(self, x, y):
                return x

            def right_inverse(self, z):
                return None, z

        with self.assertRaisesRegex(ValueError, "of the sequence with type"):
            parametrize.register_parametrization(module, "weight", WrongRightInverseSequence())
        self.assertFalse(parametrize.is_parametrized(module))

        # A parametrization from one tensor to one tensor that changes the dtype
        class ChangeDtypeInverse(nn.Module):
            def forward(self, x):
                return x.float()

            def right_inverse(self, w):
                return w.bool()

        # For parametrizations that return one tensor, right_inverse may not change the dtype
        with self.assertRaisesRegex(ValueError, "outputs one tensor, it may not change the dtype"):
            parametrize.register_parametrization(module, "weight", ChangeDtypeInverse())
        self.assertFalse(parametrize.is_parametrized(module))

        # Doesn't return a tensor
        class NotTensor(nn.Module):
            def forward(self, x):
                return 2

        # Forward must return a tensor
        with self.assertRaisesRegex(ValueError, "must return a tensor"):
            parametrize.register_parametrization(module, "weight", NotTensor())
        self.assertFalse(parametrize.is_parametrized(module))

        # A parametrization from one tensor to one tensor that changes the dtype
        class ChangeDtype(nn.Module):
            def forward(self, x):
                return x.bool()

        # forward should not change the initial dtype
        with self.assertRaisesRegex(ValueError, "may not change the dtype"):
            parametrize.register_parametrization(module, "weight", ChangeDtype())
        self.assertFalse(parametrize.is_parametrized(module))

        # Change shape
        class ChangeShape(nn.Module):
            def forward(self, x):
                return x[:-1]

        # forward should not change the original shape
        with self.assertRaisesRegex(ValueError, "may not change the shape"):
            parametrize.register_parametrization(module, "weight", ChangeShape())
        self.assertFalse(parametrize.is_parametrized(module))

        # Many to one that changes dtype
        class ChangeDtypeMulti(nn.Module):
            def forward(self, x, y):
                return (x + y).bool()

            def right_inverse(self, w):
                return w, w + 1

        # forward should not change the original shape even for parametrizations with many inputs
        with self.assertRaisesRegex(ValueError, "may not change the dtype"):
            parametrize.register_parametrization(module, "weight", ChangeDtypeMulti())
        self.assertFalse(parametrize.is_parametrized(module))

        # Returning a sequence of size one, although weird, it's correct
        class SequenceLen1(nn.Module):
            def forward(self, x):
                return x

            def right_inverse(self, w):
                return (w,)

        parametrize.register_parametrization(module, "weight", SequenceLen1())
        self.assertTrue(hasattr(module.parametrizations.weight, "original0"))
        self.assertFalse(hasattr(module.parametrizations.weight, "original1"))
        _ = module.weight   # Does not throw
        self.assertTrue(parametrize.is_parametrized(module))
        parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)

        # None of the operations above should have altered the weight
        self.assertFalse(parametrize.is_parametrized(module))
        self.assertEqual(module.weight, weight_init)

    def test_errors_parametrized_tensor_parametrization(self):
        # Test errors when registering a parametrization on a parametrized tensor

        class Identity(nn.Module):
            def forward(self, x):
                return x

        module = nn.Linear(3, 4)
        parametrize.register_parametrization(module, "weight", Identity())

        # Has to return a tensor
        class WrongReturn(nn.Module):
            def forward(self, x):
                return x, x

        with self.assertRaisesRegex(ValueError, "must return a tensor"):
            parametrize.register_parametrization(module, "weight", WrongReturn())
        self.assertTrue(parametrize.is_parametrized(module))
        self.assertEqual(len(module.parametrizations.weight), 1)
        self.assertTrue(isinstance(module.parametrizations.weight[0], Identity))

        # Cannot change dtype
        class ChangeDtype(nn.Module):
            def forward(self, x):
                return x.bool()

        with self.assertRaisesRegex(ValueError, "may not change the dtype"):
            parametrize.register_parametrization(module, "weight", ChangeDtype())
        self.assertTrue(parametrize.is_parametrized(module))
        self.assertEqual(len(module.parametrizations.weight), 1)
        self.assertTrue(isinstance(module.parametrizations.weight[0], Identity))

        # Cannot change shape
        class ChangeShape(nn.Module):
            def forward(self, x):
                return x[:-1]

        with self.assertRaisesRegex(ValueError, "may not change the shape"):
            parametrize.register_parametrization(module, "weight", ChangeShape())
        self.assertTrue(parametrize.is_parametrized(module))
        self.assertEqual(len(module.parametrizations.weight), 1)
        self.assertTrue(isinstance(module.parametrizations.weight[0], Identity))

        # The following checks are mostly due to bugs in the code of the parametrization

        # right_inverse has to return a tensor
        class WrongReturnInverse(Identity):
            def right_inverse(self, x):
                return x, x

        with self.assertRaisesRegex(ValueError, "right_inverse must return a tensor"):
            parametrize.register_parametrization(module, "weight", WrongReturnInverse())
        self.assertTrue(parametrize.is_parametrized(module))
        self.assertEqual(len(module.parametrizations.weight), 1)
        self.assertTrue(isinstance(module.parametrizations.weight[0], Identity))

        # Cannot change dtype
        class ChangeDtypeInverse(Identity):
            def right_inverse(self, x):
                return x.bool()

        with self.assertRaisesRegex(ValueError, "must have the same dtype"):
            parametrize.register_parametrization(module, "weight", ChangeDtypeInverse())
        self.assertTrue(parametrize.is_parametrized(module))
        self.assertEqual(len(module.parametrizations.weight), 1)
        self.assertTrue(isinstance(module.parametrizations.weight[0], Identity))

        # Cannot change shape
        class ChangeShapeInverse(Identity):
            def right_inverse(self, x):
                return x[:-1]

        with self.assertRaisesRegex(ValueError, "must have the same shape"):
            parametrize.register_parametrization(module, "weight", ChangeShapeInverse())
        self.assertTrue(parametrize.is_parametrized(module))
        self.assertEqual(len(module.parametrizations.weight), 1)
        self.assertTrue(isinstance(module.parametrizations.weight[0], Identity))

    # FIXME: Rewrite this test using functions not depending on LAPACK
    #        and remove the `@skipIfNoLapack` (see #70995)
    @skipIfNoLapack
    def test_multiple_inputs_parametrization(self):
        # A parametrization with several outputs
        class RankOne(nn.Module):
            def forward(self, x, y):
                # Form a rank-1 matrix from a pair of vectors
                return x.unsqueeze(-1) @ y.unsqueeze(-2)

            def right_inverse(self, Y):
                # We project the given matrix onto the rank 1 matrices
                U, S, Vh = torch.linalg.svd(Y, full_matrices=False)
                # S is ordered in a decreasing way.
                s0_sqrt = S[0].sqrt().unsqueeze(-1)
                return U[..., :, 0] * s0_sqrt, Vh[..., 0, :] * s0_sqrt

        # Simple parametrisation
        class Double(nn.Module):
            def forward(self, x):
                return 2.0 * x

            def right_inverse(self, w):
                return 0.5 * w

        model = nn.Linear(3, 3)
        # Test one parametrization
        parametrize.register_parametrization(model, "weight", RankOne())
        self.assertTrue(hasattr(model, "parametrizations"))
        self.assertTrue(parametrize.is_parametrized(model))
        self.assertTrue(parametrize.is_parametrized(model, "weight"))
        self.assertTrue(hasattr(model.parametrizations.weight, "original0"))
        self.assertIn("original0", model.parametrizations.weight._parameters)
        self.assertTrue(hasattr(model.parametrizations.weight, "original1"))
        self.assertIn("original1", model.parametrizations.weight._parameters)
        self.assertFalse(parametrize.is_parametrized(model, "bias"))
        self.assertNotIn("weight", model._parameters)
        # Result should be rank 1
        self.assertEqual(torch.linalg.matrix_rank(model.weight).item(), 1)

        with self.assertRaisesRegex(ValueError, "leave_parametrized=False"):
            # Cannot remove a parametrization with multiple inputs and not leave it parametrized
            parametrize.remove_parametrizations(model, "weight", leave_parametrized=False)
        # Remove parametrization and check consistency
        parametrize.remove_parametrizations(model, "weight", leave_parametrized=True)
        self.assertFalse(hasattr(model, "parametrizations"))
        self.assertEqual(model.__class__, nn.Linear)
        self.assertFalse(parametrize.is_parametrized(model))
        self.assertEqual(torch.linalg.matrix_rank(model.weight).item(), 1)
        self.assertIn("weight", model._parameters)

        # Registering parametrizations with one input on top of one with multiple inputs should work
        init_weight = model.weight.clone()
        parametrize.register_parametrization(model, "weight", RankOne())
        # Projecting a rank 1 matrix onto the matrices of rank one does not change the matrix
        self.assertEqual(init_weight, model.weight)
        parametrize.register_parametrization(model, "weight", Double())
        # The matrix now is twice the initial matrix
        self.assertEqual(2.0 * init_weight, model.weight)
        # Multiplying by a scalar does not change the rank
        self.assertEqual(torch.linalg.matrix_rank(model.weight).item(), 1)

        # The model has now three parameters
        self.assertEqual(len(list(model.parameters())), 3)

        sgd = torch.optim.SGD(model.parameters(), lr=0.1)

        # Test backward. Should not throw
        for _ in range(2):
            sgd.zero_grad()
            loss = (model.weight.T @ model.bias).sum()
            loss.backward()
            sgd.step()

        # Same drill as before, removing should work as expected
        with self.assertRaisesRegex(ValueError, "leave_parametrized=False"):
            # Cannot remove a parametrization with multiple inputs and not leave it parametrized
            parametrize.remove_parametrizations(model, "weight", leave_parametrized=False)
        # Remove parametrization and check consistency
        parametrize.remove_parametrizations(model, "weight", leave_parametrized=True)
        self.assertFalse(hasattr(model, "parametrizations"))
        self.assertEqual(model.__class__, nn.Linear)
        self.assertFalse(parametrize.is_parametrized(model))
        self.assertEqual(torch.linalg.matrix_rank(model.weight).item(), 1)
        self.assertIn("weight", model._parameters)

        # The model has now two parameters
        self.assertEqual(len(list(model.parameters())), 2)

        # Test backward. Should not throw
        sgd = torch.optim.SGD(model.parameters(), lr=0.1)
        for _ in range(2):
            sgd.zero_grad()
            loss = (model.weight.T @ model.bias).sum()
            loss.backward()
            sgd.step()

    # FIXME: Rewrite this test using functions not depending on LAPACK
    #        and remove the `@skipIfNoLapack` (see #70995)
    @skipIfNoLapack
    def test_caching_parametrization(self):
        r"""Test the caching system of a parametrization"""
        # Define a couple matrix parametrizations
        class Skew(nn.Module):
            def forward(self, X):
                X = X.tril(-1)
                return X - X.T

        class Orthogonal(nn.Module):
            def forward(self, X):
                Id = torch.eye(X.size(0), device=X.device)
                return torch.linalg.solve(Id + X, Id - X)

        model = nn.Linear(5, 5)
        parametrize.register_parametrization(model, "weight", Skew())
        parametrize.register_parametrization(model, "weight", Orthogonal())

        # Test that the caching system works
        with parametrize.cached():
            X = model.weight
            Y = model.weight
            self.assertEqual(id(X), id(Y))

    # FIXME: Rewrite this test using functions not depending on LAPACK
    #        and remove the `@skipIfNoLapack` (see #70995)
    @skipIfNoLapack
    def test_caching_parametrization_with_transfer_parametrizations_and_params(self):
        r"""Test that transferring parametrizations doesn't cause issues with caching"""
        class Skew(nn.Module):
            def forward(self, X):
                X = X.tril(-1)
                return X - X.T

        class Orthogonal(nn.Module):
            def forward(self, X):
                Id = torch.eye(X.size(0), device=X.device)
                return torch.linalg.solve(Id + X, Id - X)

        model = nn.Linear(5, 5)
        parametrize.register_parametrization(model, "weight", Skew())
        parametrize.register_parametrization(model, "weight", Orthogonal())

        to_model = nn.Linear(5, 5)
        parametrize.transfer_parametrizations_and_params(model, to_model)

        with parametrize.cached():
            X = model.weight
            Y = model.weight
            self.assertEqual(id(X), id(Y))

            A = to_model.weight
            B = to_model.weight
            self.assertEqual(id(A), id(B))

            # test that the results are distinct objects for each module
            self.assertNotEqual(id(A), id(X))

    def test_parametrization_same_training_mode(self):
        r"""Test training mode updated on parametrization registration"""
        class Identity(nn.Module):
            def forward(self, X):
                return X

        module = nn.Linear(4, 4)
        module.eval()
        parametrize.register_parametrization(module, "weight", Identity())
        self.assertFalse(module.parametrizations.weight[0].training)
        module.train()
        parametrize.register_parametrization(module, "weight", Identity().eval())
        self.assertTrue(module.parametrizations.weight[0].training)
        self.assertTrue(module.parametrizations.weight[1].training)

    def test_type_before_parametrizations(self):
        r"""Test that type_before_parametrizations always retrieves original type"""

        class Identity(nn.Module):
            def forward(self, X):
                return X

        model = nn.Linear(5, 5)
        original_type = type(model)
        self.assertTrue(
            parametrize.type_before_parametrizations(model) == original_type
        )
        parametrize.register_parametrization(model, "weight", Identity())
        self.assertTrue(
            parametrize.type_before_parametrizations(model) == original_type
        )

    def test_deepcopy_after_parametrization(self):
        r"""Test that we are able to create a deepcopy of the module when it's parametrized."""

        class AddOne(nn.Module):
            def forward(self, x):
                return x + 1.0

        class ModelWithoutDeepcopy(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor([1., 1., 1., 1.]), requires_grad=True)
                self.bias = nn.Parameter(torch.tensor([0., 0., 0., 0.]), requires_grad=True)
                self.attr = [1.0, 2.0, 3.0, 4.0]

        class ActualModel(ModelWithoutDeepcopy):
            # Emulate custom implementation of the deepcopying.
            def __deepcopy__(self, memo):
                result = self.__new__(self.__class__)
                memo[id(self)] = result
                result.__dict__ = deepcopy(self.__dict__, memo)
                return result

        def check_deepcopy(m1: nn.Module, m2: nn.Module):
            w1 = m1.parametrizations.weight.original
            w2 = m2.parametrizations.weight.original
            b1 = m1.parametrizations.bias.original if parametrize.is_parametrized(m1, "bias") else m1.bias
            b2 = m2.parametrizations.bias.original if parametrize.is_parametrized(m2, "bias") else m2.bias
            # Weights, biases and attributes should be equal but they must be different objects.
            self.assertEqual(m1.__dict__.keys(), m2.__dict__.keys())
            self.assertIsNot(m1, m2)
            self.assertEqual(w1, w2)
            self.assertIsNot(w1, w2)
            self.assertEqual(b1, b2)
            self.assertIsNot(b1, b2)
            self.assertEqual(m1.attr, m2.attr)
            self.assertIsNot(m1.attr, m2.attr)

        for model in (ModelWithoutDeepcopy(), ActualModel()):
            # General check that we are able to create deepcopy.
            parametrize.register_parametrization(model, "weight", AddOne())
            check_deepcopy(model, deepcopy(model))
            # Check that this works on models with several parametrized tensors.
            parametrize.register_parametrization(model, "bias", AddOne())
            check_deepcopy(model, deepcopy(model))
            # Check that this works on models where tensors have more than one parametrization.
            parametrize.register_parametrization(model, "weight", AddOne())
            check_deepcopy(model, deepcopy(model))

    def test_transfer_parametrizations_and_params(self):
        r"""Test that all parametrizations and their associated parameters are transferred."""

        class AddOne(nn.Module):
            def forward(self, x):
                return x + 1.0

        class Double(nn.Module):
            def forward(self, x):
                return 2.0 * x

            def right_inverse(self, x):
                return 0.5 * x

        class MinusOne(nn.Module):
            def forward(self, x):
                return x - 1.0

        model = nn.Linear(5, 5)
        parametrize.register_parametrization(model, "weight", AddOne())
        parametrize.register_parametrization(model, "weight", Double())
        parametrize.register_parametrization(model, "weight", MinusOne())
        hold_weight = model.weight

        to_model = torch.ao.nn.qat.Linear(
            5, 5, qconfig=torch.ao.quantization.get_default_qconfig()
        )
        parametrize.transfer_parametrizations_and_params(model, to_model)

        # checks that final and original value are correct and the to_model is parametrized
        self.assertTrue(torch.nn.utils.parametrize.is_parametrized(to_model, "weight"))
        self.assertEqual(model.weight, to_model.weight)
        self.assertEqual(
            model.parametrizations.weight.original,
            to_model.parametrizations.weight.original,
        )

        # check that the transfer didn't affect the original value
        self.assertEqual(hold_weight, model.weight)

        # testing that changes to one set of parametrizations do not affect the other
        parametrize.remove_parametrizations(to_model, "weight")
        self.assertFalse(torch.nn.utils.parametrize.is_parametrized(to_model, "weight"))
        self.assertTrue(torch.nn.utils.parametrize.is_parametrized(model, "weight"))

        # also test that parameters that don't exist in to_model get transferred
        model.test_param = Parameter(torch.randn(5, 5))

        self.assertTrue(not hasattr(to_model, "test_param"))
        parametrize.register_parametrization(model, "test_param", Double())
        hold_test_param = model.test_param
        parametrize.transfer_parametrizations_and_params(model, to_model, "test_param")

        # check that previously missing params got transferred correctly
        self.assertEqual(model.test_param, to_model.test_param)
        self.assertEqual(
            model.parametrizations.test_param.original,
            to_model.parametrizations.test_param.original,
        )

        # check that the new transfer didn't change the value for the from_module
        self.assertEqual(hold_test_param, model.test_param)

    def test_transfer_parametrizations_and_params_right_inverse(self):
        r"""Test that all parametrizations and their associated parameters are transferred."""

        class Double(nn.Module):
            def forward(self, x):
                return 2.0 * x

            def right_inverse(self, x):
                return 0.5 * x

        model = nn.Linear(5, 5)
        parametrize.register_parametrization(model, "weight", Double())
        hold_weight = model.weight

        to_model = torch.ao.nn.qat.Linear(
            5, 5, qconfig=torch.ao.quantization.get_default_qconfig()
        )
        parametrize.transfer_parametrizations_and_params(model, to_model)

        # check that transfer occurs successfully
        self.assertEqual(model.weight, to_model.weight)
        self.assertEqual(
            model.parametrizations.weight.original,
            to_model.parametrizations.weight.original,
        )

        # check that transfer doesn't affect the from_model weight
        self.assertEqual(hold_weight, model.weight)

    def test_transfer_parametrizations_and_params_single_param(self):
        r"""Test that all parametrizations and their associated parameters are transferred."""

        class AddOne(nn.Module):
            def forward(self, x):
                return x + 1.0

        class Double(nn.Module):
            def forward(self, x):
                return 2.0 * x

        class MinusOne(nn.Module):
            def forward(self, x):
                return x - 1.0

        model = nn.Linear(5, 5, bias=True)
        parametrize.register_parametrization(model, "weight", AddOne())
        parametrize.register_parametrization(model, "weight", Double())
        parametrize.register_parametrization(model, "weight", MinusOne())
        parametrize.register_parametrization(model, "bias", AddOne())
        parametrize.register_parametrization(model, "bias", Double())
        parametrize.register_parametrization(model, "bias", MinusOne())

        to_model = torch.ao.nn.qat.Linear(
            5, 5, bias=True, qconfig=torch.ao.quantization.get_default_qconfig()
        )
        parametrize.transfer_parametrizations_and_params(model, to_model, "weight")

        # check that weight and only weight was transferred
        self.assertEqual(model.weight, to_model.weight)
        self.assertEqual(
            model.parametrizations.weight.original,
            to_model.parametrizations.weight.original,
        )
        self.assertTrue("bias" not in to_model.parametrizations)

    # FIXME: Rewrite this test using functions not depending on LAPACK
    # and remove the `@skipIfNoLapack` (see #70995)
    @skipIfNoLapack
    def test_transfer_parametrizations_and_params_many_to_one(self):
        # A parametrization with several outputs
        class RankOne(nn.Module):
            def forward(self, x, y):
                # Form a rank-1 matrix from a pair of vectors
                return x.unsqueeze(-1) @ y.unsqueeze(-2)

            def right_inverse(self, Y):
                # We project the given matrix onto the rank 1 matrices
                U, S, Vh = torch.linalg.svd(Y, full_matrices=False)
                # S is ordered in a decreasing way.
                s0_sqrt = S[0].sqrt().unsqueeze(-1)
                return U[..., :, 0] * s0_sqrt, Vh[..., 0, :] * s0_sqrt

        class Double(nn.Module):
            def forward(self, x):
                return 2.0 * x

        model = nn.Linear(3, 3)
        parametrize.register_parametrization(model, "weight", RankOne())
        parametrize.register_parametrization(model, "weight", Double())
        hold_weight = model.weight

        to_model = torch.ao.nn.qat.Linear(
            3, 3, qconfig=torch.ao.quantization.get_default_qconfig()
        )

        parametrize.transfer_parametrizations_and_params(model, to_model)

        # checks that final and original value are correct and the to_model is parametrized
        self.assertTrue(torch.nn.utils.parametrize.is_parametrized(to_model, "weight"))
        self.assertEqual(model.weight, to_model.weight)
        self.assertEqual(
            model.parametrizations.weight.original0,
            to_model.parametrizations.weight.original0,
        )
        self.assertEqual(
            model.parametrizations.weight.original1,
            to_model.parametrizations.weight.original1,
        )

        # check that the transfer didn't affect the original value
        self.assertEqual(hold_weight, model.weight)

        # testing that changes to one set of parametrizations do not affect the other
        model.test_param = Parameter(torch.randn(3, 3))

        self.assertTrue(not hasattr(to_model, "test_param"))
        parametrize.register_parametrization(model, "test_param", RankOne())
        hold_test_param = model.test_param
        parametrize.transfer_parametrizations_and_params(model, to_model, "test_param")

        # also check that previously missing params got transferred correctly
        self.assertEqual(model.test_param, to_model.test_param)
        self.assertEqual(
            model.parametrizations.test_param.original0,
            to_model.parametrizations.test_param.original0,
        )
        self.assertEqual(
            model.parametrizations.test_param.original1,
            to_model.parametrizations.test_param.original1,
        )

        # check that the new transfer didn't change the value for the from_module
        self.assertEqual(hold_test_param, model.test_param)

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
                with self.subTest(m=m, name=name):
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
                with self.subTest(m=m, name=name):

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
                with self.subTest(m=m, name=name):
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
                with self.subTest(m=m, name=name):
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
        expected_mask = torch.tensor([[0, 0, 1, 0], [1, 1, 0, 1]], dtype=torch.float32)
        computed_mask = container.compute_mask(t, default_mask)
        self.assertEqual(expected_mask, computed_mask)

        # 2) test structured pruning
        q = prune.LnStructured(amount=1, n=2, dim=0)
        q._tensor_name = 'test'
        container.add_pruning_method(q)
        # since we are pruning the lowest magnitude one of the two rows, the
        # outcome of the calculation should be this:
        expected_mask = torch.tensor([[0, 0, 0, 0], [1, 1, 0, 1]], dtype=torch.float32)
        computed_mask = container.compute_mask(t, default_mask)
        self.assertEqual(expected_mask, computed_mask)

        # 2) test structured pruning, along another axis
        r = prune.LnStructured(amount=1, n=2, dim=1)
        r._tensor_name = 'test'
        container.add_pruning_method(r)
        # since we are pruning the lowest magnitude of the four columns, the
        # outcome of the calculation should be this:
        expected_mask = torch.tensor([[0, 1, 1, 0], [0, 1, 0, 1]], dtype=torch.float32)
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
        expected_weight = torch.tensor([[0, 2, 3, 4], [-4, -3, -2, 0]],
                                       dtype=m.weight.dtype)
        self.assertEqual(expected_weight, m.weight)

        # check that pruning again removes the next two smallest entries
        prune.l1_unstructured(m, 'weight', amount=2)
        expected_weight = torch.tensor([[0, 0, 3, 4], [-4, -3, 0, 0]],
                                       dtype=m.weight.dtype)
        self.assertEqual(expected_weight, m.weight)

    def test_l1_unstructured_pruning_with_importance_scores(self):
        r"""Test that l1 unstructured pruning actually removes the lowest
        entries of importance scores and not the parameter by l1 norm (by hand).
        It also checks that applying l1 unstructured pruning more than once
        respects the previous mask.
        """
        m = nn.Linear(4, 2)
        # modify its weight matrix by hand
        m.weight = torch.nn.Parameter(
            torch.tensor(
                [[1, 2, 3, 4], [-4, -3, -2, -1]], dtype=torch.float32
            )
        )
        importance_scores = torch.tensor(
            [[4, 2, 1, 3], [-3, -1, -2, -4]], dtype=torch.float32
        )

        prune.l1_unstructured(m, 'weight', amount=2, importance_scores=importance_scores)
        expected_weight = torch.tensor([[1, 2, 0, 4], [-4, 0, -2, -1]],
                                       dtype=m.weight.dtype)
        self.assertEqual(expected_weight, m.weight)

        # check that pruning again removes two entries of m.weight that are colocated with
        # the next two smallest absolute values of importance scores.
        prune.l1_unstructured(m, 'weight', amount=2, importance_scores=importance_scores)
        expected_weight = torch.tensor([[1, 0, 0, 4], [-4, 0, 0, -1]],
                                       dtype=m.weight.dtype)
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
        m.weight.data = torch.tensor(
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

    def test_ln_structured_pruning_importance_scores(self):
        r"""Check Ln structured pruning by hand.
        """
        m = nn.Conv2d(3, 1, 2)
        m.weight.data = torch.tensor(
            [[[[1., 2.], [1., 2.5]],
             [[0.5, 1.], [0.1, 0.1]],
             [[-3., -5.], [0.1, -1.]]]]
        )
        importance_scores = torch.tensor(
            [[[[10., 1.], [10., 1.]],
             [[30., 3.], [30., 3.]],
             [[-20., -2.], [-20., -2.]]]]
        )
        # expected effect of pruning 1 of the 3 channels by L2-norm
        expected_mask_axis1 = torch.ones_like(m.weight)
        expected_mask_axis1[:, 0] = 0.

        prune.ln_structured(m, 'weight', amount=1, n=2, dim=1, importance_scores=importance_scores)
        self.assertEqual(expected_mask_axis1, m.weight_mask)

        # expected effect of pruning 1 of the 2 columns along axis -1 by L1-norm
        expected_mask_axis3 = expected_mask_axis1
        expected_mask_axis3[:, :, :, 1] = 0.

        prune.ln_structured(m, 'weight', amount=1, n=1, dim=-1, importance_scores=importance_scores)
        self.assertEqual(expected_mask_axis3, m.weight_mask)

    def test_remove_pruning(self):
        r"""`prune.remove` removes the hook and the reparametrization
        and makes the pruning final in the original parameter.
        """
        modules = [nn.Linear(5, 7), nn.Conv3d(2, 2, 2)]
        names = ['weight', 'bias']

        for m in modules:
            for name in names:
                with self.subTest(m=m, name=name):
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
                with self.subTest(m=m, name=name):
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

        expected_mweight = torch.tensor([[0, 2, 3, 4], [-4, -3, -2, 0]],
                                        dtype=m.weight.dtype)
        self.assertEqual(expected_mweight, m.weight)

        expected_nweight = torch.tensor([[0, 0, -2]]).to(dtype=n.weight.dtype)
        self.assertEqual(expected_nweight, n.weight)

    def test_global_pruning_importance_scores(self):
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
        m_importance_scores = torch.tensor(
            [[4, 2, 1, 3], [-3, -1, -2, -4]], dtype=torch.float32
        )
        n.weight = torch.nn.Parameter(
            torch.tensor([[0, 0.1, -2]]).to(
                dtype=torch.float32)
        )
        n_importance_scores = torch.tensor([[0, 10., -0.2]]).to(dtype=torch.float32)

        params_to_prune = (
            (m, 'weight'),
            (n, 'weight'),
        )
        importance_scores = {
            (m, 'weight'): m_importance_scores,
            (n, 'weight'): n_importance_scores,
        }

        # prune the 4 smallest weights globally by L1 magnitude
        prune.global_unstructured(
            params_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=4,
            importance_scores=importance_scores,
        )

        expected_m_weight = torch.tensor([[1, 2, 0, 4], [-4, 0, -2, -1]],
                                         dtype=m.weight.dtype)
        self.assertEqual(expected_m_weight, m.weight)

        expected_n_weight = torch.tensor([[0, 0.1, 0]]).to(dtype=n.weight.dtype)
        self.assertEqual(expected_n_weight, n.weight)

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
        expected_mask = torch.tensor([[0, 0, 0, 0], [0, 0, 1, 1]], dtype=computed_mask.dtype)

        self.assertEqual(computed_mask, expected_mask)

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
                with self.subTest(m=m, name=name):

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

    def test_prune_importance_scores(self):
        # create a new pruning method
        p = prune.L1Unstructured(amount=2)
        # create tensor to be pruned
        t = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]).to(dtype=torch.float32)
        importance_scores = torch.tensor(
            [[1, 2, 3, 4], [1.5, 1.6, 1.7, 1.8]]
        ).to(dtype=torch.float32)
        # create prior mask by hand
        default_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 1]])
        # since we are pruning the two lowest magnitude units, the outcome of
        # the calculation should be this:
        expected_mask = torch.tensor([[0, 1, 1, 0], [0, 1, 0, 1]])
        pruned_tensor = p.prune(t, default_mask, importance_scores=importance_scores)
        self.assertEqual(t * expected_mask, pruned_tensor)

    def test_prune_importance_scores_mimic_default(self):
        # create a new pruning method
        p = prune.L1Unstructured(amount=2)
        # create tensor to be pruned
        t = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]).to(dtype=torch.float32)
        # create prior mask by hand
        default_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 1]])
        # since we are pruning the two lowest magnitude units, the outcome of
        # the calculation should be this:
        expected_mask = torch.tensor([[0, 0, 1, 0], [1, 1, 0, 1]])
        pruned_tensor_without_importance_scores = p.prune(t, default_mask)
        pruned_tensor_with_importance_scores = p.prune(t, default_mask, importance_scores=t)
        self.assertEqual(pruned_tensor_without_importance_scores, pruned_tensor_with_importance_scores)
        self.assertEqual(t * expected_mask, pruned_tensor_without_importance_scores)

    def test_rnn_pruning(self):
        l = torch.nn.LSTM(32, 32)
        # This Module has 4 parameters called:
        # 'weight_ih_l0', 'weight_hh_l0', 'bias_ih_l0', 'bias_hh_l0'

        # Pruning one of them causes one of the weights to become a tensor
        prune.l1_unstructured(l, 'weight_ih_l0', 0.5)
        assert (
            sum([isinstance(p, torch.nn.Parameter) for p in l._flat_weights])
            == 3
        )

        # Removing the pruning reparametrization restores the Parameter
        prune.remove(l, 'weight_ih_l0')
        assert (
            sum([isinstance(p, torch.nn.Parameter) for p in l._flat_weights])
            == 4
        )

        # Make sure that, upon removal of the reparametrization, the
        # `._parameters` and `.named_parameters` contain the right params.
        # Specifically, the original weight ('weight_ih_l0') should be placed
        # back in the parameters, while the reparametrization component
        # ('weight_ih_l0_orig') should be removed.
        assert 'weight_ih_l0' in l._parameters
        assert l._parameters['weight_ih_l0'] is not None
        assert 'weight_ih_l0_orig' not in l._parameters
        assert 'weight_ih_l0' in dict(l.named_parameters())
        assert dict(l.named_parameters())['weight_ih_l0'] is not None
        assert 'weight_ih_l0_orig' not in dict(l.named_parameters())

    def test_rnn_weight_norm(self):
        def check_weight_norm(l, name, num_params):
            # This Module has 4 or 5 parameters called:
            # 'weight_ih_l0', 'weight_hh_l0', 'bias_ih_l0', 'bias_hh_l0', weight_hr_l0

            # Applying weight norm on one of them causes it to become a tensor
            l = torch.nn.utils.weight_norm(l, name=name)
            self.assertEqual(
                sum([isinstance(p, torch.nn.Parameter) for p in l._flat_weights]),
                num_params - 1,
            )

            # Removing the weight norm reparametrization restores the Parameter
            l = torch.nn.utils.remove_weight_norm(l, name=name)
            self.assertEqual(
                sum([isinstance(p, torch.nn.Parameter) for p in l._flat_weights]),
                num_params,
            )

            # Make sure that, upon removal of the reparametrization, the
            # `._parameters` and `.named_parameters` contain the right params.
            # Specifically, the original weight ('weight_ih_l0') should be placed
            # back in the parameters, while the reparametrization components
            # ('weight_ih_l0_v' and 'weight_ih_l0_g') should be removed.
            self.assertTrue(name in l._parameters)
            self.assertIsNotNone(l._parameters[name])
            self.assertTrue(name + '_v' not in l._parameters)
            self.assertTrue(name + '_g' not in l._parameters)
            self.assertTrue(name in dict(l.named_parameters()))
            self.assertIsNotNone(dict(l.named_parameters())[name])
            self.assertTrue(name + '_v' not in dict(l.named_parameters()))
            self.assertTrue(name + '_g' not in dict(l.named_parameters()))

        check_weight_norm(torch.nn.LSTM(32, 32), 'weight_ih_l0', 4)
        check_weight_norm(torch.nn.LSTM(32, 32, proj_size=16), 'weight_hr_l0', 5)


    def test_weight_norm(self):
        for dtype in [torch.float, torch.bfloat16]:
            input = torch.randn(3, 4, dtype=dtype)
            m = nn.Linear(4, 5).to(dtype=dtype)
            expected_output = m(input)

            # add weight normalization
            m = torch.nn.utils.weight_norm(m)
            self.assertEqual(m.weight_v.size(), m.weight.size())
            self.assertEqual(m.weight_g.size(), (5, 1))
            self.assertEqual(m(input), expected_output, atol=dtype2prec_DONTUSE[dtype], rtol=0)

            # remove weight norm
            m = torch.nn.utils.remove_weight_norm(m)
            self.assertFalse(hasattr(m, 'weight_g'))
            self.assertFalse(hasattr(m, 'weight_v'))
            self.assertEqual(m(input), expected_output, atol=dtype2prec_DONTUSE[dtype], rtol=0)

            # test with dim=1
            m = torch.nn.utils.weight_norm(m, dim=1)
            self.assertEqual(m.weight_v.size(), m.weight.size())
            self.assertEqual(m.weight_g.size(), (1, 4))
            self.assertEqual(m(input), expected_output, atol=dtype2prec_DONTUSE[dtype], rtol=0)

            # test with dim=None
            m = nn.Linear(4, 5).to(dtype=dtype)
            expected_output = m(input)
            m = torch.nn.utils.weight_norm(m, dim=None)
            self.assertEqual(m(input), expected_output)

            with self.assertRaisesRegex(RuntimeError, 'register two weight_norm hooks'):
                m = torch.nn.utils.weight_norm(m)
                m = torch.nn.utils.weight_norm(m)

        # For float16, the forward of the Module doesn't work but we must still be able
        # to register the weight norm as this is often done before sending the Module to
        # CUDA.
        m = nn.Linear(4, 5, dtype=torch.float16)
        m = torch.nn.utils.weight_norm(m)

    def test_parameterlistdict_setting_attributes(self):
        with warnings.catch_warnings(record=True) as w:
            mod = nn.ParameterList(map(nn.Parameter, [torch.rand(2), torch.rand(2)]))
        self.assertTrue(len(w) == 0)

        with warnings.catch_warnings(record=True) as w:
            mod.train()
            mod.eval()
        self.assertTrue(len(w) == 0)

        with warnings.catch_warnings(record=True) as w:
            mod = nn.ParameterDict({"a": nn.Parameter(torch.rand(2)), "b": nn.Parameter(torch.rand(2))})
        self.assertTrue(len(w) == 0)

        with warnings.catch_warnings(record=True) as w:
            mod.train()
            mod.eval()
        self.assertTrue(len(w) == 0)

    def test_parameterlistdict_pickle(self):
        m = nn.ParameterList(map(nn.Parameter, [torch.rand(2), torch.rand(2)]))
        with warnings.catch_warnings(record=True) as w:
            m = pickle.loads(pickle.dumps(m))
        self.assertTrue(len(w) == 0)

        # Test whether loading from older checkpoints works without triggering warnings
        m = nn.ParameterList(map(nn.Parameter, [torch.rand(2), torch.rand(2)]))
        del m._forward_pre_hooks, m._state_dict_hooks, m._load_state_dict_pre_hooks, m._non_persistent_buffers_set
        with warnings.catch_warnings(record=True) as w:
            m = pickle.loads(pickle.dumps(m))
        self.assertTrue(len(w) == 0)

        m = nn.ParameterDict({"a": nn.Parameter(torch.rand(2)), "b": nn.Parameter(torch.rand(2))})
        with warnings.catch_warnings(record=True) as w:
            m = pickle.loads(pickle.dumps(m))
        self.assertTrue(len(w) == 0)

        # Test whether loading from older checkpoints works without triggering warnings
        m = nn.ParameterDict({"a": nn.Parameter(torch.rand(2)), "b": nn.Parameter(torch.rand(2))})
        del m._forward_pre_hooks, m._state_dict_hooks, m._load_state_dict_pre_hooks, m._non_persistent_buffers_set
        with warnings.catch_warnings(record=True) as w:
            m = pickle.loads(pickle.dumps(m))
        self.assertTrue(len(w) == 0)

    def test_weight_norm_pickle(self):
        m = torch.nn.utils.weight_norm(nn.Linear(5, 7))
        m = pickle.loads(pickle.dumps(m))
        self.assertIsInstance(m, nn.Linear)

    @skipIfTorchDynamo("TorchDynamo fails here for unknown reasons")
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

                gradcheck(fn, (input.clone().requires_grad_(),), check_batched_grad=False)

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

                gradcheck(fn, (input.clone().requires_grad_(),))

                # assert that backprop reaches weight_orig in eval
                if requires_grad:
                    def fn(weight):
                        return wrapped_m(input)

                    gradcheck(fn, (m.weight_orig,))

    def test_new_spectral_norm(self):
        input = torch.randn(3, 5)
        m = nn.Linear(5, 7)
        m = torch.nn.utils.parametrizations.spectral_norm(m)
        spectral_norm_m = m.parametrizations.weight[0]

        self.assertEqual(spectral_norm_m._u.size(), torch.Size([m.weight.size(0)]))

        # .parametrizations.weight.original should be trainable
        self.assertTrue(hasattr(m.parametrizations.weight, 'original'))
        self.assertTrue('original' in m.parametrizations.weight._parameters)

        # u should be just a reused buffer
        self.assertTrue(hasattr(spectral_norm_m, '_u'))
        self.assertTrue('_u' in spectral_norm_m._buffers)
        self.assertTrue('_v' in spectral_norm_m._buffers)

        # weight should be a plain attribute, not counted as a buffer or a param
        self.assertIsNotNone(m.weight)
        self.assertFalse('weight' in m._buffers)
        self.assertFalse('weight' in m._parameters)

        # it should also be sharing storage as `weight_orig`
        # self.assertEqual(m.parametrizations.weight.original.storage(), m.weight.storage())
        self.assertEqual(m.parametrizations.weight.original.size(), m.weight.size())
        self.assertEqual(m.parametrizations.weight.original.stride(), m.weight.stride())

        m = torch.nn.utils.parametrize.remove_parametrizations(m, 'weight')

        # spectral_norm is the only parametrization
        self.assertFalse(hasattr(m, 'parametrizations'))
        self.assertTrue('weight' in m._parameters)

        # We can register spectral_norm multiple times on the same parameter
        # and on multiple parameters in the same module
        m = torch.nn.utils.parametrizations.spectral_norm(m, 'weight')
        m = torch.nn.utils.parametrizations.spectral_norm(m, 'weight')
        m = torch.nn.utils.parametrizations.spectral_norm(m, 'bias')

        # If we remove the parametrization on bias, weight is still parametrized
        # Removing a parametrization runs forward in eval mode if leave_parametrized=True
        m = torch.nn.utils.parametrize.remove_parametrizations(m, 'bias')
        self.assertTrue('bias' in m._parameters)
        self.assertTrue(hasattr(m, 'parametrizations'))
        self.assertFalse('weight' in m._parameters)

        m = torch.nn.utils.parametrize.remove_parametrizations(m, 'weight')
        # Neither weight and bias are parametrized
        self.assertFalse(hasattr(m, 'parametrizations'))
        self.assertTrue('weight' in m._parameters)
        self.assertFalse(torch.nn.utils.parametrize.is_parametrized(m))

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
                def get_modules():
                    m = nn.Linear(3, 4).to(device)
                    m.weight.requires_grad_(requires_grad)
                    m = torch.nn.utils.parametrizations.spectral_norm(m)
                    wrapped_m = maybe_wrap(m)
                    spectral_norm_m = m.parametrizations.weight[0]
                    return m, wrapped_m, spectral_norm_m

                input = torch.randn(2, 3, device=device)

                m, wrapped_m, spectral_norm_m = get_modules()

                self.assertTrue(hasattr(spectral_norm_m, '_u'))
                u0 = spectral_norm_m._u.clone()
                v0 = spectral_norm_m._v.clone()

                # TEST TRAINING BEHAVIOR

                # We perform GD first to modify the initial matrix
                opt = torch.optim.SGD(wrapped_m.parameters(), lr=0.1)

                opt.zero_grad()
                wrapped_m(input).sum().backward()
                opt.step()

                out = wrapped_m(input)
                if requires_grad:
                    # run forward again and assert that u and v are updated
                    self.assertNotEqual(u0, spectral_norm_m._u)
                    self.assertNotEqual(v0, spectral_norm_m._v)

                # assert that backprop reaches original weight
                # can't use gradcheck because the function changes as we
                # activate through it in training mode
                if requires_grad:
                    torch.autograd.grad(out.sum(), m.parametrizations.weight.original)

                # test backward works with multiple forwards
                # it uses training mode so we need to reset `u` and `v` vectors
                # to same value at beginning for finite difference test to pass
                saved_u = spectral_norm_m._u.clone()
                saved_v = spectral_norm_m._v.clone()

                def fn(input):
                    spectral_norm_m._u.data.copy_(saved_u)
                    spectral_norm_m._v.data.copy_(saved_v)
                    out0 = wrapped_m(input)
                    out1 = wrapped_m(input)
                    return out0 + out1

                # Make sure we can compute gradients wrt to all the parameters in the case
                # of double forward
                fn(input.clone().requires_grad_()).sum().backward()
                gradcheck(fn, (input.clone().requires_grad_(),), check_batched_grad=False)

                # test removing
                # spectral norm module needs to be in eval mode if we'd like to
                # avoid doing another power iteration
                m, wrapped_m, _ = get_modules()
                pre_remove_out = wrapped_m(input)
                m.eval()
                m = torch.nn.utils.parametrize.remove_parametrizations(m, 'weight')
                self.assertEqual(wrapped_m(input), pre_remove_out)

                torch.nn.utils.parametrizations.spectral_norm(m)
                for _ in range(3):
                    pre_remove_out = wrapped_m(input)
                m.eval()
                m = torch.nn.utils.parametrize.remove_parametrizations(m, 'weight')
                self.assertEqual(wrapped_m(input), pre_remove_out)

                # TEST EVAL BEHAVIOR
                m, wrapped_m, spectral_norm_m = get_modules()
                wrapped_m(input)
                last_train_out = wrapped_m(input)
                last_train_u = spectral_norm_m._u.clone()
                last_train_v = spectral_norm_m._v.clone()
                wrapped_m.zero_grad()
                wrapped_m.eval()

                eval_out0 = wrapped_m(input)
                # assert eval gives same result as last training iteration
                self.assertEqual(eval_out0, last_train_out)
                # assert doing more iteartion in eval don't change things
                self.assertEqual(eval_out0, wrapped_m(input))
                self.assertEqual(last_train_u, spectral_norm_m._u)
                self.assertEqual(last_train_v, spectral_norm_m._v)

                # FIXME: the code below is flaky when executed with DataParallel
                # see https://github.com/pytorch/pytorch/issues/13818
                if apply_dp:
                    continue

                # test backward works with multiple forwards in mixed training
                # and eval modes
                # it uses training mode so we need to reset `u` and `v` vectors
                # to same value at beginning for finite difference test to pass
                saved_u = spectral_norm_m._u.clone()
                saved_v = spectral_norm_m._v.clone()

                def fn(input):
                    spectral_norm_m._u.data.copy_(saved_u)
                    spectral_norm_m._v.data.copy_(saved_v)
                    wrapped_m.train()
                    out0 = wrapped_m(input)
                    wrapped_m.eval()
                    out1 = wrapped_m(input)
                    wrapped_m.train()
                    out2 = wrapped_m(input)
                    wrapped_m.eval()
                    out3 = wrapped_m(input)
                    return out0 + out1 + out2 + out3

                gradcheck(fn, (input.clone().requires_grad_(),))

                # assert that backprop reaches weight_orig in eval
                if requires_grad:
                    def fn(weight):
                        return wrapped_m(input)

                    gradcheck(fn, (m.parametrizations.weight.original,))

    def test_new_spectral_norm_load_state_dict(self):
        for activate_times in (0, 3):
            inp = torch.randn(2, 3)
            m = nn.Linear(3, 5)
            snm = torch.nn.utils.parametrizations.spectral_norm(m)
            snm.train()

            for _ in range(activate_times):
                snm(inp)

            state_dict = deepcopy(snm.state_dict())
            self.assertEqual({
                'parametrizations.weight.original',
                'bias',
                'parametrizations.weight.0._v',
                'parametrizations.weight.0._u'
            }, set(state_dict.keys()))

            # test that non-strict loading works
            non_strict_state_dict = deepcopy(state_dict)
            non_strict_state_dict['nonsense'] = 'nonsense'
            with self.assertRaisesRegex(RuntimeError, r'Unexpected key\(s\) in state_dict: "nonsense"'):
                snm.load_state_dict(non_strict_state_dict, strict=True)
            snm.load_state_dict(non_strict_state_dict, strict=False)
            del non_strict_state_dict['parametrizations.weight.original']
            snm.load_state_dict(non_strict_state_dict, strict=False)
            del non_strict_state_dict['parametrizations.weight.0._u']
            snm.load_state_dict(non_strict_state_dict, strict=False)
            del non_strict_state_dict['parametrizations.weight.0._v']
            snm.load_state_dict(non_strict_state_dict, strict=False)
            non_strict_state_dict['weight'] = snm.weight.detach().clone()     # set W as a buffer
            snm.load_state_dict(non_strict_state_dict, strict=False)
            del non_strict_state_dict._metadata['parametrizations.weight.0']  # remove metadata info
            snm.load_state_dict(non_strict_state_dict, strict=False)
            del non_strict_state_dict['weight']                               # remove W buffer
            snm.load_state_dict(non_strict_state_dict, strict=False)
            del non_strict_state_dict['bias']
            snm.load_state_dict(non_strict_state_dict, strict=False)

            # normal state_dict

            # test that re-wrapping does not matter
            m = torch.nn.utils.parametrize.remove_parametrizations(snm, 'weight')
            snm = torch.nn.utils.parametrizations.spectral_norm(m)

            snm.load_state_dict(state_dict)
            with torch.no_grad():
                snm.eval()
                out0_eval = snm(inp)
                snm.train()
                out1_train = snm(inp)
                out2_train = snm(inp)
                snm.eval()
                out3_eval = snm(inp)

            # test that re-wrapping does not matter
            m = torch.nn.utils.parametrize.remove_parametrizations(snm, 'weight')
            snm = torch.nn.utils.parametrizations.spectral_norm(m)

            # Test normal loading
            snm.load_state_dict(state_dict)
            with torch.no_grad():
                snm.eval()
                self.assertEqual(out0_eval, snm(inp))
                snm.train()
                self.assertEqual(out1_train, snm(inp))
                self.assertEqual(out2_train, snm(inp))
                snm.eval()
                self.assertEqual(out3_eval, snm(inp))

    @skipIfNoLapack
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

    def test_new_spectral_norm_dim(self):
        inp = torch.randn(2, 3, 10, 12)
        m = nn.ConvTranspose2d(3, 4, (5, 6))
        m = torch.nn.utils.parametrizations.spectral_norm(m)
        snm = m.parametrizations.weight[0]
        # this should not run into incompatible shapes
        x = m(inp)
        # check that u refers to the same dimension
        self.assertEqual(snm._u.shape, m.parametrizations.weight.original[0, :, 0, 0].shape)

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
        self.assertEqual(expect_out, out_hat)

    def test_new_spectral_norm_forward(self):
        input = torch.randn(3, 5)
        m = nn.Linear(5, 7)
        m = torch.nn.utils.parametrizations.spectral_norm(m)
        snm = m.parametrizations.weight[0]
        # naive forward
        _weight = m.parametrizations.weight.original
        _bias, _v = m.bias, snm._v
        _weight_mat = _weight.view(_weight.size(0), -1)
        _u = torch.mv(_weight_mat, _v)
        _u = F.normalize(_u, dim=0, eps=1e-12)
        _v = torch.mv(_weight_mat.t(), _u)
        _v = F.normalize(_v, dim=0, eps=1e-12)
        _weight.data /= torch.dot(_u, torch.matmul(_weight_mat, _v))
        out_hat = torch.nn.functional.linear(input, _weight, _bias)
        expect_out = m(input)
        self.assertEqual(expect_out, out_hat)

    def test_spectral_norm_pickle(self):
        m = torch.nn.utils.spectral_norm(nn.Linear(5, 7))
        m = pickle.loads(pickle.dumps(m))
        self.assertIsInstance(m, nn.Linear)

    @skipIfNoLapack
    def test_orthogonal_parametrization(self):
        # Orthogonal implements 6 algorithms (3x parametrizations times 2 options of use_trivialization)

        def assert_is_orthogonal(X):
            n, k = X.size(-2), X.size(-1)
            if n < k:
                X = X.mT
                n, k = k, n
            Id = torch.eye(k, dtype=X.dtype, device=X.device).expand(*(X.size()[:-2]), k, k)
            eps = 10 * n * torch.finfo(X.dtype).eps
            torch.testing.assert_close(X.mH @ X, Id, atol=eps, rtol=0.)


        def assert_weight_allclose_Q(weight, W):
            # Test that weight is equal to the Q part of the QR decomposition of W
            # (or of its transpose if the matrix is wide)
            wide_matrix = W.size(-2) < W.size(-1)
            if wide_matrix:
                W = W.mT
            Q, R = torch.linalg.qr(W)
            Q *= R.diagonal(dim1=-2, dim2=-1).sgn().unsqueeze(-2)
            if wide_matrix:
                Q = Q.mT
            torch.testing.assert_close(Q, weight, atol=1e-5, rtol=0.)


        for shape, dtype, use_linear in product(((4, 4), (5, 3), (3, 5)),  # square/ tall / wide
                                                (torch.float32, torch.complex64),
                                                (True, False)):
            # Conv2d does not support complex yet
            if not use_linear:
                continue

            if use_linear:
                input = torch.randn(3, shape[0], dtype=dtype)
            else:
                input = torch.randn(2, 2, shape[0] + 2, shape[1] + 1, dtype=dtype)

            for parametrization, use_trivialization in product(("matrix_exp", "cayley", "householder"),
                                                               (False, True)):
                # right_inverse for Cayley and matrix_exp not implemented for use_trivialization=False
                # See Note [right_inverse expm cayley]
                can_initialize = use_trivialization or parametrization == "householder"

                # We generate them every time to always start with fresh weights
                if use_linear:
                    m = nn.Linear(*shape, dtype=dtype)
                else:
                    m = nn.Conv2d(2, 3, shape, dtype=dtype)

                # We do not support householder for complex inputs
                # See Note [Householder complex]
                w_init = m.weight.clone()
                if parametrization == "householder" and m.weight.is_complex():
                    msg = "householder parametrization does not support complex tensors"
                    with self.assertRaisesRegex(ValueError, msg):
                        torch.nn.utils.parametrizations.orthogonal(m,
                                                                   "weight",
                                                                   parametrization,
                                                                   use_trivialization=use_trivialization)
                    continue

                wide_matrix = w_init.size(-2) < w_init.size(-1)
                torch.nn.utils.parametrizations.orthogonal(m,
                                                           "weight",
                                                           parametrization,
                                                           use_trivialization=use_trivialization)
                # Forwards works as expected
                self.assertEqual(w_init.shape, m.weight.shape)
                assert_is_orthogonal(m.weight)
                if can_initialize:
                    assert_weight_allclose_Q(m.weight, w_init)

                # Intializing with a given orthogonal matrix works
                X = torch.randn_like(m.weight)
                if wide_matrix:
                    X = X.mT
                w_new = torch.linalg.qr(X).Q
                if wide_matrix:
                    w_new = w_new.mT
                if can_initialize:
                    m.weight = w_new
                    torch.testing.assert_close(w_new, m.weight, atol=1e-5, rtol=0.)
                else:
                    msg = "assign to the matrix exponential or the Cayley parametrization"
                    with self.assertRaisesRegex(NotImplementedError, msg):
                        m.weight = w_new

                # Intializing with a non-orthogonal matrix makes m.weight be the Q part of the given matrix
                w_new = torch.randn_like(m.weight)
                if can_initialize:
                    m.weight = w_new
                    assert_weight_allclose_Q(m.weight, w_new)
                else:
                    msg = "assign to the matrix exponential or the Cayley parametrization"
                    with self.assertRaisesRegex(NotImplementedError, msg):
                        m.weight = w_new

                opt = torch.optim.SGD(m.parameters(), lr=0.1)
                for _ in range(2):
                    opt.zero_grad()
                    m(input).norm().backward()
                    grad = m.parametrizations.weight.original.grad
                    self.assertIsNotNone(grad)
                    # We do not update the upper triangular part of the matrix if tall tril if wide
                    if grad.size(-2) >= grad.size(-1):
                        zeros_grad = grad.triu(1)
                    else:
                        zeros_grad = grad.tril(-1)
                    self.assertEqual(zeros_grad, torch.zeros_like(zeros_grad))
                    # The gradient in the diagonal can only be imaginary because a skew-Hermitian
                    # matrix has imaginary diagonal
                    diag_grad = grad.diagonal(dim1=-2, dim2=-1)
                    if grad.is_complex():
                        diag_grad = diag_grad.real
                    self.assertEqual(diag_grad, torch.zeros_like(diag_grad))
                    opt.step()
                    assert_is_orthogonal(m.weight)

    @skipIfNoLapack
    def test_orthogonal_errors(self):
        m = nn.Linear(3, 4)
        with self.assertRaisesRegex(ValueError, "has to be one of"):
            torch.nn.utils.parametrizations.orthogonal(m, "weight", "foo")

        with self.assertRaisesRegex(ValueError, "Expected a matrix"):
            torch.nn.utils.parametrizations.orthogonal(m, "bias")

        torch.nn.utils.parametrizations.orthogonal(m, "weight")
        with self.assertRaisesRegex(ValueError, "matrices of shape"):
            m.weight = torch.randn(5, 5)
        torch.nn.utils.parametrize.remove_parametrizations(m, "weight")


    def test_threshold_int(self):
        x = torch.tensor([-3, -2, -1, 0, 1, 2, 3])
        expected = torch.tensor([99, 99, 99, 99, 1, 2, 3])
        self.assertEqual(F.threshold(x, 0, 99), expected)

    def test_threshold_bfloat16(self):
        x = torch.randn(100)
        for threshold in [0, -0.5, 0.5, float('inf'), float('-inf'), float('nan')]:
            expected = F.threshold(x, threshold, 0).bfloat16().float()
            res_bf16 = F.threshold(x.bfloat16(), threshold, 0).float()
            self.assertEqual(res_bf16, expected)

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
        torch.testing.assert_close(torch.from_numpy(expected_output), actual_output.cpu(), atol=1e-3, rtol=1e-3)

    def test_pad_scalar_error(self):
        inputs = torch.tensor(0., requires_grad=True)
        self.assertRaises(RuntimeError, lambda: F.pad(inputs, (1, 1)))
        self.assertRaises(RuntimeError, lambda: F.pad(inputs, (1,)))

    def test_nested_tensor_from_mask(self):
        N, L, D = 10, 12, 14

        input = torch.rand(N, L, D)
        mask = torch.ones(N, L, dtype=torch.bool)
        # Leave first row be all True to maintain the nt's size unchanged
        for i in range(1, N):
            end = torch.randint(1, L, size=()).item()
            mask[i, end:] = False

        nt = torch._nested_tensor_from_mask(input, mask)
        input_convert = nt.to_padded_tensor(0.)
        input.masked_fill_(mask.reshape(N, L, 1).logical_not(), 0.)

        self.assertEqual(input, input_convert)

    def test_nested_tensor_from_mask_error(self):
        N, L, D = 10, 12, 14

        input = torch.rand(N, L, D)
        # Mask is not bool
        mask = torch.zeros(N, L, dtype=torch.float)
        self.assertRaises(RuntimeError, lambda: torch._nested_tensor_from_mask(input, mask))

        # Mask size is not 2
        mask = torch.zeros(N, L, D, dtype=torch.bool)
        self.assertRaises(RuntimeError, lambda: torch._nested_tensor_from_mask(input, mask))

        # Input size is not 3
        mask = torch.zeros(N, L, dtype=torch.bool)
        input = torch.rand(N, L)
        self.assertRaises(RuntimeError, lambda: torch._nested_tensor_from_mask(input, mask))

        # Mask size does not match input
        mask = torch.zeros(N + 1, L + 1, dtype=torch.bool)
        input = torch.rand(N, L, D)
        self.assertRaises(RuntimeError, lambda: torch._nested_tensor_from_mask(input, mask))

        # Mask is not padding format
        mask = torch.ones(N, L, dtype=torch.bool)
        mask[0, 0] = False
        mask[0, 2] = False
        self.assertRaises(RuntimeError, lambda: torch._nested_tensor_from_mask(input, mask))

    @unittest.skipIf(not TEST_NUMPY, "numpy not found")
    @parametrize_test("average_attn_weights", [True, False])
    def test_multihead_attention(self, average_attn_weights):
        def _scaled_dot_attn_ref(Q, K, V, dims, unseen_mask=None, key_padding_mask=None,
                                 average_attn_weights=average_attn_weights):
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
            if average_attn_weights:
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
            src_indices = torch.arange(0, max_srclen).unsqueeze(0).to(src_lengths)
            src_indices = src_indices.expand(batch_size, max_srclen)
            src_lengths = src_lengths.unsqueeze(dim=1).expand(batch_size, max_srclen)
            # returns [batch_size, max_seq_len]
            return (src_indices < src_lengths).int().detach()

        def _multihead_attn_test_helper(add_key_padding_mask=False, add_bias_kv=False, add_zero_attn=False,
                                        saved_kv=False, same_embed_dim=False,
                                        average_attn_weights=average_attn_weights):
            for _ in range(100):
                batch_sz, seq_len = [random.randint(2, 10) for r in range(2)]
                d_head = random.randint(3, 10)
                nheads = random.randint(2, 5) * 2
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
                        static_k=saved_k_tensor, static_v=saved_v_tensor,
                        average_attn_weights=average_attn_weights)
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
                        static_k=saved_k_tensor, static_v=saved_v_tensor,
                        average_attn_weights=average_attn_weights)

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

    def test_multihead_attn_3d_attn_mask(self):
        embed_dim = 8
        num_heads = 4
        batch_size = 8
        src_len = 3
        tgt_len = 2

        query = torch.rand(batch_size, tgt_len, embed_dim)  # [N, T, D]
        key = torch.rand(batch_size, src_len, embed_dim)  # [N, S, D]
        value = key  # [N, S, D]
        attn_mask = torch.randint(0, 2, (batch_size, tgt_len, src_len)).float()  # [N, T, S]
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float('-inf')).masked_fill(attn_mask == 1, float(0.0))

        mta_model = torch.nn.MultiheadAttention(embed_dim, num_heads)

        # Generate 3D results
        attn_mask_3d = torch.repeat_interleave(attn_mask, num_heads, dim=0)  # [N * H, T, S]
        output_3d = mta_model(query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1), attn_mask=attn_mask_3d)[0]
        output_3d = output_3d.transpose(0, 1)  # [N, T, D]

        for i in range(0, batch_size):
            output_2d = mta_model(query[i].unsqueeze(0).transpose(0, 1),
                                  key[i].unsqueeze(0).transpose(0, 1),
                                  value[i].unsqueeze(0).transpose(0, 1),
                                  attn_mask=attn_mask[i])[0]

            # output_2d in shape of [T, 1, D]
            self.assertEqual(output_3d[i].unsqueeze(0).transpose(0, 1), output_2d)

    def test_multihead_attn_no_bias(self):
        embed_dim = 8
        num_heads = 4
        mha = torch.nn.MultiheadAttention(embed_dim, num_heads, bias=False)

        # Verify that bias=False applies to both in and out projection layers.
        self.assertIsNone(mha.in_proj_bias)
        self.assertIsNone(mha.out_proj.bias)

    def _test_multihead_attn_invalid_shape_impl(self, mha):
        # Batched (3D) query cases
        query = torch.randn(4, 4, 4)
        key = torch.randn(4, 4, 4)
        value = torch.randn(4, 4, 4)

        msg = "expected `key` and `value` to be 3-D but found 2-D and 3-D tensors respectively"
        # 3D query, 2D key and 3D value
        with self.assertRaisesRegex(AssertionError, msg):
            mha(query, torch.randn(4, 4), value)

        msg = "expected `key` and `value` to be 3-D but found 3-D and 2-D tensors respectively"
        # 3D query, 3D key and 2D value
        with self.assertRaisesRegex(AssertionError, msg):
            mha(query, key, torch.randn(4, 4))

        msg = "expected `key_padding_mask` to be `None` or 2-D but found 1-D tensor instead"
        # 3D query, 3D key, 3D value and 1D key_padding_mask
        with self.assertRaisesRegex(AssertionError, msg):
            mha(query, key, value, key_padding_mask=torch.tensor([False, False, True, True], dtype=torch.bool))

        msg = "expected `attn_mask` to be `None`, 2-D or 3-D but found 1-D tensor instead"
        # 3D query, 3D key, 3D value and 1D attn_mask
        with self.assertRaisesRegex(AssertionError, msg):
            mha(query, key, value, attn_mask=torch.tensor([False, False, True, True], dtype=torch.bool))

        # Unbatched (2D) query cases
        query = torch.randn(4, 4)
        key = torch.randn(4, 4)
        value = torch.randn(4, 4)

        msg = "expected `key` and `value` to be 2-D but found 3-D and 2-D tensors respectively"
        # 2D query, 3D key and 2D value
        with self.assertRaisesRegex(AssertionError, msg):
            mha(query, torch.randn(4, 4, 4), value)

        msg = "expected `key` and `value` to be 2-D but found 2-D and 3-D tensors respectively"
        # 2D query, 3D key and 2D value
        with self.assertRaisesRegex(AssertionError, msg):
            mha(query, key, torch.randn(4, 4, 4))

        msg = "expected `key_padding_mask` to be `None` or 1-D but found 2-D tensor instead"
        # 2D query, 2D key, 2D value and 1D key_padding_mask
        with self.assertRaisesRegex(AssertionError, msg):
            mha(query, key, value, key_padding_mask=torch.tensor([[False, False, True, True] * 2], dtype=torch.bool))

        msg = "expected `attn_mask` to be `None`, 2-D or 3-D but found 1-D tensor instead"
        # 2D query, 2D key, 2D value and 1D attn_mask
        with self.assertRaisesRegex(AssertionError, msg):
            mha(query, key, value, attn_mask=torch.tensor([False, False, True, True], dtype=torch.bool))

        msg = r"Expected `attn_mask` shape to be \(4, 4, 4\)"
        # 2D query, 2D key, 2D value and 3D incorrect attn_mask
        with self.assertRaisesRegex(AssertionError, msg):
            mha(query, key, value, attn_mask=torch.randn(5, 4, 4).bernoulli_().to(torch.bool))

    def test_multihead_attn_invalid_shape(self):
        mha = torch.nn.MultiheadAttention(4, 4)
        self._test_multihead_attn_invalid_shape_impl(mha)
        # Give the test a chance to hit the fast path. (Right now, it
        # won't, but gating may be less restricted in the future.)
        with torch.no_grad():
            self._test_multihead_attn_invalid_shape_impl(mha.eval())

    @torch.no_grad()
    def test_multihead_attn_fast_path_invalid_shape(self):
        mha = torch.nn.MultiheadAttention(4, 4, batch_first=True).eval()

        # Batched (3D) query cases
        query = torch.randn(4, 4, 4)
        key = torch.randn(4, 4, 4)
        value = torch.randn(4, 4, 4)

        # Currently, this case will just go to the slow path and get
        # the usual message because it fails the requirement to be
        # batched.
        msg = "expected `key` and `value` to be 3-D but found 2-D and 3-D tensors respectively"
        # 3D query, 2D key and 3D value
        with self.assertRaisesRegex(AssertionError, msg):
            mha(query, torch.randn(3, 3), value, need_weights=False)

        # Currently, this case will just go to the slow path and get
        # the usual message because it fails the requirement to be
        # batched.
        msg = "expected `key` and `value` to be 3-D but found 3-D and 2-D tensors respectively"
        # 3D query, 3D key and 2D value
        with self.assertRaisesRegex(AssertionError, msg):
            mha(query, key, torch.randn(3, 3), need_weights=False)

        msg = "expected `key_padding_mask` to be `None` or 2-D but found 1-D tensor instead"
        # 3D query, 3D key, 3D value and 1D key_padding_mask
        with self.assertRaisesRegex(AssertionError, msg):
            mha(query, key, value, key_padding_mask=torch.tensor([False, True, True], dtype=torch.bool), need_weights=False)

        msg = "expected `attn_mask` to be `None`, 2-D or 3-D but found 1-D tensor instead"
        # 3D query, 3D key, 3D value and 1D attn_mask
        with self.assertRaisesRegex(AssertionError, msg):
            mha(query, key, value, attn_mask=torch.tensor([False, True, True], dtype=torch.bool), need_weights=False)

        # Unbatched (2D) query cases
        # NOTE: error messages are the same as regular path because the fast path doesn't support 2D.
        query = torch.randn(4, 4)
        key = torch.randn(4, 4)
        value = torch.randn(4, 4)

        msg = "expected `key` and `value` to be 2-D but found 3-D and 2-D tensors respectively"
        # 2D query, 3D key and 2D value
        with self.assertRaisesRegex(AssertionError, msg):
            mha(query, torch.randn(4, 4, 4), value)

        msg = "expected `key` and `value` to be 2-D but found 2-D and 3-D tensors respectively"
        # 2D query, 3D key and 2D value
        with self.assertRaisesRegex(AssertionError, msg):
            mha(query, key, torch.randn(4, 4, 4))

        msg = "expected `key_padding_mask` to be `None` or 1-D but found 2-D tensor instead"
        # 2D query, 2D key, 2D value and 1D key_padding_mask
        with self.assertRaisesRegex(AssertionError, msg):
            mha(query, key, value, key_padding_mask=torch.tensor([[False, False, True, True] * 2], dtype=torch.bool))

        msg = "expected `attn_mask` to be `None`, 2-D or 3-D but found 1-D tensor instead"
        # 2D query, 2D key, 2D value and 1D attn_mask
        with self.assertRaisesRegex(AssertionError, msg):
            mha(query, key, value, attn_mask=torch.tensor([False, False, True, True], dtype=torch.bool))

        msg = r"Expected `attn_mask` shape to be \(4, 4, 4\)"
        # 2D query, 2D key, 2D value and 3D incorrect attn_mask
        with self.assertRaisesRegex(AssertionError, msg):
            mha(query, key, value, attn_mask=torch.randn(5, 4, 4).bernoulli_().to(torch.bool))

    def test_multihead_attn_nested_tensor_outside_fast_path(self):
        mha = torch.nn.MultiheadAttention(4, 4, batch_first=True).eval()
        nt = torch.nested.nested_tensor([torch.randn(4, 4)])
        # One tested platform (linux-bionic-py3.7-clang) has a torch_function for one
        # or more of these. Take advantage of that to test the torch_function bailout.
        has_torch_func = torch.overrides.has_torch_function(
            (nt, mha.in_proj_weight, mha.in_proj_bias, mha.out_proj.weight, mha.out_proj.bias))
        if has_torch_func:
            msg = "MultiheadAttention does not support NestedTensor.*argument has_torch_function"
        else:
            msg = ("MultiheadAttention does not support NestedTensor outside of its fast path.*grad is " +
                   "enabled and.*or biases requires_grad")
        with self.assertRaisesRegex(AssertionError, msg):
            mha(nt, nt, nt)

        if has_torch_func:
            # Just give up, they're all going to fail with the same message.
            return

        with torch.no_grad():
            mha(nt, nt, nt)
        with torch.inference_mode():
            mha(nt, nt, nt)
        nt = torch.nested.nested_tensor([torch.randn(4, 4, requires_grad=False)])
        nt.requires_grad = False
        with self.assertRaisesRegex(AssertionError, msg):
            mha(nt, nt, nt)
        mha.in_proj_weight.requires_grad = False
        mha.in_proj_bias.requires_grad = False
        mha.out_proj.weight.requires_grad = False
        mha.out_proj.bias.requires_grad = False
        mha(nt, nt, nt)

    def test_normalize(self):
        inputs = torch.randn(1, 3, 4, 4, requires_grad=True)
        self.assertTrue(gradcheck(lambda x: F.normalize(x, p=1, dim=-1), (inputs,)))
        self.assertTrue(gradcheck(lambda x: F.normalize(x, p=2, dim=-2), (inputs,)))

        inputs = torch.randn((), requires_grad=True)
        self.assertTrue(gradcheck(lambda x: F.normalize(x, p=1, dim=-1), (inputs,)))

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    # Skip the test for ROCm as per https://github.com/pytorch/pytorch/issues/53190
    @skipIfRocm
    def test_broadcast_double_backwards_gpu(self):
        tensors = (torch.randn(4, 4, device='cuda', requires_grad=True),
                   torch.randn(4, 4, device='cuda', requires_grad=True),
                   torch.randn(4, 4, device='cuda', requires_grad=True))
        # TODO(#50743): the following segfaults with check_batched_grad=True
        _assertGradAndGradgradChecks(self, lambda *i: Broadcast.apply((0, 1), *i), tensors,
                                     check_batched_grad=False)

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
        self.assertFalse(any(k.startswith('empty') for k in state_dict.keys()))
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

        # Reference https://github.com/pytorch/pytorch/pull/75507#issuecomment-1110291545
        self.assertNotWarn(lambda: l.state_dict(destination=dict()), "Should not warn kwarg destination w/o _metadata")

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
        conv1_bias_dtype = block.conv1.bias.dtype

        state_dict = net.state_dict()
        state_dict.update({
            'linear1.weight': torch.ones(5, 5),
            'block.conv1.bias': torch.arange(1, 4, dtype=conv1_bias_dtype),
            'bn.running_mean': torch.randn(2),
        })
        # Also test if a DDP state_dict can be loaded from a local model.
        ddp_state_dict = net.state_dict()
        ddp_state_dict.update({
            'module.linear1.weight': torch.ones(5, 5),
            'module.block.conv1.bias': torch.arange(1, 4, dtype=conv1_bias_dtype),
            'module.bn.running_mean': torch.randn(2),
        })
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(ddp_state_dict, 'module.')
        for sd in [state_dict, ddp_state_dict]:
            incompatible_keys = net.load_state_dict(sd)
            self.assertEqual(len(incompatible_keys.missing_keys), 0)
            self.assertEqual(len(incompatible_keys.unexpected_keys), 0)
            self.assertNotIn('Incompatible', str(incompatible_keys))
            self.assertEqual(net.linear1.weight, sd['linear1.weight'])
            self.assertEqual(net.block.conv1.bias, sd['block.conv1.bias'])
            self.assertEqual(net.bn.running_mean, sd['bn.running_mean'])

        state_dict = net.state_dict()
        state_dict.update({'extra': torch.ones(5)})
        self.assertRaises(RuntimeError, lambda: net.load_state_dict(state_dict))
        incompatible_keys = net.load_state_dict(state_dict, strict=False)
        self.assertEqual(len(incompatible_keys.missing_keys), 0)
        self.assertEqual(len(incompatible_keys.unexpected_keys), 1)
        self.assertIn('extra', incompatible_keys.unexpected_keys)
        self.assertIn('Incompatible', str(incompatible_keys))

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
            'block.conv1.bias': torch.arange(1, 4, dtype=conv1_bias_dtype),
            'bn.running_mean': torch.randn(2),
            'nonexistent_key': torch.rand(3)
        }
        net.load_state_dict(state_dict, strict=False)
        self.assertEqual(net.linear1.weight, state_dict['linear1.weight'])
        self.assertEqual(net.block.conv1.bias, state_dict['block.conv1.bias'])
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

    def test_load_state_dict_child(self):
        base_module = nn.Linear(1, 1)
        model = base_module
        for _ in range(3):
            model = nn.Sequential(*[deepcopy(model) for _ in range(10)])

        def hook_fn(module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            module_state_dict = module.state_dict()
            self.assertEqual(len(module_state_dict.keys()), len(state_dict.keys()))

        model[0][0]._register_load_state_dict_pre_hook(hook_fn, with_module=True)
        model.load_state_dict(model.state_dict(), strict=True)

    @skipIfTorchDynamo("TorchDynamo fails here for unknown reasons")
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
        with torch.no_grad():
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

    def test_extra_state(self):

        class SubModule(torch.nn.Module):
            def __init__(self, foo):
                super().__init__()
                self.foo = foo

            def get_extra_state(self):
                return {
                    'foo': self.foo
                }

            def set_extra_state(self, state):
                self.foo = state['foo']

        class MyModule(torch.nn.Module):
            def __init__(self, foo, bar):
                super().__init__()
                self.sub = SubModule(foo)
                self.bar = bar

            def get_extra_state(self):
                return {
                    'bar': self.bar
                }

            def set_extra_state(self, state):
                self.bar = state['bar']

        # Ensure state_dict contains the extra state by loading it into another module.
        m = MyModule(3, 'something')
        m2 = MyModule(5, 'something else')
        m2.load_state_dict(m.state_dict())
        self.assertEqual(m.state_dict(), m2.state_dict())
        self.assertEqual(m2.bar, m.bar)
        self.assertEqual(m2.sub.foo, m.sub.foo)

    def test_extra_state_non_dict(self):

        class MyModule(torch.nn.Module):
            def __init__(self, foo):
                super().__init__()
                self.foo = foo

            def get_extra_state(self):
                return self.foo

            def set_extra_state(self, state):
                self.foo = state

        # Test various types of extra state.
        for state in ('something', 5, MyModule(3)):
            m = MyModule(state)
            m2 = MyModule('something else')
            m2.load_state_dict(m.state_dict())
            self.assertEqual(m.state_dict(), m2.state_dict())
            self.assertEqual(m.foo, m2.foo)

    def test_extra_state_missing_set_extra_state(self):

        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def get_extra_state(self):
                return {
                    'foo': 5
                }

        m = MyModule()
        with self.assertRaisesRegex(RuntimeError, 'Unexpected key'):
            m.load_state_dict(m.state_dict())

    def test_extra_state_missing_get_extra_state(self):

        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def set_extra_state(self):
                pass

        m = MyModule()
        with self.assertRaisesRegex(RuntimeError, 'Missing key'):
            m.load_state_dict(m.state_dict())

    @skipIfTorchDynamo("TorchDynamo fails here for unknown reasons")
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

    def test_RNN_cell_forward_input_size(self):
        input = torch.randn(3, 11)
        hx = torch.randn(3, 20)
        for module in (nn.RNNCell, nn.GRUCell):
            cell = module(10, 20)
            self.assertRaises(Exception, lambda: cell(input, hx))

    def test_RNN_cell_forward_hidden_size(self):
        input = torch.randn(3, 10)
        hx = torch.randn(3, 21)
        cell_shared_param = (10, 20)
        for cell in (nn.RNNCell(*cell_shared_param, nonlinearity="relu"),
                     nn.RNNCell(*cell_shared_param, nonlinearity="tanh"),
                     nn.GRUCell(*cell_shared_param)):
            self.assertRaises(Exception, lambda: cell(input, hx))

    def test_RNN_cell_forward_zero_hidden_size(self):
        input = torch.randn(3, 10)
        hx = torch.randn(3, 0)
        cell_shared_param = (10, 0)
        for cell in (nn.RNNCell(*cell_shared_param, nonlinearity="relu"),
                     nn.RNNCell(*cell_shared_param, nonlinearity="tanh"),
                     nn.GRUCell(*cell_shared_param)):
            self.assertEqual(cell(input, hx).shape, torch.Size([3, 0]))

    def _test_loss_equal_input_target_shape(self, cast):
        # Tests losses whose inputs should have the same size.
        losses = {
            'mse_loss': lambda x, y: F.mse_loss(x, y),
            'l1_loss': lambda x, y: F.l1_loss(x, y),
            'smooth_l1_loss': lambda x, y: F.smooth_l1_loss(x, y),
            'huber_loss': lambda x, y: F.huber_loss(x, y),
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

    def test_gaussian_nll_loss_reduction_modes(self):
        input = torch.tensor([[0.5, 1.5, 2.5], [2., 4., 6.]])
        target = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
        var = torch.tensor([[0.5, 1., 1.5], [1., 1.5, 2.]])
        component_wise_loss = 0.5 * (torch.log(var) + (input - target)**2 / var)
        self.assertEqual(component_wise_loss,
                         F.gaussian_nll_loss(input, target, var, reduction='none'))
        self.assertEqual(torch.sum(component_wise_loss),
                         F.gaussian_nll_loss(input, target, var, reduction='sum'))
        self.assertEqual(torch.mean(component_wise_loss),
                         F.gaussian_nll_loss(input, target, var, reduction='mean'))
        with self.assertRaisesRegex(ValueError, 'is not valid'):
            F.gaussian_nll_loss(input, target, var, reduction='total')

    def test_gaussian_nll_loss_broadcasting(self):
        input = torch.tensor([[0.5, 1.5, 2.5], [2., 4., 6.]])
        target_full = torch.tensor([[1., 2., 3.], [1., 2., 3.]])
        target_part = torch.tensor([[1., 2., 3.]])
        var_full = torch.tensor([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]])
        var_part1 = torch.tensor([[0.5], [1.5]])
        var_part2 = torch.tensor([0.5, 1.5])
        component_wise_loss = 0.5 * (torch.log(var_full) + (input - target_full)**2 / var_full)
        self.assertEqual(component_wise_loss,
                         F.gaussian_nll_loss(input, target_part, var_full, reduction='none'))
        self.assertEqual(component_wise_loss,
                         F.gaussian_nll_loss(input, target_full, var_part1, reduction='none'))
        self.assertEqual(component_wise_loss,
                         F.gaussian_nll_loss(input, target_full, var_part2, reduction='none'))
        self.assertEqual(component_wise_loss,
                         F.gaussian_nll_loss(input, target_part, var_part1, reduction='none'))
        self.assertEqual(component_wise_loss,
                         F.gaussian_nll_loss(input, target_part, var_part2, reduction='none'))

    def test_gaussian_nll_loss_args(self):
        input = torch.randn(3, 5)
        with self.assertRaisesRegex(ValueError, 'var is of incorrect size'):
            target = torch.randn(3, 5)
            var = torch.ones(3, 3)
            torch.nn.functional.gaussian_nll_loss(input, target, var)
        with self.assertRaisesRegex(ValueError, 'var has negative entry/entries'):
            var = -1 * torch.ones(3, 5)
            torch.nn.functional.gaussian_nll_loss(input, target, var)

    def test_KLDivLoss_batch_mean(self):
        input_shape = (2, 5)
        log_prob1 = F.log_softmax(torch.randn(input_shape), 1)
        prob2 = F.softmax(torch.randn(input_shape), 1)

        loss = nn.KLDivLoss(reduction='batchmean')
        l = loss(log_prob1, prob2)

        loss_none_reduce = nn.KLDivLoss(reduction='sum')(log_prob1, prob2)
        expected = loss_none_reduce / input_shape[0]

        self.assertEqual(l, expected)

    def test_KLDivLoss_batch_mean_log_target(self):
        input_shape = (2, 5)
        log_prob1 = F.log_softmax(torch.randn(input_shape), 1)
        log_prob2 = F.log_softmax(torch.randn(input_shape), 1)

        loss = nn.KLDivLoss(reduction='batchmean', log_target=True)
        l = loss(log_prob1, log_prob2)

        loss_none_reduce = nn.KLDivLoss(reduction='sum', log_target=True)(log_prob1, log_prob2)
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
        self.assertEqual(res_cpu, res_gpu, atol=1e-4, rtol=0)
        self.assertEqual(grad_cpu, grad_gpu, atol=1e-4, rtol=0)

    @unittest.skipIf(not TEST_CUDA, 'CUDA not available')
    def test_CTCLoss_critical_target_len(self):
        # cudnn has an unexpected problem with target length 256, see issue #53505
        N = 1
        S = 256
        C = 10
        T = 500
        target = torch.randint(low=1, high=C, size=(S,), dtype=torch.int)
        input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.int)
        target_lengths = torch.tensor(S, dtype=torch.int)
        inp = torch.randn(T, N, C, dtype=torch.float, device='cuda').log_softmax(2).requires_grad_()
        with cudnn.flags(enabled=True):
            res_gpu = torch.nn.functional.ctc_loss(inp, target, input_lengths, target_lengths, reduction='none')
        res_cpu = torch.nn.functional.ctc_loss(inp.cpu(), target, input_lengths, target_lengths, reduction='none')
        self.assertEqual(res_cpu, res_gpu, atol=1e-3, rtol=0)

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

        self.assertEqual(res2, res, atol=1e-4, rtol=0)
        self.assertEqual(res_cpu, res.cpu(), atol=1e-4, rtol=0)
        g1, = torch.autograd.grad(res, log_probs)
        g2, = torch.autograd.grad(res2, log_probs)
        g3, = torch.autograd.grad(res_cpu, log_probs)
        self.assertEqual(g2, g3, atol=1e-4, rtol=0)
        self.assertEqual(g1, g2, atol=1e-4, rtol=0)
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

    def test_LSTM_cell_forward_input_size(self):
        input = torch.randn(3, 11)
        hx = torch.randn(3, 20)
        cx = torch.randn(3, 20)
        lstm = nn.LSTMCell(10, 20)
        self.assertRaises(Exception, lambda: lstm(input, (hx, cx)))

    def test_LSTM_cell_forward_hidden_size(self):
        input = torch.randn(3, 10)
        hx = torch.randn(3, 21)
        cx = torch.randn(3, 20)
        lstm = nn.LSTMCell(10, 20)
        self.assertRaises(Exception, lambda: lstm(input, (hx, cx)))
        self.assertRaises(Exception, lambda: lstm(input, (cx, hx)))


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
        for batch_first, src_size, tgt_size in zip((True, False),
                                                   [(bsz, seq_length, d_model),
                                                    (seq_length, bsz, d_model)],
                                                   [(bsz, tgt_length, d_model),
                                                    (tgt_length, bsz, d_model)]):
            transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers,
                                         dim_feedforward, dropout, batch_first=batch_first)
            src = torch.randn(src_size)
            src_mask = transformer.generate_square_subsequent_mask(seq_length).double()
            tgt = torch.randn(tgt_size)
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

    def test_transformerdecoderlayer(self):
        # this is a deterministic test for TransformerDecoderLayer
        d_model = 4
        nhead = 2
        dim_feedforward = 16
        dropout = 0.0
        bsz = 2
        seq_length = 5
        tgt_length = 3

        for batch_first in (False, True):
            def perm_fn(x):
                return x.transpose(1, 0) if batch_first else x

            model = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout,
                                               batch_first=batch_first)

            # set constant weights of the model
            for idx, p in enumerate(model.parameters()):
                x = p.data
                sz = x.view(-1).size(0)
                shape = x.shape
                x = torch.cos(torch.arange(0, sz).float().view(shape))
                p.data.copy_(x)

            # deterministic input
            decoder_input = torch.tensor([[[20., 30., 40., 50.]]])
            memory_input = torch.tensor([[[60., 70., 80., 90.]]])
            result = model(decoder_input, memory_input)
            ref_output = torch.tensor([[[2.314351, 0.094805, -0.671322, 0.101977]]])
            result = result.detach().numpy()
            ref_output = ref_output.detach().numpy()
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            np.testing.assert_allclose(result, ref_output, atol=1e-5)

            # deterministic input
            decoder_input = perm_fn(torch.tensor([[[9., 10., 11., 12.]],
                                                  [[11., 12., 13., 14.]]]))
            memory_input = torch.tensor([[[1., 2., 3., 4.]]])
            result = model(decoder_input, memory_input)
            result = result.detach().numpy()
            ref_output = perm_fn(torch.tensor([[[2.422245, 0.051716, -0.606338, -0.024756]],
                                               [[2.422245, 0.051716, -0.606338, -0.024756]]]))
            ref_output = ref_output.detach().numpy()
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            np.testing.assert_allclose(result, ref_output, atol=1e-5)

            # deterministic input
            decoder_input = perm_fn(torch.tensor([[[1., 2., 3., 4.]],
                                                  [[5., 6., 7., 8.]]]))
            memory_input = perm_fn(torch.tensor([[[9., 10., 11., 12.]],
                                                 [[11., 12., 13., 14.]]]))
            result = model(decoder_input, memory_input)
            ref_output = perm_fn(torch.tensor([[[2.343536, 0.085561, -0.654954, 0.074991]],
                                               [[2.343536, 0.085561, -0.654954, 0.074991]]]))
            result = result.detach().numpy()
            ref_output = ref_output.detach().numpy()
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            np.testing.assert_allclose(result, ref_output, atol=1e-5)

            # deterministic input
            decoder_input = perm_fn(torch.tensor([[[0.4517, 0.6793, 0.5313, 0.0034],
                                                   [0.2678, 0.3677, 0.4459, 0.7166]],
                                                  [[0.8100, 0.3716, 0.4096, 0.1976],
                                                   [0.6958, 0.8844, 0.6081, 0.8315]],
                                                  [[0.0494, 0.9343, 0.5955, 0.3830],
                                                   [0.5404, 0.3464, 0.9378, 0.6200]]]))
            memory_input = perm_fn(torch.tensor([[[0.7462, 0.6653, 0.5679, 0.4891],
                                                  [0.5387, 0.1655, 0.3565, 0.0471]],
                                                 [[0.8335, 0.2799, 0.5031, 0.2947],
                                                  [0.1402, 0.0318, 0.7636, 0.1346]],
                                                 [[0.6333, 0.9344, 0.1376, 0.9938],
                                                  [0.8924, 0.2872, 0.6692, 0.2944]],
                                                 [[0.9897, 0.6915, 0.3154, 0.1733],
                                                  [0.8645, 0.3513, 0.3064, 0.0767]],
                                                 [[0.8117, 0.2366, 0.4838, 0.7881],
                                                  [0.3718, 0.4945, 0.9511, 0.0864]]]))
            result = model(decoder_input, memory_input)
            ref_output = perm_fn(torch.tensor([[[2.430065, 0.027862, -0.601136, -0.073096],
                                                [2.431935, 0.028907, -0.599809, -0.072488]],
                                               [[2.428457, 0.027053, -0.602275, -0.073462],
                                                [2.431970, 0.029387, -0.599789, -0.071621]],
                                               [[2.431934, 0.028196, -0.599802, -0.073809],
                                                [2.432306, 0.028858, -0.599542, -0.072846]]]))
            result = result.detach().numpy()
            ref_output = ref_output.detach().numpy()
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            np.testing.assert_allclose(result, ref_output, atol=1e-5)

            # key_padding_mask
            key_padding_mask = torch.zeros(2, 3) == 1
            result = model(decoder_input, memory_input, tgt_key_padding_mask=key_padding_mask)
            ref_output = perm_fn(torch.tensor([[[2.430065, 0.027862, -0.601136, -0.073096],
                                                [2.431935, 0.028907, -0.599809, -0.072488]],
                                               [[2.428457, 0.027053, -0.602275, -0.073462],
                                                [2.431970, 0.029387, -0.599789, -0.071621]],
                                               [[2.431934, 0.028196, -0.599802, -0.073809],
                                                [2.432306, 0.028858, -0.599542, -0.072846]]]))
            result = result.detach().numpy()
            ref_output = ref_output.detach().numpy()
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            np.testing.assert_allclose(result, ref_output, atol=1e-5)

            # key_padding_mask
            key_padding_mask[0, 2] = 1
            key_padding_mask[1, 1] = 1
            key_padding_mask[1, 2] = 1
            result = model(decoder_input, memory_input, tgt_key_padding_mask=key_padding_mask)
            ref_output = perm_fn(torch.tensor([[[2.430025, 0.027643, -0.601164, -0.073476],
                                                [2.4323, 0.029375, -0.599553, -0.071881]],
                                               [[2.428523, 0.026838, -0.602226, -0.07391],
                                                [2.432634, 0.029842, -0.599318, -0.071253]],
                                               [[2.432278, 0.028152, -0.599555, -0.074139],
                                                [2.432659, 0.029244, -0.599294, -0.072382]]]))
            result = result.detach().numpy()
            ref_output = ref_output.detach().numpy()
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            np.testing.assert_allclose(result, ref_output, atol=1e-5)

            # memory_key_padding_mask
            key_padding_mask = torch.zeros(2, 5) == 1
            result = model(decoder_input, memory_input, memory_key_padding_mask=key_padding_mask)
            ref_output = perm_fn(torch.tensor([[[2.430065, 0.027862, -0.601136, -0.073096],
                                                [2.431935, 0.028907, -0.599809, -0.072488]],
                                               [[2.428457, 0.027053, -0.602275, -0.073462],
                                                [2.431970, 0.029387, -0.599789, -0.071621]],
                                               [[2.431934, 0.028196, -0.599802, -0.073809],
                                                [2.432306, 0.028858, -0.599542, -0.072846]]]))
            result = result.detach().numpy()
            ref_output = ref_output.detach().numpy()
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            np.testing.assert_allclose(result, ref_output, atol=1e-5)

            # memory_key_padding_mask
            key_padding_mask[0, 4] = 1
            key_padding_mask[1, 3] = 1
            key_padding_mask[1, 4] = 1
            result = model(decoder_input, memory_input, memory_key_padding_mask=key_padding_mask)
            ref_output = perm_fn(torch.tensor([[[2.429757, 0.027358, -0.601351, -0.073816],
                                                [2.432692, 0.028583, -0.599263, -0.073634]],
                                               [[2.428247, 0.02662, -0.602419, -0.074123],
                                                [2.432657, 0.029055, -0.599293, -0.072732]],
                                               [[2.431515, 0.027687, -0.600096, -0.074459],
                                                [2.433075, 0.028543, -0.598987, -0.073985]]]))
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

        for activation, batch_first in product(('gelu', F.gelu, nn.GELU()), (True, False)):
            def perm_fn(x):
                return x.transpose(1, 0) if batch_first else x

            model = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout,
                                               activation, batch_first=batch_first)

            # set constant weights of the model
            for idx, p in enumerate(model.parameters()):
                x = p.data
                sz = x.view(-1).size(0)
                shape = x.shape
                x = torch.cos(torch.arange(0, sz).float().view(shape))
                p.data.copy_(x)

            # deterministic input
            decoder_input = torch.tensor([[[20., 30., 40., 50.]]])
            memory_input = torch.tensor([[[60., 70., 80., 90.]]])
            result = model(decoder_input, memory_input)
            ref_output = torch.tensor([[[2.306435, 0.095946, -0.675796, 0.10687]]])
            torch.testing.assert_close(result, ref_output, rtol=1e-5, atol=0)

            # deterministic input
            decoder_input = perm_fn(torch.tensor([[[9., 10., 11., 12.]],
                                                  [[11., 12., 13., 14.]]]))
            memory_input = perm_fn(torch.tensor([[[1., 2., 3., 4.]]]))
            result = model(decoder_input, memory_input)
            ref_output = perm_fn(torch.tensor([[[2.415448, 0.054389, -0.610932, -0.0156613]],
                                               [[2.415448, 0.054389, -0.610932, -0.0156613]]]))
            torch.testing.assert_close(result, ref_output, rtol=1e-5, atol=0)

            # deterministic input
            decoder_input = perm_fn(torch.tensor([[[1., 2., 3., 4.]],
                                                  [[5., 6., 7., 8.]]]))
            memory_input = perm_fn(torch.tensor([[[9., 10., 11., 12.]],
                                                 [[11., 12., 13., 14.]]]))
            result = model(decoder_input, memory_input)
            ref_output = perm_fn(torch.tensor([[[2.338531, 0.087709, -0.65776, 0.080646]],
                                               [[2.338531, 0.087709, -0.65776, 0.080646]]]))
            torch.testing.assert_close(result, ref_output, rtol=1e-5, atol=0)

            # deterministic input
            decoder_input = perm_fn(torch.tensor([[[0.4517, 0.6793, 0.5313, 0.0034],
                                                   [0.2678, 0.3677, 0.4459, 0.7166]],
                                                  [[0.8100, 0.3716, 0.4096, 0.1976],
                                                   [0.6958, 0.8844, 0.6081, 0.8315]],
                                                  [[0.0494, 0.9343, 0.5955, 0.3830],
                                                   [0.5404, 0.3464, 0.9378, 0.6200]]]))
            memory_input = perm_fn(torch.tensor([[[0.7462, 0.6653, 0.5679, 0.4891],
                                                  [0.5387, 0.1655, 0.3565, 0.0471]],
                                                 [[0.8335, 0.2799, 0.5031, 0.2947],
                                                  [0.1402, 0.0318, 0.7636, 0.1346]],
                                                 [[0.6333, 0.9344, 0.1376, 0.9938],
                                                  [0.8924, 0.2872, 0.6692, 0.2944]],
                                                 [[0.9897, 0.6915, 0.3154, 0.1733],
                                                  [0.8645, 0.3513, 0.3064, 0.0767]],
                                                 [[0.8117, 0.2366, 0.4838, 0.7881],
                                                  [0.3718, 0.4945, 0.9511, 0.0864]]]))
            result = model(decoder_input, memory_input)
            ref_output = perm_fn(torch.tensor([[[2.42049104, 0.03443088, -0.60793706, -0.05436271],
                                                [2.42210631, 0.03546578, -0.60679895, -0.05357488]],
                                               [[2.41907674, 0.0336104, -0.60892977, -0.05490462],
                                                [2.42216881, 0.03586554, -0.6067524, -0.05289126]],
                                               [[2.42205716, 0.03488046, -0.60683681, -0.05460596],
                                                [2.42240309, 0.0354595, -0.60659063, -0.05378816]]]))
            torch.testing.assert_close(result, ref_output, rtol=1e-5, atol=0)

    def test_transformerdecoder(self):
        def get_a_test_layer(use_cuda, activation, batch_first=False):
            d_model = 4
            nhead = 2
            dim_feedforward = 16
            dropout = 0.0
            device = torch.device("cuda" if use_cuda else "cpu")

            layer = nn.TransformerDecoderLayer(
                d_model,
                nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                batch_first=batch_first).to(device)

            with torch.no_grad():
                # set constant weights of the model
                for idx, p in enumerate(layer.parameters()):
                    x = p.data
                    sz = x.view(-1).size(0)
                    shape = x.shape
                    x = torch.cos(torch.arange(0, sz).float().view(shape))
                    p.data.copy_(x)

            return layer

        # this is a deterministic test for TransformerDecoder
        for batch_first in (False, True):
            def perm_fn(x):
                return x.transpose(1, 0) if batch_first else x
            activation = F.relu
            use_cuda = torch.cuda.is_available()
            device = torch.device("cuda" if use_cuda else "cpu")

            decoder_layer = get_a_test_layer(use_cuda=use_cuda, activation=activation,
                                             batch_first=batch_first)

            model = nn.TransformerDecoder(decoder_layer, 1).to(device)

            # deterministic input
            decoder_input = torch.tensor([[[20., 30., 40., 50.]]]).to(device)
            memory_input = torch.tensor([[[60., 70., 80., 90.]]]).to(device)
            result = model(decoder_input, memory_input)
            ref_output = torch.tensor(
                [[[2.314351, 0.094805, -0.671322, 0.101977]]]).to(device)
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-3)

            # deterministic input
            decoder_input = perm_fn(torch.tensor([[[9., 10., 11., 12.]],
                                                  [[11., 12., 13., 14.]]])).to(device)
            memory_input = perm_fn(torch.tensor([[[1., 2., 3., 4.]]])).to(device)
            result = model(decoder_input, memory_input)
            ref_output = perm_fn(torch.tensor([[[2.422245, 0.051716, -0.606338, -0.024756]],
                                               [[2.422245, 0.051716, -0.606338, -0.024756]]]
                                              )).to(device)
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-4)

            # deterministic input
            decoder_input = perm_fn(torch.tensor([[[1., 2., 3., 4.]],
                                                  [[5., 6., 7., 8.]]])).to(device)
            memory_input = perm_fn(torch.tensor([[[9., 10., 11., 12.]],
                                                 [[11., 12., 13., 14.]]])).to(device)
            result = model(decoder_input, memory_input)
            ref_output = perm_fn(torch.tensor([[[2.343536, 0.085561, -0.654954, 0.074991]],
                                               [[2.343536, 0.085561, -0.654954, 0.074991]]]
                                              )).to(device)
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-4)

            # deterministic input
            decoder_input = perm_fn(torch.tensor([[[0.4517, 0.6793, 0.5313, 0.0034],
                                                   [0.2678, 0.3677, 0.4459, 0.7166]],
                                                  [[0.8100, 0.3716, 0.4096, 0.1976],
                                                   [0.6958, 0.8844, 0.6081, 0.8315]],
                                                  [[0.0494, 0.9343, 0.5955, 0.3830],
                                                   [0.5404, 0.3464, 0.9378, 0.6200]]]
                                                 )).to(device)
            memory_input = perm_fn(torch.tensor([[[0.7462, 0.6653, 0.5679, 0.4891],
                                                  [0.5387, 0.1655, 0.3565, 0.0471]],
                                                 [[0.8335, 0.2799, 0.5031, 0.2947],
                                                  [0.1402, 0.0318, 0.7636, 0.1346]],
                                                 [[0.6333, 0.9344, 0.1376, 0.9938],
                                                  [0.8924, 0.2872, 0.6692, 0.2944]],
                                                 [[0.9897, 0.6915, 0.3154, 0.1733],
                                                  [0.8645, 0.3513, 0.3064, 0.0767]],
                                                 [[0.8117, 0.2366, 0.4838, 0.7881],
                                                  [0.3718, 0.4945, 0.9511, 0.0864]]]
                                                )).to(device)
            result = model(decoder_input, memory_input)
            ref_output = perm_fn(torch.tensor([[[2.430065, 0.027862, -0.601136, -0.073096],
                                                [2.431935, 0.028907, -0.599809, -0.072488]],
                                               [[2.428457, 0.027053, -0.602275, -0.073462],
                                                [2.431970, 0.029387, -0.599789, -0.071621]],
                                               [[2.431934, 0.028196, -0.599802, -0.073809],
                                                [2.432306, 0.028858, -0.599542, -0.072846]]]
                                              )).to(device)
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-5)

            # key_padding_mask
            key_padding_mask = torch.zeros(2, 3).to(device) == 1
            result = model(decoder_input, memory_input,
                           tgt_key_padding_mask=key_padding_mask)
            ref_output = perm_fn(torch.tensor([[[2.430065, 0.027862, -0.601136, -0.073096],
                                                [2.431935, 0.028907, -0.599809, -0.072488]],
                                               [[2.428457, 0.027053, -0.602275, -0.073462],
                                                [2.431970, 0.029387, -0.599789, -0.071621]],
                                               [[2.431934, 0.028196, -0.599802, -0.073809],
                                                [2.432306, 0.028858, -0.599542, -0.072846]]]
                                              )).to(device)
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-5)

            # key_padding_mask
            key_padding_mask[0, 2] = 1
            key_padding_mask[1, 1] = 1
            key_padding_mask[1, 2] = 1
            result = model(decoder_input, memory_input,
                           tgt_key_padding_mask=key_padding_mask)
            ref_output = perm_fn(torch.tensor([[[2.430025, 0.027643, -0.601164, -0.073476],
                                                [2.4323, 0.029375, -0.599553, -0.071881]],
                                               [[2.428523, 0.026838, -0.602226, -0.07391],
                                                [2.432634, 0.029842, -0.599318, -0.071253]],
                                               [[2.432278, 0.028152, -0.599555, -0.074139],
                                                [2.432659, 0.029244, -0.599294, -0.072382]]]
                                              )).to(device)
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-5)

            # memory_key_padding_mask
            key_padding_mask = torch.zeros(2, 5).to(device) == 1
            result = model(decoder_input, memory_input,
                           memory_key_padding_mask=key_padding_mask)
            ref_output = perm_fn(torch.tensor([[[2.430065, 0.027862, -0.601136, -0.073096],
                                                [2.431935, 0.028907, -0.599809, -0.072488]],
                                               [[2.428457, 0.027053, -0.602275, -0.073462],
                                                [2.431970, 0.029387, -0.599789, -0.071621]],
                                               [[2.431934, 0.028196, -0.599802, -0.073809],
                                                [2.432306, 0.028858, -0.599542, -0.072846]]]
                                              )).to(device)
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-5)

            # memory_key_padding_mask
            key_padding_mask[0, 4] = 1
            key_padding_mask[1, 3] = 1
            key_padding_mask[1, 4] = 1
            result = model(decoder_input,
                           memory_input,
                           memory_key_padding_mask=key_padding_mask)
            ref_output = perm_fn(torch.tensor([[[2.429757, 0.027358, -0.601351, -0.073816],
                                                [2.432692, 0.028583, -0.599263, -0.073634]],
                                               [[2.428247, 0.02662, -0.602419, -0.074123],
                                                [2.432657, 0.029055, -0.599293, -0.072732]],
                                               [[2.431515, 0.027687, -0.600096, -0.074459],
                                                [2.433075, 0.028543, -0.598987, -0.073985]]]
                                              )).to(device)
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-5)

            # multiple layers no norm
            model = nn.TransformerDecoder(decoder_layer, 2).to(device)

            # deterministic input
            decoder_input = torch.tensor([[[20., 30., 40., 50.]]]).to(device)
            memory_input = torch.tensor([[[60., 70., 80., 90.]]]).to(device)
            result = model(decoder_input, memory_input)
            ref_output = torch.tensor(
                [[[2.31316, 0.0950293, -0.671995, 0.102802]]]).to(device)
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-3)

            # multiple layers no norm
            model = nn.TransformerDecoder(decoder_layer, 6).to(device)

            # deterministic input
            decoder_input = perm_fn(torch.tensor([[[0.4517, 0.6793, 0.5313, 0.0034],
                                                   [0.2678, 0.3677, 0.4459, 0.7166]],
                                                  [[0.8100, 0.3716, 0.4096, 0.1976],
                                                   [0.6958, 0.8844, 0.6081, 0.8315]],
                                                  [[0.0494, 0.9343, 0.5955, 0.3830],
                                                   [0.5404, 0.3464, 0.9378, 0.6200]]]
                                                 )).to(device)
            memory_input = perm_fn(torch.tensor([[[0.7462, 0.6653, 0.5679, 0.4891],
                                                  [0.5387, 0.1655, 0.3565, 0.0471]],
                                                 [[0.8335, 0.2799, 0.5031, 0.2947],
                                                  [0.1402, 0.0318, 0.7636, 0.1346]],
                                                 [[0.6333, 0.9344, 0.1376, 0.9938],
                                                  [0.8924, 0.2872, 0.6692, 0.2944]],
                                                 [[0.9897, 0.6915, 0.3154, 0.1733],
                                                  [0.8645, 0.3513, 0.3064, 0.0767]],
                                                 [[0.8117, 0.2366, 0.4838, 0.7881],
                                                  [0.3718, 0.4945, 0.9511, 0.0864]]]
                                                )).to(device)
            result = model(decoder_input, memory_input)
            ref_output = perm_fn(torch.tensor([[[2.42794, 0.026164, -0.60263, -0.0747591],
                                                [2.43113, 0.0279516, -0.600376, -0.0736896]],
                                               [[2.42794, 0.026164, -0.60263, -0.0747591],
                                                [2.43113, 0.0279516, -0.600376, -0.0736896]],
                                               [[2.42794, 0.026164, -0.60263, -0.0747591],
                                                [2.43113, 0.0279516, -0.600376, -0.0736896]]]
                                              )).to(device)
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-5)

            # multiple layers with norm
            # d_model = 4
            norm = nn.LayerNorm(4)
            model = nn.TransformerDecoder(decoder_layer, 2, norm=norm).to(device)

            # deterministic input
            decoder_input = torch.tensor([[[20., 30., 40., 50.]]]).to(device)
            memory_input = torch.tensor([[[60., 70., 80., 90.]]]).to(device)
            result = model(decoder_input, memory_input)
            ref_output = torch.tensor(
                [[[1.66166, -0.326986, -1.01466, -0.320017]]]).to(device)
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-3)

            # multiple layers with norm
            model = nn.TransformerDecoder(decoder_layer, 6, norm=norm).to(device)

            # deterministic input
            decoder_input = perm_fn(torch.tensor([[[0.4517, 0.6793, 0.5313, 0.0034],
                                                   [0.2678, 0.3677, 0.4459, 0.7166]],
                                                  [[0.8100, 0.3716, 0.4096, 0.1976],
                                                   [0.6958, 0.8844, 0.6081, 0.8315]],
                                                  [[0.0494, 0.9343, 0.5955, 0.3830],
                                                   [0.5404, 0.3464, 0.9378, 0.6200]]]
                                                 )).to(device)
            memory_input = perm_fn(torch.tensor([[[0.7462, 0.6653, 0.5679, 0.4891],
                                                  [0.5387, 0.1655, 0.3565, 0.0471]],
                                                 [[0.8335, 0.2799, 0.5031, 0.2947],
                                                  [0.1402, 0.0318, 0.7636, 0.1346]],
                                                 [[0.6333, 0.9344, 0.1376, 0.9938],
                                                  [0.8924, 0.2872, 0.6692, 0.2944]],
                                                 [[0.9897, 0.6915, 0.3154, 0.1733],
                                                  [0.8645, 0.3513, 0.3064, 0.0767]],
                                                 [[0.8117, 0.2366, 0.4838, 0.7881],
                                                  [0.3718, 0.4945, 0.9511, 0.0864]]]
                                                )).to(device)
            result = model(decoder_input, memory_input)
            ref_output = perm_fn(torch.tensor([[[1.69559, -0.357291, -0.894741, -0.443553],
                                                [1.69571, -0.357363, -0.894154, -0.444196]],
                                               [[1.69559, -0.357291, -0.894741, -0.443553],
                                                [1.69571, -0.357363, -0.894154, -0.444196]],
                                               [[1.69559, -0.357291, -0.894741, -0.443553],
                                                [1.69571, -0.357363, -0.894154, -0.444196]]]
                                              )).to(device)
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-5)

            # gelu activation test cases
            activation = "gelu"
            use_cuda = torch.cuda.is_available()
            device = torch.device("cuda" if use_cuda else "cpu")

            decoder_layer = get_a_test_layer(use_cuda=use_cuda, activation=activation,
                                             batch_first=batch_first)

            model = nn.TransformerDecoder(decoder_layer, 1).to(device)

            # deterministic input
            decoder_input = torch.tensor([[[20., 30., 40., 50.]]]).to(device)
            memory_input = torch.tensor([[[60., 70., 80., 90.]]]).to(device)
            result = model(decoder_input, memory_input)
            ref_output = torch.tensor([[[2.306435, 0.095946, -0.675796, 0.10687]]]).to(device)
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-3)

            # deterministic input
            decoder_input = perm_fn(torch.tensor([[[9., 10., 11., 12.]],
                                                  [[11., 12., 13., 14.]]])).to(device)
            memory_input = perm_fn(torch.tensor([[[1., 2., 3., 4.]]])).to(device)
            result = model(decoder_input, memory_input)
            ref_output = perm_fn(torch.tensor([[[2.415448, 0.054389, -0.610932, -0.0156613]],
                                               [[2.415448, 0.054389, -0.610932, -0.0156613]]])).to(device)
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-4)

            # deterministic input
            decoder_input = perm_fn(torch.tensor([[[1., 2., 3., 4.]],
                                                  [[5., 6., 7., 8.]]])).to(device)
            memory_input = perm_fn(torch.tensor([[[9., 10., 11., 12.]],
                                                 [[11., 12., 13., 14.]]])).to(device)
            result = model(decoder_input, memory_input)
            ref_output = perm_fn(torch.tensor([[[2.338531, 0.087709, -0.65776, 0.080646]],
                                               [[2.338531, 0.087709, -0.65776, 0.080646]]])).to(device)
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-4)

            # deterministic input
            decoder_input = perm_fn(torch.tensor([[[0.4517, 0.6793, 0.5313, 0.0034],
                                                   [0.2678, 0.3677, 0.4459, 0.7166]],
                                                  [[0.8100, 0.3716, 0.4096, 0.1976],
                                                   [0.6958, 0.8844, 0.6081, 0.8315]],
                                                  [[0.0494, 0.9343, 0.5955, 0.3830],
                                                   [0.5404, 0.3464, 0.9378, 0.6200]]]
                                                 )).to(device)
            memory_input = perm_fn(torch.tensor([[[0.7462, 0.6653, 0.5679, 0.4891],
                                                  [0.5387, 0.1655, 0.3565, 0.0471]],
                                                 [[0.8335, 0.2799, 0.5031, 0.2947],
                                                  [0.1402, 0.0318, 0.7636, 0.1346]],
                                                 [[0.6333, 0.9344, 0.1376, 0.9938],
                                                  [0.8924, 0.2872, 0.6692, 0.2944]],
                                                 [[0.9897, 0.6915, 0.3154, 0.1733],
                                                  [0.8645, 0.3513, 0.3064, 0.0767]],
                                                 [[0.8117, 0.2366, 0.4838, 0.7881],
                                                  [0.3718, 0.4945, 0.9511, 0.0864]]]
                                                )).to(device)
            result = model(decoder_input, memory_input)
            ref_output = perm_fn(torch.tensor([[[2.42049104, 0.03443088, -0.60793706, -0.05436271],
                                                [2.42210631, 0.03546578, -0.60679895, -0.05357488]],
                                               [[2.41907674, 0.0336104, -0.60892977, -0.05490462],
                                                [2.42216881, 0.03586554, -0.6067524, -0.05289126]],
                                               [[2.42205716, 0.03488046, -0.60683681, -0.05460596],
                                                [2.42240309, 0.0354595, -0.60659063, -0.05378816]]]
                                              )).to(device)
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-5)

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
            nn.LSTM(10, 20, batch_first=True, proj_size=10),
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
                # LSTM with projections has different hx size
                if rnn.proj_size > 0:
                    hx = torch.randn(1, 5, 10, requires_grad=True, device="cuda")
                    all_vars[1] = hx
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
            nn.LSTM(10, 20, batch_first=True, bidirectional=True, proj_size=10),
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
                # LSTM with projections has different hx size
                if rnn.proj_size > 0:
                    hx = torch.randn(2, 5, 10, requires_grad=True, device="cuda")
                    all_vars[1] = hx
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
        activations = [F.relu, F.gelu]

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
        activations = [F.relu, F.gelu]

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

    def test_projections_lstm_args_check(self):
        input_size = 3
        hidden_size = 5
        proj_size = 2
        num_layers = 2
        batch_size = 4
        seq_len = 6
        num_directions = 1
        bad_size = 7  # prime number so that no size can divide it.

        def test(input_shape, hidden_h_shape, hidden_c_shape):
            for input, hidden in get_inputs(input_shape, hidden_h_shape, hidden_c_shape):
                model = nn.LSTM(input_size, hidden_size, num_layers, proj_size=proj_size)
                self.assertRaises(RuntimeError, lambda: model(input, hidden))

        correct_input_shape = (seq_len, batch_size, input_size)
        correct_hidden_h_shape = (num_layers * num_directions, batch_size, proj_size)
        correct_hidden_c_shape = (num_layers * num_directions, batch_size, hidden_size)

        def update_shape(shape, dim, new_dim_size):
            new_shape = list(shape)
            new_shape[dim] = new_dim_size
            return tuple(new_shape)

        def get_inputs(input_shape, hidden_h_shape, hidden_c_shape):
            '''returns list( tuple(input, hidden) )
            where input, hidden are inputs to a model'''
            input = torch.randn(input_shape)
            hidden_h = torch.randn(hidden_h_shape)
            hidden_c = torch.randn(hidden_c_shape)
            return [(input, (hidden_h, hidden_c))]

        # Incorrect input batch size
        input_shape = update_shape(correct_input_shape, 1, bad_size)
        test(input_shape, correct_hidden_h_shape, correct_hidden_c_shape)

        # Incorrect hidden batch size
        input_shape = correct_input_shape
        hidden_h_shape = update_shape(correct_hidden_h_shape, 1, bad_size)
        hidden_c_shape = update_shape(correct_hidden_c_shape, 1, bad_size)
        test(input_shape, hidden_h_shape, hidden_c_shape)

        # Incorrect input size
        input_shape = update_shape(correct_input_shape, 2, bad_size)
        test(input_shape, correct_hidden_h_shape, correct_hidden_c_shape)

        # Incorrect hidden size
        input_shape = correct_input_shape
        hidden_h_shape = update_shape(correct_hidden_h_shape, 2, bad_size)
        hidden_c_shape = update_shape(correct_hidden_c_shape, 2, bad_size)
        test(input_shape, hidden_h_shape, hidden_c_shape)

        # Incorrect hidden[0]
        input_shape = correct_input_shape
        hidden_h_shape = update_shape(correct_hidden_h_shape, 0, bad_size)
        hidden_c_shape = update_shape(correct_hidden_c_shape, 0, bad_size)
        test(input_shape, hidden_h_shape, hidden_c_shape)

        # Incorrect proj size = hidden size
        input_shape = correct_input_shape
        hidden_h_shape = update_shape(correct_hidden_h_shape, 0, hidden_size)
        hidden_c_shape = correct_hidden_c_shape
        test(input_shape, hidden_h_shape, hidden_c_shape)

        # Incorrect proj size != hidden size
        input_shape = correct_input_shape
        hidden_h_shape = update_shape(correct_hidden_h_shape, 0, bad_size)
        hidden_c_shape = correct_hidden_c_shape
        test(input_shape, hidden_h_shape, hidden_c_shape)

        # Incorrect cell size != hidden size
        input_shape = correct_input_shape
        hidden_h_shape = correct_hidden_h_shape
        hidden_c_shape = update_shape(correct_hidden_c_shape, 0, bad_size)
        test(input_shape, hidden_h_shape, hidden_c_shape)

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

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_projections_lstm_check_device(self):
        input_size = 3
        hidden_size = 5
        proj_size = 2
        num_layers = 2
        batch_size = 4
        seq_len = 6
        num_directions = 1

        correct_input_shape = (seq_len, batch_size, input_size)
        correct_hidden_h_shape = (num_layers * num_directions, batch_size, proj_size)
        correct_hidden_c_shape = (num_layers * num_directions, batch_size, hidden_size)

        model = nn.LSTM(input_size, hidden_size, num_layers, proj_size=proj_size)
        input = torch.randn(correct_input_shape)
        hidden_h = torch.randn(correct_hidden_h_shape)
        hidden_c = torch.randn(correct_hidden_c_shape)

        # input and weights are not at the same device
        with self.assertRaisesRegex(RuntimeError,
                                    "Input and parameter tensors are not at the same device"):
            model(input.to('cuda:0'))

        # input and hiddens are not at the same device
        with self.assertRaisesRegex(RuntimeError,
                                    r"Input and hidden tensors are not at the same device"):
            model(input, (hidden_h.to('cuda:0'), hidden_c.to('cuda:0')))

        # hidden tensors are not at the same CUDA device
        with self.assertRaisesRegex(RuntimeError,
                                    "Input and hidden tensors are not at the same device"):
            model(input.to('cuda:0'), (hidden_h.to('cuda:0'), hidden_c.to('cuda:1')))

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

    def test_projections_lstm_initial_hidden_state(self):
        for bidir in [False, True]:
            rnn = nn.LSTM(30, 20, 2, bidirectional=bidir, proj_size=10)
            num_dirs = 2 if bidir else 1
            input = torch.randn(10, 32, 30)
            hidden_h = torch.zeros(2 * num_dirs, 32, 10)
            hidden_c = torch.zeros(2 * num_dirs, 32, 20)
            hidden = (hidden_h, hidden_c)
            output1, hidden1 = rnn(input, hidden)
            output2, hidden2 = rnn(input)
            self.assertEqual(output1, output2)
            self.assertEqual(hidden1, hidden2)

    def test_projections_errors_on_gru_and_rnn(self):
        error_msg = "proj_size argument is only supported for LSTM, not RNN or GRU"
        for mode in ['RNN', 'GRU']:
            with self.assertRaisesRegex(ValueError, error_msg):
                rnn = getattr(nn, mode)(30, 20, 2, proj_size=10)

    def _test_RNN_cpu_vs_cudnn(self, dropout, dtype=torch.double):

        def forward_backward(cuda, rnn, input_val, grad_output, weights_val, hx_val, grad_hy,
                             cx_val=None, grad_cy=None):
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
                if cx_val is None:
                    hx = (hx_val.clone().requires_grad_(True),
                          hx_val.add(1).requires_grad_(True))
                else:
                    hx = (hx_val.clone().requires_grad_(True),
                          cx_val.add(1).requires_grad_(True))
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
                if grad_cy is not None:
                    grad_cy = grad_cy.cuda()
                grad_output = grad_output.cuda()

            output, hy = rnn(input, hx)

            if isinstance(output, rnn_utils.PackedSequence):
                output = output.data

            if is_lstm:
                if grad_cy is None:
                    torch.autograd.backward([output, hy[0], hy[1]], [grad_output, grad_hy, grad_hy + 1])
                else:
                    torch.autograd.backward([output, hy[0], hy[1]], [grad_output, grad_hy, grad_cy + 1])
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
        proj_size = 3
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
                    self.assertEqual(outputs_cpu[key], outputs_gpu[key], atol=5e-5, rtol=0, msg=key)

            # check grad weights separately, as nested dict
            for cpu_layer_weight, gpu_layer_weight in zip(outputs_cpu['weights'], outputs_gpu['weights']):
                for (cpu_weight, gpu_weight) in zip(cpu_layer_weight, gpu_layer_weight):
                    self.assertEqual(cpu_weight.grad.data, gpu_weight.grad.data, atol=5e-5, rtol=0)

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

                hx_val = torch.randn(num_layers * num_directions, batch, hidden_size, dtype=dtype)
                grad_hy = torch.randn(num_layers * num_directions, batch, hidden_size, dtype=dtype)

                if not contig:
                    grad_output = make_noncontig(grad_output)
                    grad_hy = make_noncontig(grad_hy)
                    input_var = make_noncontig(input_val)
                    hx_val = make_noncontig(hx_val)

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
                             batch_first=batch_first).to(dtype)

                outputs_cpu = forward_backward(
                    False, rnn, input_val, grad_output, rnn.all_weights, hx_val, grad_hy)

                rnn_gpu = module(input_size,
                                 hidden_size,
                                 num_layers,
                                 bias=bias,
                                 dropout=dropout,
                                 bidirectional=bidirectional,
                                 batch_first=batch_first).to(dtype)

                outputs_gpu = forward_backward(
                    True, rnn_gpu, input_val, grad_output, rnn.all_weights, hx_val, grad_hy)

                compare_cpu_gpu(outputs_cpu, outputs_gpu)

        for nonlinearity in ('tanh', 'relu'):
            hx_val = torch.randn(num_layers, batch, hidden_size, dtype=dtype)
            input_val = torch.randn(seq_length, batch, input_size, dtype=dtype)
            grad_output = torch.randn(
                seq_length, batch, hidden_size * num_directions, dtype=dtype)
            grad_hy = torch.randn(
                num_layers * num_directions, batch, hidden_size, dtype=dtype)

            rnn = nn.RNN(input_size, hidden_size, num_layers, bias=bias, nonlinearity=nonlinearity).to(dtype)
            outputs_cpu = forward_backward(False, rnn, input_val, grad_output, rnn.all_weights, hx_val, grad_hy)

            rnn_gpu = nn.RNN(input_size, hidden_size, num_layers, bias=bias, nonlinearity=nonlinearity).to(dtype)
            outputs_gpu = forward_backward(True, rnn_gpu, input_val, grad_output, rnn.all_weights, hx_val, grad_hy)

            compare_cpu_gpu(outputs_cpu, outputs_gpu)

        # checking LSTM with projections
        for bias, bidirectional, batch_first, contig, variable_len, lens_as_tensor \
                in product((True, False), repeat=6):
            num_directions = 2 if bidirectional else 1
            if batch_first:
                input_val = torch.randn(batch, seq_length, input_size, dtype=dtype)
                grad_output = torch.randn(batch, seq_length, proj_size * num_directions, dtype=dtype)
            else:
                input_val = torch.randn(seq_length, batch, input_size, dtype=dtype)
                grad_output = torch.randn(seq_length, batch, proj_size * num_directions, dtype=dtype)

            hx_val = torch.randn(num_layers * num_directions, batch, proj_size, dtype=dtype)
            cx_val = torch.randn(num_layers * num_directions, batch, hidden_size, dtype=dtype)
            grad_hy = torch.randn(num_layers * num_directions, batch, proj_size, dtype=dtype)
            grad_cy = torch.randn(num_layers * num_directions, batch, hidden_size, dtype=dtype)

            if not contig:
                grad_output = make_noncontig(grad_output)
                grad_hy = make_noncontig(grad_hy)
                grad_cy = make_noncontig(grad_cy)
                input_var = make_noncontig(input_val)
                hx_val = make_noncontig(hx_val)
                cx_val = make_noncontig(cx_val)

            if variable_len:
                lengths = [7, 5, 5, 2, 1, 1]
                if lens_as_tensor:
                    lengths = torch.tensor(lengths, dtype=torch.long)
                input_val = rnn_utils.pack_padded_sequence(input_val, lengths, batch_first=batch_first)
                grad_output = rnn_utils.pack_padded_sequence(grad_output, lengths, batch_first=batch_first).data

            rnn = nn.LSTM(input_size,
                          hidden_size,
                          num_layers,
                          bias=bias,
                          dropout=dropout,
                          bidirectional=bidirectional,
                          batch_first=batch_first,
                          proj_size=proj_size).to(dtype)

            outputs_cpu = forward_backward(
                False, rnn, input_val, grad_output, rnn.all_weights,
                hx_val, grad_hy, cx_val, grad_cy)

            rnn_gpu = nn.LSTM(input_size,
                              hidden_size,
                              num_layers,
                              bias=bias,
                              dropout=dropout,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              proj_size=proj_size).to(dtype)

            outputs_gpu = forward_backward(
                True, rnn_gpu, input_val, grad_output, rnn.all_weights,
                hx_val, grad_hy, cx_val, grad_cy)
            compare_cpu_gpu(outputs_cpu, outputs_gpu)

    @unittest.skipIf(not TEST_CUDNN, "needs cudnn")
    def test_RNN_cpu_vs_cudnn_no_dropout(self):
        dtype = torch.double
        self._test_RNN_cpu_vs_cudnn(0, dtype)

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

        # runs on CPU to acquire expected output
        def check_weight_norm(m, name):
            input = torch.randn(seq_length, batch, input_size)
            expected_output = m(input)

            # adds weight normalization
            m = torch.nn.utils.weight_norm(m, name=name)

            # moves to CUDA
            m = m.cuda()
            input = input.cuda()

            # otherwise, subsequent warnings will be hidden, and further tests rely on them
            warnings.simplefilter("always")
            self.assertEqual(m(input), expected_output)

            # remove weight norm
            m = torch.nn.utils.remove_weight_norm(m, name=name)
            self.assertEqual(m(input), expected_output)

        check_weight_norm(nn.LSTM(input_size, hidden_size, num_layers), 'weight_hh_l0')
        check_weight_norm(nn.LSTM(input_size, hidden_size, num_layers, proj_size=3), 'weight_hr_l0')

    @unittest.skipIf(not TEST_CUDA, 'CUDA not available')
    def test_partial_flat_weights(self):
        input_size = 10
        hidden_size = 6
        num_layers = 2

        m = nn.LSTM(input_size, hidden_size, num_layers)
        inp = torch.randn(3, 2, 10)
        out_expected = m(inp)
        # deletes an attribute of original LSTM
        weight_orig = m.weight_hh_l0
        del m.weight_hh_l0
        self.assertFalse(hasattr(m, "weight_hh_l0"))
        # verifies that moving to CUDA with only some attributes defined
        # does not throw an error
        m.cuda()
        # recompute the weight and make sure that module can be used
        m.weight_hh_l0 = weight_orig.cuda()
        inp = inp.cuda()
        # otherwise, subsequent warnings will be hidden, and further tests rely on them
        warnings.simplefilter("always")
        self.assertEqual(m(inp)[0].cpu(), out_expected[0])


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

    def test_error_RNN_seq_len_zero(self):
        # checking error message when RNN has seq_len = 0
        for module in (nn.RNN, nn.LSTM, nn.GRU):
            for bidirectional in [True, False]:
                for device in get_all_device_types():
                    input = torch.ones(0, 10, 5)
                    rnn = module(5, 6, bidirectional=bidirectional)
                    if device == 'cuda':
                        rnn.cuda()
                        input = input.cuda()

                    with self.assertRaisesRegex(RuntimeError, "Expected sequence length to be larger than 0 in RNN"):
                        rnn(input)

    def test_RNN_input_size_zero(self):
        for module in (nn.RNN, nn.LSTM, nn.GRU):
            for device in get_all_device_types():
                input = torch.zeros((5, 0, 3))
                rnn = module(input_size=3, hidden_size=4)
                if device == 'cuda':
                    rnn.cuda()
                    input = input.cuda()
                outs = rnn(input)
                self.assertEqual(outs[0].shape, torch.Size([5, 0, 4]))
                # Check that backward does not cause a hard error
                outs[0].sum().backward()

    @unittest.skipIf(not (TEST_CUDNN and (TEST_CUDNN_VERSION if TEST_CUDNN_VERSION else 0) >= 5103), "needs cudnn >= 5.1")
    def test_RNN_dropout_state(self):
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

                    buf = io.BytesIO()
                    rnn_pickle = torch.save(rnn, buf)
                    buf.seek(0)
                    rnn2 = torch.load(buf)
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


    def test_pixel_shuffle_unshuffle(self):
        def _test_pixel_shuffle_unshuffle_helper(num_input_dims, valid_channels_dim=True,
                                                 upscale_factor=None):
            # Function to imperatively ensure pixels are shuffled to the correct locations.
            # Used to validate the batch operations in pixel_shuffle.
            def _verify_pixel_shuffle(input, output, upscale_factor):
                for c in range(output.size(-3)):
                    for h in range(output.size(-2)):
                        for w in range(output.size(-1)):
                            height_idx = h // upscale_factor
                            weight_idx = w // upscale_factor
                            channel_idx = (upscale_factor * (h % upscale_factor)) + (w % upscale_factor) + \
                                          (c * upscale_factor ** 2)
                            self.assertEqual(output[..., c, h, w], input[..., channel_idx, height_idx, weight_idx])

            upscale_factor = random.randint(2, 5) if upscale_factor is None else upscale_factor
            # If valid_channels_dim=False, add 1 to make channels dim indivisible by upscale_factor ** 2.
            channels = random.randint(1, 4) * upscale_factor ** 2 + (0 if valid_channels_dim else 1)
            height = random.randint(5, 10)
            width = random.randint(5, 10)

            if num_input_dims == 1:
                input = torch.rand(channels, requires_grad=True)
            elif num_input_dims == 2:
                input = torch.rand(height, width, requires_grad=True)
            else:
                batch_sizes = [random.randint(1, 3) for _ in range(num_input_dims - 3)]
                input = torch.rand(*batch_sizes, channels, height, width, requires_grad=True)
            ps = nn.PixelShuffle(upscale_factor)
            pus = nn.PixelUnshuffle(downscale_factor=upscale_factor)

            if num_input_dims >= 3 and valid_channels_dim and upscale_factor > 0:
                output = ps(input)
                _verify_pixel_shuffle(input, output, upscale_factor)
                output.backward(output.data)
                self.assertEqual(input.data, input.grad.data)

                # Ensure unshuffle properly inverts shuffle.
                unshuffle_output = pus(output)
                self.assertEqual(input, unshuffle_output)
            else:
                self.assertRaises(RuntimeError, lambda: ps(input))

        def _test_pixel_unshuffle_error_case_helper(num_input_dims, valid_height_dim=True, valid_width_dim=True,
                                                    downscale_factor=None):
            downscale_factor = random.randint(2, 5) if downscale_factor is None else downscale_factor
            channels = random.randint(1, 4)
            # If valid_height_dim=False, add 1 to make height dim indivisible by downscale_factor.
            height = random.randint(3, 5) * abs(downscale_factor) + (0 if valid_height_dim else 1)
            # If valid_width_dim=False, add 1 to make width dim indivisible by downscale_factor.
            width = random.randint(3, 5) * abs(downscale_factor) + (0 if valid_width_dim else 1)

            if num_input_dims == 1:
                input = torch.rand(channels, requires_grad=True)
            elif num_input_dims == 2:
                input = torch.rand(height, width, requires_grad=True)
            else:
                batch_sizes = [random.randint(1, 3) for _ in range(num_input_dims - 3)]
                input = torch.rand(*batch_sizes, channels, height, width, requires_grad=True)

            pus = nn.PixelUnshuffle(downscale_factor)
            self.assertRaises(RuntimeError, lambda: pus(input))

        def _test_pixel_shuffle_unshuffle_for_input_dims(num_input_dims):
            # For 1D - 2D, this is an error case.
            # For 3D - 5D, this is a success case for pixel_shuffle + pixel_unshuffle.
            _test_pixel_shuffle_unshuffle_helper(num_input_dims=num_input_dims)

            # Error cases for pixel_shuffle.
            _test_pixel_shuffle_unshuffle_helper(num_input_dims=num_input_dims, valid_channels_dim=False)
            _test_pixel_shuffle_unshuffle_helper(num_input_dims=num_input_dims, upscale_factor=0)
            _test_pixel_shuffle_unshuffle_helper(num_input_dims=num_input_dims, upscale_factor=-2)

            # Error cases for pixel_unshuffle.
            _test_pixel_unshuffle_error_case_helper(num_input_dims=num_input_dims, valid_height_dim=False)
            _test_pixel_unshuffle_error_case_helper(num_input_dims=num_input_dims, valid_width_dim=False)
            _test_pixel_unshuffle_error_case_helper(num_input_dims=num_input_dims, downscale_factor=0)
            _test_pixel_unshuffle_error_case_helper(num_input_dims=num_input_dims, downscale_factor=-2)

        def test_pixel_shuffle_unshuffle_1D():
            _test_pixel_shuffle_unshuffle_for_input_dims(num_input_dims=1)

        def test_pixel_shuffle_unshuffle_2D():
            _test_pixel_shuffle_unshuffle_for_input_dims(num_input_dims=2)

        def test_pixel_shuffle_unshuffle_3D():
            _test_pixel_shuffle_unshuffle_for_input_dims(num_input_dims=3)

        def test_pixel_shuffle_unshuffle_4D():
            _test_pixel_shuffle_unshuffle_for_input_dims(num_input_dims=4)

        def test_pixel_shuffle_unshuffle_5D():
            _test_pixel_shuffle_unshuffle_for_input_dims(num_input_dims=5)

        test_pixel_shuffle_unshuffle_1D()
        test_pixel_shuffle_unshuffle_2D()
        test_pixel_shuffle_unshuffle_3D()
        test_pixel_shuffle_unshuffle_4D()
        test_pixel_shuffle_unshuffle_5D()

    def test_pixel_shuffle_nhwc_cpu(self):
        input = torch.randn(3, 18, 4, 4, device='cpu')
        input = input.contiguous(memory_format=torch.channels_last).requires_grad_()
        grad = torch.randn(3, 18, 4, 4, device='cpu')
        ps = torch.nn.PixelShuffle(3)
        pus = torch.nn.PixelUnshuffle(3)

        ref_input = input.detach().clone().contiguous().requires_grad_(True)
        ref_grad = grad.detach().clone().contiguous()
        ref_ps = torch.nn.PixelShuffle(3)
        ref_pus = torch.nn.PixelUnshuffle(3)

        out = pus(ps(input))
        out.backward(grad)
        ref_out = ref_pus(ref_ps(ref_input))
        ref_out.backward(ref_grad)

        self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))
        self.assertTrue(ref_out.is_contiguous())
        self.assertEqual(out, ref_out)
        self.assertEqual(input.grad, ref_input.grad)

    # These tests should be OpInfo'd
    def test_elu_inplace_on_view(self):
        v = torch.tensor([1.0, -1.0, 1.0, -1.0], requires_grad=True)

        def func(root):
            x = root.clone()
            view = x.narrow(0, 1, 2)
            res = F.elu(view, inplace=True)
            self.assertIs(res, view)
            return x

        gradcheck(func, [v])
        gradgradcheck(func, [v])

    def test_elu_inplace_gradgrad(self):
        v = torch.randn(8, requires_grad=True)

        def func(root):
            x = root.clone()
            return F.elu(x, inplace=True)

        gradcheck(func, [v])
        gradgradcheck(func, [v])

    def test_relu_inplace_on_view(self):
        v = torch.tensor([1.0, -1.0, 1.0, -1.0], requires_grad=True)

        def func(root):
            x = root.clone()
            view = x.narrow(0, 1, 2)
            res = F.relu(view, inplace=True)
            self.assertIs(res, view)
            return x

        gradcheck(func, [v])
        gradgradcheck(func, [v])

    def test_PReLU_backward_requires_grad_false(self):
        devices = ['cpu']
        devices += ['cuda'] if TEST_CUDA else []
        for d in devices:
            m = nn.PReLU().to(d)
            x = torch.randn(2, 3, 4, 5, device=d, requires_grad=False)
            y = m(x)
            y.mean().backward()
            self.assertEqual(x.grad, None)

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

    def test_bce_loss_input_range(self):
        bceloss = nn.BCELoss()

        target = torch.rand(25, 25)
        output_valid = torch.rand(25, 25)
        output_too_negative = output_valid - 1.0
        output_too_positive = output_valid + 1.0

        loss_valid = bceloss(output_valid, target)
        with self.assertRaisesRegex(RuntimeError, 'between 0 and 1'):
            loss_too_negative = bceloss(output_too_negative, target)
        with self.assertRaisesRegex(RuntimeError, 'between 0 and 1'):
            loss_too_positive = bceloss(output_too_positive, target)

    def test_bce_loss_size_mismatch(self):
        bceloss = nn.BCELoss()
        a = torch.rand(25)
        b = torch.rand(25, 1)
        with self.assertRaisesRegex(ValueError, r'Using a target size \('):
            bceloss(a, b)

    def test_bce_with_logits_gives_same_result_as_sigmoid_and_bce_loss_large_tensors_with_grad(self):
        x_size = 1024
        y_size = 256
        target = torch.rand(x_size, y_size)

        for reduction in ['none', 'mean', 'sum']:
            output_sig = torch.rand(x_size, y_size) - 0.5
            output_logits = output_sig.clone().detach()

            output_sig.requires_grad = True
            output_logits.requires_grad = True
            weight = torch.rand(y_size)

            loss_sig = nn.BCELoss(weight, reduction=reduction)(
                torch.sigmoid(output_sig), target
            )
            loss_logits = nn.BCEWithLogitsLoss(weight, reduction=reduction)(
                output_logits, target
            )

            self.assertEqual(loss_logits, loss_sig)

            if reduction == 'none':
                grad = torch.rand(x_size, y_size)
                loss_sig.backward(grad)
                loss_logits.backward(grad)
            else:
                loss_sig.backward()
                loss_logits.backward()

            self.assertEqual(output_sig.grad, output_logits.grad)

    def test_bce_with_logits_has_correct_forward_grad(self):
        output = torch.randn(3, 5, requires_grad=True)
        target = torch.randn(3, 5)
        for reduction in ('sum', 'mean', 'none'):
            gradcheck(lambda self, target: nn.BCEWithLogitsLoss(reduction=reduction)(self, target),
                      (output, target), check_forward_ad=True)

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

    def test_batchnorm_nhwc_cpu(self):
        def helper(self, size, dtype, mixed_dtype=False):
            channels = size[1]
            input = torch.randn(size, dtype=dtype, device='cpu', requires_grad=True)
            input = input.contiguous(memory_format=torch.channels_last).to(dtype)
            input.retain_grad()
            grad = torch.randn(size, dtype=dtype, device='cpu')
            grad = grad.contiguous(memory_format=torch.channels_last)
            bn = nn.BatchNorm2d(channels).cpu().to(dtype)
            bn.weight.data.uniform_()
            bn.bias.data.uniform_()

            ref_input = input.detach().clone().contiguous().requires_grad_(True)
            ref_grad = grad.detach().clone().contiguous()
            ref_bn = nn.BatchNorm2d(channels).cpu().to(dtype)
            ref_bn.load_state_dict(bn.state_dict())

            if mixed_dtype:
                bn.float()
                ref_bn.float()

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

        # test NC11 and N1HW; test mixed dtype
        for shape in [(4, 8, 10, 10), (4, 1, 9, 9), (4, 9, 1, 1)]:
            helper(self, shape, torch.float, False)
            helper(self, shape, torch.bfloat16, False)
            helper(self, shape, torch.bfloat16, True)

    def test_batchnorm_non_contig_cpu(self):
        input = torch.arange(6, dtype=torch.float).reshape(1, 3, 2, 1).cpu()
        input = input.permute(0, 2, 1, 3)

        bn = torch.nn.BatchNorm2d(2).cpu().float().eval()
        bn.weight.data.uniform_()
        bn.bias.data.uniform_()

        ref_input = input.detach().clone().contiguous()
        ref_bn = nn.BatchNorm2d(2).cpu().float().eval()
        ref_bn.load_state_dict(bn.state_dict())

        out = bn(input)
        ref_out = ref_bn(ref_input)

        self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))
        self.assertTrue(ref_out.is_contiguous())
        self.assertEqual(out, ref_out)

        input_bf = torch.arange(24, dtype=torch.bfloat16).reshape(1, 3, 2, 4)
        input_bf = input_bf.permute(0, 2, 1, 3)
        input_f = input_bf.float()
        bn_mix = torch.nn.BatchNorm2d(2).float().eval()
        ref_bn_f = deepcopy(bn_mix)
        out_bf = bn_mix(input_bf)
        ref_out_bf = ref_bn_f(input_f)
        self.assertEqual(ref_out_bf, out_bf.float(), atol=0.05, rtol=0.05)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    @unittest.skipIf(not TEST_CUDNN, "needs cudnn")
    def test_batchnorm_cudnn_nhwc(self):
        def run_test(input, grad_output):
            c = input.size(1)
            mod = nn.BatchNorm2d(c).cuda().float()
            mod.weight.data.uniform_()
            mod.bias.data.uniform_()
            ref_input = input.detach().clone().contiguous().requires_grad_(True)
            ref_grad = grad.detach().clone().contiguous()
            ref_mod = nn.BatchNorm2d(c).cuda().float()
            ref_mod.load_state_dict(mod.state_dict())
            out = mod(input)
            out.backward(grad_output)
            ref_out = ref_mod(ref_input)
            ref_out.backward(ref_grad)
            self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))
            self.assertTrue(ref_out.is_contiguous())
            self.assertEqual(out, ref_out)
            self.assertEqual(mod.weight.grad, ref_mod.weight.grad)
            self.assertEqual(mod.bias.grad, ref_mod.bias.grad)
            self.assertEqual(input.grad, ref_input.grad)

        input = torch.randint(1, 10, (4, 8, 2, 2), dtype=torch.float32, device="cuda")
        input = input.contiguous(memory_format=torch.channels_last).detach().requires_grad_()

        grad = torch.randint(1, 10, (4, 8, 2, 2), dtype=torch.float32, device="cuda")
        grad = grad.contiguous(memory_format=torch.channels_last)
        run_test(input, grad)
        # see #42588, grad is channels_last contiguous, but grad.suggest_memory_format (rightly) return "contiguous"
        # not channels_last
        input = torch.randint(1, 10, (2, 8, 8, 1), dtype=torch.float32, device="cuda")
        input = input.contiguous(memory_format=torch.channels_last).detach().requires_grad_()
        grad = torch.randint(1, 10, (2, 8, 8, 1), dtype=torch.float32, device="cuda")
        grad = grad.permute(0, 2, 1, 3)
        run_test(input, grad)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_batchnorm_cudnn_half(self):
        # THNN
        input = torch.randint(1, 10, (2, 3, 2, 2), dtype=torch.half, device="cuda", requires_grad=True)
        m = nn.BatchNorm2d(3).half().cuda()
        thnn_output = m(input)
        thnn_output.sum().backward()
        thnn_input_grad = input.grad.data.clone()
        self.assertEqualTypeString(thnn_output, input)
        # cuDNN
        if TEST_CUDNN:
            input.grad = None
            m = m.float()
            cudnn_output = m(input)
            cudnn_output.sum().backward()
            cudnn_input_grad = input.grad.data.clone()
            self.assertEqualTypeString(cudnn_output, input)
            self.assertEqual(cudnn_output, thnn_output)
            self.assertEqual(cudnn_input_grad, thnn_input_grad, atol=1e-3, rtol=0)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_batchnorm_nonaffine_cuda_half_input(self):
        input = torch.randn(16, 3, 24, 24, dtype=torch.half, device="cuda")
        m = nn.BatchNorm2d(3, affine=False).cuda().float()  # keep running stats in FP32
        output = m(input)
        self.assertEqualTypeString(output, input)
        m.eval()
        output = m(input)
        self.assertEqualTypeString(output, input)

    def test_batchnorm_raises_error_if_less_than_one_value_per_channel(self):
        x = torch.rand(10)[None, :, None]
        with self.assertRaises(ValueError):
            torch.nn.BatchNorm1d(10)(x)

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

    def test_batchnorm_raises_error_if_running_var_or_running_mean_have_forward_grad(self):
        args = (
            torch.randn(3, 2, 5),  # input
            torch.randn(2),  # running_mean
            torch.randn(2),  # running_var
        )
        kwargs = {'training': False, 'momentum': -1.2}
        fn = partial(F.batch_norm, **kwargs)

        for dual_indices in ((0,), (1,), (1, 2), (0, 1), (0, 1, 2),):
            tangents = tuple(torch.rand_like(x) for x in args)

            with fwAD.dual_level():
                duals = [fwAD.make_dual(primal, tangent) if i in dual_indices else primal
                         for i, (primal, tangent) in enumerate(zip(args, tangents))]
                msg = "batch_norm is not differentiable wrt running_mean and running_var"
                # 0 needs to have forward grad because otherwise we won't even run batch_norm_jvp
                if (1 in dual_indices or 2 in dual_indices) and 0 in dual_indices:
                    with self.assertRaisesRegex(RuntimeError, msg):
                        fn(*duals)
                else:
                    fn(*duals)

    def test_batchnorm_buffer_update_when_stats_are_not_tracked(self):
        input_size = (32, 4)
        # Instantiate BN with buffers that are not None
        bn = nn.BatchNorm1d(input_size[1], track_running_stats=True)
        # Use buffers for normalization but don't update them
        bn.track_running_stats = False
        # Store initial values
        num_batches = bn.num_batches_tracked.clone()
        running_mean = bn.running_mean.clone()
        running_var = bn.running_var.clone()
        # Forward random tensor
        _ = bn(torch.rand(input_size))
        # Ensure none of the buffers has been updated
        self.assertTrue(torch.equal(num_batches, bn.num_batches_tracked))
        self.assertTrue(torch.equal(running_mean, bn.running_mean))
        self.assertTrue(torch.equal(running_var, bn.running_var))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_batchnorm_nhwc_cuda(self):
        for dtype in (torch.half, torch.float):
            (N, C, H, W) = 2, 64, 50, 50
            model = torch.nn.BatchNorm2d(C, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            model = model.eval().cuda().to(dtype)
            inp1 = torch.randn(N, C, H, W, device=torch.device('cuda'), dtype=dtype)
            inp2 = inp1.contiguous(memory_format=torch.channels_last)
            out1 = model(inp1)
            out2 = model(inp2)
            self.assertTrue(torch.equal(out1, out2))

    def test_pairwise_distance(self):
        input1 = torch.randn(4, 4, requires_grad=True)
        input2 = torch.randn(4, 4, requires_grad=True)
        self.assertTrue(gradcheck(lambda x, y: F.pairwise_distance(x, y), (input1, input2)))

    # TODO: Create an OpInfo for pdist
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

    @unittest.expectedFailure
    def test_pdist_cuda_gradgrad_unimplemented(self):
        inp = torch.randn(4, 5, device='cuda', requires_grad=True)
        gradgradcheck(F.pdist, (inp,))

    # Merge into OpInfo?
    # test for backward in https://github.com/pytorch/pytorch/issues/15511
    def test_pdist_large(self):
        for device in device_():
            def func(x):
                return torch.pdist(x, p=2)

            # shape[0] should be able to be (roughly) arbitrarily large, but the kernel
            # is currently limited to smaller sizes (see issue above); this is just testing
            # a floor.
            shape = (1000, 1)
            x = torch.randn(shape, device=device).requires_grad_()
            output = torch.pdist(x, p=2)
            # just run a single backward, as gradcheck/gradgradcheck is expensive here
            output.sum().backward()

    def test_cosine_embedding_loss_with_diff_type(self):
        for device in device_():
            input1 = torch.tensor([[2, 3, 4], [6, 2, 4]], dtype=torch.double, device=device)
            input2 = torch.tensor([[2, 3, 5], [3, 2, 1]], dtype=torch.double, device=device)
            target = torch.tensor([1, -1], dtype=torch.int, device=device)
            expected = torch.nn.functional.cosine_embedding_loss(input1, input2, target)
            for dt1 in get_all_math_dtypes(device):
                for dt2 in get_all_math_dtypes(device):
                    for dt3 in get_all_math_dtypes(device):
                        # dt3 is used as dtype for target = [1, -1], so let's skip unsigned type
                        if dt3 == torch.uint8:
                            continue
                        if dt1.is_complex or dt2.is_complex or dt3.is_complex:
                            continue
                        input1 = input1.to(dt1)
                        input2 = input2.to(dt2)
                        target = target.to(dt3)
                        result = torch.nn.functional.cosine_embedding_loss(input1, input2, target)
                        self.assertEqual(result.item(), expected.item(), atol=0.001, rtol=0)

    def test_kl_div_with_diff_type(self):
        for device in device_():
            input = torch.tensor([[2, 3, 5], [3, 2, 1]], dtype=torch.double, device=device)
            target = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.double, device=device)
            expected = torch.nn.functional.kl_div(input, target)
            real_dtypes = (torch.float32, torch.float64, torch.float16)
            for input_dtype, target_dtype in product(real_dtypes, repeat=2):
                if (torch.device(device).type == 'cpu' and target_dtype == torch.float16):
                    continue
                input = input.to(input_dtype)
                target = target.to(target_dtype)
                result = torch.nn.functional.kl_div(input, target)
                self.assertEqual(result.item(), expected.item(), atol=0.001, rtol=0)

    def test_kl_div_with_diff_type_log_target(self):
        for device in device_():
            input = torch.tensor([[2, 3, 5], [3, 2, 1]], dtype=torch.double, device=device)
            target = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.double, device=device).log()
            expected = torch.nn.functional.kl_div(input, target, log_target=True)
            real_dtypes = (torch.float32, torch.float64, torch.float16)
            for input_dtype, target_dtype in product(real_dtypes, repeat=2):
                if (torch.device(device).type == 'cpu' and target_dtype == torch.float16):
                    continue
                input = input.to(input_dtype)
                target = target.to(target_dtype)
                result = torch.nn.functional.kl_div(input, target, log_target=True)
                self.assertEqual(result.item(), expected.item(), atol=0.001, rtol=0)

    def test_kl_div_log_softmax_target(self):
        for device in device_():
            a = torch.tensor([[1.0, 2, 3], [5.0, 5, 5]], device=device)
            b = torch.tensor([[1.0, 2, 3], [5.0, 5, 5]], device=device)
            self.assertEqual(
                F.kl_div(F.log_softmax(a, 1), F.log_softmax(b, 1), reduction='none', log_target=True),
                torch.zeros_like(a)
            )

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

    def test_cosine_embedding_loss_invalid_shape(self):
        input1 = torch.randn(15, 10)
        input2 = torch.randn(15, 10)
        target = torch.randn(15, 1).sign()

        with self.assertRaisesRegex(RuntimeError, "1D target tensor expected"):
            F.cosine_embedding_loss(input1, input2, target)

        with self.assertRaisesRegex(RuntimeError, "1D target tensor expects 2D input tensors"):
            F.cosine_embedding_loss(torch.randn(10), torch.randn(10), torch.randn(10))

        with self.assertRaisesRegex(RuntimeError, "0D target tensor expects 1D input tensors"):
            F.cosine_embedding_loss(torch.randn(2, 5), torch.randn(2, 5), torch.randn(()))

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
            'huber_loss': lambda x, y, r: F.huber_loss(x, y, reduction=r),
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
        for dtype in [torch.float, torch.cfloat]:
            for N in range(1, 50, 10):
                input = torch.rand(N, 3, 1024, 1024, dtype=dtype)
                self.assertEqual(
                    torch.nn.L1Loss()(input, torch.zeros_like(input)),
                    input.abs().mean())

    def test_smoothl1loss_intergral_target(self):
        def _input_grad(input, target, reduction):
            output = F.smooth_l1_loss(input, target, reduction=reduction, beta=0.5)
            output.sum().backward()
            return input.grad

        for device, dtype, reduction in product(device_(),
                                                integral_types(),
                                                ('none', 'sum', 'mean')):
            input = torch.randn(2, 2, device=device, requires_grad=True)
            target = torch.randint(0, 9, (2, 2), device=device, dtype=dtype)

            input_grad_with_float_target = _input_grad(input, target.float(), reduction)

            input_grad = _input_grad(input.detach().clone().requires_grad_(True),
                                     target,
                                     reduction)
            self.assertEqual(input_grad, input_grad_with_float_target)

    def test_smoothl1loss_negative_beta_not_supported(self):
        with self.assertRaises(RuntimeError):
            F.smooth_l1_loss(torch.randn(2, 2), torch.randn(2, 2), beta=-1.0)

    def test_huber_loss_invalid_delta(self):
        def _test_huber_loss_delta_error_helper(delta):
            input, target = torch.randn(2, 2), torch.randn(2, 2)
            loss = torch.nn.HuberLoss(delta=delta)
            with self.assertRaises(RuntimeError):
                loss(input, target)

        def test_huber_loss_negative_delta():
            _test_huber_loss_delta_error_helper(delta=-0.5)

        def test_huber_loss_zero_delta():
            _test_huber_loss_delta_error_helper(delta=0.0)

        test_huber_loss_negative_delta()
        test_huber_loss_zero_delta()

    def test_cosine_similarity(self):
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
        # previous behavior: <x,y>/max(eps, ||x|| * ||y||)
        # current: <x/max(eps, ||x||), y/max(eps,||y||)>
        # if f(x,y) is the cosine similarity, then
        # df/dx = y/(||x|| * ||y||) - (x * <x,y> * ||y||/||x||)/(||x|| * ||y||)^2
        # the tests below check division by zero in the backward formula when
        # x := input2 = 0, y := input1 != 0.
        # For these inputs the gradient wrt x simplifies to g(x,y) := y/(||x|| * ||y||)
        # Previous test checks g(x,y) == y/eps,
        # Current test checks g(x,y) == (y/||y||)/eps.
        input1 = torch.randn(10).requires_grad_()
        input2 = torch.zeros_like(input1).requires_grad_()
        torch.cosine_similarity(input1, input2, 0).sum().backward()
        self.assertEqual(input1.grad, torch.zeros_like(input1))
        self.assertEqual(input2.grad, input1 / input1.norm() * 1e8)

        # Check type promotion, issue #61454
        input = torch.tensor(12.)
        out = F.cosine_similarity(input.to(torch.int8), input, dim=-1)
        self.assertEqual(out, 1.)

    def test_grid_sample_error_checking(self):
        input = torch.empty(1, 1, 2, 2)
        grid = torch.empty(1, 1, 1, 2)

        # assert no error
        F.grid_sample(input, grid, align_corners=False)

        with self.assertRaisesRegex(ValueError, "but got: 'garbage'"):
            F.grid_sample(input, grid, mode='garbage', align_corners=False)

        with self.assertRaisesRegex(ValueError, "but got: 'garbage'"):
            F.grid_sample(input, grid, padding_mode='garbage', align_corners=False)

        with self.assertRaisesRegex(RuntimeError, "expected grid to have size 1 in last dimension"):
            F.grid_sample(input[0], grid, align_corners=False)

        with self.assertRaisesRegex(RuntimeError, "expected grid to have size 2 in last dimension"):
            F.grid_sample(input, torch.empty(1, 1, 1, 1, 3), align_corners=False)

        with self.assertRaisesRegex(RuntimeError, "expected grid and input to have same batch size"):
            F.grid_sample(input, torch.empty(2, 1, 1, 2), align_corners=False)

        with self.assertRaisesRegex(RuntimeError, "expected grid to have size 2 in last dimension"):
            F.grid_sample(input, torch.empty(1, 1, 1, 3), align_corners=False)

        with self.assertRaisesRegex(RuntimeError, "expected input to have non-empty spatial dimensions"):
            F.grid_sample(torch.empty(1, 1, 0, 2), grid, align_corners=False)

        with self.assertRaisesRegex(RuntimeError, "bicubic interpolation only supports 4D input"):
            F.grid_sample(torch.empty(1, 1, 2, 2, 2), torch.empty(1, 1, 1, 1, 3), mode='bicubic')

        if TEST_CUDA:
            with self.assertRaisesRegex(RuntimeError, "Expected all tensors to be on the same device"):
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
        # Backward pass of native C++ and CUDA kernels branch depending on whether input requires gradient,
        # so we test both cases.
        def test(N, C, H, W, mode, padding_mode, align_corners, input_requires_grad):
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

                    input_cpu = torch.randn(C, N, IH, IW).transpose(0, 1).requires_grad_(input_requires_grad)
                    grid_cpu = get_grid().requires_grad_()
                    out_cpu = F.grid_sample(input_cpu, grid_cpu, mode=mode, padding_mode=padding_mode,
                                            align_corners=align_corners)
                    self.assertTrue(out_cpu.size() == torch.Size([N, C, H, W]))

                    gradients = torch.randn_like(out_cpu)
                    out_cpu.backward(gradients)


                    # Compare against unvectorized CPU fallback

                    # NOTE [ grid_sample CPU fallback ]
                    # grid_sample uses AVX for 2d images, but that requires 32-bit indexing for
                    # 32-bit floats. So we also have a fallback that is used only for float tensors
                    # requiring 64-bit indexing. That requires too much memory to run on CI, so we
                    # also export the fallback and test it here to ensure feature parity with
                    # the vectorized version.
                    input_fallback = input_cpu.float().detach_().requires_grad_()
                    grid_fallback = grid_cpu.float().detach_().requires_grad_()
                    out_fallback = torch._grid_sampler_2d_cpu_fallback(
                        input_fallback, grid_fallback,
                        F.GRID_SAMPLE_INTERPOLATION_MODES[mode],
                        F.GRID_SAMPLE_PADDING_MODES[padding_mode],
                        align_corners)
                    self.assertEqual(out_fallback, out_cpu.float(), atol=1e-5, rtol=5e-5)

                    out_fallback.backward(gradients.float())
                    if input_requires_grad:
                        self.assertEqual(input_fallback.grad, input_cpu.grad.float(), atol=1e-4, rtol=5e-5)
                    self.assertEqual(grid_fallback.grad, grid_cpu.grad.float(), atol=1e-4, rtol=5e-5)

                    if TEST_CUDA:
                        input_cuda = input_cpu.detach().transpose(0, 1).cuda().transpose(0, 1).requires_grad_(input_requires_grad)
                        grid_cuda = get_grid('cuda', grid_cpu.detach()).requires_grad_()
                        out_cuda = F.grid_sample(input_cuda, grid_cuda, mode=mode, padding_mode=padding_mode,
                                                 align_corners=align_corners)
                        self.assertEqual(out_cpu, out_cuda)

                        out_cuda.backward(gradients.cuda())
                        if input_requires_grad:
                            self.assertEqual(input_cpu.grad, input_cuda.grad)
                        self.assertEqual(grid_cpu.grad, grid_cuda.grad, atol=5e-5, rtol=0)

                        # check that zero-dimensional input strides don't error out
                        base_input = torch.randn(N, C, 1, IW)
                        input_cpu = base_input.expand_as(input_cuda).requires_grad_(input_requires_grad)
                        out_cpu = F.grid_sample(input_cpu, grid_cpu, mode=mode, padding_mode=padding_mode,
                                                align_corners=align_corners)

                        input_cuda = base_input.cuda().expand_as(input_cuda).requires_grad_(input_requires_grad)
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

        for mode in ('bilinear', 'nearest', 'bicubic'):
            for padding_mode in ('zeros', 'border', 'reflection'):
                for align_corners in (True, False):
                    # test known input on CPU
                    input = torch.arange(1., 11).view(1, 1, 2, 5)
                    grid = torch.tensor(
                        [[[-0.9, -4.1], [0, 0.2000], [1, -1], [-0.333, 1e-6], [0.5, 1.0]],
                         [[-1.0, -0.5], [0, 0.3333], [1, -1], [-0.200, 1e-6], [1.5, 0.5]]]).view(1, 2, 5, 2)
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
                    elif mode == 'bicubic':
                        if padding_mode == 'zeros':
                            if align_corners:
                                groundtruth = torch.tensor(
                                    [[-0.10424726, 7.1400003, 5.0000, 5.7842274, 9.0000],
                                     [2.4492188, 7.4814040, 5.0000, 6.0277520, 0.0000]]).view(1, 1, 2, 5)
                            else:
                                groundtruth = torch.tensor(
                                    [[0.00000, 7.6287503, 1.0625, 5.5977230, 5.3270264],
                                     [0.40625, 8.0288770, 1.0625, 5.9375067, -0.3515625]]).view(1, 1, 2, 5)
                        elif padding_mode == 'border':
                            if align_corners:
                                groundtruth = torch.tensor(
                                    [[1.1520010, 6.0599990, 5.0000, 4.870930, 9.0000000],
                                     [2.1328125, 6.4258375, 5.0000, 5.076003, 8.8671875]]).view(1, 1, 2, 5)
                            else:
                                groundtruth = torch.tensor(
                                    [[0.894531, 6.6050020, 4.625, 4.7138715, 9.800781],
                                     [0.906250, 7.2822485, 4.625, 5.0000052, 10.00000]]).view(1, 1, 2, 5)
                        elif padding_mode == 'reflection':
                            if align_corners:
                                groundtruth = torch.tensor(
                                    [[3.1822524, 6.239998, 5.0000, 4.8709273, 9.00000],
                                     [1.7812500, 6.703594, 5.0000, 5.0760007, 8.21875]]).view(1, 1, 2, 5)
                            else:
                                groundtruth = torch.tensor(
                                    [[2.7993753, 6.6050020, 4.25, 4.7138715, 10.269531],
                                     [0.8125000, 7.2822485, 4.25, 5.0000052, 9.332031]]).view(1, 1, 2, 5)
                        else:
                            raise AssertionError("missing groundtruth test for padding mode '{}'".format(padding_mode))

                    else:
                        raise AssertionError("missing groundtruth test for interpolation mode '{}'".format(mode))
                    output = F.grid_sample(input, grid, mode=mode, padding_mode=padding_mode,
                                           align_corners=align_corners)
                    self.assertEqual(output, groundtruth, atol=1e-5, rtol=0,
                                     msg="groundtruth comparison failed for mode={}, "
                                     "padding_mode={}".format(mode, padding_mode))

                    # See NOTE [ grid_sample CPU fallback ]
                    output = torch._grid_sampler_2d_cpu_fallback(
                        input.float(), grid.float(),
                        F.GRID_SAMPLE_INTERPOLATION_MODES[mode],
                        F.GRID_SAMPLE_PADDING_MODES[padding_mode],
                        align_corners)
                    self.assertEqual(output, groundtruth.float(), atol=1e-5, rtol=0)

                    # explicit check for gradient edge cases
                    input = torch.arange(0., 5).expand((1, 1, 5, 5))
                    grid = torch.tensor(
                        [[[1.0, 1.0], [1.0, -1.0], [0.8, 0.8], [0.8, -0.8]],
                         [[-1.0, -1.0], [-1.0, 1.0], [-0.8, -0.8], [-0.8, 0.8]]]).view(1, 2, 4, 2).requires_grad_()
                    if mode == 'bilinear':
                        if padding_mode == 'zeros':
                            if align_corners:
                                groundtruth = torch.tensor(
                                    [[[[-8., -8.], [-8., 0.], [2., 0.], [2., 0.]],
                                      [[2., 0.], [2., 0.], [2., 0.], [2., 0.]]]]).view(1, 2, 4, 2)
                            else:
                                groundtruth = torch.tensor(
                                    [[[[-5., -5.], [-5., 5.], [-10., -10.], [-10., 10.]],
                                      [[0., 0.], [0., 0.], [0., 0.], [0., 0.]]]]).view(1, 2, 4, 2)
                        elif padding_mode == 'border':
                            if align_corners:
                                groundtruth = torch.tensor(
                                    [[[[-0., -0.], [-0., 0.], [2., 0.], [2., 0.]],
                                      [[0., 0.], [0., 0.], [2., 0.], [2., 0.]]]]).view(1, 2, 4, 2)
                            else:
                                groundtruth = torch.tensor(
                                    [[[[-0., -0.], [-0., 0.], [-0., -0.], [-0., 0.]],
                                      [[0., 0.], [0., 0.], [0., 0.], [0., 0.]]]]).view(1, 2, 4, 2)
                        elif padding_mode == 'reflection':
                            if align_corners:
                                groundtruth = torch.tensor(
                                    [[[[-0., -0.], [-0., 0.], [2., 0.], [2., 0.]],
                                      [[0., 0.], [0., 0.], [2., 0.], [2., 0.]]]]).view(1, 2, 4, 2)
                            else:
                                groundtruth = torch.tensor(
                                    [[[[-0., -0.], [-0., 0.], [-0., -0.], [-0., 0.]],
                                      [[0., 0.], [0., 0.], [0., 0.], [0., 0.]]]]).view(1, 2, 4, 2)
                        else:
                            raise AssertionError("missing gradient groundtruth test for padding mode '{}'".format(padding_mode))
                    elif mode == 'nearest':
                        groundtruth = torch.tensor(
                            [[[[-0., -0.], [-0., 0.], [-0., -0.], [-0., 0.]],
                              [[0., 0.], [0., 0.], [0., 0.], [0., 0.]]]]).view(1, 2, 4, 2)
                    elif mode == 'bicubic':
                        if padding_mode == 'zeros':
                            if align_corners:
                                groundtruth = torch.tensor(
                                    [[[[-4.5, -6.], [-4.5, 6.], [2.725679, 0.740878], [2.725679, -0.740878]],
                                      [[1.5, 0.], [1.5, 0.], [1.927921, -0.05688], [1.927921, 0.05688]]]]).view(1, 2, 4, 2)
                            else:
                                groundtruth = torch.tensor(
                                    [[[[-5.859375, -5.888672], [-5.859375, 5.888672], [-5.6250, -7.5000], [-5.6250, 7.5000]],
                                      [[-0.234375, -0.263672], [-0.234375, 0.263672], [1.8750, 0.], [1.8750, 0.]]]]
                                ).view(1, 2, 4, 2)
                        elif padding_mode == 'border':
                            if align_corners:
                                groundtruth = torch.tensor(
                                    [[[[1.5, 0.], [1.5, 0.], [1.74, 0.], [1.74, 0.]],
                                      [[1.5, 0.], [1.5, 0.], [1.74, 0.], [1.74, 0.]]]]).view(1, 2, 4, 2)
                            else:
                                groundtruth = torch.tensor(
                                    [[[[-0.46875, 0.], [-0.46875, 0.], [1.8750, 0.], [1.8750, 0.]],
                                      [[-0.46875, 0.], [-0.46875, 0.], [1.8750, 0.], [1.8750, 0.]]]]).view(1, 2, 4, 2)
                        elif padding_mode == 'reflection':
                            if align_corners:
                                groundtruth = torch.tensor(
                                    [[[[0., 0.], [0., 0.], [1.92, 0.], [1.92, 0.]],
                                      [[0., 0.], [0., 0.], [1.92, 0.], [1.92, 0.]]]]).view(1, 2, 4, 2)
                            else:
                                groundtruth = torch.tensor(
                                    [[[[0., 0.], [0., 0.], [1.875, 0.], [1.875, 0.]],
                                      [[0., 0.], [0., 0.], [1.875, 0.], [1.875, 0.]]]]).view(1, 2, 4, 2)
                        else:
                            raise AssertionError("missing gradient groundtruth test for padding mode '{}'".format(padding_mode))
                    else:
                        raise AssertionError("missing gradient groundtruth test for interpolation mode '{}'".format(mode))
                    for input_requires_grad in [False, True]:
                        input = input.requires_grad_(input_requires_grad)
                        F.grid_sample(input, grid, mode=mode, padding_mode=padding_mode,
                                      align_corners=align_corners).sum().backward()
                        self.assertEqual(grid.grad, groundtruth, atol=1e-5, rtol=0,
                                         msg="gradient groundtruth comparison failed for mode={}, "
                                         "padding_mode={}, input_requires_grad={}".format(mode, padding_mode, input_requires_grad))
                        grid.grad.zero_()

                    # See NOTE [ grid_sample CPU fallback ]
                    torch._grid_sampler_2d_cpu_fallback(
                        input.float(), grid.float(),
                        F.GRID_SAMPLE_INTERPOLATION_MODES[mode],
                        F.GRID_SAMPLE_PADDING_MODES[padding_mode],
                        align_corners).sum().backward()
                    self.assertEqual(grid.grad, groundtruth, atol=1e-5, rtol=0)

                    # do gradcheck
                    N = random.randint(2, 8)
                    C = random.randint(2, 6)
                    H = random.randint(2, 8)
                    W = random.randint(2, 8)
                    input = torch.randn(N, C, H, W, requires_grad=True)
                    grid = torch.randn(N, H, W, 2, requires_grad=True)

                    for input_requires_grad in [False, True]:
                        input.requires_grad_(input_requires_grad)
                        self.assertTrue(gradcheck(
                            lambda inp, grd: F.grid_sample(inp, grd, mode=mode, padding_mode=padding_mode,
                                                           align_corners=align_corners),
                            (input, grid)))
                        test(N, C, H, W, mode, padding_mode, align_corners, input_requires_grad)
                        if TEST_CUDNN:
                            with cudnn.flags(enabled=False):
                                test(N, C, H, W, mode, padding_mode, align_corners, input_requires_grad)

    def test_grid_sample_3d(self):
        # Backward pass of native C++ and CUDA kernels branch depending on whether input requires gradient,
        # so we test both cases.
        def test(N, C, D, H, W, mode, padding_mode, align_corners, input_requires_grad):
            def test_shape(N, C, ID, IH, IW, D, H, W, mode, padding_mode, align_corners):
                input_cpu = torch.randn(C, N, ID, IH, IW).transpose(0, 1).requires_grad_(input_requires_grad)
                grid_cpu = torch.randn(D, N, H, W, 3).transpose(0, 1).requires_grad_()
                out_cpu = F.grid_sample(input_cpu, grid_cpu, mode=mode, padding_mode=padding_mode,
                                        align_corners=align_corners)
                self.assertTrue(out_cpu.size() == torch.Size([N, C, D, H, W]))

                gradients = torch.randn_like(out_cpu)
                out_cpu.backward(gradients)

                if TEST_CUDA:
                    input_cuda = input_cpu.detach().transpose(0, 1).cuda().transpose(0, 1).requires_grad_(input_requires_grad)
                    grid_cuda = grid_cpu.detach().transpose(0, 1).cuda().transpose(0, 1).requires_grad_()
                    out_cuda = F.grid_sample(input_cuda, grid_cuda, mode=mode, padding_mode=padding_mode,
                                             align_corners=align_corners)
                    self.assertEqual(out_cpu, out_cuda)

                    out_cuda.backward(gradients.cuda())
                    if input_requires_grad:
                        self.assertEqual(input_cpu.grad, input_cuda.grad)
                    self.assertEqual(grid_cpu.grad, grid_cuda.grad, atol=5e-5, rtol=0)

                    # check that zero-dimensional input strides don't error out
                    base_input = torch.randn(N, C, 1, IH, IW)
                    input_cpu = base_input.expand_as(input_cuda).requires_grad_(input_requires_grad)
                    grid_cpu = torch.randn(N, D, H, W, 3, requires_grad=True)
                    out_cpu = F.grid_sample(input_cpu, grid_cpu, mode=mode, padding_mode=padding_mode,
                                            align_corners=align_corners)

                    input_cuda = base_input.cuda().expand_as(input_cuda).requires_grad_(input_requires_grad)
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
                    input = input.requires_grad_(False)
                    self.assertTrue(gradcheck(
                        lambda grid: F.grid_sample(input, grid, mode=mode, padding_mode=padding_mode,
                                                   align_corners=align_corners),
                        (grid,)))

                    for input_requires_grad in [False, True]:
                        test(N, C, D, H, W, mode, padding_mode, align_corners, input_requires_grad)

    def test_affine_grid(self):
        # test known input on CPU
        input = torch.arange(1., 7).view(1, 2, 3)
        output = F.affine_grid(input, torch.Size([1, 1, 2, 2]), align_corners=True)
        groundtruth = torch.tensor(
            [[[0., -3.], [2., 5.]], [[4., 7.], [6., 15.]]]).view(1, 2, 2, 2)
        self.assertEqual(output, groundtruth)
        output = F.affine_grid(input, torch.Size([1, 1, 2, 2]), align_corners=False)
        groundtruth = torch.tensor(
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
        groundtruth = torch.tensor(
            [[[[[-2., -10., -18.], [0., 0., 0.]], [[2., 2., 2.], [4., 12., 20.]]],
              [[[4., 4., 4.], [6., 14., 22.]], [[8., 16., 24.], [10., 26., 42.]]]]]).view(1, 2, 2, 2, 3)
        self.assertEqual(output, groundtruth)
        output = F.affine_grid(input, torch.Size([1, 1, 2, 2, 2]), align_corners=False)
        groundtruth = torch.tensor(
            [[[[[1., -1., -3.], [2., 4., 6.]], [[3., 5., 7.], [4., 10., 16.]]],
              [[[4., 6., 8.], [5., 11., 17.]], [[6., 12., 18.], [7., 17., 27.]]]]]).view(1, 2, 2, 2, 3)
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

    def test_channel_shuffle(self):
        #  3D tensor
        x = torch.tensor(
            [[[1, 2],
              [5, 6],
              [9, 10],
              [13, 14],
              ]]
        )
        y_ref = torch.tensor(
            [[[1, 2],
              [9, 10],
              [5, 6],
              [13, 14],
              ]]
        )
        #  ChannelsFirst
        with warnings.catch_warnings(record=True) as w:
            y = F.channel_shuffle(x, 2)
            self.assertEqual(len(w), 0)
        self.assertEqual(y, y_ref)
        #  ChannelsLast not supported for 3dim

        #  4D tensor
        x = torch.tensor(
            [[[[1, 2],
               [3, 4]],
              [[5, 6],
               [7, 8]],
              [[9, 10],
               [11, 12]],
              [[13, 14],
               [15, 16]],
              ]]
        )
        y_ref = torch.tensor(
            [[[[1, 2],
               [3, 4]],
              [[9, 10],
               [11, 12]],
              [[5, 6],
               [7, 8]],
              [[13, 14],
               [15, 16]],
              ]]
        )
        #  ChannelsFirst NCHW
        with warnings.catch_warnings(record=True) as w:
            y = F.channel_shuffle(x, 2)
            self.assertEqual(len(w), 0)
        self.assertEqual(y, y_ref)
        #  ChannelsLast NHWC
        with warnings.catch_warnings(record=True) as w:
            y = F.channel_shuffle(x.contiguous(memory_format=torch.channels_last), 2)
            self.assertEqual(len(w), 0)
        y = y.contiguous(memory_format=torch.contiguous_format)
        self.assertEqual(y, y_ref)

        #  5D tensor
        x = torch.tensor(
            [[[[[1, 2],
               [3, 4]]],
              [[[5, 6],
               [7, 8]]],
              [[[9, 10],
               [11, 12]]],
              [[[13, 14],
               [15, 16]]],
              ]]
        )
        y_ref = torch.tensor(
            [[[[[1, 2],
               [3, 4]]],
              [[[9, 10],
               [11, 12]]],
              [[[5, 6],
               [7, 8]]],
              [[[13, 14],
               [15, 16]]],
              ]]
        )
        #  ChannelsFirst NCHW
        with warnings.catch_warnings(record=True) as w:
            y = F.channel_shuffle(x, 2)
            self.assertEqual(len(w), 0)
        self.assertEqual(y, y_ref)
        #  ChannelsLast NHWC
        with warnings.catch_warnings(record=True) as w:
            y = F.channel_shuffle(x.contiguous(memory_format=torch.channels_last_3d), 2)
            self.assertEqual(len(w), 0)
        y = y.contiguous(memory_format=torch.contiguous_format)
        self.assertEqual(y, y_ref)


    def test_channel_shuffle_return_self(self):
        # gh-76616: nn.ChannelShuffle will return self with an  empty input tensor
        groups = 3
        input_tensor = torch.rand([0, 9, 4, 4])
        output = torch.nn.ChannelShuffle(groups)(input_tensor)
        torch.testing.assert_close(output, input_tensor)

    def test_upsamplingLinear1d(self):
        for align_corners in [True, False]:
            for recompute_scale_factor in [True, False]:
                kwargs = dict(
                    mode='linear', align_corners=align_corners, recompute_scale_factor=recompute_scale_factor
                )
                # test float scale factor up & downsampling
                for scale_factor in [0.5, 1.5, 2]:
                    m = nn.Upsample(scale_factor=scale_factor, **kwargs)
                    in_t = torch.ones(1, 1, 2)
                    out_size = int(math.floor(in_t.shape[-1] * scale_factor))
                    with warnings.catch_warnings(record=True) as w:
                        out_t = m(in_t)
                    self.assertEqual(torch.ones(1, 1, out_size), out_t.data)

                    input = torch.randn(1, 1, 2, requires_grad=True)
                    if not recompute_scale_factor:
                        gradcheck(lambda x: F.interpolate(x, out_size, **kwargs), (input,))
                    else:
                        gradcheck(lambda x: F.interpolate(x, scale_factor=scale_factor, **kwargs), (input,))

    def test_upsamplingLinear1d_spatial_invariance(self):
        m = nn.Upsample(scale_factor=3, mode='linear', align_corners=False)
        in_t_9 = torch.zeros(1, 1, 9)
        in_t_9[:, :, :4].normal_()
        with warnings.catch_warnings(record=True) as w:
            out_t_9 = m(in_t_9)
            out_t_5 = m(in_t_9[:, :, :5])
        self.assertEqual(out_t_9[:, :, :15], out_t_5)

    def test_upsampling_not_recompute_scale_factor(self):
        # test output against known input: result must match opencv
        in_t = torch.arange(8.).view(1, 2, 2, 2)
        expected_out_t = torch.tensor(
            [[[[-0.32725, -0.08843, 0.37933, 0.79744],
              [0.15039, 0.38921, 0.85697, 1.27508],
              [1.08591, 1.32473, 1.79249, 2.21060],
              [1.92213, 2.16095, 2.62871, 3.04682]],

             [[3.67275, 3.91157, 4.37933, 4.79744],
              [4.15039, 4.38921, 4.85697, 5.27508],
              [5.08591, 5.32473, 5.79249, 6.21060],
              [5.92213, 6.16095, 6.62871, 7.04682]]]])
        if IS_PPC:
            # Both OpenCV and PyTorch give a slightly different result on PPC
            expected_out_t = torch.tensor(
                [[[[-0.32725, -0.08843, 0.37933, 0.79744],
                  [0.15039, 0.38921, 0.85697, 1.27508],
                  [1.08591, 1.32473, 1.79249, 2.21060],
                  [1.92212, 2.16094, 2.62870, 3.04681]],

                 [[3.67275, 3.91157, 4.37933, 4.79743],
                  [4.15039, 4.38921, 4.85697, 5.27508],
                  [5.08591, 5.32473, 5.79249, 6.21059],
                  [5.92212, 6.16094, 6.62870, 7.04680]]]])
        out_t = F.interpolate(in_t, scale_factor=2.3, mode='bicubic', align_corners=False, recompute_scale_factor=False)
        torch.set_printoptions(precision=5)
        self.assertEqual(out_t, expected_out_t, atol=1e-4, rtol=0)

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
                    self.assertEqual(torch.ones(2, 2, out_size, out_size), out_t.data, atol=1e-5, rtol=0)

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

    def test_upsamplingTrilinear3d(self):
        for align_corners in [True, False]:
            kwargs = dict(mode='trilinear', align_corners=align_corners)

            for memory_format in [torch.contiguous_format, torch.channels_last_3d]:
                # test float scale factor up & downsampling
                for scale_factor in [0.5, 1.5, 2]:
                    m = nn.Upsample(scale_factor=scale_factor, **kwargs)
                    in_t = torch.ones(1, 2, 2, 2, 2).contiguous(memory_format=memory_format)
                    out_size = int(math.floor(in_t.shape[-1] * scale_factor))
                    with warnings.catch_warnings(record=True) as w:
                        out_t = m(in_t)
                    self.assertEqual(torch.ones(1, 2, out_size, out_size, out_size), out_t.data)
                    # Assert that memory format is carried through to the output
                    self.assertTrue(out_t.is_contiguous(memory_format=memory_format))

                    input = torch.randn(1, 2, 2, 2, 2, requires_grad=True)
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

    def test_upsampling_small_scale(self):
        m = torch.nn.Upsample(scale_factor=0.5, mode="bilinear")
        in_t = torch.arange(1, 5, dtype=torch.float64).reshape(1, 1, 2, 2)
        out_t = m(in_t)
        expected_out_t = torch.tensor([[[[2.5]]]])
        self.assertEqual(expected_out_t, out_t)

    def test_upsampling_bfloat16(self, dtype=torch.bfloat16):
        def helper(size, scale_factor, mode, device, memory_format=torch.contiguous_format):
            inputf = torch.randn(size, device=device, dtype=torch.float, requires_grad=True)
            input = inputf.to(dtype).to(memory_format=memory_format).detach().requires_grad_(True)
            m = nn.Upsample(scale_factor=scale_factor, mode=mode)

            outf = m(inputf)
            out = m(input)
            self.assertEqual(out.dtype, dtype)
            self.assertEqualIgnoreType(out, outf, atol=0.1, rtol=0.0)

            out.sum().backward()
            outf.sum().backward()
            self.assertEqual(input.grad.dtype, dtype)
            self.assertEqual(input.grad, inputf.grad.to(dtype), atol=0.1, rtol=0)

        for device in ['cpu']:
            helper([3, 20, 30], 2, 'nearest', device)
            helper([3, 20, 11, 7], 2, 'nearest', device)
            helper([3, 20, 11, 7], 2, 'nearest', device, torch.channels_last)
            helper([3, 20, 11, 7, 3], 2, 'nearest', device)
            helper([3, 20, 30], 2, 'linear', device)
            helper([3, 20, 11, 7], 2, 'bilinear', device)
            helper([3, 20, 11, 7], 2, 'bilinear', device, torch.channels_last)
            helper([3, 20, 11, 7, 3], 2, 'trilinear', device)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_interpolate_illegal_memory_access(self):
        in_s = 45
        out_s = 14

        input = torch.ones((1, 1, in_s), device='cuda', requires_grad=True)
        # note we allocated grad_output to be larger so out of bound access
        # woudl be visible in grad_input
        grad = torch.ones((1, 1, out_s * 2), device='cuda', requires_grad=True)
        grad = grad[:, :, :out_s]

        input_ref = input.detach().cpu().requires_grad_()
        grad_ref = grad.cpu()

        out = F.interpolate(input, size=(out_s,), mode='nearest')
        out.backward(grad)

        out_ref = F.interpolate(input_ref, size=(out_s,), mode='nearest')
        out_ref.backward(grad_ref)

        self.assertEqual(out_ref, out)
        self.assertEqual(input_ref.grad, input.grad)

    def test_interpolate(self):
        def _test_interpolate_helper(in_t, scale_factor, layer):
            out_size = int(math.floor(in_t.shape[-1] * scale_factor))
            dim = len(in_t.shape) - 2
            out_shape = [1, 1] + [out_size] * dim
            with warnings.catch_warnings(record=True) as w:
                out_t = layer(in_t)
            self.assertEqual(torch.ones(out_shape), out_t)

            self.assertEqual(
                F.interpolate(in_t, (out_size,) * dim, **kwargs),
                F.interpolate(in_t, scale_factor=scale_factor, **kwargs))
            gradcheck(lambda x: F.interpolate(x, out_size, **kwargs), [in_t], nondet_tol=GRADCHECK_NONDET_TOL)
            gradgradcheck(lambda x: F.interpolate(x, out_size, **kwargs), [in_t], nondet_tol=GRADCHECK_NONDET_TOL)

        def _make_input(dim, device):
            size = [1, 1]
            size += [2] * dim
            return torch.ones(size, requires_grad=True, device=device)

        device_list = ['cpu']
        if TEST_CUDA:
            device_list.append('cuda')

        for device in device_list:
            for scale_factor in [0.5, 1.5, 2]:
                for mode in ['nearest', 'area']:
                    kwargs = dict(mode=mode)
                    m = nn.Upsample(scale_factor=scale_factor, **kwargs).to(device)
                    for input in [_make_input(1, device), _make_input(2, device), _make_input(3, device)]:
                        _test_interpolate_helper(input, scale_factor, m)

                for align_corners in [True, False]:
                    kwargs = dict(mode='linear', align_corners=align_corners)
                    m = nn.Upsample(scale_factor=scale_factor, **kwargs).to(device)
                    _test_interpolate_helper(_make_input(1, device), scale_factor, m)

                    kwargs = dict(mode='bilinear', align_corners=align_corners)
                    m = nn.Upsample(scale_factor=scale_factor, **kwargs).to(device)
                    _test_interpolate_helper(_make_input(2, device), scale_factor, m)

                    kwargs = dict(mode='bicubic', align_corners=align_corners)

                    def m(t):
                        return F.interpolate(t, scale_factor=scale_factor, **kwargs).to(device)
                    _test_interpolate_helper(_make_input(2, device), scale_factor, m)

                    kwargs = dict(mode='trilinear', align_corners=align_corners)
                    m = nn.Upsample(scale_factor=scale_factor, **kwargs).to(device)
                    _test_interpolate_helper(_make_input(3, device), scale_factor, m)

    def test_linear_broadcasting(self):
        m = nn.Linear(5, 8)
        inp = torch.randn(2, 3, 5)
        expected = m(inp.view(6, 5)).view(2, 3, 8)
        self.assertEqual(expected, m(inp))

    @parametrize_test('device', ['cpu'] + (['cuda'] if TEST_CUDA else []))
    @parametrize_test('bias', [
        subtest(False, name='nobias'), subtest(True, name='bias')])
    @parametrize_test('weight_layout', [
        subtest(torch.strided, name='weightStrided'),
        subtest(torch.sparse_coo, name='weightCOO'),
        subtest(torch.sparse_csr, name='weightCSR'),
        subtest(torch.sparse_csc, name='weightCSC'),
        # TODO: addmm: computation on CPU is not implemented for Strided + Strided @ SparseBsr
        # subtest(torch.sparse_bsr, name='weightBSR'),
        # subtest(torch.sparse_bsc, name='weightBSC'),
    ])
    def test_linear_autograd(self, device, bias, weight_layout):
        module = nn.Linear(4, 4, bias=bias, device=device)
        if weight_layout == torch.strided:
            pass
        elif weight_layout == torch.sparse_csr:
            module.weight = nn.Parameter(module.weight.to_sparse_csr())
        elif weight_layout == torch.sparse_csc:
            module.weight = nn.Parameter(module.weight.to_sparse_csc())
        elif weight_layout == torch.sparse_bsr:
            module.weight = nn.Parameter(module.weight.to_sparse_bsr(2, 2))
        elif weight_layout == torch.sparse_bsc:
            module.weight = nn.Parameter(module.weight.to_sparse_bsc(2, 2))
        elif weight_layout == torch.sparse_coo:
            module.weight = nn.Parameter(module.weight.to_sparse_coo())
        else:
            assert(0)

        inp = torch.randn(4, requires_grad=True, device=device)
        res = module(inp)
        if bias:
            expected = (torch.einsum("i,ji->j", inp, module.weight.to_dense())) + module.bias
        else:
            expected = (torch.einsum("i,ji->j", inp, module.weight.to_dense()))
        self.assertEqual(res, expected)

        grad_output = torch.randn(4, device=device)
        grads = torch.autograd.grad(res, [module.weight, inp], grad_output)
        grads_expected = torch.autograd.grad(expected, [module.weight, inp], grad_output)

        self.assertEqual(grads_expected[0].layout, weight_layout)

        for g, ge in zip(grads, grads_expected):
            self.assertEqual(g, ge)

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

    def test_bilinear_non_contiguous(self):
        module = nn.Bilinear(7, 7, 5)
        input1 = torch.randn(4, 7, 10, requires_grad=True)
        input2 = torch.randn(4, 7, 10, requires_grad=True)
        input1_tp = input1.transpose(1, 2)
        input2_tp = input2.transpose(1, 2)

        grad_output = torch.randn(4, 10, 5)

        def run(input1_tp, input2_tp):
            input1.grad = input2.grad = None
            output = module(input1_tp, input2_tp)
            output.backward(grad_output)

            return output.data, input1.grad.data, input2.grad.data

        out_nc, g1_nc, g2_nc = run(input1_tp, input2_tp)
        input1_tp = input1_tp.contiguous()
        input2_tp = input2_tp.contiguous()
        out, g1, g2 = run(input1_tp, input2_tp)

        self.assertEqual(out, out_nc)
        self.assertEqual(g1, g1_nc)
        self.assertEqual(g2, g2_nc)

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

    def test_fold_invalid_arg(self):
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

        fold = nn.Fold(output_size=(4, 5), kernel_size=(2, 2), stride=1, dilation=8, padding=0)
        with self.assertRaisesRegex(RuntimeError, r"calculated shape of the array of sliding blocks as"):
            fold(torch.randn(1, 12, 12))

    def test_unfold_invalid_arg(self):
        # input wrong dimension

        unfold = nn.Unfold(kernel_size=(2, 3))

        # calculated output shape is too small
        with self.assertRaisesRegex(RuntimeError, r"its components must be at least one"):
            unfold = nn.Unfold(kernel_size=(2, 3))
            unfold(torch.randn(1, 2, 2, 2))

        with self.assertRaisesRegex(RuntimeError, r"its components must be at least one"):
            unfold = nn.Unfold(kernel_size=(5, 3), padding=(1, 1))
            unfold(torch.randn(1, 2, 2, 3))

        with self.assertRaisesRegex(RuntimeError, r"its components must be at least one"):
            unfold = nn.Unfold(kernel_size=(1, 3), padding=(1, 1), dilation=(1, 2))
            unfold(torch.randn(1, 2, 2, 2))

    def test_softmin(self):
        x = torch.randn(2, 16)
        self.assertEqual(F.softmin(x, 1), F.softmax(-x, 1))
        self.assertEqual(F.softmin(x, 0), F.softmax(-x, 0))

    def test_log_softmax_cpu(self, dtype=torch.bfloat16):
        for dim in [0, 1]:
            inputf = torch.rand(200, 200, device="cpu", dtype=torch.float, requires_grad=True)
            input = inputf.to(dtype).detach().requires_grad_(True)
            outf = F.log_softmax(inputf, dim=dim)
            out = F.log_softmax(input, dim=dim)
            self.assertEqual(out.dtype, dtype)
            self.assertEqual(out, outf.to(dtype=dtype), atol=0.1, rtol=0)

            out.sum().backward()
            outf.sum().backward()
            self.assertEqual(input.grad.dtype, dtype)
            self.assertEqual(input.grad, inputf.grad.to(dtype), atol=0.1, rtol=0)

    def test_softmax_cpu(self, dtype=torch.bfloat16):
        for dim in [0, 1]:
            inputf = torch.rand(200, 200, device="cpu", dtype=torch.float, requires_grad=True)
            input = inputf.to(dtype).detach().requires_grad_(True)
            outf = F.softmax(inputf, dim=dim)
            out = F.softmax(input, dim=dim)
            self.assertEqual(out.dtype, dtype)
            self.assertEqualIgnoreType(out, outf, atol=1e-3, rtol=0)

            out.sum().backward()
            outf.sum().backward()
            self.assertEqual(input.grad.dtype, dtype)
            self.assertEqual(input.grad, inputf.grad.to(dtype), atol=1e-3, rtol=0)

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

        # test no_batch_dim support
        asfm = nn.AdaptiveLogSoftmaxWithLoss(16, 20, [5, 10, 15], div_value=2.)
        x = torch.randn(1, 16)
        y = torch.tensor([17])
        x2 = x.squeeze(0)
        y2 = y.squeeze(0)
        self.assertEqual(asfm(x, y).output.squeeze(0), asfm(x2, y2).output)

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
        self.assertEqual(out, outf.to(dtype=dtype), atol=1e-1, rtol=0)

        outf.backward()
        out.backward()
        self.assertEqual(input.grad.dtype, dtype)
        self.assertEqual(input.grad, inputf.grad.to(dtype=dtype), atol=1e-1, rtol=0)

    def test_cross_entropy_loss_precision(self):
        # Regression test for #55657
        loss_cpu = nn.CrossEntropyLoss().cpu()
        inputf = torch.randn(128, 2, 768, 768, device="cpu", dtype=torch.float)
        inputd = inputf.double()
        target = torch.randint(2, (128, 768, 768), dtype=torch.long)

        outf = loss_cpu(inputf, target)
        outd = loss_cpu(inputd, target)
        self.assertEqual(outf, outd, exact_dtype=False)

    def test_cross_entropy_loss_zero_div(self):
        # Test for issue #73165
        input_1 = torch.rand([5, 0], dtype=torch.float32)
        input_2 = torch.rand([5, 0], dtype=torch.float32)
        torch.nn.CrossEntropyLoss()(input_1, input_2)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_convert_sync_batchnorm(self):
        module = torch.nn.Sequential(
            torch.nn.BatchNorm1d(100),
            torch.nn.InstanceNorm1d(100)
        ).cuda()

        # necessary to have an anchor point for comparison, in case the
        # convert_sync_batchnorm updates in place
        comp_module = torch.nn.Sequential(
            torch.nn.BatchNorm1d(100),
            torch.nn.InstanceNorm1d(100)
        ).cuda()
        comp_module.load_state_dict(module.state_dict())

        sync_bn_module = torch.nn.SyncBatchNorm.convert_sync_batchnorm(module)
        children = list(sync_bn_module.children())
        self.assertEqual(children[0].__class__, torch.nn.SyncBatchNorm)
        self.assertEqual(children[1].__class__, torch.nn.InstanceNorm1d)

        for layer, converted_layer in zip(comp_module.children(), sync_bn_module.children()):
            for key in layer.state_dict().keys():
                self.assertEqual(layer.state_dict()[key].device, converted_layer.state_dict()[key].device)
                self.assertEqual(layer.state_dict()[key], converted_layer.state_dict()[key])

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_sync_batchnorm_backward_elemt(self):
        device = 'cuda'
        saved_input = torch.rand(2, 3, 2, 1, device=device)
        grad_output = torch.rand(2, 3, 2, 1, device=device)
        mean = torch.rand(3, device=device)
        invstd = torch.rand(3, device=device)
        weight = torch.rand(3, device=device)
        sum_dy = torch.rand(3, device=device)
        sum_dy_xmu = torch.rand(3, device=device)
        count_tensor = torch.tensor([5, 5, 5], dtype=torch.int32, device=device)

        gI_contiguous = torch.batch_norm_backward_elemt(
            grad_output,
            saved_input,
            mean,
            invstd,
            weight,
            sum_dy,
            sum_dy_xmu,
            count_tensor
        )

        # Test batch_norm_backward_elemt gives the same answer for all
        # combinations of contiguous as channels_last input
        for a, b in [
                (torch.channels_last, torch.contiguous_format),
                (torch.contiguous_format, torch.channels_last),
                (torch.channels_last, torch.channels_last),
        ]:
            gI_actual = torch.batch_norm_backward_elemt(
                grad_output.contiguous(memory_format=a),
                saved_input.contiguous(memory_format=b),
                mean,
                invstd,
                weight,
                sum_dy,
                sum_dy_xmu,
                count_tensor
            )
            self.assertEqual(gI_actual, gI_contiguous)

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_sync_batchnorm_accuracy_cuda(self):
        # The target of this test is to test the functionality and accuracy of
        #   those single-GPU cuda kernels used in SyncBatchNorm
        # They are:
        #   fwd: torch.batch_norm_stats, torch.batch_norm_gather_stats_with_counts, torch.batch_norm_elemt
        #   bwd: torch.batch_norm_backward_reduce, torch.batch_norm_backward_elemt

        def _batch_norm_stats(data):
            mean1, _ = torch.batch_norm_stats(data, 1e-5)
            mean2, _ = torch.batch_norm_stats(data.to(memory_format=torch.channels_last), 1e-5)
            mean_ref = torch.mean(data, (0, 2, 3), keepdim=False)

            self.assertEqual(mean_ref, mean1)
            self.assertEqual(mean_ref, mean2)

        data = torch.randn(1, 96, 112, 112, dtype=torch.float, device='cuda')
        _batch_norm_stats(data)

    def test_flatten(self):
        tensor_input = torch.randn(2, 1, 2, 3)

        # Flatten Tensor

        flatten = nn.Flatten(start_dim=1, end_dim=-1)
        tensor_output = flatten(tensor_input)
        self.assertEqual(tensor_output.size(), torch.Size([2, 6]))

    def test_unflatten(self):
        tensor_input = torch.randn(2, 50)

        # Unflatten Tensor (unflattened_size as a tuple of ints and list of ints)

        for us in ((2, 5, 5), [2, 5, 5]):
            unflatten = nn.Unflatten(dim=1, unflattened_size=us)
            tensor_output = unflatten(tensor_input)
            self.assertEqual(tensor_output.size(), torch.Size([2, 2, 5, 5]))

        # Unflatten NamedTensor

        unflatten = nn.Unflatten(dim='features', unflattened_size=(('C', 2), ('H', 5), ('W', 5)))
        named_tensor_input = tensor_input.refine_names('N', 'features')
        named_tensor_output = unflatten(named_tensor_input)
        self.assertEqual(named_tensor_output.size(), torch.Size([2, 2, 5, 5]))

    def test_unflatten_invalid_arg(self):
        # Wrong type for unflattened_size (tuple of floats)

        with self.assertRaisesRegex(
                TypeError,
                r"unflattened_size must be tuple of ints, but found element of type float at pos 2"):
            nn.Unflatten(dim=1, unflattened_size=(2, 5, 5.0))

        # Wrong type for unflattened_size (list of lists and list of tuples)
        for us in ([['C', 2], ['W', 5], ['H', 5]], [('C', 2), ('W', 5), ('H', 5)]):
            with self.assertRaisesRegex(
                    TypeError,
                    r"unflattened_size must be a tuple of tuples, but found type list"):
                nn.Unflatten(dim='features', unflattened_size=us)

        # Wrong type for unflattened_size (tuple of lists)

        with self.assertRaisesRegex(
                TypeError,
                r"unflattened_size must be tuple of tuples, but found element of type list at pos 0"):
            nn.Unflatten(dim='features', unflattened_size=(['C', 2], ['W', 5], ['H', 5]))

        # Wrong type for unflattened_size (tuple of dicts)

        with self.assertRaisesRegex(
                TypeError,
                r"unflattened_size must be tuple of tuples, but found element of type dict at pos 0"):
            nn.Unflatten(dim='features', unflattened_size=({'C': 2}, {'W': 5}, {'H': 5}))

    def test_layer_norm_grads_with_create_graph_flag(self):
        atol = 1e-5
        rtol = 1e-3

        x = torch.randn((4, 4, 16), requires_grad=True)
        layer_norm = nn.LayerNorm((16,), 1e-5, True)
        with torch.no_grad():
            layer_norm.weight = torch.nn.Parameter(0.1 * torch.ones_like(layer_norm.weight))

        grads1 = torch.autograd.grad(layer_norm(x).sum(), x, create_graph=False)[0]
        grads2 = torch.autograd.grad(layer_norm(x).sum(), x, create_graph=True)[0]

        self.assertEqual(grads1, grads2, rtol=rtol, atol=atol)

        if TEST_CUDA:
            x = x.to('cuda')
            layer_norm = layer_norm.to('cuda')

            grads1 = torch.autograd.grad(layer_norm(x).sum(), x, create_graph=False)[0]
            grads2 = torch.autograd.grad(layer_norm(x).sum(), x, create_graph=True)[0]

            self.assertEqual(grads1, grads2, rtol=rtol, atol=atol)

    def test_padding_list(self):
        # Padding can be a list, or tuple (regression test for gh-54452)
        x = torch.randn(4, 8, 32, 32)
        net = torch.nn.ConvTranspose2d(8, 16, kernel_size=3, padding=[3, 3])
        y = net(x)

        net = torch.nn.ConvTranspose2d(8, 16, kernel_size=3, padding=(3, 3))
        y = net(x)


class TestNNInit(TestCase):
    def setUp(self):
        super(TestNNInit, self).setUp()
        random.seed(123)

    def _is_normal(self, tensor, mean, std):
        samples = tensor.view(-1).tolist()
        p_value = stats.kstest(samples, 'norm', args=(mean, std))[1]
        return p_value > 0.0001

    def _is_trunc_normal(self, tensor, mean, std, a, b):
        # scipy's trunc norm is suited for data drawn from N(0, 1),
        # so we need to transform our data to test it using scipy.
        z_samples = (tensor.view(-1) - mean) / std
        z_samples = z_samples.tolist()
        a0 = (a - mean) / std
        b0 = (b - mean) / std
        p_value = stats.kstest(z_samples, 'truncnorm', args=(a0, b0))[1]
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
            elif fn == 'selu':
                self.assertEqual(gain, 0.75)

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

    @unittest.skipIf(not TEST_SCIPY, "Scipy not found.")
    def test_trunc_normal(self):
        for dims in [1, 2, 4]:
            input_tensor = self._create_random_nd_tensor(dims, size_min=30, size_max=50)
            mean = self._random_float(-3, 3)
            std = self._random_float(.01, 1)
            a = self._random_float(mean - 2 * std, mean)
            b = self._random_float(mean, mean + 2 * std)
            init.trunc_normal_(input_tensor, mean=mean, std=std, a=a, b=b)

            assert self._is_trunc_normal(input_tensor, mean, std, a, b)

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

    def test_dirac_properties(self):
        for dims in [3, 4, 5]:
            for groups in [1, 2, 3]:
                # prepare random tensor with random sizes, but fits groups
                a, c, d, e = (random.randint(1, 5) for _ in range(4))
                b = random.randint(1, 5 * groups)  # same range as a*groups but all range allowed
                # make sure first dim divides by groups
                input_tensor = torch.randn((a * groups, b, c, d, e)[:dims])

                init.dirac_(input_tensor, groups)

                c_out, c_in = input_tensor.size(0) // groups, input_tensor.size(1)
                min_d = min(c_out, c_in)
                # Check number of nonzeros is equivalent to smallest dim (for each group)
                assert torch.nonzero(input_tensor).size(0) == min_d * groups
                # Check sum of values (can have precision issues, hence assertEqual) is also equivalent
                self.assertEqual(input_tensor.sum(), min_d * groups)


    def test_dirac_identity(self):
        for groups in [1, 3]:
            batch, in_c, out_c, size, kernel_size = 8, 3, 9, 5, 3  # in_c, out_c must divide by groups
            eff_out_c = out_c // groups

            # Test 1D
            input_var = torch.randn(batch, in_c, size)
            filter_var = torch.zeros(eff_out_c, in_c, kernel_size)
            filter_var = torch.cat([filter_var] * groups)
            init.dirac_(filter_var, groups)
            output_var = F.conv1d(input_var, filter_var)
            input_tensor, output_tensor = input_var.data, output_var.data  # Variables do not support nonzero
            for g in range(groups):
                # Assert in_c outputs are preserved (per each group)
                self.assertEqual(input_tensor[:, :, 1:-1],
                                 output_tensor[:, eff_out_c * g:eff_out_c * g + in_c, :])
                # Assert extra outputs are 0
                assert torch.nonzero(output_tensor[:, eff_out_c * g + in_c:eff_out_c * (g + 1), :]).numel() == 0

            # Test 2D
            input_var = torch.randn(batch, in_c, size, size)
            filter_var = torch.zeros(eff_out_c, in_c, kernel_size, kernel_size)
            filter_var = torch.cat([filter_var] * groups)
            init.dirac_(filter_var, groups)
            output_var = F.conv2d(input_var, filter_var)
            input_tensor, output_tensor = input_var.data, output_var.data  # Variables do not support nonzero
            for g in range(groups):
                # Assert in_c outputs are preserved (per each group)
                self.assertEqual(input_tensor[:, :, 1:-1, 1:-1],
                                 output_tensor[:, eff_out_c * g:eff_out_c * g + in_c, :, :])
                # Assert extra outputs are 0
                assert torch.nonzero(output_tensor[:, eff_out_c * g + in_c:eff_out_c * (g + 1), :, :]).numel() == 0

            # Test 3D
            input_var = torch.randn(batch, in_c, size, size, size)
            filter_var = torch.zeros(eff_out_c, in_c, kernel_size, kernel_size, kernel_size)
            filter_var = torch.cat([filter_var] * groups)
            init.dirac_(filter_var, groups)
            output_var = F.conv3d(input_var, filter_var)
            input_tensor, output_tensor = input_var.data, output_var.data
            for g in range(groups):
                # Assert in_c outputs are preserved (per each group)
                self.assertEqual(input_tensor[:, :, 1:-1, 1:-1, 1:-1],
                                 output_tensor[:, eff_out_c * g:eff_out_c * g + in_c, :, :, :])
                # Assert extra outputs are 0
                assert torch.nonzero(output_tensor[:, eff_out_c * g + in_c:eff_out_c * (g + 1), :, :, :]).numel() == 0

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

    def test_kaiming_uniform_warning_on_0element_tensor(self):
        tensor = torch.empty(0, 1)
        with self.assertWarnsRegex(UserWarning, "Initializing zero-element tensors is a no-op"):
            _ = init.kaiming_uniform_(tensor)

    def test_kaiming_normal_warning_on_0element_tensor(self):
        tensor = torch.empty(0, 1)
        with self.assertWarnsRegex(UserWarning, "Initializing zero-element tensors is a no-op"):
            _ = init.kaiming_normal_(tensor)

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
                                     torch.eye(cols) * gain ** 2, atol=1e-6, rtol=0)
                else:
                    self.assertEqual(torch.mm(flattened_tensor, flattened_tensor.t()),
                                     torch.eye(rows) * gain ** 2, atol=1e-6, rtol=0)

    def test_deprecation(self):
        x = torch.randn(3, 3)

        def fn():
            init.normal(x)

        with self.assertWarnsRegex(UserWarning, 'deprecated', msg='methods not suffixed with underscore should be deprecated'):
            fn()

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

        self.assertEqual(Y_ref, Y_hat, msg="Conv+BN fusion results are off")

        na_bn_ref = torch.nn.BatchNorm2d(oC, affine=False)
        na_bn_ref.running_mean = torch.from_numpy(running_mean[0]).to(torch.double)
        na_bn_ref.running_var = torch.from_numpy(running_var[0]).to(torch.double)
        na_bn_ref.eval()

        Y_ref = na_bn_ref(conv_ref(inputs))
        conv_na_bn_fused = torch.nn.utils.fusion.fuse_conv_bn_eval(conv_ref,
                                                                   na_bn_ref)
        Y_hat = conv_na_bn_fused(inputs)

        self.assertEqual(Y_ref, Y_hat, msg="Conv+BN(non-affine) fusion results are off")


class TestConstantPadNd(TestCase):
    def test_constant_pad_nd(self):
        a = torch.tensor([[1, 2], [3, 4]])
        res = torch.constant_pad_nd(a, [1, 2, 1, 0], 9)
        expected = torch.tensor([
            [9, 9, 9, 9, 9],
            [9, 1, 2, 9, 9],
            [9, 3, 4, 9, 9]
        ])
        self.assertEqual(res, expected)

    def test_preserves_memory_format(self):
        nchw_tensor = torch.rand((1, 2, 5, 3))
        nchw_padded = torch.constant_pad_nd(nchw_tensor, [1, 2], 0.5)
        self.assertTrue(nchw_padded.is_contiguous(memory_format=torch.contiguous_format))

        nhwc_tensor = nchw_tensor.contiguous(memory_format=torch.channels_last)
        nhwc_padded = torch.constant_pad_nd(nhwc_tensor, [1, 2], 0.5)
        self.assertTrue(nhwc_padded.is_contiguous(memory_format=torch.channels_last))


class TestAddRelu(TestCase):
    def test_add_relu(self):
        a = torch.rand((7, 11))
        b = torch.rand((7, 11))
        a = a.float()
        b = b.float()
        a = a * -10
        a = a + 5
        add_res = a + b
        relu_res = torch.relu(add_res)
        add_relu_res = torch._VF._add_relu(a, b)

        self.assertEqual(add_relu_res, relu_res)

    def test_add_relu_broadcasting(self):
        a = torch.rand((1, 32))
        b = 1
        b_scalar = torch.ones(1, 32)
        res = torch._VF._add_relu(a, b)
        broadcasted_res = torch._VF._add_relu(a, b_scalar)

        self.assertEqual(broadcasted_res, res)


def add_test(test, decorator=None):
    def add(test_name, fn):
        if hasattr(TestNN, test_name):
            raise RuntimeError('Found two tests with the same name: ' + test_name)
        if decorator is not None:
            fn = decorator(fn)
        setattr(TestNN, test_name, fn)

    test_name = test.get_name()
    if not hasattr(test, 'test_cpu') or test.test_cpu:
        add(test_name, lambda self, test=test: test(self))
    cuda_test_name = test_name + '_cuda'
    # With dtype enable, it's good enough to test against three floating types
    kwargs = {}
    if 'extra_args' in get_function_arglist(test.test_cuda):
        kwargs['extra_args'] = test.extra_args

    if 'dtype' in get_function_arglist(test.test_cuda):
        if tf32_is_not_fp32() and test.with_tf32:

            def with_tf32_off(self, test=test, kwargs=kwargs):
                with tf32_off():
                    test.test_cuda(self, dtype=torch.float, **kwargs)

            add(cuda_test_name + '_fp32', with_tf32_off)

            def with_tf32_on(self, test=test, kwargs=kwargs):
                with tf32_on(self, test.tf32_precision):
                    test.test_cuda(self, dtype=torch.float, **kwargs)

            add(cuda_test_name + '_tf32', with_tf32_on)
        else:
            add(cuda_test_name + '_float', lambda self,
                test=test, kwargs=kwargs: test.test_cuda(self, dtype=torch.float, **kwargs))
        add(cuda_test_name + '_double', lambda self,
            test=test, kwargs=kwargs: test.test_cuda(self, dtype=torch.double, **kwargs))

        def test_half(self, test=test, kwargs=kwargs):
            test.test_cuda(self, dtype=torch.half, **kwargs)
        if getattr(test, 'check_half', True):
            add(cuda_test_name + '_half', test_half)

        def test_bfloat16(self, test=test, kwargs=kwargs):
            test.test_cuda(self, dtype=torch.bfloat16, **kwargs)
        if getattr(test, 'check_bfloat16', True):
            add(cuda_test_name + '_bfloat16', test_bfloat16)

        def test_cfloat(self, test=test, kwargs=kwargs):
            test.test_cuda(self, dtype=torch.cfloat, **kwargs)

        def test_cdouble(self, test=test, kwargs=kwargs):
            test.test_cuda(self, dtype=torch.cdouble, **kwargs)
        if getattr(test, 'check_complex', False):
            add(cuda_test_name + '_cfloat', test_cfloat)
            add(cuda_test_name + '_cdouble', test_cdouble)

    else:
        def with_tf32_off(self, test=test, kwargs=kwargs):
            with tf32_off():
                test.test_cuda(self, **kwargs)

        if tf32_is_not_fp32() and test.with_tf32:
            add(cuda_test_name + '_fp32', with_tf32_off)

            def with_tf32_on(self, test=test, kwargs=kwargs):
                with tf32_on(self, test.tf32_precision):
                    test.test_cuda(self, **kwargs)

            add(cuda_test_name + '_tf32', with_tf32_on)
        else:
            add(cuda_test_name, with_tf32_off)

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
            # For bad reasons this would create LongTensors that requires gradients
            # Remove requires_grad to avoid this
            for p in m.parameters():
                p.requires_grad_(False)
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

for test_params in criterion_tests:
    if 'constructor' not in test_params:
        name = test_params.pop('module_name')
        test_params['constructor'] = getattr(nn, name)
    test = CriterionTest(**test_params)
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
        test = CriterionTest(**test_params)
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

add_test(NewModuleTest(
    constructor=lambda: UnpoolingNet(
        nn.MaxPool1d(2, return_indices=True),
        nn.MaxUnpool1d(2)),
    input_size=(1, 4),
    reference_fn=single_batch_reference_fn,
    fullname='MaxUnpool1d_net_no_batch_dim',))
add_test(NewModuleTest(
    constructor=lambda: UnpoolingNet(
        nn.MaxPool2d(2, return_indices=True),
        nn.MaxUnpool2d(2)),
    input_size=(1, 2, 4),
    reference_fn=single_batch_reference_fn,
    fullname='MaxUnpool2d_net_no_batch_dim',))

add_test(NewModuleTest(
    constructor=lambda: UnpoolingNet(
        nn.MaxPool3d(2, return_indices=True),
        nn.MaxUnpool3d(2)),
    input_size=(1, 2, 4, 6),
    reference_fn=single_batch_reference_fn,
    fullname='MaxUnpool3d_net_no_batch_dim',
    check_gradgrad=False))

class _AdaptiveLogSoftmaxWithLoss(nn.AdaptiveLogSoftmaxWithLoss):
    def __call__(self, input):
        t = torch.tensor([0, 1, 4, 8]).to(input.device)
        return nn.AdaptiveLogSoftmaxWithLoss.__call__(self, input, t).output

add_test(NewModuleTest(
    constructor=lambda: _AdaptiveLogSoftmaxWithLoss(16, 10, [2, 6]),
    input_size=(4, 16),
    fullname='AdaptiveLogSoftmax',
    with_tf32=True,
    tf32_precision=0.005))


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
    def _test_InstanceNorm_general(self, cls, input, device, dtype=torch.float):
        # default case track_running_stats=False
        b, c = input.size(0), input.size(1)
        input_var = input.to(device=device, dtype=dtype).requires_grad_()

        IN = cls(c, eps=0).to(device, dtype)

        output = IN(input_var)
        out_reshaped = output.view(b * c, -1)

        mean = out_reshaped.mean(1)
        var = out_reshaped.var(1, unbiased=False)

        self.assertEqual(torch.abs(mean.data).mean(), 0, atol=1e-5, rtol=0)
        self.assertEqual(torch.abs(var.data).mean(), 1, atol=1e-5, rtol=0)

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

        self.assertEqual(torch.abs(mean.data - IN.running_mean).mean(), 0, atol=1e-5, rtol=0)
        self.assertEqual(torch.abs(var.data.mean(1) - IN.running_var).mean(), 0, atol=1e-5, rtol=0)

        # in eval mode, adding X * std to a channel in input should make the
        # corresponding channel in output have mean X
        IN.eval()
        delta = IN.running_var.sqrt() * torch.arange(c, device=device, dtype=dtype)
        delta = delta.view(-1, *[1 for _ in range(2, input.dim())])
        output = IN(input_var + delta)
        self.assertEqual(output.transpose(0, 1).reshape(c, -1).mean(1), torch.arange(c, dtype=dtype))

    def _test_InstanceNorm_cuda_half(self, cls, input, device):
        # THNN
        input = input.to(device=device, dtype=torch.half).random_(1, 10).requires_grad_(True)
        m = cls(input.size(1), affine=True, track_running_stats=True).to(device, torch.half)
        thnn_output = m(input)
        thnn_output.sum().backward()
        thnn_input_grad = input.grad.data.clone()
        self.assertEqualTypeString(thnn_output, input)
        # cuDNN
        if TEST_CUDNN:
            input.grad = None
            m = m.float()
            cudnn_output = m(input)
            cudnn_output.sum().backward()
            cudnn_input_grad = input.grad.data.clone()
            self.assertEqualTypeString(cudnn_output, input)
            self.assertEqual(cudnn_output, thnn_output, atol=1e-4, rtol=0)
            self.assertEqual(cudnn_input_grad, thnn_input_grad, atol=1e-3, rtol=0)

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

            delta = 1e-1 if dtype == torch.bfloat16 else 1e-5
            self.assertEqual(torch.abs(mean.data).mean(), 0, atol=delta, rtol=0)
            self.assertEqual(torch.abs(var.data).mean(), 1, atol=delta, rtol=0)

            # test that LN applies weight and bias correctly
            scale, bias = torch.empty(2).uniform_(0.2, 2).tolist()
            ln.weight.data.fill_(scale)
            ln.bias.data.fill_(bias)
            output = ln(x)
            out_reshaped = output.view(*(unnormalized_shape + [-1]))
            mean = out_reshaped.mean(-1)
            var = out_reshaped.var(-1, unbiased=False)
            self.assertEqual(torch.abs(mean.data).mean(), bias, atol=delta, rtol=0)
            self.assertEqual(torch.abs(var.data).mean(), scale ** 2, atol=delta, rtol=0)

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
        self.assertEqualTypeString(output, input)

    def _test_LayerNorm_cpu_mixed_dtype(self, device):
        for elementwise_affine in [True, False]:
            # layer norm input shape is normalized to m x n, cpu vectorized on n,
            # so make sure n exceeds vector length
            input = torch.empty(2, 3, 11, 3, device=device, dtype=torch.bfloat16).random_(1, 10)
            m = nn.LayerNorm([11, 3], elementwise_affine=elementwise_affine).to(device, torch.bfloat16)
            m2 = deepcopy(m).to(device, torch.float)
            out = m(input)
            out2 = m2(input)
            self.assertEqual(out, out2)

    def _test_GroupNorm_general(self, device, dtype=torch.float):
        good_shape_g = {
            (1, 2, 3, 4): 2,
            (2, 3, 10): 3,
            (3, 1, 1, 1, 2): 1,
            (2, 6, 4, 2, 2): 3,
            (1, 256, 1, 1): 32,
        }
        for shape_g, grad in product(good_shape_g.items(), [True, False]):
            shape, g = shape_g
            x = torch.empty(*shape, device=device, dtype=dtype).uniform_(0, 10)
            x.requires_grad_(grad)
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
            self.assertEqual(torch.abs(mean).mean(), 0, atol=1e-5, rtol=0)
            self.assertEqual(torch.abs(var).mean(), 1, atol=1e-5, rtol=0)

            output.backward(torch.randn_like(output))
            if output.is_cuda:
                torch.cuda.synchronize()

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
            self.assertEqual(torch.abs(mean).mean(), 0, atol=1e-5, rtol=0)
            self.assertEqual(torch.abs(var).mean(), 1, atol=1e-5, rtol=0)

        bad_shape_g = {
            (1, 2, 3, 4): 3,
            (2, 3, 10): 2,
            (3, 1, 1, 1, 2): 10,
            (2, 6, 4, 2, 2): 4,
        }
        for shape, g in bad_shape_g.items():
            with self.assertRaises(ValueError):
                gn = nn.GroupNorm(g, shape[1])

    def _test_GroupNorm_cuda_half(self):
        input = torch.zeros(2, 4, 3, 2, requires_grad=True).cuda().half().random_(1, 10)
        m = nn.GroupNorm(2, 4).to("cuda", torch.half)
        output = m(input)
        output.sum().backward()
        self.assertEqualTypeString(output, input)

    def _test_GroupNorm_cpu_mixed_dtype(self):
        def helper(self, size, groups, memory_format):
            channels = size[1]
            input = torch.empty(size, dtype=torch.bfloat16).cpu().random_(1, 10)
            input = input.contiguous(memory_format=memory_format).detach().requires_grad_(True)
            input_bf = input.clone().detach().requires_grad_(True)
            inputf = input.float().detach().requires_grad_(True)
            m = nn.GroupNorm(groups, channels).cpu().bfloat16()
            m2 = deepcopy(m).float()
            m3 = deepcopy(m2)
            out = m(input)
            out2 = m2(input_bf)
            out3 = m3(inputf)
            self.assertEqual(out, out2, atol=5e-3, rtol=5e-3)
            self.assertEqual(out2.float(), out3, atol=5e-3, rtol=5e-3)
            grad_out = torch.rand(out2.shape, device="cpu", dtype=torch.bfloat16, requires_grad=True)
            grad_out2 = grad_out.float()
            out2.backward(grad_out, retain_graph=True)
            out3.backward(grad_out2, retain_graph=True)
            self.assertEqual(m2.weight.grad.float(), m3.weight.grad, atol=1e-5, rtol=1e-5)
            self.assertEqual(input_bf.grad.to(torch.float), inputf.grad, atol=5e-5, rtol=5e-3)

        helper(self, (1, 8, 4, 3), 2, torch.channels_last)
        helper(self, (1, 8, 4, 3), 2, torch.contiguous_format)
        helper(self, (1, 8, 3, 4), 4, torch.contiguous_format)
        helper(self, (1, 8, 3, 4), 4, torch.channels_last)
        helper(self, (1, 8, 40, 40), 4, torch.channels_last)
        helper(self, (1, 8, 40, 40), 4, torch.contiguous_format)
        helper(self, (1, 9, 3, 4, 5), 3, torch.channels_last_3d)

    def _test_module_empty_inputs(self, module, inputs):
        for _inp in inputs:
            _inp.requires_grad_(True)
        out = module(*inputs)
        gO = torch.rand_like(out)
        out.backward(gO)

        for p in module.parameters():
            if p.requires_grad:
                self.assertEqual(p.grad, torch.zeros_like(p.grad))

        for _inp in inputs:
            self.assertEqual(_inp.grad, torch.zeros_like(_inp))

    @unittest.skipIf((not TEST_NUMPY) or (not TEST_SCIPY) or (scipy.__version__ < '1.0.0'),
                     "Scipy v1.0 and/or numpy not found")
    @tf32_on_and_off()
    def test_affine_2d_rotate0(self, device):
        # scipy before 1.0.0 do not support homogeneous coordinate
        # scipy.ndimage.affine_transform, so we need to skip.
        input_size = [1, 1, 3, 3]
        input_ary = np.array(np.random.random(input_size), dtype=np.float32)
        output_size = [1, 1, 5, 5]
        angle_rad = 0.

        transform_tensor, transform_ary, offset = \
            _buildEquivalentAffineTransforms2d(device, input_size, output_size, angle_rad)

        scipy_ary = torch.from_numpy(scipy.ndimage.affine_transform(
            input_ary[0, 0],
            transform_ary,
            offset=offset,
            output_shape=output_size[2:],
            order=1,
            mode='nearest',
            prefilter=False))

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
        ).to('cpu')

        self.assertEqual(scipy_ary.mean(), gridsample_ary.mean())
        self.assertEqual(scipy_ary, gridsample_ary.reshape_as(scipy_ary))

    @unittest.skipIf((not TEST_NUMPY) or (not TEST_SCIPY) or (scipy.__version__ < '1.0.0'),
                     "Scipy v1.0 and/or numpy not found")
    @tf32_on_and_off(0.001)
    def test_affine_2d_rotate90(self, device):
        # scipy before 1.0.0 do not support homogeneous coordinate
        # scipy.ndimage.affine_transform, so we need to skip.
        for input_size2dsq, output_size2dsq in \
                itertools.product(input_size2dsq_(), output_size2dsq_()):
            input_size = input_size2dsq
            input_ary = np.array(np.random.random(input_size), dtype=np.float32)
            output_size = output_size2dsq
            angle_rad = 0.25 * math.pi * 2

            transform_tensor, transform_ary, offset = \
                _buildEquivalentAffineTransforms2d(device, input_size, output_size, angle_rad)

            scipy_ary = torch.from_numpy(scipy.ndimage.affine_transform(
                input_ary[0, 0],
                transform_ary,
                offset=offset,
                output_shape=output_size[2:],
                order=1,
                mode='nearest',
                prefilter=True))

            if input_size2dsq == output_size2dsq:
                self.assertEqual(scipy_ary.mean(), input_ary.mean())
            self.assertEqual(scipy_ary[0, 0], input_ary[0, 0, 0, -1])
            self.assertEqual(scipy_ary[0, -1], input_ary[0, 0, -1, -1])
            self.assertEqual(scipy_ary[-1, -1], input_ary[0, 0, -1, 0])
            self.assertEqual(scipy_ary[-1, 0], input_ary[0, 0, 0, 0])

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
            ).to('cpu')

            self.assertEqual(scipy_ary.mean(), gridsample_ary.mean())
            self.assertEqual(scipy_ary, gridsample_ary.reshape_as(scipy_ary))

    @unittest.skipIf((not TEST_NUMPY) or (not TEST_SCIPY) or (scipy.__version__ < '1.0.0'),
                     "Scipy v1.0 and/or numpy not found")
    @tf32_on_and_off(0.005)
    def test_affine_2d_rotate45(self, device):
        # scipy before 1.0.0 do not support homogeneous coordinate
        # scipy.ndimage.affine_transform, so we need to skip.
        input_size = [1, 1, 3, 3]
        input_ary = np.array(np.zeros(input_size), dtype=np.float32)
        input_ary[0, 0, 0, :] = 0.5
        input_ary[0, 0, 2, 2] = 1.0
        output_size = [1, 1, 3, 3]
        angle_rad = 0.125 * math.pi * 2

        transform_tensor, transform_ary, offset = \
            _buildEquivalentAffineTransforms2d(device, input_size, output_size, angle_rad)

        scipy_ary = torch.from_numpy(scipy.ndimage.affine_transform(
            input_ary[0, 0],
            transform_ary,
            offset=offset,
            output_shape=output_size[2:],
            order=1,
            mode='nearest',
            prefilter=False))

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
        ).to('cpu')

        self.assertEqual(scipy_ary, gridsample_ary.reshape_as(scipy_ary))

    @unittest.skipIf((not TEST_NUMPY) or (not TEST_SCIPY) or (scipy.__version__ < '1.0.0'),
                     "Scipy v1.0 and/or numpy not found")
    @tf32_on_and_off(0.005)
    def test_affine_2d_rotateRandom(self, device):
        # scipy before 1.0.0 do not support homogeneous coordinate
        # scipy.ndimage.affine_transform, so we need to skip.
        for angle_rad, input_size2d, output_size2d in \
                itertools.product(angle_rad_(), input_size2d_(), output_size2d_()):

            input_size = input_size2d
            input_ary = np.array(np.random.random(input_size), dtype=np.float32).round(3)
            output_size = output_size2d

            input_ary[0, 0, 0, 0] = 2
            input_ary[0, 0, 0, -1] = 4
            input_ary[0, 0, -1, 0] = 6
            input_ary[0, 0, -1, -1] = 8

            transform_tensor, transform_ary, grid_ary = \
                _buildEquivalentAffineTransforms2d(device, input_size, output_size, angle_rad)

            scipy_ary = torch.from_numpy(scipy.ndimage.affine_transform(
                input_ary[0, 0],
                transform_ary,
                output_shape=output_size[2:],
                order=1,
                mode='nearest',
                prefilter=False))

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
            ).to('cpu')

            affine_tensor = affine_tensor.to('cpu')

            for r in range(affine_tensor.size(1)):
                for c in range(affine_tensor.size(2)):
                    grid_out = np.dot(grid_ary, [r, c, 1])
                    self.assertEqual(affine_tensor[0, r, c], grid_out[:2], exact_dtype=False)

            self.assertEqual(scipy_ary, gridsample_ary.reshape_as(scipy_ary))

    @unittest.skipIf((not TEST_NUMPY) or (not TEST_SCIPY) or (scipy.__version__ < '1.0.0'),
                     "Scipy v1.0 and/or numpy not found")
    @tf32_on_and_off(0.005)
    def test_affine_3d_rotateRandom(self, device):
        # scipy before 1.0.0 do not support homogeneous coordinate
        # scipy.ndimage.affine_transform, so we need to skip.
        for angle_rad, axis_vector, input_size3d, output_size3d in \
                itertools.product(angle_rad_(), axis_vector_(), input_size3d_(), output_size3d_()):
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

            scipy_ary = torch.from_numpy(scipy.ndimage.affine_transform(
                input_ary[0, 0],
                transform_ary,
                output_shape=output_size[2:],
                order=1,
                mode='nearest',
                prefilter=False))

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
            ).to('cpu')

            affine_tensor = affine_tensor.to('cpu')

            for i in range(affine_tensor.size(1)):
                for r in range(affine_tensor.size(2)):
                    for c in range(affine_tensor.size(3)):
                        grid_out = np.dot(grid_ary, [i, r, c, 1])
                        self.assertEqual(affine_tensor[0, i, r, c], grid_out[:3], exact_dtype=False)

            self.assertEqual(scipy_ary, gridsample_ary.reshape_as(scipy_ary))


    @onlyCUDA
    @dtypes(torch.float, torch.half)
    def test_batchnorm_large_batch(self, device, dtype):
        bn = nn.BatchNorm2d(1).to(device, dtype)
        data = torch.rand(880801, 1, 1, 1, device=device, dtype=dtype)
        out = bn(data).sum().backward()

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

    def test_instancenorm_raises_error_if_less_than_one_value_per_channel(self, device):
        x = torch.rand(10)[None, :, None]
        with self.assertRaises(ValueError):
            torch.nn.InstanceNorm1d(10)(x).to(device)

    def test_instancenorm_raises_error_for_single_spatial_element_during_training(self, device):
        BATCH_SIZE = 10
        NUM_CHANNELS = 3
        norms = [torch.nn.InstanceNorm1d, torch.nn.InstanceNorm2d, torch.nn.InstanceNorm3d]
        for i, norm in enumerate(norms):
            m = norm(NUM_CHANNELS, track_running_stats=True)
            m.to(device)

            # Create an appropriately-sized input with a single spatial element.
            input = torch.randn(BATCH_SIZE, NUM_CHANNELS, *[1 for _ in range(i + 1)],
                                device=device)
            with self.assertRaises(ValueError):
                m(input)

            # Single spatial element should be fine in eval.
            m.eval()
            m(input)

    def test_LayerNorm_general(self, device):
        self._test_LayerNorm_general(device)

        if self.device_type == 'cuda' or self.device_type == 'cpu':
            self._test_LayerNorm_general(device, dtype=torch.bfloat16)

        if self.device_type == 'cuda':
            self._test_LayerNorm_cuda_half(device)

        if self.device_type == 'cpu':
            self._test_LayerNorm_cpu_mixed_dtype(device)

    @onlyNativeDeviceTypes
    def test_LayerNorm_numeric(self, device):
        def layer_norm_ref(X, gamma, beta, normalized_shape, eps):
            feature_size = np.prod(normalized_shape)
            X_view = X.view(-1, feature_size)
            mean = X_view.mean(dim=-1, keepdim=True)
            var = X_view.var(dim=-1, unbiased=False, keepdim=True)
            Y = (X_view - mean) / torch.sqrt(var + eps)
            Y = Y * gamma.view(-1) + beta.view(-1)
            return Y.view(*X.size())

        normalized_shape = [256, 256, 144]
        layer_norm = nn.LayerNorm(normalized_shape).float().to(device)
        X = torch.rand(2, *normalized_shape, dtype=torch.float32,
                       device=device)

        Y = layer_norm(X)
        Y_ref = layer_norm_ref(X, layer_norm.weight.data, layer_norm.bias.data,
                               normalized_shape, layer_norm.eps)
        self.assertEqual(Y, Y_ref, rtol=0, atol=1e-5)

        if self.device_type == 'cuda':
            layer_norm.cpu()
            Y_cpu = layer_norm(X.cpu())
            self.assertEqual(Y_cpu, Y, rtol=0, atol=1e-5)

    @onlyCPU
    def test_glu_bfloat16(self, device):
        def test_dtype(fn, input, dtype):
            input = input.detach().clone().to(dtype=dtype).requires_grad_(True)
            input2 = input.detach().clone().float().requires_grad_(True)
            out = fn(input)
            out.sum().backward()
            out2 = fn(input2)
            out2.sum().backward()
            self.assertEqual(out.dtype, dtype)
            self.assertEqual(input.grad.dtype, dtype)
            self.assertEqual(out, out2, exact_dtype=False)
            self.assertEqual(input.grad, input2.grad, atol=1e-2, rtol=0, exact_dtype=False)

        def func(device):
            return torch.nn.GLU(dim=-1).to(device)

        shapes = [[1, 3, 1, 6], [1, 3, 1, 128], [1, 3, 256, 256]]
        for shape in shapes:
            x = torch.randn(shape, device=device)
            test_dtype(func(device), x, torch.bfloat16)

    @onlyNativeDeviceTypes
    def test_GroupNorm_general(self, device):
        self._test_GroupNorm_general(device)

        if self.device_type == 'cuda':
            self._test_GroupNorm_cuda_half()

        if self.device_type == 'cpu':
            self._test_GroupNorm_cpu_mixed_dtype()

    def test_GroupNorm_raises_error_if_one_value_per_group(self, device):
        x = torch.rand(10)[None, :, None]
        with self.assertRaises(ValueError):
            torch.nn.GroupNorm(10, 10)(x).to(device)

    def test_GroupNorm_empty(self, device):
        mod = torch.nn.GroupNorm(2, 4).to(device)
        inp = torch.randn(0, 4, 2, 2, device=device)
        _test_module_empty_input(self, mod, inp)
        if self.device_type == 'cuda' and self.has_cudnn():
            with torch.backends.cudnn.flags(enabled=False):
                _test_module_empty_input(self, mod, inp)

    @onlyCPU
    @dtypes(torch.float, torch.double)
    def test_groupnorm_nhwc(self, device, dtype):
        def helper(self, size, groups, memory_format):
            channels = size[1]
            input = torch.randn(size, dtype=dtype, device=device, requires_grad=True)
            input = input.contiguous(memory_format=memory_format)
            input.retain_grad()
            grad = torch.randn(size, dtype=dtype, device=device)
            grad = grad.contiguous(memory_format=memory_format)
            gn = nn.GroupNorm(groups, channels).to(device).to(dtype)
            gn.weight.data.uniform_()
            gn.bias.data.uniform_()

            ref_input = input.detach().clone().contiguous().requires_grad_(True)
            ref_grad = grad.detach().clone().contiguous()
            ref_gn = nn.GroupNorm(groups, channels).to(device).to(dtype)
            ref_gn.load_state_dict(gn.state_dict())

            out = gn(input)
            out.backward(grad)
            ref_out = ref_gn(ref_input)
            ref_out.backward(ref_grad)

            self.assertTrue(out.is_contiguous(memory_format=memory_format))
            self.assertTrue(ref_out.is_contiguous())
            self.assertEqual(out, ref_out)
            self.assertEqual(gn.weight.grad, ref_gn.weight.grad)
            self.assertEqual(gn.bias.grad, ref_gn.bias.grad)
            self.assertEqual(input.grad, ref_input.grad)

        helper(self, (4, 8, 10, 10), 4, torch.channels_last)
        helper(self, (2, 30, 9, 9), 3, torch.channels_last)
        helper(self, (2, 9, 7, 11, 15), 3, torch.channels_last_3d)

    @onlyNativeDeviceTypes
    def test_GroupNorm_numeric(self, device):
        def group_norm_ref(X, gamma, beta, groups, channels, eps):
            batch_size = X.size()[0]
            X_view = X.view(batch_size, groups, -1)
            mean = X_view.mean(dim=-1, keepdim=True)
            var = X_view.var(dim=-1, unbiased=False, keepdim=True)
            Y = ((X_view - mean) / torch.sqrt(var + eps)).view(
                batch_size, channels, -1)
            Y = Y * gamma.view(channels, 1) + beta.view(channels, 1)
            return Y.view(*X.size())

        batch_size = 1
        groups = 2
        channels = 8
        group_norm = nn.GroupNorm(groups, channels).float().to(device)
        X = torch.rand(batch_size, channels, 256, 256, 72,
                       dtype=torch.float32, device=device)

        Y = group_norm(X)
        Y_ref = group_norm_ref(
            X, group_norm.weight.data, group_norm.bias.data, groups,
            channels, group_norm.eps)
        self.assertEqual(Y, Y_ref, rtol=0, atol=1e-5)

        if self.device_type == 'cuda':
            group_norm.cpu()
            Y_cpu = group_norm(X.cpu())
            self.assertEqual(Y_cpu, Y, rtol=0, atol=1e-5)

    @onlyNativeDeviceTypes
    @dtypes(torch.float64, torch.complex128)
    def test_pad(self, device, dtype):
        # Assert assertion errors are raised for invalid circular padding values
        inputs = torch.randn(1, 1, 4, device=device, dtype=dtype, requires_grad=True)
        # Should raise error when trying to wrap around more than once
        self.assertRaises(RuntimeError, lambda: F.pad(inputs, (5, 4), mode='circular'))
        self.assertRaises(RuntimeError, lambda: F.pad(inputs, (3, 6), mode='circular'))
        # Should raise error when negative padding results in negative output shape
        self.assertRaises(RuntimeError, lambda: F.pad(inputs, (-3, -2), mode='circular'))

        # assert that relfection padding errors when pad >= input size
        expected_err_msg = r"Padding size should be less than the corresponding input dimension"
        inputs = torch.randn(1, 1, 2, 3, device=device, dtype=dtype)
        self.assertRaisesRegex(RuntimeError, expected_err_msg,
                               lambda: F.pad(inputs, (1, 1, 3, 0), mode='reflect'))
        inputs = torch.randn(1, 1, 2, device=device, dtype=dtype)
        self.assertRaisesRegex(RuntimeError, expected_err_msg,
                               lambda: F.pad(inputs, (2, 1), mode='reflect'))

        inputs = torch.rand(1, 3, 4, 4, device=device, dtype=dtype)
        # assert that pad doesn't return a view into the input tensor
        for mode in 'constant', 'reflect', 'replicate', 'circular':
            out = F.pad(inputs, (0, 0, 0, 0), mode=mode)
            out.fill_(4)
            self.assertTrue(torch.all(torch.abs(inputs) < 2))

            out = F.pad(inputs, (0, 0, -1, -1), mode=mode)
            out.fill_(4)
            self.assertTrue(torch.all(torch.abs(inputs) < 2))

    @onlyNativeDeviceTypes
    @dtypes(torch.float64, torch.complex128)
    def test_ReplicationPad_empty(self, device, dtype):
        for mod, inp in [
                (torch.nn.ReplicationPad1d(3), torch.randn(0, 3, 10, device=device, dtype=dtype)),
                (torch.nn.ReplicationPad2d(3), torch.randn(0, 3, 10, 10, device=device, dtype=dtype)),
                (torch.nn.ReplicationPad3d(3), torch.randn(0, 3, 10, 10, 10, device=device, dtype=dtype))]:
            _test_module_empty_input(self, mod, inp, check_size=False)

        with self.assertRaisesRegex(RuntimeError, 'Expected 2D or 3D'):
            mod = torch.nn.ReplicationPad1d(2)
            inp = torch.randn(3, 0, 10, device=device, dtype=dtype)
            mod(inp)

        with self.assertRaisesRegex(RuntimeError, 'Expected 3D or 4D'):
            mod = torch.nn.ReplicationPad2d((2, 2, 2, 2))
            inp = torch.randn(43, 0, 10, 10, device=device, dtype=dtype)
            mod(inp)

        with self.assertRaisesRegex(RuntimeError, 'Expected 4D or 5D'):
            mod = torch.nn.ReplicationPad3d((2, 2, 2, 2, 2, 2))
            inp = torch.randn(3, 0, 10, 10, 10, device=device, dtype=dtype)
            mod(inp)

    def test_ReplicationPad1d_large(self, device):
        shapes = ([2, 65736, 4], [65736, 2, 4])
        pl, pr = 3, 4
        for shape in shapes:
            x = torch.randn(shape, device=device, requires_grad=True)
            model = torch.nn.ReplicationPad1d((pl, pr))

            # forward
            out = model(x)
            self.assertEqual(out[:, :, pl : -pr], x)

            left_padding = out[:, :, : pl]
            self.assertEqual(left_padding, x[:, :, :1].expand_as(left_padding))
            right_padding = out[:, :, -pr :]
            self.assertEqual(right_padding, x[:, :, -1:].expand_as(right_padding))

            # backward
            g = torch.randn_like(out)
            out.backward(g)
            self.assertEqual(x.grad[:, :, 1 : -1], g[:, :, pl + 1 : -pr - 1])

            self.assertEqual(x.grad[:, :, 0], g[:, :, : pl + 1].sum(-1))
            self.assertEqual(x.grad[:, :, -1], g[:, :, -pr - 1:].sum(-1))

    def test_ReplicationPad2d_large(self, device):
        shapes = ([2, 65736, 4, 4], [65736, 2, 4, 4])
        pl, pr, pt, pb = 3, 4, 5, 6
        for shape in shapes:
            x = torch.randn(shape, device=device, requires_grad=True)
            model = torch.nn.ReplicationPad2d((pl, pr, pt, pb))

            # forward center, edge
            out = model(x)
            self.assertEqual(out[:, :, pt : -pb, pl : -pr], x)

            left_padding = out[:, :, pt : -pb, : pl]
            self.assertEqual(left_padding, x[:, :, :, :1].expand_as(left_padding))
            right_padding = out[:, :, pt : -pb, -pr :]
            self.assertEqual(right_padding, x[:, :, :, -1:].expand_as(right_padding))
            top_padding = out[:, :, : pt, pl : -pr]
            self.assertEqual(top_padding, x[:, :, :1, :].expand_as(top_padding))
            bottom_padding = out[:, :, -pb : , pl : -pr]
            self.assertEqual(bottom_padding, x[:, :, -1:, :].expand_as(bottom_padding))

            # forward corner
            tl_padding = out[:, :, : pt + 1, : pl + 1]
            self.assertEqual(tl_padding, x[:, :, :1, :1].expand_as(tl_padding))
            tr_padding = out[:, :, : pt + 1, -pr - 1:]
            self.assertEqual(tr_padding, x[:, :, :1, -1:].expand_as(tr_padding))
            bl_padding = out[:, :, -pb - 1:, : pl + 1]
            self.assertEqual(bl_padding, x[:, :, -1:, :1].expand_as(bl_padding))
            br_padding = out[:, :, -pb - 1:, -pr - 1:]
            self.assertEqual(br_padding, x[:, :, -1:, -1:].expand_as(br_padding))

            # backward center, edge
            g = torch.randn_like(out)
            out.backward(g)
            self.assertEqual(x.grad[:, :, 1:-1, 1:-1], g[:, :, pt + 1 : -pb - 1, pl + 1 : -pr - 1])

            self.assertEqual(x.grad[:, :, 1:-1, 0], g[:, :, pt + 1 : -pb - 1, : pl + 1].sum(-1))
            self.assertEqual(x.grad[:, :, 1:-1, -1], g[:, :, pt + 1 : -pb - 1, -pr - 1 :].sum(-1))
            self.assertEqual(x.grad[:, :, 0, 1:-1], g[:, :, : pt + 1, pl + 1 : -pr - 1].sum(-2))
            self.assertEqual(x.grad[:, :, -1, 1:-1], g[:, :, -pb - 1 :, pl + 1 : -pr - 1].sum(-2))

            # backward corner
            self.assertEqual(x.grad[:, :, 0, 0], g[:, :, : pt + 1, : pl + 1].sum((-2, -1)))
            self.assertEqual(x.grad[:, :, 0, -1], g[:, :, : pt + 1, -pr - 1 :].sum((-2, -1)))
            self.assertEqual(x.grad[:, :, -1, 0], g[:, :, -pb - 1 :, : pl + 1].sum((-2, -1)))
            self.assertEqual(x.grad[:, :, -1, -1], g[:, :, -pb - 1 :, -pr - 1 :].sum((-2, -1)))

    @largeTensorTest("6GB")
    def test_ReplicationPad3d_large(self, device):
        shapes = ([1, 65736, 2, 2, 2], [65736, 1, 2, 2, 2])
        pl, pr, pt, pbt, pf, pbk = 3, 4, 5, 6, 7, 8

        for shape in shapes:
            x = torch.randn(shape, device=device, requires_grad=True)
            model = torch.nn.ReplicationPad3d((pl, pr, pt, pbt, pf, pbk))

            # forward center
            out = model(x)
            self.assertEqual(out[:, :, pf : -pbk, pt : -pbt, pl : -pr], x)

            # backward center
            g = torch.randn_like(out)
            out.backward(g)
            self.assertEqual(x.grad[:, :, 1:-1, 1:-1, 1:-1], g[:, :, pf + 1 : -pbk - 1, pt + 1 : -pbt - 1, pl + 1 : -pr - 1])

    @onlyNativeDeviceTypes
    def test_Bilinear_empty(self, device):
        mod = torch.nn.Bilinear(20, 30, 40).to(device)
        inp1 = torch.randn(0, 10, 20, requires_grad=True, device=device)
        inp2 = torch.randn(0, 10, 30, requires_grad=True, device=device)

        output = mod(inp1, inp2)
        output.sum().backward()

        self.assertEqual(inp1, torch.zeros_like(inp1))
        self.assertEqual(inp2, torch.zeros_like(inp2))

        self.assertEqual(inp1.grad, torch.zeros_like(inp1))
        self.assertEqual(inp2.grad, torch.zeros_like(inp2))

    @expectedFailureMeta  # RuntimeError: cannot reshape tensor of 0 elements into shape [1, 0, -1]
    @onlyNativeDeviceTypes
    def test_TransformerEncoderLayer_empty(self, device):
        for training in (True, False):
            for batch_first, input_shape in [(True, (0, 10, 512)),
                                             (False, (10, 0, 512))]:
                input = torch.rand(*input_shape, device=device)
                encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=batch_first).to(device)
                if not training:
                    encoder_layer = encoder_layer.eval()
                    with torch.no_grad():
                        _test_module_empty_input(self, encoder_layer, input, check_size=False, inference=True)
                    if batch_first and not TEST_WITH_CROSSREF:
                        with torch.no_grad():
                            # A NestedTensor with no tensors inside it doesn't have dim 3 (or dim
                            # 2, for that matter) so it can't hit the fast path, nor can we give a
                            # result.
                            with self.assertRaisesRegex(
                                    AssertionError, 'MultiheadAttention does not support NestedTensor outside'):
                                nt = torch.nested.nested_tensor([], device=device)
                                _test_module_empty_input(self, encoder_layer, nt, check_size=False, inference=True)

                            nt = torch.nested.nested_tensor([torch.rand(0, 512, device=device)], device=device)
                            _test_module_empty_input(self, encoder_layer, nt, check_size=False, inference=True)
                else:
                    _test_module_empty_input(self, encoder_layer, input, check_size=False)

    @expectedFailureMeta  # RuntimeError: cannot reshape tensor of 0 elements into shape [1, 0, -1]
    @onlyNativeDeviceTypes
    def test_TransformerEncoder_empty(self, device):
        for batch_first, input_shape in [(True, (0, 10, 512)),
                                         (False, (10, 0, 512))]:
            input = torch.rand(*input_shape, device=device)
            encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=batch_first).to(device)
            transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6).to(device)
            _test_module_empty_input(self, transformer_encoder, input, check_size=False)

    @expectedFailureMeta  # RuntimeError: cannot reshape tensor of 0 elements into shape [1, 0, -1]
    @onlyNativeDeviceTypes
    def test_TransformerDecoderLayer_empty(self, device):
        for batch_first, memory_shape, tgt_shape in [(True, (0, 10, 512), (0, 20, 512)),
                                                     (False, (10, 0, 512), (20, 0, 512))]:
            memory = torch.rand(*memory_shape, device=device)
            tgt = torch.rand(*tgt_shape, requires_grad=True, device=device)
            decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=batch_first).to(device)
            self._test_module_empty_inputs(decoder_layer, [tgt, memory])

    @expectedFailureMeta  # RuntimeError: cannot reshape tensor of 0 elements into shape [1, 0, -1]
    @onlyNativeDeviceTypes
    def test_TransformerDecoder_empty(self, device):
        for batch_first, memory_shape, tgt_shape in [(True, (0, 10, 512), (0, 20, 512)),
                                                     (False, (10, 0, 512), (20, 0, 512))]:
            memory = torch.rand(*memory_shape, device=device)
            tgt = torch.rand(*tgt_shape, requires_grad=True, device=device)
            decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=batch_first).to(device)
            transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6).to(device)
            self._test_module_empty_inputs(transformer_decoder, [tgt, memory])

    @expectedFailureMeta  # RuntimeError: cannot reshape tensor of 0 elements into shape [1, 0, -1]
    @onlyNativeDeviceTypes
    def test_Transformer_empty(self, device):
        for batch_first, src_shape, tgt_shape in [(True, (10, 0, 512), (20, 0, 512))]:
            transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12).to(device)
            src = torch.rand(*src_shape, requires_grad=True, device=device)
            tgt = torch.rand(*tgt_shape, requires_grad=True, device=device)
            self._test_module_empty_inputs(transformer_model, [src, tgt])

    @onlyNativeDeviceTypes
    @dtypes(torch.float32, torch.complex64)
    def test_ReflectionPad_empty(self, device, dtype):
        for mod, inp in [
                (torch.nn.ReflectionPad1d(2), torch.randn(0, 3, 10, device=device, dtype=dtype)),
                (torch.nn.ReflectionPad2d(2), torch.randn(0, 3, 10, 10, device=device, dtype=dtype)),
                (torch.nn.ReflectionPad3d(3), torch.randn(0, 3, 10, 10, 10, device=device, dtype=dtype))]:
            _test_module_empty_input(self, mod, inp, check_size=False)

        with self.assertRaisesRegex(RuntimeError, '2D or 3D'):
            mod = torch.nn.ReflectionPad1d(2)
            inp = torch.randn(3, 0, 10, device=device, dtype=dtype)
            mod(inp)

        with self.assertRaisesRegex(RuntimeError, '3D or 4D'):
            mod = torch.nn.ReflectionPad2d(2)
            inp = torch.randn(3, 0, 10, 10, device=device, dtype=dtype)
            mod(inp)

        with self.assertRaisesRegex(RuntimeError, '4D or 5D'):
            mod = torch.nn.ReflectionPad3d(3)
            inp = torch.randn(3, 0, 10, 10, 10, device=device, dtype=dtype)
            mod(inp)

    @onlyCUDA   # Test if CPU and GPU results match
    def test_ReflectionPad2d_large(self, device):
        shapes = ([2, 65736, 6, 6], [65736, 2, 6, 6])
        pad = (1, 2, 3, 4)
        for shape in shapes:
            x = torch.randn(shape, device=device, requires_grad=True)
            ref_x = x.detach().cpu().requires_grad_()

            out = F.pad(x, pad, mode='reflect')
            ref_out = F.pad(ref_x, pad, mode='reflect')

            self.assertEqual(out, ref_out)

            g = torch.randn_like(out)
            ref_g = g.cpu()

            out.backward(g)
            ref_out.backward(ref_g)

            self.assertEqual(x.grad, ref_x.grad)

    @onlyNativeDeviceTypes
    def test_LocalResponseNorm_empty(self, device):
        mod = torch.nn.LocalResponseNorm(2).to(device)
        inp = torch.ones(0, 5, 24, 24, device=device)
        _test_module_empty_input(self, mod, inp, check_size=False)

    @onlyCUDA   # Test if CPU and GPU results match
    def test_ReflectionPad3d_large(self, device):
        shapes = ([2, 1000, 7, 7, 7], [1000, 2, 7, 7, 7])
        pad = (1, 2, 3, 4, 5, 6)
        for shape in shapes:
            x = torch.randn(shape, device=device, requires_grad=True)
            ref_x = x.detach().cpu().requires_grad_()

            out = F.pad(x, pad, mode='reflect')
            ref_out = F.pad(ref_x, pad, mode='reflect')

            self.assertEqual(out, ref_out)

            g = torch.randn_like(out)
            ref_g = g.cpu()

            out.backward(g)
            ref_out.backward(ref_g)

            self.assertEqual(x.grad, ref_x.grad)

    @onlyNativeDeviceTypes
    @dtypes(torch.float, torch.double)
    def test_MarginLoss_empty(self, device, dtype):
        for mod, x, y in [
                (torch.nn.MultiMarginLoss().to(device),
                 torch.randn(0, 10, requires_grad=True, device=device, dtype=dtype),
                 torch.ones(0, device=device).type(torch.long)),
                (torch.nn.MultiLabelMarginLoss().to(device),
                 torch.randn(0, 10, requires_grad=True, device=device, dtype=dtype),
                 torch.ones(0, 10, device=device).type(torch.long))]:

            out = mod(x, y)
            out.sum().backward()

            self.assertEqual(x, torch.zeros_like(x))
            self.assertEqual(x.grad, torch.zeros_like(x))

            with self.assertRaisesRegex(RuntimeError, 'Expected'):
                x = torch.randn(0, requires_grad=True, device=device, dtype=dtype)
                y = torch.ones(10, device=device).type(torch.long)
                mod(x, y)

            with self.assertRaisesRegex(RuntimeError, 'Expected'):
                x = torch.randn(10, 0, requires_grad=True, device=device, dtype=dtype)
                y = torch.ones(10, 0, device=device).type(torch.long)
                mod(x, y)

    @onlyNativeDeviceTypes
    def test_Unfold_empty(self, device):
        inp = torch.randn(0, 3, 3, 4, device=device)
        unfold = torch.nn.Unfold(kernel_size=(2, 3)).to(device)
        _test_module_empty_input(self, unfold, inp, check_size=False)

        with self.assertRaisesRegex(RuntimeError, 'Expected 3D or 4D'):
            inp = torch.randn(3, 0, 3, 4, device=device)
            unfold = torch.nn.Unfold(kernel_size=(2, 3)).to(device)
            unfold(inp)

    @onlyCUDA
    @dtypes(torch.float, torch.double)
    @tf32_on_and_off(0.005)
    def test_rnn_fused(self, device, dtype):

        def copy_rnn(rnn1, rnn2):
            for x_layer, y_layer in zip(rnn1.all_weights, rnn2.all_weights):
                for x, y in zip(x_layer, y_layer):
                    x.data.copy_(y.data)

        def check_rnn_grads(rnn1, rnn2):
            for x_layer, y_layer in zip(rnn1.all_weights, rnn2.all_weights):
                for x, y in zip(x_layer, y_layer):
                    self.assertEqual(x.grad, y.grad, atol=5e-5, rtol=0)

        input_size = 10
        hidden_size = 6
        num_layers = 2
        seq_length = 7
        batch = 6
        input_val = torch.randn(seq_length, batch, input_size, dtype=dtype)
        grad_output = torch.randn(seq_length, batch, hidden_size, dtype=dtype)
        hx_val = torch.randn(num_layers, batch, hidden_size, dtype=dtype)
        grad_hy = torch.randn(num_layers, batch, hidden_size, dtype=dtype)
        with torch.backends.cudnn.flags(enabled=False, allow_tf32=None):
            for module in (nn.GRU, nn.LSTM):
                for bias in (True, False):
                    rnn = module(input_size, hidden_size, num_layers, bias=bias).to(dtype)
                    rnn_device = module(input_size, hidden_size, num_layers, bias=bias).to(device, dtype)
                    copy_rnn(rnn, rnn_device)

                    is_lstm = isinstance(rnn, nn.LSTM)
                    if is_lstm:
                        hx = (hx_val.clone().requires_grad_(True),
                              hx_val.clone().add(1).requires_grad_(True))
                        hx_device = (hx_val.clone().to(device).requires_grad_(True),
                                     hx_val.clone().to(device).add(1).requires_grad_(True))
                    else:
                        hx = hx_val.clone().requires_grad_(True)
                        hx_device = hx_val.clone().to(device).requires_grad_(True)

                    inp = input_val.clone().requires_grad_(True)
                    inp_cu = input_val.clone().to(device).requires_grad_(True)
                    output1, hy1 = rnn(inp, hx)
                    output2, hy2 = rnn_device(inp_cu, hx_device)
                    if is_lstm:
                        torch.autograd.backward(
                            [output1, hy1[0], hy1[1]], [grad_output, grad_hy, grad_hy + 1]
                        )
                        torch.autograd.backward(
                            [output2, hy2[0], hy2[1]],
                            [grad_output.to(device), grad_hy.to(device), (grad_hy + 1).to(device)]
                        )
                    else:
                        torch.autograd.backward([output1, hy1], [grad_output, grad_hy])
                        torch.autograd.backward([output2, hy2], [grad_output.to(device), grad_hy.to(device)])

                    self.assertEqual(output1, output2)
                    self.assertEqual(hy1, hy2)

                    check_rnn_grads(rnn, rnn_device)
                    self.assertEqual(inp.grad, inp_cu.grad)
                    if is_lstm:
                        self.assertEqual(hx[0].grad, hx_device[0].grad)
                        self.assertEqual(hx[1].grad, hx_device[1].grad)
                    else:
                        self.assertEqual(hx.grad, hx_device.grad)

    def test_BatchNorm_empty(self, device):
        mod = torch.nn.BatchNorm2d(3).to(device)
        inp = torch.randn(0, 3, 2, 2, device=device)
        _test_module_empty_input(self, mod, inp)
        if self.device_type == 'cuda' and self.has_cudnn():
            with torch.backends.cudnn.flags(enabled=False):
                _test_module_empty_input(self, mod, inp)

        self.assertEqual(mod.running_mean, torch.tensor([0., 0, 0], device=device))
        self.assertEqual(mod.running_var, torch.tensor([1., 1, 1], device=device))
        self.assertEqual(mod.weight.grad, torch.tensor([0., 0, 0], device=device))
        self.assertEqual(mod.bias.grad, torch.tensor([0., 0, 0], device=device))

    @onlyCUDA
    @largeTensorTest('16GB')
    def test_prelu_backward_32bit_indexing(self, device):
        m = torch.nn.PReLU().cuda().half()
        input_ = torch.ones((1024, 1024, 1024, 2), dtype=torch.half, device=device)
        output = m(input_)
        output.backward(input_)

    def test_linear_empty(self, device):
        mod = torch.nn.Linear(7, 7).to(device)
        inp = torch.randn(0, 7, device=device)
        _test_module_empty_input(self, mod, inp)

    def test_one_hot(self, device):
        if self.device_type != 'cuda':  # cuda throws device assert for invalid data
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
        expected = torch.empty([4, 0, 100], dtype=torch.long)
        self.assertEqual(t, expected)

        with self.assertRaises(RuntimeError):
            torch.nn.functional.one_hot(torch.empty([4, 0], dtype=torch.long, device=device))

        with self.assertRaises(RuntimeError):
            torch.nn.functional.one_hot(torch.tensor([3, 4, 1, 0], device=device), -2)

    def test_nn_empty(self, device):
        # One off tests to ensure scalars from nn.yaml are properly applied
        def verify_scalars(input, output):
            self.assertEqual(input.shape, output.shape)
            self.assertEqual(0, output.numel())

        for input_shape in [(0), (0, 2)]:
            for module in [torch.nn.ELU, torch.nn.Hardtanh, torch.nn.LeakyReLU, torch.nn.LogSigmoid,
                           torch.nn.RReLU, torch.nn.Softshrink, torch.nn.Softplus, torch.nn.Sigmoid,
                           torch.nn.Tanh]:
                input = torch.randn(input_shape, device=device, requires_grad=True)
                m = module()
                output = m(input)
                verify_scalars(input, output)

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

    # verify that bogus reduction strings are errors
    @onlyNativeDeviceTypes
    def test_invalid_reduction_strings(self, device):
        input = torch.randn(3, 5, requires_grad=True, device=device)
        cinput = torch.randn(3, 5, requires_grad=True, device=device, dtype=torch.cfloat)
        target = torch.tensor([1, 0, 4], device=device)
        var = torch.ones(size=input.size(), requires_grad=True, device=device)

        for reduction in ['none', 'invalid']:
            def v(fn):
                if reduction == 'invalid':
                    self.assertRaises(ValueError, lambda: fn())
                else:
                    fn()

            v(lambda: F.nll_loss(input, target, reduction=reduction))
            v(lambda: F.cross_entropy(input, target, reduction=reduction))
            v(lambda: F.multi_margin_loss(input, target, reduction=reduction))

            v(lambda: F.kl_div(input, input, reduction=reduction))
            v(lambda: F.huber_loss(input, input, reduction=reduction))
            v(lambda: F.smooth_l1_loss(input, input, reduction=reduction))
            v(lambda: F.l1_loss(input, input, reduction=reduction))
            v(lambda: F.l1_loss(cinput, cinput, reduction=reduction))
            v(lambda: F.mse_loss(input, input, reduction=reduction))
            v(lambda: F.hinge_embedding_loss(input, input, reduction=reduction))
            v(lambda: F.poisson_nll_loss(input, input, reduction=reduction))
            v(lambda: F.gaussian_nll_loss(input, input, var, reduction=reduction))
            v(lambda: F.binary_cross_entropy(torch.sigmoid(input), input, reduction=reduction))
            v(lambda: F.binary_cross_entropy_with_logits(input, input, reduction=reduction))

            zeros = torch.zeros_like(input).to(torch.int64)
            v(lambda: F.multilabel_soft_margin_loss(input, zeros, reduction=reduction))
            v(lambda: F.multilabel_margin_loss(input, zeros, reduction=reduction))

            v(lambda: F.triplet_margin_loss(input, input, input, reduction=reduction))
            v(lambda: F.triplet_margin_with_distance_loss(input, input, input, reduction=reduction))
            v(lambda: F.margin_ranking_loss(input, input, input.sign(), reduction=reduction))
            v(lambda: F.cosine_embedding_loss(input, input, input[:, 0].sign(), reduction=reduction))

            log_probs = torch.randn(50, 16, 20, requires_grad=True, device=device).log_softmax(2)
            targets = torch.randint(1, 20, (16, 30), dtype=torch.long, device=device)
            input_lengths = torch.full((16,), 50, dtype=torch.long, device=device)
            target_lengths = torch.randint(10, 30, (16,), dtype=torch.long, device=device)
            v(lambda: F.ctc_loss(log_probs, targets, input_lengths, target_lengths, reduction=reduction))

            # FIXME: should we allow derivatives on these?
            v(lambda: F.soft_margin_loss(input, input.sign().detach(), reduction=reduction))

    @onlyNativeDeviceTypes
    def test_smooth_l1_loss_vs_huber_loss(self, device):
        def _make_test_tensor(shape, contiguous=True):
            if contiguous:
                test_tensor = torch.randn(shape, device=device)
            else:
                # Select every other element in the innermost dimension to
                # make it non-contiguous.
                doubled_shape = list(shape)
                doubled_shape[-1] *= 2
                test_tensor = torch.randn(doubled_shape, device=device)
                test_tensor = test_tensor[..., ::2]
            return test_tensor

        def _test_smooth_l1_loss_vs_huber_loss_helper(input, target, beta, require_equal):
            for reduction in ['mean', 'sum', 'none']:
                smooth_l1 = torch.nn.SmoothL1Loss(beta=beta, reduction=reduction)
                # beta hyper-parameter is called delta for Huber
                huber = torch.nn.HuberLoss(delta=beta, reduction=reduction)
                smooth_l1_loss = smooth_l1(input, target)
                huber_loss = huber(input, target)

                if require_equal:
                    self.assertEqual(smooth_l1_loss, huber_loss)
                else:
                    # Huber loss should be larger than smooth L1 loss by a factor of beta.
                    self.assertEqual(smooth_l1_loss * beta, huber_loss)

        def _test_smooth_l1_loss_vs_huber_loss_multi_input_helper(beta, require_equal):
            # Test the non-vectorized case.
            shape = (2, 2)
            _test_smooth_l1_loss_vs_huber_loss_helper(input=_make_test_tensor(shape),
                                                      target=_make_test_tensor(shape),
                                                      beta=beta,
                                                      require_equal=require_equal)

            # Test the vectorized case (innermost dim > 32).
            shape = (64, 64)
            _test_smooth_l1_loss_vs_huber_loss_helper(input=_make_test_tensor(shape),
                                                      target=_make_test_tensor(shape),
                                                      beta=beta,
                                                      require_equal=require_equal)

            # Test the non-contiguous case.
            _test_smooth_l1_loss_vs_huber_loss_helper(input=_make_test_tensor(shape, contiguous=False),
                                                      target=_make_test_tensor(shape, contiguous=False),
                                                      beta=beta,
                                                      require_equal=require_equal)

        def test_equal_when_beta_is_one():
            _test_smooth_l1_loss_vs_huber_loss_multi_input_helper(beta=1.0, require_equal=True)

        def test_unequal_when_beta_is_less_than_one():
            _test_smooth_l1_loss_vs_huber_loss_multi_input_helper(beta=0.5, require_equal=False)

        def test_unequal_when_beta_is_greater_than_one():
            _test_smooth_l1_loss_vs_huber_loss_multi_input_helper(beta=1.5, require_equal=False)

        test_equal_when_beta_is_one()
        test_unequal_when_beta_is_less_than_one()
        test_unequal_when_beta_is_greater_than_one()

    @onlyCPU
    def test_smooth_l1_loss_bfloat16(self, device):
        def test_dtype(fn, input, target, dtype):
            input = input.detach().clone().to(dtype=dtype).requires_grad_(True)
            input2 = input.detach().clone().float().requires_grad_(True)
            target = target.detach().clone().to(dtype=dtype)
            target2 = target.detach().clone().float()
            out = fn(input, target)
            out.sum().backward()
            out2 = fn(input2, target2)
            out2.sum().backward()
            self.assertEqual(out.dtype, dtype)
            self.assertEqual(input.grad.dtype, dtype)
            self.assertEqual(out, out2, exact_dtype=False)
            self.assertEqual(input.grad, input2.grad, exact_dtype=False)

        def func(device):
            return nn.SmoothL1Loss().to(device=device)

        shapes = [[1, 3, 1, 6], [1, 3, 1, 128], [1, 3, 128, 128]]
        for shape in shapes:
            x = torch.randn(shape, device=device, requires_grad=True)
            t = torch.randn(shape, device=device)
            test_dtype(func(device), x, t, torch.bfloat16)

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

    def test_upsamplingNearest1d(self, device):
        # Forward AD does not support XLA because XLA tensors don't have storage
        check_forward_ad = torch.device(device).type != 'xla'

        def helper(mode):
            m = nn.Upsample(size=4, mode=mode)
            in_t = torch.ones(1, 1, 2, device=device)
            in_uint8_t = torch.ones(1, 1, 2, dtype=torch.uint8, device=device)
            with warnings.catch_warnings(record=True) as w:
                out_t = m(in_t)
                out_uint8_t = m(in_uint8_t)
            self.assertEqual(torch.ones(1, 1, 4, device=device), out_t.data)
            self.assertEqual(torch.ones(1, 1, 4, dtype=torch.uint8, device=device), out_uint8_t.data)

            # Checks upsampling
            input = torch.randn(1, 1, 2, requires_grad=True, device=device)
            gradcheck(lambda x: F.interpolate(x, 4, mode=mode), [input], check_forward_ad=check_forward_ad)
            gradgradcheck(lambda x: F.interpolate(x, 4, mode=mode), [input], check_fwd_over_rev=check_forward_ad)

            # Checks downsampling
            input = torch.randn(1, 1, 20, requires_grad=True, device=device)
            gradcheck(lambda x: F.interpolate(x, 11, mode=mode), [input], check_forward_ad=check_forward_ad)
            gradgradcheck(lambda x: F.interpolate(x, 4, mode=mode), [input], check_fwd_over_rev=check_forward_ad)

            # consistency CUDA/CPU check
            if torch.device(device).type == 'cuda':
                input_cuda = torch.randn(1, 1, 20, device=device)
                input_cpu = input_cuda.cpu()
                output_cuda = F.interpolate(input_cuda, 4, mode=mode)
                output_cpu = F.interpolate(input_cpu, 4, mode=mode)
                self.assertEqual(output_cuda.cpu(), output_cpu)

                output_cuda = F.interpolate(input_cuda, 24, mode=mode)
                output_cpu = F.interpolate(input_cpu, 24, mode=mode)
                self.assertEqual(output_cuda.cpu(), output_cpu)

        helper("nearest")
        helper("nearest-exact")

    def test_upsamplingNearest1d_correctness(self, device):
        # Here we check if output matches OpenCV's INTER_NEAREST-like result
        def helper(isize, osize):
            in_t = torch.arange(isize, dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)
            out_t = F.interpolate(
                in_t, size=(osize, ), recompute_scale_factor=False, mode="nearest"
            )
            # compute expected output as OpenCV
            expected_out = torch.zeros(osize, dtype=torch.float).unsqueeze(0).unsqueeze(0)
            scale = 1.0 * isize / osize
            for o in range(osize):
                i_f32 = o * scale
                i = int(i_f32)
                expected_out[0, 0, o] = in_t[0, 0, i]
            expected_out = expected_out.to(device=device)
            self.assertEqual(out_t, expected_out)

        helper(20, 11)
        helper(10, 15)

    def test_upsamplingNearestExact1d_rescale(self, device):
        # Checks https://github.com/pytorch/pytorch/issues/62237
        isize = 20
        in_t = torch.arange(isize, dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)
        # for s in [1.00001, 0.99999]:  # 0.9999 case is broken
        # See issue: https://github.com/pytorch/pytorch/issues/62396
        for s in [1.00001, ]:
            out_t = F.interpolate(
                in_t, scale_factor=s, recompute_scale_factor=False, mode="nearest-exact"
            )
            expected_out = in_t
            self.assertEqual(out_t, expected_out, msg=f"scale: {s}")

        # checks data duplication if output_size == 2 * input_size
        # for s in [2.00001, 1.99999]:  # 1.99999 case is broken
        # See issue: https://github.com/pytorch/pytorch/issues/62396
        for s in [2.00001, ]:
            out_t = F.interpolate(
                in_t, scale_factor=s, recompute_scale_factor=False, mode="nearest-exact"
            )
            # input is [[[0, 1, 2, 3, ..., 9]]]
            # expected out is [[[0, 0, 1, 1, 2, 2, ..., 9, 9]]]
            expected_out = in_t.repeat_interleave(2, dim=-1)
            self.assertEqual(out_t, expected_out)

    def test_upsamplingNearestExact1d_correctness(self, device):
        # Here we check if output matches Scikit-Image/Scipy-like result
        # Checks https://github.com/pytorch/pytorch/issues/34808
        def helper(isize, osize):
            in_t = torch.arange(isize, dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)
            out_t = F.interpolate(
                in_t, size=(osize, ), recompute_scale_factor=False, mode="nearest-exact"
            )
            # compute expected output as scikit-image/scipy
            expected_out = torch.zeros(osize, dtype=torch.float).unsqueeze(0).unsqueeze(0)
            scale = 1.0 * isize / osize
            for o in range(osize):
                i_f32 = (o + 0.5) * scale
                i = int(i_f32)
                expected_out[0, 0, o] = in_t[0, 0, i]
            expected_out = expected_out.to(device=device)
            self.assertEqual(out_t, expected_out)

        helper(20, 11)
        helper(10, 15)

    def test_upsamplingNearest2d(self, device):
        # Forward AD does not support XLA because XLA tensors don't have storage
        check_forward_ad = torch.device(device).type != 'xla'

        def helper(memory_format, mode):
            in_t = torch.ones(1, 2, 2, 2, device=device).contiguous(memory_format=memory_format)
            in_uint8_t = torch.ones(1, 2, 2, 2, dtype=torch.uint8, device=device).contiguous(memory_format=memory_format)
            with warnings.catch_warnings(record=True) as w:
                out_t = F.interpolate(in_t, size=4, mode=mode)
                out_uint8_t = F.interpolate(in_uint8_t, size=4, mode=mode)
                self.assertEqual(len(w), 0)
            self.assertEqual(torch.ones(1, 2, 4, 4, device=device), out_t)
            self.assertEqual(torch.ones(1, 2, 4, 4, dtype=torch.uint8, device=device), out_uint8_t)
            # Assert that memory format is carried through to the output
            self.assertTrue(out_t.is_contiguous(memory_format=memory_format))

            # test forward when input's height is not same as width
            in_t = torch.ones(1, 2, 2, 1, device=device).contiguous(memory_format=memory_format).requires_grad_()
            out_t = F.interpolate(in_t, size=(4, 2), mode=mode)
            self.assertEqual(torch.ones(1, 2, 4, 2, device=device), out_t)
            self.assertTrue(out_t.is_contiguous(memory_format=memory_format))

            out_t.backward(torch.randn_like(out_t))
            self.assertTrue(in_t.grad.is_contiguous(memory_format=memory_format))

            # test backward when input's height is not same as width
            input = torch.ones(1, 2, 2, 1, requires_grad=True, device=device).contiguous(memory_format=memory_format)
            gradcheck(lambda x: F.interpolate(x, size=(4, 2), mode=mode), [input], check_forward_ad=check_forward_ad)
            gradgradcheck(lambda x: F.interpolate(x, size=(4, 2), mode=mode), [input], check_fwd_over_rev=check_forward_ad)

            input = torch.randn(1, 2, 2, 2, requires_grad=True, device=device).contiguous(memory_format=memory_format)
            self.assertEqual(
                F.interpolate(input, 4, mode=mode),
                F.interpolate(input, scale_factor=2, mode=mode))
            gradcheck(lambda x: F.interpolate(x, 4, mode=mode), [input], check_forward_ad=check_forward_ad)
            gradgradcheck(lambda x: F.interpolate(x, 4, mode=mode), [input], check_fwd_over_rev=check_forward_ad)

            # Assert that cpu and cuda handle channels_last memory format in the same way
            # https://github.com/pytorch/pytorch/issues/54590
            if torch.device(device).type == 'cuda':
                for shapes, scale_factor in product([
                    (2, 2, 3, 4), (2, 3, 4, 5), (3, 1, 2, 2), (1, 5, 3, 2)
                ], [0.5, 1.5, 2]):
                    a_cuda = torch.randn(*shapes, device=device).contiguous(memory_format=memory_format).requires_grad_()
                    a_cpu = a_cuda.detach().cpu().requires_grad_()

                    out_cuda = F.interpolate(a_cuda, scale_factor=scale_factor, mode=mode)
                    out_cpu = F.interpolate(a_cpu, scale_factor=scale_factor, mode=mode)

                    self.assertEqual(out_cpu.cuda(), out_cuda)

                    g_cuda = torch.randn_like(out_cuda)
                    g_cpu = g_cuda.cpu()

                    out_cuda.backward(g_cuda)
                    out_cpu.backward(g_cpu)

                    self.assertEqual(a_cuda.grad, a_cpu.grad)

        helper(torch.contiguous_format, "nearest")
        helper(torch.channels_last, "nearest")
        # Uncomment below once F.interpolate is updated
        helper(torch.contiguous_format, "nearest-exact")
        helper(torch.channels_last, "nearest-exact")

    def test_upsamplingNearest2d_correctness(self, device):
        # Here we check if output matches OpenCV's INTER_NEAREST-like result
        def helper(memory_format, isize, osize):
            in_t = torch.arange(isize * isize, dtype=torch.float, device=device).reshape(1, 1, isize, isize)
            in_t = in_t.contiguous(memory_format=memory_format)
            out_t = F.interpolate(
                in_t, size=(osize, osize), recompute_scale_factor=False, mode="nearest"
            )
            # compute expected output as OpenCV
            expected_out = torch.zeros(1, 1, osize, osize, dtype=torch.float)
            scale = 1.0 * isize / osize
            for o1 in range(osize):
                i1_f32 = o1 * scale
                i1 = int(i1_f32)
                for o2 in range(osize):
                    i2_f32 = o2 * scale
                    i2 = int(i2_f32)
                    expected_out[0, 0, o1, o2] = in_t[0, 0, i1, i2]
            expected_out = expected_out.to(device=device)
            self.assertEqual(out_t, expected_out)

        helper(torch.contiguous_format, 20, 11)
        helper(torch.channels_last, 20, 11)
        helper(torch.contiguous_format, 10, 15)
        helper(torch.channels_last, 10, 15)

    def test_upsamplingNearestExact2d_correctness(self, device):
        # Here we check if output matches Scikit-Image/Scipy-like result
        # Checks https://github.com/pytorch/pytorch/issues/34808
        def helper(memory_format, isize, osize):
            in_t = torch.arange(isize * isize, dtype=torch.float, device=device).reshape(1, 1, isize, isize)
            in_t = in_t.contiguous(memory_format=memory_format)
            out_t = F.interpolate(
                in_t, size=(osize, osize), recompute_scale_factor=False, mode="nearest-exact"
            )
            # compute expected output as Scikit-Image/Scipy
            expected_out = torch.zeros(1, 1, osize, osize, dtype=torch.float)
            scale = 1.0 * isize / osize
            for o1 in range(osize):
                i1_f32 = (o1 + 0.5) * scale
                i1 = int(i1_f32)
                for o2 in range(osize):
                    i2_f32 = (o2 + 0.5) * scale
                    i2 = int(i2_f32)
                    expected_out[0, 0, o1, o2] = in_t[0, 0, i1, i2]
            expected_out = expected_out.to(device=device)
            self.assertEqual(out_t, expected_out)

        helper(torch.contiguous_format, 20, 11)
        helper(torch.channels_last, 20, 11)
        helper(torch.contiguous_format, 10, 15)
        helper(torch.channels_last, 10, 15)

    def test_upsamplingNearest3d(self, device):
        # Forward AD does not support XLA because XLA tensors don't have storage
        check_forward_ad = torch.device(device).type != 'xla'

        def helper(memory_format, mode):
            m = nn.Upsample(size=4, mode=mode)
            in_t = torch.ones(1, 2, 2, 2, 2, device=device).contiguous(memory_format=memory_format)
            in_uint8_t = torch.ones(
                1, 2, 2, 2, 2, dtype=torch.uint8, device=device
            ).contiguous(memory_format=memory_format)
            with warnings.catch_warnings(record=True) as w:
                out_t = m(in_t)
                out_uint8_t = m(in_uint8_t)
            expected_output = torch.ones(1, 2, 4, 4, 4, device=device)
            self.assertEqual(expected_output, out_t)
            self.assertEqual(expected_output.to(torch.uint8), out_uint8_t)
            # Assert that memory format is carried through to the output
            self.assertTrue(out_t.is_contiguous(memory_format=memory_format))

            input = torch.randn(
                1, 2, 2, 2, 2, requires_grad=True, device=device
            ).contiguous(memory_format=memory_format)
            gradcheck(lambda x: F.interpolate(x, 4, mode=mode), [input], check_forward_ad=check_forward_ad)
            gradgradcheck(lambda x: F.interpolate(x, 4, mode=mode), [input], check_fwd_over_rev=check_forward_ad)

            # Assert that cpu and cuda handle channels_last memory format in the same way
            # https://github.com/pytorch/pytorch/issues/54590
            if torch.device(device).type == 'cuda':
                a = torch.ones(
                    2, 2, 2, 3, 4, device=device, requires_grad=True
                ).contiguous(memory_format=torch.channels_last_3d)
                # make the data asymmetric; ensure that cuda/cpu handle channels_last appropriately.
                a[1][1][1][2][2] = a[1][1][1][2][3] = 0

                out_cuda = torch.nn.functional.interpolate(a, scale_factor=2, mode=mode)
                out_cpu = torch.nn.functional.interpolate(a.to('cpu'), scale_factor=2, mode=mode)
                self.assertEqual(out_cpu, out_cuda.to('cpu'))

                gradcheck(lambda x: F.interpolate(x, 4, mode=mode), [a], check_forward_ad=check_forward_ad)
                gradgradcheck(lambda x: F.interpolate(x, 4, mode=mode), [a], check_fwd_over_rev=check_forward_ad)

                gradcheck(lambda x: F.interpolate(x, 4, mode=mode), [a.to('cuda')], check_forward_ad=check_forward_ad)
                gradgradcheck(lambda x: F.interpolate(x, 4, mode=mode), [a.to('cuda')], check_fwd_over_rev=check_forward_ad)

        helper(torch.contiguous_format, "nearest")
        helper(torch.channels_last_3d, "nearest")
        helper(torch.contiguous_format, "nearest-exact")
        helper(torch.channels_last_3d, "nearest-exact")

    def test_upsamplingNearest3d_correctness(self, device):
        # Here we check if output matches OpenCV's INTER_NEAREST-like result
        def helper(memory_format, isize, osize):
            in_t = torch.arange(isize * isize * isize, dtype=torch.float, device=device)
            in_t = in_t.reshape(1, 1, isize, isize, isize)
            in_t = in_t.contiguous(memory_format=memory_format)
            out_t = F.interpolate(
                in_t, size=(osize, osize, osize), recompute_scale_factor=False, mode="nearest"
            )
            # compute expected output as OpenCV
            expected_out = torch.zeros(1, 1, osize, osize, osize, dtype=torch.float)
            scale = 1.0 * isize / osize
            for o1 in range(osize):
                i1_f32 = o1 * scale
                i1 = int(i1_f32)
                for o2 in range(osize):
                    i2_f32 = o2 * scale
                    i2 = int(i2_f32)
                    for o3 in range(osize):
                        i3_f32 = o3 * scale
                        i3 = int(i3_f32)
                        expected_out[0, 0, o1, o2, o3] = in_t[0, 0, i1, i2, i3]
            expected_out = expected_out.to(device=device)
            self.assertEqual(out_t, expected_out)

        helper(torch.contiguous_format, 20, 11)
        helper(torch.channels_last_3d, 20, 11)
        helper(torch.contiguous_format, 10, 15)
        helper(torch.channels_last_3d, 10, 15)

    def test_upsamplingNearestExact3d_correctness(self, device):
        # Here we check if output matches Scikit-Image/Scipy-like result
        # Checks https://github.com/pytorch/pytorch/issues/34808
        def helper(memory_format, isize, osize):
            in_t = torch.arange(isize * isize * isize, dtype=torch.float, device=device)
            in_t = in_t.reshape(1, 1, isize, isize, isize)
            in_t = in_t.contiguous(memory_format=memory_format)
            out_t = F.interpolate(
                in_t, size=(osize, osize, osize), recompute_scale_factor=False, mode="nearest-exact"
            )
            # compute expected output as Scikit-Image/Scipy
            expected_out = torch.zeros(1, 1, osize, osize, osize, dtype=torch.float)
            scale = 1.0 * isize / osize
            for o1 in range(osize):
                i1_f32 = (o1 + 0.5) * scale
                i1 = int(i1_f32)
                for o2 in range(osize):
                    i2_f32 = (o2 + 0.5) * scale
                    i2 = int(i2_f32)
                    for o3 in range(osize):
                        i3_f32 = (o3 + 0.5) * scale
                        i3 = int(i3_f32)
                        expected_out[0, 0, o1, o2, o3] = in_t[0, 0, i1, i2, i3]
            expected_out = expected_out.to(device=device)
            self.assertEqual(out_t, expected_out)

        helper(torch.contiguous_format, 20, 11)
        helper(torch.channels_last_3d, 20, 11)
        helper(torch.contiguous_format, 10, 15)
        helper(torch.channels_last_3d, 10, 15)

    @parametrize_test("antialias", [True, False])
    @parametrize_test("align_corners", [True, False])
    def test_upsamplingBilinear2d(self, device, antialias, align_corners):
        # Forward AD does not support XLA because XLA tensors don't have storage
        check_forward_ad = torch.device(device).type != 'xla'

        kwargs = dict(mode='bilinear', align_corners=align_corners, antialias=antialias)
        for memory_format in [torch.contiguous_format, torch.channels_last]:
            # test float scale factor up & downsampling
            for scale_factor in [0.5, 1.5, 2]:
                in_t = torch.ones(2, 3, 8, 8, device=device).contiguous(memory_format=memory_format).requires_grad_()
                out_size = int(math.floor(in_t.shape[-1] * scale_factor))
                with warnings.catch_warnings(record=True) as w:
                    out_t = F.interpolate(in_t, scale_factor=scale_factor, **kwargs)
                self.assertEqual(torch.ones(2, 3, out_size, out_size, device=device), out_t.data)
                # Assert that memory format is carried through to the output
                self.assertTrue(out_t.is_contiguous(memory_format=memory_format))
                out_t.backward(torch.randn_like(out_t))
                self.assertTrue(in_t.grad.is_contiguous(memory_format=memory_format))

                if torch.device(device).type == 'cuda':
                    # Bilinear backward is nondeterministic because of atomicAdd usage
                    nondet_tol = 1e-5
                else:
                    nondet_tol = 0.0

                input = torch.randn(2, 3, 8, 8, device=device).contiguous(memory_format=memory_format).requires_grad_()
                gradcheck(
                    lambda x: F.interpolate(x, out_size, **kwargs),
                    [input],
                    check_forward_ad=check_forward_ad, nondet_tol=nondet_tol
                )
                gradgradcheck(
                    lambda x: F.interpolate(x, out_size, **kwargs),
                    [input],
                    check_fwd_over_rev=check_forward_ad, nondet_tol=nondet_tol
                )

                # Assert that cpu and cuda give same results
                if torch.device(device).type == 'cuda':
                    for shapes in [
                        (2, 2, 3, 4), (2, 3, 4, 5), (3, 1, 2, 2), (1, 5, 3, 2)
                    ]:
                        a_cuda = torch.randn(
                            *shapes, device=device
                        ).contiguous(memory_format=memory_format).requires_grad_()
                        a_cpu = a_cuda.detach().cpu().requires_grad_()

                        with warnings.catch_warnings(record=True):
                            out_cuda = F.interpolate(a_cuda, scale_factor=scale_factor, **kwargs)
                            out_cpu = F.interpolate(a_cpu, scale_factor=scale_factor, **kwargs)

                        self.assertEqual(out_cpu, out_cuda.cpu())

                        g_cuda = torch.randn_like(out_cuda)
                        g_cpu = g_cuda.cpu()

                        out_cuda.backward(g_cuda)
                        out_cpu.backward(g_cpu)

                        self.assertEqual(a_cuda.grad, a_cpu.grad)

    @parametrize_test("memory_format", [torch.contiguous_format, torch.channels_last])
    def test_upsamplingBilinear2d_aa_correctness(self, device, memory_format):
        t_in = torch.arange(3 * 8 * 8, dtype=torch.float, device=device).reshape(1, 3, 8, 8)
        t_in = t_in.contiguous(memory_format=memory_format)
        # This expected result is obtain using PIL.Image.resize
        # for c in range(3):
        #   a_in = t_in.numpy()[0, c, ...]
        #   pil_in = Image.fromarray(a_in)
        #   pil_out = pil_in.resize((2, 2), resample=Image.LINEAR)
        expected_out = torch.tensor([
            17.035713, 20.25, 42.75, 45.964287, 81.03572, 84.25,
            106.75, 109.96428, 145.0357, 148.25, 170.75, 173.9643
        ], device=device, dtype=t_in.dtype).reshape(1, 3, 2, 2)
        t_out = F.interpolate(t_in, size=(2, 2), mode="bilinear", align_corners=False, antialias=True)
        self.assertEqual(expected_out, t_out)

    @parametrize_test("antialias", [True, False])
    @parametrize_test("align_corners", [True, False])
    def test_upsamplingBicubic2d(self, device, antialias, align_corners):
        kwargs = dict(mode='bicubic', align_corners=align_corners, antialias=antialias)
        # test float scale factor up & downsampling
        # for scale_factor in [0.5, 1, 1.5, 2]:
        for scale_factor in [2, ]:
            in_t = torch.ones(2, 3, 8, 8, device=device)
            out_t = F.interpolate(in_t, scale_factor=scale_factor, **kwargs)
            out_size = int(math.floor(in_t.shape[-1] * scale_factor))
            expected_out = torch.ones(2, 3, out_size, out_size, device=device)
            self.assertEqual(expected_out, out_t, atol=1e-5, rtol=0)

            if torch.device(device).type == 'cuda':
                # Bicubic backward is nondeterministic because of atomicAdd usage
                nondet_tol = 1e-5
            else:
                nondet_tol = 0.0
            inpt = torch.ones(2, 3, 8, 8, requires_grad=True, device=device)
            gradcheck(lambda x: F.interpolate(x, out_size, **kwargs), [inpt], nondet_tol=nondet_tol)

    def test_upsamplingBicubic2d_correctness(self, device):
        # test output against known input: align_corners=False result must match opencv
        in_t = torch.arange(8., device=device).view(1, 2, 2, 2)
        expected_out_t = torch.tensor(
            [[[[-0.31641, 0.01562, 0.56250, 0.89453],
              [0.34766, 0.67969, 1.22656, 1.55859],
              [1.44141, 1.77344, 2.32031, 2.65234],
              [2.10547, 2.43750, 2.98438, 3.31641]],

             [[3.68359, 4.01562, 4.56250, 4.89453],
              [4.34766, 4.67969, 5.22656, 5.55859],
              [5.44141, 5.77344, 6.32031, 6.65234],
              [6.10547, 6.43750, 6.98438, 7.31641]]]], device=device)
        out_t = F.interpolate(in_t, scale_factor=2, mode='bicubic', align_corners=False)
        torch.set_printoptions(precision=5)
        self.assertEqual(out_t, expected_out_t, atol=1e-5, rtol=0)

    @parametrize_test("memory_format", [torch.contiguous_format, torch.channels_last])
    def test_upsamplingBicubic2d_aa_correctness(self, device, memory_format):
        t_in = torch.arange(3 * 8 * 8, dtype=torch.float, device=device).reshape(1, 3, 8, 8)
        t_in = t_in.contiguous(memory_format=memory_format)
        # This expected result is obtain using PIL.Image.resize
        # for c in range(3):
        #   a_in = t_in.numpy()[0, c, ...]
        #   pil_in = Image.fromarray(a_in)
        #   pil_out = pil_in.resize((2, 2), resample=Image.BICUBIC)
        expected_out = torch.tensor([
            15.1205635, 18.760439, 44.23956, 47.879436, 79.12056, 82.76044,
            108.23956, 111.87944, 143.12057, 146.76044, 172.23956, 175.87943
        ], device=device, dtype=t_in.dtype).reshape(1, 3, 2, 2)
        t_out = F.interpolate(t_in, size=(2, 2), mode="bicubic", align_corners=False, antialias=True)
        self.assertEqual(expected_out, t_out)

    @onlyCUDA
    @dtypes(torch.half)
    @largeTensorTest('40GB')
    def test_upsampling_64bit_indexing_channels_last(self, device, dtype):
        x = torch.rand((32, 64, 512, 512), dtype=dtype, device=device)
        out = torch.nn.functional.interpolate(x.to(memory_format=torch.channels_last), scale_factor=2, mode='nearest')
        out_ref = torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        del x
        self.assertTrue(torch.allclose(out, out_ref))

    def _slow_masked_softmax(self, input, mask):
        exp = torch.exp(input)
        exp = exp * mask
        s = exp.sum(dim=3, keepdim=True).expand(exp.size())
        return exp / s

    def test_masked_softmax_mask_types_0_1(self, device):
        # Test that mask type 0 (LxL attention mask) and mask type 1 (BxL padding mask)
        # are processed correctly on the fast path and the results match explicit slow
        # calculation.
        sizes = [(1, 1, 32), (3, 16, 310), (12, 4, 1024), (4, 2, 1200)]

        for (B, num_heads, L) in sizes:

            # mask_type == 0 => attention mask of shape LxL
            src_mask_orig = torch.randint(0, 2, (L, L)).bool()
            src_mask = src_mask_orig.reshape(1, 1, L, L).expand(B, num_heads, L, L).bool()

            # mask_type == 1 => padding mask of shape BxL
            src_key_padding_mask_orig = torch.randint(0, 2, (B, L)).bool()
            src_key_padding_mask = src_key_padding_mask_orig.reshape(B, 1, 1, L).expand(B, num_heads, L, L).bool()

            masks = [(src_mask_orig, src_mask, 0), (src_key_padding_mask_orig, src_key_padding_mask, 1)]
            for dim in [0, 3]:
                for mask_orig, mask, mask_type in masks:
                    if (self.device_type == "cuda") and (num_heads % 2) and (mask_type == 1):
                        # CUDA path doesn't support padding mask when the number of heads is odd
                        continue
                    input = torch.randn((B, num_heads, L, L))
                    if (self.device_type == "cuda"):
                        input = input.cuda()
                        mask = mask.cuda()
                        mask_orig = mask_orig.cuda()
                    native_res = torch._masked_softmax(input, mask_orig, dim, mask_type)
                    mask = ~mask

                    def slow_masked_softmax(input, mask):
                        exp = torch.exp(input)
                        exp = exp * mask
                        s = exp.sum(dim=dim, keepdim=True).expand(exp.size())
                        return exp / s

                    pt_res = slow_masked_softmax(input, mask)
                    pt_res = torch.nan_to_num(pt_res)

                    mask_not = mask.logical_not()
                    # In result, should only fill the entirely masked out rows since those are non-deterministic (*may* be 0)
                    # Converts rows with all True's to False
                    mask_out = mask_not.all(dim, keepdim=True).expand(mask_not.shape)
                    self.assertEqual(
                        pt_res.masked_fill(mask_out, 0),
                        native_res.masked_fill(mask_out, 0),
                        exact_dtype=True
                    )

    @onlyCUDA
    def test_masked_softmax_devices_parity(self):
        # Test that softmax with mask type 0 (LxL attention mask) and mask type 1 (BxL padding mask)
        # gives the same result on CPU and on CUDA

        sizes = [(1, 1, 32), (3, 16, 310), (12, 4, 1024), (4, 2, 1200)]
        for (B, num_heads, L) in sizes:
            # mask_type == 0 => attention mask of shape LxL
            src_mask = torch.randint(0, 2, (L, L)).bool()
            # mask_type == 1 => padding mask of shape BxL
            src_key_padding_mask = torch.randint(0, 2, (B, L)).bool()
            masks = [(src_mask, 0), (src_key_padding_mask, 1)]
            input = torch.randn((B, num_heads, L, L))
            for dim in [0, 3]:
                for mask, mask_type in masks:
                    if (num_heads % 2) and (mask_type == 1):
                        # CUDA path doesn't support padding mask when the number of heads is odd
                        continue

                    def softmax_on_device(mask, input, device):
                        # Compute softmax on a given device
                        input_device = input.to(device)
                        mask_device = mask.to(device)
                        softmax_res = torch._masked_softmax(input_device, mask_device, dim, mask_type)
                        if mask_type == 0:
                            mask_expanded = mask_device.reshape(1, 1, L, L).expand(B, num_heads, L, L).bool()
                        else:
                            mask_expanded = mask_device.reshape(B, 1, 1, L).expand(B, num_heads, L, L).bool()
                        # In result, should only fill the entirely masked out rows since those are non-deterministic (*may* be 0)
                        # Fill rows with all True's with 0
                        mask_out = mask_expanded.all(dim, keepdim=True).expand(mask_expanded.shape)
                        softmax_res = softmax_res.masked_fill(mask_out, 0)
                        return softmax_res

                    cpu_res = softmax_on_device(mask, input, "cpu")
                    cuda_res = softmax_on_device(mask, input, "cuda")
                    self.assertEqual(cpu_res, cuda_res, exact_dtype=True)

    def test_masked_softmax(self, device):
        sizes = [(1, 1, 32), (3, 16, 310), (12, 4, 1024), (4, 2, 1200)]
        for (B, num_heads, L) in sizes:
            for dim in [0, 3]:
                input = torch.randn((B, num_heads, L, L))
                mask = torch.randint(0, 2, (B, L))
                mask = mask.reshape(B, 1, 1, L).expand(B, num_heads, L, L).bool()
                mask_type = 1   # BxL => src_key_padding_mask
                if (self.device_type == "cuda"):
                    input = input.cuda()
                    mask = mask.cuda()
                native_res = torch._masked_softmax(input, mask, dim, mask_type)
                mask = ~mask

                def slow_masked_softmax(input, mask):
                    exp = torch.exp(input)
                    exp = exp * mask
                    s = exp.sum(dim=dim, keepdim=True).expand(exp.size())
                    return exp / s

                pt_res = slow_masked_softmax(input, mask)
                pt_res = torch.nan_to_num(pt_res)

                mask_not = mask.logical_not()
                # In result, should only fill the entirely masked out rows since those are non-deterministic (*may* be 0)
                # Converts rows with all True's to False
                mask_out = mask_not.all(dim, keepdim=True).expand(mask_not.shape)
                self.assertEqual(
                    pt_res.masked_fill(mask_out, 0),
                    native_res.masked_fill(mask_out, 0),
                    exact_dtype=True
                )

    def _test_masked_softmax_helper(self, input, dim, mask, mask_type):
        input_ref = input.detach().clone().requires_grad_()
        result = torch._masked_softmax(input, mask, dim, mask_type)

        expected = torch._softmax(input_ref.masked_fill(mask, float('-inf')), dim, False)
        grad = torch.randn_like(expected).to(dtype=expected.dtype)

        result.backward(grad)
        expected.backward(grad)

        # Make sure the optional argument works as well
        if dim == input.dim() - 1:
            input_ref_default = input.detach().clone().requires_grad_()
            result_default = torch._masked_softmax(input_ref_default, mask, None, mask_type)
            result_default.backward(grad)
            self.assertEqual(result, result_default)
            self.assertEqual(input.grad, input_ref_default.grad)

        # In result, should only fill the entirely masked out rows since those are non-deterministic (*may* be 0)
        # Converts rows with all True's to False
        mask_out = mask.all(dim, keepdim=True).expand(mask.shape)
        self.assertEqual(result.masked_fill(mask_out, 0), expected.masked_fill(mask_out, 0))

        self.assertEqual(input.grad, torch.nan_to_num(input_ref.grad))
        self.assertEqual(input.grad, input.grad.masked_fill(mask, 0.0))

    def test_masked_softmax_grad(self, device):
        shapes = [(1, 1, 32), (3, 16, 310), (12, 4, 1024), (4, 2, 1200)]
        for shape in shapes:
            dims = [0, len(shape) - 1] if len(shape) > 0 else [0]
            for dim in dims:
                for mask_type in [1, 2]:  # 1 = BxL => src_key_padding_mask
                    input = torch.randn(shape, requires_grad=True)
                    mask = torch.randint(0, 2, shape).bool()
                    if (self.device_type == "cuda"):
                        input = input.cuda().detach().requires_grad_()
                        mask = mask.cuda()
                    self._test_masked_softmax_helper(input, dim, mask, mask_type)

    # In this test, the forward pass is expected to produce nan's because when dim=0, we only have unspecified values
    def test_masked_softmax_forward_with_nans(self, device):
        dim = 0
        shapes = [(4, 5), (50, 100), (1500, 1200)]
        for (x, y) in shapes:
            for mask_type in [1, 2]:  # 1 = BxL => src_key_padding_mask
                input = torch.randn((x, y), requires_grad=True)
                mask = torch.tensor([i % 2 for i in range(y)]).expand((x, y)).bool()
                if (self.device_type == "cuda"):
                    input = input.cuda().detach().requires_grad_()
                    mask = mask.cuda()
                self._test_masked_softmax_helper(input, dim, mask, mask_type)

    @onlyCUDA
    def test_masked_softmax_transformer_layout(self, device):
        B = 211
        num_heads = 16
        L = 42
        input = torch.randn((B, num_heads, L, L))
        dim = input.dim() - 1
        mask = torch.randint(0, 2, (B, L))
        mask_type = 1   # BxL => src_key_padding_mask
        if (self.device_type == "cuda"):
            input = input.cuda()
            mask = mask.cuda()
        mask = mask.bool()
        native_res = torch._masked_softmax(input, mask, dim, mask_type)
        mask = mask.reshape(B, 1, 1, L).expand(B, num_heads, L, L)
        mask = ~mask
        mask = mask.float()

        pt_res = self._slow_masked_softmax(input, mask)
        self.assertEqual(pt_res, native_res, exact_dtype=True)

    @onlyCUDA
    def test_masked_softmax_TxT_layout(self, device):
        B = 211
        num_heads = 16
        L = 42
        input = torch.randn((B, num_heads, L, L))
        dim = input.dim() - 1
        mask = torch.randint(0, 2, (L, L))
        mask_type = 0   # LxL => src_mask
        if (self.device_type == "cuda"):
            input = input.cuda()
            mask = mask.cuda()
        mask = mask.bool()
        native_res = torch._masked_softmax(input, mask, dim, mask_type)
        mask = mask.expand(B, num_heads, L, L)
        mask = ~mask
        mask = mask.float()

        pt_res = self._slow_masked_softmax(input, mask)
        self.assertEqual(pt_res, native_res, exact_dtype=True)

    @dtypesIfCUDA(torch.half, torch.float)
    @dtypes(torch.float)
    def test_softmax_results(self, device, dtype):
        # Non-even sizes and non-zero shifts test fallback paths in vectorized kernel
        # Note: dim1 > 1024 is needed to exercise the vectorized (non-persistent) path, (16, 30576) is BERT-esque
        sizes = [(0, 10), (32, 20), (10, 0), (31, 20), (32, 21), (31, 23), (32, 1536), (31, 2048), (33, 2049), (16, 30576)]
        shifts = [(0, 0), (1, 0), (0, 1), (1, 1)]
        for fn in [F.softmax, F.log_softmax]:
            for size in sizes:
                for shift in shifts:
                    input = torch.rand(size, device=device, dtype=dtype)
                    # Note: With the largest tests we can hit upper limit of fp16 when we
                    # sum, so scale the input down to stay in a nicer range.
                    if dtype == torch.float16:
                        input = input / 100.
                    input = input[shift[0]:, shift[1]:]
                    # Note; Don't want to bprop back through slice op
                    input = input.detach().requires_grad_(True)
                    ref_input = input.clone().cpu().detach().requires_grad_(True)
                    for dim in [0, 1]:
                        ref_output = fn(ref_input, dtype=torch.float, dim=dim)
                        output = fn(input, dtype=torch.float, dim=dim)
                        grad_output = torch.rand(size, device=device, dtype=dtype)
                        grad_output = grad_output[shift[0]:, shift[1]:]
                        ref_grad_output = grad_output.clone().cpu().detach()
                        grad_input, = torch.autograd.grad(output, input, grad_outputs=(grad_output), create_graph=True)
                        ref_grad_input, = torch.autograd.grad(ref_output, ref_input,
                                                              grad_outputs=(ref_grad_output), create_graph=True)
                        grad_input.sum().backward()
                        ref_grad_input.sum().backward()

                        self.assertEqual(output, ref_output)
                        self.assertEqual(grad_input, ref_grad_input)
                        self.assertEqual(input.grad, ref_input.grad)

    @onlyCUDA
    @dtypes(torch.float, torch.half)
    @largeTensorTest("20GB")
    @largeTensorTest("64GB", "cpu")
    def test_warp_softmax_64bit_indexing(self, device, dtype):
        def run_test(*shape):
            x = torch.randn(shape, device="cuda", dtype=torch.float16, requires_grad=True)
            y = F.log_softmax(x, dim=-1, dtype=dtype)
            y.backward(y)
            with torch.no_grad():
                xx = x.cpu().requires_grad_()
            yy = F.log_softmax(xx.float(), dim=-1).to(dtype)
            yy.backward(yy)
            # workaround to reduce memory usage vs. self.assertEqual, see #84944
            rtol, atol = torch.testing._comparison.get_tolerances(dtype, rtol=None, atol=None)
            self.assertTrue(torch.allclose(y.cpu(), yy, rtol=rtol, atol=atol))
            # x is half
            rtol, _ = torch.testing._comparison.get_tolerances(torch.half, rtol=None, atol=None)
            self.assertTrue(torch.allclose(x.grad.cpu(), xx.grad, rtol=rtol, atol=1e-3))

        run_test(1100000000, 2)  # Illegal memory access https://github.com/pytorch/pytorch/issues/52715
        run_test(2200000000, 1)  # invalid configuration argument https://github.com/pytorch/pytorch/issues/52716

    @onlyCUDA
    @dtypes(torch.half)
    @largeTensorTest("20GB")
    @largeTensorTest("2GB", "cpu")
    @precisionOverride({torch.half: 0.001})
    def test_softmax_64bit_indexing(self, device, dtype):
        def run_test(*shape):
            x = torch.ones(shape, device=device, dtype=dtype, requires_grad=True)
            y = F.log_softmax(x, dim=-1, dtype=dtype)
            y.backward(y)
            self.assertEqual(y[0], y[-1])
            self.assertEqual(x.grad[0], x.grad[-1])

        run_test(1024 * 256 + 1, 8192)  # https://github.com/pytorch/pytorch/issues/84144


    @dtypes(torch.float)
    @dtypesIfCUDA(torch.float, torch.half)
    def test_log_softmax_big(self, device, dtype):
        def _test_helper(shape):
            # generate a tensor with big numbers that are exactly representable in dtype
            # and are at a constant offset from tensor with small numbers
            # the logsoftmax of a small and big tensors should be equal
            x_small = torch.randint(100, shape, dtype=dtype, device=device)
            offset = 1.5e3 if dtype == torch.half else 1e7
            x_big = x_small + offset
            self.assertEqual(F.log_softmax(x_small, -1), F.log_softmax(x_big, -1))
        _test_helper((16, 4))
        if self.device_type == 'cuda':
            # test non-persistent softmax kernel
            _test_helper((4, 1536))

    def test_save_lstm_compatibility(self, device):
        # Test that saving an LSTM in PyTorch 1.7 and older can still be
        # loaded in newer versions of PyTorch.
        model = nn.LSTM(2, 3)
        x = torch.randn(32, 5, 2)
        expected = model(x)

        # Get a state dict for PyTorch 1.7 LSTM. Before PyTorch 1.8, proj_size
        # didn't exist.
        assert model.proj_size == 0
        state_dict = model.__dict__
        del state_dict['proj_size']

        # load a model
        loaded_model = nn.LSTM(2, 3)
        loaded_model.__setstate__(state_dict)
        result = loaded_model(x)
        self.assertEqual(result, expected)

    @onlyCUDA
    @tf32_on_and_off(0.005)
    def test_grid_sample_large(self, device):
        def issue_35202():
            input_tensor = torch.rand(1, 1, 480, 640, dtype=torch.float, device=device, requires_grad=True)
            coords = torch.tensor([[-10059144, 67680944], [67680944, 67680944]], dtype=torch.float, device=device)
            coords = coords.unsqueeze(0).unsqueeze(0).repeat(1, 1, 1, 1)
            result = torch.nn.functional.grid_sample(input_tensor, coords)
            self.assertEqual(result, torch.tensor([[[[0., 0.]]]], dtype=torch.float, device=device))
            result.backward(torch.ones_like(result))
            torch.cuda.synchronize()
        issue_35202()

        def issue_24823_1(dtype):
            image = torch.arange(27, 0, -1, dtype=dtype, device=device).view(1, 1, 3, 3, 3)
            image.requires_grad_()
            grid = torch.nn.functional.affine_grid(
                torch.tensor([[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]], dtype=dtype, device=device),
                (1, 1, 3, 3, 3))
            grid[:, 1, 1, 1, 0] = float('inf')
            result = torch.nn.functional.grid_sample(image, grid, padding_mode='zeros')
            self.assertEqual(result, torch.tensor([[[[[27., 26., 25.], [24., 23., 22.], [21., 20., 19.]],
                                                     [[18., 17., 16.], [15., 0., 13.], [12., 11., 10.]],
                                                     [[9., 8., 7.], [6., 5., 4.], [3., 2., 1.]]]]],
                                                  device=device, dtype=dtype))
            result.backward(torch.ones_like(result))
            expected_grad = torch.ones_like(image)
            expected_grad[0, 0, 1, 1, 1] = 0
            self.assertEqual(image.grad, expected_grad, atol=0.005, rtol=0)
        issue_24823_1(torch.half)
        issue_24823_1(torch.float)
        issue_24823_1(torch.double)

        def issue_24823_2():
            param = torch.tensor([[[-1.0e+20, 0.0, 0.0], [0.0, -1.0e+20, 0.0]]], dtype=torch.float, device=device)
            img = torch.zeros((1, 1, 4, 4), dtype=torch.float, device=device, requires_grad=True)
            grid = torch.nn.functional.affine_grid(param, img.size())
            result = torch.nn.functional.grid_sample(img, grid)
            self.assertEqual(result, torch.zeros(1, 1, 4, 4, device=device, dtype=torch.float))
            result.backward(torch.ones_like(result))
            torch.cuda.synchronize()
        issue_24823_2()

    @dtypes(torch.float, torch.double)
    @largeTensorTest(lambda self, device, dtype:
                     # Compute sum of the large tensor sizes:
                     # (im.numel() + small_image.numel() + small_image.grad.numel() +
                     #   large_view.grad.numel()) * sizeof(dtype)
                     32769 * (65536 + 3 * 65536 / 128) *
                     torch.tensor([], dtype=dtype).element_size())
    def test_grid_sample_large_index_2d(self, device, dtype):
        # Test 64-bit indexing with grid_sample (gh-41656)
        # Try accessing the corners, there should be no segfault
        coords = torch.tensor([[[-1., -1.],
                                [+1., -1.]],

                               [[-1., +1.],
                                [+1., +1.]]], device=device, dtype=dtype)
        coords = coords.expand(1, 2, 2, 2)
        im = torch.zeros([1, 1, 32769, 65536], device=device, dtype=dtype)

        # Compare sampling with large strides to the same op on a contiguous tensor
        coords = torch.rand(1, 4, 4, 2, device=device, dtype=dtype)
        large_view = im[..., 127::128]
        small_image = torch.rand_like(large_view)
        large_view[...] = small_image
        large_view.requires_grad, small_image.requires_grad = True, True
        self.assertTrue(
            sum(i * s for i, s in zip(large_view.size(), large_view.stride())) >= 2 ** 31,
            msg="View must use 64-bit indexing")
        for mode, padding_mode, align_corners in itertools.product(
                ('nearest', 'bilinear', 'bicubic'), ('zeros', 'border', 'reflection'), (True, False)):
            a = F.grid_sample(
                small_image, coords, mode=mode,
                padding_mode=padding_mode, align_corners=align_corners)
            a.sum().backward()

            b = F.grid_sample(
                large_view, coords, mode=mode,
                padding_mode=padding_mode, align_corners=align_corners)
            b.sum().backward()

            self.assertEqual(a, b)
            self.assertEqual(small_image.grad, large_view.grad)

            small_image.grad.zero_()
            large_view.grad.zero_()

    @dtypes(torch.float, torch.double)
    @largeTensorTest(lambda self, device, dtype:
                     # Compute sum of the large tensor sizes:
                     # (im.numel() + small_image.numel() + small_image.grad.numel() +
                     #   large_view.grad.numel()) * sizeof(dtype)
                     2 * 32769 * (32768 + 3 * 32768 / 128) *
                     torch.tensor([], dtype=dtype).element_size())
    def test_grid_sample_large_index_3d(self, device, dtype):
        # Test 64-bit indexing with grid_sample (gh-41656)
        # Try accessing the corners, there should be no segfault
        coords = torch.full((1, 2, 2, 2, 3), 1., device=device, dtype=dtype)
        im = torch.zeros([1, 1, 2, 32769, 32768], device=device, dtype=dtype)

        result = F.grid_sample(im, coords, align_corners=False)
        self.assertEqual(result, torch.zeros((1, 1, 2, 2, 2), device=device, dtype=dtype))

        # Compare sampling with large strides to the same op on a contiguous tensor
        coords = torch.rand(1, 1, 4, 4, 3, device=device, dtype=dtype)
        large_view = im[..., 127::128]
        small_image = torch.rand_like(large_view)
        large_view[...] = small_image
        small_image.requires_grad, large_view.requires_grad = True, True
        self.assertTrue(
            sum(i * s for i, s in zip(large_view.size(), large_view.stride())) >= 2 ** 31,
            msg="View must use 64-bit indexing")
        for mode, padding_mode, align_corners in itertools.product(
                ('nearest', 'bilinear'), ('zeros', 'border', 'reflection'), (True, False)):
            a = F.grid_sample(
                small_image, coords, mode=mode,
                padding_mode=padding_mode, align_corners=align_corners)
            a.sum().backward()

            b = F.grid_sample(
                large_view, coords, mode=mode,
                padding_mode=padding_mode, align_corners=align_corners)
            b.sum().backward()

            self.assertEqual(a, b)
            self.assertEqual(small_image.grad, large_view.grad)

            small_image.grad.zero_()
            large_view.grad.zero_()

    def _test_gumbel_softmax_st_shapes(self, device, dtype, shape, dim, count_expected):
        logits = torch.randn(shape, dtype=torch.float, device=device)
        logits = logits.to(dtype)

        y_draw = F.gumbel_softmax(logits, hard=True, dim=dim)

        # All values positive
        self.assertGreaterEqual(y_draw.min(), 0)
        # Shape unchanged
        self.assertTrue(y_draw.shape == logits.shape)
        # One choice per draw
        self.assertEqual(y_draw.sum(), count_expected, atol=torch.finfo(y_draw.dtype).eps, rtol=0)

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
        self.assertEqual(counts.sum(), num_draws, atol=torch.finfo(counts.dtype).eps, rtol=0)

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
        self.assertEqual(logits_soft.grad, logits_hard.grad, atol=tol, rtol=0)

    @skipIfMps
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
    @dtypes(torch.double)
    def test_lstmcell_backward_only_one_output_grad(self, device, dtype):
        # checks that undefined gradients doen't hamper the backward
        # see #11872
        l = torch.nn.LSTMCell(2, 3).to(device).to(dtype=dtype)
        s = torch.randn(1, 2, device=device, dtype=dtype, requires_grad=True)
        for i in range(2):
            out = l(s)[i]
            out.sum().backward()
            self.assertFalse(s.grad is None or s.grad.abs().sum().item() == 0)

    def _test_rnn_mod(self, mod, inp):
        def flatten_out(mod, inp):
            out = mod(inp)
            return tuple([t if isinstance(t, torch.Tensor) else tt for t in out for tt in t])
        gradcheckfunc = partial(flatten_out, mod)
        with torch.backends.cudnn.flags(enabled=False):
            gradcheck(gradcheckfunc, inp, check_batched_grad=False)
            gradgradcheck(gradcheckfunc, inp, check_batched_grad=False)

        if inp.is_cuda and not TEST_WITH_ROCM:
            # Assert that we have good error message around unsupported CuDNN double backward
            # NB: we trigger double backward using .backward() instead of autograd.grad due to
            # https://github.com/pytorch/pytorch/issues/37874
            with torch.backends.cudnn.flags(enabled=True):
                result = gradcheckfunc(inp)
                result[0].sum().backward(create_graph=True)
                grad0 = next(mod.parameters()).grad
                with self.assertRaisesRegex(RuntimeError,
                                            "please disable the CuDNN backend temporarily"):
                    grad0.sum().backward()

                # Here we avoid the backward(create_graph=True) memory leak
                # described in https://github.com/pytorch/pytorch/issues/7343
                for param in mod.parameters():
                    param.grad = None
                inp.grad = None

    # Merge into OpInfo?
    @skipMeta  # LSTM cell reuses output which was resized
    @dtypes(torch.double)
    def test_LSTM_grad_and_gradgrad(self, device, dtype):
        hsize = 4
        inp = torch.rand(1, 3, hsize, device=device, dtype=dtype, requires_grad=True)
        for bias in [True, False]:
            mod = torch.nn.LSTM(hsize, hsize, bias=bias).to(device).to(dtype)
            self._test_rnn_mod(mod, inp)

    @skipMeta  # GRU cell reuses output which was resized
    @dtypes(torch.double)
    def test_GRU_grad_and_gradgrad(self, device, dtype):
        hsize = 4
        inp = torch.rand(1, 3, hsize, device=device, dtype=dtype, requires_grad=True)
        for bias in [True, False]:
            mod = torch.nn.GRU(hsize, hsize, bias=bias).to(device).to(dtype)
            self._test_rnn_mod(mod, inp)

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
    @skipCUDAIfNotRocm
    def test_upsamplingNearest2d_launch_rocm(self, device):
        # test_upsamplingNearest2d_launch_fail should run OK on ROCm
        m = nn.Upsample(scale_factor=2)
        inp = torch.rand(1, 1, 2**15, 2**8, device=device)
        out = m(inp)

    @onlyCUDA
    @skipCUDAIfCudnnVersionLessThan(7600)
    def test_CTCLoss_cudnn(self, device):
        def _helper(zero_infinity):
            target_lengths = [30, 25, 20]
            input_lengths = [50, 50, 50]
            targets = torch.randint(1, 15, (sum(target_lengths),), dtype=torch.int)
            log_probs = torch.randn(50, 3, 15, dtype=torch.float, device=device).log_softmax(2).requires_grad_()

            log_probs_ref = log_probs.detach().clone().requires_grad_()

            with torch.backends.cudnn.flags(enabled=True):
                res = torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths, zero_infinity=zero_infinity)
                res.backward()

            expected = ctcloss_reference(log_probs, targets.cuda(), input_lengths, target_lengths).float()

            with torch.backends.cudnn.flags(enabled=False):
                res2 = torch.nn.functional.ctc_loss(log_probs_ref, targets.cuda().long(), input_lengths, target_lengths,
                                                    zero_infinity=zero_infinity)
                res2.backward()

            self.assertEqual(res, expected)
            self.assertEqual(res2, res)
            self.assertEqual(log_probs.grad, log_probs_ref.grad)

        _helper(zero_infinity=True)
        _helper(zero_infinity=False)

    def _CTCLoss_gen_losses(self, device, input_length, vocab_size, target_length, reduction, use_module_form):
        batch_size = 1
        log_probs = torch.randn(input_length, batch_size, vocab_size, dtype=torch.float, device=device) \
                         .log_softmax(2).requires_grad_()
        targets = torch.randint(low=1, high=vocab_size - 1, size=(batch_size, target_length),
                                dtype=torch.int, device=device)
        input_lengths = batch_size * [input_length]
        target_lengths = batch_size * [target_length]

        log_probs_no_bd = log_probs.squeeze(1).detach().clone().requires_grad_()
        targets_no_bd = targets.squeeze(0).detach().clone()
        input_lengths_no_bd = torch.tensor(input_length)
        target_lengths_no_bd = torch.tensor(target_length)

        # currently only length 2 and 1 right now, but left flexible for additional potential cases
        log_probs_refs = [log_probs.detach().clone().requires_grad_() for _ in range(2)]
        log_probs_no_bd_refs = [log_probs_no_bd.detach().clone().requires_grad_() for _ in range(1)]

        losses = []
        losses_no_bd = []

        has_cuda = torch.cuda.is_available()
        has_cudnn = has_cuda and 'cuda' in device and self.has_cudnn()
        # cudnn requires a cpu target
        if has_cuda and has_cudnn:
            targets = targets.cpu()
            targets_no_bd = targets_no_bd.cpu()

        ctc_loss = (
            nn.CTCLoss(reduction=reduction, zero_infinity=True)
            if use_module_form
            else partial(torch.nn.functional.ctc_loss, reduction=reduction, zero_infinity=True)
        )

        with torch.backends.cudnn.flags(enabled=has_cudnn):
            # batched case. log_probs.shape = (T, N, C), targets = (N, S), input_lengths/target_lengths = (N,)
            losses.append(ctc_loss(log_probs_refs[0], targets, input_lengths, target_lengths))
            # batched case. input.shape = (T, N, C), targets = (S,), input_lengths/target_lengths = (N,)
            losses.append(ctc_loss(log_probs_refs[1], targets_no_bd, input_lengths, target_lengths))
            # unbatched case. input.shape = (T, C), targets = (S,), input_lengths/target_lengths = (N,)
            losses_no_bd.append(ctc_loss(log_probs_no_bd_refs[0], targets_no_bd,
                                         input_lengths_no_bd, target_lengths_no_bd))

            for loss in losses + losses_no_bd:
                loss.backward()

        return losses, losses_no_bd, log_probs_refs, log_probs_no_bd_refs

    def _assertEqual_list(self, expected, list_to_compare, atol=None, rtol=None):
        for ele in list_to_compare:
            self.assertEqual(expected, ele, atol=atol, rtol=rtol)

    @parametrize_test("reduction", ['none', 'mean', 'sum'])
    @parametrize_test("use_module_form", [True, False])
    def test_CTCLoss_no_batch_dim(self, device, reduction, use_module_form):
        input_length = 40
        vocab_size = 3
        target_length = 12

        args = self._CTCLoss_gen_losses(device, input_length, vocab_size, target_length, reduction, use_module_form)
        losses, losses_no_bd, log_probs_refs, log_probs_no_bd_refs = args

        # test output values
        self._assertEqual_list(losses[0], losses[1:], atol=1e-4, rtol=0)
        self._assertEqual_list(losses[0].squeeze(0), losses_no_bd, atol=1e-4, rtol=0)

        # test gradient values
        self._assertEqual_list(log_probs_refs[0].grad, [t.grad for t in log_probs_refs[1:]], atol=1e-4, rtol=0)
        self._assertEqual_list(
            log_probs_refs[0].grad.squeeze(1),
            [t.grad for t in log_probs_no_bd_refs],
            atol=1e-4,
            rtol=0,
        )

        # checking the output's shape
        # batch dim case should be (N,). no batch dim case should be ()
        self._assertEqual_list((1,) if reduction == 'none' else (), [loss.shape for loss in losses])
        self._assertEqual_list((), [loss.shape for loss in losses_no_bd])

        # checking the gradient's shape
        # batch dim case should have shape (T, N, C). no batch dim case should have shape (T, C)
        self._assertEqual_list((input_length, 1, vocab_size), [t.grad.shape for t in log_probs_refs])
        self._assertEqual_list((input_length, vocab_size), [t.grad.shape for t in log_probs_no_bd_refs])

    def _ordered_sequence(self, device, dtype):
        """Create ordered list of random sequences"""
        seqs = [torch.empty(random.randint(1, 6), device=device, dtype=dtype)
                for _ in range(5)]
        seqs = [s.random_(-128, 128) for s in seqs]
        ordered = sorted(seqs, key=len, reverse=True)
        return ordered

    def _padded_sequence(self, device, dtype):
        """Create Tensor of random padded sequences"""
        ordered = self._ordered_sequence(device, dtype)
        lengths = [len(i) for i in ordered]
        padded_tensor = rnn_utils.pad_sequence(ordered)
        return padded_tensor, lengths

    @onlyCUDA
    def test_device_mask(self, device):
        for enforce_sorted in [True, False]:
            padded, lengths = self._padded_sequence('cpu', torch.float)
            packed = rnn_utils.pack_padded_sequence(
                padded, lengths, enforce_sorted=enforce_sorted)
            self.assertFalse(packed.is_cuda)
            packed = packed.to(device)
            self.assertTrue(packed.is_cuda)
            unpacked, _ = rnn_utils.pad_packed_sequence(packed)
            self.assertTrue(unpacked.is_cuda)
            self.assertEqual(unpacked.dtype, torch.float)

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
            with torch.no_grad():
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
    @dtypes(torch.half, torch.float)
    def test_softmax(self, device, dtype):
        input = torch.rand(32, 100, device=device, dtype=dtype, requires_grad=True)
        inputf = input.to(torch.float).detach().requires_grad_(True)
        out = F.softmax(input, dim=-1, dtype=torch.float)
        outf = F.softmax(inputf, dim=-1)
        # should be bitwise equal
        self.assertEqual(out, outf, atol=0, rtol=0)
        gO = torch.empty_like(outf).uniform_()
        out.backward(gO)
        outf.backward(gO)
        # should be bitwise equal
        self.assertEqual(input.grad, inputf.grad.to(dtype), atol=0, rtol=0)

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

    @onlyCUDA
    @dtypes(torch.half, torch.float, torch.double)
    def test_multihead_attention_dtype_batch_first(self, device, dtype):
        embed_dim = 128
        num_heads = 8
        sl = 10
        bs = 8
        # With batch_first=True, we have the possibility of hitting
        # the native fast path if we call .eval() and enable inference
        # mode. Test both paths.
        for training in (True, False):
            model = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True).cuda().to(dtype)
            if not training:
                model = model.eval()
                cm = torch.no_grad()
            else:
                cm = contextlib.nullcontext()
            with cm:
                q = torch.randn(bs, sl, embed_dim, device=device, dtype=dtype)
                k = torch.randn(bs, sl, embed_dim, device=device, dtype=dtype)
                v = torch.randn(bs, sl, embed_dim, device=device, dtype=dtype)
                # fast path currently doesn't support weights
                out = model(q, k, v, need_weights=False)
                self.assertEqual(q.size(), out[0].size())
                self.assertEqual(dtype, out[0].dtype)

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

    @onlyCUDA
    def test_layernorm_half_precision(self):
        width = 128
        input = torch.rand(1, 5, width, device="cuda", dtype=torch.half) * 0.1
        normalized_shape = (width,)
        weight = torch.ones(width, device="cuda", dtype=torch.half)
        bias = torch.zeros(width, device="cuda", dtype=torch.half)
        eps = 1e-5

        output_fp16 = torch.layer_norm(input, normalized_shape, weight, bias, eps)
        output_fp32 = torch.layer_norm(input.float(), normalized_shape, weight.float(), bias.float(), eps).half()
        self.assertEqual(output_fp16, output_fp32, atol=0, rtol=0)

    @onlyCUDA
    def test_layernorm_weight_bias(self):
        width = 128
        input = torch.rand(1, 5, width, device="cuda", dtype=torch.float32) * 0.1
        normalized_shape = (width,)
        data = torch.randn(width, device="cuda", dtype=torch.float32)
        weight = torch.ones(width, device="cuda", dtype=torch.float32)
        bias = torch.zeros(width, device="cuda", dtype=torch.float32)
        eps = 1e-5

        out_none_weight = torch.layer_norm(input, normalized_shape, None, data, eps)
        out_one_weight = torch.layer_norm(input, normalized_shape, weight, data, eps)
        self.assertEqual(out_none_weight, out_one_weight)

        out_none_bias = torch.layer_norm(input, normalized_shape, data, None, eps)
        out_zero_bias = torch.layer_norm(input, normalized_shape, data, bias, eps)
        self.assertEqual(out_none_bias, out_zero_bias)

    def test_hardsigmoid_grad(self, device):
        inputs = (torch.randn(4, 16, 16, device=device) - 0.5) * 10
        inputs.requires_grad = True
        self.assertTrue(gradcheck(F.hardsigmoid, (inputs,)))

    # currently fails on XLA
    @onlyNativeDeviceTypes
    def test_hardswish_grad(self, device):
        inputs = (torch.randn(4, 16, 16, device=device) - 0.5) * 10
        inputs.requires_grad = True
        self.assertTrue(gradcheck(F.hardswish, (inputs,)))


    def _test_batchnorm_eval(self, ndim, device, dtype, module_dtype=None):
        module_dtype = module_dtype or dtype
        module = nn.BatchNorm1d(3).to(device, module_dtype)
        module.eval()

        data = torch.rand([3] * ndim, device=device, dtype=dtype, requires_grad=True)
        grad = torch.rand([3] * ndim, device=device, dtype=dtype)

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
        module = nn.BatchNorm1d(3, track_running_stats=False).to(device, module_dtype)

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

    @dtypes(torch.float)
    @dtypesIfCUDA(torch.float, torch.bfloat16)
    def test_batchnorm_eval(self, device, dtype):
        self._test_batchnorm_eval(2, device, dtype)
        self._test_batchnorm_eval(3, device, dtype)

        if self.device_type == 'cuda' and self.has_cudnn():
            with torch.backends.cudnn.flags(enabled=False):
                self._test_batchnorm_eval(2, device, dtype)
                self._test_batchnorm_eval(3, device, dtype)

    @onlyCUDA
    @dtypes(torch.bfloat16, torch.half)
    def test_batchnorm_eval_mixed(self, device, dtype):
        # Test bfloat16 input with float module
        self._test_batchnorm_eval(2, device, dtype, torch.float)
        self._test_batchnorm_eval(3, device, dtype, torch.float)

        if self.device_type == 'cuda' and self.has_cudnn():
            with torch.backends.cudnn.flags(enabled=False):
                self._test_batchnorm_eval(2, device, dtype, torch.float)
                self._test_batchnorm_eval(3, device, dtype, torch.float)

    def _test_batchnorm_affine(self, ndim, device, dtype, module_dtype=None):
        # Compare affine against no-op weights and bias
        module_dtype = module_dtype or dtype
        module = nn.BatchNorm1d(3, affine=False).to(device, module_dtype)
        module_affine = nn.BatchNorm1d(3, affine=True).to(device, module_dtype)
        with torch.no_grad():
            module_affine.weight.fill_(1.0)
            module_affine.bias.zero_()

        data = torch.rand([3] * ndim, device=device, dtype=dtype, requires_grad=True)
        grad = torch.ones_like(data, requires_grad=False)

        # With weights all ones and bias all zeros
        res1 = module_affine(data)
        res1.backward(grad)
        grad1 = data.grad.clone()
        data.grad.zero_()

        # Without any weights or bias
        res2 = module(data)
        res2.backward(grad)
        grad2 = data.grad

        self.assertEqual(res1, res2)
        self.assertEqual(grad1, grad2)

    @dtypes(torch.float)
    @dtypesIfCUDA(torch.float, torch.bfloat16)
    def test_batchnorm_affine(self, device, dtype):
        self._test_batchnorm_affine(2, device, dtype)
        self._test_batchnorm_affine(3, device, dtype)

        if self.device_type == 'cuda' and self.has_cudnn():
            with torch.backends.cudnn.flags(enabled=False):
                self._test_batchnorm_affine(2, device, dtype)
                self._test_batchnorm_affine(3, device, dtype)

    @onlyCUDA
    @dtypes(torch.bfloat16, torch.half)
    def test_batchnorm_affine_mixed(self, device, dtype):
        cudnn_enabled = [False]
        if self.device_type == 'cuda' and self.has_cudnn():
            # TODO: Test fails with cudnn, see gh-62034
            # cudnn_enabled = [False, True]
            pass

        # Test bfloat16 input with float module
        for enabled in cudnn_enabled:
            with torch.backends.cudnn.flags(enabled=enabled):
                self._test_batchnorm_affine(2, device, dtype, torch.float)
                self._test_batchnorm_affine(3, device, dtype, torch.float)

    def _test_batchnorm_simple_average(self, device, dtype, module_dtype=None):
        module_dtype = module_dtype or dtype
        module = nn.BatchNorm1d(3, momentum=None).to(dtype=module_dtype, device=device)
        zeros = torch.zeros(3, dtype=module_dtype, device=device)
        ones = torch.ones(3, dtype=module_dtype, device=device)
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
        self.assertEqual(module.running_mean, (running_mean1 + running_mean2) / 2)
        self.assertEqual(module.running_var, (running_var1 + running_var2) / 2)

    @dtypes(torch.float)
    @dtypesIfCUDA(torch.float, torch.bfloat16)
    def test_batchnorm_simple_average(self, device, dtype):
        self._test_batchnorm_simple_average(device, dtype)

        if self.device_type == 'cuda' and self.has_cudnn():
            with torch.backends.cudnn.flags(enabled=False):
                self._test_batchnorm_simple_average(device, dtype)

    @onlyCUDA
    @dtypes(torch.bfloat16, torch.half)
    def test_batchnorm_simple_average_mixed(self, device, dtype):
        self._test_batchnorm_simple_average(device, dtype, torch.float)

        if self.device_type == 'cuda' and self.has_cudnn():
            with torch.backends.cudnn.flags(enabled=False):
                self._test_batchnorm_simple_average(device, dtype, torch.float)

    @onlyNativeDeviceTypes
    @dtypes(torch.float, torch.double)
    def test_grid_sample_nan_inf(self, device, dtype):
        input = torch.zeros([1, 1, 3, 3], device=device, dtype=dtype)
        grid = torch.tensor([[[[nan, 0], [0, inf]]]], device=device, dtype=dtype)
        for padding_mode in ('reflection', 'border', 'zeros'):
            sample = torch.nn.functional.grid_sample(input=input, grid=grid, mode='nearest',
                                                     padding_mode=padding_mode, align_corners=False)
            self.assertEqual(sample, torch.zeros([1, 1, 1, 2], device=device, dtype=dtype))

    def test_CTCLoss_empty_target(self, device):
        target_lengths = [0, 0, 0]
        input_lengths = [50, 50, 50]
        targets = torch.randint(1, 15, (0,), dtype=torch.long, device=device)
        log_probs = torch.randn(50, 3, 15, dtype=torch.double, device=device).log_softmax(2)
        loss = torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths, reduction='none')
        self.assertTrue((loss >= 0).all().item())
        self.assertEqual(-log_probs.sum(0)[:, 0], loss)

        target_lengths = [0, 9, 0]
        input_lengths = [50, 50, 50]
        targets = torch.randint(1, 15, (9,), dtype=torch.long, device=device)
        log_probs = torch.randn(50, 3, 15, dtype=torch.double, device=device).log_softmax(2)
        loss = torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths, reduction='none')
        self.assertTrue((loss >= 0).all().item())
        self.assertEqual(-log_probs.sum(0)[[0, 2], 0], loss[[0, 2]])

    # Merge into OpInfo?
    @skipCUDAIf(True, """Test is flaky on Linux and Windows, typical error message:
                          https://github.com/pytorch/pytorch/issues/34870""")
    def test_ctc_loss(self, device):
        batch_size = 64
        num_labels = 101
        target_length = 15
        gradcheck_input_size = 10

        ZERO_NONE = 0
        ZERO_SOME = 1
        ZERO_ALL = 2

        # input_length, vary_lengths, zero_lengths
        tests = [(150, False, ZERO_NONE),
                 (150, True, ZERO_NONE),
                 (50, True, ZERO_SOME),
                 (50, True, ZERO_ALL)]

        if 'cuda' in device:
            tests += [(50, False, ZERO_NONE),
                      (50, True, ZERO_NONE),
                      (150, True, ZERO_SOME),
                      (150, True, ZERO_ALL)]

        for input_length, vary_lengths, zero_mode in tests:
            targets = torch.randint(1, num_labels, (batch_size, target_length),
                                    device=device, dtype=torch.long)
            x = torch.randn(gradcheck_input_size, dtype=torch.double, device=device, requires_grad=True)
            tile_factors = torch.randn(input_length * batch_size * num_labels // gradcheck_input_size + 1,
                                       device=device)
            input_lengths = [(torch.randint(input_length // 2, input_length + 1, ()).item()
                              if vary_lengths or i == 0 else input_length) for i in range(batch_size)]
            if zero_mode == ZERO_ALL:
                target_lengths = [0 for _ in range(batch_size)]
            else:
                target_lengths = [(torch.randint(target_length // 2, target_length + 1, ()).item()
                                   if vary_lengths else target_length) for _ in range(batch_size)]
                if zero_mode == ZERO_SOME:
                    idxes = torch.randint(0, batch_size, (10,))
                    for i in idxes:
                        target_lengths[i] = 0

            def ctc_after_softmax(x):
                x_full = ((x[:, None] * tile_factors[None, :]).view(-1)[:input_length * batch_size * num_labels]
                          .view(input_length, batch_size, num_labels))
                log_probs = torch.log_softmax(x_full, 2)
                return torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths)

            gradcheck(ctc_after_softmax, [x])

    @onlyCUDA
    @skipCUDAIfRocm
    @skipCUDAIfCudnnVersionLessThan(7600)
    def test_ctc_loss_cudnn(self, device):
        batch_size = 16
        input_length = 30
        num_labels = 101
        target_length = 15
        targets = torch.randint(1, num_labels, (batch_size * target_length,),
                                device='cuda', dtype=torch.long)
        log_probs = torch.log_softmax(torch.randn(input_length, batch_size, num_labels, device='cuda', dtype=torch.float), 2)
        log_probs.requires_grad_()

        input_lengths = batch_size * [input_length]
        target_lengths = batch_size * [target_length]
        grad_out = torch.randn(batch_size, device='cuda', dtype=torch.float)
        with torch.backends.cudnn.flags(enabled=False):
            loss_native = torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths, reduction='none')
            grad_native, = torch.autograd.grad(loss_native, log_probs, grad_out)
        loss_cudnn = torch.nn.functional.ctc_loss(log_probs, targets.to('cpu', torch.int32),
                                                  input_lengths, target_lengths, reduction='none')
        self.assertTrue("Cudnn" in str(loss_cudnn.grad_fn))
        grad_cudnn, = torch.autograd.grad(loss_cudnn, log_probs, grad_out)
        self.assertEqual(grad_cudnn, grad_native, atol=1e-4, rtol=0)

    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    @dtypes(torch.float)
    @tf32_on_and_off(0.005)
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

        def check_lengths(lengths, enforce_sorted, use_default_hiddens, proj_size):
            input_size = 3
            hidden_size = 4
            num_layers = 2
            bidirectional = True

            max_length = max(lengths)
            x_leaf = torch.randn(max_length, len(lengths), input_size, device=device,
                                 dtype=dtype, requires_grad=True)
            num_directions = 2 if bidirectional else 1
            lstm = nn.LSTM(input_size, hidden_size, bidirectional=bidirectional,
                           num_layers=num_layers, proj_size=proj_size).to(device, dtype)
            lstm2 = deepcopy(lstm).to(device, dtype)
            x = x_leaf

            hidden0 = None
            if not use_default_hiddens:
                real_hidden_size = hidden_size if proj_size == 0 else proj_size
                hidden0 = (torch.randn(num_directions * num_layers, len(lengths), real_hidden_size,
                                       device=device, dtype=dtype),
                           torch.randn(num_directions * num_layers, len(lengths), hidden_size,
                                       device=device, dtype=dtype))

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
            prec = dtype2prec_DONTUSE[dtype]
            self.assertEqual(packed_hidden, seq_hidden, atol=prec, rtol=0)
            self.assertEqual(unpacked, seq_out, atol=prec, rtol=0)
            self.assertEqual(unpacked_len, lengths, atol=prec, rtol=0)

            # Check backward
            seq_out.sum().backward()
            grad_x = x_leaf.grad.data.clone()
            x_leaf.grad.data.zero_()
            unpacked.sum().backward()

            self.assertEqual(x_leaf.grad, grad_x, atol=dtype2prec_DONTUSE[dtype], rtol=0)
            for p1, p2 in zip(lstm.parameters(), lstm2.parameters()):
                prec = dtype2prec_DONTUSE[dtype]
                if dtype == torch.float16:
                    prec = 4e-2
                self.assertEqual(p1.grad, p2.grad, atol=prec, rtol=0)

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
                for proj_size in [0, 2]:
                    check_lengths(seq_lens, enforce_sorted, use_default_hiddens, proj_size)

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

    @onlyCPU
    def test_activations_bfloat16_cpu(self, device):
        def test_bfloat16(fn, device, inp_dims, prec):
            # bfloat16 compute
            input = torch.randn(inp_dims, dtype=torch.bfloat16, device=device, requires_grad=True)
            out = fn(input)
            grad_input = torch.randn_like(out, dtype=torch.bfloat16, device=device)
            out.backward(grad_input)

            # fp32 compute
            input2 = input.detach().clone().float().requires_grad_(True)
            out2 = fn(input2)
            grad_input2 = grad_input.detach().clone().float()
            out2.backward(grad_input2)

            self.assertEqual(out.dtype, torch.bfloat16)
            self.assertEqual(input.grad.dtype, torch.bfloat16)
            self.assertEqual(out, out2, atol=prec, rtol=0, exact_dtype=False)
            self.assertEqual(input.grad.data, input2.grad.data, atol=prec, rtol=0, exact_dtype=False)

        shapes = [[1, 3, 1, 6], [1, 3, 1, 128], [1, 3, 256, 256]]
        for shape in shapes:
            test_bfloat16(torch.nn.LogSigmoid(), device, shape, prec=2e-2)
            test_bfloat16(torch.nn.Hardsigmoid(), device, shape, prec=1e-2)
            test_bfloat16(torch.nn.Hardshrink(), device, shape, prec=1e-2)
            test_bfloat16(torch.nn.Softshrink(), device, shape, prec=1e-2)
            test_bfloat16(torch.nn.Hardswish(), device, shape, prec=2e-2)
            test_bfloat16(torch.nn.Softplus(), device, shape, prec=1e-2)

    @onlyCUDA
    def test_activations_bfloat16(self, device):
        _test_bfloat16_ops(self, torch.nn.ReLU(), device, inp_dims=(5), prec=1e-2)
        _test_bfloat16_ops(self, torch.nn.Threshold(0.1, 20), device, inp_dims=(5), prec=1e-2)
        _test_bfloat16_ops(self, torch.nn.ELU(), device, inp_dims=(5), prec=1e-2)
        _test_bfloat16_ops(self, torch.nn.Softplus(), device, inp_dims=(5), prec=1e-2)
        _test_bfloat16_ops(self, torch.nn.Hardshrink(), device, inp_dims=(5), prec=1e-2)
        _test_bfloat16_ops(self, torch.nn.Softshrink(), device, inp_dims=(5), prec=1e-2)
        _test_bfloat16_ops(self, torch.nn.LeakyReLU(), device, inp_dims=(5), prec=1e-2)

    @onlyNativeDeviceTypes
    def test_softmax_bfloat16(self, device):
        for dim in [0, 1, 2, 3]:
            _test_bfloat16_ops(self, torch.nn.Softmax(dim=dim), device, inp_dims=(16, 33, 15, 16), prec=1e-2)
            # test softmax with large input value which casues exp() to overflow
            _test_bfloat16_ops(self, torch.nn.Softmax(dim=dim), device, inp_dims=(16, 33, 15, 16), prec=0.05, scale_factor=1000.0)

    def test_nll_loss_mismatched_batch(self, device):
        x = torch.randn((10, 3), requires_grad=True, device=device)
        # t should have size (10,)
        t = torch.zeros((3,), dtype=torch.int64, device=device)
        with self.assertRaisesRegex(ValueError, 'Expected.*batch_size'):
            F.nll_loss(x, t)

    def test_nll_loss_out_of_bounds_ignore_index(self, device):
        x = torch.randn(6, 3, requires_grad=True, device=device)
        t = torch.tensor([0, 1, 255, 0, 1, 2], dtype=torch.int64, device=device)
        for reduction in ['mean', 'none']:
            F.nll_loss(x, t, ignore_index=255, reduction=reduction).sum().backward()

    def test_nll_loss_invalid_target_dim(self, device):
        x = torch.randn((10, 3), device=device)
        t = torch.zeros((10, 2), dtype=torch.int64, device=device)
        with self.assertRaisesRegex(RuntimeError, "1D target tensor expected"):
            F.nll_loss(x, t)

    def test_nll_loss_invalid_weights(self, device):
        x = torch.randn((10, 3), device=device)
        t = torch.empty(10, dtype=torch.int64, device=device).random_(0, 3)
        invalid_weights = [
            torch.randn(4, device=device),
            torch.randn(1, 3, device=device),
        ]
        msg = "weight tensor should be defined either for all 3 classes or no classes"
        for weight in invalid_weights:
            with self.assertRaisesRegex(RuntimeError, msg):
                F.nll_loss(x, t, weight=weight)

    # Ref: https://github.com/pytorch/pytorch/issue/85005
    @onlyCUDA
    @largeTensorTest("45GB", "cpu")
    @largeTensorTest("45GB", "cuda")
    @parametrize_test("reduction", ("none", "mean", "sum"))
    def test_nll_loss_large_tensor(self, device, reduction):
        shape = [int(2 ** 16), int(2 ** 16) + 1]

        input = torch.randn(shape, device=device, dtype=torch.float32, requires_grad=True)
        labels = torch.randint(shape[0], (shape[0],), dtype=torch.long, device=device)

        out = F.nll_loss(input, labels, reduction=reduction)

        with torch.no_grad():
            input_cpu = input.cpu().float().requires_grad_()
            labels_cpu = labels.cpu()
        out_cpu = F.nll_loss(input_cpu, labels_cpu, reduction=reduction)
        # workaround to reduce memory usage vs. self.assertEqual, see #84944
        rtol, atol = torch.testing._comparison.get_tolerances(torch.float32, rtol=None, atol=None)
        if reduction == "sum":
            orig_rtol, orig_atol = rtol, atol
            rtol, atol = 7 * rtol, 3 * atol
        with torch.no_grad():
            self.assertTrue(torch.allclose(out.cpu(), out_cpu, rtol=rtol, atol=atol))
        if reduction == "sum":
            rtol, atol = orig_rtol, orig_atol

        if reduction != "none":
            out.backward()
            out_cpu.backward()
            with torch.no_grad():
                self.assertTrue(torch.allclose(input.grad.cpu(), input_cpu.grad, rtol=rtol, atol=atol))

    def _nll_loss_helper(self, input_size, reduction, expected, device):
        input = torch.rand(input_size, requires_grad=True, device=device)
        num_channels = input_size[1]
        target_size = (input_size[0], ) + tuple(input_size[2:])
        target = torch.randint(num_channels, target_size, device=device)

        output = F.nll_loss(input, target, reduction=reduction)
        self.assertEqual(output, expected, exact_dtype=False)

        output.sum().backward()
        self.assertEqual(input.grad.size(), input.size())

    def test_nll_loss_empty_tensor_reduction_none(self, device):
        self._nll_loss_helper([0, 3], "none", torch.empty([0], device=device), device)
        self._nll_loss_helper([0, 3, 5, 7], "none", torch.empty([0, 5, 7], device=device), device)
        self._nll_loss_helper([2, 3, 0, 7], "none", torch.empty([2, 0, 7], device=device), device)
        self._nll_loss_helper([2, 3, 5, 0], "none", torch.empty([2, 5, 0], device=device), device)
        self._nll_loss_helper([2, 3, 5, 7, 0], "none", torch.empty([2, 5, 7, 0], device=device), device)

    @unittest.skipIf(TEST_WITH_UBSAN, "division-by-zero error with UBSAN")
    def test_nll_loss_empty_tensor_reduction_mean(self, device):
        nan = torch.tensor(float('nan'), device=device)
        self._nll_loss_helper([0, 3], "mean", nan, device)
        self._nll_loss_helper([0, 3, 5, 7], "mean", nan, device)
        self._nll_loss_helper([2, 3, 0, 7], "mean", nan, device)
        self._nll_loss_helper([2, 3, 5, 0], "mean", nan, device)
        self._nll_loss_helper([2, 3, 5, 7, 0], "mean", nan, device)

    def test_nll_loss_empty_tensor_reduction_sum(self, device):
        zero = torch.tensor(0, device=device)
        self._nll_loss_helper([0, 3], "sum", zero, device)
        self._nll_loss_helper([0, 3, 5, 7], "sum", zero, device)
        self._nll_loss_helper([2, 3, 0, 7], "sum", zero, device)
        self._nll_loss_helper([2, 3, 5, 0], "sum", zero, device)
        self._nll_loss_helper([2, 3, 5, 7, 0], "sum", zero, device)

    @unittest.skipIf(TEST_WITH_UBSAN, "division-by-zero error with UBSAN")
    def test_nll_loss_total_weight_is_zero(self, device):

        def helper(input_size):
            input = torch.ones(input_size, requires_grad=True, device=device)
            num_channels = input_size[1]
            target_size = (input_size[0], ) + tuple(input_size[2:])
            target = torch.zeros(target_size, dtype=torch.long, device=device)
            weight = torch.zeros([num_channels], device=device)
            self.assertEqual(F.nll_loss(input, target, weight, reduction="sum").item(), 0.)
            self.assertEqual(F.nll_loss(input, target, weight, reduction="mean").item(), float("nan"))
            self.assertEqual(F.nll_loss(input, target, weight, reduction="none"), torch.zeros(target.shape, device=device))

        helper([2, 3])
        helper([2, 3, 5, 7])
        helper([2, 3, 5, 7, 9])

    @unittest.skipIf(TEST_WITH_UBSAN, "division-by-zero error with UBSAN")
    def test_nll_loss_all_ignored(self, device):

        def helper(input_size):
            input = torch.ones(input_size, device=device)
            num_channels = input_size[1]
            target_size = (input_size[0], ) + tuple(input_size[2:])
            target = torch.zeros(target_size, dtype=torch.long, device=device)
            self.assertEqual(F.nll_loss(input, target, ignore_index=0, reduction="sum").item(), 0)
            self.assertEqual(F.nll_loss(input, target, ignore_index=0, reduction="mean").item(), float("nan"))
            self.assertEqual(F.nll_loss(input, target, ignore_index=0, reduction="none"), torch.zeros(target.shape, device=device))

        helper([2, 3])
        helper([2, 3, 5, 7])
        helper([2, 3, 5, 7, 9])

    def test_nll_loss_byte_target_matches_long(self, device):
        N, C = 10, 4
        input = torch.randn(N, C, device=device, requires_grad=True)
        target = torch.empty(N, dtype=torch.long, device=device).random_(0, C)

        def compute_result_and_gradient(reduction, target_dtype):
            input_ = input.detach()
            input_.requires_grad_()

            prob = F.log_softmax(input_, dim=-1)
            loss = nn.NLLLoss(reduction=reduction)
            result = loss(prob, target.to(target_dtype))
            result.sum().backward()

            return result, input_.grad

        for reduction in ["none", "mean", "sum"]:
            result_long, grad_long = compute_result_and_gradient(reduction, torch.long)
            result_byte, grad_byte = compute_result_and_gradient(reduction, torch.uint8)
            self.assertEqual(result_long, result_byte)
            self.assertEqual(grad_long, grad_byte)

    def test_cross_entropy_loss_prob_target_all_reductions(self, device):
        # Test with k-dimensional loss.
        for k in range(5):
            N, C = 5, 4
            other_dims = [torch.randint(2, 5, size=(1,)).item() for _ in range(k)]
            input = torch.randn(N, C, *other_dims, device=device, requires_grad=True)
            target = torch.randn(N, C, *other_dims, device=device, requires_grad=True)
            weight = torch.randn(C, device=device).abs()

            for reduction, w in product(['none', 'mean', 'sum'], [None, weight]):
                m = torch.nn.CrossEntropyLoss(weight=w, reduction=reduction)
                output = m(input, target)
                output_ref = loss_reference_fns['CrossEntropyLoss'](
                    input, target, reduction=reduction, weight=w)
                self.assertEqual(output, output_ref)

    def test_cross_entropy_loss_prob_target_unit_weights(self, device):
        # Test with k-dimensional loss.
        for k in range(5):
            N, C = 5, 4
            other_dims = [torch.randint(2, 5, size=(1,)).item() for _ in range(k)]
            input = torch.randn(N, C, *other_dims, device=device, requires_grad=True)
            target = torch.randn(N, C, *other_dims, device=device, requires_grad=True)

            for reduction in ['none', 'mean', 'sum']:
                # Ensure result with unit weights is equivalent to result without weights.
                m = torch.nn.CrossEntropyLoss(reduction=reduction)
                unit_weight = torch.ones(C, device=device, dtype=target.dtype)
                m_unit = torch.nn.CrossEntropyLoss(weight=unit_weight, reduction=reduction)
                output = m(input, target)
                output_unit = m_unit(input, target)
                self.assertEqual(output, output_unit)

    @parametrize_test('reduction', ['none', 'mean', 'sum'])
    @parametrize_test('weighted', [False, True])
    def test_cross_entropy_loss_prob_target_no_batch_dim(self, device, reduction, weighted):
        C = 5
        input = torch.randn(C, device=device).log_softmax(dim=-1)
        target = torch.randn(C, device=device).softmax(dim=-1)
        weight = torch.randn(C, device=device) if weighted else None
        m = nn.CrossEntropyLoss(reduction=reduction, weight=weight)
        loss_no_batch = m(input, target)
        loss_batch = m(input.unsqueeze(0), target.unsqueeze(0))
        if reduction == 'none':
            loss_batch = loss_batch.squeeze(0)
        self.assertEqual(loss_no_batch, loss_batch)

    def test_cross_entropy_loss_index_target_unit_weights(self, device):
        # Test with k-dimensional loss.
        for k in range(5):
            N, C = 5, 4
            other_dims = [torch.randint(2, 5, size=(1,)).item() for _ in range(k)]
            input = torch.randn(N, C, *other_dims, device=device, requires_grad=True)
            target = torch.empty(N, *other_dims, dtype=torch.long, device=device).random_(0, C)

            for reduction in ['none', 'mean', 'sum']:
                # Ensure result with unit weights is equivalent to result without weights.
                m = torch.nn.CrossEntropyLoss(reduction=reduction)
                unit_weight = torch.ones(C, device=device, dtype=input.dtype)
                m_unit = torch.nn.CrossEntropyLoss(weight=unit_weight, reduction=reduction)
                output = m(input, target)
                output_unit = m_unit(input, target)
                self.assertEqual(output, output_unit)

    def test_cross_entropy_loss_one_hot_target(self, device):
        # Test with k-dimensional loss.
        for k in range(5):
            N, C = 5, 4
            other_dims = [torch.randint(2, 5, size=(1,)).item() for _ in range(k)]
            input = torch.randn(N, C, *other_dims, device=device, requires_grad=True)
            target = torch.empty(N, *other_dims, dtype=torch.long, device=device).random_(0, C)
            weight = torch.randn(C, device=device).abs()

            # Get one-hot representation of the target.
            target_one_hot = F.one_hot(target, num_classes=C).to(input.dtype)
            # Need to put the C dim at index 1.
            target_one_hot = target_one_hot.permute(0, -1, *range(1, target_one_hot.dim() - 1))

            for reduction, w in product(['none', 'mean', 'sum'], [None, weight]):
                # Skip this case for now because soft and hard label CE are not consistent
                # in the way they apply class weights (see issue #61309).
                if reduction == 'mean' and weight is not None:
                    continue

                # Ensure loss computed with class indices matches loss
                # computed with one-hot class probs.
                m = torch.nn.CrossEntropyLoss(weight=w, reduction=reduction)
                output = m(input, target)
                output_one_hot = m(input, target_one_hot)
                self.assertEqual(output, output_one_hot)

    def test_cross_entropy_label_smoothing_errors(self, device):
        N, C = 3, 4
        input_args = [
            (torch.randn((N, C), device=device), torch.arange(0, C, device=device)),
            (torch.randn((N, C), device=device), torch.randn(N, C, device=device))
        ]
        for input_arg in input_args:
            loss = nn.CrossEntropyLoss(label_smoothing=1.2)
            with self.assertRaisesRegex(RuntimeError,
                                        r"label_smoothing must be between 0\.0"):
                loss(*input_arg)

    def test_cross_entropy_label_smoothing_consistent_index_target_and_probs(self, device):
        N, C = 10, 4
        ks = range(5)
        reductions = ['none', 'mean', 'sum']
        label_smoothings = [0.05, 0.15]

        for k, reduction, label_smoothing in product(ks, reductions, label_smoothings):
            other_dims = [torch.randint(2, 5, size=(1,)).item() for _ in range(k)]
            input = torch.randn(N, C, *other_dims, device=device, requires_grad=True)
            target = torch.empty(N, *other_dims, dtype=torch.long, device=device).random_(0, C)

            # construct target probablity that should have the same result as label_smoothing
            target_proba = F.one_hot(target, num_classes=C)
            # Need to put the C dim at index 1.
            target_proba = target_proba.permute(0, -1, *range(1, target_proba.dim() - 1))
            target_mask = (target_proba == 1)
            target_proba = target_proba.to(dtype=input.dtype)

            # y_k^ls = y_k * (1 - label_smoothing) + label_smoothing / n_classes
            # Get one-hot representation of the target.
            target_proba.masked_fill_(target_mask, 1 - label_smoothing + label_smoothing / C)
            target_proba.masked_fill_(~target_mask, label_smoothing / C)

            loss = nn.CrossEntropyLoss(reduction=reduction)
            output_with_prob = loss(input, target_proba)

            loss = nn.CrossEntropyLoss(
                reduction=reduction, label_smoothing=label_smoothing)
            output_with_index = loss(input, target)

            self.assertEqual(output_with_prob, output_with_index,
                             rtol=1e-07, atol=1e-05)

    def test_cross_entropy_label_smoothing_with_probs(self, device):
        N, C = 10, 4
        ks = range(5)
        reductions = ['none', 'mean', 'sum']
        label_smoothings = [0.05, 0.15]

        # Test with k-dimensional loss.
        for k, label_smoothing in product(ks, label_smoothings):
            other_dims = [torch.randint(2, 5, size=(1,)).item() for _ in range(k)]
            input = torch.randn(N, C, *other_dims, device=device, requires_grad=True)
            target = F.log_softmax(torch.randn(N, C, *other_dims, device=device), dim=1)

            for reduction in reductions:
                # use with label_smoothing
                loss = nn.CrossEntropyLoss(reduction=reduction, label_smoothing=label_smoothing)
                output_with_smoothing = loss(input, target)

                # manually smoothing target
                # class_proba^ls = class_proba * (1 - label_smoothing) +
                #                  label_smoothing / n_classes
                target_with_smoothing = target * (1 - label_smoothing) + label_smoothing / C
                loss = nn.CrossEntropyLoss(reduction=reduction)
                output_with_manual_smoothing = loss(input, target_with_smoothing)

                self.assertEqual(output_with_smoothing, output_with_manual_smoothing)


    def test_cross_entropy_label_smoothing_weight_ignore_indices(self, device):
        reductions = ['none', 'sum', 'mean']
        label_smoothings = [0.05, 0.15]

        weight = torch.tensor([0.3, 0.6], device=device)
        inp1 = torch.tensor([[0.3, 0.4], [1, 2]], device=device)
        inp2 = torch.tensor([[0.3, 0.6], [1, 2]], device=device)

        targ_default_ignore_index = torch.tensor([-100, 1], device=device)
        targ_negative_ignore_index = torch.tensor([-2, 1], device=device)
        targ_positive_ignore_index = torch.tensor([2, 1], device=device)

        for reduction, label_smoothing, weight in product(reductions, label_smoothings, (None, weight)):
            def check_equal(loss, inp_targ_1, inp_targ_2):
                inp1, targ1 = inp_targ_1
                inp2, targ2 = inp_targ_2
                l1 = loss(inp1, targ1)
                l2 = loss(inp2, targ2)
                self.assertEqual(l1, l2)

            # Default ignore_index
            loss = nn.CrossEntropyLoss(reduction=reduction,
                                       label_smoothing=label_smoothing,
                                       weight=weight)
            check_equal(loss, (inp1, targ_default_ignore_index), (inp2, targ_default_ignore_index))
            if reduction != 'none':
                # Check that we correctly tally the denominator for `mean`
                # i.e. we don't count the ignored_idx at all.
                check_equal(loss, (inp1, targ_default_ignore_index), (inp2[1:], targ_default_ignore_index[1:]))

            # negative ignore_index
            loss = nn.CrossEntropyLoss(reduction=reduction,
                                       label_smoothing=label_smoothing,
                                       ignore_index=-2,
                                       weight=weight)
            check_equal(loss, (inp1, targ_negative_ignore_index), (inp2, targ_negative_ignore_index))
            if reduction != 'none':
                # Check that we correctly tally the denominator for `mean`
                # i.e. we don't count the ignored_idx at all.
                check_equal(loss, (inp1, targ_negative_ignore_index), (inp2[1:], targ_negative_ignore_index[1:]))

            # positive ignore_index
            loss = nn.CrossEntropyLoss(reduction=reduction,
                                       label_smoothing=label_smoothing,
                                       ignore_index=2,
                                       weight=weight)
            check_equal(loss, (inp1, targ_positive_ignore_index), (inp2, targ_positive_ignore_index))
            if reduction != 'none':
                # Check that we correctly tally the denominator for `mean`
                # i.e. we don't count the ignored_idx at all.
                check_equal(loss, (inp1, targ_positive_ignore_index), (inp2[1:], targ_positive_ignore_index[1:]))

    # Ref: https://github.com/pytorch/pytorch/issue/85005
    @onlyCUDA
    @largeTensorTest("45GB", "cpu")
    @largeTensorTest("45GB", "cuda")
    @parametrize_test("reduction", ("none", "mean", "sum"))
    def test_cross_entropy_large_tensor(self, device, reduction):
        logits = torch.randn(int(2 ** 16), int(2 ** 16) + 1, dtype=torch.float32, device='cuda', requires_grad=True)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device='cuda')
        loss = F.cross_entropy(logits, labels, reduction=reduction)
        if reduction != "none":
            loss.backward()

        with torch.no_grad():
            logits_cpu = logits.cpu().detach().requires_grad_()
            labels_cpu = labels.cpu().detach()
        loss_cpu = F.cross_entropy(logits_cpu, labels_cpu, reduction=reduction)
        if reduction != "none":
            loss_cpu.backward()

        # workaround to reduce memory usage vs. self.assertEqual, see #84944
        rtol, atol = torch.testing._comparison.get_tolerances(torch.float32, rtol=None, atol=None)
        self.assertTrue(torch.allclose(loss.cpu(), loss_cpu, rtol=rtol, atol=atol))
        if reduction != "none":
            self.assertTrue(torch.allclose(logits.grad.cpu(), logits_cpu.grad, rtol=rtol, atol=atol))

    def test_softshrink_negative(self, device):
        input = torch.randn(5, device=device, requires_grad=True)
        m = torch.nn.Softshrink(-1)
        with self.assertRaisesRegex(RuntimeError,
                                    r'lambda must be greater or equal to 0, but found to be -1\.'):
            m(input)

    def test_fold(self, device):
        def test_dtype(fn, input, dtype):
            input = input.detach().clone().to(dtype=dtype).requires_grad_(True)
            input2 = input.detach().clone().float().requires_grad_(True)
            out = fn(input)
            out.sum().backward()
            out2 = fn(input2)
            out2.sum().backward()
            self.assertEqual(out.dtype, dtype)
            self.assertEqual(input.grad.dtype, dtype)
            self.assertEqual(out, out2.to(dtype=dtype), atol=0.05, rtol=0)
            self.assertEqual(input.grad, input2.grad.to(dtype=dtype))

        def func(x):
            return F.fold(x, output_size=(4, 5), kernel_size=(2, 2))

        seeds = (44, 83, 71, 25, 999)
        for sd in seeds:
            torch.manual_seed(sd)
            x = torch.randn(1, 12, 12, device=device, requires_grad=True)
            gradcheck(func, [x], check_forward_ad=True)
            gradgradcheck(func, [x], check_fwd_over_rev=True)
            if device == 'cpu':
                test_dtype(func, x, torch.bfloat16)


    def test_logsigmoid_out(self, device):
        # this isn't actually documented, but was broken previously:
        # https://github.com/pytorch/pytorch/issues/36499
        x = torch.randn(2, 3, device=device).t()
        empty_out = torch.randn(0, device=device)
        self.assertEqual(F.logsigmoid(x), F.logsigmoid(x, out=empty_out))

        noncontig_out = torch.randn(2, 3, device=device).t()
        self.assertEqual(F.logsigmoid(x), F.logsigmoid(x, out=noncontig_out))

    # Check that clip_grad_norm_ raises an error if the total norm of the
    # parameters' gradients is non-finite
    def test_clip_grad_norm_error_if_nonfinite(self, device):
        norms_pos = [0.1, 1, 2, 3.5, inf]
        norms_neg = [-0.1, -1, -2, -3.5]
        norms_except_0 = norms_pos + norms_neg
        norms_all = norms_except_0 + [0]

        # Each entry in test_cases has the following values, in this order:
        #
        # grad_only_one_elem    If True, only one element of the parameter's
        #                       gradient is set to the scalar grad, and the
        #                       rest of the elements are 0. If False, all grad
        #                       elements are equal to the scalar.
        #
        # prefix_finite_grad_param  If True, prefix a parameter that has a grad
        #                           of 1.
        #
        # scalars           Scalars to use as the parameter's grad, through
        #                   multiplication
        #
        # norms_nonfinite   Norm types that should produce nonfinite total norm
        #
        # norms_finite      Norm types that should produce finite total norm
        test_cases = [
            # Test errors from an infinite grad
            (False, False, [inf, -inf], norms_except_0, [0]),
            (False, True, [inf, -inf], norms_pos, norms_neg + [0]),
            (True, False, [inf, -inf], norms_pos, norms_neg + [0]),
            (True, True, [inf, -inf], norms_pos, norms_neg + [0]),

            # Test errors from a NaN grad
            (False, False, [nan], norms_except_0, [0]),
            (False, True, [nan], norms_except_0, [0]),
            (True, False, [nan], norms_except_0, [0]),
            (True, True, [nan], norms_except_0, [0]),

            # Test a grad that should never error
            (False, False, [2e22, -2e22], [], norms_all),
            (False, True, [2e22, -2e22], [], norms_all),
            (True, False, [2e22, -2e22], [], norms_all),
            (True, True, [2e22, -2e22], [], norms_all),

            # Test a grad that will overflow to inf for only some norm orders
            (False, False, [2e200, -2e200], [3.5, 2, -2, -3.5], [inf, 1, 0.1, 0, -1, -0.1]),
            (False, True, [2e200, -2e200], [3.5, 2], norms_neg + [inf, 1, 0.1, 0]),
            (True, False, [2e200, -2e200], [3.5, 2], norms_neg + [inf, 1, 0.1, 0]),
            (True, True, [2e200, -2e200], [3.5, 2], norms_neg + [inf, 1, 0.1, 0]),
        ]

        def gen_parameters(scalar, grad_only_one_elem, prefix_finite_grad_param):
            param = torch.ones(10, dtype=torch.float64, device=device, requires_grad=True)

            if grad_only_one_elem:
                param[1].mul(scalar).sum().backward()
            else:
                param.mul(scalar).sum().backward()

            if prefix_finite_grad_param:
                prefix_param = torch.ones(1, dtype=torch.float64, device=device, requires_grad=True)
                prefix_param.mul(1).sum().backward()
                parameters = [prefix_param, param]
            else:
                parameters = [param]

            return parameters

        def run_test_case(norm_type, error_if_nonfinite, scalar, grad_only_one_elem, prefix_finite_grad_param, is_norm_nonfinite):
            msg = (
                f'norm_type: {norm_type}, ',
                f'error_if_nonfinite: {error_if_nonfinite}, '
                f'scalar: {scalar}, '
                f'grad_only_one_elem: {grad_only_one_elem}, '
                f'prefix_finite_grad_param: {prefix_finite_grad_param}, '
                f'is_norm_nonfinite: {is_norm_nonfinite}')

            parameters = gen_parameters(scalar, grad_only_one_elem, prefix_finite_grad_param)

            # Should only throw an error if the total norm is expected to be
            # nonfinite and `error_if_nonfinite=True`
            if is_norm_nonfinite and error_if_nonfinite:
                error_msg = f'The total norm of order {float(norm_type)} for gradients'

                grads_before = [p.grad.clone() for p in parameters]

                with self.assertRaisesRegex(RuntimeError, error_msg, msg=msg):
                    clip_grad_norm_(parameters, 1, norm_type=norm_type, error_if_nonfinite=True)

                # Grad should not change if error is thrown
                grads_after = [p.grad for p in parameters]
                self.assertEqual(grads_before, grads_after, msg=msg)
            else:
                clip_grad_norm_(parameters, 1, norm_type=norm_type, error_if_nonfinite=error_if_nonfinite)

        for grad_only_one_elem, prefix_finite_grad_param, scalars, norms_nonfinite, norms_finite in test_cases:
            for error_if_nonfinite in [False, True]:
                for norm_type, scalar in product(norms_nonfinite, scalars):
                    run_test_case(norm_type, error_if_nonfinite, scalar, grad_only_one_elem, prefix_finite_grad_param, True)

                for norm_type, scalar in product(norms_finite, scalars):
                    run_test_case(norm_type, error_if_nonfinite, scalar, grad_only_one_elem, prefix_finite_grad_param, False)

    @onlyCUDA
    @deviceCountAtLeast(2)
    def test_clip_grad_norm_multi_device(self, devices):
        class TestModel(nn.Module):
            def __init__(self):
                super(TestModel, self).__init__()
                self.layer1 = nn.Linear(10, 10)
                self.layer2 = nn.Linear(10, 10)

        test_model = TestModel()
        test_model.layer1.to(devices[0])
        test_model.layer2.to(devices[1])
        ref_model = TestModel().to(devices[0])
        for norm_type in [2., math.inf]:
            for p in test_model.parameters():
                p.grad = torch.ones_like(p)
            for p in ref_model.parameters():
                p.grad = torch.ones_like(p)
            norm = clip_grad_norm_(test_model.parameters(), 0.5, norm_type=norm_type)
            expected = clip_grad_norm_(ref_model.parameters(), 0.5, norm_type=norm_type)
            self.assertEqual(norm, expected)
            for p, pe in zip(test_model.parameters(), ref_model.parameters()):
                self.assertEqual(p.grad.to(devices[0]), pe.grad)

    def test_elu_inplace_overlap(self, device):
        x = torch.randn((1, 6), dtype=torch.bfloat16, device=device).expand((6, 6))
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            F.elu(x, inplace=True)
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            F.elu_(x)

    # Merge into OpInfo?
    @onlyNativeDeviceTypes
    def test_elu_inplace_with_neg_alpha(self, device):
        a = torch.tensor([-1., 1.], device=device, requires_grad=True)
        b = torch.nn.functional.elu_(a.clone(), alpha=-2)
        with self.assertRaisesRegex(RuntimeError, "call out-of-place version"):
            b.backward(torch.ones(2, device=device))

        a = torch.tensor([-1., 1.], device=device, requires_grad=True)
        b = torch.nn.functional.celu_(a.clone(), alpha=-2)
        with self.assertRaisesRegex(RuntimeError, "call out-of-place version"):
            b.backward(torch.ones(2, device=device))

    @expectedFailureMeta  # https://github.com/pytorch/pytorch/issues/54897
    def test_hardswish_inplace_overlap(self, device):
        x = torch.randn((1, 6), device=device).expand((6, 6))
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            F.hardswish(x, inplace=True)

    def test_silu_inplace_overlap(self, device):
        x = torch.randn((1, 6), device=device).expand((6, 6))
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            F.silu(x, inplace=True)

    @onlyNativeDeviceTypes
    def test_mish_inplace_overlap(self, device):
        x = torch.randn((1, 6), device=device).expand((6, 6))
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            F.mish(x, inplace=True)

    def test_softplus_inplace_overlap(self, device):
        x = torch.randn((1, 6), device=device).expand((6, 6))
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            F.softplus(x, out=x)

    def test_softplus_low_threshold(self, device):
        # Ensure gradients are computed correctly with a low threshold.
        model = torch.nn.Softplus(threshold=1).double()
        input = torch.tensor(0.9, device=device, dtype=torch.double,
                             requires_grad=True)
        output = model(input)
        torch.autograd.gradcheck(model, input)

    def test_softshrink_inplace_overlap(self, device):
        x = torch.randn((1, 6), device=device).expand((6, 6))
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            F.softshrink(x, out=x)

    def test_leaky_relu_inplace_overlap(self, device):
        x = torch.randn((1, 6), device=device).expand((6, 6))
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            F.leaky_relu(x, inplace=True)
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            F.leaky_relu_(x)

    # Merge into OpInfo?
    def test_leaky_relu_inplace_with_neg_slope(self, device):
        a = torch.tensor([-1., 1.], device=device, requires_grad=True)
        b = torch.nn.functional.leaky_relu_(a.clone(), -2)
        with self.assertRaisesRegex(RuntimeError, "call out-of-place version"):
            b.backward(torch.ones(2, device=device))

        a = torch.tensor([-1., 1.], device=device, requires_grad=True)
        b = torch.nn.functional.rrelu_(a.clone(), -5.0, 1.0)
        with self.assertRaisesRegex(RuntimeError, "call out-of-place version"):
            b.backward(torch.ones(2, device=device))

    # Merge into OpInfo?
    def test_leaky_relu_inplace_with_zero_slope(self, device):
        a = torch.tensor([-2., 0., 2.], device=device, requires_grad=True)
        b = torch.nn.functional.leaky_relu_(a.clone(), 0.0)
        b.backward(torch.ones(3, device=device))
        expected = torch.tensor([0., 0., 1.], device=device)
        self.assertEqual(a.grad, expected)

        a_bf16 = torch.tensor([-2., 0., 2.], device=device, dtype=torch.bfloat16, requires_grad=True)
        b_bf16 = torch.nn.functional.leaky_relu_(a_bf16.clone(), 0.0)
        b_bf16.backward(torch.ones(3, device=device))
        expected_bf16 = torch.tensor([0., 0., 1.], device=device, dtype=torch.bfloat16)
        self.assertEqual(a_bf16.grad, expected_bf16)

    @onlyCPU
    def test_softshrink(self, device):
        x = torch.tensor([[1.21, 0.56, 0.5001, 0.4999, 1.2357, -0.4999, -0.5001, -1.154,
                           0.254, -0.24, -0.225, 0.104, 0.002, -0.001, 0.0574, 1.2344,
                           0.1748, -0.1797, -0.8125, 0.2051, -1.1328, 1.2344, -0.1562, 2.3554,
                           -0.1953, 0.0304, -0.3613, -1.3047, 1.0312, 0.1436, -0.6953, 0.5664,
                           -0.5820, -0.3301, 0.8203, 0.6133, 0.5938],
                          [-0.8203, -1.2344, -0.5234, 2.5312, -0.4551, -0.6875, -1.5547, -0.2217,
                           -0.3027, 2.6406, 1.3047, 0.2344, -1.6719, 0.2773, -1.3516, 3.4575,
                           0.4414, 0.2656, 2.1094, -1.5156, 1.2344, -0.4336, 0.6797, -3.5486,
                           0.9766, -0.4062, 1.4844, 0.7500, -1.7578, 0.7461, 1.6094, 8.5458,
                           0.3730, -0.3477, -1.0625, 0.3848, 0.0557]], device=device)
        expected = torch.tensor([[0.71, 0.06, 0.0001, 0., 0.7357, 0., -0.0001, -0.654,
                                  0., 0., 0., 0., 0., 0., 0., 0.7344,
                                  0., 0., -0.3125, 0., -0.6328, 0.7344, 0., 1.8554,
                                  0., 0., 0., -0.8047, 0.5312, 0., -0.1953, 0.0664,
                                  -0.0820, 0.0, 0.3203, 0.1133, 0.0938],
                                 [-0.3203, -0.7344, -0.0234, 2.0312, 0.0, -0.1875, -1.0547, 0.,
                                  0.0, 2.1406, 0.8047, 0., -1.1719, 0., -0.8516, 2.9575,
                                  0., 0., 1.6094, -1.0156, 0.7344, 0., 0.1797, -3.0486,
                                  0.4766, 0., 0.9844, 0.2500, -1.2578, 0.2461, 1.1094, 8.0458,
                                  0., 0., -0.5625, 0., 0.]])
        softshrink = torch.nn.Softshrink()
        out = softshrink(x)
        self.assertEqual(out, expected, atol=1e-2, rtol=0)

    def test_threshold_inplace_overlap(self, device):
        # Inplace threshold is okay, because it is idempotent
        x = torch.randn((1, 6), device=device).expand((6, 6))
        F.threshold(x, 0.5, 0.5, inplace=True)
        F.threshold_(x, 0.5, 0.5)

    @onlyNativeDeviceTypes
    def test_triplet_margin_with_distance_loss_default_parity(self, device):
        # Test for `nn.TripletMarginWithDistanceLoss` and
        # `F.triplet_margin_with_distance_loss`.  Checks
        # for parity against the respective non-distance-agnostic
        # implementations of triplet margin loss (``nn.TripletMarginLoss`
        # and `F.triplet_margin_loss`) under *default args*.

        for extra_args in \
                itertools.product((0.5, 1, 1.5), (True, False), ('none', 'mean', 'sum')):
            kwargs = {'margin': extra_args[0], 'swap': extra_args[1], 'reduction': extra_args[2]}

            anchor = torch.randn(5, 10, device=device, requires_grad=True)
            positive = torch.randn(5, 10, device=device, requires_grad=True)
            negative = torch.randn(5, 10, device=device, requires_grad=True)

            # Test forward, functional
            expected = F.triplet_margin_loss(anchor, positive, negative, **kwargs)
            actual = F.triplet_margin_with_distance_loss(anchor, positive, negative, **kwargs)
            self.assertEqual(actual, expected, rtol=1e-6, atol=1e-6)

            # Test forward, module
            loss_ref = nn.TripletMarginLoss(**kwargs)
            loss_op = nn.TripletMarginWithDistanceLoss(**kwargs)
            self.assertEqual(loss_op(anchor, positive, negative),
                             loss_ref(anchor, positive, negative),
                             rtol=1e-6, atol=1e-6)

            # Test backward
            self.assertTrue(gradcheck(lambda a, p, n: F.triplet_margin_with_distance_loss(
                a, p, n, **kwargs), (anchor, positive, negative)))
            self.assertTrue(gradcheck(lambda a, p, n: loss_op(a, p, n),
                            (anchor, positive, negative)))

    @onlyNativeDeviceTypes
    def test_triplet_margin_with_distance_loss(self, device):
        # Test for parity between `nn.TripletMarginWithDistanceLoss` and
        # `F.triplet_margin_with_distance_loss`.

        pairwise_distance = nn.PairwiseDistance()

        def cosine_distance(x, y):
            return 1.0 - F.cosine_similarity(x, y)

        distance_functions = (pairwise_distance, cosine_distance,
                              lambda x, y: 1.0 - F.cosine_similarity(x, y))

        reductions = ('mean', 'none', 'sum')
        margins = (1.0, 1.5, 0.5)
        swaps = (True, False)

        for distance_fn, reduction, margin, swap \
                in itertools.product(distance_functions, reductions, margins, swaps):
            anchor = torch.randn(5, 10, device=device, requires_grad=True)
            positive = torch.randn(5, 10, device=device, requires_grad=True)
            negative = torch.randn(5, 10, device=device, requires_grad=True)

            # Test backward
            self.assertTrue(gradcheck(lambda a, p, n: F.triplet_margin_with_distance_loss(
                a, p, n, distance_function=distance_fn, reduction=reduction, margin=margin, swap=swap),
                (anchor, positive, negative)))
            loss_op = nn.TripletMarginWithDistanceLoss(distance_function=distance_fn,
                                                       reduction=reduction, margin=margin, swap=swap)
            self.assertTrue(gradcheck(lambda a, p, n: loss_op(
                a, p, n), (anchor, positive, negative)))
            traced_loss_op = torch.jit.trace(loss_op, (anchor, positive, negative))
            self.assertTrue(gradcheck(lambda a, p, n: traced_loss_op(
                a, p, n), (anchor, positive, negative)))

            # Test forward parity
            functional = F.triplet_margin_with_distance_loss(anchor, positive, negative,
                                                             distance_function=distance_fn,
                                                             reduction=reduction, margin=margin, swap=swap)
            modular = loss_op(anchor, positive, negative)
            traced = traced_loss_op(anchor, positive, negative)
            self.assertEqual(functional, modular, atol=1e-6, rtol=1e-6)
            self.assertEqual(traced, modular, atol=1e-6, rtol=1e-6)

    def test_to_complex(self, device):
        m = nn.Linear(3, 5).to(device)
        self.assertIs(m, m.to(device))
        m.to(torch.cfloat)
        self.assertIs(m.weight.dtype, torch.cfloat)
        m.to(torch.cdouble)
        self.assertIs(m.weight.dtype, torch.cdouble)
        m.to(torch.float)
        self.assertIs(m.weight.dtype, torch.float)
        with warnings.catch_warnings(record=True) as w:
            # Trigger warning
            m.to(torch.cfloat)
            # Check warning occurs
            self.assertEqual(len(w), 1)
            self.assertTrue("Complex modules are a new feature" in str(w[-1].message))

    @skipMeta
    @dtypes(torch.float32, torch.float64)
    def test_module_to_empty(self, device, dtype):
        class MyModule(nn.Module):
            def __init__(self, in_features, out_features, device=None, dtype=None):
                super().__init__()
                factory_kwargs = {"device": device, "dtype": dtype}
                self.weight = nn.Parameter(torch.randn(in_features, out_features, **factory_kwargs))

            def forward(self, x):
                return x @ self.weight

        # Test meta module instantiation.
        input = torch.randn(5, 10, device=device, dtype=dtype)
        m = MyModule(10, 1, device='meta', dtype=dtype)
        m(input)

        # Test materializing meta module on a real device.
        m.to_empty(device=device)
        m(input)
        with torch.no_grad():
            torch.nn.init.kaiming_uniform_(m.weight)
        m(input)

        # Test creating meta module from materialized module.
        m.to_empty(device='meta')
        m(input)

    @skipMeta
    def test_skip_init(self, device):
        torch.manual_seed(1)
        m_initialized = torch.nn.Linear(5, 1)
        m_initialized.to(device)

        torch.manual_seed(1)
        m_uninitialized = torch.nn.utils.skip_init(torch.nn.Linear, 5, 1, device=device)

        self.assertEqual(m_initialized.weight.device, m_uninitialized.weight.device)
        self.assertFalse(torch.allclose(m_initialized.weight, m_uninitialized.weight))

    @dtypes(torch.float)
    @dtypesIfCUDA(torch.double, torch.float, torch.half)
    def test_transformerencoderlayer(self, device, dtype):
        # this is a deterministic test for TransformerEncoderLayer
        d_model = 4
        nhead = 2
        dim_feedforward = 16
        dropout = 0.0
        bsz = 2

        atol = 1e-5
        rtol = 1e-7
        if "cuda" in device:
            atol = 1e-3
            rtol = 1e-2

        def _test(training, batch_first, atol, rtol):
            def perm_fn(x):
                return x.transpose(1, 0) if batch_first else x

            model = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                               batch_first=batch_first, device=device, dtype=dtype)

            if not training:
                assert dropout == 0
                model = model.eval()

            # set constant weights of the model
            for idx, p in enumerate(model.parameters()):
                x = p.data
                sz = x.view(-1).size(0)
                shape = x.shape
                x = torch.cos(torch.arange(0, sz).float().view(shape))
                p.data.copy_(x)

            # deterministic input
            encoder_input = torch.tensor([[[20., 30., 40., 50.]]], device=device, dtype=dtype)
            result = model(encoder_input)
            ref_output = torch.tensor([[[2.258703, 0.127985, -0.697881, 0.170862]]], device=device, dtype=dtype)
            self.assertEqual(result.shape, ref_output.shape)
            torch.testing.assert_close(result, ref_output, atol=atol, rtol=rtol)
            # 0 values are NOT masked. This shouldn't mask anything.
            mask = torch.tensor([[0]], device=device) == 1
            # TODO: enable fast path for calls with a mask!
            result = model(encoder_input, src_key_padding_mask=mask)
            self.assertEqual(result.shape, ref_output.shape)
            torch.testing.assert_close(result, ref_output, atol=atol, rtol=rtol)
            # 1 values are masked. Since there is only 1 input embedding this
            # will result in nan.
            mask = torch.tensor([[1]], device=device) == 1
            result = model(encoder_input, src_key_padding_mask=mask)
            result = result.cpu().detach().numpy()
            self.assertTrue(np.isnan(result).all())

            # deterministic input
            encoder_input = perm_fn(torch.tensor([[[1., 2., 3., 4.]],
                                                  [[5., 6., 7., 8.]]], device=device, dtype=dtype))
            result = model(encoder_input)
            ref_output = perm_fn(torch.tensor([[[2.272644, 0.119035, -0.691669, 0.153486]],
                                               [[2.272644, 0.119035, -0.691669, 0.153486]]], device=device, dtype=dtype))
            self.assertEqual(result.shape, ref_output.shape)
            torch.testing.assert_close(result, ref_output, atol=atol, rtol=rtol)
            # all 0 which is no masking
            mask = torch.tensor([[0, 0]], device=device) == 1
            result = model(encoder_input, src_key_padding_mask=mask)
            self.assertEqual(result.shape, ref_output.shape)
            torch.testing.assert_close(result, ref_output, atol=atol, rtol=rtol)
            mask = torch.tensor([[1, 0]], device=device) == 1
            result = model(encoder_input, src_key_padding_mask=mask)
            ref_output = perm_fn(torch.tensor([[[2.301516, 0.092249, -0.679101, 0.103088]],
                                               [[2.301516, 0.092249, -0.679101, 0.103088]]], device=device, dtype=dtype))
            self.assertEqual(result.shape, ref_output.shape)
            torch.testing.assert_close(result, ref_output, atol=atol, rtol=rtol)

            # deterministic input
            encoder_input = perm_fn(torch.tensor([[[0.7462, 0.6653, 0.5679, 0.4891],
                                                   [0.5387, 0.1655, 0.3565, 0.0471]],
                                                  [[0.8335, 0.2799, 0.5031, 0.2947],
                                                   [0.1402, 0.0318, 0.7636, 0.1346]],
                                                  [[0.6333, 0.9344, 0.1376, 0.9938],
                                                   [0.8924, 0.2872, 0.6692, 0.2944]],
                                                  [[0.9897, 0.6915, 0.3154, 0.1733],
                                                   [0.8645, 0.3513, 0.3064, 0.0767]],
                                                  [[0.8117, 0.2366, 0.4838, 0.7881],
                                                   [0.3718, 0.4945, 0.9511, 0.0864]]], device=device, dtype=dtype))
            result = model(encoder_input)
            ref_output = perm_fn(torch.tensor([[[2.428589, 0.020835, -0.602055, -0.085249],
                                                [2.427987, 0.021213, -0.602496, -0.084103]],
                                               [[2.424689, 0.019155, -0.604793, -0.085672],
                                                [2.413863, 0.022211, -0.612486, -0.072490]],
                                               [[2.433774, 0.021598, -0.598343, -0.087548],
                                                [2.425104, 0.019748, -0.604515, -0.084839]],
                                               [[2.436185, 0.022682, -0.596625, -0.087261],
                                                [2.433556, 0.021891, -0.598509, -0.086832]],
                                               [[2.416246, 0.017512, -0.610712, -0.082961],
                                                [2.422901, 0.024187, -0.606178, -0.074929]]], device=device, dtype=dtype))
            self.assertEqual(result.shape, ref_output.shape)
            torch.testing.assert_close(result, ref_output, atol=atol, rtol=rtol)

            # all 0
            mask = torch.zeros([2, 5], device=device) == 1
            result = model(encoder_input, src_key_padding_mask=mask)
            self.assertEqual(result.shape, ref_output.shape)
            torch.testing.assert_close(result, ref_output, atol=atol, rtol=rtol)
            mask[0, 1] = 1
            mask[1, 3] = 1
            mask[1, 4] = 1
            result = model(encoder_input, src_key_padding_mask=mask)
            ref_output = perm_fn(torch.tensor([[[2.429026, 0.020793, -0.601741, -0.085642],
                                                [2.428811, 0.021445, -0.601912, -0.084252]],
                                               [[2.425009, 0.019155, -0.604566, -0.085899],
                                                [2.415408, 0.02249 , -0.611415, -0.073]],
                                               [[2.434199, 0.021682, -0.598039, -0.087699],
                                                [2.42598, 0.019941, -0.603896, -0.085091]],
                                               [[2.436457, 0.022736, -0.59643 , -0.08736],
                                                [2.434021, 0.022093, -0.598179, -0.08679]],
                                               [[2.416531, 0.017498, -0.610513, -0.083181],
                                                [2.4242, 0.024653, -0.605266, -0.074959]]], device=device, dtype=dtype))
            self.assertEqual(result.shape, ref_output.shape)
            torch.testing.assert_close(result, ref_output, atol=atol, rtol=rtol)

            # NestedTensor is only supported for the fast path
            # currently, which won't be used if training.
            if (batch_first and not training and
                    ('cuda' in str(device) or 'cpu' in str(device)) and not TEST_WITH_CROSSREF):
                encoder_input[0][-1] = torch.zeros_like(encoder_input[0][1])
                mask = torch.zeros(encoder_input.shape[:-1], device=device, dtype=torch.bool)
                mask[0][-1] = True

                nt = torch.nested.nested_tensor([encoder_input[0][:-1], encoder_input[1]], device=device)
                result = model(nt)
                ref_output = torch.tensor(
                    [
                        [
                            [2.4268184, 0.02042419, -0.603311, -0.08476824],
                            [2.423306, 0.01889652, -0.6057701, -0.08519465],
                            [2.431538, 0.02078694, -0.5999354, -0.08746159],
                            [2.4348664, 0.02212971, -0.5975677, -0.08733892],
                            [2.423133, 0.02097577, -0.60594773, -0.08113337],
                        ],
                        [
                            [2.4279876, 0.02121329, -0.60249615, -0.08410317],
                            [2.4138637, 0.02221113, -0.6124869, -0.07249016],
                            [2.4251041, 0.01974815, -0.6045152, -0.08483928],
                            [2.4335563, 0.0218913, -0.59850943, -0.08683228],
                            [2.4229012, 0.02418739, -0.6061784, -0.07492948],
                        ],
                    ],
                    device=device, dtype=dtype
                )
                result = result.to_padded_tensor(0)
                ref_output[0][-1] = torch.zeros_like(
                    ref_output[0][-1], device=device, dtype=dtype
                )
                result[0][-1] = torch.zeros_like(
                    result[0][-1], device=device, dtype=dtype
                )
                self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
                if 'cuda' in device:
                    if dtype == torch.float:
                        atol = 2e-4
                        rtol = 4e-3
                    else:
                        atol = 7e-4
                        rtol = 2e-2
                    torch.testing.assert_close(result, ref_output, atol=atol, rtol=rtol)
                else:
                    torch.testing.assert_close(result, ref_output)


        for batch_first in (True, False):
            for training in (True, False):
                if training:
                    cm = contextlib.nullcontext()
                else:
                    # Fast path requires inference mode.
                    cm = torch.no_grad()
                with cm:
                    _test(batch_first=batch_first, training=training, atol=atol, rtol=rtol)

    @dtypes(torch.double)
    @torch.no_grad()
    def test_multihead_attn_fast_path_query_and_bias_have_different_dtypes(self, device, dtype):
        mha = torch.nn.MultiheadAttention(4, 4, batch_first=True, dtype=dtype, device=device).eval()
        mha.in_proj_bias = torch.nn.Parameter(mha.in_proj_bias.to(torch.half).to(device))
        query = torch.randn(4, 4, 4, dtype=dtype, device=device)
        mha(query, query, query)

    @dtypes(torch.double)
    @torch.no_grad()
    def test_multihead_attn_fast_path_small_test(self, device, dtype):
        mha = torch.nn.MultiheadAttention(4, 4, batch_first=True, dtype=dtype, device=device).eval()
        query = torch.randn(4, 4, 4, dtype=dtype, device=device)
        mha(query, query, query)

    @dtypes(torch.double)
    @torch.no_grad()
    def test_multihead_attn_in_proj_bias_none(self, device, dtype):
        mha = torch.nn.MultiheadAttention(2, 2, bias=False, dtype=dtype, device=device)
        query = torch.rand(2, 2, 2, dtype=dtype, device=device)
        mha(query, query, query)

    @dtypes(torch.double)
    @torch.no_grad()
    def test_multihead_attn_in_proj_weight_none(self, device, dtype):
        # Setting kdim == vdim == 2 means that vdim != embed_dim
        # will cause the logic to use per-input project weights, thereby
        # forcing self.in_proj_weight = None
        mha = torch.nn.MultiheadAttention(4, 4, vdim=2, kdim=2, dtype=dtype, device=device)
        query = torch.rand(4, 4, 4, dtype=dtype, device=device)
        key = torch.rand(4, 4, 2, dtype=dtype, device=device)
        mha(query, key, key)

    @onlyCPU
    @dtypes(torch.double)
    def test_transformerencoderlayer_fast_path(self, device, dtype):
        """
        Test transformer fast path on CPU with different valid mask types and shapes
        """
        model = torch.nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True, device=device, dtype=dtype)
        model.eval()

        # Batched inputs
        src = torch.rand(32, 10, 512)

        # Attention mask of shape (src_len, src_len)
        src_mask = torch.zeros(10, 10).to(torch.bool)
        with torch.no_grad():
            model(src, src_mask=src_mask)

        # Padding mask of shape (batch_size, src_len)
        src_key_padding_mask = torch.zeros(32, 10).to(torch.bool)
        with torch.no_grad():
            model(src, src_key_padding_mask=src_key_padding_mask)


    @dtypes(torch.float)
    @dtypesIfCUDA(torch.half, torch.float)
    def test_transformerencoderlayer_gelu(self, device, dtype):
        # this is a deterministic test for TransformerEncoderLayer with gelu activation
        d_model = 4
        nhead = 2
        dim_feedforward = 16
        dropout = 0.0
        bsz = 2

        atol = 0
        rtol = 1e-5
        if "cuda" in device:
            atol = 1e-3
            rtol = 1e-2

        def _test(activation, batch_first, training):
            def perm_fn(x):
                return x.transpose(1, 0) if batch_first else x

            model = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                               activation, batch_first=batch_first, device=device, dtype=dtype)
            if not training:
                assert dropout == 0
                model = model.eval()

            # set constant weights of the model
            for idx, p in enumerate(model.parameters()):
                x = p.data
                sz = x.view(-1).size(0)
                shape = x.shape
                x = torch.cos(torch.arange(0, sz).float().view(shape))
                p.data.copy_(x)

            # deterministic input
            encoder_input = torch.tensor([[[20., 30., 40., 50.]]], device=device, dtype=dtype)
            result = model(encoder_input)
            ref_output = torch.tensor([[[2.249815, 0.131006, -0.702199, 0.177868]]], device=device, dtype=dtype)
            torch.testing.assert_close(result, ref_output, rtol=rtol, atol=atol)

            # deterministic input
            encoder_input = perm_fn(torch.tensor([[[1., 2., 3., 4.]],
                                                  [[5., 6., 7., 8.]]], device=device, dtype=dtype))
            result = model(encoder_input)
            ref_output = perm_fn(torch.tensor([[[2.264103, 0.121417, -0.696012, 0.159724]],
                                               [[2.264103, 0.121417, -0.696012, 0.159724]]], device=device, dtype=dtype))
            torch.testing.assert_close(result, ref_output, rtol=rtol, atol=atol)

            # deterministic input
            encoder_input = perm_fn(torch.tensor([[[0.7462, 0.6653, 0.5679, 0.4891],
                                                  [0.5387, 0.1655, 0.3565, 0.0471]],
                                                  [[0.8335, 0.2799, 0.5031, 0.2947],
                                                  [0.1402, 0.0318, 0.7636, 0.1346]],
                                                  [[0.6333, 0.9344, 0.1376, 0.9938],
                                                  [0.8924, 0.2872, 0.6692, 0.2944]],
                                                  [[0.9897, 0.6915, 0.3154, 0.1733],
                                                  [0.8645, 0.3513, 0.3064, 0.0767]],
                                                  [[0.8117, 0.2366, 0.4838, 0.7881],
                                                  [0.3718, 0.4945, 0.9511, 0.0864]]], device=device, dtype=dtype))
            result = model(encoder_input)
            ref_output = perm_fn(torch.tensor([[[2.42163188, 0.03227153, -0.60714219, -0.05908082],
                                                [2.42151276, 0.03302179, -0.60722523, -0.05762651]],
                                               [[2.41926761, 0.02974034, -0.60879519, -0.0621269],
                                                [2.41626395, 0.03539356, -0.61087842, -0.04978623]],
                                               [[2.42382808, 0.03218872, -0.6055963, -0.06073591],
                                                [2.41983477, 0.03085259, -0.60840145, -0.06046414]],
                                               [[2.42500749, 0.03328855, -0.60476388, -0.0595334],
                                                [2.4237977, 0.03290575, -0.60561789, -0.05940082]],
                                               [[2.41383916, 0.02686345, -0.61256377, -0.06380707],
                                                [2.42000277, 0.03800944, -0.60824798, -0.04754947]]], device=device, dtype=dtype))
            torch.testing.assert_close(result, ref_output, rtol=rtol, atol=atol)
        for activation, batch_first, training in product(('gelu', F.gelu, nn.GELU()), (True, False), (True, False)):
            # Fast path requires inference mode.
            if training:
                cm = contextlib.nullcontext()
            else:
                cm = torch.no_grad()
            with cm:
                _test(activation=activation, batch_first=batch_first, training=training)


class TestModuleGlobalHooks(TestCase):

    def tearDown(self):
        nn.modules.module._global_backward_hooks = OrderedDict()
        nn.modules.module._global_forward_hooks = OrderedDict()
        nn.modules.module._global_forward_pre_hooks = OrderedDict()

    @skipIfTorchDynamo("TorchDynamo does not work well with hooks")
    def test_module_global_hooks(self):
        module = nn.Sigmoid

        module_1 = module()
        module_2 = module()
        module_3 = module()

        input = torch.ones(5, 5, requires_grad=True)

        counter = {
            'forwards': 0,
            'backwards': 0
        }

        def fw_hook(inc, h_module, input, output):
            self.assertIsInstance(input, tuple)
            self.assertTrue(isinstance(output, torch.Tensor))
            self.assertTrue(isinstance(h_module, module))
            self.assertEqual(input[0], torch.ones(5, 5))
            self.assertEqual(output, torch.empty(5, 5).fill_(1 / (1 + 1 / math.e)))
            counter['forwards'] += inc

        def bw_hook(inc, h_module, grad_input, grad_output):
            self.assertIsInstance(grad_input, tuple)
            self.assertIsInstance(grad_output, tuple)
            self.assertTrue(isinstance(h_module, module))
            self.assertEqual(grad_output[0], torch.ones(5, 5) * 2)
            counter['backwards'] += inc

        test_fwd = nn.modules.module.register_module_forward_hook(lambda *args: fw_hook(1, *args))

        module_1(input)
        module_2(input)
        module_3(input)
        self.assertEqual(counter['forwards'], 3)
        self.assertEqual(counter['backwards'], 0)

        test_bwd = nn.modules.module.register_module_backward_hook(
            lambda *args: bw_hook(1, *args))

        output_1 = module_1(input)
        output_2 = module_2(input)
        output_3 = module_3(input)
        self.assertEqual(counter['forwards'], 6)
        self.assertEqual(counter['backwards'], 0)

        output_1.backward(torch.ones(5, 5) * 2, retain_graph=True)
        output_2.backward(torch.ones(5, 5) * 2, retain_graph=False)
        output_3.backward(torch.ones(5, 5) * 2, retain_graph=False)
        self.assertEqual(counter['forwards'], 6)
        self.assertEqual(counter['backwards'], 3)

        output_1.backward(torch.ones(5, 5) * 2, retain_graph=True)
        self.assertEqual(counter['forwards'], 6)
        self.assertEqual(counter['backwards'], 4)

        test2_fwd = nn.modules.module.register_module_forward_hook(lambda *args: fw_hook(2, *args))

        output = module_1(input)
        output = module_2(input)
        output = module_3(input)
        self.assertEqual(counter['forwards'], 15)
        self.assertEqual(counter['backwards'], 4)

        test2_bwd = nn.modules.module.register_module_backward_hook(lambda *args: bw_hook(2, *args))

        module_1(input).backward(torch.ones(5, 5) * 2)
        self.assertEqual(counter['forwards'], 18)
        self.assertEqual(counter['backwards'], 7)

        test2_bwd.remove()

        module_2(input).backward(torch.ones(5, 5) * 2)
        self.assertEqual(counter['forwards'], 21)
        self.assertEqual(counter['backwards'], 8)

        test2_fwd.remove()

        module_3(input).backward(torch.ones(5, 5) * 2)
        self.assertEqual(counter['forwards'], 22)
        self.assertEqual(counter['backwards'], 9)

        test_fwd.remove()
        test_bwd.remove()

    def test_module_global_hook_invalid_outputs(self):
        module = nn.Sigmoid()
        input = torch.randn(5, 5, requires_grad=True)

        def bw_fail1(self, grad_input, grad_output):
            return grad_input[:-1]

        def bw_fail2(self, grad_input, grad_output):
            return grad_input + (torch.randn(2, 2),)

        with nn.modules.module.register_module_backward_hook(bw_fail1):
            with self.assertRaisesRegex(RuntimeError, 'got 0, but expected 1'):
                module(input).sum().backward()

        with nn.modules.module.register_module_backward_hook(bw_fail2):
            with self.assertRaisesRegex(RuntimeError, 'got 2, but expected 1'):
                module(input).sum().backward()

    @skipIfTorchDynamo("https://github.com/pytorch/torchdynamo/issues/847")
    def test_module_backward_global_hook_writeable(self):
        module = nn.Sigmoid()
        input = torch.randn(5, 5, requires_grad=True)
        sig_x = torch.sigmoid(input)

        def bw_hook(module, grad_input, grad_output):
            for grad in grad_input:
                self.assertTrue(isinstance(grad, torch.Tensor))
            for grad in grad_output:
                self.assertTrue(isinstance(grad, torch.Tensor))
            return tuple(gi * 2 for gi in grad_input)

        nn.modules.module.register_module_backward_hook(bw_hook)
        module(input).backward(torch.ones(5, 5))
        expected_grad = sig_x * (1 - sig_x) * 2
        self.assertEqual(input.grad, expected_grad)

    @skipIfTorchDynamo("TorchDynamo does not work well with hooks")
    def test_module_global_forward_preforward_hook_writeable(self):
        module = nn.Sigmoid()
        input = torch.randn(5, 5, requires_grad=True)
        sig_x = torch.sigmoid(input)

        def forward_pre_hook(m, input):
            return torch.nn.functional.relu(input[0])

        def forward_hook(m, input, output):
            return -output

        nn.modules.module.register_module_forward_pre_hook(forward_pre_hook)
        nn.modules.module.register_module_forward_hook(forward_hook)
        output = module(input)
        expected_res = -torch.sigmoid(torch.nn.functional.relu(input))
        self.assertEqual(output, expected_res)
        output.backward(torch.ones(5, 5) * 2, retain_graph=True)
        mask = (input > 0).double()
        expected_grad = -sig_x * (1 - sig_x) * 2 * mask
        self.assertEqual(input.grad, expected_grad)

    @skipIfTorchDynamo("TorchDynamo does not work well with hooks")
    def test_module_forward_preforward_hook_removable(self):
        """
        This test is to test when multiple pre-forward hook functions can be
        registered successfully and used correctly, if the handle can be removable
        during the pre-forward hook function call.
        """
        module = nn.Sigmoid()

        def removable_hook(m, input):
            nonlocal handle
            handle.remove()
            return input

        def removable_hook_2(m, input):
            nonlocal handle_2
            handle_2.remove()
            return input

        handle = module.register_forward_pre_hook(removable_hook)
        handle_2 = module.register_forward_pre_hook(removable_hook_2)

        # make sure hook register is successful
        self.assertEqual(len(handle.hooks_dict_ref()), 2)
        self.assertEqual(len(handle_2.hooks_dict_ref()), 2)

        input = torch.randn(2, 2)
        output = module(input)
        self.assertEqual(torch.sigmoid(input), output)

        # make sure hook removal is successful
        self.assertFalse(handle.id in handle.hooks_dict_ref())
        self.assertFalse(handle_2.id in handle.hooks_dict_ref())
        self.assertEqual(len(handle.hooks_dict_ref()), 0)
        self.assertEqual(len(handle_2.hooks_dict_ref()), 0)

    @skipIfTorchDynamo("TorchDynamo does not work well with hooks")
    def test_module_forward_forward_hook_removable(self):
        """
        This test is to test when multiple forward hook functions can be registered
        successfully and used correctly, if the handle can be removable during the
        forward hook function call.
        """
        module = nn.Sigmoid()

        def removable_hook(m, input, output):
            nonlocal handle
            handle.remove()
            return output

        def removable_hook_2(m, input, output):
            nonlocal handle_2
            handle_2.remove()
            return output

        handle = module.register_forward_hook(removable_hook)
        handle_2 = module.register_forward_hook(removable_hook_2)

        # make sure hook register is successful
        self.assertEqual(len(handle.hooks_dict_ref()), 2)
        self.assertEqual(len(handle_2.hooks_dict_ref()), 2)

        input = torch.randn(2, 2)
        output = module(input)
        self.assertEqual(torch.sigmoid(input), output)

        # make sure hook removal is successful
        self.assertFalse(handle.id in handle.hooks_dict_ref())
        self.assertFalse(handle_2.id in handle.hooks_dict_ref())
        self.assertEqual(len(handle.hooks_dict_ref()), 0)
        self.assertEqual(len(handle_2.hooks_dict_ref()), 0)

    @skipIfTorchDynamo("TorchDynamo does not work well with hooks")
    def test_global_and_local_hooks_order(self):
        module = nn.Sigmoid()

        global_forward_pre_called = False
        local_forward_pre_called = False
        global_forward_called = False
        local_forward_called = False
        global_backward_called = False
        local_backward_called = False

        def global_forward_pre_hook(m, input):
            nonlocal global_forward_pre_called
            self.assertTrue(not local_forward_pre_called)
            global_forward_pre_called = True
            return input

        def local_forward_pre_hook(m, input):
            nonlocal local_forward_pre_called
            self.assertTrue(global_forward_pre_called)
            local_forward_pre_called = True
            return input

        def global_forward_hook(m, input, output):
            nonlocal global_forward_called
            self.assertTrue(not local_forward_called)
            global_forward_called = True
            return output

        def local_forward_hook(m, input, output):
            nonlocal local_forward_called
            self.assertTrue(global_forward_called)
            local_forward_called = True
            return output

        def global_backward_hook(m, input, output):
            nonlocal global_backward_called
            self.assertTrue(not local_backward_called)
            global_backward_called = True
            return input

        def local_backward_hook(m, input, output):
            nonlocal local_backward_called
            self.assertTrue(global_backward_called)
            local_backward_called = True
            return input

        input = torch.randn(5, 5, requires_grad=True)
        nn.modules.module.register_module_forward_pre_hook(global_forward_pre_hook)
        module.register_forward_pre_hook(local_forward_pre_hook)
        nn.modules.module.register_module_forward_hook(global_forward_hook)
        module.register_forward_hook(local_forward_hook)
        nn.modules.module.register_module_backward_hook(global_backward_hook)
        module.register_backward_hook(local_backward_hook)

        output = module(input)
        self.assertTrue(local_forward_called and local_forward_pre_called and global_forward_called and global_forward_pre_called)

        output.backward(torch.ones(5, 5), retain_graph=True)
        self.assertTrue(local_backward_called and global_backward_called)


class TestFunctionalPickle(TestCase):

    # issue gh-38137
    def test_pickle_softsign(self):
        # Make sure it does not throw an exception
        s = pickle.dumps(F.softsign)

def _hook_to_pickle(*args, **kwargs):
    pass

class TestStateDictHooks(TestCase):

    def test_load_state_dict_pre_hook(self):

        m = nn.Linear(10, 10)
        m_state_dict = m.state_dict()

        m_load = nn.Linear(10, 10)

        hook_called = 0

        def hook_without_module(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            self.assertEqual(m_state_dict, state_dict)
            nonlocal hook_called
            hook_called += 1

        def hook_with_module(module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            self.assertEqual(m_state_dict, state_dict)
            self.assertTrue(m_load is module)
            nonlocal hook_called
            hook_called += 1

        hook_called = 0
        m_load._register_load_state_dict_pre_hook(hook_without_module)
        m_load.load_state_dict(m_state_dict)
        self.assertEqual(1, hook_called)

        hook_called = 0
        m_load._register_load_state_dict_pre_hook(hook_with_module, True)
        m_load.load_state_dict(m_state_dict)
        self.assertEqual(2, hook_called)

    def test_no_extra_ref_to_module(self):
        try:
            gc.disable()
            m = nn.Linear(10, 10)

            m._register_load_state_dict_pre_hook(_hook_to_pickle, True)
            weak_m = weakref.ref(m)
            del m

            self.assertEqual(weak_m(), None)
        finally:
            gc.enable()

    def test_pickled_hook(self):
        m = nn.Linear(10, 10)
        m._register_load_state_dict_pre_hook(_hook_to_pickle, True)
        pickle.loads(pickle.dumps(m))

    def test_load_state_dict_module_pre_hook(self):
        hook_called = 0

        # Test with module instance method as hook
        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.foo = torch.nn.Parameter(torch.rand(10))

            def my_pre_load_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
                assert [] == error_msgs
                assert [] == unexpected_keys
                assert [] == missing_keys
                assert strict
                nonlocal hook_called
                hook_called += 1

            def my_pre_load_hook_with_module(
                self,
                module,
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            ):
                assert [] == error_msgs
                assert [] == unexpected_keys
                assert [] == missing_keys
                assert strict
                assert self is module
                nonlocal hook_called
                hook_called += 1

        # Test that hooks registered on a submodule are also called
        # appropriately, i.e. with the submodule as module argument in
        # my_pre_load_hook_with_module.
        class MyModuleContainer(nn.Module):
            def __init__(self, mod):
                super().__init__()
                self.mod = mod

        for ctor in [MyModuleContainer, lambda x: x]:
            m = ctor(MyModule())
            state_dict = m.state_dict()
            if isinstance(m, MyModuleContainer):
                mod = m.mod
            else:
                mod = m

            hook_called = 0
            mod._register_load_state_dict_pre_hook(
                mod.my_pre_load_hook
            )
            m.load_state_dict(state_dict)
            self.assertEqual(1, hook_called)

            hook_called = 0
            mod._register_load_state_dict_pre_hook(
                mod.my_pre_load_hook_with_module, True
            )
            m.load_state_dict(state_dict)
            self.assertEqual(2, hook_called)

    def test_load_state_dict_post_hook(self):
        hook_called = 0

        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.foo = torch.nn.Parameter(torch.rand(10))

            def my_post_load_hook(self, module, incompatible_keys):
                assert module is self
                nonlocal hook_called
                incompatible_keys.missing_keys.append("foo")
                incompatible_keys.unexpected_keys.append("bar")
                hook_called += 1

        nested = MyModule()
        wrapped = nn.ModuleList([nested])
        handle = nested.register_load_state_dict_post_hook(
            nested.my_post_load_hook,
        )
        # Hook must be called even if it is wrapped
        ret = wrapped.load_state_dict(wrapped.state_dict(), strict=False)
        self.assertEqual(hook_called, 1)
        # Ensure that the hook modified missing_keys and unexpected_keys
        missing = ret.missing_keys
        unexpected = ret.unexpected_keys
        self.assertEqual(missing, ["foo"])
        self.assertEqual(unexpected, ["bar"])
        # When called with strict=True, the error raised should mention the
        # missing and unexpected keys the hook added.
        with self.assertRaisesRegex(RuntimeError, "foo.*\n.*bar"):
            wrapped.load_state_dict(wrapped.state_dict(), strict=True)
        self.assertEqual(hook_called, 2)
        # Removing the hook via handle.remove() should cause it not to
        # fire anymore.
        handle.remove()
        # Hook did not run so it should not have added any keys
        ret = wrapped.load_state_dict(wrapped.state_dict(), strict=False)
        self.assertEqual(ret.missing_keys, [])
        self.assertEqual(ret.unexpected_keys, [])
        # hook_called should not have been incremented
        self.assertEqual(hook_called, 2)

        def load_hook_clear_incompatible(module, incompatible_keys):
            incompatible_keys.missing_keys.clear()
            incompatible_keys.unexpected_keys.clear()

        nested.register_load_state_dict_post_hook(load_hook_clear_incompatible)
        state_dict = wrapped.state_dict()
        state_dict["extra"] = torch.ones(1)
        # load state_dict with strict=True should not throw.
        ret = wrapped.load_state_dict(state_dict, strict=True)
        # explicitly ensure that the post hook clearned out incompatible_keys
        self.assertEqual([], ret.missing_keys)
        self.assertEqual([], ret.unexpected_keys)

    @unittest.skipIf(IS_WINDOWS, "Tempfile permission issue on windows")
    def test_load_state_dict_post_hook_backward_compatibility(self):
        def my_post_load_hook(mod, _):
            nonlocal called
            called = True

        for m in [nn.Softmin(10), nn.Softmax(10), nn.LogSoftmax(10)]:
            called = False
            sd = deepcopy(m.state_dict())
            self.assertTrue(hasattr(m, '_load_state_dict_post_hooks'))
            # Simulate an older model that did not have this attr
            delattr(m, '_load_state_dict_post_hooks')
            # Save and load, and ensure that load_state_dict works (without proper
            # BC we would run into errors because this attribute would be expected).
            # In particular, Softmax runs into the issue described here:
            # https://github.com/pytorch/pytorch/issues/77280
            with NamedTemporaryFile() as f:
                # Note that torch.save / torch.load is not recommended to save/load
                # modules.
                torch.save(m, f.name)
                m = torch.load(f.name)
                m.load_state_dict(sd)
                self.assertFalse(called)

            # Ensure hooks can be registered and called.
            m.register_load_state_dict_post_hook(my_post_load_hook)
            m.load_state_dict(sd)
            self.assertTrue(called)


instantiate_device_type_tests(TestNNDeviceType, globals())
instantiate_parametrized_tests(TestNN)

if __name__ == '__main__':
    run_tests()
