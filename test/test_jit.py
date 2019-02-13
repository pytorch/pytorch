from __future__ import division
import torch
import torch.jit
import torch.nn as nn
import torch.nn.functional as F
import torch.jit.quantized
from contextlib import contextmanager
from itertools import product, chain
import torch.jit.frontend
from torch.autograd import Variable, Function
from torch.autograd.function import traceable
from torch.testing import assert_allclose
from torch.onnx import OperatorExportTypes
from torch._six import inf, PY2, builtins
from common_utils import TestCase, run_tests, IS_WINDOWS, TEST_WITH_UBSAN, \
    skipIfRocm, skipIfNoLapack, suppress_warnings, load_tests, IS_SANDCASTLE, \
    freeze_rng_state, set_rng_seed
from common_nn import module_tests, new_module_tests, criterion_tests
from textwrap import dedent
from functools import wraps
import os
import io
import itertools
import sys
import unittest
import inspect
import textwrap
import numpy as np
import tempfile
import shutil
import warnings
import math
import types
import pickle
import copy

from common_methods_invocations import method_tests as autograd_method_tests
from common_methods_invocations import create_input, unpack_variables, \
    exclude_tensor_method, non_differentiable, EXCLUDE_GRADCHECK, EXCLUDE_FUNCTIONAL
from copy import deepcopy
import random
from typing import List, Optional
from torch.jit.frontend import NotSupportedError
from torch.jit import BatchTensor

# For testing truediv in python 2
from test_module.future_div import div_int_future, div_float_future
from test_module.no_future_div import div_int_nofuture, div_float_nofuture


# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

try:
    import torchvision
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")

RUN_CUDA = torch.cuda.is_available()
RUN_CUDA_HALF = RUN_CUDA
if torch.cuda.is_available():
    CUDA_VERSION = torch._C._cuda_getCompiledVersion()
    for d in range(torch.cuda.device_count()):
        major = torch.cuda.get_device_capability(d)[0]
        if (CUDA_VERSION < 8000 and major >= 6) or (CUDA_VERSION < 9000 and major >= 7):
            RUN_CUDA = False
        if (CUDA_VERSION < 9000 or major < 6):
            RUN_CUDA_HALF = False

RUN_CUDA_MULTI_GPU = RUN_CUDA and torch.cuda.device_count() > 1

PY35 = sys.version_info >= (3, 5)
WINDOWS = sys.platform == 'win32'


if WINDOWS:
    @contextmanager
    def TemporaryFileName():
        # Ideally we would like to not have to manually delete the file, but NamedTemporaryFile
        # opens the file, and it cannot be opened multiple times in Windows. To support Windows,
        # close the file after creation and try to remove it manually
        f = tempfile.NamedTemporaryFile(delete=False)
        try:
            f.close()
            yield f.name
        finally:
            os.unlink(f.name)
else:
    @contextmanager
    def TemporaryFileName():
        with tempfile.NamedTemporaryFile() as f:
            yield f.name


def LSTMCellF(input, hx, cx, *params):
    return LSTMCell(input, (hx, cx), *params)


def LSTMCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
    hx, cx = hidden
    gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * torch.tanh(cy)
    return hy, cy


def LSTMCellC(*args, **kwargs):
    hy, cy = LSTMCellF(*args, **kwargs)
    return torch.cat((hy, cy))


def LSTMCellS(x, hx, cx, w_ih, w_hh, b_ih, b_hh):
    gates = x.mm(w_ih.t()) + hx.mm(w_hh.t()) + b_ih + b_hh
    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)
    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * torch.tanh(cy)
    return hy, cy


# Code reference: https://github.com/pytorch/translate/blob/master/pytorch_translate/rnn_cell.py#L27:44
def MiLSTMCell(x, hx, cx, w_ih, w_hh, alpha, beta_i, beta_h, bias):
    Wx = x.mm(w_ih.t())
    Uz = hx.mm(w_hh.t())
    # Section 2.1 in https://arxiv.org/pdf/1606.06630.pdf
    gates = alpha * Wx * Uz + beta_i * Wx + beta_h * Uz + bias
    # Same as LSTMCell after this point
    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
    ingate = ingate.sigmoid()
    forgetgate = forgetgate.sigmoid()
    cellgate = cellgate.tanh()
    outgate = outgate.sigmoid()
    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * cy.tanh()
    return hy, cy


def canonical(graph):
    return str(torch._C._jit_pass_canonicalize(graph))


def get_lstm_inputs(device, training=False, seq_length=None):
    input_shape = (3, 10) if seq_length is None else (seq_length, 3, 10)
    input = torch.randn(*input_shape, dtype=torch.float, device=device, requires_grad=training)
    hx = torch.randn(3, 20, dtype=torch.float, device=device, requires_grad=training)
    cx = torch.randn(3, 20, dtype=torch.float, device=device, requires_grad=training)
    module = nn.LSTMCell(10, 20).to(device, torch.float)  # Just to allocate weights with correct sizes
    if training:
        params = tuple(module.parameters())
    else:
        params = tuple(p.requires_grad_(False) for p in module.parameters())
    return (input, hx, cx) + params


def get_milstm_inputs(device, training=False):
    minibatch = 3
    input_size = 10
    hidden_size = 20
    x = torch.randn(minibatch, input_size, device=device, dtype=torch.float)
    hx = torch.randn(minibatch, hidden_size, device=device, dtype=torch.float)
    cx = torch.randn(minibatch, hidden_size, device=device, dtype=torch.float)

    ih = torch.randn(4 * hidden_size, input_size, device=device, dtype=torch.float, requires_grad=training)
    hh = torch.randn(4 * hidden_size, hidden_size, device=device, dtype=torch.float, requires_grad=training)
    alpha = torch.randn(4 * hidden_size, dtype=torch.float, device=device, requires_grad=training)
    ibeta = torch.randn(4 * hidden_size, dtype=torch.float, device=device, requires_grad=training)
    hbeta = torch.randn(4 * hidden_size, dtype=torch.float, device=device, requires_grad=training)
    bias = torch.randn(4 * hidden_size, dtype=torch.float, device=device, requires_grad=training)
    return x, hx, cx, ih, hh, alpha, ibeta, hbeta, bias


def get_fn(file_name, script_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(file_name, script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = module.fn
    return fn


def get_execution_plan(graph_executor_state):
    execution_plans = list(graph_executor_state.execution_plans.values())
    num_plans = len(execution_plans)
    if num_plans != 1:
        raise RuntimeError('This test assumes this GraphExecutor should '
                           'only have one execution plan, got: {}'.format(num_plans))
    return execution_plans[0]


def get_grad_executor(plan_state, diff_graph_idx=None):
    if diff_graph_idx is None:
        nodes = list(plan_state.graph.nodes())
        if len(nodes) == 1 or (len(nodes) == 2 and nodes[1].kind() == "prim::TupleConstruct"):
            pass
        else:
            raise RuntimeError("Can't get a grad_executor for a non-differentiable graph")
    grad_executors = list(plan_state.code.grad_executors())
    return grad_executors[diff_graph_idx or 0]


def backward_graph(script_module, diff_graph_idx=None):
    if not isinstance(script_module, torch.jit.ScriptModule):
        raise RuntimeError('Expected ScriptModule')
    ge_state = script_module.get_debug_state()
    fwd_plan = get_execution_plan(ge_state)
    grad_executor = get_grad_executor(fwd_plan, diff_graph_idx=diff_graph_idx)
    bwd_plan = get_execution_plan(grad_executor.get_debug_state())
    # Running JIT passes requires that we own the graph (with a shared_ptr).
    # The debug state struct does not own its graph so we make a copy of it.
    return bwd_plan.graph.copy()


# make it easy to quicky define/trace a function for these tests
def _trace(*args, **kwargs):
    def wrapper(func):
        return torch.jit.trace(func, args, **kwargs)
    return wrapper


def enable_cpu_fuser(fn):
    def wrapper(*args, **kwargs):
        torch._C._jit_override_can_fuse_on_cpu(True)
        try:
            fn(*args, **kwargs)
        finally:
            torch._C._jit_override_can_fuse_on_cpu(False)
    return wrapper


class JitTestCase(TestCase):
    _do_cuda_memory_leak_check = True
    _restored_warnings = False

    def setUp(self):
        # unittest overrides all warning filters and forces all of them to show up
        # after we install our own to silence those coming from inside PyTorch.
        # This will ensure that our filter still takes precedence.
        if not JitTestCase._restored_warnings:
            torch.jit.TracerWarning.ignore_lib_warnings()
            JitTestCase._restored_warnings = True
        torch._C._jit_set_emit_module_hook(self.emitModuleHook)

    def tearDown(self):
        # needs to be cleared because python might be unloaded before
        # the callback gets destucted
        torch._C._jit_set_emit_module_hook(None)

    @contextmanager
    def disableModuleHook(self):
        torch._C._jit_set_emit_module_hook(None)
        yield None
        torch._C._jit_set_emit_module_hook(self.emitModuleHook)

    def emitModuleHook(self, module):
        def copy_structure_and_params(m):
            c = torch.jit.ScriptModule()
            for name, v, buffer in m._get_parameters():
                c._register_parameter(name, v, buffer)
            for name, s in m._get_modules():
                c._register_module(name, copy_structure_and_params(s))
            return c

        # disable the hook while we parse code, otherwise we will re-enter the hook
        with self.disableModuleHook():
            try:
                pp, constant_table = module._python_print()
            except RuntimeError as e:
                se = str(e)
                if "could not export python function" not in se and \
                   "closures are not exportable" not in se:
                    raise
                else:
                    return
            ppv = "op_version_set = 0\n{}".format(pp)
            sm = copy_structure_and_params(module)
            torch._C._jit_import_methods(sm, ppv, constant_table)
            pp2, _ = sm._python_print()
            if pp != pp2:
                self.assertMultiLineEqual(pp, pp2)

    def getExportImportCopy(self, m, also_test_file=True, map_location=None):
        buffer = io.BytesIO()
        torch.jit.save(m, buffer)
        buffer.seek(0)
        imported = torch.jit.load(buffer, map_location=map_location)

        if not also_test_file:
            return imported

        with TemporaryFileName() as fname:
            imported.save(fname)
            return torch.jit.load(fname, map_location=map_location)

    def getExportImportCopyWithPacking(self, m, also_test_file=True, map_location=None):
        buffer = io.BytesIO()
        m.apply(lambda s: s._pack() if s._has_method('_pack') else None)
        torch.jit.save(m, buffer)
        m.apply(lambda s: s._unpack() if s._has_method('_unpack') else None)
        buffer.seek(0)
        imported = torch.jit.load(buffer, map_location=map_location)
        imported.apply(lambda s: s._unpack() if s._has_method('_unpack') else None)

        if not also_test_file:
            return imported

        # Ideally we would like to not have to manually delete the file, but NamedTemporaryFile
        # opens the file, and it cannot be opened multiple times in Windows. To support Windows,
        # close the file after creation and try to remove it manually
        f = tempfile.NamedTemporaryFile(delete=False)
        try:
            f.close()
            imported.save(f.name)
            result = torch.jit.load(f.name, map_location=map_location)
        finally:
            os.unlink(f.name)

        result.apply(lambda s: s._unpack() if s._has_method('_unpack') else None)
        return result

    def assertGraphContains(self, graph, kind):
        self.assertTrue(any(n.kind() == kind for n in graph.nodes()))

    def assertGraphContainsExactly(self, graph, kind, num_kind_nodes, consider_subgraphs=False):
        def perform_assert(graph, kind, actual, expected, consider_subgraphs):
            if actual == expected:
                return
            subgraph = 'including' if consider_subgraphs else 'excluding'
            raise AssertionError(
                '{}\nError: graph contains {} {} nodes ({} subgraphs) but expected {}'.format(
                    graph, actual, kind, subgraph, expected))

        if consider_subgraphs:
            strgraph = str(graph)
            count = strgraph.count(kind) - strgraph.count('with {}'.format(kind))
            perform_assert(graph, kind, count, num_kind_nodes,
                           consider_subgraphs)
            return

        nodes = [node for node in graph.nodes()
                 if node.kind() == kind]
        perform_assert(graph, kind, len(nodes), num_kind_nodes,
                       consider_subgraphs)

    def assertExpectedONNXGraph(self, trace, *args, **kwargs):
        torch.onnx._optimize_trace(trace, operator_export_type=OperatorExportTypes.ONNX)
        self.assertExpectedGraph(trace, *args, **kwargs)

    def assertExpectedGraph(self, trace, *args, **kwargs):
        if isinstance(trace, torch._C.Graph):
            graph = trace
        else:
            graph = trace.graph()

        torch._C._jit_pass_lint(graph)
        torch._C._jit_pass_dce(graph)
        torch._C._jit_pass_lint(graph)
        graph = torch._C._jit_pass_canonicalize(graph)
        torch._C._jit_pass_lint(graph)
        self.assertExpected(str(graph), *args, **kwargs)

    def run_pass(self, name, trace):
        if isinstance(trace, torch._C.Graph):
            graph = trace
            set_graph = False
        else:
            set_graph = True
            graph = trace.graph()

        torch._C._jit_pass_lint(graph)
        result = getattr(torch._C, '_jit_pass_' + name)(graph)
        if result is not None:
            graph = result
        torch._C._jit_pass_lint(graph)

        if set_graph:
            trace.set_graph(graph)
        return graph

    def checkScript(self,
                    script,
                    inputs,
                    optimize=True,
                    outputs=None,
                    name='func',
                    capture_output=False,
                    frames_up=1,
                    check_expected=False):
        if isinstance(script, str):
            cu = torch.jit.CompilationUnit(script, optimize, _frames_up=frames_up)
            ge = getattr(cu, name)
        else:
            if capture_output:
                with self.capture_stdout() as captured:
                    outputs = script(*inputs)
            else:
                outputs = script(*inputs)
            # Check the string frontend first
            source = textwrap.dedent(inspect.getsource(script))
            self.checkScript(
                source,
                inputs,
                optimize,
                outputs,
                script.__name__,
                capture_output,
                frames_up=2,
                check_expected=check_expected)
            # Continue checking the Python frontend
            ge = torch.jit.script(script, optimize, _frames_up=1)

        if capture_output:
            with self.capture_stdout() as captured:
                outputs_ge = ge(*inputs)
            if not WINDOWS:
                self.assertExpected(captured[0], subname='stdout')
        else:
            outputs_ge = ge(*inputs)
        self.assertEqual(outputs, outputs_ge)

        if check_expected:
            self.assertExpectedGraph(ge.graph)

        return ge

    def checkTrace(self, func, reference_tensors, input_tensors=None,
                   optimize=True, drop=None, allow_unused=False, verbose=False,
                   inputs_require_grads=True, check_tolerance=1e-5, export_import=True):
        # TODO: check gradients for parameters, not just inputs
        def allSum(vs):
            # drop allows us to remove some values from ever being used
            # to test unused outputs
            if drop is not None:
                vs = vs[:-drop]
            # we don't want all the grad for all the outputs to be the same
            # so we multiply each by a constant
            return sum(math.log(i + 2) * v.sum() for i, v in enumerate(vs) if v is not None)
        if input_tensors is None:
            input_tensors = reference_tensors

        nograd_inputs = reference_tensors
        if inputs_require_grads:
            recording_inputs = [t.clone().requires_grad_() for t in reference_tensors]
        else:
            recording_inputs = reference_tensors

        if isinstance(func, torch._C.Graph):
            ge = torch._C.GraphExecutor(func, optimize)
        else:
            ge = torch.jit.trace(func, input_tensors, optimize=optimize, check_tolerance=check_tolerance)

        if export_import:
            ge = self.getExportImportCopy(ge)

        if verbose:
            print(ge.graph)

        # test no gradients case
        outputs = func(*nograd_inputs)
        outputs_ge = ge(*nograd_inputs)
        self.assertEqual(outputs, outputs_ge)

        # test single grad case
        outputs = func(*recording_inputs)
        if inputs_require_grads:
            grads = torch.autograd.grad(allSum(outputs), recording_inputs,
                                        allow_unused=allow_unused)

        outputs_ge = ge(*recording_inputs)
        if inputs_require_grads:
            grads_ge = torch.autograd.grad(allSum(outputs_ge), recording_inputs,
                                           allow_unused=allow_unused)
        self.assertEqual(outputs, outputs_ge)
        if inputs_require_grads:
            self.assertEqual(grads, grads_ge)

        # test the grad grad case

        outputs = func(*recording_inputs)
        l1 = allSum(outputs)
        if inputs_require_grads:
            grads = torch.autograd.grad(l1, recording_inputs, create_graph=True,
                                        allow_unused=allow_unused)
        if inputs_require_grads:
            l2 = (allSum(grads) * l1)
            grads2 = torch.autograd.grad(l2, recording_inputs, allow_unused=allow_unused)

        if inputs_require_grads:
            recording_inputs = [Variable(t, requires_grad=True)
                                for t in reference_tensors]

        outputs_ge = ge(*recording_inputs)
        l1_ge = allSum(outputs_ge)
        if inputs_require_grads:
            grads_ge = torch.autograd.grad(
                l1_ge, recording_inputs, create_graph=True, allow_unused=allow_unused)

        if inputs_require_grads:
            l2_ge = (allSum(grads_ge) * l1_ge)
            grads2_ge = torch.autograd.grad(l2_ge, recording_inputs, allow_unused=allow_unused)

        self.assertEqual(outputs, outputs_ge)
        if inputs_require_grads:
            self.assertEqual(grads, grads_ge)
            for g2, g2_ge in zip(grads2, grads2_ge):
                if g2 is None and g2_ge is None:
                    continue
                self.assertTrue(torch.allclose(g2, g2_ge, atol=8e-4, rtol=8e-4))

        return ge

    def assertExportImport(self, trace, inputs):
        graph = trace if isinstance(trace, torch._C.Graph) else trace.graph()
        m = torch.jit.ScriptModule()
        m._create_method_from_graph("forward", graph)
        self.assertExportImportModule(m, inputs)

    def assertExportImportModule(self, m, inputs):
        m_import = self.getExportImportCopy(m)
        self.assertEqual(self.runAndSaveRNG(m.forward, inputs),
                         self.runAndSaveRNG(m_import.forward, inputs))

    def runAndSaveRNG(self, func, inputs, kwargs=None):
        kwargs = kwargs if kwargs else {}
        with freeze_rng_state():
            results = func(*inputs, **kwargs)
        return results


# has to be at top level or Pickle complains
class FooToPickle(torch.nn.Module):
    def __init__(self):
        super(FooToPickle, self).__init__()
        self.bar = torch.jit.ScriptModule()


class TestJit(JitTestCase):

    @unittest.skip("Requires a lot of RAM")
    def test_big(self):
        m = torch.jit.ScriptModule()
        gig = int(1024 * 1024 * 1024 / 4)
        # a small tensor in the first 4GB
        m.v0 = nn.Parameter(torch.full((2,), 1, dtype=torch.float))
        # a large tensor in the first 4GB that ends outside of it
        m.v1 = nn.Parameter(torch.full((5, gig), 2, dtype=torch.float))
        # a small tensor in >4GB space
        m.v2 = nn.Parameter(torch.full((2,), 3, dtype=torch.float))
        # s large tensor in the > 4GB space
        m.v3 = nn.Parameter(torch.full((5, gig), 4, dtype=torch.float))

        m2 = self.getExportImportCopy(m)

        self.assertEqual(tuple(m.parameters()), tuple(m2.parameters()))

    def test_simple(self):
        x = torch.tensor([0.4], requires_grad=True)
        y = torch.tensor([0.7], requires_grad=True)

        def f(x, y):
            return torch.sigmoid(torch.tanh(x * (x + y)))

        self.checkTrace(f, (x, y))

    def test_restore_device(self):
        # main purpose is checking map_location works
        m = torch.jit.ScriptModule()
        cpu_device_str = 'cpu'
        m.p0 = nn.Parameter(torch.tensor([0.3], dtype=torch.float,
                                         device=cpu_device_str))
        m.register_buffer('b0', torch.tensor([0.9], dtype=torch.float,
                                             device=cpu_device_str))
        m2 = self.getExportImportCopy(m)
        self.assertEqual(tuple(m.parameters()), tuple(m2.parameters()))
        self.assertEqual(tuple(m.buffers()), tuple(m2.buffers()))
        self.assertFalse(m2.p0.is_cuda)
        self.assertFalse(m2.b0.is_cuda)

    def test_model_save_error(self):
        with TemporaryFileName() as fname:
            with self.assertRaisesRegex(pickle.PickleError, "not supported"):
                torch.save(FooToPickle(), fname)

    def test_single_tuple_trace(self):
        x = torch.tensor(2.)

        def f2(x):
            return (x,)
        jit_f2 = torch.jit.trace(f2, x)
        assert f2(x) == jit_f2(x)  # fails

    @unittest.skipIf(not RUN_CUDA, "restore device requires CUDA")
    def test_restore_device_cuda(self):
        class MyModule(torch.jit.ScriptModule):
            def __init__(self):
                super(MyModule, self).__init__(False)
                self.register_buffer('b0', torch.randn(1, 3))
                self.p0 = nn.Parameter(torch.randn(2, 3))

            @torch.jit.script_method
            def forward(self, x):
                return x + self.b0 + self.p0

        m = MyModule()
        m.cuda(torch.cuda.device_count() - 1)
        cuda_device_str = 'cuda:' + str(torch.cuda.device_count() - 1)

        self.assertTrue(m.p0.is_cuda)
        self.assertTrue(m.b0.is_cuda)

        # restore to the saved devices
        m2 = self.getExportImportCopy(m)
        self.assertEqual(tuple(m.parameters()), tuple(m2.parameters()))
        self.assertEqual(tuple(m.buffers()), tuple(m2.buffers()))
        self.assertEqual(str(m2.p0.device), cuda_device_str)
        self.assertEqual(str(m2.b0.device), cuda_device_str)

        # restore all to cpu using string
        cpu_device_str = 'cpu'
        m3 = self.getExportImportCopy(m, map_location=cpu_device_str)
        self.assertEqual(str(m3.p0.device), cpu_device_str)
        self.assertEqual(str(m3.b0.device), cpu_device_str)

        # restore all to first gpu using device
        m4 = self.getExportImportCopy(
            m3, map_location=torch.device('cuda:0'))
        self.assertEqual(str(m4.p0.device), 'cuda:0')
        self.assertEqual(str(m4.b0.device), 'cuda:0')

        # compute and compare the results
        input = torch.rand(2, 3).cuda(torch.cuda.device_count() - 1)
        origin_result = m(input)
        self.assertEqual(origin_result, m2(input))
        self.assertEqual(origin_result, m3(input.cpu()))
        self.assertEqual(origin_result, m4(input.cuda(0)))

    @unittest.skipIf(not RUN_CUDA, "restore device requires CUDA")
    def test_restore_shared_storage_on_cuda(self):
        whole_tensor = torch.randn(4, 5, dtype=torch.float, device='cpu')
        m = torch.jit.ScriptModule()
        m.p0 = nn.Parameter(whole_tensor.narrow(0, 0, 1))
        m.register_buffer('b0', whole_tensor.narrow(0, 3, 1))
        m2 = self.getExportImportCopy(m, map_location=torch.device('cuda:0'))
        self.assertEqual(tuple(m.parameters()), tuple(m2.parameters()))
        self.assertEqual(tuple(m.buffers()), tuple(m2.buffers()))
        self.assertTrue(m2.p0.is_cuda)
        self.assertTrue(m2.b0.is_cuda)
        self.assertTrue(m2.p0.is_shared())
        self.assertTrue(m2.b0.is_shared())
        self.assertEqual(m2.b0.storage().data_ptr(), m2.p0.storage().data_ptr())

    def test_typeas_trace_check(self):
        a = torch.tensor([0.4], requires_grad=True)
        b = torch.tensor([0.7], requires_grad=True)

        def f(x, y):
            return x.type_as(y)

        trace = torch.jit.trace(f, (a, b))

    def test_peephole(self):
        a = torch.tensor([0.4])
        b = torch.tensor([0.7])
        c = torch.tensor([0], dtype=torch.int32)

        def f(x, y):
            return x.type_as(y)

        tf = torch.jit.trace(f, (a, b))
        self.run_pass('peephole', tf.graph)
        self.assertExpectedGraph(tf.graph)
        tf2 = torch.jit.trace(f, (a, c))
        s = str(tf2.graph)
        self.run_pass('peephole', tf2.graph)
        self.assertEqual(s, str(s))

    def test_peephole_dynamic(self):
        def f(x, y):
            return x.type_as(y)

        fn = torch.jit.script(f)
        s = str(fn.graph)
        torch._C._jit_pass_peephole(fn.graph)
        self.assertEqual(s, str(fn.graph))

    @unittest.skipIf(not RUN_CUDA, "cpp tests require CUDA")
    def test_peephole_cuda(self):
        a = torch.tensor([0.4], device='cpu')
        b = torch.tensor([0.7], device='cuda')
        c = torch.tensor([0.7], device='cuda')

        def f(x, y):
            return x.type_as(y)

        trace = torch.jit.trace(f, (a, c))
        s = str(trace.graph)
        self.run_pass('peephole', trace.graph)
        self.assertEqual(s, str(trace.graph))
        trace = torch.jit.trace(f, (b, c))
        self.run_pass('peephole', trace.graph)
        self.assertExpectedGraph(trace.graph, subname="same_device")

    def test_index(self):
        x = torch.tensor([0.4], requires_grad=True)
        y = torch.tensor([0], dtype=torch.int64)

        def fn(x, y):
            return x[y]

        fn_traced = torch.jit.trace(fn, (x, y,))

        self.assertEqual(fn(x, y), fn_traced(x, y))

    def test_disabled(self):
        torch.jit._enabled = False
        try:
            def f(x, y):
                return x + y

            self.assertIs(torch.jit.trace(f, (torch.randn(2, 2), torch.randn(2, 2))), f)
            self.assertIs(torch.jit.script(f), f)

            class MyModule(torch.jit.ScriptModule):
                @torch.jit.script_method
                def method(self, x):
                    return x

            # XXX: Unfortunately ScriptModule won't simply become Module now,
            # because that requires disabling the JIT at startup time, which
            # we can't do in here.
            # We need to or those two conditions to make it work with all versions of Python
            self.assertTrue(inspect.ismethod(MyModule.method) or inspect.isfunction(MyModule.method))
        finally:
            torch.jit._enabled = True

    def test_train_eval(self):
        class Sub(nn.Module):
            def forward(self, input):
                if self.training:
                    return input
                else:
                    return -input

        class MyModule(torch.jit.ScriptModule):
            def __init__(self, module):
                super(MyModule, self).__init__()
                self.module = module

            @torch.jit.script_method
            def forward(self, input):
                return self.module(input) + 1

        m = MyModule(Sub())
        input = torch.rand(3, 4)
        self.assertEqual(input + 1, m(input))
        m.eval()
        self.assertEqual(-input + 1, m(input))

        # test batchnorm and dropout train/eval
        input = torch.randn(6, 10)
        batchnorm = nn.BatchNorm1d(10)
        dropout = nn.Dropout(p=0.2)

        m_batchnorm = MyModule(batchnorm)
        self.assertEqual(batchnorm(input) + 1, m_batchnorm(input))
        batchnorm.eval()
        m_batchnorm.eval()
        self.assertEqual(batchnorm(input) + 1, m_batchnorm(input))

        m_dropout = MyModule(dropout)
        dropout.eval()
        m_dropout.eval()
        self.assertEqual(dropout(input) + 1, m_dropout(input))

    def test_diff_subgraph_clones_constants(self):
        @torch.jit.script
        def f(x, y):
            return x + x + y + x + y + x + y + x + y + x

        def count_constants(graph):
            return sum(node.kind() == 'prim::Constant' for node in graph.nodes())

        graph = f.graph.copy()
        self.run_pass('cse', graph)
        self.run_pass('create_autodiff_subgraphs', graph)
        nodes = list(graph.nodes())
        self.assertEqual(count_constants(graph), 1)
        self.assertEqual(count_constants(nodes[1].g('Subgraph')), 1)

    # Backwards tracing was broken for indexing by a constant,
    # because it's internally implemented using as_strided,
    # and we attempted to trace its derivative (which is not
    # currently supported.)  It currently works because
    # slice() is now not marked as traceable.
    def test_index_constant(self):
        x = torch.tensor([0.4], requires_grad=True)

        def fn(x):
            return x[0]

        def run(f):
            y = f(x)
            grad = torch.autograd.grad(y, x)[0].clone()
            return y, grad

        traced_fn = torch.jit.trace(fn, torch.ones(1))
        self.assertEqual(run(fn), run(traced_fn))

    def test_scopes(self):
        x = torch.tensor([0.4], requires_grad=True)
        y = torch.tensor([0.7], requires_grad=True)

        def f(x, y):
            out = x + y
            with torch.jit.scope('Foo'):
                out = x * out
                with torch.jit.scope('Bar'):
                    out = torch.tanh(out)
                out = torch.sigmoid(out)
            return out

        self.checkTrace(f, (x, y))

    def test_scopes_intermediate_node(self):

        class Net(nn.Module):
            def forward(self, x):
                return F.log_softmax(x, dim=0)

        net = Net()
        t = torch.ones(2, requires_grad=True)
        trace, _ = torch.jit.get_trace_graph(net, (t,))
        self.assertExportImport(trace, (t,))
        self.assertExpectedONNXGraph(trace)

    def test_scopes_identity_node(self):

        class Net(nn.Module):

            def __init__(self):
                super(Net, self).__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                )

            def forward(self, x):
                x = self.features(x)
                return x

        model = Net()

        t = torch.ones(1, 3, 227, 227, requires_grad=True)

        with torch.onnx.set_training(model, False):
            trace, _ = torch.jit.get_trace_graph(model, (t,))

        self.assertExportImport(trace, (t,) + tuple(model.parameters()))
        self.assertExpectedONNXGraph(trace)

    def test_canonicalize_tensor_iterator(self):
        x = torch.randn(4, 4)

        def f(x):
            x = x + 2
            x = x - 4
            x = x * 6
            x = x / 8
            return x

        traced = torch.jit.trace(f, (x,))
        f(x)
        graph = traced.graph_for(x)
        # There should be 4 int constants for the right sides of operators, plus two
        # for alpha arguments for add and sub
        self.assertTrue(str(traced.graph_for(x)).count(': int = prim::Constant'), 6)

    # TODO: adapt this test to check that GraphExecutor treats them differently
    @unittest.skip("Need to be adjusted to Graph Executor")
    def test_arg_configurations(self):
        """Different arg configurations should trigger different traces"""
        x = Variable(torch.FloatTensor(4, 4).uniform_())
        x_double = Variable(x.data.double())
        x_grad = Variable(x.data.clone(), requires_grad=True)
        y = Variable(torch.randn(4))

        configurations = [
            (x,),
            (x_double,),
            (x_grad,),
            (y,),
            ([x, x],),
            ([x, y],),
        ]
        if torch.cuda.is_available():
            x_cuda = Variable(x.data.cuda())
            configurations += [
                (x_cuda,),
                ([x, x_cuda],),
                ([x_cuda, x],),
                ([[x_cuda, x]],),
            ]
            if torch.cuda.device_count() > 1:
                x_cuda_1 = Variable(x.data.cuda(1))
                configurations += [
                    (x_cuda_1,),
                    ([x_cuda, x_cuda_1],),
                ]

        @torch.jit.compile(nderivs=0)
        def fn(*args):
            in_vars, _ = torch._C._jit_flatten(args)
            return in_vars[0] + 1

        for i, config in enumerate(configurations):
            self.assertFalse(fn.has_trace_for(*config))
            fn(*config)
            self.assertTrue(fn.has_trace_for(*config))
            for unk_config in configurations[i + 1:]:
                self.assertFalse(fn.has_trace_for(*unk_config))
        self.assertEqual(fn.hits, 0)

    def test_cse(self):
        x = torch.tensor([0.4, 0.3], requires_grad=True)
        y = torch.tensor([0.7, 0.5], requires_grad=True)

        def fn(x, y):
            w = (x + y) * (x + y) * (x + y)
            t = torch.tanh(w) + torch.tanh(w)
            z = (x + y) * (x + y) * (x + y) + t
            return z

        trace, _ = torch.jit.get_trace_graph(fn, (x, y))
        self.run_pass('cse', trace)
        self.assertExpectedGraph(trace)
        self.assertExportImport(trace, (x, y))

    def test_recursive_cse(self):
        x = torch.tensor([0.1])
        y = torch.tensor([0.2])

        def fn(x, y):
            z = x
            if bool(x + y > x):
                z = x + y
            return z

        graph = torch.jit.script(fn).graph
        self.run_pass('cse', graph)
        self.assertExpectedGraph(graph)

    def test_scalar(self):
        # NB: must not require grad; if it requires grad, it's always a Tensor
        x = torch.tensor(2.)
        y = torch.tensor(3.)

        def fn(x, y):
            return x - y
        trace, _ = torch.jit.get_trace_graph(fn, (x, y))

    def test_shape_analysis_broadcast(self):
        def broadcast(a, b):
            return a + b

        x = torch.randn(3, 1, 5, requires_grad=True)
        y = torch.randn(4, 1, 8, 5, requires_grad=True)

        graph = torch.jit.script(broadcast).graph
        torch._C._jit_pass_complete_shape_analysis(graph, (x, y), False)
        self.assertExpectedGraph(graph)

    # TODO: update verify to work with GraphExecutors
    @unittest.skip("verify needs to be updated to work with GraphExecutors")
    def test_verify(self):
        x = torch.tensor([0.4], requires_grad=True)
        y = torch.tensor([0.7], requires_grad=True)

        @torch.jit.compile
        def f(x, y):
            z = torch.sigmoid(x * (x + y))
            w = torch.abs(x * x * x + y) + Variable(torch.ones(1))
            return z, w

        torch.jit.verify(f, (x, y), loss_fn=lambda z, w: z * w, devices=[])

    @suppress_warnings
    def test_constant(self):
        x = torch.randn(2, 2, requires_grad=True)

        def f(x):
            return x.matmul(torch.diag(torch.tensor([2., 2.])))

        self.checkTrace(f, (x,), (torch.ones(2, 2, requires_grad=True),))

    def test_legacy_fail(self):
        class MyLegacyFn(Function):
            def forward(self, x):
                return x

            def backward(self, grad_output):
                return grad_output

        x = torch.tensor([0.], requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, "MyLegacyFn"):
            torch.jit.get_trace_graph(lambda x: MyLegacyFn()(x), (x,))

    def test_inplace_transplant(self):
        x = torch.tensor([0.], requires_grad=True)

        def fn(x):
            y = x.clone()
            y.add_(2)
            y.add_(3)
            return y

        trace, _ = torch.jit.get_trace_graph(fn, (x,))
        self.assertExpectedGraph(trace)
        self.assertExportImport(trace, (x,))

    def test_inplace_flags(self):
        class InplaceFn(Function):
            @staticmethod
            def forward(ctx, x):
                ctx.mark_dirty(x)
                return x.add_(1)

            @staticmethod
            def backward(ctx, go):
                return go

        class RegularFn(Function):
            @staticmethod
            def forward(ctx, x):
                return x.add(1)

            @staticmethod
            def backward(ctx, go):
                return go

        x = torch.tensor([0.], requires_grad=True)

        def fn(x):
            y = RegularFn.apply(x)
            y = InplaceFn.apply(y)
            y = InplaceFn.apply(y)
            y = RegularFn.apply(y)
            return y

        trace, _ = torch.jit.get_trace_graph(fn, (x,), _force_outplace=True)
        self.run_pass('dce', trace)
        ops = [n for n in trace.graph().nodes()]
        for op in ops:
            self.assertTrue(op.hasAttribute('inplace'))
        inplace_flags = [False, True, True, False]
        for op, is_inplace in zip(ops, inplace_flags):
            self.assertEqual(op.i('inplace'), is_inplace)

    def test_inplace_check(self):
        class MyInplaceFn(Function):
            @staticmethod
            def forward(self, x):
                x.add_(1)
                self.mark_dirty(x)
                return x

            @staticmethod
            def backward(self, grad):
                return grad

        def fn(x):
            return MyInplaceFn.apply(x)

        x = torch.randn(5, 5)
        ge = torch._C.GraphExecutor(fn, (x,), lambda var: '', _force_outplace=True)
        with self.assertRaisesRegex(RuntimeError, 'inplace MyInplaceFn'):
            ge(x)

    def do_trace_size(self, requires_grad):
        def fn(x):
            return x.view(x.shape[1] * 2, x.size(0), 2)

        x = torch.randn(5, 2, 4, requires_grad=requires_grad)
        y = torch.randn(4, 8, 4, requires_grad=requires_grad)

        # Check that it behaves as expected
        traced_fn = torch.jit.trace(fn, x)
        self.assertEqual(traced_fn(y), fn(y))
        self.assertEqual(traced_fn(x), fn(x))

        # Check that the trace looks ok
        trace, _ = torch.jit.get_trace_graph(fn, (x,))
        self.assertExpectedGraph(trace)

    def test_trace_size(self):
        self.do_trace_size(False)

    # test the different graph_executor path that happens when
    # gradients are required and sizes are involved
    def test_trace_size_with_grad(self):
        self.do_trace_size(True)

    def test_trace_casts(self):
        casts = [
            lambda x: x.byte(),
            lambda x: x.float(),
            lambda x: x.cpu(),
            lambda x: x.to(device='cpu'),
            lambda x: x.to(dtype=torch.int64),
            lambda x: x.to(device='cpu', dtype=torch.float),
            lambda x: x.to(x)
        ]

        def assertContainsCast(trace):
            self.assertEqual(sum(n.kind() == 'aten::to' for n in trace.graph.nodes()), 1)

        for cast in casts:
            trace = torch.jit.trace(cast, torch.randn(2, 2))
            assertContainsCast(trace)
            x = torch.randn(2, 2)
            self.assertEqual(trace(x), cast(x))

        def to_tensor(x, y):
            return x.to(y)

        to_tensor_trace = torch.jit.trace(to_tensor, (torch.randn(2, 2), torch.randn(1, 8)))
        assertContainsCast(to_tensor_trace)
        x, y = torch.randn(2, 2), torch.randn(1, 10)
        self.assertEqual(to_tensor_trace(x, y), to_tensor(x, y))

    def test_trace_warn(self):
        def fn(x):
            int(x)  # Warning 1.
            y = x * 1
            if y:   # Warning 2.
                pass
            q = [x, x * 4]
            z = q[y]  # Warning 3.
            float(z)  # Warning 4.
            z.tolist()  # Warning 5.
            z.numpy()  # Warning 6.
            for _ in torch.ones(4, 4):  # Warning 7.
                pass
            return z + 4

        with warnings.catch_warnings(record=True) as warns:
            traced_fn = torch.jit.trace(fn, torch.tensor([1]))
        warns = [str(w.message) for w in warns]
        self.assertEqual(len(warns), 7)
        self.assertIn('a Python integer', warns[0])
        self.assertIn('a Python boolean', warns[1])
        self.assertIn('a Python index', warns[2])
        self.assertIn('a Python float', warns[3])
        self.assertIn('a Python list', warns[4])
        self.assertIn('a NumPy array', warns[5])
        self.assertIn('Iterating over', warns[6])

    def test_trace_tuple(self):
        def fn(x, y):
            return x, (x * y[1], x * y[0])

        x, y = torch.randn(2, 2), (torch.ones(2, 2), torch.randn(2, 2))
        traced_fn = torch.jit.trace(fn, (x, y))
        self.assertEqual(traced_fn(x, y), fn(x, y))
        self.assertExpectedGraph(traced_fn.graph)
        self.assertExportImport(traced_fn.graph, (x, y))

    def test_trace_random(self):
        def f(mean, std):
            return torch.normal(mean, std)

        traced = torch.jit.trace(f, (torch.zeros(2, 3), torch.ones(2, 3)), check_trace=False)
        mean, std = torch.zeros(5, 5), torch.ones(5, 5)
        with torch.random.fork_rng(devices=[]):
            output = f(mean, std)
        traced_output = traced(mean, std)
        self.assertEqual(output, traced_output)

    def test_trace_tensor_factory(self):
        def run(**kwargs):
            inputs_require_grads = kwargs.pop('inputs_require_grads', True)

            def fn(x):
                return x + torch.ones(2, 3, **kwargs)

            input_kwargs = kwargs.copy()
            if 'out' in input_kwargs:
                del input_kwargs['out']
            input = torch.ones(2, 3, **input_kwargs)
            self.checkTrace(fn, (input,), inputs_require_grads=inputs_require_grads)
            # check we recorded 'ones' and did not just record a constant
            tfn = torch.jit.trace(fn, input)
            self.assertTrue("ones" in str(tfn.graph))
        run()
        run(dtype=torch.int, inputs_require_grads=False)
        run(out=torch.tensor([]))
        if RUN_CUDA:
            run(device="cuda:0")
        if RUN_CUDA_MULTI_GPU:
            run(device="cuda:1")

    def test_trace_indexed_assignment(self):
        def stuff(x, y):
            x = x.clone()
            x[0] = y
            return x
        example = torch.rand(3, 4)
        self.checkTrace(stuff, (example, example[0] + 1))

    # TODO: implement
    @unittest.expectedFailure
    def test_output_unflatten(self):
        """Check that outputs of traced functions retain the original structure and nesting"""
        def fn(x):
            return (x * 2, (x ** 2, x + 4, (x + 2,), ), x * 4)

        self.checkTrace(fn, (torch.randn(2, 2),))

    # TODO: implement
    @unittest.expectedFailure
    def test_input_flatten(self):
        """Check that inputs to traced functions are flattened"""

        def fn(x, t):
            y, z = t
            return x * y * z

        inputs = (torch.randn(1), (torch.randn(1), torch.randn(1)))
        self.checkTrace(fn, inputs)

    # TODO: adapt to a GraphExecutor test
    @unittest.skip("Need to instrument GraphExecutors a bit more")
    def test_flags(self):
        x, y = torch.randn(2, 2)
        y = Variable(torch.randn(2, 2))

        @torch.jit.compile
        def fn(x, y):
            return (x * x + y * y + x * y).sum()

        grads = {}
        for rx, ry in product((True, False), repeat=2):
            x.requires_grad = rx
            y.requires_grad = ry

            self.assertFalse(fn.has_trace_for(x, y))
            out = fn(x, y)

            self.assertFalse(fn.has_trace_for(x, y))
            for v, name, compute in [(x, 'x', rx), (y, 'y', ry)]:
                if not compute:
                    continue
                grad_v, = torch.autograd.grad(out, v, retain_graph=True)
                expected_grad = grads.setdefault(name, grad_v)
                self.assertEqual(grad_v, expected_grad)
            self.assertEqual(fn.has_trace_for(x, y), rx or ry)

    def test_python_ir(self):
        x = torch.tensor([0.4], requires_grad=True)
        y = torch.tensor([0.7], requires_grad=True)

        def doit(x, y):
            return torch.sigmoid(torch.tanh(x * (x + y)))

        trace, _ = torch.jit.get_trace_graph(doit, (x, y))
        self.run_pass('dce', trace)
        self.run_pass('canonicalize', trace)
        g = trace.graph()
        g2 = torch._C.Graph()
        g_to_g2 = {}
        for node in g.inputs():
            g_to_g2[node] = g2.addInput()
        for node in g.nodes():
            n_ = g2.createClone(node, lambda x: g_to_g2[x])
            g2.appendNode(n_)
            for o, no in zip(node.outputs(), n_.outputs()):
                g_to_g2[o] = no

        for node in g.outputs():
            g2.registerOutput(g_to_g2[node])

        t_node = g2.create("prim::TensorTest").t_("a", torch.ones([2, 2]))
        self.assertEqual(t_node.attributeNames(), ["a"])
        g2.appendNode(t_node)
        self.assertTrue(torch.equal(torch.ones(2, 2), t_node.t("a")))
        self.assertExpected(str(g2))

    @unittest.skipIf(IS_WINDOWS, "NYI: fuser support for Windows")
    @unittest.skipIf(not RUN_CUDA, "cpp tests require CUDA")
    @skipIfRocm
    def test_cpp_cuda(self):
        from cpp.jit import tests_setup
        tests_setup.setup()
        # rather than rebuild assertExpected in cpp,
        # just glob all the cpp outputs into one file for now
        self.assertExpected(torch._C._jit_run_cpp_tests())
        tests_setup.shutdown()

    def test_batchnorm(self):
        x = torch.ones(2, 2, 2, 2)
        trace, _ = torch.jit.get_trace_graph(nn.BatchNorm2d(2), x, _force_outplace=True)
        self.assertExpectedGraph(trace)

    def test_dropout(self):
        x = torch.ones(2, 2)
        trace, _ = torch.jit.get_trace_graph(nn.Dropout(0.6), x)
        self.assertExpectedGraph(trace)

    def test_conv(self):
        x = torch.ones(20, 16, 50, 40)
        trace, _ = torch.jit.get_trace_graph(nn.Conv2d(16, 13, 3, bias=False), x)
        self.assertExpectedGraph(trace)

    def test_repeated_input(self):
        def fn(a, b):
            return a + b

        ge = self.checkTrace(fn, [torch.randn(2, 2)] * 2)
        self.assertExpectedGraph(ge.graph)

    def test_repeated_output(self):
        def fn(a, b):
            z = a + b
            return z, z

        ge = self.checkTrace(fn, [torch.randn(2, 2) for _ in range(2)])
        self.assertExpectedGraph(ge.graph)

    @skipIfNoTorchVision
    def test_alexnet(self):
        x = torch.ones(1, 3, 224, 224)
        trace, _ = torch.jit.get_trace_graph(torchvision.models.AlexNet(), x)
        self.run_pass('cse', trace)
        self.assertExpectedGraph(trace)

    # Inplace copies don't work with tracer yet.
    # This is actually somewhat important to support correctly
    # as all backwards functions of views are implemented
    # as a zero filled tensor with a gradient fill on the
    # viewed portion.
    def test_inplace_copy(self):
        x = torch.randn(4, 4, requires_grad=True)

        def f(x):
            out = Variable(torch.zeros(x.size()))
            out.copy_(x)
            return out

        trace, z = torch.jit.get_trace_graph(f, (x, ))
        self.run_pass('dce', trace)
        self.assertExpectedGraph(trace)
        self.assertExportImport(trace, (x,))

    def test_shared_param(self):

        class MyModule(torch.nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.b = self.a = nn.Parameter(torch.randn(2, 2))

            def forward(self, x):
                return x * self.a + self.b

        m = MyModule()
        trace, _ = torch.jit.get_trace_graph(m, (torch.randn(2, 2),))
        self.assertEqual(len(list(trace.graph().inputs())), 2)
        self.assertExpectedGraph(trace)

    def test_nested_inplace(self):
        x = torch.randn(2, 2)
        trace, _ = torch.jit.get_trace_graph(
            lambda x: F.threshold(x, 0, 0, inplace=True), (x, ))
        self.assertExpectedGraph(trace)
        self.assertExportImport(trace, (x,))

    def run_ge_tests(self, optimize, use_cuda):
        def rand(*args):
            t = torch.rand(*args).float()
            if use_cuda:
                t = t.cuda()
            return t
        self.checkTrace(lambda a, b: a * b + b,
                        [rand(1), rand(1)], [rand(2, 3), rand(2, 3)],
                        optimize=optimize)
        # trivial identity
        self.checkTrace(lambda a, b: (
            b, a), [rand(1), rand(1)], optimize=optimize)

        def foo(a):
            t = a * a
            return t * t, 4 * t
        self.checkTrace(foo, [rand(1)], optimize=optimize)
        # unused input
        self.checkTrace(
            lambda a, b: a * a, [rand(1), rand(1)], optimize=optimize,
            allow_unused=True)
        # test outputs that do not get used in grad
        self.checkTrace(foo, [rand(1)], drop=1, optimize=optimize)
        # test autograd fallback
        self.checkTrace(lambda a, b: a * b /
                        (a - 2 * b) + b, [rand(1), rand(1)],
                        optimize=optimize)

    def test_ge_unoptimized(self):
        self.run_ge_tests(False, False)

    @unittest.skipIf(IS_WINDOWS or IS_SANDCASTLE, "NYI: fuser support for Windows or Sandcastle")
    @enable_cpu_fuser
    def test_ge_optimized(self):
        self.run_ge_tests(True, False)

    @unittest.skipIf(IS_WINDOWS, "NYI: fuser support for Windows")
    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    def test_ge_cuda(self):
        self.run_ge_tests(True, True)

    # more manual test of graph executor that can be used as a scratchpad
    def test_ge(self):
        def foo(a, b):
            return a * b / (a - b) + b
        V = Variable
        a, b = V(torch.rand(1)), V(torch.rand(1))
        ge = torch._C.GraphExecutor(foo, (a, b), lambda var: '')
        a, b = V(torch.rand(1), requires_grad=True), V(
            torch.rand(1), requires_grad=True)
        r, = ge(a, b)
        da, db = torch.autograd.grad(r + 3, [a, b], create_graph=True)

        l2 = (da * db + db * db)
        g2result = torch.autograd.grad(l2, [da, db])

        r = foo(a, b)
        da2, db2 = torch.autograd.grad(r + 3, [a, b], create_graph=True)
        self.assertEqual(da, da2)
        self.assertEqual(db, db2)
        l3 = (da2 * db2 + db2 * db2)
        g2result2 = torch.autograd.grad(l3, [da2, db2])
        self.assertEqual(g2result, g2result2)

    def test_trace_annotation(self):
        @_trace(torch.rand(1))
        def foo(a):
            return a + a + a

        x = torch.randn(5, 5)
        self.assertEqual(foo(x), x + x + x)

    def test_trace_script(self):
        @torch.jit.script
        def func1(x):
            # type: (Tuple[Tensor, Tensor]) -> Tensor
            return x[0] + x[1]

        @torch.jit.script
        def func2(x):
            # type: (List[Tensor]) -> Tensor
            return x[0] + x[1]

        a = torch.randn(5)
        b = torch.randn(5)

        expected = func1((a, b))
        traced = torch.jit.trace(func1, ((a, b),))
        result = traced((a, b))
        self.assertEqual(expected, result)

        expected = func2((a, b))
        traced = torch.jit.trace(func2, ((a, b),))
        result = traced((a, b))
        self.assertEqual(expected, result)

    def test_einsum(self):
        def outer(x, y):
            return torch.einsum('i,j->ij', (x, y))

        traced = torch.jit.trace(outer, (torch.randn(4), torch.randn(5)))
        script = torch.jit.script(outer)
        fns = [traced, script]
        x, y = torch.randn(10), torch.randn(2)
        for fn in [traced, script]:
            self.assertGraphContains(fn.graph, kind='aten::einsum')
            self.assertEqual(fn(x, y), outer(x, y))

    @unittest.skipIf(IS_WINDOWS, "NYI: fuser support for Windows")
    @unittest.skipIf(not RUN_CUDA, "calls .cuda()")
    def test_traced_module_cuda(self):
        class Model(nn.Module):
            def __init__(self, num_features, num_layers):
                super(Model, self).__init__()
                self.num_layers = num_layers
                layers = [[nn.Linear(num_features, num_features), nn.Sigmoid()]
                          for _ in range(num_layers)]
                self.submodule = nn.Sequential(*chain(*layers))

            def forward(self, x):
                for i in range(self.num_layers):
                    x = self.submodule[i](x) + x
                return x

        model = Model(5, 3)
        x = torch.randn(2, 5)
        traced_model = torch.jit.trace(model, x)

        # We're missing some attributes these modules had initially. Make sure we can
        # still get the __repr__()
        model.__repr__()

        # XXX: indexing sequentials is broken
        linear_submodule = next(iter(traced_model.submodule._modules.values()))

        # All attributes that aren't parameters should raise
        with self.assertRaises(AttributeError):
            linear_submodule.in_features
        linear_submodule.weight
        with self.assertRaises(RuntimeError):
            traced_model.asdf = 4
        linear_submodule.weight = nn.Parameter(torch.randn(linear_submodule.weight.shape))
        with self.assertRaises(RuntimeError):
            del linear_submodule.weight

        # Submodules can't be called
        with self.assertRaises(RuntimeError):
            linear_submodule(x)

        # Type casts
        linear_submodule.cuda()
        traced_model.float().cuda()
        cuda_out = traced_model(x.float().cuda())
        traced_model.cpu()
        cpu_out = traced_model(x.float())
        self.assertEqual(cpu_out, cuda_out)
        traced_model.to('cuda')
        cuda_out = traced_model(x.float().cuda())
        traced_model.to('cpu')
        cpu_out = traced_model(x.float())
        self.assertEqual(cpu_out, cuda_out)
        traced_model.double()

        # state_dict + load_state_dict
        state = {k: v.clone() for k, v in traced_model.state_dict().items()}
        new_state = {k: v.clone().fill_(1) for k, v in state.items()}
        out = traced_model(x)
        traced_model.load_state_dict(new_state)
        out_ones = traced_model(x)
        traced_model.load_state_dict(state)
        out_state = traced_model(x)
        self.assertEqual(out, out_state)
        self.assertNotEqual(out, out_ones)

    def test_python_function(self):
        class MyFn(Function):
            @staticmethod
            def forward(ctx, x):
                return x + 1

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output

        @_trace(torch.zeros(2))
        def fn(x):
            return MyFn.apply(x + 2) + 3

        x = torch.tensor([1., 2., 3.])
        y = torch.randn(2, 2, requires_grad=True)
        fn(x)
        fn(y)

    def test_python_function_tup(self):
        class MyFn(Function):
            @staticmethod
            def forward(ctx, x):
                return x + 1, x - 1

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output, grad_output

        @_trace(torch.zeros(2))
        def fn(x):
            a, b = MyFn.apply(x + 2)
            return a + b + 3
        x = torch.tensor([1., 2., 3.])
        y = torch.randn(2, 2, requires_grad=True)
        fn(x)
        fn(y)

    def test_decompose_addmm(self):
        @torch.jit.script
        def addmm(mat, mat1, mat2, alpha, beta):
            a = mat.addmm(mat1, mat2)
            b = mat.addmm(mat1, mat2, alpha=1.0, beta=1.0)
            c = mat.addmm(mat1, mat2, alpha=4.20, beta=2.0)
            d = mat.addmm(mat1, mat2, alpha=int(alpha), beta=int(beta))

            return a + b + c + d

        mat = torch.randn(2, 2)
        mat1 = torch.randn(2, 4)
        mat2 = torch.randn(4, 2)
        alpha = torch.FloatTensor([123.0])
        beta = torch.FloatTensor([321.0])

        out_ref = addmm(mat, mat1, mat2, alpha, beta)
        self.run_pass('canonicalize_ops', addmm.graph)
        out_test = addmm(mat, mat1, mat2, alpha, beta)
        self.assertEqual(out_ref, out_test)
        self.assertExpected(canonical(addmm.graph))

    def test_index_put(self):
        ten = torch.zeros(3, 3)
        mask = torch.Tensor([[True, True, True],
                             [True, False, False],
                             [True, True, False]]).byte()

        def test_fn(ten, mask):
            ten[mask] = torch.ones(6)
            return ten

        traced_test_fn = torch.jit.trace(test_fn, (ten, mask))

        ten = torch.rand(3, 3)
        self.assertEqual(test_fn(ten, mask), traced_test_fn(ten, mask))

    def test_sparse_tensors_error(self):
        def get_sparse():
            return torch.sparse.FloatTensor(2, 3)

        @torch.jit.script
        def sparse(input):
            output = get_sparse()
            return output, input

        with self.assertRaisesRegex(RuntimeError, "sparse tensors not supported"):
            sparse(get_sparse())

        with self.assertRaisesRegex(RuntimeError, "sparse tensors not supported"):
            sparse(torch.tensor([1]))

    def test_tuple_specialization(self):
        @torch.jit.script
        def f(t):
            # type: (Tuple[Tensor, Tensor]) -> Tensor
            x, y = t
            return x + y

        t = torch.randn(2, 2), torch.randn(2, 2)
        f(t)
        graph = f.graph_for(t)
        input_types = list(next(graph.inputs()).type().elements())
        for t in input_types:
            self.assertEqual(t.kind(), 'DimensionedTensorType')

    def test_constant_prop_simple(self):
        @torch.jit.script
        def constant_prop(input_int):
            # type: (int) -> int
            a = 2 * 3
            b = a + 2
            return b - input_int

        out_ref = constant_prop(2)
        self.run_pass('constant_propagation', constant_prop.graph)
        out_test = constant_prop(2)
        self.assertEqual(out_ref, out_test)
        graph_str = str(constant_prop.graph)
        self.assertTrue("aten::add" not in graph_str and "aten::mul" not in graph_str)
        const = constant_prop.graph.findNode("prim::Constant").output().toIValue()
        self.assertEqual(const, 8)

    def test_constant_prop_nested(self):
        @torch.jit.script
        def constant_prop(a):
            b = 2 + 1
            if bool(a < 2):
                c = b + 2
            else:
                c = b - 2
            return c
        out_ref = constant_prop(torch.tensor(2))
        self.run_pass('constant_propagation', constant_prop.graph)
        out_test = constant_prop(torch.tensor(2))
        self.assertEqual(out_ref, out_test)
        if_node = constant_prop.graph.findNode("prim::If")
        for block in if_node.blocks():
            for node in block.nodes():
                self.assertTrue(node.kind() == "prim::Constant")

    def test_constant_prop_print(self):
        @torch.jit.script
        def constant_prop(input_tensor):
            a = 2 * 3
            print(a)
            b = a + 2
            return b + input_tensor

        self.run_pass('constant_propagation', constant_prop.graph)
        graph = constant_prop.graph
        print_node = graph.findNode("prim::Print")
        self.assertTrue(print_node.input().toIValue() == 6)

    def test_constant_prop_rand(self):
        @torch.jit.script
        def constant_prop():
            a = torch.randn([3])
            b = a + 2
            return b

        self.run_pass('constant_propagation', constant_prop.graph)
        self.assertTrue("aten::randn" in str(constant_prop.graph))

    def test_constant_prop_none(self):
        @torch.jit.script
        def typed_none():
            # type: () -> Optional[int]
            return None

        @torch.jit.script
        def constant_prop():
            a = typed_none()
            b = typed_none()
            if (a is None and b is None):
                a = 2
            else:
                a = 1
            return a

        self.run_pass('constant_propagation', constant_prop.graph)
        graph_str = str(constant_prop.graph)
        self.assertTrue(graph_str.count("prim::None") == 0)

    def test_constant_prop_if_inline(self):
        @torch.jit.script
        def constant_prop():
            cond = True
            a = 1
            if cond:
                a = 1 * 2
            else:
                a = 1 // 0
            return a

        # testing that 1 // 0 error is not thrownn
        self.run_pass('constant_propagation', constant_prop.graph)

    def test_trace_records_names(self):
        def foo(bar, baz):
            baz = bar + 3
            quick_brown_fox = torch.neg(baz)
            for _ in range(20):
                yeet = quick_brown_fox - 3.14
            return yeet

        traced = torch.jit.trace(foo, (torch.rand(3, 3), torch.rand(3, 3)))
        graph_str = str(traced.graph)
        assert 'bar' in graph_str
        assert 'baz' in graph_str
        assert 'quick_brown_fox' in graph_str

    def test_constant_prop_if_constant(self):
        @torch.jit.script
        def constant_prop(a, b):
            c0 = 1
            c1 = 1
            c2 = 1
            if bool(a):  # -> c0, c1
                if bool(b):  # -> c0
                    if True:  # -> c0
                        c0 = c0 + 1
                        if False:
                            c1 = c1 + 1
                            c2 = c2 + 1
            else:  # -> c0, c1
                c1 = c1 + 1

            if True:  # inlined
                c0 = c0 + 1  # dynamic
                c2 = c2 + 4  # set to 5
            return a + c0 + c1 + c2

        graph = constant_prop.graph
        self.run_pass('constant_propagation', graph)
        ifs = graph.findAllNodes("prim::If", recurse=False)
        snd_if_inlined = len(ifs) == 1
        self.assertTrue(snd_if_inlined)
        first_if = ifs[0]
        self.assertTrue(first_if.outputsSize() == 2)
        second_if = first_if.findNode("prim::If", recurse=False)
        self.assertTrue(second_if.outputsSize() == 1)
        self.assertTrue(second_if.findNode("prim::If") is None)

    def test_constant_prop_loop_constant(self):
        @torch.jit.script
        def constant_prop(cond, iter):
            # type: (bool, int) -> int
            b = 0
            while True:
                print("stays")
            for _ in range(2):
                print("stays")
            for _ in range(iter):
                print("stays")
            while cond:
                print("stays")
            while False:
                print("removed")
            for _i in range(0):
                print("removed")
            for _i in range(-4):
                print("removed")
            return b

        self.run_pass('constant_propagation', constant_prop.graph)
        graph = canonical(constant_prop.graph)
        self.assertTrue(graph.count("removed") == 0)
        self.assertTrue(graph.count("stays") == 1)  # constant gets pooled
        self.assertTrue(graph.count("prim::Print") == 4)

    def test_constant_prop_remove_output(self):
        @torch.jit.script
        def constant_prop(iter):
            # type: (int) -> None
            a = 1
            b = 1
            c = 1
            for i in range(iter):
                if False:
                    a = 10
                if i == 5:
                    b = 2
                    c = 3
            print(a, b, c)

        graph = constant_prop.graph
        self.run_pass('constant_propagation', graph)
        self.assertTrue(graph.findNode("prim::Loop").outputsSize() == 2)

    def test_trace_detach(self):
        def foo(x, w):
            return torch.matmul(x, w).detach()

        traced = torch.jit.trace(foo, (torch.rand(3, 4), torch.rand(4, 5)))

        self.assertExpectedGraph(traced.graph)
        x, w = torch.rand(3, 4), torch.rand(4, 5, requires_grad=True)
        traced_result = traced(x, w)
        self.assertEqual(foo(x, w), traced_result)
        self.assertFalse(traced_result.requires_grad)
        self.assertIsNone(traced_result.grad_fn)

    def test_trace_detach_inplace(self):
        def foo(x, w):
            y = torch.matmul(x, w)
            y.detach_()
            return y

        traced = torch.jit.trace(foo, (torch.rand(3, 4), torch.rand(4, 5)))

        self.assertExpectedGraph(traced.graph)
        x, w = torch.rand(3, 4), torch.rand(4, 5)
        traced_result = traced(x, w)
        self.assertEqual(foo(x, w), traced_result)
        self.assertFalse(traced_result.requires_grad)
        self.assertIsNone(traced_result.grad_fn)

    def test_trace_detach_onnx_erase(self):
        class Mod(torch.nn.Module):
            def forward(self, x, w):
                return torch.matmul(x, w).detach()

        f = io.BytesIO()
        self.assertExpected(torch.onnx.export_to_pretty_string(
            Mod(), (torch.rand(3, 4), torch.rand(4, 5)), f))

    def test_trace_slice_full_dim(self):
        def foo(x):
            return x[0:5, 0] + 1.0

        traced = torch.jit.trace(foo, (torch.rand(5, 4),))
        test_x = torch.rand(6, 3)
        self.assertEqual(foo(test_x), traced(test_x))

    def test_export_dropout(self):
        test = torch.nn.Dropout()
        test.eval()

        traced = torch.jit.trace(test, (torch.rand(3, 4),), check_trace=False)
        imported = self.getExportImportCopy(traced)
        x = torch.randn(3, 4)
        self.assertEqual(traced(x), imported(x))

    def test_onnx_transpose_incomplete_tensor_type(self):
        # Smoke test to get us into the state where we are attempting to export
        # a transpose op, where the input is a TensorType rather than a
        # CompleteTensorType. This would previously not work, since we would
        # take the size of the input and use the length of its sizes as the
        # number of dimensions in the permutation.
        class Foo(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                return x.contiguous().transpose(0, 1).sum()

        class TraceMe(torch.nn.Module):
            def __init__(self):
                super(TraceMe, self).__init__()
                self.foo = Foo()

            def forward(self, x):
                return self.foo(x)

        tm = TraceMe()
        tm = torch.jit.trace(tm, torch.rand(3, 4))
        example_outputs = (tm(torch.rand(3, 4)),)
        f = io.BytesIO()
        torch.onnx._export(tm, (torch.rand(3, 4),), f, example_outputs=example_outputs)

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    def test_cuda_export_restore(self):
        class Sub(torch.jit.ScriptModule):
            def __init__(self):
                super(Sub, self).__init__()
                self.weight = nn.Parameter(torch.randn(3, 4))

            @torch.jit.script_method
            def forward(self, thing):
                return self.weight + thing

        class M(torch.jit.ScriptModule):
            def __init__(self):
                super(M, self).__init__()
                self.mod = Sub()

            @torch.jit.script_method
            def forward(self, v):
                return self.mod(v)
        m = M()
        m.cuda()
        m2 = self.getExportImportCopy(m)
        m2.cuda()
        input = torch.rand(3, 4).cuda()
        self.assertEqual(m(input), m2(input))

    def test_export_batchnorm(self):
        for mode in ['eval', 'train']:
            for clazz in [
                    torch.nn.BatchNorm1d(100),
                    torch.nn.BatchNorm1d(100, affine=False),
                    torch.nn.BatchNorm2d(100),
                    torch.nn.BatchNorm2d(100, affine=False)]:
                getattr(clazz, mode)()

                input = torch.randn(20, 100) if isinstance(clazz, torch.nn.BatchNorm1d) else \
                    torch.randn(20, 100, 35, 45)

                traced = torch.jit.trace(clazz, (input,))
                imported = self.getExportImportCopy(traced)
                x = torch.randn(20, 100) if isinstance(clazz, torch.nn.BatchNorm1d) else \
                    torch.randn(20, 100, 35, 45)
                self.assertEqual(traced(x), imported(x))

    def test_export_rnn(self):
        for clazz in [nn.RNN(10, 20, 2), nn.GRU(10, 20, 2)]:
            class RNNTest(torch.nn.Module):
                def __init__(self):
                    super(RNNTest, self).__init__()
                    self.rnn = clazz

                def forward(self, x, lengths, h0):
                    packed = torch.nn.utils.rnn.pack_padded_sequence(x, lengths)
                    out, h = self.rnn(packed, h0)
                    padded_outs, _ = torch.nn.utils.rnn.pad_packed_sequence(out)
                    return padded_outs

            test = RNNTest()

            traced = torch.jit.trace(test, (torch.randn(5, 3, 10), torch.LongTensor([3, 2, 1]), torch.randn(2, 3, 20)))
            imported = self.getExportImportCopy(traced)
            # NB: We make sure to pass in a batch with a different max sequence
            # length to ensure that the argument stashing for pad_packed works
            # properly.
            x, lengths, h0 = torch.randn(7, 4, 10), torch.LongTensor([7, 3, 2, 1]), torch.randn(2, 4, 20)
            self.assertEqual(traced(x, lengths, h0), imported(x, lengths, h0))

    def test_export_lstm(self):
        class LSTMTest(torch.nn.Module):
            def __init__(self):
                super(LSTMTest, self).__init__()
                self.rnn = nn.LSTM(10, 20, 2)

            def forward(self, x, lengths, hiddens):
                h0, c0 = hiddens
                packed = torch.nn.utils.rnn.pack_padded_sequence(x, lengths)
                out, (h, c) = self.rnn(packed, (h0, c0))
                padded_outs, _ = torch.nn.utils.rnn.pad_packed_sequence(out)
                return padded_outs

        test = LSTMTest()

        traced = torch.jit.trace(test, (torch.randn(5, 3, 10),
                                        torch.LongTensor([3, 2, 1]),
                                        (torch.randn(2, 3, 20), torch.randn(2, 3, 20))))
        imported = self.getExportImportCopy(traced)
        x, lengths, h0, c0 = \
            torch.randn(7, 3, 10), torch.LongTensor([7, 5, 2]), torch.randn(2, 3, 20), torch.randn(2, 3, 20)
        self.assertEqual(traced(x, lengths, (h0, c0)), imported(x, lengths, (h0, c0)))

    def test_trace_dict_input(self):
        class Bar(torch.nn.Module):
            def __init__(self):
                super(Bar, self).__init__()
                self.foo = Foo()

            def forward(self, a, b):
                return self.foo({'a': a, 'b': b})['a']

        class Foo(torch.nn.Module):
            def forward(self, x):
                return {'a': x['a'] * x['b']}

        x = (torch.rand(3), torch.rand(3))
        model = Bar()
        self.checkTrace(model, x)

    def test_trace_variable_instantiation(self):
        def random_foo(x):
            return Variable(Variable(x) + 1.0)

        random_foo_traced = torch.jit.trace(random_foo, (torch.rand(3, 4),))

        x = torch.rand(5, 6)
        self.assertEqual(random_foo(x), random_foo_traced(x))

    def test_trace_slice_expr_complete_type(self):
        def random_foo(x):
            return x + 1.0

        random_foo_traced = torch.jit.trace(random_foo, (torch.rand(3, 4),))

        @torch.jit.script
        def random_bar(x):
            return random_foo_traced(x)[0:1]

        x = torch.rand(3, 4)
        self.assertEqual(random_bar(x), (x + 1)[0:1])

    def test_export_tensoroption_to(self):
        def foo(x):
            return x.new_tensor(x[0]).cpu() + x

        traced = torch.jit.trace(foo, (torch.rand([2])))
        example_outputs = traced(torch.rand([2]))

        f = io.BytesIO()
        self.assertExpected(torch.onnx._export_to_pretty_string(traced, (torch.rand([2]),), f,
                                                                example_outputs=example_outputs))

    def test_pretty_printer(self):
        @torch.jit.script
        def if_test(a, b):
            # FIXME: use 0 instead of a.
            # c = 0
            c = a
            if bool(a < b):
                c = b
            else:
                c = a
            return c

        @torch.jit.script
        def if_one(a, b):
            c = b
            if bool(a < b):
                c = a
            return c

        @torch.jit.script
        def while_test(a, i):
            while bool(i < 3):
                a *= a
                i += 1
            return a

        @torch.jit.script
        def while_if_test(a, b):
            c = 0
            while bool(a < 10):
                a = a + 1
                b = b + 1
                if bool(a > b):
                    c = 2
                else:
                    c = 3
            return a + 1 + c

        @torch.jit.script
        def loop_use_test(y):
            x = y + 1
            z = x + 5
            while bool(y < 8):
                y += 1
                z = x
            return x, z

        def python_fn(x):
            return x + 10

        @torch.jit.script
        def python_op_name_test(y):
            return python_fn(y)

        @torch.jit.script
        def empty_int_list_test(y):
            x = torch.jit.annotate(List[int], [])
            return x[0]

        @torch.jit.script
        def empty_float_list_test(y):
            return [1.0, 2.0, 3.0]

        @torch.jit.script
        def print_weird_test(y):
            print("hi\016")

        self.assertExpected(if_test.graph.pretty_print(), "if_test")
        self.assertExpected(if_one.graph.pretty_print(), "if_one")
        self.assertExpected(while_test.graph.pretty_print(), "while_test")
        self.assertExpected(while_if_test.graph.pretty_print(), "while_if_test")
        self.assertExpected(loop_use_test.graph.pretty_print(), "loop_use_test")
        self.assertExpected(python_op_name_test.graph.pretty_print(), "python_op_name_test")
        self.assertExpected(empty_int_list_test.graph.pretty_print(), "empty_int_list_test")
        self.assertExpected(empty_float_list_test.graph.pretty_print(), "empty_float_list_test")
        self.assertExpected(print_weird_test.graph.pretty_print(), "print_weird_test")

    def test_cu_escaped_number(self):
        cu = torch.jit.CompilationUnit('''
            def foo(a):
                print("hi\016")
        ''')
        self.assertExpected(cu.foo.graph.pretty_print())

    def test_import_method(self):
        @torch.jit.script
        def foo(x, y):
            return 2 * x + y

        r, _ = foo._python_print()
        mod = torch.jit.ScriptModule()
        torch._C._jit_import_methods(mod, "op_version_set = 0\n{}".format(r), [])
        self.assertExpected(mod.graph.pretty_print())

    def test_function_default_values(self):
        outer_var = torch.tensor(20)
        outer_var2 = torch.tensor(30)
        a = torch.tensor(0.5)
        b = torch.tensor(10)

        @torch.jit.script
        def simple_fn(x, a=a, b=b, c=outer_var + outer_var2):
            return x + a + b + c

        self.assertEqual(
            simple_fn(torch.ones(1)),
            torch.ones(1) + 0.5 + 10 + (20 + 30))
        self.assertEqual(
            simple_fn(torch.ones(1), torch.tensor(1), torch.tensor(3), torch.tensor(4)),
            torch.ones(1) + 1 + 3 + 4)

        outer_c = torch.tensor(9)
        outer_flag = torch.tensor(False)

        @torch.jit.script
        def bool_fn(x, a=outer_c, flag=outer_flag):
            if bool(flag):
                result = x
            else:
                result = x + a
            return result

        self.assertEqual(bool_fn(torch.ones(1)), torch.ones(1) + 9)
        self.assertEqual(
            bool_fn(torch.ones(1), torch.tensor(1), torch.tensor(True)),
            torch.ones(1))

        @torch.jit.script
        def none_fn(x=None):
            # type: (Optional[int]) -> Optional[int]
            return x

        self.assertEqual(none_fn(), None)
        self.assertEqual(none_fn(1), 1)

        @torch.jit.script
        def hints(x, a=0.5, b=10):
            # type: (Tensor, float, int) -> Tensor
            return x + a + b

        self.assertEqual(hints(torch.ones(1)), torch.ones(1) + 0.5 + 10)

        with self.assertRaisesRegex(RuntimeError, "Expected a default value"):

            @torch.jit.script
            def hints_bad_types(x, a=10, b=0.5):
                # type: (Tensor, float, int) -> Tensor
                return x + a + b

    def test_module_default_values(self):
        four = torch.tensor(4)

        class Test(torch.jit.ScriptModule):
            def __init__(self):
                super(Test, self).__init__()

            @torch.jit.script_method
            def forward(self, input, other=four):
                return input + other

        t = Test()
        self.assertEqual(t(torch.ones(1)), torch.ones(1) + 4)

    def test_warnings(self):
        import warnings

        @torch.jit.script
        def fn(x):
            if bool(x < 2):
                warnings.warn("x is less than 2")
            return x

        self.assertExpectedGraph(fn.graph)

    def test_no_erroneous_warnings(self):
        import warnings

        def fn(x):
            if bool(x > 0):
                warnings.warn('This should NOT be printed')
                x += 1
            return x

        with warnings.catch_warnings(record=True) as warns:
            fn_script = torch.jit.script(fn)
            fn_script(torch.tensor(0))
        warns = [str(w.message) for w in warns]
        self.assertEqual(len(warns), 0)

    @unittest.skipIf(sys.platform == "win32", "TODO: need to fix this test case for Windows")
    def test_torch_load_error(self):
        class J(torch.jit.ScriptModule):
            def __init__(self):
                super(J, self).__init__()

            @torch.jit.script_method
            def forward(self, input):
                return input + 100

        j = J()
        with tempfile.NamedTemporaryFile() as f:
            j.save(f.name)
            with self.assertRaisesRegex(RuntimeError, "is a zip"):
                torch.load(f.name)

    def test_legacy_constructors(self):
        def fn(x):
            return x.new_zeros(5, 5, requires_grad=False)

        with warnings.catch_warnings(record=True) as warns:
            torch.jit.trace(fn, (torch.ones(2, 2)))
        warns = [str(w.message) for w in warns]
        self.assertEqual(len(warns), 1)
        self.assertEqual(warns[0], "new_zeros is a legacy constructor and is not supported in the JIT.")


class TestBatched(TestCase):
    # generate random examples and create an batchtensor with them
    def rand_batch(self, *dims):
        dims = [dim for dim in dims if dim != ()]
        xs = [torch.rand(1, *(random.randint(1, size) if b else size for b, size in dims[1:]),
                         requires_grad=True) for i in range(dims[0])]
        xb = BatchTensor(xs, torch.tensor([b for b, d in dims[1:]]).byte())
        return xs, xb

    def test_create_batchtensor(self):
        # create from tensorlist
        xs, batch = self.rand_batch(4, (True, 3), (False, 2), (True, 5))
        self.assertEqual(xs, batch.examples())
        # create from data, mask, dims
        batch2 = BatchTensor(batch.get_data(), batch.get_mask(), batch.get_dims())
        self.assertEqual(xs, batch2.examples())
        # expand a tensor to a batchtensor given batch_size
        xs = torch.rand(3, 4, 5)
        batch3 = BatchTensor(xs, 2)
        xs = xs.unsqueeze(0)
        self.assertEqual([xs, xs], batch3.examples())

    def test_batch_elementwise_unary(self):
        @torch.jit.batch(batch_size=4)
        def tanh(a):
            return torch.tanh(a)

        xs, batch = self.rand_batch(4, (True, 3), (False, 2))
        res_batch = tanh(batch)
        res = [torch.tanh(xs[j]) for j in range(4)]
        self.assertEqual(res, res_batch.examples())

    def test_batch_elementwise_binary(self):
        @torch.jit.batch(batch_size=4)
        def add(a, b):
            return a + b

        xs, batch = self.rand_batch(4, (True, 3), (False, 2))
        xs2, batch2 = xs, batch
        res_batch = add(batch, batch2)
        res = [torch.add(xs[j], xs2[j]) for j in range(4)]
        self.assertEqual(res, res_batch.examples())

        # test broadcast
        xs, batch = self.rand_batch(4, (False, 3), (False, 2))
        b = torch.rand(3, 2)
        res_batch = add(batch, b)
        res = [torch.add(xs[j], b) for j in range(4)]
        self.assertEqual(res, res_batch.examples())

    def test_batch_mm(self):
        @torch.jit.batch(batch_size=4)
        def mm(a, b):
            return torch.mm(a, b)

        xs, batch = self.rand_batch(4, (True, 3), (False, 2))
        xs2, batch2 = self.rand_batch(4, (False, 2), (True, 3))
        res_batch = mm(batch, batch2)
        res = [torch.mm(xs[j].squeeze(0), xs2[j].squeeze(0)).unsqueeze(0) for j in range(4)]
        self.assertEqual(res, res_batch.examples())

        # test broadcast
        b = torch.rand(2, 4)
        res_batch = mm(batch, b)
        res = [torch.mm(xs[j].squeeze(0), b).unsqueeze(0) for j in range(4)]
        self.assertEqual(res, res_batch.examples())

    def test_batch_matmul(self):
        @torch.jit.batch(batch_size=4)
        def matmul(a, b):
            return torch.matmul(a, b)

        def matmul_test(xs, batch, xs2, batch2):
            ys = [torch.matmul(xs[j].squeeze(0), xs2[j].squeeze(0)).unsqueeze(0) for j in range(4)]
            ybs = matmul(batch, batch2)
            self.assertEqual(ys, ybs.examples())

        # 1 dimension * 1 dimension
        xs, batch = self.rand_batch(4, (False, 2))
        xs2, batch2 = self.rand_batch(4, (False, 2))
        matmul_test(xs, batch, xs2, batch2)
        # 1 dimension * 2 dimension
        xs, batch = self.rand_batch(4, (False, 2))
        xs2, batch2 = self.rand_batch(4, (False, 2), (True, 3))
        matmul_test(xs, batch, xs2, batch2)
        # 2 dimension * 1 dimensions
        xs, batch = self.rand_batch(4, (True, 3), (False, 2))
        xs2, batch2 = self.rand_batch(4, (False, 2))
        matmul_test(xs, batch, xs2, batch2)
        # 2 dimension * 2 dimension
        xs, batch = self.rand_batch(4, (True, 3), (False, 2))
        xs2, batch2 = self.rand_batch(4, (False, 2), (True, 3))
        matmul_test(xs, batch, xs2, batch2)

    def test_batch_select(self):
        @torch.jit.batch(batch_size=4)
        def select(x):
            return torch.select(x, 1, 0)

        xs, batch = self.rand_batch(4, (True, 3), (True, 2))
        res_batch = select(batch)
        res = [torch.select(xs[j], 1, 0) for j in range(4)]
        self.assertEqual(res, res_batch.examples())

        xs, batch = self.rand_batch(4, (False, 3), (True, 2))
        res_batch = select(batch)
        res = [torch.select(xs[j], 1, 0) for j in range(4)]
        self.assertEqual(res, res_batch.examples())

    def test_batch_index_select(self):
        @torch.jit.batch(batch_size=4)
        def index_select(x, ind):
            return x.index_select(1, ind)

        xs, batch = self.rand_batch(4, (False, 5), (True, 2))
        ind = [torch.randint(0, 4, (1,), dtype=torch.long) for i in range(4)]
        ind_batch = BatchTensor(ind, torch.tensor([]).byte())
        res_batch = index_select(batch, ind_batch)
        res = [torch.index_select(xs[j], 1, ind[j]) for j in range(4)]
        self.assertEqual(res, res_batch.examples())

    def test_batch_where(self):
        @torch.jit.batch(batch_size=4)
        def where(c, a, b):
            return torch.where(c, a, b)

        xs, batch = self.rand_batch(4, (False, 3), (False, 2))
        xs2, batch2 = self.rand_batch(4, (False, 3), (False, 2))

        dims = [4, (False, 3), (False, 2)]
        xs_cond = [torch.rand(1, 3, 2).byte() for i in range(dims[0])]
        batch_cond = BatchTensor(xs_cond, torch.tensor([b for b, d in dims[1:]]))

        res_batch = where(batch_cond, batch, batch2)
        res = [torch.where(xs_cond[j], xs[j], xs2[j]) for j in range(4)]
        self.assertEqual(res, res_batch.examples())

    def test_batch_argmax(self):
        @torch.jit.batch(batch_size=4)
        def argmax(a):
            return torch.argmax(a, 1)

        xs, batch = self.rand_batch(4, (True, 5), (True, 6))
        res_batch = argmax(batch)
        res = [torch.argmax(xs[j], 1) for j in range(4)]
        self.assertEqual(res, res_batch.examples())

        @torch.jit.batch(batch_size=4)
        def argmax(a):
            return torch.argmax(a, 1, False)

        res_batch = argmax(batch)
        res = [torch.argmax(xs[j], 1, False) for j in range(4)]
        self.assertEqual(res, res_batch.examples())

    def test_batch_topk(self):
        @torch.jit.batch(batch_size=4)
        def topk(a):
            return torch.topk(a, 3, 1)

        xs, batch = self.rand_batch(4, (False, 5), (True, 6))

        # along static dim
        res_batch = topk(batch)
        res = [torch.topk(xs[j], 3, 1)[0] for j in range(4)]
        res_idx = [torch.topk(xs[j], 3, 1)[1] for j in range(4)]
        self.assertEqual(res, res_batch[0].examples())
        self.assertEqual(res_idx, res_batch[1].examples())

        @torch.jit.batch(batch_size=4)
        def topk(a):
            return torch.topk(a, 1, 2)

        # along dynamic dim
        res_batch = topk(batch)
        res = [torch.topk(xs[j], 1, 2)[0] for j in range(4)]
        res_idx = [torch.topk(xs[j], 1, 2)[1] for j in range(4)]
        self.assertEqual(res, res_batch[0].examples())
        self.assertEqual(res_idx, res_batch[1].examples())

    def test_batch_softmax(self):
        @torch.jit.batch(batch_size=4)
        def softmax(a):
            return torch.softmax(a, 1)

        xs, batch = self.rand_batch(4, (False, 5), (True, 6))

        # along static dim
        res_batch = softmax(batch)
        res = [torch.softmax(xs[j], 1) for j in range(4)]
        self.assertEqual(res, res_batch.examples())

        @torch.jit.batch(batch_size=4)
        def softmax(a):
            return torch.softmax(a, 2)

        # along dynamic dim
        res_batch = softmax(batch)
        res = [torch.softmax(xs[j], 2) for j in range(4)]
        self.assertEqual(res, res_batch.examples())

    def test_batch_view(self):
        @torch.jit.batch(batch_size=4)
        def view(a):
            return a.view([4, -1, 3])

        xs, batch = self.rand_batch(4, (True, 5), (False, 3))
        res_batch = view(batch)
        res = [xs[j].view([1, -1, 3]) for j in range(4)]
        self.assertEqual(res, res_batch.examples())

    def test_batch_cat(self):
        @torch.jit.batch(batch_size=4)
        def cat2(a, b):
            return torch.cat([a, b], 2)

        xs, batch = self.rand_batch(4, (True, 5), (False, 3))
        xs2, batch2 = xs, batch
        res_batch = cat2(batch, batch2)
        res = [torch.cat([xs[j], xs2[j]], 2) for j in range(4)]
        self.assertEqual(res, res_batch.examples())

    def test_batch_sum(self):
        @torch.jit.batch(batch_size=4)
        def batch_sum(a):
            return a.sum()

        xs, batch = self.rand_batch(4, (True, 5), (False, 3))
        res_batch = batch_sum(batch)
        res = [xs[j].sum().unsqueeze(0) for j in range(4)]
        self.assertEqual(res, res_batch.examples())

    def test_if_else(self):
        def single_if(a, b):
            if bool(a > b):
                a = a + b
            else:
                a = a - b
            return a

        batch_if = torch.jit.batch(batch_size=4)(single_if)

        a, batch_a = self.rand_batch(4, ())
        b, batch_b = self.rand_batch(4, ())
        res_batch = batch_if(batch_a, batch_b)
        res = [single_if(a[j], b[j]) for j in range(4)]
        self.assertEqual(res, res_batch.examples())

        script_if = torch.jit.script(single_if)
        graph = torch.to_batch_graph(script_if.graph)
        self.assertExpected(canonical(graph))

    def test_if_else_with_scalar(self):
        def single_if(a, b):
            if bool(a > 0.1):
                a = a + b
            else:
                a = a - b
            return a

        batch_if = torch.jit.batch(batch_size=4)(single_if)

        a, batch_a = self.rand_batch(4, ())
        b, batch_b = self.rand_batch(4, ())
        res_batch = batch_if(batch_a, batch_b)
        res = [single_if(a[j], b[j]) for j in range(4)]
        self.assertEqual(res, res_batch.examples())

        script_if = torch.jit.script(single_if)
        graph = torch.to_batch_graph(script_if.graph)
        self.assertExpected(canonical(graph))

    def test_if_noelse(self):
        def single_if(a, b):
            if bool(a > b):
                a = a + b
            return a

        batch_if = torch.jit.batch(batch_size=4)(single_if)

        a, batch_a = self.rand_batch(4, ())
        b, batch_b = self.rand_batch(4, ())
        res_batch = batch_if(batch_a, batch_b)
        res = [single_if(a[j], b[j]) for j in range(4)]
        self.assertEqual(res, res_batch.examples())

        script_if = torch.jit.script(single_if)
        graph = torch.to_batch_graph(script_if.graph)
        self.assertExpected(canonical(graph))

    def test_if_noelse_with_scalar(self):
        def single_if(a, b):
            if bool(a > 0.1):
                a = a + b
            return a

        batch_if = torch.jit.batch(batch_size=4)(single_if)

        a, batch_a = self.rand_batch(4, ())
        b, batch_b = self.rand_batch(4, ())
        res_batch = batch_if(batch_a, batch_b)
        res = [single_if(a[j], b[j]) for j in range(4)]
        self.assertEqual(res, res_batch.examples())

        script_if = torch.jit.script(single_if)
        graph = torch.to_batch_graph(script_if.graph)
        self.assertExpected(canonical(graph))

    def test_while(self):
        def single_while(a, b):
            while bool(a > b):
                a = a - b
            return a

        batch_while = torch.jit.batch(batch_size=4)(single_while)

        a, batch_a = self.rand_batch(4, ())
        b = [torch.abs(torch.rand(1)) for i in range(4)]
        batch_b = BatchTensor(b, torch.tensor([]).byte())
        res_batch = batch_while(batch_a, batch_b)
        res = [single_while(a[j], b[j]) for j in range(4)]
        self.assertEqual(res, res_batch.examples())

        script_while = torch.jit.script(single_while)
        graph = torch.to_batch_graph(script_while.graph)
        self.assertExpected(canonical(graph))

    def test_for(self):
        def single_for(x, y):
            for _ in range(10):
                x = x + y
            return x

        batch_for = torch.jit.batch(batch_size=4)(single_for)

        a, batch_a = self.rand_batch(4, ())
        b, batch_b = self.rand_batch(4, ())
        res_batch = batch_for(batch_a, batch_b)
        res = [single_for(a[j], b[j]) for j in range(4)]
        self.assertEqual(res, res_batch.examples())

        script_for = torch.jit.script(single_for)
        graph = torch.to_batch_graph(script_for.graph)
        self.assertExpected(canonical(graph))

    def test_lstm(self):
        def LSTM(x_all, h, c, w_xi, w_xf, w_xo, w_xc, w_hi, w_hf, w_ho, w_hc, b_i, b_f, b_o, b_c):
            for i in range(x_all.size(1)):
                x = x_all.select(1, i)
                i_t = torch.matmul(x, w_xi) + torch.matmul(h, w_hi) + b_i
                f_t = torch.matmul(x, w_xf) + torch.matmul(h, w_hf) + b_f
                o_t = torch.matmul(x, w_xo) + torch.matmul(h, w_ho) + b_o
                # activations
                i_t = torch.sigmoid(i_t)
                f_t = torch.sigmoid(f_t)
                o_t = torch.sigmoid(o_t)
                # cell computations
                c_t = torch.matmul(x, w_xc) + torch.matmul(h, w_hc) + b_c
                c_t = torch.tanh(c_t)
                c_t = torch.mul(c_t, f_t) + torch.mul(i_t, c_t)
                h_t = torch.mul(o_t, torch.tanh(c_t))
                h = h_t
                c = c_t
            return h

        LSTM_batch = torch.jit.batch(batch_size=4)(LSTM)

        batch_size, input_size, hidden_size = 4, 3, 2
        xs, batch = self.rand_batch(batch_size, (True, 4), (False, input_size))
        hx, h_batch = self.rand_batch(batch_size, (False, hidden_size))
        cx, c_batch = self.rand_batch(batch_size, (False, hidden_size))

        # input to hidden weights
        w_xi = torch.rand(input_size, hidden_size)
        w_xf = torch.rand(input_size, hidden_size)
        w_xo = torch.rand(input_size, hidden_size)
        w_xc = torch.rand(input_size, hidden_size)
        # hidden to hidden weights
        w_hi = torch.rand(hidden_size, hidden_size)
        w_hf = torch.rand(hidden_size, hidden_size)
        w_ho = torch.rand(hidden_size, hidden_size)
        w_hc = torch.rand(hidden_size, hidden_size)
        # bias terms
        b_i = torch.rand(hidden_size)
        b_f = torch.rand(hidden_size)
        b_o = torch.rand(hidden_size)
        b_c = torch.rand(hidden_size)

        ys = [LSTM(xs[j], hx[j], cx[j], w_xi, w_xf, w_xo, w_xc,
                   w_hi, w_hf, w_ho, w_hc, b_i, b_f, b_o, b_c) for j in range(batch_size)]
        ybs = LSTM_batch(batch, h_batch, c_batch, w_xi, w_xf, w_xo, w_xc,
                         w_hi, w_hf, w_ho, w_hc, b_i, b_f, b_o, b_c)
        self.assertEqual(ys, ybs.examples())

    def test_greedy_search(self):
        def greedy(x, h, c, embed, w_xi, w_xf, w_xo, w_xc, w_hi, w_hf, w_ho, w_hc,
                   b_i, b_f, b_o, b_c, w_hs, b_s, iter_num):
            iter_count = torch.zeros_like(iter_num)
            while bool(iter_count < iter_num):
                iter_count = iter_count + 1
                # LSTM Cell
                i_t = torch.matmul(x, w_xi) + torch.matmul(h, w_hi) + b_i
                f_t = torch.matmul(x, w_xf) + torch.matmul(h, w_hf) + b_f
                o_t = torch.matmul(x, w_xo) + torch.matmul(h, w_ho) + b_o
                # activations
                i_t = torch.sigmoid(i_t)
                f_t = torch.sigmoid(f_t)
                o_t = torch.sigmoid(o_t)
                # cell computations
                c_t = torch.matmul(x, w_xc) + torch.matmul(h, w_hc) + b_c
                c_t = torch.tanh(c_t)
                c_t = torch.mul(c_t, f_t) + torch.mul(i_t, c_t)
                h_t = torch.mul(o_t, torch.tanh(c_t))
                h = h_t
                c = c_t
                # calculate feature with max probability
                s_t = torch.matmul(h_t, w_hs) + b_s
                p_t = torch.softmax(s_t, 1)
                i_t = torch.argmax(p_t, 1)
                x = embed.index_select(1, i_t).squeeze(1)
            return h

        greedy_batch = torch.jit.batch(batch_size=4)(greedy)

        batch_size, input_size, hidden_size, vocab_size = 4, 6, 8, 7
        xs, batch = self.rand_batch(batch_size, (False, input_size))
        hx, h_batch = self.rand_batch(batch_size, (False, hidden_size))
        cx, c_batch = self.rand_batch(batch_size, (False, hidden_size))
        embed, embed_batch = self.rand_batch(batch_size, (False, vocab_size), (False, input_size))
        iter_num = [torch.randint(2, 5, (1,)) for i in range(batch_size)]
        iter_num_batch = BatchTensor(iter_num, torch.tensor([]).byte())

        # input to hidden weights
        w_xi = torch.rand(input_size, hidden_size)
        w_xf = torch.rand(input_size, hidden_size)
        w_xo = torch.rand(input_size, hidden_size)
        w_xc = torch.rand(input_size, hidden_size)
        # hidden to hidden weights
        w_hi = torch.rand(hidden_size, hidden_size)
        w_hf = torch.rand(hidden_size, hidden_size)
        w_ho = torch.rand(hidden_size, hidden_size)
        w_hc = torch.rand(hidden_size, hidden_size)
        # bias terms
        b_i = torch.rand(hidden_size)
        b_f = torch.rand(hidden_size)
        b_o = torch.rand(hidden_size)
        b_c = torch.rand(hidden_size)
        # hidden to vocab weights, bias
        w_hs = torch.rand(hidden_size, vocab_size)
        b_s = torch.rand(vocab_size)

        ys = [greedy(xs[j], hx[j], cx[j], embed[j], w_xi, w_xf, w_xo, w_xc,
                     w_hi, w_hf, w_ho, w_hc, b_i, b_f, b_o, b_c, w_hs, b_s, iter_num[j]) for j in range(batch_size)]
        ybs = greedy_batch(batch, h_batch, c_batch, embed_batch, w_xi, w_xf, w_xo, w_xc,
                           w_hi, w_hf, w_ho, w_hc, b_i, b_f, b_o, b_c, w_hs, b_s, iter_num_batch)
        self.assertEqual(ys, ybs.examples())

    def test_beam_search(self):
        def beam(x, h, c, embed, w_xi, w_xf, w_xo, w_xc, w_hi, w_hf, w_ho, w_hc,
                 b_i, b_f, b_o, b_c, w_hs, b_s, iter_num, idx):
            k = 5
            vocab_size = embed.size(1)
            iter_count = torch.zeros_like(iter_num)
            max_len = idx.size(2)
            while bool(iter_count < iter_num):
                iter_count = iter_count + 1
                # LSTM Cell
                i_t = torch.matmul(x, w_xi) + torch.matmul(h, w_hi) + b_i
                f_t = torch.matmul(x, w_xf) + torch.matmul(h, w_hf) + b_f
                o_t = torch.matmul(x, w_xo) + torch.matmul(h, w_ho) + b_o
                # activations
                i_t = torch.sigmoid(i_t)
                f_t = torch.sigmoid(f_t)
                o_t = torch.sigmoid(o_t)
                # cell computations
                c_t = torch.matmul(x, w_xc) + torch.matmul(h, w_hc) + b_c
                c_t = torch.tanh(c_t)
                c_t = torch.mul(c_t, f_t) + torch.mul(i_t, c_t)
                h_t = torch.mul(o_t, torch.tanh(c_t))
                h = h_t
                c = c_t
                # calculate features with max probability
                s_t = torch.matmul(h_t, w_hs) + b_s
                s_t = s_t.view([1, s_t.size(1) * s_t.size(2)])
                p_t = torch.softmax(s_t, 1)
                prob_t, idx_t = torch.topk(p_t, k, 1)
                if(int(idx_t.dim()) > 1):
                    idx_t_tmp = idx_t.squeeze(0)
                else:
                    idx_t_tmp = idx_t
                new_y = torch.fmod(idx_t_tmp, vocab_size)
                pre_y = idx_t_tmp / vocab_size
                x = embed.index_select(1, new_y)
                h = h_t.index_select(1, pre_y)
                c = c_t.index_select(1, pre_y)
                iter = int(iter_count[0])
                idx = torch.cat([idx.narrow(2, 0, iter).index_select(1, pre_y),
                                torch.fmod(idx_t, vocab_size).unsqueeze(-1),
                                idx.narrow(2, iter, max_len - iter)], 2)
                idx = idx.narrow(2, 0, max_len)
            return idx

        beam_batch = torch.jit.batch(batch_size=4)(beam)

        k = 5
        batch_size, input_size, hidden_size, vocab_size = 4, 6, 8, 7
        max_len = 5
        xs, batch = self.rand_batch(batch_size, (False, 1), (False, input_size))
        hx, h_batch = self.rand_batch(batch_size, (False, 1), (False, hidden_size))
        cx, c_batch = self.rand_batch(batch_size, (False, 1), (False, hidden_size))
        embed, embed_batch = self.rand_batch(batch_size, (False, vocab_size), (False, input_size))
        iter_num = [torch.randint(2, max_len + 1, (1,)) for i in range(batch_size)]
        iter_num_batch = BatchTensor(iter_num, torch.tensor([]).byte())

        # input to hidden weights
        w_xi = torch.rand(input_size, hidden_size)
        w_xf = torch.rand(input_size, hidden_size)
        w_xo = torch.rand(input_size, hidden_size)
        w_xc = torch.rand(input_size, hidden_size)
        # hidden to hidden weights
        w_hi = torch.rand(hidden_size, hidden_size)
        w_hf = torch.rand(hidden_size, hidden_size)
        w_ho = torch.rand(hidden_size, hidden_size)
        w_hc = torch.rand(hidden_size, hidden_size)
        # bias terms
        b_i = torch.rand(1, hidden_size)
        b_f = torch.rand(1, hidden_size)
        b_o = torch.rand(1, hidden_size)
        b_c = torch.rand(1, hidden_size)
        # hidden to vocab weights, bias
        w_hs = torch.rand(hidden_size, vocab_size)
        b_s = torch.rand(1, vocab_size)

        idx_batch = torch.jit.BatchTensor(torch.zeros([batch_size, k, max_len], dtype=torch.long),
                                          torch.zeros([batch_size, 1, max_len]).byte(),
                                          torch.tensor([0, 1]).byte())
        idx = [torch.zeros([1, k, max_len], dtype=torch.long) for _ in range(batch_size)]

        ys = [beam(xs[j], hx[j], cx[j], embed[j], w_xi, w_xf, w_xo, w_xc, w_hi, w_hf, w_ho, w_hc,
                   b_i, b_f, b_o, b_c, w_hs, b_s, iter_num[j], idx[j]).narrow(2, 0, int(iter_num[j]))
              for j in range(batch_size)]
        ybs = beam_batch(batch, h_batch, c_batch, embed_batch, w_xi, w_xf, w_xo, w_xc,
                         w_hi, w_hf, w_ho, w_hc, b_i, b_f, b_o, b_c, w_hs, b_s, iter_num_batch, idx_batch)
        self.assertEqual(ys, ybs.examples())


def execWrapper(code, glob, loc):
    if PY2:
        exec(code) in glob, loc
    else:
        exec(code, glob, loc)


class TestScript(JitTestCase):
    @contextmanager
    def capture_stdout(self):
        # No idea how to capture stdout from C++ on Windows
        if WINDOWS:
            yield ['']
            return
        import os
        import fcntl
        import errno
        sys.stdout.flush()
        stdout_fd = os.dup(1)
        r, w = os.pipe()
        try:
            # Override stdout with r - dup is guaranteed to return the lowest free fd
            os.close(1)
            os.dup(w)

            captured_stdout = ['']
            yield captured_stdout
            sys.stdout.flush()  # Make sure that Python hasn't buffered anything

            # Do the ugly dance to read all the data that was written into the pipe
            fcntl.fcntl(r, fcntl.F_SETFL, os.O_NONBLOCK)
            total_stdout = ''
            while True:
                try:
                    total_stdout += os.read(r, 1000).decode('ascii')
                except OSError as e:
                    if e.errno != errno.EAGAIN:
                        raise
                    break
            captured_stdout[0] = total_stdout
        finally:
            # Revert the change, and clean up all fds
            os.close(1)
            os.dup(stdout_fd)
            os.close(stdout_fd)
            os.close(r)
            os.close(w)

    def checkScriptRaisesRegex(self, script, inputs, exception, regex,
                               optimize=True, outputs=None, capture_output=False):
        """
        Checks that a given function will throw the correct exception,
        when executed with normal python, the string frontend, and the AST frontend
        """
        # normal python
        with self.assertRaisesRegex(exception, regex):
            script(*inputs)
        # string frontend
        with self.assertRaisesRegex(exception, regex):
            source = textwrap.dedent(inspect.getsource(script))
            cu = torch.jit.CompilationUnit(source, optimize)
            ge = getattr(cu, script.__name__)
            ge(*inputs)
        # python AST frontend
        with self.assertRaisesRegex(exception, regex):
            ge = torch.jit.script(script, optimize)
            ge(*inputs)

    def test_training_param(self):
        class What(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                # type: (int) -> int
                if self.training:
                    r = x
                else:
                    r = x + 4
                # check double use of training
                if self.training:
                    r = r + 1
                return r

        w = What()
        self.assertEqual(4, w(3))
        w.train(False)
        self.assertEqual(7, w(3))

    def test_jitter_bug(self):
        @torch.jit.script
        def fn2(input, kernel_size):
            # type: (Tensor, List[int]) -> Tensor
            if kernel_size[0] > 1:
                _stride = [2]
            else:
                _stride = kernel_size
            print(_stride, kernel_size)
            return input

        @torch.jit.script
        def fn(input):
            # type: (Tensor) -> Tensor
            return fn2(input, [1])

    def test_annoying_doubles(self):
        mod = types.ModuleType("temp")
        mod.inf = float("inf")
        mod.ninf = float("-inf")
        mod.nan = float("nan")

        with self.disableModuleHook():
            @torch.jit.script
            def foo():
                return math.pi, 0.1, mod.inf, mod.ninf, 2.225073858507201e-308, mod.nan

            pp, table = foo._get_method('forward').python_print()
            ppv = "op_version_set = 0\n{}".format(pp)
            sm = torch.jit.ScriptModule()
            torch._C._jit_import_methods(sm, ppv, table)
            r = foo()
            r2 = sm()
            # use precise assert, we are checking floating point details
            self.assertTrue(r[:-1] == r2[:-1])
            self.assertTrue(math.isnan(r[-1]) and math.isnan(r2[-1]))

    def test_type_annotate(self):

        def foo(a):
            return torch.jit.annotate(torch.Tensor, a)

        self.checkScript(foo, (torch.rand(3),))

        def bar():
            a = torch.jit.annotate(List[int], [])
            for _ in range(10):
                a.append(4)
            return a

        self.checkScript(bar, ())

        def baz(a):
            return torch.jit.annotate(float, a)
        self.checkScript(baz, (torch.rand(()),))

        # test annotate none types
        def annotate_none():
            return torch.jit.annotate(Optional[torch.Tensor], None)

        def annotate_none_no_optional():
            return torch.jit.annotate(torch.Tensor, None)

        self.checkScript(annotate_none, ())
        self.checkScript(annotate_none_no_optional, ())

    def test_robust_op_resolution(self):
        neg = torch.add  # misleading name to make sure we resolve by function

        def stuff(x):
            return neg(x, x)

        a = (torch.rand(3),)
        self.checkScript(stuff, a)

    def test_tuple_io(self):
        def stuff(x):
            # type: (Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]
            a, b = x
            return b, a

        a = (torch.rand(3), torch.rand(3))
        self.checkScript(stuff, (a,))

    def test_tuple_create_return(self):
        def stuff2(x):
            # type: (int) -> Tuple[Tensor, Tensor]
            a = (torch.ones(x), torch.zeros(x))
            return a
        self.checkScript(stuff2, (3,))

    def test_list_io(self):
        def stuff3(x):
            # type: (List[int]) -> Tuple[Tensor, List[int]]
            return torch.ones(x), x
        self.checkScript(stuff3, ([3, 2],))

    def test_bool_list_io(self):
        @torch.jit.script
        def stuff4(x):
            # type: (List[bool]) -> Tuple[List[bool], List[bool], List[List[bool]]]
            return x, [True, False], [[True]]

        li_1, li_2, li_3 = stuff4([True])
        li_3 = li_3[0]
        for li in [li_1, li_2, li_3]:
            self.assertTrue(type(li[0]) == type(True))

    def test_nested_list(self):
        def foo(z):
            # type: (Tuple[int, List[List[int]]]) -> int
            x, y = z
            return y[0][1]
        self.checkScript(foo, ((1, [[1, 2], [3, 4]]),))

    def test_nested_list_construct(self):
        def foo():
            return [[4]] + [[4, 5]]
        self.checkScript(foo, ())

    def test_tensor_shape(self):
        x = torch.empty(34, 56, 78)

        def f(x):
            return x.shape

        self.checkScript(f, (x,))

    def test_tensor_grad(self):
        x = torch.tensor(1.0, requires_grad=True)
        y = torch.tensor(1.0, requires_grad=False)

        def f(x):
            return x.requires_grad

        self.checkScript(f, (x,))
        self.checkScript(f, (y,))

    def test_tensor_dtype(self):
        x_byte = torch.empty(34, 56, 78, dtype=torch.uint8)
        x_long = torch.empty(34, 56, 78, dtype=torch.long)
        x_float32 = torch.empty(34, 56, 78, dtype=torch.float32)

        @torch.jit.script
        def byte(x):
            return x.dtype == torch.uint8

        @torch.jit.script
        def long(x):
            return x.dtype == torch.long

        @torch.jit.script
        def float32(x):
            return x.dtype == torch.float32

        self.assertTrue(byte(x_byte))
        self.assertFalse(byte(x_long))
        self.assertFalse(byte(x_float32))
        self.assertFalse(long(x_byte))
        self.assertTrue(long(x_long))
        self.assertFalse(long(x_float32))
        self.assertFalse(float32(x_byte))
        self.assertFalse(float32(x_long))
        self.assertTrue(float32(x_float32))

    @unittest.skipIf(not RUN_CUDA, "device tests require CUDA")
    def test_tensor_device(self):
        cpu = torch.empty(34, 56, 78, device='cpu')
        gpu = torch.empty(34, 56, 78, device='cuda')

        @torch.jit.script
        def same_device(x, y):
            return x.device == y.device

        self.assertTrue(same_device(cpu, cpu))
        self.assertTrue(same_device(gpu, gpu))
        self.assertFalse(same_device(cpu, gpu))

    @unittest.skipIf(not RUN_CUDA, "device tests require CUDA")
    def test_tensor_to_device(self):
        def to_device(x):
            return x.to(device="cuda").to(device=torch.device("cpu"))

        self.checkScript(to_device, (torch.ones(3, 4),))

    def test_tensor_to_cpu(self):
        def to_cpu(x):
            return x.cpu()

        x = torch.ones(3, 4)
        script_fn = torch.jit.script(to_cpu)
        self.assertEqual(to_cpu(x).device, script_fn(x).device)
        self.checkScript(to_cpu, (x,))

    @unittest.skipIf(not RUN_CUDA, "device tests require CUDA")
    def test_tensor_to_cuda(self):
        def to_cuda(x):
            return x.cuda()

        x = torch.ones(3, 4)
        script_fn = torch.jit.script(to_cuda)
        self.assertEqual(to_cuda(x).device, script_fn(x).device)
        self.checkScript(to_cuda, (x,))

    def test_generic_list_errors(self):
        with self.assertRaisesRegex(RuntimeError, "previously matched to type"):
            @torch.jit.script
            def foo(x):
                return [[x]] + [[1]]

    def test_script_cu(self):
        cu = torch.jit.CompilationUnit('''
            def foo(a):
                b = a
                return b
        ''')
        a = Variable(torch.rand(1))
        self.assertEqual(a, cu.foo(a))

    # because the compilation unit ingests python strings
    # to use an escape sequence escape the backslash (\\n = \n)
    def test_string_cu(self):
        cu = torch.jit.CompilationUnit('''
            def foo(a):
                print(a, """a\\n\tb\\n""", 2, "a\
a")
                return a
        ''')
        self.assertExpected(str(cu.foo.graph))

    def test_string_ops(self):
        def foo():
            a = "a" + "b"
            return a + a, "ab" == "b", "ab" != "b", "ab" == "ab", "ab" != "ab"

        self.checkScript(foo, ())

    def test_string_new_line(self):
        with self.assertRaisesRegex(RuntimeError, "expected a valid token*"):
            torch.jit.CompilationUnit('''
            def test_while(a):
                print("
                    a")
                return a
            ''')

    def test_string_single_escape(self):
        with self.assertRaisesRegex(RuntimeError, "expected a valid token*"):
            torch.jit.CompilationUnit('''
            def test_while(a):
                print("\\")
                return a
            ''')

    def test_script_annotation(self):
        @torch.jit.script
        def foo(a):
            return a + a + a
        s = Variable(torch.rand(2))
        self.assertEqual(s + s + s, foo(s))

    def test_inf(self):
        @torch.jit.script
        def foo(a):
            return a < float('inf')
        s = torch.rand(1)
        self.assertTrue(foo(s))

        @torch.jit.script
        def bar(a):
            return a > float('-inf')
        s = torch.rand(1)
        self.assertTrue(foo(s))

    def test_add(self):
        def func(a, b):
            c = a + b
            c += a
            return c

        a = torch.rand(1, requires_grad=True)
        b = torch.rand(1, requires_grad=True)
        self.checkScript(func, (a, b), optimize=True)

    def test_mul(self):
        def func(a, b):
            return a * b

        a = torch.rand(1, requires_grad=True)
        b = torch.rand(1, requires_grad=True)
        self.checkScript(func, (a, b), optimize=True)

    @unittest.skipIf(not PY35, "Python 3.5 needed")
    def test_matmul_py3(self):
        code = dedent("""
        def fn(a, b):
            return a @ b
        """)

        with tempfile.TemporaryDirectory() as tmp_dir:
            script_path = os.path.join(tmp_dir, 'script.py')
            with open(script_path, 'w') as f:
                f.write(code)
            fn = get_fn('test_matmul_py3', script_path)

            a = torch.rand(4, 3, requires_grad=True)
            b = torch.rand(3, 2, requires_grad=True)
            self.checkScript(fn, (a, b), optimize=True)

    def test_pow(self):
        def func(a, b):
            return a ** b

        def func2(a, b, c, d):
            return c + a ** b ** d

        a = torch.rand(1, requires_grad=True)
        b = torch.rand(1, requires_grad=True)
        c = torch.rand(1, requires_grad=True)
        d = torch.rand(1, requires_grad=True)
        self.checkScript(func, (a, b), optimize=True)
        self.checkScript(func2, (a, b, c, d), optimize=True)

    def test_triple(self):
        def func(x):
            return 3. * x

        x = torch.rand(1, dtype=torch.float, requires_grad=True)
        self.checkScript(func, [x], optimize=True)

    def test_slice(self):
        def func(x):
            return x[:5]

        x = torch.rand(10, dtype=torch.float, requires_grad=True)
        self.checkScript(func, [x], optimize=True)

        def func2(x):
            return x[5:]

        self.checkScript(func2, [x], optimize=True)

    def test_gather(self):
        def func(x):
            return x[0]

        x = torch.rand(10, dtype=torch.float, requires_grad=True)
        self.checkScript(func, [x], optimize=True)

    def test_random(self):
        @torch.jit.script
        def f(mean, std):
            return torch.normal(mean, std)

        mean, std = torch.zeros(5, 5), torch.ones(5, 5)
        with torch.random.fork_rng(devices=[]):
            output = torch.normal(mean, std)
        with torch.random.fork_rng(devices=[]):
            script_output = f(mean, std)
        self.assertEqual(output, script_output)

    def _check_code(self, code_str, fn_name, inputs):
        scope = {}
        exec(code_str, globals(), scope)
        cu = torch.jit.CompilationUnit(code_str)
        self.assertEqual(cu.func(*inputs), scope[fn_name](*inputs))

    @unittest.skipIf(not RUN_CUDA, 'no CUDA')
    def test_scriptmodule_releases_tensors_cuda(self):
        @torch.jit.script
        def fn(x, y):
            return x.sigmoid() * y.tanh()

        def test(backward=False):
            x = torch.randn(3, 3, dtype=torch.double, device='cuda', requires_grad=True)
            y = torch.randn(3, 3, dtype=torch.double, device='cuda', requires_grad=True)
            out = fn(x, y)
            if backward:
                out.sum().backward()

        with self.assertLeaksNoCudaTensors():
            test()
            test()
            test()

        with self.assertLeaksNoCudaTensors():
            test(backward=True)
            test(backward=True)
            test(backward=True)

    def test_index(self):
        def consec(size, start=0):
            numel = torch.tensor(size).prod().item()
            return torch.arange(numel).view(size)

        def check_indexing(indexing, tensor):
            template = dedent("""
            def func(x):
                return x{}
            """)

            self._check_code(template.format(indexing), "func", [tensor])

        def check_dynamic_indexing(indexing, tensor, value1, value2):
            value1 = torch.tensor(value1)
            value2 = torch.tensor(value2)

            template = dedent("""
            def func(x, value1, value2):
                i = int(value1)
                j = int(value2)
                return x{}
            """)

            self._check_code(template.format(indexing), "func", [tensor, value1, value2])

        # basic slices
        check_indexing('[0]', consec((3, 3)))
        check_indexing('[1]', consec((3, 3), 10))
        check_indexing('[2]', consec((3, 3), 19))
        check_indexing('[2]', consec((3,)))
        check_indexing('[-1]', consec((3, 3), 19))
        check_indexing('[0:2]', consec((3, 3, 3)))
        check_indexing('[1:-1]', consec((3, 3, 3)))
        check_indexing('[-3:-1]', consec((6, 3)))
        check_indexing('[1:]', consec((3, 3)))
        check_indexing('[:1]', consec((3, 3)))
        check_indexing('[:]', consec((3, 2)))

        # multi-dim: indexes
        check_indexing('[0, 1]', consec((3, 3)))
        check_indexing('[0, 1]', consec((3, 3, 2)))
        check_indexing('[1, 0, 2]', consec((3, 3, 3)))
        check_indexing('[2, -1]', consec((3, 3)))

        # multi-dim: mixed slicing and indexing
        check_indexing('[0, 1:2]', consec((3, 3)))
        check_indexing('[0, :1]', consec((3, 3, 2)))
        check_indexing('[1, 2:]', consec((3, 3, 3)))
        check_indexing('[-1, 1:, 0]', consec((3, 3, 3, 3)))
        check_indexing('[1:, -1, 0]', consec((3, 3, 3, 3)))
        check_indexing('[-1, 2:, 1:2]', consec((3, 3, 3, 3)))
        check_indexing('[-1, 1:, 0]', consec((3, 3, 3, 3)))
        check_indexing('[-1, :, 0, 2]', consec((3, 3, 3, 3)))

        # zero-sized slices
        check_indexing('[0:0]', consec((2, 2)))
        check_indexing('[0:0, 1]', consec((3, 3)))

        # trivial expression usage
        check_indexing('[1+1]', consec((3, 3)))
        check_indexing('[1:(0 + 2)]', consec((3, 3, 3)))

        # dynamic expression usage
        check_dynamic_indexing("[i + j]", consec((3, 3)), 0, 1)
        check_dynamic_indexing("[i:j, i]", consec((3, 3, 2)), 0, 2)

    def test_method_on_number(self):
        def func():
            c = 1
            return c.add(1)
        with self.assertRaisesRegex(RuntimeError, 'Cannot call methods on numbers'):
            torch.jit.script(func)

    # testing implicit conversion of tensors to scalars to match function arguments
    def test_scalar_to_num_conversions(self):
        @torch.jit.script
        def multiple_defs(x):
            c = 1
            x = x + c
            return x

        self.assertTrue("ImplicitTensorToNum" not in str(multiple_defs.graph))

        @torch.jit.script
        def tensor_to_int_script(x, tensor):
            return x.unsqueeze(tensor)

        def tensor_to_int(x, tensor):
            return x.unsqueeze(tensor)

        @torch.jit.script
        def tensor_to_float_script(x, tensor):
            return x.addcmul(tensor, tensor, value=tensor)

        def tensor_to_float(x, tensor):
            return x.addcmul(tensor, tensor, value=tensor)

        x = torch.zeros(10)
        # float tensor, float tensor with grad, int tensor (can't set grad on int tensor)
        tensors = [torch.tensor(1.1),
                   torch.tensor(1.1, requires_grad=True),
                   torch.tensor(0),
                   torch.tensor([2])]

        script_funs = [tensor_to_int_script, tensor_to_float_script]
        funs = [tensor_to_int, tensor_to_float]

        # return the result, or whether exception was thrown
        def test_func(func, x, tensor):
            try:
                result = func(x, tensor)
            except RuntimeError as e:
                result = True
            except TypeError as e:
                result = True
            return result

        # assert result or exception equal for each (function, inputs)
        for tensor in tensors:
            for i in range(len(script_funs)):
                self.assertEqual(test_func(script_funs[i], x, tensor), test_func(funs[i], x, tensor))

    def test_tuple_to_opt_list(self):
        @torch.jit.script
        def foo(x):
            # type: (Optional[List[int]]) -> int
            return 1

        @torch.jit.script
        def tuple_call():
            return foo((1, 2))

    def test_advancedindex(self):
        def consec(size, start=0):
            numel = torch.tensor(size).prod().item()
            return torch.arange(numel).view(size)

        def check_indexing(indexing, tensor, **kwargs):
            indices_dict = kwargs

            template = dedent("""
            def func(x{formals}):
                return x{expr}
            """)

            formals = []
            values = []
            for formal, value in indices_dict.items():
                formals.append(formal)
                values.append(value)

            formals = ''.join(map(', {}'.format, formals))
            inputs = [tensor] + values
            self._check_code(template.format(formals=formals, expr=indexing),
                             "func", inputs)

        # Indexing with tensor (basic)
        check_indexing('[i]', consec((3, 3)), i=torch.tensor([0]))
        check_indexing('[i]', consec((3, 3)), i=torch.tensor(1))
        check_indexing('[i]', consec((3, 3)), i=torch.tensor([-2]))
        check_indexing('[i]', consec((3, 3), 2), i=torch.tensor([0, 0]))
        check_indexing('[i]', consec((3, 3, 2, 2)), i=torch.tensor([0, -2, 1]))

        # NB: indexing with tensors and indexing with sequences can be implemented
        # in a very similar way (sequences are converted to tensors), so only one
        # case needs to be tested extensively.
        # XXX: When we can index with sequences, replace these cases with
        # sequence indexing expressions; those are much easier to read.

        # Misc sequence advanced indexing
        inp = consec((4, 8, 5))
        to_check = [
            # [[0, 2], [1, 3]]
            ['[i, j]', {'i': [0, 2], 'j': [1, 3]}],
            # [[0, 2], [1, 3], [1, 1]]
            ['[i, j, k]', {'i': [0, 2], 'j': [1, 3], 'k': [1, 1]}],
            # [[0, 2], 1, [1, 1]]
            ['[i, j, k]', {'i': [0, 2], 'j': 1, 'k': [1, 1]}],
            # [:, :, [0, 3, 4]]
            ['[:, :, i]', {'i': [0, 3, 4]}],
            # [:, [2, 4, 5, 7], 2:4]
            ['[:, i, 2:4]', {'i': [0, 2, 3]}],
            # [[2, 3], :, :]
            ['[i, :, :]', {'i': [2, 3]}],
            # [:, [0, 2, 3], [1, 3, 4]]
            ['[:, i, j]', {'i': [0, 2, 3], 'j': [1, 3, 4]}],
            # [:, [0], [1, 2, 4]]
            ['[:, i, j]', {'i': [0], 'j': [1, 2, 4]}],
            # [:, [0, 1, 3], [4]]
            ['[:, i, j]', {'i': [0, 1, 3], 'j': [4]}],
            # [:, [[0, 1], [1, 0]], [[2, 3]]]
            ['[:, i, j]', {'i': [[0, 1], [1, 0]], 'j': [[2, 3]]}],
            # [:, [[0, 1], [2, 3]], [[0]]]
            ['[:, i, j]', {'i': [[0, 1], [2, 3]], 'j': [[0]]}],
            # [:, [[5, 6]], [[0, 3], [4, 4]]]
            ['[:, i, j]', {'i': [[5, 6]], 'j': [[0, 3], [4, 4]]}],
            # [[0, 2, 3], [1, 3, 4], :]
            ['[i, j, :]', {'i': [0, 2, 3], 'j': [1, 3, 4]}],
            # [0, [1, 2, 4], :]
            ['[i, j, :]', {'i': 0, 'j': [1, 2, 4]}],
            # [[0, 1, 3], 4, :]
            ['[i, j, :]', {'i': [0, 1, 3], 'j': 4}],
            # [[[0, 1], [1, 0]], [[2, 1], [3, 5]], :]
            ['[i, j, :]', {'i': [[0, 1], [1, 0]], 'j': [[2, 1], [3, 5]]}],
            # [[[0, 1], [1, 0]], [[2, 3]], :]
            ['[i, j, :]', {'i': [[0, 1], [1, 0]], 'j': [[2, 3]]}],
            # [[[0, 1], [2, 3]], [[0]], :]
            ['[i, j, :]', {'i': [[0, 1], [2, 3]], 'j': [[0]]}],
            # [[[2, 1]], [[0, 3], [4, 4]], :]
            ['[i, j, :]', {'i': [[2, 1]], 'j': [[0, 3], [4, 4]]}],
            # [[[2]], [[0, 3], [4, 1]], 0:2]
            ['[i, j, 0:2]', {'i': [[2]], 'j': [[0, 3], [4, 1]]}],
        ]

        for expr, argdict in to_check:
            tensordict = {k: torch.tensor(v) for (k, v) in argdict.items()}
            check_indexing(expr, inp, **tensordict)

    def test_keyword(self):
        @torch.jit.script
        def func(x):
            return torch.sum(x, dim=0)

        x = torch.rand(10, dtype=torch.float, requires_grad=True)
        y = func(x)
        y2 = torch.sum(x, dim=0)
        self.assertEqual(y, y2)

    def test_constant_pooling(self):
        def func(cond):
            a = 1
            b = 4
            c = 0
            d = "abc"
            e = "bcd"
            f = "abc"
            x = torch.ones([2])
            y = x * 4
            z = torch.ones([2])
            if bool(cond):
                c = b - a
            else:
                y = torch.rand(0)
                if bool(cond):
                    y = torch.rand(1)
                print(d, e, f, x, y, z)
            b = b - a
            return a, b, c, x, y

        self.checkScript(func, torch.tensor([1]))
        graph = torch.jit.script(func).graph
        self.run_pass('constant_propagation', graph)
        self.run_pass('constant_pooling', graph)
        self.assertExpectedGraph(graph)

    def test_constant_pooling_none(self):
        @torch.jit.script
        def typed_nones(a=None, b=None, c=None):
            # type: (Optional[int], Optional[bool], Optional[Tensor]) -> Tuple[Optional[int], Optional[bool], Optional[Tensor]] # noqa
            return a, b, c

        @torch.jit.script
        def test(a):
            # type: (bool) -> None
            if a:
                print(typed_nones())
            else:
                print(typed_nones())

        graph_str = str(test.graph)
        self.assertTrue(graph_str.count("bool? = prim::None") == 1)
        self.assertTrue(graph_str.count("int? = prim::None") == 1)
        self.assertTrue(graph_str.count("None = prim::None") == 1)

    def test_literal(self):
        def func1(a, b):
            c = a, b
            d, e = c
            return d + e

        def func2(a, b):
            c = a, (a, b)
            d, e = c
            f, g = e
            return d + f + g

        def func3(a, b):
            # type: (float, float) -> float
            c = 0., (0., 0.)
            x = True
            while x:
                x = False
                c = a, (a, b)
            d, e = c
            f, g = e
            return d + f + g

        a = torch.rand(1, requires_grad=True)
        b = torch.rand(1, requires_grad=True)
        self.checkScript(func1, (a, b), optimize=True)
        self.checkScript(func2, (a, b), optimize=True)
        self.checkScript(func3, (a.item(), b.item()), optimize=True)

    def test_expand(self):
        @torch.jit.script
        def func(x, y):
            return x + y

        x = torch.rand(2, 3, dtype=torch.float, requires_grad=True)
        y = torch.rand(3, dtype=torch.float, requires_grad=True)
        out = func(x, y)
        self.assertEqual(func(x, y), x + y)

        grad = torch.randn(2, 3, dtype=torch.float)
        out.backward(grad)
        self.assertEqual(x.grad, grad)
        self.assertEqual(y.grad, grad.sum(dim=0))

    def test_sum(self):
        @torch.jit.script
        def func(x):
            return x.sum(dim=[4])

        @torch.jit.script
        def func2(x):
            return x.sum(dim=4)

        self.assertExpected(canonical(func.graph), subname='1')
        # test that shape analysis is written correctly for sum with IntArrayRef[1] dim argument
        torch._C._jit_pass_shape_analysis(
            func2.graph, (torch.zeros(1, 1, 1, 1, 4),), False)
        self.assertExpected(canonical(func2.graph), subname='2')

    def test_cat(self):
        @torch.jit.script
        def func(x):
            return torch.cat((x, x), dim=0)

        x = torch.rand(10, dtype=torch.float, requires_grad=True)
        self.assertEqual(func(x), torch.cat((x, x), dim=0))

        @torch.jit.script
        def func2(x, y):
            return torch.cat((x, x), y)

        x = torch.rand([2, 2])
        y = torch.tensor(1)
        self.assertEqual(func2(x, y), torch.cat((x, x), y))

    def test_cat_lifts(self):
        @torch.jit.script
        def foo(x):
            return torch.cat([x, x], dim=1)

        @torch.jit.script
        def foo2(x):
            return torch.cat([], dim=1)

        @torch.jit.script
        def foo3(x):
            return torch.cat([x], dim=1)

        self.assertExpected(
            canonical(foo.graph) +
            canonical(foo2.graph) +
            canonical(foo3.graph))

    def test_list_literal(self):
        def reassign():
            x = [1]
            if True:
                x = [2, 3]
            return
        self.checkScript(reassign, (), optimize=False)

        def reassign_arity_change():
            x = [1]
            if True:
                x = [1, 2, 3]
            return
        self.checkScript(reassign_arity_change, (), optimize=False)

        def reassign_from_empty_literal():
            x = []
            if True:
                x = [1, 2, 3]
            return
        with self.assertRaisesRegex(RuntimeError, r"previously has type Tensor\[\]"):
            self.checkScript(reassign_from_empty_literal, (), optimize=False)

        def reassign_from_empty_builtin():
            x = torch.jit.annotate(List[int], [])
            if True:
                x = [1, 2, 3]
            y = torch.jit.annotate(List[float], [])
            if True:
                y = [1.0, 2.0, 3.0]
            z = []
            if True:
                z = [torch.randn([1])]
            return
        self.checkScript(reassign_from_empty_builtin, (), optimize=False)

        def reassign_bad_type():
            x = [1]
            if True:
                x = [1.0]
            return
        with self.assertRaisesRegex(RuntimeError, "previously has type"):
            self.checkScript(reassign_bad_type, (), optimize=False)

        def reassign_nested():
            x = torch.jit.annotate(List[int], [])
            if True:
                x = [1, 2, 3]
                if True:
                    x = [1.0]
            return
        with self.assertRaisesRegex(RuntimeError, "previously has type"):
            self.checkScript(reassign_nested, (), optimize=False)

    def test_list_gather(self):
        def index():
            a = [1, 2, 3]
            return a[1]

        self.checkScript(index, ())

        def negative_index():
            a = [1, 2, 3]
            return a[-1]

        self.checkScript(negative_index, ())

        def bad_index():
            a = [1, 2, 3]
            return a[4]

        self.checkScriptRaisesRegex(bad_index, (), IndexError,
                                    "list index out of range")

        def bad_negative_index():
            a = [1, 2, 3]
            return a[-5]

        self.checkScriptRaisesRegex(bad_negative_index, (), IndexError,
                                    "list index out of range")

    def test_tensor_len(self):
        def func(x):
            return len(x)

        self.checkScript(func, [torch.ones(4, 5, 6)])

    def test_list_len(self):
        def func():
            a = [1, 2, 3]
            return len(a) == 3

        self.checkScript(func, ())

        def func2():
            a = []
            return len(a) == 0

        self.checkScript(func2, ())

    def test_list_ops(self):
        def test_equality():
            a = [1, 2, 3]
            b = [1, 2, 3]
            return a == b

        self.checkScript(test_equality, (), optimize=True)

        def test_inequality():
            a = [1, 2, 3]
            b = [1, 2, 3]
            return a != b

        self.checkScript(test_equality, (), optimize=True)

        def test_non_equality():
            a = [1, 2, 3]
            b = [3]
            return a == b

        self.checkScript(test_non_equality, (), optimize=True)

        def test_non_inequality():
            a = [1, 2, 3]
            b = [3]
            return a != b

        self.checkScript(test_non_equality, (), optimize=True)

        def test_list_equality_as_cond():
            a = [1, 2, 3]
            b = [3]
            if a == b:
                c = 1
            else:
                c = 2
            return c

        self.checkScript(test_list_equality_as_cond, (), optimize=True)

        def test_list_add():
            a = [1, 2, 3]
            b = [2]
            c = a + b
            return c == [1, 2, 3, 2]

        self.checkScript(test_list_add, (), optimize=True)

        def test_list_add_empty():
            a = [1, 2, 3]
            b = torch.jit.annotate(List[int], [])
            c = a + b
            return c == [1, 2, 3]

        self.checkScript(test_list_add_empty, (), optimize=True)

        def test_tensor_list_equality():
            t1 = torch.ones([1, 1])
            t2 = torch.ones([1, 1])
            x = [t1, t2]
            y = [t2, t1]
            return x == y

        self.checkScript(test_tensor_list_equality, (), optimize=True)

        def test_invalid_list_equality():
            t1 = torch.ones([2, 2])
            t2 = torch.ones([2, 2])
            x = [t1, t2]
            y = [t2, t1]
            # will throw since the tensors have more than one element
            return x == y

        self.checkScriptRaisesRegex(
            test_invalid_list_equality,
            (),
            RuntimeError,
            "bool value of Tensor")

    def test_list_slice(self):
        def test_regular_slice():
            a = [0, 1, 2, 3, 4]
            return a[2:3] == [2]
        self.checkScript(test_regular_slice, ())

        def test_open_ended_slice():
            a = [0, 1, 2, 3, 4]
            return a[2:] == [2, 3, 4]
        self.checkScript(test_open_ended_slice, ())

        def test_open_ended_slice2():
            a = [0, 1, 2, 3, 4]
            return a[:2] == [0, 1]
        self.checkScript(test_open_ended_slice2, ())

        def test_negative_slice():
            a = [0, 1, 2, 3, 4]
            return a[:-1] == [0, 1, 2, 3]
        self.checkScript(test_negative_slice, ())

        def test_negative_slice2():
            a = [0, 1, 2, 3, 4]
            return a[-3:-1] == [2, 3]
        self.checkScript(test_negative_slice2, ())

        def test_backward_slice():
            a = [0, 1, 2, 3, 4]
            return a[3:2] == torch.jit.annotate(List[int], [])
        self.checkScript(test_backward_slice, ())

        def test_over_slice():
            a = [0, 1, 2, 3, 4]
            return a[3:10] == [3, 4]
        self.checkScript(test_backward_slice, ())

    def test_mutable_list_append(self):
        def test_append():
            a = [0, 1]
            a.append(2)
            a.append(3)
            return a == [0, 1, 2, 3]
        self.checkScript(test_append, ())

    def test_mutable_list_append_2(self):
        def test_append_2():
            a = [0, 1]
            a.append(2)
            a = [1]
            a.append(4)
            return a == [1, 4]
        self.checkScript(test_append_2, ())

    def test_mutable_list_append_if(self):
        def test_append_if():
            a = [1]
            if True:
                a.append(4)
            return a == [1, 4]
        self.checkScript(test_append_if, ())

    def test_mutable_list_append_if_else(self):
        def test_append_if_else():
            a = [1]
            if False:
                a.append(4)
            else:
                a.append(10)
            return a == [1, 10]
        self.checkScript(test_append_if_else, ())

    def test_mutable_list_append_loop(self):
        def test_append_loop():
            a = torch.jit.annotate(List[int], [])
            for i in range(5):
                a.append(i)

            return a == [0, 1, 2, 3, 4]
        self.checkScript(test_append_loop, ())

    def test_mutable_list_append_loop_if(self):
        def test_append_loop_if():
            a = torch.jit.annotate(List[int], [])
            for i in range(5):
                if i > 3:
                    a.append(i)
                else:
                    a.append(0)

            return a == [0, 0, 0, 0, 4]
        self.checkScript(test_append_loop_if, ())

    def test_mutable_list_nested_loop(self):
        def test_nested_loop():
            a = torch.jit.annotate(List[int], [])
            for i in range(2):
                for j in range(2):
                    a.append(i + j)

            return a == [0, 1, 1, 2]
        self.checkScript(test_nested_loop, ())

    def test_mutable_list_function_inline(self):
        @torch.jit.script
        def bar(y):
            # type: (List[int]) -> None
            y.append(4)

        @torch.jit.script
        def foo():
            x = [1, 2, 3]
            bar(x)
            return x

        self.assertEqual(foo(), [1, 2, 3, 4])

    def test_func_call(self):
        script = '''
        def add(a, b):
            return a + b

        def mul(a, x):
            return a * x

        def func(alpha, beta, x, y):
            return add(mul(alpha, x), mul(beta, y))
        '''
        alpha = torch.rand(1, dtype=torch.float, requires_grad=True)
        beta = torch.rand(1, dtype=torch.float, requires_grad=True)
        x = torch.rand(3, dtype=torch.float, requires_grad=True)
        y = torch.rand(3, dtype=torch.float, requires_grad=True)
        outputs = alpha * x + beta * y
        # NOTE: cannot optimize yet because broadcasts are not inserted before the fuser runs
        self.checkScript(script, [alpha, beta, x, y], optimize=False, outputs=outputs)

    def test_view_shape_prop(self):
        cu = torch.jit.CompilationUnit('''
        def test_view_shape_prop(a):
            return a.view(size=[-1])
        ''')
        inputs = [torch.zeros(10, 10)]
        outputs = torch.zeros(100)

        real_outs = cu.test_view_shape_prop(*inputs)
        self.assertEqual(real_outs, outputs)

    def test_view_listconstruct_shape_prop(self):
        def fn(x):
            B = x.size(0)
            C = x.size(1)
            T = x.size(2)
            return x.view(T, B, C)

        x = torch.randn(3, 1, 5, requires_grad=True)
        graph = torch.jit.script(fn).graph
        torch._C._jit_pass_shape_analysis(graph, (x,), False)
        self.assertTrue(next(graph.outputs()).type().kind() != 'DynamicType')

    def test_integral_shape_inference(self):
        cu = torch.jit.CompilationUnit('''
        def test_integral_shape_inference(a):
            return a / a
        ''')
        inputs = [torch.ones(10, 10).type(torch.LongTensor)]
        outputs = torch.ones(10, 10)

        self.assertEqual(cu.test_integral_shape_inference(*inputs), outputs)

    def test_fuser_multiple_blocks(self):
        cu = torch.jit.CompilationUnit('''
        def test_fuser_multiple_blocks(this, that, theother, meme):
            i = 0
            while i < 20:
                this = torch.cat([this, meme], dim=0)
                that = torch.cat([that, meme], dim=0)
                theother = torch.cat([theother, meme], dim=0)
                i = i + 1
            return this, that, theother
        ''')

        inputs = [torch.ones(0, 10, 10)] * 3
        inputs += [torch.ones(1, 10, 10)]
        outputs = [torch.ones(20, 10, 10)] * 3

        self.assertEqual(cu.test_fuser_multiple_blocks(*inputs), outputs)

    def test_dropout_script(self):

        eg = torch.zeros(1, 2, 3, requires_grad=True)

        @_trace(eg)
        def foo(x):
            x = torch.neg(x)
            return F.dropout(x)

        class MyDrop(nn.Module):
            def forward(self, x):
                return foo(x)

        f = io.BytesIO()
        torch.onnx.export(MyDrop(), (eg,), f, verbose=False)

    @unittest.skip("RuntimeError: VariableType::ID() not implemented")
    def test_cast(self):
        script = '''
        def to_int(x):
            return int(x)
        '''
        x = Variable(torch.FloatTensor([1.1, 2.3]), requires_grad=True)
        out = Variable(torch.IntTensor([1, 2]), requires_grad=True)
        self.checkScript(script, [x], optimize=True, outputs=[out], func='to_int')

    def test_python_frontend(self):
        def fn(x, y, z):
            q = None
            q = x + y - z.sigmoid()
            print(q)
            w = -z
            if not x and not y and z:
                m = x if not z else y
            while x < y > z:
                q = x
            assert 1 == 1, "hello"
            return x

        ast = torch.jit.frontend.get_jit_ast(fn, is_method=False)
        self.assertExpected(str(ast))

    @unittest.skipIf(not PY2, "Requires python 2")
    def test_python_frontend_py2(self):
        def fn():
            raise Exception("hello")
        ast = torch.jit.frontend.get_jit_ast(fn, is_method=False)
        self.assertExpected(str(ast))

    @unittest.skipIf(PY2, "Requires python 3")
    def test_python_frontend_py3(self):
        def fn():
            raise Exception("hello")
        ast = torch.jit.frontend.get_jit_ast(fn, is_method=False)
        self.assertExpected(str(ast))

    def _make_scalar_vars(self, arr, dtype):
        return [torch.tensor(val, dtype=dtype) for val in arr]

    def test_string_print(self):
        def func(a):
            print(a, "a" 'b' '''c''' """d""", 2, 1.5)
            return a

        inputs = self._make_scalar_vars([1], torch.int64)
        self.checkScript(func, inputs, capture_output=True)

    def test_while(self):
        def func(a, b, max):
            while bool(a < max):
                a = a + 1
                b = b + 1
            c = a + b
            return c

        inputs = self._make_scalar_vars([1, 1, 10], torch.int64)
        self.checkScript(func, inputs, optimize=True)

    def test_fibb(self):
        def func(lim):
            first = 1
            second = 1
            i = 1
            somenum = 5
            dontmutateme = 3
            third = 0
            while bool(i < lim):
                third = first + second
                first = second
                second = third
                j = 0
                while j < 10:
                    somenum = somenum * 2
                    j = j + 1
                i = i + j
                i = i + dontmutateme

            st = second + third
            fs = first + second
            return third, st, fs

        inputs = self._make_scalar_vars([10], torch.int64)
        self.checkScript(func, inputs, optimize=True)

    def test_if(self):
        def func(a, b):
            # type: (int, int) -> int
            d = 3
            if bool(a > 10):
                a = 3 + d
            else:
                b = 3 + d
                d = 4
            c = a + b
            return c

        inputs = self._make_scalar_vars([1, -1], torch.int64)
        self.checkScript(func, inputs, optimize=True)

    def test_if_for_in_range(self):
        def func(a, b):
            # type: (int, int) -> int
            d = 3
            for _ in range(20):
                if bool(a > 10):
                    a = 3 + d
                else:
                    b = 3 + d
                    d = 4
                c = a + b
            return d
        inputs = self._make_scalar_vars([1, -1], torch.int64)
        self.checkScript(func, inputs, optimize=True)

    def test_if_noelse(self):
        def func(a, b):
            if bool(a > 10):
                a = 3 + b
            c = a + b
            return c

        inputs = self._make_scalar_vars([-1, 1], torch.int64)
        self.checkScript(func, inputs, optimize=True)

    def test_if_is_none_dispatch(self):
        class Test(torch.jit.ScriptModule):
            __constants__ = ['b']

            def __init__(self, b=None):
                super(Test, self).__init__()
                self.b = b

            @torch.jit.script_method
            def forward(self, input, opt=None):
                # type: (Tensor, Optional[Tensor]) -> Tensor
                x = input
                if self.b is not None:
                    x = self.b(input)

                if self.b is None:
                    x = input + 2

                if opt is not None:
                    opt = torch.jit._unwrap_optional(opt)
                    x = opt + x

                if opt is None:
                    x = x + 4

                return x

        inputs = torch.zeros(1, 2)
        self.assertExpectedGraph(Test().graph)
        out = Test()(inputs)
        self.assertEqual(out, inputs + 6)

    def test_explicit_bool_cast(self):
        with self.assertRaisesRegex(RuntimeError, "expected a boolean"):
            @torch.jit.script
            def test_bool_cast(a):
                if a:
                    return a + 2
                return a + 1

    def test_while_nonexistent_value(self):
        with self.assertRaisesRegex(RuntimeError, "undefined value x"):
            torch.jit.CompilationUnit('''
            def test_while(a, b):
                while bool(a < 10):
                    a = a + x
                    b = b + 1
                return a + b
            ''')

    def test_while_nonexistent_cond_value(self):
        with self.assertRaisesRegex(RuntimeError, "undefined value x"):
            torch.jit.CompilationUnit('''
            def test_while(a, b):
                while a < x:
                    a = a + 1
                    b = b + 1
                return a + b
            ''')

    def test_optional_refinement(self):
        @torch.jit.script
        def test_if_none_assignment(x):
            # type: (Optional[int]) -> int
            if x is None:
                x = 1
            return x + 1

        self.assertEqual(test_if_none_assignment(1), 2)

        @torch.jit.script
        def test_ternary(x):
            # type: (Optional[int]) -> int
            x = x if x is not None else 2
            return x

        @torch.jit.script
        def test_not_none(x):
            # type: (Optional[int]) -> None
            if x is not None:
                print(x + 1)

        @torch.jit.script
        def test_and(x, y):
            # type: (Optional[int], Optional[int]) -> None
            if x is not None and y is not None:
                print(x + y)

        @torch.jit.script
        def test_not(x, y):
            # type: (Optional[int], Optional[int]) -> None
            if not (x is not None and y is not None):
                pass
            else:
                print(x + y)

        @torch.jit.script
        def test_bool_expression(x):
            # type: (Optional[int]) -> None
            if x is not None and x < 2:
                print(x + 1)

        @torch.jit.script
        def test_nested_bool_expression(x, y):
            # type: (Optional[int], Optional[int]) -> int
            if x is not None and x < 2 and y is not None:
                x = x + y
            else:
                x = 5
            return x + 2

        @torch.jit.script
        def test_or(x, y):
            # type: (Optional[int], Optional[int]) -> None
            if y is None or x is None:
                pass
            else:
                print(x + y)

        # backwards compatibility
        @torch.jit.script
        def test_manual_unwrap_opt(x):
            # type: (Optional[int]) -> int
            if x is None:
                x = 1
            else:
                x = torch.jit._unwrap_optional(x)
            return x

        with self.assertRaisesRegex(RuntimeError, "arguments for call are not valid"):
            @torch.jit.script
            def or_error(x, y):
                # type: (Optional[int], Optional[int]) -> int
                if x is None or y is None:
                    print(x + y)

        with self.assertRaisesRegex(RuntimeError, "arguments for call are not valid"):
            @torch.jit.script
            def and_error(x, y):
                # type: (Optional[int], Optional[int]) -> int
                if x is None and y is None:
                    pass
                else:
                    print(x + y)

        with self.assertRaisesRegex(RuntimeError, "arguments for call are not valid"):
            @torch.jit.script
            def named_var(x):
                # type: (Optional[int]) -> None
                x_none = x is not None
                if x_none:
                    print(x + 1)

        with self.assertRaisesRegex(RuntimeError, "arguments for call are not valid"):
            @torch.jit.script
            def named_var_and(x, y):
                # type: (Optional[int], Optional[int]) -> None
                x_none = x is not None
                if y is not None and x_none:
                    print(x + y)

    def test_while_write_outer_then_read(self):
        def func(a, b):
            while bool(a < 10):
                a = a + 1
                b = a + 1
            return a + b

        inputs = self._make_scalar_vars([42, 1337], torch.int64)
        self.checkScript(func, inputs, optimize=True)

    def test_while_nest_if(self):
        def func(a, b):
            # type: (int, int) -> int
            c = 0
            while a < 10:
                a = a + 1
                b = b + 1
                if a > b:
                    c = -a
                else:
                    c = -b
            return c + 1

        inputs = self._make_scalar_vars([-1234, 4321], torch.int64)
        self.checkScript(func, inputs, optimize=True)

    def test_math_schema(self):
        # This should use the add(Tensor, Tensor) schema.
        # Also tests to see if alpha={1} is lifted correctly.
        def fn(x, y):
            return x + y

        graph = torch.jit.script(fn).graph
        self.assertExpectedGraph(graph)

    def test_math_tensor_number(self):
        # Test that 7 is casted to tensor, then casted to the
        # correct type, and finally added to x.
        def fn(x):
            return x + 7

        graph = torch.jit.script(fn).graph
        self.assertExpectedGraph(graph)

    def test_math_numbers(self):
        # Test that the numbers are casted to tensor,
        # added, and then casted back.
        def fn1(x):
            return 7 + 8

        def fn2(x):
            return 1.1 + 3.1

        graph1 = torch.jit.script(fn1).graph
        self.assertExpectedGraph(graph1, subname="int")
        graph2 = torch.jit.script(fn2).graph
        self.assertExpectedGraph(graph2, subname="float")

    def test_math_ops(self):

        def test_floor():
            return math.floor(1.5)

        self.checkScript(test_floor, ())

    def test_if_nest_while(self):
        def func(a, b):
            # type: (int, int) -> int
            c = 0
            if a > b:
                while a > b:
                    b = b + 1
                    c = -b
            return c

        inputs = self._make_scalar_vars([4321, 1234], torch.int64)
        self.checkScript(func, inputs, optimize=True)

    def test_script_for_in_range(self):
        def fn():
            c = 0
            for i in range(100):
                c += i
            return c
        self.checkScript(fn, (), outputs=4950, optimize=True)

    def test_script_for_in_range_dynamic(self):
        def fn():
            c = 0
            for i in range(100):
                acc = 0
                for j in range(i):
                    acc += j
                c += acc
            return c
        self.checkScript(fn, (), optimize=False)

    def test_script_for_in_range_ast(self):
        @torch.jit.script
        def test_script_for_in_range_ast():
            c = 0
            for i in range(100):
                acc = 0
                for j in range(i):
                    acc += j
                c += acc
            return c

        self.assertEqual(test_script_for_in_range_ast(), 161700)

    def test_script_for_in_range_if_ast(self):
        @torch.jit.script
        def test_script_for_in_range_if_ast(x):
            output = x
            for i in range(20):
                if i == 0:
                    output = x.unsqueeze(0)
                else:
                    output = torch.cat((output, x.unsqueeze(0)), dim=0)
            return output
        inputs = self._make_scalar_vars([0], torch.int64)

        self.assertEqual(test_script_for_in_range_if_ast(*inputs).shape[0], 20)

    def test_script_optional_none(self):
        def none_stmt(x):
            output = None
            output = x
            return output

        def none_args(x):
            # type: (Optional[Tensor]) -> Optional[Tensor]
            return None

        self.checkScript(none_stmt, [torch.arange(0, 2)], optimize=True)
        self.checkScript(none_args, [None], optimize=True)

        # test undefined tensor None as default param
        def test_script_optional_tensor_none(x=None):
            # type: (Optional[Tensor]) -> Tensor
            res = torch.zeros(1, dtype=torch.int8)
            if x is None:
                res = res + 1
            else:
                res = torch.jit._unwrap_optional(x)
            return res

        fn = test_script_optional_tensor_none
        scripted_fn = torch.jit.script(fn)
        self.assertEqual(fn(), scripted_fn())
        self.assertEqual(fn(torch.zeros(1)), scripted_fn(torch.zeros(1)))

        # test typical None as default param
        def test_script_optional_other_none(x=None):
            # type: (Optional[float]) -> float
            res = 2.0
            if x is None:
                res = res + 1.0
            else:
                res = torch.jit._unwrap_optional(x)
            return res

        fn = test_script_optional_other_none
        scripted_fn = torch.jit.script(fn)
        self.assertEqual(fn(), scripted_fn())
        self.assertEqual(fn(1.0), scripted_fn(1.0))

    def test_script_clamp_none(self):
        def test_script_clamp_max_none(x):
            return torch.clamp(x, min=2, max=None)

        def test_script_clamp_max(x):
            return torch.clamp(x, max=2)

        def test_script_clamp_min_none(x):
            return torch.clamp(x, min=None, max=2)

        def test_script_clamp_min(x):
            return torch.clamp(x, min=2)

        input = [torch.arange(0, 3)]
        self.checkScript(test_script_clamp_max_none, input, optimize=True)
        self.checkScript(test_script_clamp_max, input, optimize=True)
        self.checkScript(test_script_clamp_min_none, input, optimize=True)
        self.checkScript(test_script_clamp_min, input, optimize=True)

    def test_script_bool_constant(self):
        script = '''
        def test_script_bool_constant():
            a = True
            return a
        '''
        outputs = [1]
        self.checkScript(script, [], outputs[0], True, 'test_script_bool_constant')

    def test_ternary(self):
        def func(a, b):
            c = 3
            c = a + b if bool(a > 3) else b
            return c

        inputs_true = self._make_scalar_vars([5, 2], torch.int64)
        inputs_false = self._make_scalar_vars([1, 0], torch.int64)
        self.checkScript(func, inputs_true, optimize=True)
        self.checkScript(func, inputs_false, optimize=True)

    def test_print(self):
        def func(x, y):
            q = (x + y).sigmoid()
            print(q, 1, 2, [1, 2], [1.0, 2.0])
            w = -q
            return w * w

        x = torch.arange(4., requires_grad=True)
        y = torch.arange(0., 8, 2, requires_grad=True)
        self.checkScript(func, [x, y], optimize=True, capture_output=True)

    def test_format(self):
        def func(x):
            print("{}, I'm a {}".format("Hello", "test"))
            print("format blank".format())
            print("stuff before {}".format("hi"))
            print("{} stuff after".format("hi"))
            return x + 1

        x = torch.arange(4., requires_grad=True)
        self.checkScript(func, [x], optimize=True, capture_output=True)

    def test_logical_short_circuit(self):
        @torch.jit.script
        def testNoThrows(t):
            c1 = 1
            if (False and bool(t[1])) or (True or bool(t[1])):
                c1 = 0
            return c1

        @torch.jit.script
        def throwsOr(t):
            c0 = False or bool(t[1])
            print(c0)

        @torch.jit.script
        def throwsAnd(t):
            c0 = True and bool(t[1])
            print(c0)

        t = torch.randn(0)
        self.assertEqual(0, testNoThrows(torch.randn(0)))
        self.assertExpectedGraph(testNoThrows.graph)
        with self.assertRaisesRegex(RuntimeError, "index 1 out of range for tensor of size"):
            throwsOr(t)
        with self.assertRaisesRegex(RuntimeError, "index 1 out of range for tensor of size"):
            throwsAnd(t)

    def test_type_cast(self):
        template = dedent('''
        def cast(v):
            # type: ({from_type}) -> {to_type}
            return {to_type}(v)
        ''')

        def check_cast(from_type, to_type, value, raises=False):
            code = template.format(from_type=from_type, to_type=to_type)
            expected = getattr(builtins, to_type)(value)
            if raises:
                with self.assertRaisesRegex(RuntimeError, "Cannot cast"):
                    cu = torch.jit.CompilationUnit(code)
            else:
                self.checkScript(code, (value,), name='cast', outputs=expected)

        check_cast('int', 'float', 1)
        check_cast('int', 'bool', 1)
        check_cast('int', 'bool', 0)

        check_cast('float', 'int', 1.)
        check_cast('float', 'bool', 1.)
        check_cast('float', 'bool', 0.)

        check_cast('bool', 'int', True)
        check_cast('bool', 'float', True)

    def test_multiple_assignment(self):
        def outer_func(x):
            return x * 2, x + 2

        @torch.jit.script
        def func(x):
            y, z = outer_func(x)
            return y + z

        x = torch.arange(4)
        self.assertEqual(func(x), x * 2 + x + 2)

    def test_literals(self):
        def func(a):
            return a.view(size=[1, 2, 3])

        a = torch.randn(6)
        self.checkScript(func, [a], optimize=True)

    def test_return(self):
        def no_return(a):
            a + 1

        def void_return(a):
            return

        def one_return(a):
            return a + 1.

        def multiple_returns(a):
            return a * 1., a * 2., a * 3.

        a = torch.randn(1, dtype=torch.float)
        self.checkScript(no_return, [a], optimize=True)
        self.checkScript(void_return, [a], optimize=True)
        self.checkScript(one_return, [a], optimize=True)
        self.checkScript(multiple_returns, [a], optimize=True)

        with self.assertRaisesRegex(RuntimeError, "but is actually of type None"):
            @torch.jit.script
            def no_return_bad_annotation(a):
                # type: (Tensor) -> Tensor
                a + 1

    def test_error(self):
        @torch.jit.script
        def foo(a):
            return a.t()
        s = Variable(torch.rand(10))
        # XXX: this should stay quiet in stay propagation and only fail in the interpreter
        with self.assertRaisesRegex(RuntimeError, "failed in interpreter"):
            foo(s)

        @torch.jit.script
        def bar(c, b):
            return c + b

        with self.assertRaisesRegex(RuntimeError, "failed in interpreter"):
            bar(Variable(torch.rand(10), requires_grad=True), Variable(torch.rand(9), requires_grad=True))

    def test_binop_unsupported_error(self):
        with self.assertRaisesRegex(NotSupportedError, "unsupported binary operator:"):
            @torch.jit.script
            def binop(x, y):
                # Replace this with another unsupported op when/if it gets supported
                return x << y

    def test_bitwise_ops(self):

        def int_test():
            return 2 & 3, 2 ^ 3, 2 | 3

        self.checkScript(int_test, ())

        def bool_test(x, y):
            # type: (bool, bool) -> Tuple[bool, bool, bool]
            return x & y, x ^ y, x | y

        self.checkScript(bool_test, (True, False))
        self.checkScript(bool_test, (True, True))

        def tensor_test(x, y):
            return x & y, x ^ y, x | y

        x = torch.tensor(2)
        y = torch.tensor(3)

        self.checkScript(tensor_test, (x, y))

    def test_number_math(self):
        ops_template = dedent('''
        def func():
            return {scalar1} {op} {scalar2}
        ''')
        ops = ['+', '-', '*', '%', '<', '<=', '>', '>=', '==', '!=', '//']
        funcs_template = dedent('''
        def func():
            return {func}({scalar1}, {scalar2})
        ''')
        funcs = ['min', 'max']
        scalars = ['7', '2', '3', '-3', '3.14', '0.125', '-0.5', '2.0', '-2.0']
        scalar_pairs = [(scalar1, scalar2) for scalar1 in scalars for scalar2 in scalars]

        def run_test(code):
            scope = {}
            execWrapper(code, globals(), scope)
            cu = torch.jit.CompilationUnit(code)

            self.assertEqual(cu.func(), scope['func']())

        for scalar1, scalar2 in scalar_pairs:
            for op in ops:
                code = ops_template.format(op=op, scalar1=scalar1, scalar2=scalar2)
                run_test(code)
            for func in funcs:
                code = funcs_template.format(func=func, scalar1=scalar1, scalar2=scalar2)
                run_test(code)

    def test_number_div(self):
        self.checkScript(div_int_future, (), optimize=True)
        self.checkScript(div_float_future, (), optimize=True)

        if PY2:
            with self.assertRaisesRegex(RuntimeError, 'from __future__ import division'):
                torch.jit.script(div_int_nofuture)
            with self.assertRaisesRegex(RuntimeError, 'from __future__ import division'):
                torch.jit.script(div_float_nofuture)
        else:
            self.checkScript(div_int_nofuture, (), optimize=True)
            self.checkScript(div_float_nofuture, (), optimize=True)

    def test_floor_div(self):
        @torch.jit.script
        def foo(a, b):
            # type: (int, int) -> int
            return a // b
        for i in range(-8, 8):
            for j in range(-8, 8):
                if j != 0:
                    self.assertEqual(foo(i, j), i // j)
                else:
                    with self.assertRaisesRegex(RuntimeError, 'division by 0'):
                        foo(i, j)

    def test_number_augassign(self):
        def func():
            z = 1
            z += 2
            return z

        self.checkScript(func, (), optimize=True)

    def test_number_neg(self):
        # int -> int
        def func1():
            return -8

        # float -> float
        def func2():
            return -3.14

        self.checkScript(func1, (), optimize=True)
        self.checkScript(func2, (), optimize=True)

    def _test_tensor_number_math(self, device='cpu'):
        template = dedent('''
        def func(t):
            return {lhs} {op} {rhs}
        ''')

        def test(op, const, swap_args):
            args = ('t', const)
            if swap_args:
                args = (const, 't')

            code = template.format(lhs=args[0], rhs=args[1], op=op)
            scope = {}
            execWrapper(code, globals(), scope)
            cu = torch.jit.CompilationUnit(code)
            self.assertEqual(cu.func(tensor), scope['func'](tensor))

        var_int = [2, -2]
        var_float = [1.4321, -1.2]

        ops = ['+', '-', '*', '%', '<', '<=', '>', '>=', '==', '!=', '/']

        float_tensor = torch.randn(5, 5, device=device)
        double_tensor = torch.randn(5, 5, dtype=torch.double, device=device)
        long_tensor = torch.randint(-5, 5, (5, 5), dtype=torch.long, device=device)
        long_tensor[long_tensor == 0] = 2

        tensors = [float_tensor, double_tensor, long_tensor]
        consts = var_int + var_float

        for op, tensor, const, swap_args in product(ops, tensors, consts, [True, False]):
            # FIXME: things like 2 / long_tensor are not implemented correctly
            # Look in torch/tensor.py to see how pytorch implements it.
            if op == '/' and tensor.data_ptr() == long_tensor.data_ptr():
                continue

            # % operator does not take: const % tensor
            if op == '%' and swap_args is True:
                continue

            test(op, const, swap_args)

    def test_tensor_number_math(self):
        self._test_tensor_number_math()

    def test_torch_tensor_bad_input(self):
        with self.assertRaisesRegex(RuntimeError, "Input list to torch.tensor must be of ints, floats, "
                                    "or bools, got None"):
            @torch.jit.script
            def test():
                return torch.tensor([None])

        with self.assertRaisesRegex(RuntimeError, "Note: empty lists are constructed as Tensor"):
            @torch.jit.script
            def tmp():
                return torch.tensor([])

        @torch.jit.script
        def foo():
            return torch.tensor([[2, 2], [1]])
        with self.assertRaisesRegex(RuntimeError, "Expected sequence of length"):
            foo()

    @suppress_warnings
    def test_torch_tensor_empty_list(self):
        def func():
            return torch.tensor(torch.jit.annotate(List[int], []))
        cu = torch.jit.script(func)
        t1 = cu()
        t2 = func()

        # torchscript returns int tensor, python returns float tensor
        self.assertNotEqual(t1.dtype, t2.dtype)

        def func():
            li = torch.jit.annotate(List[int], [])
            return torch.tensor([li, li])

        self.checkScript(func, ())

        def func():
            li = torch.jit.annotate(List[int], [])
            return torch.tensor([[[li]]])

        self.checkScript(func, ())

    def test_torch_tensor(self):
        template = dedent('''
        def func():
            li = {list_create}
            return torch.tensor(li {options})
        ''')

        lists = ["2.5", "4", "True", "False", "[2]", "[-.5]", "[False, True, False]", "[2, 2]",
                 "torch.jit.annotate(List[int], [])", "[2.5, 2.5]", "[[2], [2]]", "[[-.5], [2.2]]", "[[False], [True]]"]

        dtypes = ["", ", dtype=torch.float", ", dtype=torch.double", ", dtype=torch.half",
                  ", dtype=torch.uint8", ", dtype=torch.int8", ", dtype=torch.short",
                  ", dtype=torch.int", ", dtype=torch.long"]

        devices = ['', ", device='cpu'"]
        if RUN_CUDA:
            devices.append(", device='cuda'")

        option_pairs = [dtype + device for dtype in dtypes for device in devices]
        for li in lists:
            for option in option_pairs:
                # tensor from empty list is type float in python and annotated type in torchscript
                if "annotate" in li and "dtype" not in option:
                    continue
                code = template.format(list_create=li, options=option)
                scope = {}
                exec(code, globals(), scope)
                cu = torch.jit.CompilationUnit(code)
                t1 = cu.func()
                t2 = scope['func']()
                if t1.dtype == torch.float16:  # equality NYI for half tensor
                    self.assertTrue(str(t1) == str(t2))
                else:
                    self.assertEqual(t1, t2)
                self.assertEqual(t1.dtype, t2.dtype)
                self.assertEqual(t1.device, t2.device)

    # adapted from test in test_torch
    def test_tensor_to(self):
        template = dedent('''
        def func(t):
            cuda = "{cuda}"
            device = "{device}"
            non_blocking = {non_blocking}
            return {to_str}
        ''')

        def s(t, to_str, non_blocking=None, device=None, cuda=None):
            device = device if device is not None else str(t.device)
            non_blocking = non_blocking if non_blocking is not None else False
            cuda = "cuda" if cuda is None else cuda
            code = template.format(to_str=to_str, device=device, non_blocking=non_blocking, cuda=cuda)
            scope = {}
            cu = torch.jit.CompilationUnit(code)
            return cu.func(t)

        def test_copy_behavior(t, non_blocking=False):
            self.assertIs(t, s(t, 't.to(t, non_blocking=non_blocking)', non_blocking))
            self.assertIs(t, s(t, 't.to(t.dtype, non_blocking=non_blocking)', non_blocking))
            self.assertIs(t, s(t, 't.to(torch.empty_like(t), non_blocking=non_blocking)', non_blocking))
            self.assertIsNot(t, s(t, 't.to(t, non_blocking=non_blocking, copy=True)', non_blocking))
            self.assertIsNot(t, s(t, 't.to(t.dtype, non_blocking=non_blocking, copy=True)', non_blocking))
            self.assertIsNot(t, s(t, 't.to(torch.empty_like(t), non_blocking=non_blocking, copy=True)', non_blocking))

            devices = [t.device]
            if t.device.type == 'cuda':
                if t.device.index == -1:
                    devices.append('cuda:{}'.format(torch.cuda.current_device()))
                elif t.device.index == torch.cuda.current_device():
                    devices.append('cuda')
            for device in devices:
                self.assertIs(t, s(t, 't.to(device, non_blocking=non_blocking)', non_blocking, device))
                self.assertIs(t, s(t, 't.to(device, t.dtype, non_blocking=non_blocking)', non_blocking, device))
                self.assertIsNot(t, s(t, 't.to(device, non_blocking=non_blocking, copy=True)', non_blocking, device))
                self.assertIsNot(t, s(t, 't.to(device, t.dtype, non_blocking=non_blocking, copy=True)',
                                      non_blocking, device))

        t = torch.tensor(5)
        test_copy_behavior(t)

        self.assertEqual(t.device, s(t, "t.to('cpu')").device)
        self.assertEqual(t.device, s(t, "t.to('cpu', dtype=torch.float32)").device)
        self.assertIs(torch.float32, s(t, "t.to('cpu', dtype=torch.float32)").dtype)
        self.assertEqual(t.device, s(t, "t.to(torch.float32)").device)
        self.assertIs(torch.float32, s(t, "t.to(dtype=torch.float32)").dtype)
        self.assertEqual(t.data_ptr(), s(t, "t.to('cpu')").data_ptr())
        self.assertEqual(t.data_ptr(), s(t, "t.to(dtype=t.dtype, device=t.device, copy=False)").data_ptr())
        self.assertEqual(t.data_ptr(), s(t, "t.to('cpu', copy=False)").data_ptr())
        self.assertNotEqual(t.data_ptr(), s(t, "t.to('cpu', copy=True)").data_ptr())

        a = torch.tensor(5)
        if torch.cuda.is_available():
            for non_blocking in [True, False]:
                for cuda in ['cuda', 'cuda:0' if torch.cuda.device_count() == 1 else 'cuda:1']:
                    b = torch.tensor(5., device=cuda)
                    test_copy_behavior(b, non_blocking)
                    self.assertEqual(b.device, s(b, "t.to(cuda, non_blocking=non_blocking).device", cuda=cuda))
                    self.assertEqual(a.device, s(b, "t.to('cpu', non_blocking=non_blocking).device"))
                    self.assertEqual(b.device, s(b, "t.to(cuda, non_blocking=non_blocking).device", cuda=cuda))
                    self.assertIs(torch.int32, s(b, "t.to('cpu', dtype=torch.int32, non_blocking=non_blocking)").dtype)
                    self.assertEqual(a.device, s(b, "t.to('cpu', dtype=torch.int32, non_blocking=non_blocking)").device)
                    self.assertIs(torch.int32, s(b, "t.to(dtype=torch.int32)").dtype)
                    self.assertEqual(b.device, s(b, "t.to(dtype=torch.int32)").device)

    @unittest.skipIf(not RUN_CUDA, "No CUDA")
    def test_tensor_number_math_cuda(self):
        self._test_tensor_number_math(device='cuda')

    def test_not(self):
        # test not operator in python
        # TODO: add more tests when bool conversions ready
        def test_not_op(a):
            return not bool(a > 1)

        self.checkScript(test_not_op, (torch.tensor(2), ), optimize=True)

    def test_is_isnot(self):
        # test is and is not operator in python
        template = dedent('''
        def func():
            # type: () -> bool
            return {lhs} {op} {rhs}
        ''')

        def test(op, args):
            code = template.format(lhs=args[0], rhs=args[1], op=op)
            scope = {}
            execWrapper(code, globals(), scope)
            cu = torch.jit.CompilationUnit(code)
            self.assertEqual(
                cu.func(),
                scope['func'](),
                "Failed with op: {}, lhs: {}, rhs: {}"
                .format(op, args[0], args[1])
            )

        ops = ['is', 'is not']
        type_literals = [True, False, None, [1, 1]]

        # do literals product to try any types combinations
        for op, lhs, rhs in product(ops, type_literals, type_literals):
            test(op, [lhs, rhs])

    def test_isinstance(self):
        # test isinstance operator for static type checking
        template = dedent('''
        def func(x):
            # type: ({type_hint}) -> bool
            return isinstance(x, {typ})
        ''')

        def test(inp, typ, type_hint):
            code = template.format(typ=typ, type_hint=type_hint)
            scope = {}
            execWrapper(code, globals(), scope)
            cu = torch.jit.CompilationUnit(code)
            self.assertEqual(
                cu.func(inp),
                scope['func'](inp),
                "Failed with typ: {}"
                .format(typ)
            )

        inputs = [True, 1, 1.0, torch.tensor(1), [1, 2], (1.0,), [1, 2], 1]
        type_literals = ['bool', 'int', 'float', 'torch.Tensor', 'list', 'tuple',
                         '(list, tuple)', '(int, float, bool)']
        type_annotations = ['bool', 'int', 'float', 'Tensor', 'List[int]', 'Tuple[float]',
                            'List[int]', 'int']

        # do zipping to try different types
        for inp, typ, type_hint in zip(inputs, type_literals, type_annotations):
            test(inp, typ, type_hint)

        # test optional isintance check
        with self.assertRaisesRegex(RuntimeError, "Optional isinstance check is not supported"):
            @torch.jit.script
            def opt_func(x):
                # type: (Optional[int]) -> bool
                return isinstance(x, int)

    def test_python_call(self):
        def pyfunc(a):
            return a * 3.0

        cu = torch.jit.CompilationUnit('''
        def other_func(a):
            return a + a

        def test_call_python(a):
            b = pyfunc(a)
            b = other_func(b)
            i = 0
            step = 1
            while i < 10:
                b = pyfunc(b)
                if bool(b > 3.0):
                    b = pyfunc(b)
                i = 11
            return b
        ''')
        inputs = self._make_scalar_vars([1], torch.float)
        outputs = self._make_scalar_vars([54], torch.float)

        self.assertEqual(cu.test_call_python(*inputs), outputs[0])

    def test_python_call_failure(self):
        with self.assertRaisesRegex(RuntimeError, "undefined value pyfunc2"):
            def pyfunc(a):
                return a * 3.0

            cu = torch.jit.CompilationUnit('''
            def other_func(a):
                return a + a

            def test_call_python(a):
                b = pyfunc(a)
                b = other_func(b)
                i = 0
                step = 1
                while i < 10:
                    b = pyfunc2(b)
                    if b > 3.0:
                        b = pyfunc(b)
                    i = 11
                return b
            ''')
            inputs = self._make_scalar_vars([1], torch.float)
            outputs = self._make_scalar_vars([54], torch.float)

            self.assertEqual(cu.test_call_python(*inputs), outputs)

    def test_python_call_annotation(self):
        def pyfunc(a):
            return a * 3.0

        @torch.jit.script
        def foo(a):
            return pyfunc(a) + pyfunc(a)

        inputs = self._make_scalar_vars([1], torch.float)
        outputs = self._make_scalar_vars([6], torch.float)
        self.assertEqual(foo(*inputs), outputs[0])

    def test_python_call_annoytation_failure(self):
        with self.assertRaisesRegex(RuntimeError, "undefined value pyfunc2"):
            def pyfunc(a):
                return a * 3.0

            @torch.jit.script
            def foo(a):
                return pyfunc2(a) + pyfunc(a)

            inputs = self._make_scalar_vars([1], torch.float)
            outputs = self._make_scalar_vars([6], torch.float)

            self.assertEqual(foo(*inputs), outputs[0])

    def test_desugar_module(self):
        import torch.nn.functional as F

        def fn(x, slope):
            a = torch.abs(x)
            b = torch.nn.functional.prelu(x, slope)
            c = F.prelu(x, slope)
            return a, b, c

        x = torch.arange(-3., 4)
        slope = torch.tensor([0.5])
        self.checkScript(fn, [x, slope], optimize=True)

    def test_script_docstring(self):
        @torch.jit.script
        def with_docstring(x):
            """test str"""
            y = x
            """y is the same as x"""
            return y
        self.assertEqual(with_docstring.__doc__, 'test str')

    def test_script_method_docstring(self):
        class A(torch.jit.ScriptModule):
            @torch.jit.script_method
            def with_docstring(self, x):
                """test str"""
                y = x
                """y is the same as x"""
                return y
        a = A()
        self.assertEqual(a.with_docstring.__doc__, 'test str')

    @unittest.skipIf(TEST_WITH_UBSAN or not torch.fbgemm_is_cpu_supported(),
                     'Quantized RNN requires FBGEMM. FBGEMM does not play'
                     ' well with UBSAN at the moment, so we skip the test if'
                     ' we are in a UBSAN environment.')
    def test_rnn_cell_quantized(self):
        d_in, d_hid = 2, 2

        for cell in [
            torch.nn.LSTMCell(d_in, d_hid).float(),
            torch.nn.GRUCell(d_in, d_hid).float(),
            torch.nn.RNNCell(d_in, d_hid).float(),
        ]:
            if isinstance(cell, torch.nn.LSTMCell):
                num_chunks = 4
            elif isinstance(cell, torch.nn.GRUCell):
                num_chunks = 3
            elif isinstance(cell, torch.nn.RNNCell):
                num_chunks = 1

            # Replace parameter values s.t. the range of values is exactly
            # 255, thus we will have 0 quantization error in the quantized
            # GEMM call. This i s for testing purposes.
            #
            # Note that the current implementation does not support
            # accumulation values outside of the range representable by a
            # 16 bit integer, instead resulting in a saturated value. We
            # must take care that in our test we do not end up with a dot
            # product that overflows the int16 range, e.g.
            # (255*127+255*127) = 64770. So, we hardcode the test values
            # here and ensure a mix of signedness.
            vals = [[100, -155],
                    [100, -155],
                    [-155, 100],
                    [-155, 100],
                    [100, -155],
                    [-155, 100],
                    [-155, 100],
                    [100, -155]]
            vals = vals[:d_hid * num_chunks]
            cell.weight_ih = torch.nn.Parameter(
                torch.tensor(vals, dtype=torch.float),
                requires_grad=False)
            cell.weight_hh = torch.nn.Parameter(
                torch.tensor(vals, dtype=torch.float),
                requires_grad=False)

            ref = copy.deepcopy(cell)

            cell = torch.jit.quantized.quantize_rnn_cell_modules(cell)
            x = torch.tensor([[100, -155],
                              [-155, 100],
                              [100, -155]], dtype=torch.float)
            h0_vals = [[-155, 100],
                       [-155, 155],
                       [100, -155]]
            hx = torch.tensor(h0_vals, dtype=torch.float)
            if isinstance(cell, torch.jit.quantized.QuantizedLSTMCell):
                cx = torch.tensor(h0_vals, dtype=torch.float)
                hiddens = (hx, cx)
            else:
                hiddens = hx

            if isinstance(cell, torch.jit.quantized.QuantizedLSTMCell):
                from typing import Tuple

                class ScriptWrapper(torch.jit.ScriptModule):
                    def __init__(self, cell):
                        super(ScriptWrapper, self).__init__()
                        self.cell = cell

                    @torch.jit.script_method
                    def forward(self, x, hiddens):
                        # type: (torch.Tensor, Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]
                        return self.cell(x, hiddens)
            else:

                class ScriptWrapper(torch.jit.ScriptModule):
                    def __init__(self, cell):
                        super(ScriptWrapper, self).__init__()
                        self.cell = cell

                    @torch.jit.script_method
                    def forward(self, x, hiddens):
                        # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
                        return self.cell(x, hiddens)

            cell = ScriptWrapper(cell)
            outs = cell(x, hiddens)
            cell = self.getExportImportCopyWithPacking(cell)

            outs = cell(x, hiddens)
            ref_outs = ref(x, hiddens)

            self.assertEqual(len(outs), len(ref_outs))
            for out, ref_out in zip(outs, ref_outs):
                torch.testing.assert_allclose(out, ref_out)

    def test_script_module(self):
        class M1(torch.jit.ScriptModule):
            def __init__(self):
                super(M1, self).__init__(False)
                self.weight = nn.Parameter(torch.randn(2))

            @torch.jit.script_method
            def forward(self, thing):
                return self.weight + thing

        class PModule(nn.Module):
            def __init__(self):
                super(PModule, self).__init__()
                self.a = nn.Parameter(torch.randn(2, 3))

            def forward(self, a):
                return self.a.mm(a)

        class M2(torch.jit.ScriptModule):
            def __init__(self):
                super(M2, self).__init__(False)
                # test submodule
                self.sub = M1()
                self.sub2 = PModule()
                # test parameters
                self.weight = nn.Parameter(torch.randn(2, 3))
                self.bias = nn.Parameter(torch.randn(2))
                # test defining a method from a string
                self.define("""
                    def hi(self, a):
                        return self.weight.mm(a)
                """)
            # test script methods

            @torch.jit.script_method
            def doit(self, input):
                # test use of parameter
                return self.weight.mm(input)

            @torch.jit.script_method
            def doit2(self, input):
                return self.weight.mm(input)

            @torch.jit.script_method
            def forward(self, input):
                a = self.doit(input)
                b = self.doit2(input)
                c = self.hi(input)
                d = self.sub2(input)
                return a + b + self.bias + self.sub(a) + c + d
        m2 = M2()
        input = torch.randn(3, 2)
        a = m2.weight.mm(input)
        b = m2.weight.mm(input)
        c = m2.weight.mm(input)
        d = m2.sub2.a.mm(input)
        ref = a + b + m2.bias + m2.sub.weight + a + c + d
        self.assertEqual(ref, m2.forward(input))
        m2.weight = nn.Parameter(torch.zeros_like(m2.weight))
        m2.bias = nn.Parameter(torch.zeros_like(m2.bias))
        m2.sub.weight = nn.Parameter(torch.zeros_like(m2.sub.weight))
        m2.sub2.a.data.zero_()
        self.assertEqual(torch.zeros(2, 2), m2.forward(torch.randn(3, 2)))

    def test_script_module_call_noscript(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super(M, self).__init__(False)
                self.value = 1

            def foo(self):
                return torch.ones(2, 2) + self.value

            @torch.jit.script_method
            def forward(self, input):
                return input + self.foo()

        m = M()
        input = torch.randn(2, 2)
        o = m(input)
        self.assertEqual(o, input + torch.ones(2, 2) + 1)
        # check that we can change python attributes
        # and that those changes are picked up in script methods
        m.value = 2
        o = m(input)
        self.assertEqual(o, input + torch.ones(2, 2) + 2)

    def test_script_module_nochange_submodule(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super(M, self).__init__(False)
                self.sub = nn.Linear(5, 5)

            @torch.jit.script_method
            def forward(self, input):
                return self.sub(input)

        m = M()
        input = torch.randn(1, 5, 5)
        o = m(input)
        self.assertEqual(o, m.sub(input))
        with self.assertRaisesRegex(RuntimeError, "cannot re-assign"):
            m.sub = nn.Linear(5, 5)

    def test_script_inline_trace_multiple_args(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super(M, self).__init__(False)

            def forward(self, input, input2):
                return input + input2

        class M2(torch.jit.ScriptModule):
            def __init__(self):
                super(M2, self).__init__(False)
                self.m = torch.jit.trace(M(), (torch.zeros(4, 3), torch.zeros(4, 3)))

            @torch.jit.script_method
            def forward(self, inp):
                return self.m(inp, inp)

        m2 = M2()
        m2(torch.zeros(4, 3))

    def test_script_module_const(self):
        class M(torch.jit.ScriptModule):

            __constants__ = ['b', 'i', 'c']

            def __init__(self):
                super(M, self).__init__(False)
                self.b = False
                self.i = 1
                self.c = 3.5

            @torch.jit.script_method
            def forward(self):
                return self.b, self.i, self.c

        m = M()
        o0, o1, o2 = m()
        self.assertEqual(o0, 0)
        self.assertEqual(o1, 1)
        self.assertEqual(o2, 3.5)

    def test_script_module_fail_const(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super(M, self).__init__(False)
                self.b = False

            @torch.jit.script_method
            def forward(self):
                return self.b
        with self.assertRaisesRegex(RuntimeError, "is not usable in a script method"):
            M()

    def test_script_module_valid_consts(self):
        tester = self

        class Foo(torch.jit.ScriptModule):
            __constants__ = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

            def __init__(self):
                super(Foo, self).__init__(False)
                self.a = 1
                self.b = 1.2
                self.c = False
                with tester.assertRaisesRegex(
                        TypeError,
                        "'Linear' object for attribute 'd' is not a valid constant"):
                    self.d = [nn.Linear(3, 4)]
                self.e = lambda x: x
                self.f = [3, 4, 5]
                tester.assertTrue(type(self.f) is tuple)
                self.g = [3, (3, 4), 5]
                with tester.assertRaisesRegex(TypeError, "not a valid constant"):
                    self.h = type(1)
                with tester.assertRaisesRegex(TypeError, "not a valid constant"):
                    self.i = (3, 4, {})

        f = Foo()

    def test_script_module_param_buffer_mutation(self):
        # TODO: add param mutation test case after JIT support it
        class ModuleBufferMutate(torch.jit.ScriptModule):
            def __init__(self):
                super(ModuleBufferMutate, self).__init__(False)
                self.register_buffer('running_var', torch.tensor(0, dtype=torch.long))

            @torch.jit.script_method
            def forward(self):
                if self.training:
                    self.running_var += 1
                return self.running_var

        m = ModuleBufferMutate()
        self.assertEqual(m(), 1)
        m.eval()
        self.assertEqual(m(), 1)

    def test_script_module_for(self):
        class M(torch.jit.ScriptModule):
            __constants__ = ['b']

            def __init__(self):
                super(M, self).__init__(False)
                self.b = [1, 2, 3, 4]

            @torch.jit.script_method
            def forward(self):
                sum = 0
                for i in self.b:
                    sum += i
                return sum

        m = M()
        self.assertEqual(m(), 10)

    def test_script_module_for2(self):
        class Sub(torch.jit.ScriptModule):
            def __init__(self):
                super(Sub, self).__init__(False)
                self.weight = nn.Parameter(torch.randn(2))

            @torch.jit.script_method
            def forward(self, thing):
                return self.weight + thing

        class M(torch.jit.ScriptModule):
            __constants__ = ['mods']

            def __init__(self):
                super(M, self).__init__(False)
                self.mods = nn.ModuleList([Sub() for i in range(10)])

            @torch.jit.script_method
            def forward(self, v):
                for m in self.mods:
                    v = m(v)
                return v

        i = torch.Tensor(2)
        m = M()
        o = m(i)
        v = i
        for sub in m.mods:
            v = sub(v)
        self.assertEqual(o, v)

    def test_script_module_const_submodule_fail(self):
        class Sub(torch.jit.ScriptModule):
            def __init__(self):
                super(Sub, self).__init__(False)
                self.weight = nn.Parameter(torch.randn(2))

            @torch.jit.script_method
            def forward(self, thing):
                return self.weight + thing

        class M(torch.jit.ScriptModule):
            def __init__(self):
                super(M, self).__init__(False)
                self.mods = [Sub() for _ in range(10)]

            @torch.jit.script_method
            def forward(self):
                for _ in self.mods:
                    print(1)
                return 4

        with self.assertRaisesRegex(RuntimeError, "did you forget to add it __constants__"):
            M()

        # Specialized error for Tensors
        class S(torch.jit.ScriptModule):
            def __init__(self):
                self.tensor_constant = torch.ones(2)

            @torch.jit.script_method
            def forward(self):
                return self.tensor_constant + 2

        with self.assertRaisesRegex(RuntimeError, "Tensors must be added to a module as a buffer or parameter"):
            S()

    class DerivedStateModule(torch.jit.ScriptModule):
        def __init__(self):
            super(TestScript.DerivedStateModule, self).__init__()
            self.param = torch.nn.Parameter(torch.ones(3, 4, dtype=torch.float))
            self.register_buffer('derived', torch.neg(self.param).detach().clone())

            # This is a flag so we can test that the pack method was called
            self.register_buffer('pack_called', torch.zeros(1, dtype=torch.long))
            # This is a flag so we can test that the unpack method was called
            self.register_buffer('unpack_called', torch.zeros(1, dtype=torch.long))

        @torch.jit.script_method
        def _pack(self):
            self.pack_called.set_(torch.ones(1, dtype=torch.long))
            self.derived.set_(torch.rand(1, dtype=torch.float).detach())

        @torch.jit.script_method
        def _unpack(self):
            self.unpack_called.set_(torch.ones(1, dtype=torch.long))
            self.derived.set_(torch.neg(self.param).detach())

        @torch.jit.script_method
        def forward(self, x):
            return x + self.derived

    def test_pack_unpack_state(self):
        sm = TestScript.DerivedStateModule()
        x = torch.rand(3, 4, dtype=torch.float)
        torch.testing.assert_allclose(sm(x), x + torch.neg(torch.ones(3, 4, dtype=torch.float)))

        # Test save path
        self.assertFalse(sm.pack_called.item())
        self.assertFalse(sm.unpack_called.item())
        imported = self.getExportImportCopyWithPacking(sm)
        # ensure pack was called before serialization
        self.assertTrue(sm.pack_called.item())
        # ensure unpack was called after serialization so as to leave the module in an initialized state
        self.assertTrue(sm.unpack_called.item())

        torch.testing.assert_allclose(sm.derived, torch.neg(sm.param))

        # Test load paths
        self.assertTrue(imported.unpack_called.item())
        torch.testing.assert_allclose(imported(x), x + torch.neg(torch.ones(3, 4, dtype=torch.float)))

    def test_pack_unpack_nested(self):
        class SubSubMod(torch.jit.ScriptModule):
            def __init__(self):
                super(SubSubMod, self).__init__()
                self.register_buffer('buf', torch.ones(3, 4) * 3)

            @torch.jit.script_method
            def _pack(self):
                self.buf.set_(torch.zeros(1, dtype=torch.double))

            @torch.jit.script_method
            def _unpack(self):
                self.buf.set_(torch.ones(3, 4, dtype=torch.double) * 3)

            @torch.jit.script_method
            def forward(self, x):
                return x + self.buf

        class SubMod(torch.jit.ScriptModule):
            def __init__(self):
                super(SubMod, self).__init__()
                self.register_buffer('buf', torch.ones(3, 4) * 2)
                self.ssm = SubSubMod()

            @torch.jit.script_method
            def _pack(self):
                self.buf.set_(torch.zeros(1, dtype=torch.double))

            @torch.jit.script_method
            def _unpack(self):
                self.buf.set_(torch.ones(3, 4, dtype=torch.double) * 2)

            @torch.jit.script_method
            def forward(self, x):
                return self.ssm(x + self.buf)

        class Mod(torch.jit.ScriptModule):
            def __init__(self):
                super(Mod, self).__init__()
                self.submod = SubMod()
                self.register_buffer('buf', torch.ones(3, 4) * 1)

            @torch.jit.script_method
            def _pack(self):
                self.buf.set_(torch.zeros(1, dtype=torch.double))

            @torch.jit.script_method
            def _unpack(self):
                self.buf.set_(torch.ones(3, 4, dtype=torch.double))

            @torch.jit.script_method
            def forward(self, x):
                return self.submod(x + self.buf)

        m = Mod()
        torch.testing.assert_allclose(m(torch.zeros(3, 4)), torch.ones(3, 4) * 6)
        m.apply(lambda s: s._pack())
        torch.testing.assert_allclose(m(torch.zeros(3, 4)), torch.zeros(3, 4))
        m.apply(lambda s: s._unpack())
        torch.testing.assert_allclose(m(torch.zeros(3, 4)), torch.ones(3, 4) * 6)

    def test_script_module_not_tuple(self):
        class M(torch.jit.ScriptModule):
            __constants__ = ['mods']

            def __init__(self):
                super(M, self).__init__(False)
                self.mods = 1

            @torch.jit.script_method
            def forward(self, v):
                for m in self.mods:
                    print(m)
                return v
        with self.assertRaisesRegex(RuntimeError, "cannot be used as a tuple"):
            M()

    def test_script_sequential_for(self):
        class Sub(torch.jit.ScriptModule):
            def __init__(self):
                super(Sub, self).__init__(False)
                self.weight = nn.Parameter(torch.randn(2))

            @torch.jit.script_method
            def forward(self, thing):
                return self.weight + thing

        class M(torch.jit.ScriptModule):
            __constants__ = ['mods']

            def __init__(self):
                super(M, self).__init__(False)
                self.mods = nn.Sequential(Sub(), Sub(), Sub())

            @torch.jit.script_method
            def forward(self, v):
                for m in self.mods:
                    v = m(v)
                return v

            @torch.jit.script_method
            def forward2(self, v):
                return self.mods(v)

        i = torch.Tensor(2)
        m = M()
        o = m(i)
        v = i
        for sub in m.mods:
            v = sub(v)
        self.assertEqual(o, v)

        o2 = m.forward2(i)
        self.assertEqual(o2, v)

    def test_script_sequential_multi_output_fail(self):
        class Sub(torch.jit.ScriptModule):
            def __init__(self):
                super(Sub, self).__init__(False)
                self.weight = nn.Parameter(torch.randn(2))

            @torch.jit.script_method
            def forward(self, thing):
                return self.weight + thing

        class ReturnMulti(torch.jit.ScriptModule):
            def __init__(self):
                super(ReturnMulti, self).__init__(False)

            @torch.jit.script_method
            def forward(self, x):
                return x, x, x

        class HaveSequential(torch.jit.ScriptModule):
            __constants__ = ['someseq']

            def __init__(self):
                super(HaveSequential, self).__init__(False)
                self.someseq = nn.Sequential(
                    Sub(),
                    ReturnMulti(),
                    Sub()
                )

            @torch.jit.script_method
            def forward(self, x):
                return self.someseq(x)

        with self.assertRaisesRegex(RuntimeError, "(Tensor, Tensor, Tensor)"):
            hs = HaveSequential()
            i = torch.Tensor(2)
            hs(i)

    def test_script_sequential_in_mod_list(self):
        class Sub(torch.jit.ScriptModule):
            def __init__(self):
                super(Sub, self).__init__(False)
                self.weight = nn.Parameter(torch.randn(2))

            @torch.jit.script_method
            def forward(self, thing):
                return self.weight + thing

        class M(torch.jit.ScriptModule):
            __constants__ = ['mods']

            def __init__(self):
                super(M, self).__init__(False)
                self.mods = nn.ModuleList([Sub(), nn.Sequential(Sub(), nn.Sequential(Sub(), Sub()), Sub())])

            @torch.jit.script_method
            def forward(self, v):
                for mod in self.mods:
                    v = mod(v)
                return v

        m = M()
        graph = str(m.graph)
        print(graph)
        return
        self.assertTrue(graph.count("aten::add") == 4)
        self.assertTrue("python" not in graph)

    def test_script_nested_mod_list(self):
        class Sub(torch.jit.ScriptModule):
            def __init__(self):
                super(Sub, self).__init__(False)
                self.weight = nn.Parameter(torch.randn(2))

            @torch.jit.script_method
            def forward(self, thing):
                return self.weight + thing

        class M(torch.jit.ScriptModule):
            __constants__ = ['mods']

            def __init__(self):
                super(M, self).__init__(False)
                self.mods = nn.ModuleList([nn.ModuleList([Sub()]), nn.Sequential(Sub()), nn.ModuleList([Sub(), Sub()])])

            @torch.jit.script_method
            def forward(self, v):
                for mod in self.mods:
                    for m in mod:
                        v = m(v)
                return v

        m = M()
        graph = str(m.graph)
        self.assertTrue(graph.count("aten::add") == 4)
        self.assertTrue("python" not in graph)

    def test_constant_as_attr(self):
        class M(torch.jit.ScriptModule):
            __constants__ = ['dim']

            def __init__(self):
                super(M, self).__init__(False)
                self.dim = 1

            @torch.jit.script_method
            def forward(self, v):
                return torch.cat([v, v, v], dim=self.dim)
        v = torch.zeros(1, 1)
        self.assertEqual(torch.cat([v, v, v], dim=1), M()(v))

    class StarTestSumStarred(torch.nn.Module):
        def __init__(self):
            super(TestScript.StarTestSumStarred, self).__init__()

        def forward(self, *inputs):
            output = inputs[0]
            for i in range(1, len(inputs)):
                output += inputs[i]
            return output

    class StarTestReturnThree(torch.nn.Module):
        def __init__(self):
            super(TestScript.StarTestReturnThree, self).__init__()

        def forward(self, rep):
            return rep, rep, rep

    def test_script_star_expr(self):

        class M2(torch.jit.ScriptModule):
            def __init__(self):
                super(M2, self).__init__(True)
                self.m = torch.jit.trace(TestScript.StarTestSumStarred(),
                                         (torch.ones(4, 3), torch.ones(4, 3), torch.ones(4, 3)))
                self.g = torch.jit.trace(TestScript.StarTestReturnThree(), torch.ones(4, 3))

            @torch.jit.script_method
            def forward(self, rep):
                tup = self.g(rep)
                return self.m(*tup)

        m = M2()
        self.assertEqual(m(torch.zeros(4, 3)), 3 * torch.zeros(4, 3))

    def test_script_star_expr_string(self):
        class M2(torch.jit.ScriptModule):
            def __init__(self):
                super(M2, self).__init__(True)
                self.m = torch.jit.trace(TestScript.StarTestSumStarred(),
                                         (torch.ones(4, 3), torch.ones(4, 3), torch.ones(4, 3)))
                self.g = torch.jit.trace(TestScript.StarTestReturnThree(), torch.ones(4, 3))

                self.define('''
            def forward(self, rep):
                tup = self.g(rep)
                return self.m(*tup)
                ''')

        m = M2()
        self.assertEqual(m(torch.zeros(4, 3)), 3 * torch.zeros(4, 3))

    class StarTestSumAndReturnThree(torch.nn.Module):
        def __init__(self):
            super(TestScript.StarTestSumAndReturnThree, self).__init__()

        def forward(self, *inputs):
            output = inputs[0]
            for i in range(1, len(inputs)):
                output += inputs[i]
            return output, output, output

    def test_script_star_assign(self):
        class M2(torch.jit.ScriptModule):
            def __init__(self):
                super(M2, self).__init__(True)
                self.g = torch.jit.trace(TestScript.StarTestSumAndReturnThree(), torch.ones(4, 3))
                self.define('''
            def forward(self, rep):
                head, *tail = self.g(rep)
                return head
                ''')

        m = M2()
        self.assertEqual(m(torch.zeros(4, 3)), 3 * torch.zeros(4, 3))

    def test_script_module_star_assign2(self):
        class M2(torch.jit.ScriptModule):
            def __init__(self):
                super(M2, self).__init__(True)
                self.g = torch.jit.trace(
                    TestScript.StarTestSumAndReturnThree(),
                    (torch.ones(4, 3), torch.ones(4, 3), torch.ones(4, 3)),
                    _force_outplace=True)
                self.define('''
            def forward(self, rep):
                *head, tail = self.g(rep, rep, rep)
                return tail
                ''')

        m = M2()
        self.assertEqual(m(torch.ones(4, 3)), 3 * torch.ones(4, 3))

    def test_script_module_star_assign2_inplace(self):
        class M2(torch.jit.ScriptModule):
            def __init__(self):
                super(M2, self).__init__(True)
                self.g = torch.jit.trace(
                    TestScript.StarTestSumAndReturnThree(),
                    (torch.ones(4, 3), torch.ones(4, 3), torch.ones(4, 3)),
                    _force_outplace=False)
                self.define('''
            def forward(self, rep):
                *head, tail = self.g(rep, rep, rep)
                return tail
                ''')

        m = M2()
        # since forward() makes three aliases to the input `rep` before passing
        # it to StarTestSumAndReturnThree(), in-place behavior will be different
        # than the above out of place.
        self.assertEqual(m(torch.ones(4, 3)), 4 * torch.ones(4, 3))

    def test_script_module_star_assign_fail_pythonop(self):

        with self.assertRaisesRegex(RuntimeError, "cannot be used as a tuple"):
            class M2(torch.jit.ScriptModule):
                def __init__(self):
                    super(M2, self).__init__(True)

                    def myfunc():
                        return torch.zeros(1, 2, 3), torch.zeros(1, 2, 3)

                    self.define('''
                def forward(self, rep):
                    a, *b = myfunc()
                    return a
                    ''')

            m = M2()
            m(torch.zeros(4, 3))

    def test_script_module_star_assign_fail_builtin(self):
        with self.assertRaisesRegex(RuntimeError, "cannot be used as a tuple"):
            class M2(torch.jit.ScriptModule):
                def __init__(self):
                    super(M2, self).__init__(True)

                    self.define('''
                def forward(self, rep):
                    a, *b = torch.neg(rep)
                    return a
                    ''')

            m = M2()
            m(torch.zeros(4, 3))

    def test_pack_padded_pad_packed_trace(self):
        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
        T, B, C = 3, 5, 7

        class PadPackedWrapper(torch.nn.Module):
            def __init__(self):
                super(PadPackedWrapper, self).__init__()

            def forward(self, x, seq_lens):
                x = pack_padded_sequence(x, seq_lens)
                x, _ = pad_packed_sequence(x)
                return x

        x = np.ones((T, B, C))
        seq_lens = np.array([3, 3, 2, 2, 1], dtype=np.int32)
        # set padding value so we can test equivalence
        for b in range(B):
            if seq_lens[b] < T:
                x[seq_lens[b]:, b, :] = 0
        seq_lens = torch.from_numpy(seq_lens)
        x = torch.autograd.Variable(torch.from_numpy(x), requires_grad=True)

        m = PadPackedWrapper()
        m_traced = torch.jit.trace(m, (x, seq_lens,))

        y = m(x, seq_lens)
        loss = torch.sum(y)
        loss.backward()
        grad = x.grad.clone()
        x.grad.zero_()

        y_traced = m_traced(x, seq_lens)
        loss_traced = torch.sum(y_traced)
        loss_traced.backward()
        grad_traced = x.grad.clone()

        self.assertEqual(y_traced, x)
        self.assertEqual(y_traced, y)
        self.assertEqual(grad, grad_traced)

        f = io.BytesIO()
        torch.onnx._export(m, (x, seq_lens), f, verbose=False)

    def test_script_outputs(self):
        with self.assertRaisesRegex(RuntimeError, "cannot be used as a tuple"):
            @torch.jit.script
            def foo(a):
                c, d = a + a
                return c + d

        @torch.jit.script
        def return3():
            return 1, 2, 3

        with self.assertRaisesRegex(RuntimeError, "too many values to unpack"):
            @torch.jit.script
            def bind2():
                a, b = return3()
                print(a)
                print(b)

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    def test_script_get_device_cuda(self):
        @torch.jit.script
        def foo(a):
            return a.get_device()

        v = torch.randn(1, device='cuda')
        self.assertEqual(foo(v), 0)

    def test_script_chunk(self):
        @torch.jit.script
        def foo(a):
            b, c = torch.chunk(a, dim=0, chunks=2)
            return b
        v = torch.rand(10, 3)
        self.assertEqual(torch.chunk(v, dim=0, chunks=2)[0], foo(v))

    def test_rnn_trace_override(self):
        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
        num_layers = 3
        T, B, C = 11, 5, 7

        class RNNTraceWrapper(torch.nn.Module):
            def __init__(self, cell_type):
                super(RNNTraceWrapper, self).__init__()
                if cell_type == 'RNN':
                    self.rnn = torch.nn.RNN(input_size=C, hidden_size=C, num_layers=num_layers)
                elif cell_type == 'LSTM':
                    self.rnn = torch.nn.LSTM(input_size=C, hidden_size=C, num_layers=num_layers)
                elif cell_type == 'GRU':
                    self.rnn = torch.nn.GRU(input_size=C, hidden_size=C, num_layers=num_layers)

            def forward(self, x, seq_lens):
                x = pack_padded_sequence(x, seq_lens)
                x, _ = self.rnn(x)
                x, _ = pad_packed_sequence(x)
                return x

        for cell_type in ['RNN', 'LSTM', 'GRU']:
            x = torch.ones(T, B, C, requires_grad=True)
            seq_lens = torch.from_numpy(np.array([11, 3, 2, 2, 1], dtype=np.int32))

            m = RNNTraceWrapper(cell_type)
            m_traced = torch.jit.trace(m, (x, seq_lens,))

            y = m(x, seq_lens)
            loss = torch.sum(y)
            loss.backward()
            grad = x.grad.clone()
            x.grad.zero_()

            y_traced = m_traced(x, seq_lens)
            loss_traced = torch.sum(y_traced)
            loss_traced.backward()
            grad_traced = x.grad.clone()

            self.assertEqual(y_traced, y)
            self.assertEqual(grad, grad_traced)

            f = io.BytesIO()
            torch.onnx._export(m, (x, seq_lens), f, verbose=False)

    def test_python_call_non_tensor(self):
        def foo(a, b, c):
            # type: (Tensor, int, Tuple[Tensor, int]) -> Tuple[int, Tensor]
            d, e = c
            return b + e, a + d

        @torch.jit.script
        def bar():
            x = torch.ones(3, 4)
            a, b = foo(x, 3, (x, 3))
            return a, b

        self.assertEqual((6, torch.ones(3, 4) + 1), bar())

    def test_python_call_non_tensor_wrong(self):
        with self.assertRaisesRegex(RuntimeError, r"but instead got value of type tuple"):
            def foo():
                # type: () -> Tensor
                return ((3, 4),)

            @torch.jit.script
            def bar():
                return foo()

            bar()

    def test_tuples(self):
        def foo(i):
            a = (i + 4, i * 2)
            c = a
            # some nonsense with if-statements and loops to check
            # that tuple lowering doesn't fail
            if True:
                c = (i * 9, i + 1)
            t0, t1 = c
            while False:
                t0, t1 = c
                c = (t1, t0)
            x = (1,)
            y = 1,
            return t0, x, y

        v = torch.rand(10, 3)
        self.checkScript(foo, (v,))

        with self.assertRaisesRegex(RuntimeError, r"variable 'a' previously has type \(Tensor, Tensor\)"):
            @torch.jit.script
            def mixtypes(x):
                a = (x, x)
                if True:
                    a = 4

    def test_if_tuple_sizes(self):
        with self.assertRaisesRegex(RuntimeError, "Type mismatch"):
            @torch.jit.script
            def diff_tuple_sizes(x):
                if False:
                    c0 = ((x, x), (x, x, x))
                else:
                    c0 = ((x, x, x), (x, x))
                return c0

    def test_if_different_type(self):
        with self.assertRaisesRegex(RuntimeError, "Type mismatch: c0 is set to type int "
                                    "in the true branch and type float in the false branch:"):
            @torch.jit.script
            def diff_type_used():
                if False:
                    c0 = 1
                else:
                    c0 = 1.0
                return c0

        with self.assertRaisesRegex(RuntimeError, "variable 'c0' previously has type float"):
            @torch.jit.script
            def diff_existing_type(x):
                c0 = 1.0
                if False:
                    c0 = 1
                    print(x)
                return x

        @torch.jit.script
        def diff_type_unused():
            if True:
                c0 = 1
                print(c0)
            else:
                c0 = 1.0
                print(c0)
            return 1

    def test_if_list(self):
        # testing that different length lists don't throw error
        @torch.jit.script
        def test_list(x):
            if True:
                c = [x, x]
            else:
                c = [x, x, x]
            return torch.cat(c)

        b = torch.zeros(2, 4)
        test_list.graph.propagate_shapes((b,), False)
        self.assertExpected(canonical(test_list.graph))

    def test_if_supertype(self):
        @torch.jit.script
        def tensor_unifying(x, y, z):

            # testing dynamic is appropriately set for y and z
            if True:
                x, y, z = x, y, z
            else:
                x, y, z = x, x, y

            return x, y, z

        a = torch.zeros(2, 2, dtype=torch.float)
        b = torch.zeros(2, 4, dtype=torch.long)
        c = torch.zeros(2, 4, dtype=torch.float)

        tensor_unifying.graph.propagate_shapes((a, b, c), False)
        self.assertExpected(canonical(tensor_unifying.graph))

    def test_type_annotations_repeated_list(self):
        @torch.jit.script
        def float_fn(x, y):
            # type: (float, BroadcastingList3[float]) -> List[float]
            return y
        self.assertEqual(float_fn(2.0, 1.0), float_fn(2.0, [1.0, 1.0, 1.0]))
        self.assertEqual(float_fn(2.0, 1.0), float_fn(2.0, (1.0, 1.0, 1.0)))

        @torch.jit.script
        def float_fn_call():
            print(float_fn(1.0, 1.0))
            print(float_fn(1.0, (1.0, 1.0, 1.0)))

        @torch.jit.script
        def int_fn(x):
            # type: (BroadcastingList3[int]) -> List[int]
            return x
        self.assertEqual(int_fn(1), int_fn([1, 1, 1]))
        self.assertEqual(int_fn(1), int_fn((1, 1, 1)))

        @torch.jit.script
        def int_fn_call():
            print(int_fn(1))
            print(int_fn((1, 1, 1)))

        with self.assertRaisesRegex(RuntimeError, "must be a positive integer:"):
            @torch.jit.script
            def fn(x):
                # type: (BroadcastingListx[int]) -> List[int]
                return x

        # TODO: the type comment in this seems to trip up flake8 for some reason
        # even though we have a noqa comment. Figure out why
        with self.assertRaisesRegex(RuntimeError, "Unknown type constructor"):
            @torch.jit.script
            def nested(x, y):
                # type: (int, Tuple[int, int[2]]) -> List[int] # noqa: T484
                return x

    def test_ntuple_builtins(self):
        from torch.nn.modules.utils import _single, _pair, _triple, _quadruple

        def test_ints():
            return _single(1), _pair(2), _triple(3), _quadruple(4)

        def test_floats():
            return _single(1), _pair(2.1), _triple(3.1), _quadruple(4.1)

        self.checkScript(test_ints, ())
        self.checkScript(test_floats, ())

    def test_embedding_renorm_grad_error(self):
        # Testing that the builtin call to embedding_renorm_ correctly throws
        # Error when .backward() is called on its input

        def embedding_norm(input, embedding_matrix, max_norm):
            F.embedding(input, embedding_matrix, max_norm=0.01)

        @torch.jit.script
        def embedding_norm_script(input, embedding_matrix, max_norm):
            # type: (Tensor, Tensor, float) -> None
            F.embedding(input, embedding_matrix, max_norm=0.01)

        for _ in [embedding_norm, embedding_norm_script]:
            input = torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]])
            embedding_matrix = torch.randn(10, 3)

            var1 = torch.randn(10, 3, requires_grad=True)
            var2 = var1.detach().requires_grad_()
            output1 = var1 * embedding_matrix
            output2 = var2 * embedding_matrix

            output1.sum().backward()

            ignore = F.embedding(input, embedding_matrix, max_norm=0.01)
            with self.assertRaisesRegex(RuntimeError, "modified"):
                output2.sum().backward()

    def test_type_annotations(self):
        def fn(x, y):
            # type: (Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor]
            return x, x * 2, x * 3

        with self.assertRaisesRegex(RuntimeError, r"need 4 values .* found only 3"):
            @torch.jit.script
            def script_fn(x):
                x, y, z, w = fn(x, x)

        with self.assertRaisesRegex(RuntimeError, r"too many values .* need 2 but found 3"):
            @torch.jit.script
            def script_fn2(x):
                x, y = fn(x, x)

        def fn_unpack(x):
            y, z, w = fn(x, x)
            return y

        def fn_index(x):
            q = fn(x, x)
            return x

        def fn_string(str, strpair):
            # type: (str, Tuple[str, str]) -> Tuple[str, int, str, str]
            str1, str2 = strpair
            return str, 2, str1, str2

        x = torch.ones(2, 2)
        self.checkScript(fn_unpack, (x,), optimize=True)
        self.checkScript(fn_index, (x,), optimize=True)
        self.checkScript(fn_string, ("1", ("3", "4")), optimize=True)

    def test_type_annotations_varargs(self):
        def fn_varargs(x, *args):
            return args[0] if args else x

        def fn1(x, y, z):
            return fn_varargs(x)

        def fn2(x, y, z):
            return fn_varargs(x, y)

        def fn3(x, y, z):
            return fn_varargs(x, y, z)

        x, y, z = [torch.randn(2, 2) for _ in range(3)]
        self.checkScript(fn1, (x, y, z), optimize=True)
        self.checkScript(fn2, (x, y, z), optimize=True)
        self.checkScript(fn3, (x, y, z), optimize=True)

    @unittest.skipIf(not PY35, "Python 3.5 needed")
    def test_type_annotation_py3(self):
        import importlib.util

        code = dedent("""
        import torch
        from torch import Tensor
        from typing import Tuple

        def fn(x : torch.Tensor, y : Tensor, z) -> Tuple[Tensor, Tensor, Tensor]:
            return (x, y + z, z)
        """)

        with tempfile.TemporaryDirectory() as tmp_dir:
            script_path = os.path.join(tmp_dir, 'script.py')
            with open(script_path, 'w') as f:
                f.write(code)
            fn = get_fn('test_type_annotation_py3', script_path)

            with self.assertRaisesRegex(RuntimeError, r"expected a value of type Tensor for argument"
                                                      r" '0' but found \(Tensor, Tensor\)"):
                @torch.jit.script
                def bad_fn(x):
                    x, y = fn((x, x), x, x)
                    return y

            with self.assertRaisesRegex(RuntimeError, r"too many values .* need 2 but found 3"):
                @torch.jit.script
                def bad_fn2(x):
                    x, y = fn(x, x, x)
                    return y

            with self.assertRaisesRegex(RuntimeError, r"need 4 values .* found only 3"):
                @torch.jit.script
                def bad_fn3(x):
                    x, y, z, w = fn(x, x, x)
                    return y

            def good_fn(x):
                y, z, w = fn(x, x, x)
                return y, z, w

            self.checkScript(good_fn, (torch.ones(2, 2),), optimize=True)

    def test_type_annotation_module(self):
        class BaseModule(torch.jit.ScriptModule):
            def foo(self, x):
                # type: (Tensor) -> Tensor
                return x + 1

            def bar(self, x, y):
                # type: (Tensor, Tensor) -> Tuple[Tensor, Tensor]
                return x + y, y

            def baz(self, x, y):
                return x

        class ModuleTooMany(BaseModule):
            @torch.jit.script_method
            def method(self, x):
                return self.foo(x, x)

        class ModuleTooFew(BaseModule):
            @torch.jit.script_method
            def method(self, x):
                return self.bar(x)

        class ModuleTooManyAssign(BaseModule):
            @torch.jit.script_method
            def method(self, x):
                y, z, w = self.bar(x, x)
                return x

        class ModuleDefault(BaseModule):
            @torch.jit.script_method
            def method(self, x):
                y = self.baz(x)
                return x

        with self.assertRaisesRegex(RuntimeError, "expected at most 1 arguments but found 2"):
            ModuleTooMany()
        with self.assertRaisesRegex(RuntimeError, "argument 1 not provided"):
            ModuleTooFew()
        with self.assertRaisesRegex(RuntimeError, "need 3 values .* found only 2"):
            ModuleTooManyAssign()
        with self.assertRaisesRegex(RuntimeError, "argument 1 not provided."):
            ModuleDefault()

    def test_script_define_order(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                pass

            @torch.jit.script_method
            def call_foo(self, input):
                return self.foo(input)

            @torch.jit.script_method
            def foo(self, input):
                return input + 1
        m = M()
        self.assertEqual(2, m.call_foo(torch.ones((), dtype=torch.int64)))

    def test_script_define_order_recursive_fail(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                pass

            @torch.jit.script_method
            def call_foo(self, input):
                return self.foo(input)

            @torch.jit.script_method
            def foo(self, input):
                self.call_foo(input)

        with self.assertRaisesRegex(RuntimeError, 'called recursively involving'):
            M()

    def test_script_kwargs_fn_call(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                pass

            @torch.jit.script_method
            def call_foo(self, input):
                return self.foo(input=input, bar=1)

            @torch.jit.script_method
            def foo(self, bar, input):
                # type: (int, Tensor) -> Tensor
                return input + bar
        m = M()
        self.assertEqual(2, m.call_foo(torch.ones((), dtype=torch.int64)))

    @unittest.skipIf(IS_WINDOWS, "NYI: fuser support for Windows")
    def test_trace_of_script(self):
        @torch.jit.script
        def foo(a, c):
            b = 0.0
            if bool(a == 0.0):
                b = 1.0
            return b + c

        a = torch.ones(1, dtype=torch.float)

        @_trace(torch.zeros(1, dtype=torch.float))
        def use(b):
            return foo(b - 1.0, a) + 1.0

        # test we propagated shapes through the function
        self.assertTrue("Dynamic" not in str(use.graph))

        self.assertEqual(3, use(torch.ones(1, dtype=torch.float)))
        self.assertEqual(2, use(torch.zeros(1, dtype=torch.float)))

    def test_if_define(self):
        @torch.jit.script
        def foo(a):
            if bool(a == 0):
                b = 1
            else:
                b = 0
            return b + 1

        @torch.jit.script
        def foo2(a):
            b = 0
            if bool(a == 0):
                b = 1
            return b + 1

        @torch.jit.script
        def foo3(a):
            b = 1
            if bool(a == 0):
                c = 4
            else:
                b = 0
            return b + 1

        a = torch.ones(1, dtype=torch.long)
        b = torch.zeros(1, dtype=torch.long)
        self.assertEqual(1, foo(a))
        self.assertEqual(2, foo(b))
        self.assertEqual(1, foo2(a))
        self.assertEqual(2, foo2(b))
        self.assertEqual(1, foo3(a))
        self.assertEqual(2, foo3(b))

    def test_script_module_export_submodule(self):
        class M1(torch.jit.ScriptModule):
            def __init__(self):
                super(M1, self).__init__(False)
                self.weight = nn.Parameter(torch.randn(2))

            @torch.jit.script_method
            def forward(self, thing):
                return self.weight + thing

        class M2(torch.jit.ScriptModule):
            def __init__(self):
                super(M2, self).__init__(False)
                # test submodule
                self.sub = M1()
                self.weight = nn.Parameter(torch.randn(2, 3))
                self.bias = nn.Parameter(torch.randn(2))
                self.define("""
                    def hi(self, a):
                        return self.weight.mm(a)
                """)

            @torch.jit.script_method
            def doit(self, input):
                return self.weight.mm(input)

            @torch.jit.script_method
            def doit2(self, input):
                return self.weight.mm(input)

            @torch.jit.script_method
            def doit3(self, input):
                return input + torch.ones([1], dtype=torch.double)

            @torch.jit.script_method
            def forward(self, input):
                a = self.doit(input)
                b = self.doit2(input)
                c = self.hi(input)
                return a + b + self.bias + c

        m_orig = M2()
        m_import = self.getExportImportCopy(m_orig)

        input = torch.randn(3, 2)
        self.assertEqual(m_orig.doit(input), m_import.doit(input))
        self.assertEqual(m_orig.hi(input), m_import.hi(input))
        self.assertEqual(m_orig.doit3(input), m_import.doit3(input))
        self.assertEqual(m_orig.forward(input), m_import.forward(input))

    @skipIfNoTorchVision
    def test_script_module_trace_resnet18(self):
        x = torch.ones(1, 3, 224, 224)
        m_orig = torch.jit.trace(torchvision.models.resnet18(), torch.ones(1, 3, 224, 224))
        m_import = self.getExportImportCopy(m_orig)

        input = torch.randn(1, 3, 224, 224, requires_grad=True)
        output_orig = m_orig(input)
        output_orig.sum().backward()
        grad_orig = input.grad.clone()
        input.grad.zero_()

        output_import = m_import(input)
        output_import.sum().backward()
        grad_import = input.grad.clone()

        self.assertEqual(output_orig, output_import)
        self.assertEqual(grad_orig, grad_import)

    @skipIfNoTorchVision
    def test_script_module_script_resnet(self):
        def conv1x1(in_planes, out_planes, stride=1):
            """1x1 convolution"""
            return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

        def conv3x3(in_planes, out_planes, stride=1):
            """3x3 convolution with padding"""
            return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                             padding=1, bias=False)

        class BasicBlock(torch.jit.ScriptModule):
            expansion = 1
            __constants__ = ['downsample']

            def __init__(self, inplanes, planes, stride=1, downsample=None):
                super(BasicBlock, self).__init__()
                self.conv1 = conv3x3(inplanes, planes, stride)
                self.bn1 = nn.BatchNorm2d(planes)
                self.relu = nn.ReLU(inplace=True)
                self.conv2 = conv3x3(planes, planes)
                self.bn2 = nn.BatchNorm2d(planes)
                self.downsample = downsample
                self.stride = stride

            @torch.jit.script_method
            def forward(self, x):
                residual = x

                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)

                out = self.conv2(out)
                out = self.bn2(out)

                if self.downsample is not None:
                    residual = self.downsample(x)

                out += residual
                out = self.relu(out)

                return out

        class ResNet(torch.jit.ScriptModule):
            __constants__ = ['layer1', 'layer2', 'layer3', 'layer4']

            def __init__(self, block, layers, num_classes=1000):
                super(ResNet, self).__init__()
                self.inplanes = 64
                self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                       bias=False)
                self.bn1 = nn.BatchNorm2d(64)
                self.relu = nn.ReLU(inplace=True)
                self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                self.layer1 = self._make_layer(block, 64, layers[0])
                self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
                self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
                self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(512 * block.expansion, num_classes)

                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    elif isinstance(m, nn.BatchNorm2d):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)

            def _make_layer(self, block, planes, blocks, stride=1):
                downsample = None
                if stride != 1 or self.inplanes != planes * block.expansion:
                    downsample = nn.Sequential(
                        conv1x1(self.inplanes, planes * block.expansion, stride),
                        nn.BatchNorm2d(planes * block.expansion),
                    )

                layers = []
                layers.append(block(self.inplanes, planes, stride, downsample))
                self.inplanes = planes * block.expansion
                for _ in range(1, blocks):
                    layers.append(block(self.inplanes, planes))

                return nn.Sequential(*layers)

            @torch.jit.script_method
            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)

                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)

                x = self.avgpool(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)

                return x

        resnet18 = ResNet(BasicBlock, [2, 2, 2, 2])

        resnet18_imported = self.getExportImportCopy(resnet18)

        input = torch.randn(1, 3, 224, 224, requires_grad=True)
        output_orig = resnet18(input)
        output_orig.sum().backward()
        grad_orig = input.grad.clone()
        input.grad.zero_()
        output_import = resnet18_imported(input)
        output_import.sum().backward()
        grad_import = input.grad.clone()

        self.assertEqual(output_orig, output_import)
        self.assertEqual(grad_orig, grad_import)

    def test_script_module_export_tensor_type(self):
        class M(torch.jit.ScriptModule):

            def __init__(self, type):
                super(M, self).__init__(False)
                self.param = torch.nn.Parameter(torch.zeros((5, 5), dtype=type).random_())

            @torch.jit.script_method
            def foo(self):
                return self.param

        for type in [torch.float, torch.double]:
            m_orig = M(type)
            m_import = self.getExportImportCopy(m_orig)
            # check to make sure the storage wasn't resized
            self.assertTrue(m_orig.param.storage().size() == 25)
            self.assertEqual(m_orig.foo(), m_import.foo())
            self.assertTrue(m_orig.foo().dtype == m_import.foo().dtype)

    @unittest.skipIf(not RUN_CUDA, "testing cuda tensors require CUDA")
    def test_script_module_export_tensor_cuda(self):
        class M(torch.jit.ScriptModule):

            def __init__(self):
                super(M, self).__init__(False)
                self.param = torch.nn.Parameter(torch.zeros((5, 5), device='cuda:0').random_())

            @torch.jit.script_method
            def foo(self):
                return self.param

        m_orig = M()
        m_import = self.getExportImportCopy(m_orig)
        # check to make sure the storage wasn't resized
        self.assertTrue(m_orig.param.storage().size() == 25)
        self.assertTrue(m_import.foo().device == torch.device('cuda:0'))
        self.assertEqual(m_orig.foo(), m_import.foo())
        self.assertTrue(m_orig.foo().dtype == m_import.foo().dtype)

    def test_script_module_export_blocks(self):
        class M(torch.jit.ScriptModule):
            def __init__(self, n, m):
                super(M, self).__init__()
                self.weight = torch.nn.Parameter(torch.rand(n, m))

            @torch.jit.script_method
            def forward(self, input):
                if bool(input.sum() > 0):
                    output = self.weight.mv(input)
                else:
                    output = self.weight + input
                return output

        m_orig = M(200, 200)
        m_import = self.getExportImportCopy(m_orig)

        t = torch.rand(200)
        self.assertEqual(m_orig(t), m_import(t))

    def test_script_module_export_shared_storage(self):
        class M(torch.jit.ScriptModule):

            def __init__(self):
                super(M, self).__init__(False)
                self.param1 = torch.nn.Parameter(torch.rand(5, 5))
                self.param2 = torch.nn.Parameter(self.param1[3])
                self.param3 = torch.nn.Parameter(torch.rand(5, 5))
                self.param4 = torch.nn.Parameter(torch.rand(11, 5)[1:6])

            @torch.jit.script_method
            def foo(self):
                return self.param1 + self.param2 + self.param3 + self.param4

        m_orig = M()
        m_import = self.getExportImportCopy(m_orig)

        self.assertEqual(m_orig.foo(), m_import.foo())
        self.assertTrue(m_import.param1.storage().data_ptr() == m_import.param2.storage().data_ptr())
        self.assertTrue(m_import.param1.storage().data_ptr() != m_import.param3.storage().data_ptr())

    def test_onnx_export_script_module(self):
        class ModuleToExport(torch.jit.ScriptModule):
            def __init__(self):
                super(ModuleToExport, self).__init__()

            @torch.jit.script_method
            def forward(self, x):
                y = x - x
                return x + x

        mte = ModuleToExport()
        outputs = mte(torch.zeros(1, 2, 3))
        self.assertExpected(torch.onnx.export_to_pretty_string(
            mte, (torch.zeros(1, 2, 3),), None, verbose=False,
            example_outputs=outputs))

    def test_trace_nested_datatypes(self):
        @torch.jit.script
        def foo(x):
            return [[x + 1, x - 1], [x + 2, x - 2]]

        def bar(x):
            list_stuff = foo(x)
            return list_stuff[0][0], list_stuff[1][1]

        traced = torch.jit.trace(bar, torch.rand(3, 4))
        x = torch.rand(5, 6)
        self.assertEqual(bar(x), traced(x))

    @suppress_warnings
    def test_onnx_export_func_with_warnings(self):
        @torch.jit.script
        def func_with_warning(inp):
            return torch.nn.functional.sigmoid(inp)  # triggers a deprecation warning

        class WarningTest(torch.nn.Module):
            def __init__(self):
                super(WarningTest, self).__init__()

            def forward(self, x):
                return func_with_warning(x)

        outputs = WarningTest()(torch.randn(42))
        # no exception
        torch.onnx.export_to_pretty_string(
            WarningTest(), torch.randn(42), None, verbose=False,
            example_outputs=outputs)

    def test_onnx_export_script_python_fail(self):
        class ModuleToInline(torch.jit.ScriptModule):
            def __init__(self):
                super(ModuleToInline, self).__init__()

            def forward(self, x):
                return torch.neg(x)

        class ModuleToExport(torch.jit.ScriptModule):
            def __init__(self):
                super(ModuleToExport, self).__init__()
                self.mod = ModuleToInline()

            @torch.jit.script_method
            def forward(self, x):
                y = self.mod(x)
                return y + y

        mte = ModuleToExport()
        outputs = mte(torch.zeros(1, 2, 3))
        f = io.BytesIO()
        with self.assertRaisesRegex(RuntimeError, "Couldn't export Python operator"):
            torch.onnx._export(mte, (torch.zeros(1, 2, 3),), f, verbose=False,
                               example_outputs=outputs)

    def test_onnx_export_script_inline_trace(self):
        class ModuleToInline(torch.jit.ScriptModule):
            def __init__(self):
                super(ModuleToInline, self).__init__()

            def forward(self, x):
                return torch.neg(x)

        class ModuleToExport(torch.jit.ScriptModule):
            def __init__(self):
                super(ModuleToExport, self).__init__()
                self.mod = torch.jit.trace(ModuleToInline(), torch.zeros(1, 2, 3))

            @torch.jit.script_method
            def forward(self, x):
                y = self.mod(x)
                return y + y

        mte = ModuleToExport()
        outputs = mte(torch.zeros(1, 2, 3))
        self.assertExpected(torch.onnx.export_to_pretty_string(
            mte, (torch.zeros(1, 2, 3),), None, verbose=False,
            example_outputs=outputs))

    def test_onnx_export_script_inline_script(self):
        class ModuleToInline(torch.jit.ScriptModule):
            def __init__(self):
                super(ModuleToInline, self).__init__()

            @torch.jit.script_method
            def forward(self, x):
                return torch.neg(x)

        class ModuleToExport(torch.jit.ScriptModule):
            def __init__(self):
                super(ModuleToExport, self).__init__()
                self.mod = ModuleToInline()

            @torch.jit.script_method
            def forward(self, x):
                y = self.mod(x)
                return y + y

        mte = ModuleToExport()
        outputs = mte(torch.zeros(1, 2, 3))
        self.assertExpected(torch.onnx.export_to_pretty_string(
            mte, (torch.zeros(1, 2, 3),), None, verbose=False,
            example_outputs=outputs))

    def test_onnx_export_script_module_loop(self):
        class ModuleToExport(torch.jit.ScriptModule):
            def __init__(self):
                super(ModuleToExport, self).__init__()

            @torch.jit.script_method
            def forward(self, x):
                # test if we support end to end onnx export on loop and
                # nested loops with and without loop index
                for _ in range(5):
                    for i in range(3):
                        x = x + i
                return x

        mte = ModuleToExport()
        outputs = mte(torch.zeros(1, 2, 3))
        self.assertExpected(torch.onnx.export_to_pretty_string(
            mte, (torch.zeros(1, 2, 3),), None, verbose=False,
            example_outputs=outputs))

    def test_onnx_export_script_truediv(self):
        class ModuleToExport(torch.jit.ScriptModule):
            def __init__(self):
                super(ModuleToExport, self).__init__()

            @torch.jit.script_method
            def forward(self, x):
                z = x.size(0) / 2
                return x + z

        mte = ModuleToExport()
        outputs = mte(torch.zeros(1, 2, 3))
        self.assertExpected(torch.onnx.export_to_pretty_string(
            mte, (torch.zeros(1, 2, 3),), None, verbose=False,
            example_outputs=outputs))

    def test_onnx_raw_export_script_truediv(self):
        class ModuleToExport(torch.jit.ScriptModule):
            def __init__(self):
                super(ModuleToExport, self).__init__()

            @torch.jit.script_method
            def forward(self, x):
                z = x.size(0) / 2
                return x + z

        mte = ModuleToExport()
        outputs = mte(torch.zeros(1, 2, 3))
        self.assertExpected(torch.onnx.export_to_pretty_string(
            mte, (torch.zeros(1, 2, 3),), None, verbose=False,
            example_outputs=outputs, export_raw_ir=True))

    def test_onnx_export_script_non_alpha_add_sub(self):
        class ModuleToExport(torch.jit.ScriptModule):
            def __init__(self):
                super(ModuleToExport, self).__init__()

            @torch.jit.script_method
            def forward(self, x):
                bs = x.size(0) + 1
                return bs - 1

        mte = ModuleToExport()
        outputs = torch.LongTensor([mte(torch.rand(3, 4))])
        self.assertExpected(torch.onnx.export_to_pretty_string(
            mte, (torch.rand(3, 4),), None, verbose=False,
            example_outputs=outputs))

    def test_onnx_export_script_module_if(self):
        class ModuleToExport(torch.jit.ScriptModule):
            def __init__(self):
                super(ModuleToExport, self).__init__()

            @torch.jit.script_method
            def forward(self, x):
                if bool(torch.sum(x) > 0):
                    x = torch.neg(x)
                return x

        mte = ModuleToExport()
        outputs = mte(torch.zeros(1, 2, 3, dtype=torch.long))
        self.assertExpected(torch.onnx.export_to_pretty_string(
            mte, (torch.zeros(1, 2, 3),), None, verbose=False,
            example_outputs=outputs))

    def test_onnx_export_script_inline_params(self):
        class ModuleToInline(torch.jit.ScriptModule):
            def __init__(self):
                super(ModuleToInline, self).__init__()
                self.m = torch.nn.Parameter(torch.ones(3, 3))
                self.unused = torch.nn.Parameter(torch.ones(1, 2, 3))

            @torch.jit.script_method
            def forward(self, x):
                return torch.mm(x, self.m)

        class ModuleToExport(torch.jit.ScriptModule):
            def __init__(self):
                super(ModuleToExport, self).__init__()
                self.mod = ModuleToInline()
                self.param = torch.nn.Parameter(torch.ones(3, 4))

            @torch.jit.script_method
            def forward(self, x):
                y = self.mod(x)
                return torch.mm(y, self.param)

        mte = ModuleToExport()
        result = mte(torch.zeros(2, 3))
        reference = torch.mm(torch.mm(torch.zeros(2, 3), torch.ones(3, 3)), torch.ones(3, 4))
        self.assertEqual(result, reference)
        self.assertExpected(torch.onnx.export_to_pretty_string(
            mte, (torch.ones(2, 3),), None, verbose=False,
            example_outputs=result, propagate=True))

    def test_trace_with_size(self):
        @_trace(torch.zeros(1, 1))
        def foo(x):
            return x + 1

        @torch.jit.script
        def bar(x):
            y = int(foo(x))
            if True:
                y = 7
            return y + 1

        self.assertEqual(8, bar(torch.ones(1, 1)))

    def test_tracing_slicing(self):
        @_trace(torch.zeros(10))
        def foo_trace(x):
            return x[-5:-3]

        @torch.jit.script
        def foo_script(x):
            return x[-5:-3]

        def foo(x):
            return x[-5:-3]

        a = torch.arange(0, 8)
        b = torch.arange(0, 20)
        self.assertEqual(foo_trace(a), foo_script(a))
        self.assertEqual(foo_trace(a), foo(a))
        self.assertNotEqual(foo_trace(a), foo_trace(b))

    def test_tracing_indexing(self):
        @_trace(torch.zeros(10))
        def foo_trace(x):
            return x[-2]

        @torch.jit.script
        def foo_script(x):
            return x[-2]

        def foo(x):
            return x[-2]

        a = torch.arange(0, 8)
        b = torch.arange(0, 20)
        self.assertEqual(foo_script(a), foo_trace(a))
        self.assertEqual(foo_trace(a), foo(a))
        self.assertNotEqual(foo_trace(a), foo_trace(b))

    def test_index_select_shape_prop(self):

        @torch.jit.script
        def foo(x, y):
            return torch.index_select(x, index=y, dim=1)

        a = torch.zeros(2, 2)
        b = torch.zeros(4, dtype=torch.long)
        torch._C._jit_pass_complete_shape_analysis(foo.graph, (a, b), False)
        self.assertExpected(canonical(foo.graph))

    def test_onnx_export_speculate(self):

        class Foo(torch.jit.ScriptModule):
            def __init__(self, m):
                super(Foo, self).__init__()
                self.m = m

            @torch.jit.script_method
            def forward(self, x):
                x += x
                # because we are testing if we emit `if` statement correctly
                # we cannot use `True` as the condition. Constant prop
                # would remove the `if` statements.
                c = torch.sum(x) > 4
                if bool(c):
                    if bool(c):
                        y = self.m(x)
                    else:
                        y = self.m(x)
                else:
                    y = self.m(x)
                return y

        linear = torch.jit.trace(nn.Linear(10, 20).float(), torch.zeros(1, 10, dtype=torch.float))

        @torch.jit.script
        def transpose(x):
            return x.t()

        f1 = Foo(transpose)
        outputs_f1 = f1(torch.ones(1, 10, dtype=torch.float))
        f2 = Foo(linear)
        outputs_f2 = f2(torch.ones(1, 10, dtype=torch.float))

        onnx_ish = torch.onnx.export_to_pretty_string(
            f1,
            (torch.ones(1, 10, dtype=torch.float), ),
            None, verbose=False, example_outputs=outputs_f1)
        self.assertExpected(onnx_ish, subname='f1')
        onnx_ish = torch.onnx.export_to_pretty_string(
            f2,
            (torch.ones(1, 10, dtype=torch.float), ),
            None, verbose=False, example_outputs=outputs_f2)
        self.assertExpected(onnx_ish, subname='f2')

    def test_onnx_export_shape_reshape(self):
        class Foo(torch.nn.Module):
            def forward(self, x):
                import torch.onnx.operators
                x = x.repeat(5, 1, 1)
                shape = torch.onnx.operators.shape_as_tensor(x)
                reshaped = torch.onnx.operators.reshape_from_tensor_shape(x, shape)
                return reshaped

        foo = torch.jit.trace(Foo(), torch.zeros(1, 2, 3))
        outputs = foo(torch.zeros(1, 2, 3))
        f = io.BytesIO()
        s = torch.onnx.export_to_pretty_string(foo, (torch.zeros(1, 2, 3)), f,
                                               example_outputs=outputs)
        self.assertExpected(s)

    def test_shape_analysis_loop(self):
        def foo(a, b, x):
            c = a
            # on the first iteration of the loop it appears that
            # c should have a expand to the size of b
            # but on the second+ iterations, there is no broadcast and the
            # sizes are different.
            # previously this would cause the compiler to (1) enter an infinite
            # loop trying to compute the shape, and (2) insert invalid
            # broadcasts.
            # this test ensure we don't regress on these issues
            for _ in range(2):
                a = c + b
                c = x
                b = x
            return a

        self.checkScript(foo, (torch.zeros(1), torch.zeros(4), torch.zeros(5)), optimize=False)

    def test_intlist_args(self):
        def func_1(x):
            return torch.nn.functional.adaptive_avg_pool1d(x, 1)

        def func_2(x):
            return torch.nn.functional.adaptive_avg_pool1d(x, output_size=1)

        def func_3(x):
            return torch.nn.functional.adaptive_avg_pool1d(x, output_size=[1])

        x = torch.randn(8, 8, 8)
        self.checkScript(func_1, [x], optimize=True)
        self.checkScript(func_2, [x], optimize=True)
        self.checkScript(func_3, [x], optimize=True)

    def test_wrong_implicit_expand(self):

        @_trace(torch.zeros(3), torch.zeros(1))
        def foo(a, b):
            return a + b

        a = torch.rand(4)
        b = torch.rand(4)
        self.assertEqual(a + b, foo(a, b))

    def test_builtin_args_fails(self):

        with self.assertRaisesRegex(RuntimeError, 'expected at most'):
            @torch.jit.script
            def f0(a):
                torch.sum(a, a, a, a)

        with self.assertRaisesRegex(RuntimeError, 'argument self not provided'):
            @torch.jit.script
            def f1(a):
                torch.sum(foo=4)

        with self.assertRaisesRegex(RuntimeError, 'specified twice'):
            @torch.jit.script
            def f2(a):
                torch.sum(a, self=a)

        with self.assertRaisesRegex(RuntimeError, 'not provided'):
            @torch.jit.script
            def f3(a):
                torch.sum(dim=4)

        with self.assertRaisesRegex(RuntimeError, 'for argument \'tensors\' but found Tensor'):
            @torch.jit.script
            def f4(a):
                torch.cat(a)

        with self.assertRaisesRegex(RuntimeError, r'argument \'tensors\' but found int\[\]'):
            @torch.jit.script
            def f5(a):
                torch.cat([3])

        with self.assertRaisesRegex(RuntimeError, 'Lists must contain only a single type'):
            @torch.jit.script
            def f6(a):
                a.expand(size=[3, [4]])

        with self.assertRaisesRegex(RuntimeError, 'xpected a value of type Tensor for argument \'self\''):
            @torch.jit.script
            def f7(a):
                torch.sum([4])

    def test_builtin_args(self):

        def t0(a):
            # default arg dim
            return torch.cat([a, a])

        self.checkScript(t0, (torch.zeros(1, 1),))

        def t1(a):
            # keywords out of order
            return torch.cat(dim=1, tensors=[a, a])

        self.checkScript(t1, (torch.zeros(1, 1, 2),))

        def t2(a):
            # mix const/non-const attributes
            if True:
                b = 1
            else:
                b = 0
            return torch.sum(a, dim=b, keepdim=False)

        self.checkScript(t2, (torch.zeros(1, 1, 2),))

    def test_parser_type_annotations(self):
        cu = torch.jit.CompilationUnit('''
            def foo(x : Tensor, y : Tuple[Tuple[Tensor, Tensor], Tensor]) -> Tuple[Tensor, Tensor]:
                return x, x
        ''')

        self.assertExpected(cu.__getattr__('foo').pretty_print_schema())

    def test_parser_type_annotations_comment(self):
        cu = torch.jit.CompilationUnit('''
            def foo(x, y):
                # type: (Tensor, Tuple[Tuple[Tensor, Tensor], Tensor]) -> Tuple[Tensor, Tensor]
                return x, x
        ''')

        self.assertExpected(cu.__getattr__('foo').pretty_print_schema())

    def test_parser_type_annotations_unknown_type(self):
        with self.assertRaisesRegex(RuntimeError, r'Unknown type name Foo'):
            cu = torch.jit.CompilationUnit('''
                def foo(x : Tensor, y : Tuple[Tuple[Foo, Tensor], Tensor]) -> Tuple[Tensor, Tensor]:
                    return x, x
            ''')

    def test_parser_type_annotations_subscript_non_ident(self):
        with self.assertRaisesRegex(RuntimeError, r'Subscripted type must be a type identifier'):
            cu = torch.jit.CompilationUnit('''
                def foo(x : Tensor, y : Tuple[Tensor, Tensor][Tensor]) -> Tuple[Tensor, Tensor]:
                    return x, x
            ''')

    def test_parser_type_annotations_subscript_tensor(self):
        with self.assertRaisesRegex(RuntimeError, r'Unknown type constructor Tensor'):
            cu = torch.jit.CompilationUnit('''
                def foo(x : Tensor, y : Tensor[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
                    return x, x
            ''')

    def test_parser_type_annotations_incompatible_expression(self):
        with self.assertRaisesRegex(RuntimeError, r'Expression of type \+ cannot be used in a type expression'):
            cu = torch.jit.CompilationUnit('''
                def foo(x : Tensor, y : Tuple[3 + 4, Tensor]) -> Tuple[Tensor, Tensor]:
                    return x, x
            ''')

    def test_gather_dynamic_index(self):
        def t(x):
            gather1 = x[0]
            idx = 0 + 1
            gather2 = x[idx]
            return gather1 + gather2

        self.checkScript(t, (torch.zeros(3, 2, 3),))

    def test_slice_dynamic_index(self):
        def t(x):
            slice1 = x[0:1]
            zero = 0
            one = zero + 1
            slice2 = x[zero:one]
            return slice1 + slice2

        self.checkScript(t, (torch.zeros(3, 2, 3),))

    def test_addmm_grad(self):
        """ This test checks several things:
            1. An expand node was inserted before the addmm operating on the
               bias term.
            2. The fused form of addmm appears in the ultimate graph that's
               executed.
            3. A sum op was emitted for accumulating gradients along the 0th
               (expanded) dimension of the bias term.
            4. The correct symbolic representation for the backward pass of the
               mm operator was emitted (x.t() -> mm)

            TODO: we should actually check these conditions once we have a way
            to dump the GraphExecutor state. Namely the processed forward graph
            and the backward graph.
        """
        @torch.jit.script
        def addmm_grad_test(b, x, w):
            return torch.addmm(b, x, w)

        # Initialize param and input values
        w_init = torch.rand(2, 5)
        b_init = torch.rand(5)
        x = torch.rand(3, 2)

        # Clone trainable params
        b = b_init.clone()
        b.requires_grad_()
        w = w_init.clone()
        w.requires_grad_()

        # Test symbolic differentiation
        y = addmm_grad_test(b, x, w)
        y.sum().backward()

        # clone params for autograd reference
        b_ref = b_init.clone()
        b_ref.requires_grad_()
        w_ref = w_init.clone()
        w_ref.requires_grad_()
        y_ref = torch.addmm(b_ref, x, w_ref)
        y_ref.sum().backward()

        self.assertEqual(w.grad, w_ref.grad)
        self.assertEqual(b.grad, b_ref.grad)

    def test_zeros(self):
        class M(torch.jit.ScriptModule):
            __constants__ = ['d']

            def __init__(self):
                self.d = torch.device('cpu')

            @torch.jit.script_method
            def create(self):
                return torch.zeros([1, 1, 2], dtype=torch.float, device=self.d, layout=torch.strided)

        r = M().create()
        self.assertEqual(r.dtype, torch.float)
        self.assertEqual(torch.zeros([1, 1, 2], dtype=torch.float), r)

    def test_vararg_zeros(self):
        def foo():
            return torch.zeros(3, 4, 5, dtype=torch.int)

        self.checkScript(foo, ())

    def test_rand(self):
        def test_rand():
            a = torch.rand([3, 4])
            return a + 1.0 - a

        self.checkScript(test_rand, ())

    def test_erase_number_types(self):
        def func(a):
            b = 7 + 1 + 3
            c = a + b
            c += b
            return c

        graph = torch.jit.script(func).graph
        self.run_pass('remove_inplace_ops', graph)
        self.run_pass('erase_number_types', graph)
        self.assertExpectedGraph(graph)

    def test_mm_batching(self):
        lstm_cell = torch.jit.script(LSTMCellS)

        def lstm(x, hx, cx, w_ih, w_hh, b_ih, b_hh):
            for i in range(x.size(0)):
                hx, cx = lstm_cell(x[i], hx, cx, w_ih, w_hh, b_ih, b_hh)
            return hx

        slstm = torch.jit.script(lstm)

        inputs = get_lstm_inputs('cpu', training=True, seq_length=10)
        slstm(*inputs).sum().backward()

        fw_graph = slstm.graph_for(*inputs)
        bw_graph = backward_graph(slstm, diff_graph_idx=0)
        self.assertTrue('prim::MMBatchSide' in str(fw_graph))
        self.assertTrue('prim::MMTreeReduce' in str(bw_graph))

        sout = slstm(*inputs)
        out = lstm(*inputs)
        self.assertEqual(slstm(*inputs), lstm(*inputs))
        self.assertEqual(torch.autograd.grad(slstm(*inputs).sum(), inputs),
                         torch.autograd.grad(lstm(*inputs).sum(), inputs))

    def test_loop_unrolling(self):
        def fn(x):
            y = 0
            for i in range(int(x)):
                y += i
            return y

        graph = torch.jit.script(fn).graph
        self.run_pass('loop_unrolling', graph)
        self.assertExpectedGraph(graph)
        self.checkScript(fn, (torch.tensor(10),))

    def test_loop_unrolling_const(self):
        def fn():
            y = 0
            for _ in range(10):
                y += 1
            return y

        def fn2():
            y = 0
            for i in range(10):
                y += i
            return y

        def check(fn, name):
            graph = torch.jit.script(fn).graph
            self.run_pass('loop_unrolling', graph)
            self.assertExpectedGraph(graph, subname=name)
            self.checkScript(fn, ())

        check(fn, 'add_const')
        check(fn2, 'add_iter')

    def test_loop_unrolling_nested(self):
        def fn(x):
            y = 0
            for _ in range(10):
                for j in range(int(x)):
                    y += j
            return y

        graph = torch.jit.script(fn).graph
        self.run_pass('loop_unrolling', graph)
        self.assertExpectedGraph(graph)
        self.checkScript(fn, (torch.tensor(10),))

    def test_loop_unroll_unused_counter(self):
        def fn(x):
            y = 0
            for _ in range(int(x)):
                y += 1
            return y

        graph = torch.jit.script(fn).graph
        self.run_pass('loop_unrolling', graph)
        self.assertExpectedGraph(graph)

    def test_loop_unroll_negative(self):
        def fn(x):
            y = 0
            for _ in range(int(x)):
                y += 1
            return y

        self.checkScript(fn, (torch.tensor(-20),))
        self.checkScript(fn, (torch.tensor(-2),))
        self.checkScript(fn, (torch.tensor(-1),))
        self.checkScript(fn, (torch.tensor(0),))
        self.checkScript(fn, (torch.tensor(1),))
        self.checkScript(fn, (torch.tensor(2),))

    def test_where(self):
        def fn(x, y):
            return torch.where(x > 0.0, x, y)

        self.checkScript(fn, (torch.randn(3, 2, dtype=torch.float), torch.ones(3, 2, dtype=torch.float)))

    def test_where_method(self):
        def fn(x, y):
            return x.where(x > 0.0, y)

        self.checkScript(fn, (torch.randn(3, 2, dtype=torch.float), torch.ones(3, 2, dtype=torch.float)))

    def test_reassign_module_lhs(self):
        with self.assertRaisesRegex(RuntimeError, 'Cannot re-assign \'self\' because it has type value and self is'
                                    ' not a first-class value.  Only reassignments to first-class values are allowed'):
            class ReassignSelfLHS(torch.jit.ScriptModule):
                @torch.jit.script_method
                def forward(self, x):
                    for _ in range(20):
                        self = x
                    return self

            ReassignSelfLHS()

    def test_reassign_module_rhs(self):
        with self.assertRaisesRegex(RuntimeError, 'Cannot re-assign \'x\' to a value of type module because x is not a'
                                    ' first-class value.  Only reassignments to first-class values are allowed'):
            class ReassignSelfRHS(torch.jit.ScriptModule):
                @torch.jit.script_method
                def forward(self, x):
                    for _ in range(20):
                        x = self
                    return self

            ReassignSelfRHS()

    def test_unknown_builtin(self):
        with self.assertRaisesRegex(RuntimeError, 'unknown builtin op'):
            @torch.jit.script
            def unknown_builtin(x):
                return x.splork(3)

    def test_return_tuple(self):
        def return_tuple(x):
            a = (x, x)
            return a, x
        self.checkScript(return_tuple, (torch.rand(4),))

    def test_method_no_self(self):
        with self.assertRaisesRegex(RuntimeError, 'methods must have a self argument'):
            class MethodNoSelf(torch.jit.ScriptModule):
                @torch.jit.script_method
                def forward():
                    return torch.zeros(3, 4)

            MethodNoSelf()

    def test_return_stmt_not_at_end(self):
        def return_stmt(x):
            if bool(x > 3):
                return x + 3
            else:
                return x
        self.checkScript(return_stmt, (torch.rand(1),))

    def test_for_range_no_arg(self):
        with self.assertRaisesRegex(RuntimeError, r'range\(\) expects 1 argument but got 0'):
            @torch.jit.script
            def range_no_arg(x):
                for _ in range():
                    x += 1
                return x

    def test_list_iterables(self):
        with self.assertRaisesRegex(RuntimeError, 'List of iterables is not supported currently'):
            cu = torch.jit.CompilationUnit('''
            def list_iterables(x):
                for i, j in [2, 3, 4], [5, 6, 7]:
                    x += i
                    x += j
                return x
            ''')

    def test_for_tuple_unpack(self):
        with self.assertRaisesRegex(RuntimeError, 'Iteration variable unpacking is not supported'):
            cu = torch.jit.CompilationUnit('''
            def for_tuple_unpack(x, y):
                for i, j in [[3, 4], [5, 6], [7, 8]]:
                    x += i
                    y += j
                return x, y
            ''')

    def test_single_starred_lhs(self):
        with self.assertRaisesRegex(RuntimeError, 'A Starred expression may only appear on the lhs within the presence'
                                                  ' of another non-starred expression'):
            cu = torch.jit.CompilationUnit('''
            def single_starred_lhs(x):
                a = (x, x, x)
                *b, = a
                return b
            ''')

    def test_singleton_tuple_unpack(self):
        def foo(a):
            b, = (a,)
            return b + 1
        self.checkScript(foo, (torch.rand(3),))

    def test_multi_reduction(self):
        with self.assertRaisesRegex(
                RuntimeError,
                'augmented assignment can only have one LHS expression'):
            cu = torch.jit.CompilationUnit('''
            def multi_reduction(x):
                a, b += x
                return a, b
            ''')

    def test_invalid_call_arguments(self):
        with self.assertRaisesRegex(RuntimeError, 'arguments for call are not valid'):
            @torch.jit.script
            def invalid_call_arguments(x):
                return torch.unsqueeze(3, 4, 5, 6, 7, 8)

    def test_invalid_lhs_assignment(self):
        with self.assertRaisesRegex(RuntimeError, 'unexpected expression'):
            cu = torch.jit.CompilationUnit('''
            def invalid_lhs_assignment(x):
                x + 1 = x
                return x
            ''')

    def test_multi_starred_expr_lhs(self):
        with self.assertRaisesRegex(RuntimeError, 'Only one starred expression is allowed on the lhs'):
            cu = torch.jit.CompilationUnit('''
            def multi_starred_expr_lhs():
                a, *b, *c = [1, 2, 3, 4, 5, 6]
                return a
            ''')

    def test_pack_tuple_into_non_var(self):
        with self.assertRaisesRegex(RuntimeError, 'Cannot pack a tuple into a non-variable'):
            cu = torch.jit.CompilationUnit('''
            def pack_tuple_into_non_var(x):
                a, *1 = (3, 4, 5)
                return x
            ''')

    def test_print_kwargs(self):
        with self.assertRaisesRegex(RuntimeError, 'print doesn\'t accept any keyword arguments'):
            cu = torch.jit.CompilationUnit('''
            def print_kwargs(x):
                print(x, flush=True)
                return x
            ''')

    def test_builtin_use_as_value(self):
        with self.assertRaisesRegex(RuntimeError, 'builtin cannot be used as a value'):
            @torch.jit.script
            def builtin_use_as_value(x):
                return x.unsqueeze

    def test_wrong_use_as_tuple(self):
        with self.assertRaisesRegex(RuntimeError, 'cannot be used as a tuple'):
            def test_fn():
                return 3

            @torch.jit.script
            def wrong_use_as_tuple(self):
                a, b = test_fn
                return a

    def test_wrong_attr_lookup(self):
        with self.assertRaisesRegex(RuntimeError, 'attribute lookup is not defined on builtin'):
            @torch.jit.script
            def wrong_attr_lookup(self, x):
                a = x.unsqueeze.myattr
                return a

    def test_wrong_use_as_callable(self):
        with self.assertRaisesRegex(RuntimeError, 'cannot call a value'):
            @torch.jit.script
            def wrong_use_as_callable(x):
                return x(3, 4, 5)

    def test_python_val_doesnt_have_attr(self):
        with self.assertRaisesRegex(RuntimeError, 'object has no attribute abcd'):

            @torch.jit.script
            def python_val_doesnt_have_attr():
                # this has to be a module otherwise attr lookup would not be
                # allowed in the first place
                return shutil.abcd

    def test_wrong_module_attr_lookup(self):
        with self.assertRaisesRegex(RuntimeError, 'python value of type \'type\' cannot be used as a value:'):
            import io

            @torch.jit.script
            def wrong_module_attr_lookup():
                return io.BytesIO

    def test_wrong_method_call_inputs(self):
        with self.assertRaisesRegex(RuntimeError, 'argument y not provided'):
            class SomeModule(torch.jit.ScriptModule):

                @torch.jit.script_method
                def foo(self, x, y):
                    return x

                @torch.jit.script_method
                def forward(self, x, y):
                    return self.foo(x)
            SomeModule()

    def test_single_starred_expr_for_loop(self):
        with self.assertRaisesRegex(RuntimeError, 'unexpected expression'):
            cu = torch.jit.CompilationUnit('''
            def test():
                x = 0
                for *a in [1, 2, 3]:
                    x = x + 1
                return x
            ''')

    def test_duplicate(self):
        with self.assertRaisesRegex(RuntimeError, 'Method \'test\' already defined'):
            cu = torch.jit.CompilationUnit('''
            def test():
                return 1

            def test():
                return 2
            ''')

    def test_call_ge(self):
        with self.assertRaisesRegex(RuntimeError, 'expected at most 1 arguments but found 3'):
            @_trace(torch.zeros(1, 2, 3))
            def foo(x):
                return x

            @torch.jit.script
            def test_fn():
                return foo(torch.full([1], 1), torch.full([1], 2), torch.full([1], 3))

    def test_wrong_return_type(self):
        with self.assertRaisesRegex(RuntimeError, 'but instead got value of type tuple'):
            def somefunc():
                # type: () -> Tuple[Tuple[Tensor, Tensor]]
                return torch.zeros(3, 4), torch.zeros(4, 5)

            @torch.jit.script
            def wrong_return_type():
                return somefunc()
            wrong_return_type()

    # Tests for calling between different front-end modes
    def test_call_python_fn_from_tracing_fn(self):
        def python_fn(x):
            return torch.neg(x)

        @_trace(torch.rand(3, 4))
        def traced_fn(x):
            return python_fn(x) + 1

        # The neg op in the python function should be properly inlined to the
        # graph
        self.assertExpected(canonical(traced_fn.graph))

    def test_call_python_mod_from_tracing_fn(self):
        class PythonMod(torch.nn.Module):
            def __init__(self):
                super(PythonMod, self).__init__()
                self.param = torch.nn.Parameter(torch.rand(4, 3), requires_grad=False)

            def forward(self, x):
                return torch.mm(x, self.param)

        pm = PythonMod()

        @_trace(torch.rand(3, 4))
        def traced_fn(x):
            return pm(x) + 1.0

        # Note: the parameter self.param from the Python module is inlined
        # into the graph
        self.assertExpected(canonical(traced_fn.graph))

    def test_call_traced_fn_from_tracing_fn(self):
        @_trace(torch.rand(3, 4))
        def traced_fn1(x):
            return torch.neg(x)

        @_trace(torch.rand(3, 4))
        def traced_fn(x):
            return traced_fn1(x) + 1

        self.assertExpected(canonical(traced_fn.graph))

    def test_call_traced_mod_from_tracing_fn(self):
        class TracedModule(torch.nn.Module):
            def __init__(self):
                super(TracedModule, self).__init__()
                self.param = torch.nn.Parameter(torch.rand(4, 3), requires_grad=False)

            def forward(self, x):
                return torch.mm(x, self.param)

        tm = torch.jit.trace(TracedModule(), torch.rand(3, 4))

        @_trace(torch.rand(3, 4))
        def traced_fn(x):
            return tm(x) + 1.0

        # Note: the parameter self.param from the Python module is inlined
        # into the graph
        self.assertExpected(canonical(traced_fn.graph))

    def test_call_script_fn_from_tracing_fn(self):
        @torch.jit.script
        def script_fn(x):
            return torch.neg(x)

        @_trace(torch.rand(3, 4))
        def traced_fn(x):
            return script_fn(x) + 1

        self.assertExpected(canonical(traced_fn.graph))

    def test_call_script_mod_from_tracing_fn(self):
        class ScriptMod(torch.jit.ScriptModule):
            def __init__(self):
                super(ScriptMod, self).__init__()
                self.param = torch.nn.Parameter(torch.rand(4, 3), requires_grad=False)

            @torch.jit.script_method
            def forward(self, x):
                return torch.mm(x, self.param)

        sm = ScriptMod()

        @_trace(torch.rand(3, 4))
        def traced_fn(x):
            return sm(x) + 1.0

        self.assertExpected(canonical(traced_fn.graph))

    def test_call_python_fn_from_traced_module(self):
        def python_fn(x):
            return torch.neg(x)

        class TracedModule(torch.nn.Module):
            def __init__(self):
                super(TracedModule, self).__init__()
                self.param = torch.nn.Parameter(torch.rand(4, 3))

            def forward(self, x):
                return torch.mm(python_fn(x), self.param)

        tm = torch.jit.trace(TracedModule(), torch.rand(3, 4))

        # Note: parameter self.param from the traced module should appear as
        # an input to the graph and the neg op from the Python function should
        # be properly inlined
        self.assertExpected(canonical(tm.graph))

    def test_call_python_mod_from_traced_module(self):
        class PythonModule(torch.nn.Module):
            def __init__(self):
                super(PythonModule, self).__init__()
                self.param = torch.nn.Parameter(torch.rand(5, 7))

            def forward(self, x):
                return torch.mm(x, self.param)

        class TracedModule(torch.nn.Module):
            def __init__(self):
                super(TracedModule, self).__init__()
                self.param = torch.nn.Parameter(torch.rand(4, 5))
                self.mod = PythonModule()

            def forward(self, x):
                return self.mod(torch.mm(x, self.param)) + 1.0

        tm = torch.jit.trace(TracedModule(), torch.rand(3, 4))

        # Note: the parameters from both modules should appear in the flattened
        # inputs of the graph. All ops from both modules should be inlined.
        self.assertExpected(canonical(tm.graph))

    def test_call_traced_fn_from_traced_module(self):
        @_trace(torch.rand(3, 4))
        def traced_fn(x):
            return torch.neg(x)

        class TracedModule(torch.nn.Module):
            def __init__(self):
                super(TracedModule, self).__init__()
                self.param = torch.nn.Parameter(torch.rand(4, 5))

            def forward(self, x):
                return traced_fn(torch.mm(x, self.param))

        tm = torch.jit.trace(TracedModule(), torch.rand(3, 4))
        # Note: neg op from the traced function should be properly inlined
        self.assertExpected(canonical(tm.graph))

    def test_trace_hierarchy(self):
        # Test that we preserve the module hierarchy for a ScriptModule
        # submodule during tracing

        class AnotherScriptMod(torch.jit.ScriptModule):
            def __init__(self):
                super(AnotherScriptMod, self).__init__()
                self.param = torch.nn.Parameter(torch.rand(1, 2, 3))

            @torch.jit.script_method
            def bar(self):
                return torch.zeros(4, 5)

        class SomeScriptMod(torch.jit.ScriptModule):
            def __init__(self):
                super(SomeScriptMod, self).__init__()
                self.asm = AnotherScriptMod()

            @torch.jit.script_method
            def foo(self):
                return torch.zeros(3, 4)

            @torch.jit.script_method
            def bar(self):
                return torch.zeros(4, 3)

        class TraceMe(torch.nn.Module):
            def __init__(self):
                super(TraceMe, self).__init__()
                self.ssm = SomeScriptMod()

            def forward(self, x):
                return self.ssm.bar() + x

        orig = TraceMe()
        traced = torch.jit.trace(orig, (torch.rand(4, 3, dtype=torch.float),))
        # for each of these checks, check that *BOTH* the underlying
        # _C.ScriptModule object has the expected method/param, as well as the
        # Python object that wraps it.
        self.assertTrue(traced.ssm._has_method('foo'))
        self.assertTrue(hasattr(traced.ssm, 'foo'))

        imported = self.getExportImportCopy(traced)

        self.assertTrue(imported.ssm._has_method('foo'))
        self.assertTrue(hasattr(imported.ssm, 'foo'))

        self.assertTrue(imported.ssm.asm._has_method('bar'))
        self.assertTrue(hasattr(imported.ssm.asm, 'bar'))

        self.assertTrue(imported.ssm.asm._has_parameter('param'))
        self.assertTrue(hasattr(imported.ssm.asm, 'param'))

    def test_trace_parameter(self):
        class Param(nn.Module):
            def __init__(self):
                super(Param, self).__init__()
                self.register_parameter("bias", nn.Parameter(torch.Tensor(4, 4)))

            def forward(self, x):
                return x

        class M3(torch.jit.ScriptModule):
            def __init__(self, model):
                super(M3, self).__init__(False)
                self.traced = torch.jit.trace(model, (torch.rand(3, 3)))

            @torch.jit.script_method
            def forward(self, x):
                return self.traced(x)

        class M2(nn.Module):
            def __init__(self, model):
                super(M2, self).__init__()
                self.module = M3(model)

            def forward(self, x):
                return self.module(x)

        class M1(torch.jit.ScriptModule):
            def __init__(self, model):
                super(M1, self).__init__(False)
                self.traced = torch.jit.trace(M2(model), (torch.rand(3, 3)))

            @torch.jit.script_method
            def forward(self, x):
                return self.traced(x)

        module = M1(Param())
        f = io.BytesIO()
        torch.jit.save(module, f)

    def test_call_traced_module_from_traced_module(self):
        class TracedModule1(torch.nn.Module):
            def __init__(self):
                super(TracedModule1, self).__init__()
                self.param = torch.nn.Parameter(torch.rand(5, 7))

            def forward(self, x):
                return torch.mm(x, self.param)

        class TracedModule(torch.nn.Module):
            def __init__(self):
                super(TracedModule, self).__init__()
                self.param = torch.nn.Parameter(torch.rand(4, 5))
                self.mod = torch.jit.trace(TracedModule1(), torch.rand(3, 5))

            def forward(self, x):
                return self.mod(torch.mm(x, self.param)) + 1.0

        tm = torch.jit.trace(TracedModule(), torch.rand(3, 4))

        # Note: the parameters from both modules should appear in the flattened
        # inputs of the graph. All ops from both modules should be inlined.
        self.assertExpected(canonical(tm.graph))

    def test_call_script_fn_from_traced_module(self):
        @torch.jit.script
        def traced_fn(x):
            return torch.neg(x)

        class TracedModule(torch.nn.Module):
            def __init__(self):
                super(TracedModule, self).__init__()
                self.param = torch.nn.Parameter(torch.rand(4, 5))

            def forward(self, x):
                return traced_fn(torch.mm(x, self.param))

        tm = torch.jit.trace(TracedModule(), torch.rand(3, 4))
        # Note: neg op from the script function should be properly inlined
        self.assertExpected(canonical(tm.graph))

    def test_call_script_module_from_traced_module(self):
        class ScriptMod(torch.jit.ScriptModule):
            def __init__(self):
                super(ScriptMod, self).__init__()
                self.param_foo = torch.nn.Parameter(torch.rand(5, 7))

            @torch.jit.script_method
            def forward(self, x):
                return torch.mm(x, self.param_foo)

        class TracedModule(torch.nn.Module):
            def __init__(self):
                super(TracedModule, self).__init__()
                self.param = torch.nn.Parameter(torch.rand(4, 5))
                self.mod = ScriptMod()

            def forward(self, x):
                return self.mod(torch.mm(x, self.param)) + 1.0

        tm = torch.jit.trace(TracedModule(), torch.rand(3, 4))

        # Note: the parameters from both modules should appear in the flattened
        # inputs of the graph. All ops from both modules should be inlined.
        self.assertExpected(canonical(tm.graph))

    def test_call_python_fn_from_script_fn(self):
        def python_fn(x):
            return torch.neg(x)

        @torch.jit.script
        def script_fn(x):
            return python_fn(x) + 1

        # Note: the call to python_fn appears as `^python_fn()` and is called
        # as a PythonOp in the interpreter
        self.assertExpected(canonical(script_fn.graph))

    def test_call_python_mod_from_script_fn(self):
        class PythonModule(torch.nn.Module):
            def __init__(self):
                super(PythonModule, self).__init__()
                self.param = torch.nn.Parameter(torch.rand(5, 7))

            def forward(self, x):
                return torch.mm(x, self.param)

        pm = PythonModule()

        @torch.jit.script
        def script_fn(x):
            return pm(x) + 1

        # Note: call to pm(x) appears as ^<python_value>() in the trace.
        # Parameters are NOT inlined.
        self.assertExpected(str(script_fn.graph))

    def test_call_traced_fn_from_script_fn(self):
        @_trace(torch.rand(3, 4))
        def traced_fn(x):
            return torch.neg(x)

        @torch.jit.script
        def script_fn(x):
            return traced_fn(x) + 1

        # Note: the neg op from traced_fn should be properly inlined into the
        # script function's graph
        self.assertExpected(str(script_fn.graph))

    def test_call_traced_mod_from_script_fn(self):
        class TracedModule(torch.nn.Module):
            def __init__(self):
                super(TracedModule, self).__init__()

            def forward(self, x):
                return torch.mm(x, torch.zeros(4, 3))

        tm = torch.jit.trace(TracedModule(), torch.rand(3, 4))

        @torch.jit.script
        def script_fn(x):
            return tm(x) + 1

        self.assertExpected(str(script_fn.graph))

    def test_call_script_fn_from_script_fn(self):
        @torch.jit.script
        def script_fn1(x):
            return torch.neg(x)

        @torch.jit.script
        def script_fn(x):
            return script_fn1(x) + 1

        # Note: the neg op from script_fn1 should be properly inlined into the
        # graph of script_fn
        self.assertExpected(canonical(script_fn.graph))

    def test_call_script_mod_from_script_fn(self):
        class ScriptMod(torch.jit.ScriptModule):
            def __init__(self):
                super(ScriptMod, self).__init__()

            @torch.jit.script_method
            def forward(self, x):
                return torch.mm(x, torch.zeros([4, 3]))

        sm = ScriptMod()

        @torch.jit.script
        def script_fn(x):
            return sm(x) + 1

        self.assertExpected(canonical(script_fn.graph))

    def test_call_python_fn_from_script_module(self):
        def python_fn(x):
            return torch.neg(x)

        class ScriptMod(torch.jit.ScriptModule):
            def __init__(self):
                super(ScriptMod, self).__init__()
                self.param = torch.nn.Parameter(torch.rand(4, 3))

            @torch.jit.script_method
            def forward(self, x):
                return python_fn(torch.mm(x, self.param))

        sm = ScriptMod()
        self.assertExpected(str(sm.__getattr__('forward').graph))

    def test_call_python_mod_from_script_module(self):
        class PythonMod(torch.nn.Module):
            def __init__(self):
                super(PythonMod, self).__init__()
                self.param = torch.nn.Parameter(torch.rand(3, 5))

            def forward(self, x):
                return torch.mm(x, self.param)

        class ScriptMod(torch.jit.ScriptModule):
            def __init__(self):
                super(ScriptMod, self).__init__()
                self.param = torch.nn.Parameter(torch.rand(4, 3))
                self.pm = PythonMod()

            @torch.jit.script_method
            def forward(self, x):
                return self.pm(torch.mm(x, self.param))

        sm = ScriptMod()
        # Note: the call into PythonMod appears as ^<python_value>(). Parameters
        # are NOT inlined
        self.assertExpected(str(sm.graph))

    def test_call_tracing_fn_from_script_module(self):
        @_trace(torch.rand(3, 3))
        def traced_fn(x):
            return torch.neg(x)

        class ScriptMod(torch.jit.ScriptModule):
            def __init__(self):
                super(ScriptMod, self).__init__()
                self.param = torch.nn.Parameter(torch.rand(4, 3))

            @torch.jit.script_method
            def forward(self, x):
                return traced_fn(torch.mm(x, self.param))

        sm = ScriptMod()
        self.assertExpected(str(sm.__getattr__('forward').graph))

    def test_call_tracing_mod_from_script_module(self):
        class TracedMod(torch.nn.Module):
            def __init__(self):
                super(TracedMod, self).__init__()
                self.param = torch.nn.Parameter(torch.rand(3, 5))

            def forward(self, x):
                return torch.mm(x, self.param)

        class ScriptMod(torch.jit.ScriptModule):
            def __init__(self):
                super(ScriptMod, self).__init__()
                self.param = torch.nn.Parameter(torch.rand(4, 3))
                self.tm = torch.jit.trace(TracedMod(), torch.rand(3, 3))

            @torch.jit.script_method
            def forward(self, x):
                return self.tm(torch.mm(x, self.param))

        sm = ScriptMod()
        # Note: the parameters from both modules should appear in the flattened
        # input list to the graph. The mm op from TracedMod should be properly
        # inlined
        self.assertExpected(str(sm.graph))

    def test_call_script_fn_from_script_module(self):
        @torch.jit.script
        def script_fn(x):
            return torch.neg(x)

        class ScriptMod(torch.jit.ScriptModule):
            def __init__(self):
                super(ScriptMod, self).__init__()
                self.param = torch.nn.Parameter(torch.rand(4, 3))

            @torch.jit.script_method
            def forward(self, x):
                return script_fn(torch.mm(x, self.param))

        sm = ScriptMod()
        self.assertExpected(canonical(sm.__getattr__('forward').graph))

    def test_call_script_mod_from_script_module(self):
        class ScriptMod1(torch.jit.ScriptModule):
            def __init__(self):
                super(ScriptMod1, self).__init__()
                self.param = torch.nn.Parameter(torch.rand(3, 5))

            @torch.jit.script_method
            def forward(self, x):
                return torch.mm(x, self.param)

        class ScriptMod(torch.jit.ScriptModule):
            def __init__(self):
                super(ScriptMod, self).__init__()
                self.param = torch.nn.Parameter(torch.rand(4, 3))
                self.tm = ScriptMod1()

            @torch.jit.script_method
            def forward(self, x):
                return self.tm(torch.mm(x, self.param))

        sm = ScriptMod()
        # Note: the parameters from both modules should appear in the flattened
        # input list to the graph. The mm op from ScriptMod1 should be properly
        # inlined
        self.assertExpected(canonical(sm.graph))

    def test_module_with_params_called_fails(self):
        with self.assertRaisesRegex(RuntimeError, "Attempted to inline a Module with parameters. Stateful "
                                                  "modules to be inlined must be submodules of the callee."):
            class ScriptMod(torch.jit.ScriptModule):
                def __init__(self):
                    super(ScriptMod, self).__init__()
                    self.param = torch.nn.Parameter(torch.rand(3, 3))

                @torch.jit.script_method
                def forward(self, x):
                    return torch.mm(x, self.param)

            sm = ScriptMod()

            @torch.jit.script
            def some_func(x):
                return sm(x)

    def test_index_put_trace_with_view(self):
        @_trace(torch.rand(100), torch.tensor([1, 2, 3, 4]), torch.rand(1, 1, 1, 4))
        def test_index_put(target, indices, rhs):
            target[indices] = rhs
            return target

        self.assertExpectedGraph(test_index_put.graph)

    def test_index_put_trace_without_view(self):
        @_trace(torch.rand(100), torch.tensor([1, 2, 3, 4]), torch.rand(4))
        def test_index_put(target, indices, rhs):
            target[indices] = rhs
            return target

        self.assertExpectedGraph(test_index_put.graph)

    def test_tuple_indexing(self):
        def tuple_index(a):
            if bool(a):
                b = (1, 2)
            else:
                b = (0, 2)
            return b[-2], b[1]

        self.checkScript(tuple_index, (torch.tensor([1]),))
        self.checkScript(tuple_index, (torch.tensor([1]),), optimize=True)
        tuple_comp = torch.jit.script(tuple_index)
        self.assertExpectedGraph(tuple_comp.graph)
        self.assertEqual(tuple_comp(torch.tensor(1)), (1, 2))

        with self.assertRaisesRegex(RuntimeError, "tuple indices must be integer constants"):
            @torch.jit.script
            def test_non_constant_input(a):
                if bool(a):
                    b = 1
                else:
                    b = 0
                c = (0, 1)
                return c[b]

        def test_indexing_float():
            c = (1, 2)
            return c[0.1]
        self.checkScriptRaisesRegex(test_indexing_float, (), Exception,
                                    "tuple indices must")

        def test_indexing_out_of_bounds_pos():
            c = (1, 2)
            return c[2]

        self.checkScriptRaisesRegex(test_indexing_out_of_bounds_pos, (), Exception,
                                    "out of range")

        def test_indexing_out_of_bounds_neg():
            c = (1, 2)
            return c[-3]

        self.checkScriptRaisesRegex(test_indexing_out_of_bounds_pos, (), Exception,
                                    "out of range")

    def test_namedtuple_attr(self):
        def f(x):
            return x.max(dim=1).indices + torch.max(x, dim=1).indices

        self.checkScript(f, (torch.rand(20, 20, 20),), optimize=True)

        with self.assertRaisesRegex(RuntimeError, "Unknown attribute to named tuple"):
            @torch.jit.script
            def g1(x):
                return x.max(dim=1).unknown_symbol

        with self.assertRaisesRegex(RuntimeError, "Getting attributes of tuples is not supported"):
            @torch.jit.script
            def g2(x):
                print((x, x, x).__doc__)
                return x

    def test_tuple_slicing(self):
        def tuple_slice(a):
            if bool(a):
                b = (1, 2, 3, 4)
            else:
                b = (4, 3, 2, 1)
            c = b[-4:4]
            d = b[0:]
            e = c[1:-1]
            return e

        self.checkScript(tuple_slice, (torch.tensor([1]),), optimize=True)
        tuple_graph = torch.jit.script(tuple_slice)
        self.assertExpectedGraph(tuple_graph.graph)
        self.run_pass('lower_all_tuples', tuple_graph.graph)
        self.assertTrue('Tuple' not in str(tuple_graph.graph))
        tuple_comp = torch.jit.script(tuple_slice)
        self.assertEqual(tuple_comp(torch.tensor(1)), (2, 3))

        @torch.jit.script
        def test_indexing_end_out_of_bounds():
            c = (1, 2)
            return c[2:10]

        self.assertEqual(test_indexing_end_out_of_bounds(), ())

    def test_unwrap_optional_builtin(self):
        def test(x):
            # type: (Optional[int]) -> int
            x = torch.jit._unwrap_optional(x)
            x = x + x
            return x

        self.checkScript(test, (3,))

        with self.assertRaisesRegex(AssertionError, "Unwrapping null optional"):
            test(None)

        test_script = torch.jit.script(test)
        with self.assertRaisesRegex(RuntimeError, "Unwrapping null optional"):
            test_script(None)

        @torch.jit.script
        def test_test():
            return torch.jit._unwrap_optional(1)

        with self.assertRaisesRegex(RuntimeError, "cannot match an Optional\\[T\\] to None"):
            @torch.jit.script
            def test_no_type():
                # type: () -> int
                return torch.jit._unwrap_optional(None)

    def test_indexing_error(self):
        with self.assertRaisesRegex(RuntimeError, "only supported on lists, dictionaries, tensors, and tuples"):
            @torch.jit.script
            def test_wrong_type():
                a = 8
                return a[0]

    def test_annotated_script_fn(self):
        @torch.jit.script
        def foo(x, y, z):
            # type: (Tensor, Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, Tuple[Tensor, Tensor]]) -> Tensor
            return x

        self.assertExpected(foo.__getattr__('forward').pretty_print_schema())

    def test_annotated_script_method(self):
        class SM(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x, y):
                # type: (Tuple[Tensor, Tensor], Tensor) -> Tuple[Tensor, Tensor, Tensor]
                return y, y, y

        sm = SM()

        self.assertExpected(sm.__getattr__('forward').pretty_print_schema())

    def test_annotated_script_fn_return_mismatch(self):
        with self.assertRaisesRegex(RuntimeError, "but is actually of type"):
            @torch.jit.script
            def return_tup(x):
                # type: (Tensor) -> Tuple[Tuple[Tensor, Tensor], Tensor]
                return x, x

    def test_annotated_script_fn_arg_mismatch(self):
        with self.assertRaisesRegex(RuntimeError, r"arguments for call are not valid"):
            @torch.jit.script
            def tuple_arg(x):
                # type: (Tuple[Tensor, Tensor]) -> Tensor
                return x + 1

    def test_script_non_tensor_args_outputs(self):
        @torch.jit.script
        def fn(x, y):
            # type: (Tensor, float) -> float
            return float((x + y).sum())

        x = torch.ones(2, 2)
        z = fn(x, 1)
        self.assertIsInstance(z, float)
        self.assertEqual(z, 8.)

    @unittest.skip('https://github.com/pytorch/pytorch/issues/9595')
    def test_inline_and_run_annotated_script_fn(self):
        @torch.jit.script
        def to_inline(x, y):
            # type: (Tuple[Tensor, Tensor], Tensor) -> Tensor
            return y

        @torch.jit.script
        def some_func(x):
            return to_inline((x, x), x)

        x = torch.rand(3, 4)
        self.assertEqual(some_func(x), x)

    def test_file_format_serialization(self):
        import tempfile
        filename = tempfile.mktemp()
        writer = torch._C.PyTorchFileWriter(filename)
        import os
        import random
        buffers = [os.urandom(size) for size in [random.randint(1, 100) for i in range(20)]]
        offsets = []
        for i, buf in enumerate(buffers):
            writer.write_record(str(i), buf, len(buf))
            offsets.append(i)
        import pickle
        serialized_offsets = pickle.dumps(offsets)
        writer.write_record("meta", serialized_offsets, len(serialized_offsets))
        writer.write_end_of_file()

        reader = torch._C.PyTorchFileReader(filename)
        serialized_offsets_read = reader.get_record("meta")
        parsed_serialized_offsets = pickle.loads(serialized_offsets)

        for i, offset in enumerate(parsed_serialized_offsets):
            data = reader.get_record(str(offset))
            assert(data == buffers[i])

    # for each type, the input type annotation and corresponding return type annotation
    def type_input_return_pairs(self):
        return [
            ('Tensor', 'Tensor'),
            ('torch.Tensor', 'Tensor'),
            ('str', 'str'),
            ('int', 'int'),
            ('bool', 'bool'),
            ('BroadcastingList3[float]', 'List[float]'),
            ('BroadcastingList2[int]', 'List[int]'),
            ('List[int]', 'List[int]'),
            ('Optional[int]', 'Optional[int]'),
        ]

    # replacing code input & return type pair
    def format_code(self, code, pair):
        return code.format(input=pair[0], output=pair[1])

    # ***** Type annotation tests ****
    # Test combinations of:
    # {String frontend, Python AST Frontend}
    # {Python 3-style type annotations, MyPy-style type comments}
    # {Script method, Script function}

    #  String frontend , Python 3-style type annotations , Script function
    def test_annot_string_py3_fn(self):
        code = '''
            def foo(x : {input}, y : Tuple[Tensor, Tensor]) -> Tuple[{output}, {output}]:
                return x, x
        '''
        test_str = []
        for pair in self.type_input_return_pairs():
            cu = torch.jit.CompilationUnit(self.format_code(code, pair))
            test_str.append(cu.__getattr__('foo').pretty_print_schema())
        self.assertExpected("\n".join(test_str))

    #  String frontend , Python 3-style type annotations , Script method
    def test_annot_string_py3_method(self):
        class TestModule(torch.jit.ScriptModule):
            def __init__(self):
                super(TestModule, self).__init__()

        code = '''
            def foo(self, x : {input}, y : Tuple[Tensor, Tensor]) -> Tuple[{output}, {output}]:
                return x, x
        '''
        test_str = []
        for pair in self.type_input_return_pairs():
            tm = TestModule()
            tm.define(self.format_code(code, pair))
            test_str.append(tm.__getattr__('foo').pretty_print_schema())
        self.assertExpected("\n".join(test_str))

    #  String frontend , MyPy-style type comments , Script function
    def test_annot_string_mypy_fn(self):
        code = '''
            def foo(x, y):
                # type: ({input}, Tuple[Tensor, Tensor]) -> Tuple[{output}, {output}]
                return x, x
        '''
        test_str = []
        for pair in self.type_input_return_pairs():
            cu = torch.jit.CompilationUnit(self.format_code(code, pair))
            test_str.append(cu.__getattr__('foo').pretty_print_schema())
        self.assertExpected("\n".join(test_str))

    #  String frontend , MyPy-style type comments , Script method
    def test_annot_string_mypy_method(self):
        class TestModule(torch.jit.ScriptModule):
            def __init__(self):
                super(TestModule, self).__init__()

        code = '''
        def foo(self, x, y):
            # type: ({input}, Tuple[Tensor, Tensor]) -> Tuple[{output}, {output}]
            return x, x
        '''

        test_str = []
        for pair in self.type_input_return_pairs():
            tm = TestModule()
            tm.define(self.format_code(code, pair))
            test_str.append(tm.__getattr__('foo').pretty_print_schema())
        self.assertExpected("\n".join(test_str))

    # Helper function to eval Python3 code without causing a syntax error for
    # this file under py2
    def _get_py3_code(self, code, fn_name):
        with tempfile.TemporaryDirectory() as tmp_dir:
            script_path = os.path.join(tmp_dir, 'script.py')
            with open(script_path, 'w') as f:
                f.write(code)
            import importlib.util
            spec = importlib.util.spec_from_file_location(fn_name, script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            fn = getattr(module, fn_name)
            return fn

    #  Python AST Frontend , Python 3-style type annotations , Script function
    @unittest.skipIf(not PY35, "Python 3.5 needed")
    def test_annot_ast_py3_fn(self):
        code = dedent('''
            from typing import Tuple, List, Optional
            from torch import Tensor
            from torch.jit.annotations import BroadcastingList2, BroadcastingList3
            import torch
            @torch.jit.script
            def foo(x : {input}, y : Tuple[Tensor, Tensor]) -> Tuple[{output}, {output}]:
                return x, x
        ''')
        test_str = []
        for pair in self.type_input_return_pairs():
            fn = self._get_py3_code(self.format_code(code, pair), 'foo')
            test_str.append(fn.__getattr__('forward').pretty_print_schema())
        self.assertExpected("\n".join(test_str))

    #  Python AST Frontend , Python 3-style type annotations , Script method
    @unittest.skipIf(not PY35, "Python 3.5 needed")
    def test_annot_ast_py3_method(self):
        code = dedent('''
            from typing import Tuple, List, Optional
            from torch import Tensor
            from torch.jit.annotations import BroadcastingList2, \\
                BroadcastingList3
            import torch
            class FooModule(torch.jit.ScriptModule):
                @torch.jit.script_method
                def foo(self, x : {input}, y : Tuple[Tensor, Tensor]) -> Tuple[{output}, {output}]:
                    return x, x
            instance = FooModule()
        ''')

        test_str = []
        for pair in self.type_input_return_pairs():
            fn = self._get_py3_code(self.format_code(code, pair), 'instance')
            test_str.append(fn.__getattr__('foo').pretty_print_schema())
        self.assertExpected("\n".join(test_str))

    #  Python AST Frontend , MyPy-style type comments , Script function
    @unittest.skipIf(not PY35, "Python 3.5 needed")
    def test_annot_ast_mypy_fn(self):
        code = dedent('''
            import torch
            @torch.jit.script
            def foo(x, y):
                # type: ({input}, Tuple[Tensor, Tensor]) -> Tuple[{output}, {output}]
                return x, x
        ''')

        test_str = []
        for pair in self.type_input_return_pairs():
            fn = self._get_py3_code(self.format_code(code, pair), 'foo')
            test_str.append(fn.__getattr__('forward').pretty_print_schema())
        self.assertExpected("\n".join(test_str))

    #  Python AST Frontend , MyPy-style type comments , Script method
    @unittest.skipIf(not PY35, "Python 3.5 needed")
    def test_annot_ast_mypy_method(self):
        code = dedent('''
            import torch
            class FooModule(torch.jit.ScriptModule):
                @torch.jit.script_method
                def foo(self, x, y):
                    # type: ({input}, Tuple[Tensor, Tensor]) -> Tuple[{output}, {output}]
                    return x, x
            instance = FooModule()
        ''')

        test_str = []
        for pair in self.type_input_return_pairs():
            fn = self._get_py3_code(self.format_code(code, pair), 'instance')
            test_str.append(fn.__getattr__('foo').pretty_print_schema())
        self.assertExpected("\n".join(test_str))

    def test_method_casts_script(self):
        cast_types = [
            'byte', 'char', 'double', 'float', 'int', 'long', 'short'
        ]

        for cast_type in cast_types:
            cu = torch.jit.CompilationUnit('''
            def cast_to(x):
                return x.{cast_type}()
            '''.format(cast_type=cast_type))

            x = torch.rand(3, 4, 5) * 128
            cu_result = cu.cast_to(x)
            reference = getattr(x, cast_type)()
            self.assertEqual(cu_result, reference)

    def test_listconstruct_erasure(self):
        class FooMod(torch.nn.Module):
            def forward(self, x):
                mask = x < 0.0
                return x[mask]

        import io
        f = io.BytesIO()
        self.assertExpected(torch.onnx.export_to_pretty_string(
            FooMod(), (torch.rand(3, 4),), f,
            operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK))

    def test_trace_checker_arange_as_constant(self):
        with self.assertRaisesRegex(torch.jit.TracingCheckError, r'Graphs differed across invocations!'):
            @_trace(torch.rand(3, 4), check_inputs=[(torch.rand(4, 5),)])
            def foo(x):
                y = torch.arange(0, x.shape[0]).double()
                return x + y.unsqueeze(1)

    @suppress_warnings
    def test_trace_checker_dot_data(self):
        with self.assertRaisesRegex(torch.jit.TracingCheckError, r'Tensor-valued Constant nodes differed in value '
                                                                 r'across invocations'):
            @_trace(torch.rand(3, 4), check_inputs=[(torch.rand(3, 4),)])
            def foo(x):
                y = x.data
                return x + y

    @suppress_warnings
    def test_trace_checker_control_flow(self):
        def foo(x):
            for _ in range(x.size(0)):
                x = torch.neg(x)
            return x

        with self.assertRaisesRegex(torch.jit.TracingCheckError, r'Graphs differed across invocations!'):
            torch.jit.trace(foo, torch.randn(3, 4), check_inputs=[torch.randn(4, 4)])

    @suppress_warnings
    def test_trace_checker_memoization(self):
        with self.assertRaisesRegex(torch.jit.TracingCheckError, r'Graphs differed across invocations!'):
            def foo(x):
                if not hasattr(foo, 'cache'):
                    foo.cache = torch.neg(x)
                return x + foo.cache

            traced = torch.jit.trace(foo, torch.rand(3, 4), check_inputs=[(torch.rand(3, 4),)])

    # These tests don't work because UBSAN has a false positive about accessing
    # out of bounds on a dynamically sized struct internal to asmjit
    if not TEST_WITH_UBSAN and torch.fbgemm_is_cpu_supported():
        def test_int8_quantization_module(self):
            K1, N1 = 2, 2

            class FooBar(torch.nn.Module):
                def __init__(self):
                    super(FooBar, self).__init__()
                    self.linear1 = torch.nn.Linear(K1, N1).float()

                def forward(self, x):
                    x = self.linear1(x)
                    return x

            fb = FooBar()
            fb.linear1.weight = torch.nn.Parameter(
                torch.tensor([[-150, 100], [100, -150]], dtype=torch.float), requires_grad=False)
            fb.linear1.bias = torch.nn.Parameter(torch.zeros_like(fb.linear1.bias), requires_grad=False)
            fb_ref = FooBar()
            fb_ref.linear1.weight = torch.nn.Parameter(fb.linear1.weight.clone(), requires_grad=False)
            fb_ref.linear1.bias = torch.nn.Parameter(fb.linear1.bias.clone(), requires_grad=False)
            fb = torch.jit.quantized.quantize_linear_modules(fb)

            x = (torch.rand(1, K1).float() - 0.5) / 10.0
            traced = torch.jit.trace(fb, (x,))
            fb = self.getExportImportCopyWithPacking(traced)

            x = torch.tensor([[100, -150]], dtype=torch.float)
            y = fb(x)
            y_ref = fb_ref(x)
            torch.testing.assert_allclose(y, y_ref, rtol=0.0001, atol=1e-3)

    def checkTracerWarning(self, *args, **kwargs):
        with warnings.catch_warnings(record=True) as warns:
            torch.jit.trace(*args, **kwargs)
        self.assertGreater(len(warns), 0)
        for warn in warns:
            self.assertIn("cause the trace to be incorrect", str(warn.message))

    def test_trace_checker_slice_lhs(self):
        def foo(x):
            for i in range(3):
                x[i, :] = torch.zeros(4)
            return x

        self.checkTrace(foo, (torch.rand(3, 4),))

    def test_trace_checker_inplace_on_view(self):
        def foo(x):
            x.view(-1).add_(-x.view(-1))
            return x

        self.assertWarnsRegex(lambda: torch.jit.trace(foo,
                                                      torch.rand(3, 4),
                                                      check_inputs=[torch.rand(5, 6)],
                                                      _force_outplace=True),
                              'Output nr 1. of the traced function does not match the '
                              'corresponding output of the Python function')

    def test_lhs_index_fails(self):
        def foo(x):
            x[0, 1] = 4
            return x
        self.checkTracerWarning(foo, torch.rand(3, 4), _force_outplace=True)

    def test_lhs_index_trivial(self):
        def foo(y, x):
            y[...] = x
            return y
        self.checkTrace(foo, (torch.rand(3, 4), torch.rand(4)), inputs_require_grads=False)

    def test_inplace_warn(self):
        def foo(x):
            x.view(-1).add_(-x.view(-1))
            return x
        self.checkTracerWarning(foo, torch.rand(3, 4), _force_outplace=True)

    @suppress_warnings
    def test_trace_checker_dropout_train(self):
        def foo(x):
            return torch.dropout(x, p=0.5, train=True)

        self.assertWarnsRegex(lambda: torch.jit.trace(foo, torch.rand(3, 4), check_inputs=[torch.rand(5, 6)]),
                              'Output nr 1. of the traced function does not match the '
                              'corresponding output of the Python function')
        self.assertWarnsRegex(lambda: torch.jit.trace(foo, torch.rand(3, 4), check_inputs=[torch.rand(5, 6)]),
                              'Trace had nondeterministic nodes')

    def test_trace_checker_dropout_notrain(self):
        input = torch.rand(3, 4)

        @_trace(input)
        def foo(x):
            return torch.dropout(x, p=0.5, train=False)

        self.assertEqual(foo(input), input)

    def test_export_dynamic_slice(self):
        class DynamicSliceExportMod(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                retval = x[0]
                for i in range(x.size(1)):
                    retval += torch.sum(x[0:i], dim=0)
                return retval

        mod = DynamicSliceExportMod()

        input = torch.rand(3, 4, 5)
        example_outs = mod(input)

        f = io.BytesIO()
        exported = torch.onnx.export_to_pretty_string(
            DynamicSliceExportMod(), (input,), f, example_outputs=example_outs)
        self.assertExpected(exported)

    def test_string_frontend_elif(self):
        code = '''
            def elif_test(niter : int):
                rv = 0
                for i in range(niter):
                    if i % 3 == 0 and i % 5 == 0:
                        rv += 35
                    elif i % 3 == 0:
                        rv += 3
                    elif i % 5 == 0:
                        rv += 5
                    else:
                        rv += i
                return rv
        '''

        self.checkScript(code, (101,), name='elif_test', outputs=3028)

    def test_addmm_fusion(self):
        class AddmmWrapper(torch.nn.Module):
            def forward(self, x, y, c):
                return torch.mm(x, y) + c

        # Test addmm fusion is disabled for normal Jit
        x, y, c = torch.rand(3, 4), torch.rand(4, 5), torch.rand(3, 5)
        f = io.BytesIO()
        pretty = torch.onnx.export_to_pretty_string(AddmmWrapper(), (x, y, c), f)
        self.assertExpected(pretty, 'onnx')

        jit_trace = torch.jit.trace(AddmmWrapper(), (x, y, c))
        ge_graph = jit_trace.__getattr__('forward').graph_for(x, y, c)
        self.assertExpectedGraph(ge_graph, 'jit')

    def test_pyop_exception_message(self):
        class Foo(torch.jit.ScriptModule):
            def __init__(self):
                super(Foo, self).__init__()
                self.conv = nn.Conv2d(1, 10, kernel_size=5)

            @torch.jit.script_method
            def forward(self, x):
                return self.conv(x)
        foo = Foo()
        # testing that the correct error message propagates
        with self.assertRaisesRegex(RuntimeError, "Expected 4-dimensional input for 4-dimensional weight"):
            foo(torch.ones([123]))  # wrong size

    def test_builtin_error_messsage(self):
        from torch.nn.modules.utils import _single, _pair, _triple, _quadruple

        with self.assertRaisesRegex(RuntimeError, "aten::masked_fill_"):
            @torch.jit.script
            def close_match(x):
                return x.masked_fill(True)

        with self.assertRaisesRegex(RuntimeError, "This op may not exist or may not be currently "
                                    "supported in TorchScript"):
            @torch.jit.script
            def unknown_op(x):
                torch.set_grad_enabled(True)
                return x

    def test_exceptions(self):
        cu = torch.jit.CompilationUnit('''
            def foo(cond):
                if bool(cond):
                    raise ValueError(3)
                return 1
        ''')

        cu.foo(torch.tensor(0))
        with self.assertRaisesRegex(torch.jit.Error, "Exception"):
            cu.foo(torch.tensor(1))

        @torch.jit.script
        def foo(cond):
            a = 3
            if bool(cond):
                raise ArbitraryError(a, "hi")
                if False:
                    raise ArbitraryError
            return a

        foo(torch.tensor(0))
        # we don't currently validate the name of the exception
        with self.assertRaisesRegex(torch.jit.Error, "Exception"):
            foo(torch.tensor(1))

        @torch.jit.script
        def foo_except_used():
            a = Exception()
            print(a)
            raise a

        # a not DCEd
        with self.assertRaisesRegex(RuntimeError, "expected value of type Tensor"):
            foo_except_used()

        # We don't validate the expr following raise
        @torch.jit.script
        def foo():
            raise 3 + 4

        # no control flow analysis yet
        with self.assertRaisesRegex(RuntimeError, "undefined value a"):
            @torch.jit.script
            def foo():
                if True:
                    a = 1
                else:
                    raise Exception("Hi")
                return a

    def test_assertions(self):
        cu = torch.jit.CompilationUnit('''
            def foo(cond):
                assert bool(cond), "hi"
                return 0
        ''')

        cu.foo(torch.tensor(1))
        with self.assertRaisesRegex(torch.jit.Error, "Exception"):
            cu.foo(torch.tensor(0))

        @torch.jit.script
        def foo(cond):
            assert bool(cond), "hi"

        foo(torch.tensor(1))
        # we don't currently validate the name of the exception
        with self.assertRaisesRegex(torch.jit.Error, "Exception"):
            foo(torch.tensor(0))

    def test_weak_script_function(self):
        outer_var = 10
        outer_var2 = 11

        def not_a_script_fn(x):
            return x + 2

        @torch.jit.script
        def even_more_inner(x):
            return x + 1

        @torch.jit.script
        def inner(x):
            return not_a_script_fn(x) + x + even_more_inner(x)

        @torch.jit.script
        def strong_script_fn(x):
            if bool(x.norm() > 2):
                x = x + 3
            return x + 4 + inner(x)

        @torch._jit_internal.weak_script
        def weak_script_fn_inner(x):
            return x + 6 + not_a_script_fn(x)

        @torch._jit_internal.weak_script
        def weak_script_fn(x):
            return x + 5 + weak_script_fn_inner(x) + weak_script_fn_inner(x)

        def fn(x):
            x = not_a_script_fn(x)
            x = strong_script_fn(x)
            return weak_script_fn(x)

        input = torch.randn(3, 4, 5)
        self.checkScript(fn, (input,))

    def test_python_op_exception(self):
        def python_op(x):
            raise Exception("bad!")

        @torch.jit.script
        def fn(x):
            return python_op(x)

        with self.assertRaisesRegex(RuntimeError, "operation failed in interpreter"):
            fn(torch.tensor(4))

    def test_trace_contiguous(self):
        def foo(x):
            return x[:, :, ::2].contiguous().view(12)

        x = torch.rand(2, 3, 4)
        traced = torch.jit.trace(foo, (x,))
        y = traced(x)
        self.assertNotEqual(x.storage().data_ptr(), y.storage().data_ptr())

    # This tests the logic in THPVariable_contiguous. There is short-circuiting
    # code that prevents us from even getting to VariableType::contiguous, since
    # it is an optimization that prevents us from acquiring the GIL for touching
    # the device. We needed to add the tracing logic directly into the
    # THPVariable_contiguous function only for the path where we are skipping
    # dispatch into contiguous. We should see an aten::contiguous in this trace!
    def test_trace_contiguous_short_circuit(self):
        def foo(x):
            return x.contiguous()

        x = torch.rand(2, 3, 4)
        traced = torch.jit.trace(foo, (x,))
        self.assertExpectedGraph(traced.graph)

    def test_weak_module(self):

        @torch._jit_internal.weak_module
        class Weak(torch.nn.Module):
            __constants__ = ['number']

            def __init__(self):
                super(Weak, self).__init__()
                self.number = 199

            def python_op_in_weak_module(self, x):
                return x + 123

            @torch._jit_internal.weak_script_method
            def forward(self, x):
                return 55 + self.number + self.python_op_in_weak_module(x)

        class OtherStrong(torch.jit.ScriptModule):
            __constants__ = ['number']

            def __init__(self):
                super(OtherStrong, self).__init__()
                self.number = 357

            def python_op_in_strong_module(self, x):
                return x + 456

            @torch.jit.script_method
            def forward(self, x):
                return x + self.number + self.python_op_in_strong_module(x)

        class Passthrough(torch.jit.ScriptModule):
            def __init__(self):
                super(Passthrough, self).__init__()
                self.weak = Weak()

            @torch.jit.script_method
            def forward(self, x):
                return self.weak(x)

        weak_mod = Weak()
        x = torch.ones(1)
        expected_result = 55 + 199 + (x + 123)

        # Ensure weak mod is running without the JIT by passing the wrong type
        # (i.e. not a tensor)
        weak_mod(2)

        python_result = weak_mod(x)
        strong_mod = Passthrough()
        script_result = strong_mod(x)

        self.assertEqual(python_result, expected_result)
        self.assertEqual(script_result, expected_result)

        class Strong(torch.jit.ScriptModule):
            def __init__(self):
                super(Strong, self).__init__()
                self.weak = Weak()
                self.strong = OtherStrong()

            @torch.jit.script_method
            def forward(self, x):
                y = 2 * x
                return y + 1 + self.weak(y) + self.strong(y)

        strong_mod = Strong()
        strong_mod2 = Strong()
        x = torch.ones(1)
        expected_result = (x * 2) + 1 + (55 + 199 + x * 2 + 123) + (x * 2 + 357 + x * 2 + 456)
        script_result = strong_mod(x)
        script_result2 = strong_mod2(x)
        self.assertEqual(script_result, expected_result)
        self.assertEqual(script_result, script_result2)

    def test_weak_module_parameters_and_buffers(self):
        weights = torch.randn(10, 10)
        bias = torch.randn(10)
        weights2 = torch.randn(10, 10)
        bias2 = torch.randn(10)

        @torch._jit_internal.weak_module
        class TestLinear(torch.nn.Module):
            def __init__(self, in_features, out_features):
                super(TestLinear, self).__init__()
                self.in_features = in_features
                self.out_features = out_features
                self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
                self.bias = torch.nn.Parameter(torch.Tensor(out_features))
                self.register_buffer('counter', torch.ones(out_features))
                self.reset_parameters()

            def reset_parameters(self):
                torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
                if self.bias is not None:
                    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
                    bound = 1 / math.sqrt(fan_in)
                    torch.nn.init.uniform_(self.bias, -bound, bound)

            @torch._jit_internal.weak_script_method
            def forward(self, input):
                return F.linear(input, self.weight, self.bias) + self.counter

        # Initialize a ScriptModule that uses the weak module above multiple times
        class Strong(torch.jit.ScriptModule):
            def __init__(self):
                super(Strong, self).__init__()
                self.fc1 = TestLinear(10, 10)
                self.fc1.weight = torch.nn.Parameter(weights)
                self.fc1.bias = torch.nn.Parameter(bias)
                self.fc2 = TestLinear(10, 10)
                self.fc2.weight = torch.nn.Parameter(weights2)
                self.fc2.bias = torch.nn.Parameter(bias2)

            @torch.jit.script_method
            def forward(self, x):
                return x + self.fc1(x) + self.fc1(x) + self.fc2(x)

        strong_mod = Strong()

        # Run same calculation as module
        inp = torch.ones(10)
        lin = torch.nn.Linear(10, 10)
        lin.weight = torch.nn.Parameter(weights)
        lin.bias = torch.nn.Parameter(bias)
        lin2 = torch.nn.Linear(10, 10)
        lin2.weight = torch.nn.Parameter(weights2)
        lin2.bias = torch.nn.Parameter(bias2)
        expected_result = inp + (lin(inp) + torch.ones(10)) * 2 + lin2(inp) + torch.ones(10)

        self.assertEqual(strong_mod(inp), expected_result)
        self.assertExportImportModule(strong_mod, (inp,))

    def test_weak_module_nested(self):
        @torch._jit_internal.weak_module
        class OtherWeak(torch.nn.Module):
            __constants__ = ['constant']

            def __init__(self, in_features, out_features):
                super(OtherWeak, self).__init__()
                self.in_features = in_features
                self.out_features = out_features
                self.weight = torch.nn.Parameter(torch.ones(out_features, in_features))
                self.bias = torch.nn.Parameter(torch.ones(out_features))
                self.constant = 3

            @torch._jit_internal.weak_script_method
            def forward(self, x):
                return x * x + self.constant + F.linear(x, self.weight, self.bias)

        class OtherStrong(torch.jit.ScriptModule):

            def __init__(self):
                super(OtherStrong, self).__init__()

            @torch.jit.script_method
            def forward(self, x):
                return x + 27

        @torch._jit_internal.weak_module
        class Weak(torch.nn.Module):
            def __init__(self, in_features, out_features):
                super(Weak, self).__init__()
                self.in_features = in_features
                self.out_features = out_features
                self.weight = torch.nn.Parameter(2 * torch.ones(out_features, in_features))
                self.bias = torch.nn.Parameter(2 * torch.ones(out_features))
                self.weak_submodule = OtherWeak(10, 10)
                self.strong_submodule = OtherStrong()

            @torch._jit_internal.weak_script_method
            def forward(self, x):
                return x + self.weak_submodule(x) + self.strong_submodule(x) \
                    + F.linear(x, self.weight, self.bias)

        class Strong(torch.jit.ScriptModule):
            __constants__ = ['constant']

            def __init__(self):
                super(Strong, self).__init__()
                self.weak = Weak(10, 10)

            @torch.jit.script_method
            def forward(self, x):
                return x + self.weak(x)

        strong_mod = Strong()
        inp = torch.randn(10)
        result = strong_mod(inp)
        expected_result = inp + (inp + inp * inp + inp + 27) + 3 \
            + F.linear(inp, torch.ones(10, 10), torch.ones(10)) \
            + F.linear(inp, 2 * torch.ones(10, 10), 2 * torch.ones(10))
        self.assertEqual(result, expected_result)

    def test_weak_module_submodule(self):
        @torch._jit_internal.weak_module
        class Weak(torch.nn.Module):
            def __init__(self):
                super(Weak, self).__init__()
                self.param = torch.nn.Parameter(100 * torch.ones(5))

            @torch._jit_internal.weak_script_method
            def forward(self, x):
                return x + self.param

        weak = Weak()

        class OtherStrong(torch.jit.ScriptModule):
            def __init__(self):
                super(OtherStrong, self).__init__()
                self.weak = weak
                self.weak2 = weak

            @torch.jit.script_method
            def forward(self, x):
                return x + self.weak(x)

        class Strong(torch.jit.ScriptModule):
            def __init__(self):
                super(Strong, self).__init__()
                self.weak = Weak()

            @torch.jit.script_method
            def forward(self, x):
                return self.weak(x) + weak(x)

        other_strong_mod = OtherStrong()

        self.assertIs(other_strong_mod.weak, other_strong_mod.weak2)

        with self.assertRaisesRegex(RuntimeError, "Attempted to inline a Module with param"):
            strong_mod = Strong()

    def test_weak_module_copying(self):
        class Submodule(torch.nn.Module):
            def __init__(self):
                super(Submodule, self).__init__()

            def forward(self, x):
                return x + 100

        @torch._jit_internal.weak_module
        class Weak(torch.nn.Module):
            def __init__(self, in_features, out_features):
                super(Weak, self).__init__()
                self.weight = torch.nn.Parameter(torch.ones(out_features, in_features))
                self.bias = torch.nn.Parameter(torch.ones(out_features))
                self.register_buffer("buffer", torch.ones(out_features))
                self.submodule = Submodule()

            @torch._jit_internal.weak_script_method
            def forward(self, x):
                return F.linear(x, self.weight, self.bias) \
                    + self.buffer + self.submodule(x)

        class Strong(torch.jit.ScriptModule):
            def __init__(self, weak):
                super(Strong, self).__init__()
                self.weak = weak

            @torch.jit.script_method
            def forward(self, x):
                return self.weak(x)

        inp = torch.ones(5, 5) * 5
        weak_mod = Weak(5, 5)
        strong_mod = Strong(weak_mod)

        self.assertTrue(isinstance(strong_mod.weak, torch.jit.ScriptModule))
        self.assertFalse(isinstance(weak_mod, torch.jit.ScriptModule))

        self.assertIs(strong_mod.weak.weight, weak_mod.weight)
        self.assertIs(strong_mod.weak.buffer, weak_mod.buffer)
        self.assertIs(strong_mod.weak.submodule, weak_mod.submodule)

        # Test lookup fallback
        weak_mod.new_attribute = 10
        self.assertIs(strong_mod.weak.new_attribute, weak_mod.new_attribute)

        weak_mod.weight.data += torch.ones(5, 5) * 100
        self.assertTrue(strong_mod(inp).allclose(weak_mod(inp)))

        # Re-assignment is not tracked
        weak_mod.weight = torch.nn.Parameter(torch.ones(5, 5) * 100)
        self.assertFalse(strong_mod(inp).allclose(weak_mod(inp)))

    def test_backend_cudnn_enabled(self):
        # Only test that this compiles
        @torch.jit.script
        def fn(x):
            if torch.backends.cudnn.enabled:
                x = x + 2
            else:
                x = x + 3
            return x

    def test_inplace_add(self):

        def foo(a, b):
            c = a + b
            c.add_(b)
            return c
        self.checkScript(foo, (torch.rand(3), torch.rand(3)))

    def test_add_out(self):
        def foo(a, b):
            c = a + b
            e = 2 * a
            torch.add(c, b, out=e)
            return e
        self.checkScript(foo, (torch.rand(3), torch.rand(3)))

    def test_augmented_assign(self):
        def foo(a, b):
            a += b
            a -= b
            a /= b
            a *= b
            return a, b
        self.checkScript(foo, (torch.rand(3), torch.rand(3)), check_expected=True)

    def test_pass(self):
        def foo(x):
            # type: (bool) -> int
            for _i in range(3):
                pass
            if x:
                pass
            else:
                pass
            return 3

        self.checkScript(foo, (True,))

    def test_optional_conversion(self):
        @torch.jit.script
        def other_fn(x=None):
            # type: (Optional[int]) -> int
            return torch.jit._unwrap_optional(x)

        @torch.jit.script
        def fn(x):
            # type: (int) -> int
            return other_fn(x)

        self.assertEqual(fn(2), 2)

        @torch.jit.script
        def unify_to_optional(x):
            # type: (bool) -> Optional[int]
            if x:
                a = None
            else:
                a = 2
            return a

        self.assertEqual(unify_to_optional(True), None)
        self.assertEqual(unify_to_optional(False), 2)

        @torch.jit.script
        def opt_list(x):
            # type: (Optional[List[float]]) -> int
            return 2

        @torch.jit.script
        def broadcast_opt_list(x):
            # type: (Optional[BroadcastingList2[float]]) -> int
            return 2

        @torch.jit.script
        def opt_list_tuple_caller(x):
            # type: (Tuple[float, float]) -> int
            return opt_list(x) + broadcast_opt_list(x)

        self.assertEqual(opt_list_tuple_caller((2., 3.)), 4)

    def test_lhs_indexing(self):
        def foo(a, b):
            a = a.clone()
            a[0] = b
            return a
        self.checkScript(foo, (torch.rand(2, 3), torch.rand(3)))

    def test_lhs_advanced_indexing_assignment(self):
        def foo(x, y):
            a = torch.exp(x)
            b = x == 1
            a[b] = y[b]
            return a
        self.checkScript(foo, (torch.ones(4, 3), torch.ones(4, 3)))

    def test_lhs_advanced_indexing_augmented_assignment(self):
        def foo(x, y):
            a = torch.exp(x)
            b = x == 1
            a[b] += y[b]
            return a
        self.checkScript(foo, (torch.ones(4, 3), torch.ones(4, 3)))

    def test_lhs_indexing_list(self):
        def foo(a, b):
            ls = [a]
            ls[0] = b
            return ls
        self.checkScript(foo, (torch.rand(2, 3), torch.rand(3)))

    def test_inplace_copy_script(self):
        def foo(x):
            a = torch.rand(3, 4)
            a.copy_(x)
            return a
        self.checkScript(foo, (torch.rand(3, 4),))

    def test_lhs_indexing_increment(self):
        def foo(a, b):
            a[0] += b
            return a
        self.checkScript(foo, (torch.rand(2, 3), torch.rand(3)))

    def test_lhs_indexing_increment_list(self):
        def foo(a, b):
            a = a.clone()
            ls = [a, b]
            ls[0] += b
            return ls
        self.checkScript(foo, (torch.rand(2, 3), torch.rand(3)))

    def test_lhs_indexing_increment_list_prim(self):
        def foo():
            ls = [1, 2, 3]
            ls[0] += 5
            return ls
        self.checkScript(foo, ())

    def test_lhs_indexing_multi(self):
        def foo(a, b):
            a = a.clone()
            foo, a[0], bar = (1, b, 3)
            return foo, a, bar
        self.checkScript(foo, (torch.rand(2, 3), torch.rand(3)))

    def test_bool_dispatch(self):
        with self.disableModuleHook():  # TODO: Python print broadcasting list
            def kwarg_false(x):
                # type: (Tensor) -> Tensor
                return F.max_pool1d(x, 1, 1, return_indices=False)
            self.checkScript(kwarg_false, (torch.randn(3, 3, 3),))

            def kwarg_true(x):
                # type: (Tensor) -> Tuple[Tensor, Tensor]
                return F.max_pool1d(x, 1, 1, return_indices=True)
            self.checkScript(kwarg_true, (torch.randn(3, 3, 3),))

            def full_kwarg_false(x):
                # type: (Tensor) -> Tensor
                return F.max_pool1d(x, 1, 1, ceil_mode=False, return_indices=False)
            self.checkScript(full_kwarg_false, (torch.randn(3, 3, 3),))

            def full_kwarg_true(x):
                # type: (Tensor) -> Tuple[Tensor, Tensor]
                return F.max_pool1d(x, 1, 1, ceil_mode=False, return_indices=True)
            self.checkScript(full_kwarg_true, (torch.randn(3, 3, 3),))

            def use_default(x):
                # type: (Tensor) -> Tensor
                return F.max_pool1d(x, 1, 1)
            self.checkScript(use_default, (torch.randn(3, 3, 3),))

            def arg_false(x):
                # type: (Tensor) -> Tensor
                return F.max_pool1d(x, 1, 1, 0, 1, False, False)
            self.checkScript(arg_false, (torch.randn(3, 3, 3),))

            def arg_true(x):
                # type: (Tensor) -> Tuple[Tensor, Tensor]
                return F.max_pool1d(x, 1, 1, 0, 1, False, True)
            self.checkScript(arg_true, (torch.randn(3, 3, 3),))

    def test_infer_size(self):
        from torch._C import _infer_size

        def fn(x, y):
            # type: (Tensor, Tensor) -> List[int]
            return _infer_size(x.size(), y.size())

        self.checkScript(fn, (torch.ones(2, 4, 2), torch.ones(2, 4, 2)))

    def test_mutable_dce(self):
        @torch.jit.script
        def foo():
            a = torch.rand(2, 3)
            a += torch.rand(2, 3)
            b = torch.rand(2, 3)
            b += torch.rand(2, 3)
            # b should be cleaned up but not a
            return a

        self.assertExpectedGraph(foo.graph)

    def test_mutable_dce_block(self):
        @torch.jit.script
        def foo():
            a = torch.rand(2, 3)
            a += torch.rand(2, 3)
            b = torch.rand(2, 3)
            if bool(a > torch.zeros(2, 3)):
                b += torch.rand(2, 3)
                a += torch.rand(2, 3)
            # a should be cleaned up but not b
            return b

        self.assertExpectedGraph(foo.graph)

    def test_mutable_dce_graph_input(self):
        @torch.jit.script
        def foo(a):
            a += torch.rand(2, 3)
            # shouldn't clean up `a` even though it's not used in the output

        self.assertExpectedGraph(foo.graph)

    def test_mutable_dce_list(self):
        @torch.jit.script
        def foo(a):
            l = []
            l.append(a)
            c = l[0]
            b = torch.rand(2, 3)
            c += torch.rand(2, 3)
            return b

        self.assertExpectedGraph(foo.graph)

    def test_mutable_dce_loop(self):
        @torch.jit.script
        def foo(a):
            l = []
            l.append(a)
            i = 0
            b = torch.rand(2, 3)
            while i < 1:
                dead = torch.rand(2, 3)
                c = l[0]
                c += torch.rand(2, 3)
                i += 1
            return b

        self.assertExpectedGraph(foo.graph)

    def test_mutable_dce_wildcards(self):
        def fn():
            x = torch.ones(2, 3)
            l = []
            l.append(x)
            x_view = l[0]
            x.add_(torch.ones(2, 3))
            return x_view

        self.checkScript(fn, ())

    def test_cpp_function_tensor_str(self):
        x = torch.randn(2, 2)
        scale = torch.randn(2, 2, requires_grad=True)
        shift = torch.randn(2, 2, requires_grad=True)

        @torch.jit.script
        def fn(x, scale, shift):
            return scale * x + shift

        with self.capture_stdout() as captured:
            print(fn(x, scale, shift))

    def test_non_final_return(self):

        def simple(x):
            if bool(x > 3):
                return x + 1
            else:
                return x + 2
            raise RuntimeError("nope")

        def nest(x):
            x = x + 1
            if bool(x > 3):
                if bool(x > 4):
                    x += 1
                return x + 1
            else:
                return x + 2

        def early_ret(x):
            x = x + 1
            if bool(x > 3):
                return x + 1
            x = x + 1
            return x + 2

        def nest_early_ret(x):
            x = x + 1
            if bool(x > 3):
                if bool(x > 4):
                    return x + 2
                return x + 1
            x = x + 1
            return x + 2

        self.checkScript(simple, torch.rand(1))
        self.checkScript(nest, torch.rand(1))
        self.checkScript(early_ret, torch.rand(1))
        self.checkScript(nest_early_ret, torch.rand(1))

        with self.assertRaisesRegex(RuntimeError, "early"):
            @torch.jit.script
            def not_early_ret(x):
                if bool(x > 3):
                    if bool(x > 4):
                        return 1
                    print("foo")
                else:
                    print("5")
                return 7

        with self.assertRaisesRegex(RuntimeError, "some paths"):
            @torch.jit.script
            def not_total_ret(x):
                if bool(x > 3):
                    if bool(x > 4):
                        return 1
                    else:
                        return 2
                else:
                    print("5")
                return 7

        with self.assertRaisesRegex(RuntimeError, "from a loop"):
            @torch.jit.script
            def nest_while_ret(x):
                while bool(x > 4):
                    if bool(x < 3):
                        return 4
                return 5

        with self.assertRaisesRegex(RuntimeError, "from a loop"):
            @torch.jit.script
            def nest_for_ret(x):
                for _ in range(3):
                    if bool(x < 3):
                        return 4
                return 5

    def test_overloading(self):
        @torch._jit_internal.weak_module
        class W(torch.nn.Module):
            __overloads__ = {'forward': ['forward_tuple', 'forward_tensor']}

            def __init__(self):
                super(W, self).__init__()

            @torch._jit_internal.weak_script_method
            def forward_tuple(self, x):
                # type: (Tuple[Tensor, Tensor]) -> Tensor
                return x[0] + 5

            def forward(self, x):
                # manually do argument switching
                if isinstance(x, tuple):
                    return self.forward_tuple(x)
                else:
                    return self.forward_tensor(x)

            @torch._jit_internal.weak_script_method
            def forward_tensor(self, x):
                # type: (Tensor) -> Tensor
                return x + 20

        class S(torch.jit.ScriptModule):
            def __init__(self):
                super(S, self).__init__()
                self.weak = W()

            @torch.jit.script_method
            def forward(self, x):
                return self.weak(x) + self.weak((x, x))

        s = S()
        x = torch.ones(1)
        self.assertEqual(s(x), x + 20 + 5 + x)

        w = W()
        self.assertEqual(w((x, x)), x + 5)
        self.assertEqual(w((x)), x + 20)

    def test_select_after_chunk(self):
        def foo(x):
            chunked = torch.chunk(x, 1)
            foo = chunked[0]
            foo.add_(5)
            return x

        self.checkScript(foo, [torch.rand(2, 3)])

    def test_list_python_op(self):
        def python_list_op(lst):
            # type: (List[Tensor]) -> Tensor
            return lst[0]

        def fn(lst):
            # type: (List[Tensor]) -> Tensor
            return python_list_op(lst)

        self.checkScript(fn, ([torch.ones(2) + 2, torch.ones(2)],))

    def test_ignore_decorator(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super(M, self).__init__()
                tensor = torch.zeros(1, requires_grad=False)
                self.register_buffer('some_state', torch.nn.Parameter(tensor))

            @torch.jit.script_method
            def forward(self, x):
                self.ignored_code(x)
                return x

            @torch.jit.ignore
            def ignored_code(self, x):
                self.some_state = torch.tensor((100,))

        # Assert ignored code is run
        m = M()
        self.assertEqual(m.some_state, torch.zeros(1))
        m(torch.ones(1))
        self.assertEqual(m.some_state, torch.zeros(1) + 100)

        # Export and ensure ignored code not present
        pp, constants = m._python_print()
        printed = torch.jit.ScriptModule()
        ppv = "op_version_set = 0\n{}".format(pp)
        torch._C._jit_import_methods(printed, ppv, constants)
        self.assertIn('IgnoredPythonOp', ppv)
        self.assertNotIn('ignored_code', ppv)

        with self.assertRaisesRegex(torch.jit.Error, "This Python function is annotated to be ignored"):
            printed(torch.ones(1))

    def test_view_write(self):
        def fn(x, y):
            l = []
            l.append(x)
            x_view = l[0]
            a = x + x
            x_view.add_(y)
            b = x + x
            return a == b
        self.checkScript(fn, (torch.rand(2, 3), torch.rand(2, 3)))

    def test_dict_view(self):
        def fn(x, y):
            l = {"a": x}
            x_view = l["a"]
            a = x + x
            x_view.add_(y)
            b = x + x
            return a == b
        self.checkScript(fn, (torch.rand(2, 3), torch.rand(2, 3)))

    def test_dict_ops(self):
        d = {'a': torch.ones(1), 'b': torch.ones(1) + 1, 'c': torch.ones(1) + 2}

        @torch.jit.script
        def keys(x):
            # type: (Dict[str, Tensor]) -> List[str]
            return list(x.keys())

        self.assertEqual(set(keys(d)), set(d.keys()))

        @torch.jit.script
        def values(x):
            # type: (Dict[str, Tensor]) -> List[Tensor]
            return list(x.values())

        self.assertEqual(set(values(d)), set(d.values()))

        def length(x):
            # type: (Dict[str, Tensor]) -> int
            return len(x)

        self.checkScript(length, (d,))

    def test_dict(self):
        def simple(x):
            # type: (Dict[str, int]) -> Dict[str, int]
            return x

        self.checkScript(simple, ({'item': 20, 'other_item': 120},))

        def index(x):
            # type: (Dict[str, int]) -> int
            return x['item']

        self.checkScript(index, ({'item': 20, 'other_item': 120},))

        def type_default():
            # type: () -> Dict[str, Tensor]
            return {}

        self.checkScript(type_default, ())

        @torch.jit.script
        def missing_index(x):
            # type: (Dict[str, int]) -> int
            return x['dne']

        with self.assertRaisesRegex(RuntimeError, "KeyError"):
            missing_index({'item': 20, 'other_item': 120})

        code = dedent('''
            def literal1():
                return torch.jit.annotate(Dict[int, float], {})
            def literal2():
                return torch.jit.annotate(Dict[int, float], {10: 1.2})
        ''')
        cu = torch.jit.CompilationUnit(code)
        self.assertEqual({}, cu.literal1())
        self.assertEqual({10: 1.2}, cu.literal2())

        cu = torch.jit.CompilationUnit(dedent('''
            def literal3():
                return torch.jit.annotate(Dict[int, float], {10: 1.2, 11: 1.3})
        '''))
        self.assertEqual({10: 1.2, 11: 1.3}, cu.literal3())

        def list_of_dicts():
            # type: () -> List[Dict[str, Tensor]]
            return [{'word': torch.ones(2) + 3}, {'other word': torch.ones(1) + 2}]

        self.checkScript(list_of_dicts, ())

    def dict_to_python(self):
        def python_lookup(my_dict, keys):
            # type: (Dict[str, int], List[str]) -> List[int]
            return [my_dict[k] for k in keys]

        def fn(my_dict, keys):
            # type: (Dict[str, int], List[str]) -> List[int]
            return python_lookup(my_dict, keys)

        a_dict = {'a': torch.ones(1), 'b': torch.ones(1) + 1, 'c': torch.ones(1) + 2}
        self.checkScript(fn, (a_dict, ('a', 'c')))


class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class TestEndToEndHybridFrontendModels(JitTestCase):
    @staticmethod
    def _test_dcgan_models(self, device, check_export_import=True):
        class DCGANGenerator(nn.Module):
            def __init__(self, nz, ngf, nc):
                super(DCGANGenerator, self).__init__()
                self.main = nn.Sequential(
                    # input is Z, going into a convolution
                    nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
                    nn.BatchNorm2d(ngf * 8),
                    nn.ReLU(True),
                    # state size. (ngf*8) x 4 x 4
                    nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ngf * 4),
                    nn.ReLU(True),
                    # state size. (ngf*4) x 8 x 8
                    nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ngf * 2),
                    nn.ReLU(True),
                    # state size. (ngf*2) x 16 x 16
                    nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ngf),
                    nn.ReLU(True),
                    # state size. (ngf) x 32 x 32
                    nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
                    nn.Tanh()
                    # state size. (nc) x 64 x 64
                )

            def forward(self, input):
                return self.main(input)

        class DCGANDiscriminator(nn.Module):
            def __init__(self, nc, ndf):
                super(DCGANDiscriminator, self).__init__()
                self.main = nn.Sequential(
                    # input is (nc) x 64 x 64
                    nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True),
                    # state size. (ndf) x 32 x 32
                    nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ndf * 2),
                    nn.LeakyReLU(0.2, inplace=True),
                    # state size. (ndf*2) x 16 x 16
                    nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ndf * 4),
                    nn.LeakyReLU(0.2, inplace=True),
                    # state size. (ndf*4) x 8 x 8
                    nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ndf * 8),
                    nn.LeakyReLU(0.2, inplace=True),
                    # state size. (ndf*8) x 4 x 4
                    nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                    nn.Sigmoid()
                )

            def forward(self, input):
                return self.main(input).view(-1, 1).squeeze(1)

        bs, nz, ngf, nc, ndf = 5, 6, 9, 3, 10
        self.checkTrace(DCGANGenerator(nz, ngf, nc).to(device),
                        (torch.rand(bs, nz, 1, 1, device=device),),
                        export_import=check_export_import)
        example_input = DCGANGenerator(nz, ngf, nc).to(device)(torch.rand(bs, nz, 1, 1, device=device))
        self.checkTrace(DCGANDiscriminator(nc, ndf).to(device), (example_input,),
                        export_import=check_export_import)

    def test_dcgan_models(self):
        self._test_dcgan_models(self, device='cpu')

    @unittest.skipIf(not RUN_CUDA, "no CUDA")
    def test_dcgan_models_cuda(self):
        # XXX: export_import on CUDA modules doesn't work (#11480)
        self._test_dcgan_models(self, device='cuda', check_export_import=False)

    @staticmethod
    def _test_neural_style(self, device, check_export_import=True):
        class TransformerNet(torch.nn.Module):
            def __init__(self):
                super(TransformerNet, self).__init__()
                # Initial convolution layers
                self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
                self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
                self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
                self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
                self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
                self.in3 = torch.nn.InstanceNorm2d(128, affine=True)
                # Residual layers
                self.res1 = ResidualBlock(128)
                self.res2 = ResidualBlock(128)
                self.res3 = ResidualBlock(128)
                self.res4 = ResidualBlock(128)
                self.res5 = ResidualBlock(128)
                # Upsampling Layers
                self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
                self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
                self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
                self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
                self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
                # Non-linearities
                self.relu = torch.nn.ReLU()

            def forward(self, X):
                y = self.relu(self.in1(self.conv1(X)))
                y = self.relu(self.in2(self.conv2(y)))
                y = self.relu(self.in3(self.conv3(y)))
                y = self.res1(y)
                y = self.res2(y)
                y = self.res3(y)
                y = self.res4(y)
                y = self.res5(y)
                y = self.relu(self.in4(self.deconv1(y)))
                y = self.relu(self.in5(self.deconv2(y)))
                y = self.deconv3(y)
                return y

        class ConvLayer(torch.nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size, stride):
                super(ConvLayer, self).__init__()
                reflection_padding = kernel_size // 2
                self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
                self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

            def forward(self, x):
                out = self.reflection_pad(x)
                out = self.conv2d(out)
                return out

        class ResidualBlock(torch.nn.Module):
            """ResidualBlock
            introduced in: https://arxiv.org/abs/1512.03385
            recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
            """

            def __init__(self, channels):
                super(ResidualBlock, self).__init__()
                self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
                self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
                self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
                self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                residual = x
                out = self.relu(self.in1(self.conv1(x)))
                out = self.in2(self.conv2(out))
                out = out + residual
                return out

        class UpsampleConvLayer(torch.nn.Module):
            """UpsampleConvLayer
            Upsamples the input and then does a convolution. This method gives better results
            compared to ConvTranspose2d.
            ref: http://distill.pub/2016/deconv-checkerboard/
            """

            def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
                super(UpsampleConvLayer, self).__init__()
                self.upsample = upsample
                if upsample:
                    self.upsample_layer = torch.nn.Upsample(mode='nearest', scale_factor=upsample)
                reflection_padding = kernel_size // 2
                self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
                self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

            def forward(self, x):
                x_in = x
                if self.upsample:
                    x_in = self.upsample_layer(x_in)
                out = self.reflection_pad(x_in)
                out = self.conv2d(out)
                return out

        self.checkTrace(TransformerNet(), (torch.rand(5, 3, 16, 16),), export_import=check_export_import)

    def test_neural_style(self):
        self._test_neural_style(self, device='cpu')

    @unittest.skipIf(not RUN_CUDA, "no CUDA")
    def test_neural_style_cuda(self):
        # XXX: export_import on CUDA modules doesn't work (#11480)
        self._test_neural_style(self, device='cuda', check_export_import=False)

    @staticmethod
    def _test_mnist(self, device, check_export_import=True):
        # eval() is present because dropout makes this nondeterministic
        self.checkTrace(MnistNet().to(device).eval(), (torch.rand(5, 1, 28, 28, device=device),),
                        export_import=check_export_import)

    def test_mnist(self):
        self._test_mnist(self, device='cpu')

    @unittest.skipIf(not RUN_CUDA, "no CUDA")
    def test_mnist_cuda(self):
        # XXX: export_import on CUDA modules doesn't work (#11480)
        self._test_mnist(self, device='cuda', check_export_import=False)

    @unittest.skipIf(not RUN_CUDA, "no CUDA")
    def test_mnist_training_leaks_no_memory_cuda(self):
        net = MnistNet().cuda()
        # MnistNet uses dropout, don't check its trace
        traced_net = torch.jit.trace(net, [torch.randn(5, 1, 28, 28, device='cuda')],
                                     check_trace=False)

        def train(iters):
            for _ in range(iters):
                # Get some fake data
                inp = torch.randn(5, 1, 28, 28, device='cuda')
                out = traced_net(inp)

                # Here's some fake loss
                out.sum().backward()

                # Zero out grads
                traced_net.zero_grad()

        # Set it up so the params have .grad fields so they are not reported as leaks
        train(1)

        with self.assertLeaksNoCudaTensors():
            train(5)

    @staticmethod
    def _test_reinforcement_learning(self, device, test_export_import=True):
        class Policy(nn.Module):
            def __init__(self):
                super(Policy, self).__init__()
                self.affine1 = nn.Linear(4, 128)
                self.affine2 = nn.Linear(128, 2)

            def forward(self, x):
                x = F.relu(self.affine1(x))
                action_scores = self.affine2(x)
                return F.softmax(action_scores, dim=1)

        self.checkTrace(Policy().to(device), (torch.rand(1, 4, device=device),),
                        export_import=test_export_import)

    def test_reinforcement_learning(self):
        self._test_reinforcement_learning(self, device='cpu')

    @unittest.skipIf(not RUN_CUDA, "no CUDA")
    def test_reinforcement_learning_cuda(self):
        # XXX: export_import on CUDA modules doesn't work (#11480)
        self._test_reinforcement_learning(self, device='cuda', test_export_import=False)

    @staticmethod
    def _test_snli(self, device, check_export_import=True, quantized=False):
        class Bottle(nn.Module):

            def forward(self, input):
                if len(input.size()) <= 2:
                    return super(Bottle, self).forward(input)
                size = input.size()[:2]
                out = super(Bottle, self).forward(input.view(size[0] * size[1], -1))
                return out.view(size[0], size[1], -1)

        class Linear(Bottle, nn.Linear):
            pass

        class Encoder(nn.Module):

            def __init__(self, config):
                super(Encoder, self).__init__()
                self.config = config
                input_size = config.d_proj if config.projection else config.d_embed
                dropout = 0 if config.n_layers == 1 else config.dp_ratio
                self.rnn = nn.LSTM(input_size=input_size, hidden_size=config.d_hidden,
                                   num_layers=config.n_layers, dropout=dropout,
                                   bidirectional=config.birnn)

            def forward(self, inputs):
                batch_size = inputs.size()[1]
                state_shape = self.config.n_cells, batch_size, self.config.d_hidden
                h0 = c0 = inputs.new_zeros(state_shape)
                outputs, (ht, ct) = self.rnn(inputs, (h0, c0))
                return ht[-1] if not self.config.birnn else ht[-2:].transpose(0, 1).contiguous().view(batch_size, -1)

        class SNLIClassifier(nn.Module):

            def __init__(self, config):
                super(SNLIClassifier, self).__init__()
                self.config = config
                self.embed = nn.Embedding(config.n_embed, config.d_embed)
                self.projection = Linear(config.d_embed, config.d_proj)
                self.encoder = Encoder(config)
                self.dropout = nn.Dropout(p=config.dp_ratio)
                self.relu = nn.ReLU()
                seq_in_size = 2 * config.d_hidden
                if self.config.birnn:
                    seq_in_size *= 2
                lin_config = [seq_in_size] * 2
                self.out = nn.Sequential(
                    Linear(*lin_config),
                    self.relu,
                    self.dropout,
                    Linear(*lin_config),
                    self.relu,
                    self.dropout,
                    Linear(*lin_config),
                    self.relu,
                    self.dropout,
                    Linear(seq_in_size, config.d_out))

            def forward(self, premise, hypothesis):
                prem_embed = self.embed(premise)
                hypo_embed = self.embed(hypothesis)
                if self.config.fix_emb:
                    prem_embed = prem_embed.detach()
                    hypo_embed = hypo_embed.detach()
                if self.config.projection:
                    prem_embed = self.relu(self.projection(prem_embed))
                    hypo_embed = self.relu(self.projection(hypo_embed))
                premise = self.encoder(prem_embed)
                hypothesis = self.encoder(hypo_embed)
                scores = self.out(torch.cat([premise, hypothesis], 1))
                return scores

        class Config:
            n_embed = 100
            d_embed = 100
            d_proj = 300
            dp_ratio = 0.0  # For deterministic testing TODO: change by fixing seed in checkTrace?
            d_hidden = 30
            birnn = True
            d_out = 300
            fix_emb = True
            projection = True
            n_layers = 2
            n_cells = 4  # 2 * n_layers because birnn = True

        premise = torch.LongTensor(48, 64).random_(0, 100).to(device)
        hypothesis = torch.LongTensor(24, 64).random_(0, 100).to(device)

        if quantized:
            snli = SNLIClassifier(Config()).cpu()
            torch.jit.quantized.quantize_linear_modules(snli)
            # we don't do export/import checks because we would need to call
            # _pack/_unpack
            self.checkTrace(snli, (premise, hypothesis), inputs_require_grads=False,
                            export_import=False)
        else:
            self.checkTrace(SNLIClassifier(Config()).to(device), (premise, hypothesis),
                            inputs_require_grads=False, export_import=check_export_import)

    def test_snli(self):
        self._test_snli(self, device='cpu')

    if not TEST_WITH_UBSAN and torch.fbgemm_is_cpu_supported():
        def test_snli_quantized(self):
            self._test_snli(self, device='cpu', quantized=True)

    @unittest.skipIf(not RUN_CUDA, "no CUDA")
    def test_snli_cuda(self):
        # XXX: export_import on CUDA modules doesn't work (#11480)
        self._test_snli(self, device='cuda', check_export_import=False)

    @staticmethod
    def _test_super_resolution(self, device, check_export_import=True):
        import torch.nn.init as init

        class Net(nn.Module):

            def __init__(self, upscale_factor):
                super(Net, self).__init__()

                self.relu = nn.ReLU()
                self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
                self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
                self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
                self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
                self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

            def forward(self, x):
                x = self.relu(self.conv1(x))
                x = self.relu(self.conv2(x))
                x = self.relu(self.conv3(x))
                x = self.pixel_shuffle(self.conv4(x))
                return x

        net = Net(upscale_factor=4).to(device)
        self.checkTrace(net, (torch.rand(5, 1, 32, 32, device=device),),
                        export_import=check_export_import)

    def test_super_resolution(self):
        self._test_super_resolution(self, device='cpu')

    @unittest.skipIf(not RUN_CUDA, 'no CUDA')
    def test_super_resolution_cuda(self):
        # XXX: export_import on CUDA modules doesn't work (#11480)
        self._test_super_resolution(self, device='cuda', check_export_import=False)

    @suppress_warnings
    def test_time_sequence_prediction(self):
        class Sequence(torch.jit.ScriptModule):
            def __init__(self):
                super(Sequence, self).__init__()
                self.lstm1 = nn.LSTMCell(1, 51)
                self.lstm2 = nn.LSTMCell(51, 51)
                self.linear = nn.Linear(51, 1)

            # TODO: could not pass tuple to a python Op and type annotations
            # is not descending to python signature, hence the wrapper
            # see https://github.com/pytorch/pytorch/issues/8778
            # and https://github.com/pytorch/pytorch/issues/8777
            def test_lstm1(self, input, hx, cx):
                # type: (Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
                return self.lstm1(input, (hx, cx))

            def test_lstm2(self, input, hx, cx):
                # type: (Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
                return self.lstm2(input, (hx, cx))

            # TODO: could not support tensor constructors in script
            # see https://github.com/pytorch/pytorch/issues/8814
            def test_tensor(self):
                return torch.tensor([], dtype=torch.double)

            @torch.jit.script_method
            def forward(self, input):
                # TODO: add future as input with default val
                # see https://github.com/pytorch/pytorch/issues/8724
                outputs = self.test_tensor()
                h_t = torch.zeros((3, 51), dtype=torch.double)
                c_t = torch.zeros((3, 51), dtype=torch.double)
                h_t2 = torch.zeros((3, 51), dtype=torch.double)
                c_t2 = torch.zeros((3, 51), dtype=torch.double)

                output = torch.zeros([3, 51])
                future = 2

                # TODO: chunk call should appear as the for loop iterable
                # We hard-code it to 4 for now.
                a, b, c, d = input.chunk(input.size(1), dim=1)
                for input_t in (a, b, c, d):
                    h_t, c_t = self.test_lstm1(input_t, h_t, c_t)
                    h_t2, c_t2 = self.test_lstm2(h_t, h_t2, c_t2)
                    output = self.linear(h_t2)
                    outputs = torch.cat((outputs, output), 1)
                for _ in range(future):  # if we should predict the future
                    h_t, c_t = self.test_lstm1(output, h_t, c_t)
                    h_t2, c_t2 = self.test_lstm2(h_t, h_t2, c_t2)
                    output = self.linear(h_t2)
                    outputs = torch.cat((outputs, output), 1)
                return outputs

        # TODO: toggle export_import once above issues are fixed
        self.checkTrace(Sequence(), (torch.rand(3, 4),),
                        export_import=False)

    @staticmethod
    def _test_vae(self, device, check_export_import=True, quantized=False):
        class VAE(nn.Module):
            def __init__(self):
                super(VAE, self).__init__()

                self.fc1 = nn.Linear(784, 400)
                self.fc21 = nn.Linear(400, 20)
                self.fc22 = nn.Linear(400, 20)
                self.fc3 = nn.Linear(20, 400)
                self.fc4 = nn.Linear(400, 784)

            def encode(self, x):
                h1 = F.relu(self.fc1(x))
                return self.fc21(h1), self.fc22(h1)

            def reparameterize(self, mu, logvar):
                if self.training:
                    std = torch.exp(0.5 * logvar)
                    eps = torch.randn_like(std)
                    return eps.mul(std).add_(mu)
                else:
                    return mu

            def decode(self, z):
                h3 = F.relu(self.fc3(z))
                return torch.sigmoid(self.fc4(h3))

            def forward(self, x):
                mu, logvar = self.encode(x.view(-1, 784))
                z = self.reparameterize(mu, logvar)
                return self.decode(z), mu, logvar

        if quantized:
            vae = VAE().to(device).eval()
            torch.jit.quantized.quantize_linear_modules(vae)
            # We don't do export/import checks because we would need to call
            # _unpack and _pack
            self.checkTrace(vae, (torch.rand(128, 1, 28, 28, device=device),),
                            export_import=False, allow_unused=True,
                            inputs_require_grads=False)
        else:
            # eval() is present because randn_like makes this nondeterministic
            self.checkTrace(VAE().to(device).eval(), (torch.rand(128, 1, 28, 28, device=device),),
                            export_import=check_export_import)

    def test_vae(self):
        self._test_vae(self, device='cpu')

    if not TEST_WITH_UBSAN and torch.fbgemm_is_cpu_supported():
        def test_vae_quantized(self):
            self._test_vae(self, device='cpu', quantized=True)

    @unittest.skipIf(not RUN_CUDA, "no CUDA")
    def test_vae_cuda(self):
        # XXX: export_import on CUDA modules doesn't work (#11480)
        self._test_vae(self, device='cuda', check_export_import=False)


# Smoke tests for export methods
class TestPytorchExportModes(JitTestCase):
    class MyModel(nn.Module):
        def __init__(self):
            super(TestPytorchExportModes.MyModel, self).__init__()

        def forward(self, x):
            return x.transpose(0, 1)

    def test_protobuf(self):
        torch_model = TestPytorchExportModes.MyModel()
        fake_input = Variable(torch.randn(1, 1, 224, 224), requires_grad=True)
        f = io.BytesIO()
        torch.onnx._export(torch_model, (fake_input), f, verbose=False,
                           export_type=torch.onnx.ExportTypes.PROTOBUF_FILE)

    def test_zipfile(self):
        torch_model = TestPytorchExportModes.MyModel()
        fake_input = Variable(torch.randn(1, 1, 224, 224), requires_grad=True)
        f = io.BytesIO()
        torch.onnx._export(torch_model, (fake_input), f, verbose=False,
                           export_type=torch.onnx.ExportTypes.ZIP_ARCHIVE)

    def test_compressed_zipfile(self):
        torch_model = TestPytorchExportModes.MyModel()
        fake_input = Variable(torch.randn(1, 1, 224, 224), requires_grad=True)
        f = io.BytesIO()
        torch.onnx._export(torch_model, (fake_input), f, verbose=False,
                           export_type=torch.onnx.ExportTypes.COMPRESSED_ZIP_ARCHIVE)

    def test_directory(self):
        torch_model = TestPytorchExportModes.MyModel()
        fake_input = Variable(torch.randn(1, 1, 224, 224), requires_grad=True)
        d = tempfile.mkdtemp()
        torch.onnx._export(torch_model, (fake_input), d, verbose=False,
                           export_type=torch.onnx.ExportTypes.DIRECTORY)
        shutil.rmtree(d)

    def test_onnx_multiple_return(self):
        @torch.jit.script
        def foo(a):
            return (a, a)
        f = io.BytesIO()
        x = torch.ones(3)
        torch.onnx._export(foo, (x,), f, example_outputs=(x, x))

    @skipIfNoLapack
    def test_aten_fallback(self):
        class ModelWithAtenNotONNXOp(nn.Module):
            def forward(self, x, y):
                abcd = x + y
                defg = torch.qr(abcd)
                return defg

        x = torch.rand(3, 4)
        y = torch.rand(3, 4)
        f = io.BytesIO()
        exported = torch.onnx.export_to_pretty_string(
            ModelWithAtenNotONNXOp(), (x, y), f,
            operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK)
        self.assertExpected(exported)

    # torch.fmod is using to test ONNX_ATEN.
    # If you plan to remove fmod from aten, or found this test failed.
    # please contact @Rui.
    def test_onnx_aten(self):
        class ModelWithAtenFmod(nn.Module):
            def forward(self, x, y):
                return torch.fmod(x, y)

        f = io.BytesIO()
        x = torch.randn(3, 4, dtype=torch.float32)
        y = torch.randn(3, 4, dtype=torch.float32)
        exported = torch.onnx.export_to_pretty_string(
            ModelWithAtenFmod(), (x, y), f,
            operator_export_type=OperatorExportTypes.ONNX_ATEN)
        self.assertExpected(exported)


# known to be failing in tracer
EXCLUDE_TRACED = {
    'test_split_dim',
    'test_split_dim_neg0',

    # The following fail due to #12024.
    # A prim::ListConstruct is involved and the indices get traced as DynamicType,
    # which always require_grad. This causes a crash in autodiff.
    'test___getitem___adv_index',
    'test___getitem___adv_index_beg',
    'test___getitem___adv_index_comb',
    'test___getitem___adv_index_dup',
    'test___getitem___adv_index_sub',
    'test___getitem___adv_index_sub_2',
    'test___getitem___adv_index_sub_3',
    'test___getitem___adv_index_var',

}

EXCLUDE_TYPE_CHECK = {
    # slogdet tests use itemgetter to select its only differentiable output,
    # but this happens outside of the graph we handle, so there are fewer
    # reference outputs than graph outputs.
    'test_slogdet_1x1_neg_det',
    'test_slogdet_1x1_pos_det',
    'test_slogdet_distinct_singular_values',
    'test_slogdet_neg_det',
    'test_slogdet_pos_det',
    'test_slogdet_symmetric',
    'test_slogdet_symmetric_pd',
}

# known to be failing in script
EXCLUDE_SCRIPT = {
    'test_norm_fro',
    'test_norm_fro_default',
    'test_norm_nuc',

    # aten op has additional cudnn argument
    'test_nn_unfold',

    # flaky test - TODO fix
    'test_nn_ctc_loss',

    # unknown builtin op
    'test_nn_fold',
}

EXCLUDE_PYTHON_PRINT = {
    # no support for BroadcastingList in python printer
    'test_nn_max_unpool1d',
    'test_nn_max_unpool2d',
    'test_nn_max_unpool3d',
    'test_nn_max_pool1d',
    'test_nn_max_pool2d',
    'test_nn_max_pool3d',
    'test_nn_max_pool1d_with_indices',
}

EXCLUDE_SCRIPT_MODULES = {
    'test_nn_AdaptiveAvgPool2d_tuple_none',
    'test_nn_AdaptiveAvgPool3d_tuple_none',
    'test_nn_AdaptiveMaxPool2d_tuple_none',
    'test_nn_AdaptiveMaxPool3d_tuple_none',
}

DISABLE_AUTODIFF_SUBGRAPH_INLINING = {
    'test_nn_avg_pool2d',
    'test_nn_adaptive_avg_pool2d',
    'test_nn_log_softmax',
    'test_nn_threshold',
    'test_nn_nll_loss',
}


# make a new function where all non-tensor arguments in 'args' have been partially
# applied, and all tensor arguments remain.
# used to trace functions when some arguments are not tensors
def partial_apply_nontensors(fn, args, **kwargs):
    source = ['t' if isinstance(arg, torch.Tensor) else 's' for arg in args]

    def new_fn(*tensors_):
        tensors = iter(tensors_)
        return fn(*(args[i] if s == 's' else next(tensors) for i, s in enumerate(source)), **kwargs)

    return new_fn, [arg for arg in args if isinstance(arg, torch.Tensor)]


# create a trace function from input fn
#
# disable_autodiff_subgraph_inlining:
#   Don't inline autodiff subgraphs so we can test autodiff
def create_traced_fn(self, fn,
                     disable_autodiff_subgraph_inlining=False):
    def traced_fn(*inputs, **kwargs):
        fn_tensors, inputs_tensors = partial_apply_nontensors(fn, inputs, **kwargs)
        traced = torch.jit.trace(fn_tensors, inputs_tensors)
        self.assertExportImport(traced.graph, inputs_tensors)
        if disable_autodiff_subgraph_inlining:
            traced.debug_disable_autodiff_subgraph_inlining()
        output = traced(*inputs_tensors)
        traced_fn.last_graph = traced.graph_for(*inputs_tensors)
        return output
    return traced_fn

script_template = '''
def the_method({}):
    return {}
'''

script_method_template = '''
def forward({}):
    return {}
'''


def get_constant(x):
    if x == inf:
        return 'float(\'inf\')' if PY2 else 'math.inf'
    if x == -inf:
        return 'float(\'-inf\')' if PY2 else '-math.inf'
    return x


def get_script_args(args):
    formals = []
    tensors = []
    actuals = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            name = 'i{}'.format(len(formals))
            formals.append(name)
            actuals.append(name)
            tensors.append(arg)
        elif isinstance(arg, str):
            actuals.append("'{}'".format(arg))
        else:
            actuals.append(str(get_constant(arg)))
    return (formals, tensors, actuals)


# create a script function from (name, func_type, output_process_fn),
# returns a function takes in (args, kwargs) and runs the compiled function and
# then applies the post process fn to the outputs
def create_script_fn(self, method_name, func_type, output_process_fn,
                     disable_autodiff_subgraph_inlining=False):
    def script_fn(*args, **kwargs):
        formals, tensors, actuals = get_script_args(args)
        kwargs_str = ''
        for k, v in kwargs.items():
            kwargs_str += ', ' + k + '=' + str(v)
        if func_type == 'functional':
            call = 'torch.{}({}{})'.format(method_name, ', '.join(actuals), kwargs_str)
        elif func_type == 'method':
            call = '{}.{}({}{})'.format(actuals[0], method_name, ', '.join(actuals[1:]), kwargs_str)
        elif func_type == 'nn_functional':
            call = 'torch.nn.functional.{}({}{})'.format(method_name, ', '.join(actuals), kwargs_str)
        else:
            raise 'Unsupported function type'

        script = script_template.format(', '.join(formals), call)

        CU = torch.jit.CompilationUnit(script)
        if disable_autodiff_subgraph_inlining:
            CU.the_method.debug_disable_autodiff_subgraph_inlining()
        self.assertExportImport(CU.the_method.graph, tensors)
        output = output_process_fn(CU.the_method(*tensors))
        script_fn.last_graph = CU.the_method.graph_for(*tensors)
        return output
    return script_fn


def check_alias_annotation(method_name, args, kwargs):
    formals, tensors, actuals = get_script_args(args)
    kwargs_str = ''
    for k, v in kwargs.items():
        kwargs_str += ', ' + k + '=' + str(v)
    call = '{}.{}({}{})'.format(actuals[0], method_name, ', '.join(actuals[1:]), kwargs_str)
    script = script_template.format(', '.join(formals), call)
    CU = torch.jit.CompilationUnit(script)
    torch._C._jit_check_alias_annotation(CU.the_method.graph, tuple(tensors), method_name)


def check_output_types(self, func, ref_outputs, args, kwargs):
    graph = getattr(func, 'last_graph', None)
    types = [o.type() for o in graph.outputs()]
    self.assertTrue(len(types) == 1)
    t = types[0]
    torch._C._jit_assert_is_instance(ref_outputs, t)


def check_against_reference(self, func, reference_func, args, kwargs=None,
                            allow_unused=True, check_types=True, no_grad=False):
    kwargs = kwargs if kwargs else {}

    def allSum(vs):
        if isinstance(vs, torch.Tensor):
            vs = (vs,)
        return sum((i + 1) * v.sum()
                   for i, v in enumerate(vs)
                   if v is not None and v.dtype.is_floating_point)

    def clone_inputs(requires_grad):
        inputs = [
            arg.detach().clone().requires_grad_(requires_grad and arg.requires_grad)
            if isinstance(arg, torch.Tensor) else arg for arg in args
        ]
        return inputs, [input for input in inputs if isinstance(input, torch.Tensor) and input.requires_grad]

    nograd_inputs, nograd_tensors = clone_inputs(False)
    recording_inputs, recording_tensors = clone_inputs(True)

    # test no gradients case
    outputs = self.runAndSaveRNG(reference_func, nograd_inputs, kwargs)
    outputs_test = self.runAndSaveRNG(func, nograd_inputs, kwargs)
    self.assertEqual(outputs, outputs_test)

    if check_types:
        check_output_types(self, func, outputs_test, nograd_inputs, kwargs)

    if no_grad:
        # skip grad tests
        return

    # test single grad case
    outputs = self.runAndSaveRNG(reference_func, recording_inputs, kwargs)
    grads = torch.autograd.grad(allSum(outputs), recording_tensors,
                                allow_unused=allow_unused)

    outputs_test = self.runAndSaveRNG(func, recording_inputs, kwargs)
    grads_test = torch.autograd.grad(allSum(outputs_test), recording_tensors,
                                     allow_unused=allow_unused)
    self.assertEqual(outputs, outputs_test)
    self.assertEqual(grads, grads_test)

    # test the grad grad case
    if self._testMethodName in nn_functional_single_grad:
        return

    outputs = self.runAndSaveRNG(reference_func, recording_inputs, kwargs)
    l1 = allSum(outputs)
    grads = torch.autograd.grad(l1, recording_tensors, create_graph=True,
                                allow_unused=allow_unused)
    l2 = (allSum(grads) * l1)
    grads2 = torch.autograd.grad(l2, recording_tensors, allow_unused=allow_unused)

    recording_inputs, recording_tensors = clone_inputs(True)

    outputs_test = self.runAndSaveRNG(func, recording_inputs, kwargs)
    l1_test = allSum(outputs_test)
    grads_test = torch.autograd.grad(
        l1_test, recording_tensors, create_graph=True, allow_unused=allow_unused)
    l2_test = (allSum(grads_test) * l1_test)
    grads2_test = torch.autograd.grad(l2_test, recording_tensors, allow_unused=allow_unused)

    self.assertEqual(outputs, outputs_test)
    self.assertEqual(grads, grads_test)
    for g2, g2_test in zip(grads2, grads2_test):
        if g2 is None and g2_test is None:
            continue
        self.assertTrue(torch.allclose(g2, g2_test, atol=5e-4, rtol=1e-4))


class TestFuser(JitTestCase):
    def assertAllFused(self, graph, except_for=()):
        if [n.kind() for n in graph.nodes()] == ['prim::DifferentiableGraph']:
            graph = next(graph.nodes()).g('Subgraph')
        allowed_nodes = {'prim::Constant', 'prim::FusionGroup'} | set(except_for)
        self.assertTrue(all(node.kind() in allowed_nodes for node in graph.nodes()),
                        'got {}'.format(graph))
        self.assertTrue([node.kind() for node in graph.nodes()].count('prim::FusionGroup') == 1)

    def _test_fused_abs(self, device='cpu'):

        @torch.jit.script
        def func(x):
            return x.abs() * 2

        a = torch.randn(5, device=device)
        self.assertEqual(func(a), a.abs() * 2)
        self.assertAllFused(func.graph_for(a))

    @unittest.skipIf(IS_WINDOWS or IS_SANDCASTLE, "NYI: fuser support for Windows or Sandcastle")
    @enable_cpu_fuser
    def test_abs_cpu(self):
        self._test_fused_abs()

    @unittest.skipIf(IS_WINDOWS, "NYI: fuser support for Windows")
    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    @skipIfRocm
    def test_abs_cuda(self):
        self._test_fused_abs(device="cuda")

    @unittest.skipIf(IS_WINDOWS, "NYI: fuser support for Windows")
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_arg_configurations_smoke_cuda(self):
        # A smoke test to make sure we won't use the same kernel for contiguous
        # and non-contiguous arguments.
        # TODO: add optionally enabled debug counters to the fuser to verify
        #       that we really can tell the difference between configurations
        def f(x, y):
            z1, z2 = (x + y).chunk(2, dim=1)
            return z1 * z2

        x = torch.randn(4, 4, dtype=torch.float, device='cuda')
        y = torch.randn(4, 4, dtype=torch.float, device='cuda')
        traced_f = torch.jit.trace(f, (x, y,))
        self.assertEqual(traced_f(x.t().contiguous(), y), traced_f(x.t(), y))

    @unittest.skipIf(IS_WINDOWS, "NYI: fuser support for Windows")
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @skipIfRocm
    def test_broadcast_cuda(self):
        def scaleshift(x, scale, shift):
            return x * scale + shift

        inputs = [
            torch.randn(4, 4, dtype=torch.float, device='cuda'),
            torch.randn(4, dtype=torch.float, device='cuda'),
            torch.randn(4, dtype=torch.float, device='cuda'),
        ]
        ge = self.checkTrace(scaleshift, inputs)
        self.assertExpectedGraph(ge.graph_for(*inputs))

    @unittest.skipIf(IS_WINDOWS, "NYI: fuser support for Windows")
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @unittest.skipIf(not RUN_CUDA_HALF, "no half support")
    def test_cuda_half(self):
        x = torch.randn(4, 4, dtype=torch.half, device='cuda')
        y = torch.randn(4, 4, dtype=torch.half, device='cuda')

        funcs = [
            self.fn_test_comparison_gt_lt,
            self.fn_test_relu,
            self.fn_test_exp
        ]

        # Note: Non fused inputs must be float to prevent loss of precision
        inputs = (x.float(), y.float())
        fusion_inputs = (x, y)
        for fn in funcs:
            local_inputs = [t.clone().requires_grad_() for t in inputs]
            local_fusion_inputs = [t.clone().requires_grad_() for t in fusion_inputs]

            # Verifies outputs
            fusion = torch.jit.trace(fn, local_fusion_inputs, check_trace=False, optimize=True)
            outputs = fn(*local_inputs)
            fusion_outputs = fusion(*local_fusion_inputs)
            outputs_half = [t.half() for t in outputs]
            self.assertEqual(outputs_half, fusion_outputs)

            # Verifies gradients
            for output, fusion_output in zip(outputs_half, fusion_outputs):
                grads = torch.autograd.grad(
                    output.float().sum(), local_inputs, allow_unused=True, retain_graph=True)
                fusion_grads = torch.autograd.grad(
                    fusion_output.sum(), local_fusion_inputs, allow_unused=True, retain_graph=True)
                grads_half = [t.half() for t in grads]
                self.assertEqual(grads_half, fusion_grads)

    @unittest.skipIf(IS_WINDOWS, "NYI: fuser support for Windows")
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @skipIfRocm
    def test_checks_cat_inputs(self):
        # We shouldn't treat cat nodes as broadcasting. All their inputs
        # need to be checked for having the same map size, before we can
        # run the kernel.
        @torch.jit.script
        def f(x, y):
            return torch.cat([x + 2 * x + x ** 2, y + 4 * y + y ** 3], dim=0)

        # NOTE: y is broadcastable to x, but output of f(x, y) should have
        # shape 3x4, and not 4x4.
        x = torch.randn(2, 4, dtype=torch.float, device='cuda')
        y = torch.randn(1, 4, dtype=torch.float, device='cuda')

        self.assertEqual(f(x, y).shape, (3, 4))
        self.assertAllFused(f.graph_for(x, y))

    @unittest.skipIf(IS_WINDOWS, "NYI: fuser support for Windows")
    @unittest.skipIf(not RUN_CUDA, "No CUDA")
    @skipIfRocm
    def test_chunk_cuda(self):
        def fn(x):
            a, b, c = x.chunk(3, 1)
            return a * b + c

        inputs = [torch.randn(10, 6, dtype=torch.float, device='cuda')]

        ge = self.checkScript(fn, inputs)
        self.assertExpectedGraph(ge.graph_for(*inputs))

    @staticmethod
    def _test_chunk_correctness(self, device='cpu'):
        def chunk_4_0(x):
            x0, x1, x2, x3 = x.chunk(4, 0)
            return x0 + x1 + x2 + x3

        def chunk_4_1(x):
            x0, x1, x2, x3 = x.chunk(4, 1)
            return x0 + x1 + x2 + x3

        def chunk_4_last(x):
            x0, x1, x2, x3 = x.chunk(4, 2)
            return x0 + x1 + x2 + x3

        fns = [chunk_4_0, chunk_4_1, chunk_4_last]
        tensors = [
            # splitSize = 1
            torch.randn(4, 4, 4, dtype=torch.float, device=device),

            # contiguous case
            torch.randn(12, 8, 16, dtype=torch.float, device=device),

            # non-contiguous case
            torch.randn(12, 8, 16, dtype=torch.float, device=device).transpose(1, 2),
        ]

        for tensor in tensors:
            for fn in fns:
                self.checkScript(fn, [tensor])

    @unittest.skipIf(IS_WINDOWS or IS_SANDCASTLE, "NYI: fuser support for Windows or Sandcastle")
    @enable_cpu_fuser
    def test_chunk_correctness(self):
        return self._test_chunk_correctness(self, 'cpu')

    @unittest.skipIf(IS_WINDOWS, "NYI: fuser support for Windows")
    @unittest.skipIf(not RUN_CUDA, "No CUDA")
    def test_chunk_correctness_cuda(self):
        return self._test_chunk_correctness(self, 'cuda')

    @unittest.skipIf(IS_WINDOWS, "NYI: fuser support for Windows")
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @skipIfRocm
    def test_chunk_distributes_cuda(self):
        def f(x, y):
            z1, z2 = (x + y).chunk(2, dim=1)
            return z1 * z2

        x = torch.randn(4, 4, dtype=torch.float, device='cuda')
        y = torch.randn(4, 4, dtype=torch.float, device='cuda')

        ge = self.checkTrace(f, (x, y))
        self.assertExpectedGraph(ge.graph_for(x, y))

    @unittest.skipIf(IS_WINDOWS, "NYI: fuser support for Windows")
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @skipIfRocm
    def test_chunk_motion_deduplicates_inputs(self):
        def func1(x):
            z = x * x
            z0, z1 = z.chunk(2)
            return z0 * z1

        def func2(x):
            z = x * x * x
            z0, z1 = z.chunk(2)
            return z0 * z1

        inputs = [
            torch.tensor([1.1, 1.2], device='cuda', dtype=torch.float),
        ]
        for func in [func1, func2]:
            module = self.checkScript(func, inputs)
            forward_graph = module.graph_for(*inputs)
            self.assertGraphContainsExactly(forward_graph, 'prim::FusionGroup', 1)
            fusion_group = list(forward_graph.nodes())[-1]
            self.assertEqual(len(list(fusion_group.inputs())), 1)

    @unittest.skipIf(IS_WINDOWS, "NYI: fuser support for Windows")
    @unittest.skipIf(not RUN_CUDA, "No CUDA")
    @skipIfRocm
    def test_chunk_multiple_cuda(self):
        # The arguments are intentionally used out of order as a test to see
        # if the fusion compiler adds extra args in the correct order
        def fn(s, x, y, z):
            z1, z2 = z.chunk(2, 2)
            x1, x2, x3 = x.chunk(3, 1)
            y1, y2 = y.chunk(2, 0)
            return s + x1 + x2 + x3 + y1 + y2 + z1 + z2

        inputs = [
            torch.randn(5, 2, 3, dtype=torch.float, device='cuda'),
            torch.randn(5, 6, 3, dtype=torch.float, device='cuda'),
            torch.randn(10, 2, 3, dtype=torch.float, device='cuda'),
            torch.randn(5, 2, 6, dtype=torch.float, device='cuda'),
        ]

        ge = self.checkScript(fn, inputs)
        self.assertExpectedGraph(ge.graph_for(*inputs))

    @unittest.skipIf(IS_WINDOWS, "NYI: fuser support for Windows")
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @skipIfRocm
    def test_clamp(self):
        def func2(a, b):
            return torch.clamp(a + b, min=0, max=2)

        def funcInf(a, b):
            return torch.clamp(a + b, min=0, max=float('inf'))

        a = torch.randn(4, 4, dtype=torch.float, device='cuda', requires_grad=True)
        b = torch.randn(4, 4, dtype=torch.float, device='cuda')

        funcs = (func2, funcInf)
        for f in funcs:
            s = self.checkScript(f, (a, b))
            self.assertAllFused(s.graph_for(a, b), except_for={'aten::size'})

            c = s(a, b)
            c.sum().backward()
            graph = backward_graph(s)
            self.assertAllFused(graph)

    @unittest.skipIf(IS_WINDOWS, "NYI: fuser support for Windows")
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @skipIfRocm
    def test_comparison_eq_ne(self):
        def f(x, y):
            mask = (x == 0).type_as(x)
            z = x * mask + y
            mask = (x != 0).type_as(x)
            z = z * mask + y
            return z

        x = torch.randn(4, 4, dtype=torch.float, device='cuda')
        y = torch.randn(4, 4, dtype=torch.float, device='cuda')

        ge = self.checkTrace(f, (x, y))
        self.assertAllFused(ge.graph_for(x, y))

    @staticmethod
    def fn_test_comparison_gt_lt(x, y):
        mask = (x > 0).type_as(x)
        z = x * mask + y
        mask = (x < 0).type_as(x)
        z = z * mask + y
        return z

    @unittest.skipIf(IS_WINDOWS, "NYI: fuser support for Windows")
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @skipIfRocm
    def test_comparison_gt_lt_cuda(self):
        x = torch.randn(4, 4, dtype=torch.float, device='cuda')
        y = torch.randn(4, 4, dtype=torch.float, device='cuda')

        ge = self.checkTrace(self.fn_test_comparison_gt_lt, (x, y))
        self.assertAllFused(ge.graph_for(x, y))

    @unittest.skipIf(IS_WINDOWS, "NYI: fuser support for Windows")
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @skipIfRocm
    def test_comparison_ge_le_cuda(self):
        def f(x, y):
            mask = (x >= 0).type_as(x)
            z = x * mask + y
            mask = (x <= 0).type_as(x)
            z = z * mask + y
            return z

        x = torch.randn(4, 4, dtype=torch.float, device='cuda')
        y = torch.randn(4, 4, dtype=torch.float, device='cuda')

        ge = self.checkTrace(f, (x, y))
        self.assertAllFused(ge.graph_for(x, y))
        x.requires_grad_(True)
        y.requires_grad_(True)
        self.assertAllFused(ge.graph_for(x, y), except_for=("aten::size", "prim::BroadcastSizes"))

    @unittest.skipIf(IS_WINDOWS, "NYI: fuser support for Windows")
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @skipIfRocm
    def test_concat_cuda(self):
        hx = torch.randn(3, 20, dtype=torch.float, device='cuda')
        cx = torch.randn(3, 20, dtype=torch.float, device='cuda')

        def foo(hx, cx):
            return torch.cat((hx + cx, hx * cx))

        ge = self.checkTrace(foo, (hx, cx))
        self.assertExpectedGraph(ge.graph_for(hx, cx))

    @unittest.skipIf(IS_WINDOWS, "NYI: fuser support for Windows")
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @skipIfRocm
    def test_concat_invariant_cuda(self):
        # Invariant: the output of prim::FusedConcat may
        # not be an input to any node inside the FusionGroup.
        def fn(x, y, z):
            x1 = x + y
            y1 = x - y
            w = torch.cat([x1, y1])
            return w + z

        x = torch.randn(2, 2, dtype=torch.float, device='cuda')
        y = torch.randn(2, 2, dtype=torch.float, device='cuda')
        z = torch.randn(4, 2, dtype=torch.float, device='cuda')
        ge = self.checkTrace(fn, (x, y, z))
        self.assertExpectedGraph(ge.graph_for(x, y, z))

    @staticmethod
    def fn_test_exp(x, y):
        return (x + .5 * y).exp()

    @unittest.skipIf(IS_WINDOWS, "NYI: fuser support for Windows")
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @skipIfRocm
    def test_exp_cuda(self):
        x = torch.randn(4, 4, dtype=torch.float, device='cuda')
        y = torch.randn(4, 4, dtype=torch.float, device='cuda')

        ge = self.checkTrace(self.fn_test_exp, (x, y))
        self.assertAllFused(ge.graph_for(x, y))

    @unittest.skipIf(IS_WINDOWS, "NYI: fuser support for Windows")
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @skipIfRocm
    def test_fuse_batch_norm(self):

        class ResLike(torch.jit.ScriptModule):
            def __init__(self, optimize=True):
                super(ResLike, self).__init__(optimize)
                self.bn = nn.BatchNorm2d(16)

            @torch.jit.script_method
            def forward(self, x, y):
                return y + torch.relu(self.bn(x))

        model = ResLike().cuda()
        model_noopt = ResLike(optimize=False).cuda()
        model_noopt.load_state_dict(model.state_dict())
        x = torch.randn(2, 16, 8, 8, device='cuda')
        y = torch.randn(2, 16, 8, 8, device='cuda')
        # FIXME: We need differentiation for CNNs for this optimization to trigger
        with torch.no_grad():
            out = model(x, y)
            graph = model.graph_for(x, y)
            rep = str(graph)

            out_noopt = model_noopt(x, y)
            rep_noopt = str(model_noopt.graph_for(x, y))
            self.assertEqual(out, out_noopt, prec=3e-5)

        # Check that batch_norm has really been decomposed
        self.assertIn('aten::batch_norm_update_stats', rep)
        self.assertNotIn('aten::batch_norm(', rep)
        self.assertIn('aten::batch_norm(', rep_noopt)

        # Make sure the fusion group is big, and contains aten::sqrt, which could
        # originate only from decomposing batch_norm in this case
        fusion_groups = [node for node in graph.nodes() if node.kind() == 'prim::FusionGroup']
        self.assertEqual(len(fusion_groups), 1)
        fused_graph = fusion_groups[0].g('Subgraph')
        self.assertTrue(any(node.kind() == 'aten::sqrt' for node in fused_graph.nodes()))

    @unittest.skipIf(IS_WINDOWS, "NYI: fuser support for Windows")
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @skipIfRocm
    def test_threshold(self):
        def f(x):
            return torch.threshold(x, 0, -10) + x + x + x

        x = torch.tensor([-1, -0.5, 0, 1, 2, 3], device='cuda')
        scripted = torch.jit.script(f)

        self.assertEqual(f(x), scripted(x))
        self.assertAllFused(scripted.graph_for(x))

    @unittest.skipIf(IS_WINDOWS or IS_SANDCASTLE, "NYI: fuser support for Windows or Sandcastle")
    @enable_cpu_fuser
    def test_fuser_deduplication(self):
        # See that fusion kernel outputs are deduplicated when removing  _grad_sum_to_size in the fuser's compilation
        # see the discussion in PR #14957.
        def f(x, y):
            return torch.sigmoid(x + y)

        b = torch.randn(5, 5, requires_grad=True)
        a = torch.randn(5, 5, requires_grad=True)
        s = self.checkScript(f, (a, b))
        self.assertAllFused(s.graph_for(a, b), except_for={'aten::size'})

        c = s(a, b)
        ga, gb = torch.autograd.grad(c.sum(), [a, b])
        graph = backward_graph(s)
        self.assertAllFused(graph)
        # check that a, b share storage, i.e. were generated as a single output in the fuser
        self.assertEqual(ga.data_ptr(), gb.data_ptr())

    @unittest.skipIf(IS_WINDOWS or IS_SANDCASTLE, "NYI: fuser support for Windows or Sandcastle")
    @enable_cpu_fuser
    def test_fuser_iou(self):
        # This checks if most of Intersection over Union is fused.
        # In particular, the backward contains many _grad_sum_to_size.
        def iou(b1x1, b1y1, b1x2, b1y2, b2x1, b2y1, b2x2, b2y2):
            ltx = torch.max(b1x1, b2x1)  # [N,M]
            lty = torch.max(b1y1, b2y1)
            rbx = torch.min(b1x2, b2x2)
            rby = torch.min(b1y2, b2y2)

            w = (rbx - ltx).clamp(min=0, max=float('inf'))  # [N,M]
            h = (rby - lty).clamp(min=0, max=float('inf'))  # [N,M]
            inter = w * h  # [N,M]

            area1 = (b1x2 - b1x1) * (b1y2 - b1y2)  # [N,1]
            area2 = (b2x2 - b2x1) * (b2y2 - b2y2)  # [1,M]
            iou = inter / (area1 + area2 - inter)
            return iou

        box1 = torch.randn(5, 4, requires_grad=True)
        box2 = torch.randn(5, 4, requires_grad=True)
        # unsqueezing can currently not be fused
        b1x1 = box1[:, 0].unsqueeze(1)  # [N,1]
        b1y1 = box1[:, 1].unsqueeze(1)
        b1x2 = box1[:, 2].unsqueeze(1)
        b1y2 = box1[:, 3].unsqueeze(1)
        b2x1 = box2[:, 0].unsqueeze(0)  # [1,N]
        b2y1 = box2[:, 1].unsqueeze(0)
        b2x2 = box2[:, 2].unsqueeze(0)
        b2y2 = box2[:, 3].unsqueeze(0)

        s = self.checkScript(iou, (b1x1, b1y1, b1x2, b1y2, b2x1, b2y1, b2x2, b2y2))
        self.assertAllFused(s.graph_for(b1x1, b1y1, b1x2, b1y2, b2x1, b2y1, b2x2, b2y2),
                            except_for={'aten::size', 'prim::BroadcastSizes'})

        c = s(b1x1, b1y1, b1x2, b1y2, b2x1, b2y1, b2x2, b2y2)
        torch.autograd.grad(c.sum(), [b1x1, b1y1, b1x2, b1y2, b2x1, b2y1, b2x2, b2y2])
        graph = backward_graph(s)
        self.assertAllFused(graph, except_for={'aten::size', 'prim::BroadcastSizes'})

    @unittest.skipIf(IS_WINDOWS, "NYI: fuser support for Windows")
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @unittest.skipIf(not RUN_CUDA_MULTI_GPU, "needs non-zero device")
    @skipIfRocm
    @enable_cpu_fuser
    def test_fusion_reuse_multi_gpu(self):
        def fn(x, y):
            return x * y * x * y

        inputs_cpu = [
            torch.randn(4, 4, dtype=torch.float),
            torch.randn(4, 4, dtype=torch.float),
        ]
        inputs_cuda0 = [x.cuda(0) for x in inputs_cpu]
        inputs_cuda1 = [y.cuda(1) for y in inputs_cpu]

        # Should not crash; these should compile different kernels.
        ge = self.checkScript(fn, inputs_cpu)
        self.assertAllFused(ge.graph_for(*inputs_cpu))
        ge(*inputs_cuda0)
        ge(*inputs_cuda1)

    @unittest.skipIf(IS_WINDOWS, "NYI: fuser support for Windows")
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @unittest.skipIf(not RUN_CUDA_MULTI_GPU, "needs non-zero device")
    @skipIfRocm
    @enable_cpu_fuser
    def test_kernel_cache_multi_gpu(self):
        def not_fusible(x):
            return x

        def fn(x, y, z):
            x_out = x * x * x * x * x  # fusion: lambda x. x * x * x * x * x
            y_out = y * y * y * y * y
            z_out = z * z * z * z * z
            return not_fusible(x_out), not_fusible(y_out), not_fusible(z_out)

        inputs = [
            torch.randn(4, 4, dtype=torch.float),
            torch.randn(4, 4, dtype=torch.float, device='cuda:0'),
            torch.randn(4, 4, dtype=torch.float, device='cuda:1'),
        ]

        prev_cache_size = torch._C._jit_debug_fuser_num_cached_kernel_specs()

        # There are 3 FusionGroups. Because they have the same graph, they
        # should reuse the same KernelSpec in the KernelSpec cache.
        ge = self.checkScript(fn, inputs)
        self.assertGraphContainsExactly(
            ge.graph_for(*inputs), 'prim::FusionGroup', 3, True)
        new_cache_size = torch._C._jit_debug_fuser_num_cached_kernel_specs()
        # XXX: This assumes that the same kernel isn't already used by another test
        self.assertEqual(new_cache_size - prev_cache_size, 1)

    @unittest.skipIf(IS_WINDOWS, "NYI: fuser support for Windows")
    @unittest.skipIf(not RUN_CUDA_MULTI_GPU, "needs non-zero device")
    @skipIfRocm
    def test_nonzero_device_cuda(self):
        device = 'cuda:' + str(1)
        x = torch.tensor([0.4], dtype=torch.float, device=device)
        y = torch.tensor([0.7], dtype=torch.float, device=device)

        def doit(x, y):
            return torch.sigmoid(torch.tanh(x * (x + y) + x))

        ge = self.checkTrace(doit, (x, y))
        self.assertAllFused(ge.graph_for(x, y))

    @unittest.skipIf(IS_WINDOWS, "NYI: fuser support for Windows")
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @skipIfRocm
    def test_lstm_cuda(self):
        inputs = get_lstm_inputs('cuda', training=True)
        module = self.checkScript(LSTMCellS, inputs)
        forward_graph = module.graph_for(*inputs)
        self.assertGraphContainsExactly(
            forward_graph, 'prim::FusionGroup', 1, consider_subgraphs=True)
        self.assertExpectedGraph(forward_graph, subname='forward')

        hy, cy = module(*inputs)
        (hy + cy).sum().backward()
        self.assertExpectedGraph(backward_graph(module), subname='backward')

    @unittest.skipIf(IS_WINDOWS, "NYI: fuser support for Windows")
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @skipIfRocm
    def test_lstm_concat_cuda(self):
        inputs = get_lstm_inputs('cuda')
        ge = self.checkTrace(LSTMCellC, inputs)
        self.assertExpectedGraph(ge.graph_for(*inputs))

    @unittest.skipIf(IS_WINDOWS, "NYI: fuser support for Windows")
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @skipIfRocm
    def test_lstm_gates_permutations_cuda(self):
        # lstm has gates = x.mm(w_ih.t()) + hx.mm(w_hh.t()) + b_ih + b_hh.
        # Test that any permutation of this will still result in one FusionGroup.
        choices = ['x.mm(w_ih.t())', 'hx.mm(w_hh.t())', 'b_ih', 'b_hh']
        template = dedent('''
        def cell(x, hx, cx, w_ih, w_hh, b_ih, b_hh):
            gates = {} + {} + {} + {}
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
            return ingate * forgetgate * cellgate * outgate
        ''')
        for permutation in itertools.permutations(choices, len(choices)):
            code = template.format(*permutation)
            scope = {}
            exec(code, globals(), scope)
            cu = torch.jit.CompilationUnit(code)

            inputs = get_lstm_inputs('cuda', training=False)
            self.assertEqual(cu.cell(*inputs), scope['cell'](*inputs))
            forward_graph = cu.cell.graph_for(*inputs)
            self.assertGraphContainsExactly(forward_graph, 'prim::FusionGroup', 1)

    # TODO: Fuser doesn't work at all when inputs require grad. Fix that
    @unittest.skipIf(IS_WINDOWS, "NYI: fuser support for Windows")
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @skipIfRocm
    def test_lstm_traced_cuda(self):
        inputs = get_lstm_inputs('cuda')
        ge = self.checkTrace(LSTMCellF, inputs)
        self.assertExpectedGraph(ge.graph_for(*inputs))

    @unittest.skipIf(IS_WINDOWS or IS_SANDCASTLE, "NYI: fuser support for Windows or Sandcastle")
    @unittest.skip("Test is flaky, see https://github.com/pytorch/pytorch/issues/8746")
    @enable_cpu_fuser
    def test_lstm_traced_cpu(self):
        inputs = get_lstm_inputs('cpu')
        try:
            ge = self.checkTrace(LSTMCellF, inputs)
            self.assertExpectedGraph(ge.graph_for(*inputs))
        except RuntimeError as e:
            if 'Failed to compile' in e.args[0]:
                warnings.warn('CPU fuser test has failed! This is not a hard failure, '
                              'because the kernels sometimes trigger bugs in compilers '
                              '(most notably GCC 7.2).')
                raise unittest.SkipTest('Failed to compile')
            else:
                raise

    @unittest.skipIf(IS_WINDOWS, "NYI: fuser support for Windows")
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @skipIfRocm
    def test_milstm_cuda(self):
        inputs = get_milstm_inputs('cuda', training=True)
        module = self.checkScript(MiLSTMCell, inputs)
        forward_graph = module.graph_for(*inputs)
        self.assertGraphContainsExactly(
            forward_graph, 'prim::FusionGroup', 1, consider_subgraphs=True)
        self.assertExpectedGraph(forward_graph, subname='forward')

        hy, cy = module(*inputs)
        (hy + cy).sum().backward()
        self.assertExpectedGraph(backward_graph(module), subname='backward')

    # TODO: At some point we supported fusion of torch.rand_like but not anymore
    @unittest.expectedFailure
    @unittest.skipIf(IS_WINDOWS, "NYI: fuser support for Windows")
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @skipIfRocm
    def test_rand_cuda(self):
        class M(torch.jit.ScriptModule):
            __constants__ = ['d']

            def __init__(self):
                self.d = torch.device('cuda')

            @torch.jit.script_method
            def create(self, x):
                return x * x + x + torch.rand_like(x)

        x = torch.zeros([3, 4, 5], dtype=torch.float, device='cuda')
        m = M()
        out1 = m.create(x)
        out2 = m.create(x)
        self.assertNotEqual(out1, out2)
        self.assertTrue(torch.all(out1 >= 0))
        self.assertTrue(torch.all(out1 < 1))
        self.assertTrue(torch.all(out2 >= 0))
        self.assertTrue(torch.all(out2 < 1))
        self.assertAllFused(m.create.graph_for(x))

    @staticmethod
    def fn_test_relu(x, y):
        return F.relu(x + .5 * y)

    @unittest.skipIf(IS_WINDOWS, "NYI: fuser support for Windows")
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @skipIfRocm
    def test_relu_cuda(self):
        x = torch.randn(4, 4, dtype=torch.float, device='cuda')
        y = torch.randn(4, 4, dtype=torch.float, device='cuda')

        ge = self.checkTrace(self.fn_test_relu, (x, y))
        self.assertAllFused(ge.graph_for(x, y))

    @staticmethod
    def fn_test_erf(x):
        return F.relu(torch.erf(x) - torch.erfc(x))

    @unittest.skipIf(IS_WINDOWS, "NYI: fuser support for Windows")
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @skipIfRocm
    def test_erf_cuda(self):
        x = torch.randn(4, 4, dtype=torch.float, device='cuda')
        ge = self.checkTrace(self.fn_test_erf, (x,))
        self.assertAllFused(ge.graph_for(x))
        x.requires_grad_(True)
        self.assertAllFused(ge.graph_for(x), except_for=("aten::size", "prim::BroadcastSizes"))

    @unittest.skipIf(IS_WINDOWS or IS_SANDCASTLE, "NYI: fuser support for Windows or Sandcastle")
    @enable_cpu_fuser
    def test_scalar(self):
        def fn(x, y):
            return 2 * x + y

        x = torch.tensor(0.1, dtype=torch.float, device='cpu')
        y = torch.tensor(1, dtype=torch.float, device='cpu')
        ge = self.checkScript(fn, (x, y))
        self.assertExpectedGraph(ge.graph_for(x, y))

    @unittest.skipIf(IS_WINDOWS, "NYI: fuser support for Windows")
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @skipIfRocm
    def test_small_constant_cuda(self):
        def fn_test_small_constant(x, y):
            return (1e-8 * x + 5e-9 * y) * 1e8
        x = torch.randn(4, 4, dtype=torch.float, device='cuda')
        y = torch.randn(4, 4, dtype=torch.float, device='cuda')

        ge = self.checkTrace(fn_test_small_constant, (x, y))
        self.assertAllFused(ge.graph_for(x, y))

    @unittest.skipIf(IS_WINDOWS, "NYI: fuser support for Windows")
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @skipIfRocm
    def test_tensor_scalar_ops_cuda(self):
        def should_fuse(x):
            z = 3.
            y = x + z
            return x * y

        # XXX: right now we only support fusing scalars if
        # they're constant (#9940)
        def should_not_fuse(x, z):
            y = x + int(z)
            return x * y

        inputs = [torch.randn(2, 2, dtype=torch.float, device='cuda')]
        ge = self.checkScript(should_fuse, inputs)
        self.assertAllFused(ge.graph_for(*inputs))

        inputs = [
            torch.randn(2, 2, dtype=torch.float, device='cuda'),
            torch.tensor(3., dtype=torch.float, device='cuda'),
        ]
        ge = self.checkScript(should_not_fuse, inputs)
        self.assertGraphContainsExactly(
            ge.graph_for(*inputs), 'prim::FusionGroup', 0, consider_subgraphs=True)

    @unittest.skipIf(IS_WINDOWS or IS_SANDCASTLE, "NYI: fuser support for Windows or Sandcastle")
    @enable_cpu_fuser
    def test_where_and_typing(self):
        def f(x, y):
            mask = x > y
            res = torch.where(mask, x, y)
            return mask, res

        script_f = torch.jit.script(f)

        x = torch.randn(4, 4, dtype=torch.double)
        y = torch.randn(4, 4, dtype=torch.double)

        result1, result2 = script_f(x, y)
        expected1, expected2 = f(x, y)
        self.assertEqual(result1, expected1)
        self.assertEqual(result2, expected2)
        self.assertAllFused(script_f.graph_for(x, y), except_for={'prim::TupleConstruct'})

    @unittest.skipIf(not IS_WINDOWS, "Test that the fuser is disabled on Windows")
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_windows_cuda(self):
        def scaleshift(x, scale, shift):
            return x * scale + shift

        inputs = [
            torch.randn(4, 4, dtype=torch.float, device='cuda'),
            torch.randn(4, dtype=torch.float, device='cuda'),
            torch.randn(4, dtype=torch.float, device='cuda'),
        ]

        ge = self.checkScript(scaleshift, inputs)
        self.assertGraphContainsExactly(
            ge.graph_for(*inputs), 'prim::FusionGroup', 0, consider_subgraphs=True)


# NB: torch.jit.script, when used as a function, uses the current scope
# to resolve variable names. This function cannot be made local to
# TestAutodiffSubgraphSlicing because those tests call torch.jit.script on functions
# in a different scope than they are defined in.
def pyfn(a, b):
    return a * b


class TestAutodiffSubgraphSlicing(JitTestCase):
    # TODO: It is better if we can test directly on graphs instead of the current
    # end-to-end fashion.
    def _perform_ad_subgraph_slicing(self, fn, *input_sizes):
        ge = torch.jit.script(fn)
        ge.debug_disable_autodiff_subgraph_inlining()
        inputs = [torch.randn(size, requires_grad=True) for size in input_sizes]
        ge(*inputs)
        return ge.graph_for(*inputs)

    def assertGraphSize(self, graph, size):
        self.assertEqual(len(list(graph.nodes())), size)

    def test_simple_merge(self):
        # o --> o
        def fn(x, y, z):
            a = x * y
            b = a * z
            return b

        graph = self._perform_ad_subgraph_slicing(fn, 1, 1, 1)

        self.assertGraphSize(graph, 1)
        self.assertGraphContainsExactly(graph, 'prim::DifferentiableGraph', 1)

    def test_simple_no_merge(self):
        # o: autodiff supported. x: not autodiff supported.
        # o --> x
        def fn(x, y, z):
            a = x * y
            b = pyfn(a, z)
            return b

        graph = self._perform_ad_subgraph_slicing(fn, 1, 1, 1)

        self.assertGraphSize(graph, 2)
        self.assertGraphContainsExactly(graph, 'prim::DifferentiableGraph', 1)

    def test_does_not_merge_unrelated(self):
        # o  o
        def fn(w, x, y, z):
            a = x * y
            b = w * z
            return a, b

        graph = self._perform_ad_subgraph_slicing(fn, 1, 1, 1, 1)

        self.assertGraphSize(graph, 3)
        self.assertGraphContainsExactly(graph, 'prim::DifferentiableGraph', 2)

    def test_merges_without_cycles(self):
        # o --> o --> o
        # |           ^
        #  \_________/
        def fn(w, x, y):
            a = w * x
            b = a * y
            c = a * b
            return c

        graph = self._perform_ad_subgraph_slicing(fn, 1, 1, 1)

        self.assertGraphSize(graph, 1)
        self.assertGraphContainsExactly(graph, 'prim::DifferentiableGraph', 1)

    def test_merges_dense(self):
        #   o      o
        #   |\    /|
        #   | \  / |
        #   |  /\  |
        #   vv    vv
        #   o      o
        def fn(x, y):
            a, b = x.chunk(2)
            c, d = y.chunk(2)
            return a + c, b + d

        graph = self._perform_ad_subgraph_slicing(fn, 2, 2)

        self.assertGraphSize(graph, 2)
        self.assertGraphContainsExactly(graph, 'prim::DifferentiableGraph', 1)

    def test_does_not_create_cycles(self):
        # o --> x --> o
        # |           ^
        #  \_________/
        def fn(w, x, y):
            a = w * x
            b = pyfn(a, y)
            c = a * b
            return c

        graph = self._perform_ad_subgraph_slicing(fn, 1, 1, 1)

        self.assertGraphSize(graph, 3)
        self.assertGraphContainsExactly(graph, 'prim::DifferentiableGraph', 2)

    def test_merges_up(self):
        # o --> x     o
        # |           ^
        #  \_________/
        def fn(w, x, y, z):
            a = w * x
            b = pyfn(a, y)
            c = a * z
            return b, c

        graph = self._perform_ad_subgraph_slicing(fn, 1, 1, 1, 1)

        self.assertGraphSize(graph, 3)
        self.assertGraphContainsExactly(graph, 'prim::DifferentiableGraph', 1)

    def test_merges_down(self):
        # o     x --> o
        # |           ^
        #  \_________/
        def fn(v, w, x, y):
            a = v * w
            b = pyfn(x, y)
            c = b * a
            return a, c

        graph = self._perform_ad_subgraph_slicing(fn, 1, 1, 1, 1)

        self.assertGraphSize(graph, 3)
        self.assertGraphContainsExactly(graph, 'prim::DifferentiableGraph', 1)

    def test_respects_lexical_scoping(self):
        def fn(x, k):
            y = x * 1.1
            if bool(k):
                k = k + y
            z = y * k
            return z, k

        graph = self._perform_ad_subgraph_slicing(fn, 1, 1)

        # We should not have combined the two multiplications into
        # the same group; they should each be a separate DiffGraph
        self.assertGraphContainsExactly(graph, 'prim::DifferentiableGraph', 2)


class TestCustomOperators(JitTestCase):

    def test_dynamic_op_registry(self):
        from torch._ops import _OpNamespace
        self.assertTrue(hasattr(torch, 'ops'))

        if '_test' in torch.ops.__dict__:
            torch.ops.__dict__.pop('_test')

        # Don't use `hasattr()` because it will call `__getattr__`.
        self.assertNotIn('_test', torch.ops.__dict__)
        torch.ops._test
        self.assertIn('_test', torch.ops.__dict__)
        self.assertEqual(type(torch.ops._test), _OpNamespace)

        self.assertNotIn('leaky_relu', torch.ops._test.__dict__)
        op = torch.ops._test.leaky_relu
        self.assertTrue(callable(op))
        self.assertIn('leaky_relu', torch.ops._test.__dict__)
        op2 = torch.ops._test.leaky_relu
        self.assertEqual(op, op2)

    def test_simply_calling_an_operator(self):
        input = torch.randn(100)
        output = torch.ops.aten.relu(input)
        self.assertEqual(output, input.relu())

    def test_default_arguments_are_used(self):
        output = torch.ops._test.leaky_relu(torch.tensor([-1.0, 1.0]))
        self.assertEqual(output, torch.tensor([-0.01, 1]))

    def test_only_kwargs(self):
        output = torch.ops._test.leaky_relu(self=torch.tensor(-1.0))
        self.assertEqual(output, torch.tensor(-0.01))

    def test_passing_too_many_args(self):
        with self.assertRaisesRegex(
            RuntimeError,
            r"aten::relu\(\) expected at most 1 argument\(s\) but received 2 argument\(s\)"
        ):
            torch.ops.aten.relu(1, 2)

    def test_passing_too_few_args(self):
        with self.assertRaisesRegex(
            RuntimeError,
            r"aten::relu\(\) is missing value for argument 'self'."
        ):
            torch.ops.aten.relu()

    def test_passing_one_positional_but_not_the_second(self):
        with self.assertRaisesRegex(
            RuntimeError,
            r"aten::transpose\(\) is missing value for argument 'dim0'."
        ):
            torch.ops.aten.transpose(torch.ones(5, 5))

    def test_passing_an_argument_both_as_positional_and_kwarg(self):
        with self.assertRaisesRegex(
            RuntimeError,
            "Argument 'self' specified both as positional and keyword argument"
        ):
            torch.ops._test.leaky_relu(torch.ones(5), self=torch.ones(5))

    def test_passing_unknown_kwargs(self):
        with self.assertRaisesRegex(
            RuntimeError,
            "Unknown keyword argument 'foo' for operator '_test::leaky_relu'"
        ):
            torch.ops._test.leaky_relu(torch.ones(5), foo=torch.ones(5))

    def test_passing_and_returning_lists(self):
        # Replace with actual test once we support lists.
        a, b = torch.rand(5), torch.rand(5)
        output = torch.ops._test.cat([a, b])
        output_ref = torch.cat([a, b])
        self.assertEqual(output, output_ref)

    def test_calling_scripted_custom_op(self):
        @torch.jit.script
        def func(x):
            return torch.ops.aten.relu(x)
        input = torch.ones(5, 5)
        self.assertEqual(func(input), input.relu())

    def test_calling_traced_custom_op(self):
        input = torch.ones(5, 5)
        func = torch.jit.trace(torch.ops.aten.relu, [input])
        self.assertEqual(func(input), input.relu())

    def test_script_graph_for_custom_ops_matches_traced_graph(self):
        input = torch.ones(5, 5)
        trace = torch.jit.trace(torch.ops.aten.relu, [input])
        self.assertExpectedInline(canonical(trace.graph), '''\
graph(%0 : Double(5, 5)):
  %1 : Double(5, 5) = aten::relu(%0)
  return (%1)
''')

    def test_script_graph_contains_custom_op(self):
        @torch.jit.script
        def func(x):
            return torch.ops.aten.relu(x)
        self.assertExpectedInline(canonical(func.graph), '''\
graph(%x : Tensor):
  %1 : Tensor = aten::relu(%x)
  return (%1)
''')


class TestJitGeneratedAutograd(JitTestCase):
    pass


class TestJitGeneratedModule(JitTestCase):
    pass


class TestJitGeneratedFunctional(JitTestCase):
    pass


# UBSAN per-function exclusions don't seem to work with OpenMP pragmas,
# and we have to disable the failing tests here instead.
UBSAN_BLACKLISTED_TESTS = [
    "test___rdiv___constant",
    "test___rdiv___scalar_constant",
    "test_addcdiv",
    "test_addcdiv_broadcast_all",
    "test_addcdiv_broadcast_rhs",
    "test_addcdiv_scalar",
    "test_addcdiv_scalar_broadcast_lhs",
    "test_addcdiv_scalar_broadcast_rhs",
    "test_addcdiv_scalar_scale",
    "test_addcdiv_scalar_scale_broadcast_lhs",
    "test_addcdiv_scalar_scale_broadcast_rhs",
    "test_addcdiv_scale",
    "test_addcdiv_scale_broadcast_all",
    "test_addcdiv_scale_broadcast_rhs",
    "test_add_broadcast_all",
    "test_add_broadcast_lhs",
    "test_add_broadcast_rhs",
    "test_add_constant",
    "test_add_scalar",
    "test_add_scalar_broadcast_lhs",
    "test_add_scalar_broadcast_rhs",
    "test_div",
    "test_div_broadcast_all",
    "test_div_broadcast_lhs",
    "test_div_broadcast_rhs",
    "test_div_scalar",
    "test_div_scalar_broadcast_lhs",
    "test_div_scalar_broadcast_rhs",
    "test_rsqrt",
    "test_rsqrt_scalar",
    "test_add",
    "test_reciprocal",
    "test_reciprocal_scalar",
]

L = 20
M = 10
S = 5

#  module cannot be exported /imported currently
EXCLUDE_MODULE_EXPORT_IMPORT = {
    'EmbeddingBag',
    'MaxPool1d',
    'MaxPool2d',
    'MaxPool3d',
    'AdaptiveAvgPool2d',
    'AdaptiveAvgPool3d',
    'Fold',
    'Unfold',
}

# NB: JIT script tests for all nn functional interfaces, script mode does
# not support in_place operations yet, so no inplace operation tests added.
# removed all the deprecated functions
#
# (
#   method name,
#   input size/constructing fn,
#   args (tuple represents shape of a tensor arg),
#   test variant name(will be used at test name suffix,
#       'inplace' skips grad tests),                         // optional
#   fn to determine if test should be skipped,               // optional
#   fn mapping output to part that should be gradcheck'ed,   // optional
#   kwargs for function,                                     // optional
# )
nn_functional_tests = [
    ('conv1d', (S, S, S), ((S, S, S),)),
    ('conv2d', (S, S, S, S), ((S, S, S, S),)),
    ('conv3d', (S, S, S, S, S), ((S, S, S, S, S),)),
    ('conv_transpose1d', (S, S, S), ((S, S, S),)),
    ('conv_transpose2d', (S, S, S, S), ((S, S, S, S),)),
    ('conv_transpose3d', (S, S, S, S, S), ((S, S, S, S, S),)),
    ('conv_tbc', (S, S, S), ((S, S, S), (S,), 2)),
    ('avg_pool1d', (S, S, S), (3,)),
    ('avg_pool2d', (S, S, S, S), (3,)),
    ('avg_pool3d', (S, S, S, S, S), (3,)),
    ('fractional_max_pool2d', (S, S, S, S), (3, [2, 3],)),
    ('max_pool1d', (S, S, S), (2, 1)),
    ('max_pool1d', (S, S, S), (2, 1, 1, 1, False, True), 'with_indices'),
    ('max_pool2d', (S, S, S, S), (2, 1)),
    ('max_pool3d', (S, S, S, S, S), (2, 1)),
    ('max_unpool1d', torch.tensor([[[2., 4]]]), (torch.tensor([[[1, 3]]]), 2, 2, 0)),
    ('max_unpool2d', torch.tensor([[[[2., 4]]]]), (torch.tensor([[[[1, 3]]]]), 2, 2, 0)),
    ('max_unpool3d', torch.tensor([[[[[2., 4]]]]]), (torch.tensor([[[[[1, 3]]]]]), 2, 2, 0)),
    ('lp_pool1d', (S, S, S), (2., 3, 2,)),
    ('lp_pool2d', (S, S, S, S), (2., 3, 2,)),
    ('adaptive_max_pool1d', (S, S, S), (5,)),
    ('adaptive_max_pool2d', (S, S, S, S), ([5, 7],)),
    ('adaptive_max_pool3d', (S, S, S, S, S), ([3, 2, 2],)),
    ('adaptive_avg_pool1d', (S, S, S), (5,)),
    ('adaptive_avg_pool2d', (S, S, S, S), ([5, 7],)),
    ('adaptive_avg_pool3d', (S, S, S, S, S), ([3, 2, 2],)),
    ('dropout', (S, S, S), (0.5,)),
    ('alpha_dropout', (S, S, S), (0.5,)),
    ('dropout2d', (S, S, S), (0.5,)),
    ('dropout3d', (S, S, S), (0.5,)),
    ('feature_alpha_dropout', (S, S, S), (0.5,)),
    ('threshold', (S, S, S), (0.1, 2.),),
    ('threshold', (S, S, S), (0.1, 2., True), 'inplace'),
    ('relu', (S, S, S), (),),
    ('relu', (S, S, S), (), 'inplace'),
    ('glu', (S - 1, S - 1, S - 1), (),),
    ('hardtanh', (S, S, S), (-0.5, 0.5),),
    ('hardtanh', (S, S, S), (-0.5, 0.5, True), 'inplace'),
    ('relu6', (S, S, S), (),),
    ('relu6', (S, S, S), (True), 'inplace'),
    ('elu', (S, S, S), (0.9,),),
    ('elu', (S, S, S), (0.9, True), 'inplace'),
    ('selu', (S, S, S), (),),
    ('selu', (S, S, S), (True), 'inplace'),
    ('celu', (S, S, S), (0.9,),),
    ('celu', (S, S, S), (0.9, True), 'inplace'),
    ('leaky_relu', (S, S, S), (0.02,),),
    ('leaky_relu', (S, S, S), (0.02,), 'inplace'),
    ('rrelu', (S, S), (0.1, 0.3, False),),
    ('rrelu', (S, S), (0.1, 0.3, False, True), 'inplace'),
    ('hardshrink', (S, S, S), (0.4,),),
    ('tanhshrink', (S, S, S), (),),
    ('softsign', (S, S, S), (),),
    ('softplus', (S, S, S), (),),
    ('softmin', (S, S, S), (0,),),
    ('softmax', (S, S, S), (0,),),
    ('softmax', (S, S, S), (0, 3, torch.double), 'with_all_args'),
    ('tanh', (S, S, S), (),),
    ('sigmoid', (S, S, S), (),),
    ('log_softmax', (S, S, S), (0,),),
    ('linear', (S, S), ((M, S),),),
    ('bilinear', (S, S, S), ((S, S, M), torch.zeros(M, S, M),),),
    ('embedding', torch.tensor([[1, 2, 4, 5], [4, 3, 2, 5]]), (torch.rand(6, 3), ),),
    ('embedding_bag', torch.tensor([1, 2, 4, 2]), (torch.rand(5, 3), torch.tensor([0, 4]),),),
    ('batch_norm', (S, S), (non_differentiable(torch.randn(S)), non_differentiable(torch.ones(S)), ),),
    ('instance_norm', (S, S, S), (non_differentiable(torch.zeros(S)), non_differentiable(torch.ones(S))),),
    ('layer_norm', (S, S, S, S), ([5],),),
    ('group_norm', (S, S, S), (1, torch.rand(5),),),
    ('local_response_norm', (S, S, S), (2, ),),
    ('nll_loss', F.log_softmax(torch.randn(3, 5), dim=0), (torch.tensor([1, 0, 4]),),),
    ('poisson_nll_loss', torch.rand(S, 2), (torch.rand(S, 2),),),
    ('poisson_nll_loss', torch.rand(S, 2), (torch.rand(S, 2), True, True), 'full'),
    ('kl_div', F.log_softmax(torch.randn(S, 10), 1), (F.softmax(torch.randn(S, 10), 1),),),
    ('cross_entropy', (3, S), (torch.randint(S, (3,), dtype=torch.int64),),),
    ('binary_cross_entropy_with_logits', (3,), (torch.empty(3).random_(2), ),),
    ('smooth_l1_loss', (3, S), (non_differentiable(torch.rand(3, S)),),),
    ('l1_loss', (3, S), (non_differentiable(torch.rand(3, S)),),),
    ('mse_loss', (3, S), (non_differentiable(torch.rand(3, S)),),),
    ('smooth_l1_loss', (3, S), ((torch.rand(3, S)),), 'with_grad'),
    ('l1_loss', (3, S), ((torch.rand(3, S)),), 'with_grad'),
    ('mse_loss', (3, S), ((torch.rand(3, S)),), 'with_grad'),
    ('margin_ranking_loss', (3, S), ((3, S), (S,)),),
    ('hinge_embedding_loss', (3, S), (non_differentiable(torch.rand(3, S)),),),
    ('soft_margin_loss', (3, S), (non_differentiable(torch.rand(3, S)),),),
    ('multilabel_soft_margin_loss', (3, S), (non_differentiable(torch.rand(3, S)),),),
    ('cosine_embedding_loss', (S, S), ((S, S), non_differentiable(torch.rand(S,))),),
    ('pixel_shuffle', (1, 9, 4, 4), (3,),),
    ('interpolate', torch.zeros(3, 3).view(1, 1, 3, 3), (2,),),
    ('affine_grid', (S, 2, 3), (torch.Size([S, 1, 7, 7]),),),
    ('pad', (3, 3, 4, 2), ([1, 1],),),
    ('pairwise_distance', (S, S), ((S, S),),),
    ('pdist', (S, S), (),),
    ('cosine_similarity', (S, S), ((S, S),),),
    ('triplet_margin_loss', (S, S), ((S, S), (S, S)),),
    ('normalize', (S, S, S), (),),
    ('unfold', (S, S, S, S), ([2, 3]),),
    ('fold', (1, 3 * 2 * 2, 12), ([4, 5], [2, 2]),),
    ('grid_sample', (S, S, S, S), (non_differentiable(torch.rand(S, S, S, 2)),),),
    ('gumbel_softmax', (S, S), (2.,),),
    ('gumbel_softmax', (S, S), (2., True,), 'hard'),
    ('multilabel_margin_loss', torch.tensor([[0.2, -0.2, 0.07]]), (torch.tensor([[0, 0, 1]]),),),
    ('multi_margin_loss', (S, S), (non_differentiable(torch.randint(S, (S, ), dtype=torch.int64)),
                                   1, 1., non_differentiable(torch.randn(S))),),
    ('binary_cross_entropy', torch.randn(3, 2).sigmoid(), (non_differentiable(torch.rand(3, 2)),
                                                           non_differentiable(torch.randn(3, 2))),),
    ('binary_cross_entropy', torch.randn(3, 2).sigmoid(),
        (non_differentiable(torch.rand(3, 2)),
         non_differentiable(torch.randn(3, 2)), None, None, 'mean'), 'size_average'),
    ('ctc_loss', torch.rand(S, S, S).log_softmax(2).detach().requires_grad_(),
     (torch.randint(1, S, (S, S), dtype=torch.long), torch.full((S,), S, dtype=torch.long),
      torch.randint(1, S, (S,), dtype=torch.long))),
    ('upsample', torch.randn(S, S, M, M), (None, 2), 'with_scale'),
    ('upsample', torch.randn(S, S, M, M), (4,), 'with_size'),
    ('interpolate', torch.randn(S, S, M, M), (None, 2.), 'with_scale'),
    ('interpolate', torch.randn(S, S, M, M), (4,), 'with_size'),
]


# Test names in this set are only checked for a single derivative
nn_functional_single_grad = frozenset('test_nn_' + name for name in [
    'pdist',
    'multilabel_margin_loss',
    'max_unpool3d',
    'multi_margin_loss',
    'binary_cross_entropy',
    'binary_cross_entropy_size_average',
    'ctc_loss',
    'grid_sample',
])

# additional modules test
# TODO: delete this list once we make all nn_tests work
additional_module_tests = [
    {
        'module_name': 'Bilinear',
        'constructor_args': (S, S, M),
        'input_size': (S, S),
        'extra_args': ((S, S),)
    },
    {
        'module_name': 'RNNCell',
        'constructor_args': (S, S),
        'input_size': (S, S),
    },
    {
        'module_name': 'LSTMCell',
        'constructor_args': (S, S),
        'input_size': (S, S),
    },
    {
        'module_name': 'GRUCell',
        'constructor_args': (S, S),
        'input_size': (S, S),
    },
]


def add_autograd_test(
        name,
        self_size,
        args,
        variant_name='',
        dim_args_idx=(),
        skipTestIf=(),
        output_process_fn=lambda x: x,
        kwargs=None):
    basic_test_name = 'test_' + name
    if variant_name != '':
        basic_test_name += '_' + variant_name

    for dim_perm in product([-1, 1], repeat=len(dim_args_idx)):
        test_name = basic_test_name
        new_args = [arg * dim_perm[dim_args_idx.index(i)] if i in dim_args_idx else arg for i, arg in enumerate(args)]
        test_name = basic_test_name + ''.join('_neg' + str(i) for i, idx in enumerate(dim_perm) if idx < 0)
        new_args = tuple(new_args)

        # for-loop bodies don't define scopes, so we have to save the variables
        # we want to close over in some way
        def do_test(self, name=name, self_size=self_size, args=new_args, test_name=test_name,
                    output_process_fn=output_process_fn):
            def check(name):
                set_rng_seed(2)
                is_magic_method = name[:2] == '__' and name[-2:] == '__'
                is_inplace = name[-1] == "_" and not is_magic_method
                self_variable = create_input((self_size,))[0][0]
                # FixMe: run grad checks on inplace self
                if is_inplace:
                    self_variable.requires_grad = False
                # need to record this because methods can change the size (e.g. unsqueeze)
                args_variable, kwargs_variable = create_input(args, requires_grad=not is_inplace, call_kwargs=kwargs)
                self_tensor = deepcopy(self_variable.data)
                args_tensor = deepcopy(unpack_variables(args_variable))

                def fn(*inputs, **kwargs):
                    output = getattr(inputs[0], name)(*inputs[1:], **kwargs)
                    return output_process_fn(output)

                check_types = test_name not in EXCLUDE_TYPE_CHECK

                if not is_inplace and name not in EXCLUDE_GRADCHECK and not exclude_tensor_method(name, test_name):
                    # Test with disable_autodiff_subgraph_inlining, which forces the graph
                    # to contain DifferentiableGraph nodes whenever possible. This allows us
                    # to test autodiff; we assume that autograd is correct and use autodiff for backprop
                    if test_name not in EXCLUDE_TRACED:
                        check_against_reference(self,
                                                create_traced_fn(self, fn,
                                                                 disable_autodiff_subgraph_inlining=True),
                                                fn, (self_variable,) + args_variable, kwargs_variable,
                                                check_types=check_types)

                    if not is_magic_method and test_name not in EXCLUDE_SCRIPT:
                        check_against_reference(self,
                                                create_script_fn(self, name, 'method', output_process_fn,
                                                                 disable_autodiff_subgraph_inlining=True),
                                                fn, (self_variable,) + args_variable, kwargs_variable,
                                                check_types=check_types)

                # functional interface tests
                if hasattr(torch, name) and name not in EXCLUDE_FUNCTIONAL:
                    def fn(*inputs, **kwargs):
                        output = getattr(torch, name)(*inputs, **kwargs)
                        return output_process_fn(output)

                    f_args_variable = (self_variable,) + args_variable
                    f_args_tensor = (self_tensor,) + args_tensor

                    if not is_inplace and test_name not in EXCLUDE_TRACED:
                        check_against_reference(self,
                                                create_traced_fn(self, fn,
                                                                 disable_autodiff_subgraph_inlining=True),
                                                fn, f_args_variable, kwargs_variable, check_types=check_types)

                    if not is_inplace and test_name not in EXCLUDE_SCRIPT:
                        check_against_reference(self,
                                                create_script_fn(self, name, 'functional', output_process_fn,
                                                                 disable_autodiff_subgraph_inlining=True),
                                                fn, f_args_variable, kwargs_variable,
                                                check_types=check_types)

                # alias annotation testing
                if is_inplace and test_name not in EXCLUDE_SCRIPT:
                    check_alias_annotation(name, (self_variable,) + args_variable, kwargs_variable)

            check(name)
            inplace_name = name + '_'
            # can't broadcast inplace to left hand side
            broadcast_skip_inplace = 'broadcast_lhs' in test_name or 'broadcast_all' in test_name
            if hasattr(torch.ones(1), inplace_name) and not broadcast_skip_inplace:
                check(inplace_name)

        post_add_test(test_name, skipTestIf, do_test, TestJitGeneratedAutograd)


def suppress_warnings(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings(record=True):
            return fn(*args, **kwargs)
    return wrapper


def add_nn_functional_test(name, self_size, args, variant_name='', skipTestIf=(),
                           output_process_fn=lambda x: x, kwargs=None):
    test_name = 'test_nn_' + name

    if variant_name != '':
        test_name = test_name + '_' + variant_name

    no_grad = variant_name == 'inplace'

    @suppress_warnings
    def do_test(self, name=name, args=args, test_name=test_name):
        torch.manual_seed(2)

        self_variable = create_input((self_size,))[0][0]

        # need to record this because methods can change the size (e.g. unsqueeze)
        args_variable, kwargs_variable = create_input(args, call_kwargs=kwargs)

        self_tensor = deepcopy(self_variable.data)
        args_tensor = deepcopy(unpack_variables(args_variable))

        if not no_grad:
            output_variable = getattr(F, name)(self_variable, *args_variable, **kwargs_variable)

        def fn(*inputs, **kwargs):
            output = getattr(F, name)(*inputs, **kwargs)
            return output_process_fn(output)

        f_args_variable = (self_variable,) + args_variable
        f_args_tensor = (self_tensor,) + args_tensor

        if test_name not in EXCLUDE_SCRIPT:
            disable_ad_subgraph_inlining = test_name in DISABLE_AUTODIFF_SUBGRAPH_INLINING

            def run_test():
                script_fn = create_script_fn(self, name, 'nn_functional', output_process_fn,
                                             disable_autodiff_subgraph_inlining=disable_ad_subgraph_inlining)
                check_against_reference(self, script_fn, fn, f_args_variable, kwargs_variable, no_grad=no_grad)

            if test_name in EXCLUDE_PYTHON_PRINT:
                with self.disableModuleHook():
                    run_test()
            else:
                run_test()

    post_add_test(test_name, skipTestIf, do_test, TestJitGeneratedFunctional)


def add_nn_module_test(*args, **kwargs):
    if 'module_name' in kwargs:
        name = kwargs['module_name']
    elif 'fullname' in kwargs:
        name = kwargs['fullname']
    elif 'constructor' in kwargs:
        name = kwargs['constructor'].__name__

    no_grad = False if 'no_grad' not in kwargs else kwargs['no_grad']

    module_name = name.split("_")[0]

    module = getattr(torch.nn, module_name, None)
    if module is None or torch._jit_internal.weak_types.get(module) is None:
        return

    if 'desc' in kwargs and 'eval' in kwargs['desc']:
        # eval() is not supported, so skip these tests
        return

    test_name = name
    if 'desc' in kwargs:
        test_name = "{}_{}".format(test_name, kwargs['desc'])
    test_name = 'test_nn_{}'.format(test_name)

    @suppress_warnings
    def do_test(self):
        if test_name in EXCLUDE_SCRIPT_MODULES:
            return
        if 'constructor' in kwargs:
            nn_module = kwargs['constructor']
        else:
            nn_module = getattr(torch.nn, name)

        if "FunctionalModule" in str(nn_module):
            return

        if 'constructor_args_fn' in kwargs:
            constructor_args = kwargs['constructor_args_fn']()
        else:
            constructor_args = kwargs.get('constructor_args', ())

        # Construct a script module that passes arguments through
        # to self.submodule
        def create_script_module(*args, **kwargs):
            formals, tensors, actuals = get_script_args(args)

            method_args = ', '.join(['self'] + actuals)
            call_args_str = ', '.join(actuals)
            call = "self.submodule({})".format(call_args_str)
            script = script_method_template.format(method_args, call)

            submodule_constants = []
            if kwargs.get('is_constant'):
                submodule_constants = ['submodule']

            # Create module to use the script method
            class TheModule(torch.jit.ScriptModule):
                __constants__ = submodule_constants

                def __init__(self):
                    super(TheModule, self).__init__()
                    self.submodule = nn_module(*constructor_args)
            # module cannot be imported / exported
            if module_name in EXCLUDE_MODULE_EXPORT_IMPORT:
                with self.disableModuleHook():
                    module = TheModule()
                    module.define(script)
                    create_script_module.last_graph = module.graph
                    mod = module(*args)
            else:
                module = TheModule()
                module.define(script)
                self.assertExportImportModule(module, tensors)
                create_script_module.last_graph = module.graph
                mod = module(*args)
            return mod

        # Construct a normal nn module to stay consistent with create_script_module
        # and make use of a single global rng_state in module initialization
        def create_nn_module(*args, **kwargs):
            module = nn_module(*constructor_args)
            return module(*args)

        # Set up inputs from tuple of sizes or constructor fn
        if 'input_fn' in kwargs:
            input = kwargs['input_fn']()
        else:
            input = (kwargs['input_size'],)

        # Extra parameters to forward()
        if 'extra_args' in kwargs:
            input = input + kwargs['extra_args']

        if 'target_size' in kwargs:
            input = input + (kwargs['target_size'],)
        elif 'target_fn' in kwargs:
            if torch.is_tensor(input):
                input = (input,)
            input = input + (kwargs['target_fn'](),)

        args_variable, kwargs_variable = create_input(input)
        f_args_variable = deepcopy(unpack_variables(args_variable))

        # Check against Python module as reference
        check_against_reference(self, create_script_module, create_nn_module, f_args_variable, no_grad=no_grad)

    post_add_test(test_name, (), do_test, TestJitGeneratedModule)


def post_add_test(test_name, skipTestIf, do_test, test_class):
    assert not hasattr(test_class, test_name), 'Two tests have the same name: ' + test_name

    for skip in skipTestIf:
        do_test = skip(do_test)

    if not (TEST_WITH_UBSAN and test_name in UBSAN_BLACKLISTED_TESTS):
        setattr(test_class, test_name, do_test)


class TestAsync(JitTestCase):
    def test_async_python(self):
        @torch.jit.script
        def foo(x):
            return torch.neg(x)

        x = torch.rand(3, 4)
        fut = torch.jit._fork(foo, x)
        y_hat = foo(x)
        y = torch.jit._wait(fut)
        # assert nothing; only to make sure the fake python path works

    def test_async_parsing(self):
        @torch.jit.script
        def foo(x):
            # type: (Tensor) -> List[Tensor]
            return [torch.neg(x), x.t()]

        @torch.jit.script
        def bar(x):
            futures = torch.jit.annotate(List[Future[List[Tensor]]], [])
            for _ in range(3):
                future = torch.jit.annotate(
                    Future[List[Tensor]],
                    torch.jit._fork(foo, x)
                )
                futures.append(future)

            output = torch.jit.annotate(List[List[Tensor]], [])
            for i in range(3):
                output.append(torch.jit._wait(futures[i]))
            return output

        x = torch.rand(3, 3)
        result = bar(x)
        self.assertEqual(len(result), 3)

    def test_async_script(self):
        @torch.jit.script
        def foo(x):
            return torch.neg(x), x

        x = torch.rand(3, 4)

        @torch.jit.script
        def wait_script(x):
            fut = torch.jit._fork(foo, x)
            y_hat = foo(x)
            y = torch.jit._wait(fut)
            return y, y_hat

        y, y_hat = wait_script(x)

        self.assertEqual(y, y_hat)

    def test_async_script_capture(self):
        class Module(torch.jit.ScriptModule):
            __constants__ = ['const']

            def __init__(self):
                super(Module, self).__init__(False)
                self.const = 42
                self.param = nn.Parameter(torch.randn(2, 2))

            @torch.jit.script_method
            def foo(self, x1, x2):
                return torch.neg(x1), self.param, self.const, torch.neg(x2), self.param

            @torch.jit.script_method
            def wait_script(self, x1, x2):
                fut = torch.jit._fork(self.foo, x1, x2)
                y_hat = self.foo(x1, x2)
                y = torch.jit._wait(fut)
                return y, y_hat

        x1 = torch.rand(3, 4)
        x2 = torch.rand(5, 6)

        m = Module()
        y, y_hat = m.wait_script(x1, x2)

        self.assertEqual(y, y_hat)

    def test_async_script_nested(self):
        @torch.jit.script
        def foo(x):
            return torch.neg(x), x

        x = torch.rand(3, 4)

        @torch.jit.script
        def wait_script(x):
            fut = torch.jit._fork(foo, x)
            y_hat = foo(x)
            y = torch.jit._wait(fut)
            return y, y_hat

        @torch.jit.script
        def wait_script_nest(x):
            fut = torch.jit._fork(wait_script, x)
            return torch.jit._wait(fut)

        y, y_hat = wait_script_nest(x)

        self.assertEqual(y, y_hat)

    def test_async_script_no_script_mod(self):
        x = torch.rand(3, 4)

        with self.assertRaisesRegex(RuntimeError, 'cannot call a value'):
            @torch.jit.script
            def wait_script(x):
                fut = torch.jit._fork(x)
                return fut

    def test_async_script_multi_waits(self):
        @torch.jit.script
        def foo(x):
            return torch.neg(x).t() + x

        @torch.jit.script
        def wait_script(x):
            fut = torch.jit._fork(foo, x)

            # wait twice on the same future
            y1 = torch.jit._wait(fut)
            y2 = torch.jit._wait(fut)
            return y1, y2

        x = torch.rand(2, 2)
        y1, y2 = wait_script(x)
        self.assertEqual(y1, y2)

    def test_async_script_multi_forks(self):
        @torch.jit.script
        def foo1(x):
            return torch.neg(x).t() + x

        @torch.jit.script
        def foo2(x, y):
            return torch.neg(x).t() + x + torch.neg(y).t()

        @torch.jit.script
        def foo3(x, y, z):
            return torch.neg(z).t() + y.t() + x

        x1 = torch.rand(10, 10)
        x2 = torch.rand(10, 10)
        x3 = torch.rand(10, 10)

        @torch.jit.script
        def wait_script(x1, x2, x3):
            f1 = torch.jit._fork(foo1, x1)
            f2 = torch.jit._fork(foo2, x1, x2)
            f3 = torch.jit._fork(foo3, x1, x2, x3)
            f4 = torch.jit._fork(foo1, x2)
            f5 = torch.jit._fork(foo2, x2, x3)

            # ignore some forks
            y1 = torch.jit._wait(f1)
            y2 = torch.jit._wait(f2)
            y3 = torch.jit._wait(f3)

            return y1, y2, y3

        y1, y2, y3 = wait_script(x1, x2, x3)
        self.assertEqual(y1, foo1(x1))
        self.assertEqual(y2, foo2(x1, x2))
        self.assertEqual(y3, foo3(x1, x2, x3))

    def test_async_script_trace(self):
        class Traced(nn.Module):
            def __init__(self):
                super(Traced, self).__init__()

            def forward(self, x):
                return (torch.neg(x), x)

        class Module(torch.jit.ScriptModule):
            def __init__(self):
                super(Module, self).__init__(False)
                x = torch.rand(3, 3)
                self.traced = torch.jit.trace(Traced(), (x), _force_outplace=True)

            @torch.jit.script_method
            def forward(self, x):
                # type: (Tensor) -> Tuple[List[Tensor], Tuple[Tensor, Tensor], Tensor]
                future1 = torch.jit._fork(self.traced, x)
                future2 = torch.jit._fork(torch.neg, x)

                tensor_tuple = torch.jit._wait(future1)
                tensor_single = torch.jit._wait(future2)

                tensor_list = []
                tensor_list.append(tensor_tuple[0])
                tensor_list.append(tensor_single)

                # return a nested structure of tensors
                return (tensor_list, tensor_tuple, tensor_tuple[1])

        class Tuple(nn.Module):
            def __init__(self):
                super(Tuple, self).__init__()
                self.module = Module()

            def forward(self, x):
                z = torch.neg(x)
                y = self.module(x)
                list = [z, y[0][0], y[0][1], y[1][0], y[1][1], y[2]]
                return tuple(list)

        x = torch.rand(3, 3)
        module = torch.jit.trace(Tuple(), (x), _force_outplace=True)

        # Make sure we have forks
        self.assertGraphContainsExactly(module.graph, kind='prim::fork', num_kind_nodes=2)
        # Make sure 1 ::neg is in the root graph and 2 ::negs are in the subgraphs
        self.assertGraphContainsExactly(module.graph, kind='aten::neg', num_kind_nodes=1)
        self.assertGraphContainsExactly(module.graph, kind='aten::neg', num_kind_nodes=3, consider_subgraphs=True)

        y = torch.neg(x)
        self.assertEqual(module(x), (y, y, y, y, x, x))

    def test_async_script_error(self):
        x = torch.rand(3, 4)

        @torch.jit.script
        def foo(x):
            # error here
            return x.t() + x

        @torch.jit.script
        def wait_script(x):
            fut = torch.jit._fork(foo, x)
            return torch.jit._wait(fut)

        @torch.jit.script
        def wait_script_nest(x):
            fut = torch.jit._fork(wait_script, x)
            return torch.jit._wait(fut)

        # no future
        error_msg = 'The size.*must match the size of tensor'
        with self.assertRaisesRegex(Exception, error_msg):
            foo(x)

        # one future
        with self.assertRaisesRegex(Exception, error_msg):
            wait_script(x)

        # two futures with a different error
        x = torch.rand(3, 4, 5)
        with self.assertRaisesRegex(Exception, 'expects a 2D tensor'):
            wait_script_nest(x)

    def test_async_grad_guard_with_grad(self):
        @torch.jit.script
        def foo(x):
            y = x * 2
            return y.requires_grad

        @torch.jit.script
        def bar(x):
            fut = torch.jit._fork(foo, x)
            requires_grad_in_fork = torch.jit._wait(fut)
            z = x * 2
            return (requires_grad_in_fork, z.requires_grad)

        x = torch.randn(3, requires_grad=True)

        with torch.enable_grad():
            (inside_fork, after_wait) = bar(x)

        self.assertEqual(inside_fork, True)
        self.assertEqual(after_wait, True)

    def test_async_grad_guard_no_grad(self):
        @torch.jit.script
        def foo(x):
            y = x * 2
            return y.requires_grad

        @torch.jit.script
        def bar(x):
            fut = torch.jit._fork(foo, x)
            requires_grad_in_fork = torch.jit._wait(fut)
            z = x * 2
            return (requires_grad_in_fork, z.requires_grad)

        x = torch.randn(3, requires_grad=True)

        with torch.no_grad():
            (inside_fork, after_wait) = bar(x)

        self.assertEqual(inside_fork, False)
        self.assertEqual(after_wait, False)

    def test_trace_fork_wait(self):
        def fork_body(x):
            return x.neg(), x.neg() + 1

        def fn(x):
            fut = torch.jit._fork(fork_body, x)
            vals = torch.jit._wait(fut)
            return vals[0], vals[1], x - 1

        traced = torch.jit.trace(fn, (torch.rand(3, 4),))
        x = torch.rand(3, 4)
        self.assertEqual(fn(x), traced(x))

        self.assertGraphContainsExactly(traced.graph, kind='prim::fork', num_kind_nodes=1)
        self.assertGraphContainsExactly(traced.graph, kind='aten::wait', num_kind_nodes=1)
        self.assertGraphContainsExactly(traced.graph, kind='aten::neg', num_kind_nodes=2, consider_subgraphs=True)

    def test_trace_fork_wait_leaking(self):
        my_list = []

        def fork_body(x):
            my_list.append(x + 1)
            return x + 1

        def fn(x):
            fut = torch.jit._fork(fork_body, x)
            val = torch.jit._wait(fut)
            return my_list[0]

        with self.assertRaisesRegex(RuntimeError, 'did not have observable data dependence with trace inputs; '
                                                  'this probably indicates your program cannot be understood '
                                                  'by the tracer.'):
            traced = torch.jit.trace(fn, (torch.rand(3, 4),), check_trace=False)

    def test_trace_fork_wait_inline(self):
        def fork_body(x):
            return x + 1, x + 2

        def fn(x):
            fut = torch.jit._fork(fork_body, x)
            val = torch.jit._wait(fut)
            return val[1]

        traced = torch.jit.trace(fn, (torch.rand(3, 4),))
        torch._C._jit_pass_inline_fork_wait(traced.graph)
        torch._C._jit_pass_dce(traced.graph)
        self.assertGraphContainsExactly(traced.graph, kind='prim::fork', num_kind_nodes=0)
        self.assertGraphContainsExactly(traced.graph, kind='aten::wait', num_kind_nodes=0)
        self.assertGraphContainsExactly(traced.graph, kind='aten::add', num_kind_nodes=2)

    def test_trace_fork_wait_inline_onnx(self):
        def fork_body(x):
            return torch.neg(x), torch.neg(x)

        class MyMod(torch.nn.Module):
            def forward(self, x):
                fut = torch.jit._fork(fork_body, x)
                val = torch.jit._wait(fut)
                return val[1]

        # smoke test for ONNX export
        f = io.BytesIO()
        torch.onnx.export(MyMod(), (torch.rand(3, 4),), f)


for test in autograd_method_tests():
    add_autograd_test(*test)

for test in nn_functional_tests:
    add_nn_functional_test(*test)

for test in module_tests + new_module_tests + additional_module_tests:
    add_nn_module_test(**test)

for test in criterion_tests:
    test['no_grad'] = True
    add_nn_module_test(**test)

if __name__ == '__main__':
    run_tests()
