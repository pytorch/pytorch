# -*- coding: utf-8 -*-
# Owner(s): ["oncall: jit"]

import torch

# This is how we include tests located in test/jit/...
# They are included here so that they are invoked when you call `test_jit.py`,
# do not run these test files directly.
from jit.test_tracer import TestTracer, TestMixTracingScripting  # noqa: F401
from jit.test_recursive_script import TestRecursiveScript  # noqa: F401
from jit.test_type_sharing import TestTypeSharing  # noqa: F401
from jit.test_logging import TestLogging  # noqa: F401
from jit.test_backends import TestBackends, TestBackendsWithCompiler  # noqa: F401
from jit.test_backend_nnapi import TestNnapiBackend  # noqa: F401
from jit.test_list_dict import TestList, TestDict, TestNamedTuple, TestScriptDict, TestScriptList  # noqa: F401
from jit.test_async import TestAsync  # noqa: F401
from jit.test_data_parallel import TestDataParallel  # noqa: F401
from jit.test_models import TestModels  # noqa: F401
from jit.test_modules import TestModules  # noqa: F401
from jit.test_autodiff_subgraph_slicing import TestAutodiffSubgraphSlicing  # noqa: F401
from jit.test_custom_operators import TestCustomOperators  # noqa: F401
from jit.test_export_modes import TestExportModes  # noqa: F401
from jit.test_graph_rewrite_passes import TestGraphRewritePasses  # noqa: F401
from jit.test_class_type import TestClassType  # noqa: F401
from jit.test_builtins import TestBuiltins, TestTensorBuiltins  # noqa: F401
from jit.test_ignore_context_manager import TestIgnoreContextManager  # noqa: F401
from jit.test_symbolic_shape_analysis import TestSymbolicShapeAnalysis  # noqa: F401
from jit.test_unsupported_ops import TestUnsupportedOps  # noqa: F401
from jit.test_freezing import TestFreezing, TestFrozenOptimizations, TestMKLDNNReinplacing  # noqa: F401
from jit.test_peephole import TestPeephole  # noqa: F401
from jit.test_alias_analysis import TestAliasAnalysis  # noqa: F401
from jit.test_save_load import TestSaveLoad  # noqa: F401
from jit.test_module_containers import TestModuleContainers  # noqa: F401
from jit.test_python_bindings import TestPythonBindings  # noqa: F401
from jit.test_python_ir import TestPythonIr  # noqa: F401
from jit.test_functional_blocks import TestFunctionalBlocks  # noqa: F401
from jit.test_remove_mutation import TestRemoveMutation  # noqa: F401
from jit.test_torchbind import TestTorchbind  # noqa: F401
from jit.test_module_interface import TestModuleInterface  # noqa: F401
from jit.test_onnx_export import TestONNXExport  # noqa: F401
from jit.test_with import TestWith  # noqa: F401
from jit.test_enum import TestEnum  # noqa: F401
from jit.test_string_formatting import TestStringFormatting  # noqa: F401
from jit.test_profiler import TestProfiler  # noqa: F401
from jit.test_slice import TestSlice  # noqa: F401
from jit.test_ignorable_args import TestIgnorableArgs  # noqa: F401
from jit.test_hooks import TestHooks  # noqa: F401
from jit.test_warn import TestWarn  # noqa: F401
from jit.test_isinstance import TestIsinstance  # noqa: F401
from jit.test_cuda import TestCUDA  # noqa: F401
from jit.test_python_builtins import TestPythonBuiltinOP  # noqa: F401
from jit.test_typing import TestTyping  # noqa: F401
from jit.test_hash import TestHash  # noqa: F401
from jit.test_complex import TestComplex  # noqa: F401
from jit.test_jit_utils import TestJitUtils  # noqa: F401
from jit.test_scriptmod_ann import TestScriptModuleInstanceAttributeTypeAnnotation  # noqa: F401
from jit.test_types import TestTypesAndAnnotation  # noqa: F401
from jit.test_misc import TestMisc  # noqa: F401
from jit.test_pdt import TestPDT  # noqa: F401
from jit.test_tensor_creation_ops import TestTensorCreationOps  # noqa: F401
from jit.test_module_apis import TestModuleAPIs  # noqa: F401
from jit.test_script_profile import TestScriptProfile  # noqa: F401
from jit.test_convert_activation import TestFunctionalToInplaceActivation, TestInplaceToFunctionalActivation  # noqa: F401
from jit.test_parametrization import TestParametrization  # noqa: F401
from jit.test_attr import TestGetDefaultAttr  # noqa: F401
from jit.test_aten_pow import TestAtenPow  # noqa: F401
from jit.test_optimize_for_mobile_preserve_debug_info import TestOptimizeForMobilePreserveDebugInfo  # noqa: F401
from jit.test_union import TestUnion  # noqa: F401
from jit.test_models import MnistNet
from jit.test_batch_mm import TestBatchMM  # noqa: F401
from jit.test_dtype_analysis import TestDtypeAnalysis  # noqa: F401
from jit.test_python_frontend import TestFrontend  # noqa: F401
from jit.test_jit_script import TestScript  # noqa: F401
from jit.test_jit_autocast import TestJitAutocast  # noqa: F401

# Torch
from torch import Tensor
from torch._C import TensorType, BoolType, parse_ir, _propagate_shapes
from torch._six import PY37
from torch.autograd import Variable
from torch.jit.annotations import BroadcastingList2, BroadcastingList3, Any  # noqa: F401
from torch.nn.utils.rnn import PackedSequence
from torch.testing import FileCheck, make_tensor
import torch.autograd.profiler
import torch.cuda
import torch.jit
import torch.jit._logging
import torch.nn as nn
import torch.nn.functional as F

# Testing utils
from torch.testing._internal import jit_utils
from torch.testing._internal.common_jit import check_against_reference
from torch.testing._internal.common_utils import run_tests, IS_WINDOWS, TEST_WITH_UBSAN, \
    suppress_warnings, BUILD_WITH_CAFFE2, IS_SANDCASTLE, GRAPH_EXECUTOR, ProfilingMode, TestCase, \
    freeze_rng_state, slowTest, TemporaryFileName, skipIfCompiledWithoutNumpy, \
    enable_profiling_mode_for_profiling_tests, TEST_MKL, set_default_dtype, num_profiled_runs
from torch.testing._internal.jit_utils import JitTestCase, enable_cpu_fuser, disable_autodiff_subgraph_inlining, \
    _trace, do_input_map, get_execution_plan, make_global, \
    execWrapper, _inline_everything, _tmp_donotuse_dont_inline_everything, \
    RUN_CUDA
from torch.testing._internal.jit_metaprogramming_utils import create_script_fn, nn_functional_tests, get_script_args, \
    EXCLUDE_SCRIPT, additional_module_tests, EXCLUDE_SCRIPT_MODULES, \
    get_nn_module_name_from_kwargs, get_nn_mod_test_name, script_method_template

from torch.testing._internal.common_nn import module_tests, new_module_tests, criterion_tests
from torch.testing._internal.common_methods_invocations import (
    create_input, unpack_variables)

# For testing truediv in python 2
from torch.testing._internal.test_module.future_div import div_int_future, div_float_future
from torch.testing._internal.test_module.no_future_div import div_int_nofuture, div_float_nofuture

# Standard library
from collections import defaultdict, namedtuple, OrderedDict
from copy import deepcopy
from itertools import product
from textwrap import dedent
from typing import List, Dict, NamedTuple, Optional, Tuple, Union
import copy
import functools
import inspect
import io
import itertools
import math
import numpy as np
import os
import pickle
import pickletools
import random
import re
import shutil
import string
import sys
import tempfile
import types
import typing
import unittest
import warnings
import zipfile

from jit._test_helpers import (
 FooToPickle,
 LSTMCell,
 LSTMCellC,
 LSTMCellS,
 MiLSTMCell,
 _sum_of_list,
 all_backward_graphs,
 backward_graph,
 get_fn,
 get_grad_executor,
 get_lstm_inputs,
 get_milstm_inputs,
 canonical,
 LSTMCellF,
 doAutodiffCheck
)




# TODO: enable TE in PE when all tests are fixed
torch._C._jit_set_texpr_fuser_enabled(GRAPH_EXECUTOR == ProfilingMode.PROFILING)
torch._C._jit_set_profiling_executor(GRAPH_EXECUTOR != ProfilingMode.LEGACY)
# even though FULL_PROFILER should be our default
# we haven't tested every single test in this file
# but we enable FULL_PROFILER for a large subset
# of the tests with "with enable_profiling_mode_for_profiling_tests"
torch._C._jit_set_profiling_mode(False)

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

    def test_inferred_as_tensor(self):
        with self.assertRaisesRegex(RuntimeError, "Inferred the value for argument 'dim' to be of type 'Tensor' "
                                                  "because it was not annotated with an explicit type"):
            @torch.jit.script
            def dot(points, query, dim):
                return (points * query).sum(dim)

    def test_constants_pkl(self):
        # This test asserts that the serialization archive includes a `constants.pkl`
        # file. This file is used by `torch.load` to determine whether a zip file
        # is a normal eager-mode serialization zip or a jit serialization zip. If
        # you are deleting `constants.pkl`, make sure to update `torch.serialization.load`
        # so it is still able to figure out which is which.
        @torch.jit.script
        def fn(x):
            return x

        buf = io.BytesIO()
        torch.jit.save(fn, buf)
        buf.seek(0)

        files = zipfile.ZipFile(buf).filelist
        self.assertTrue(any(['archive/constants.pkl' == f.filename for f in files]))

    def test_restore_device(self):
        class M(torch.jit.ScriptModule):
            def __init__(self, cpu_device_str):
                super(M, self).__init__()
                self.p0 = nn.Parameter(torch.tensor([0.3], dtype=torch.float,
                                                    device=cpu_device_str))
                self.b0 = torch.tensor([0.9], dtype=torch.float,
                                       device=cpu_device_str)

        # main purpose is checking map_location works
        m = M("cpu")
        m2 = self.getExportImportCopy(m)
        self.assertEqual(tuple(m.parameters()), tuple(m2.parameters()))
        self.assertEqual(tuple(m.buffers()), tuple(m2.buffers()))
        self.assertFalse(m2.p0.is_cuda)
        self.assertFalse(m2.b0.is_cuda)

    @unittest.skipIf(not RUN_CUDA, "restore device requires CUDA")
    def test_restore_device_cuda(self):
        class MyModule(torch.jit.ScriptModule):
            def __init__(self):
                super(MyModule, self).__init__()
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

    def test_trace_retains_train(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return x
        m = M()
        m.eval()
        tm = torch.jit.trace(m, (torch.rand(3)))
        self.assertEqual(tm.training, m.training)

    @unittest.skipIf(not RUN_CUDA, "restore device requires CUDA")
    def test_restore_shared_storage_on_cuda(self):
        class Foo(torch.jit.ScriptModule):
            def __init__(self):
                super(Foo, self).__init__()
                whole_tensor = torch.randn(4, 5, dtype=torch.float, device='cpu')
                self.p0 = nn.Parameter(whole_tensor.narrow(0, 0, 1))
                self.register_buffer('b0', whole_tensor.narrow(0, 3, 1))

        m = Foo()
        m2 = self.getExportImportCopy(m, map_location=torch.device('cuda:0'))
        self.assertEqual(tuple(m.parameters()), tuple(m2.parameters()))
        self.assertEqual(tuple(m.buffers()), tuple(m2.buffers()))
        self.assertTrue(m2.p0.is_cuda)
        self.assertTrue(m2.b0.is_cuda)
        self.assertTrue(m2.p0.is_shared())
        self.assertTrue(m2.b0.is_shared())
        self.assertEqual(m2.b0.storage().data_ptr(), m2.p0.storage().data_ptr())

    def test_add_relu_fusion(self):
        class M(torch.nn.Module):
            def __init__(self, relu_op):
                super(M, self).__init__()
                self.relu_op = relu_op

            def forward(self, a, b, c):
                tmp = torch.add(a, b)
                x = self.relu_op(tmp)
                d = torch.add(a, c)
                return x + d
        a = torch.rand((7, 11))
        a = a * -10
        a = a + 5
        b = torch.rand((7, 11))
        c = torch.rand((7, 11))
        m = torch.jit.script(M(torch.relu))
        orig_res = m(a, b, c)
        torch._C._jit_pass_fuse_add_relu(m.graph)
        buffer = io.BytesIO()
        torch.jit.save(m, buffer)
        buffer.seek(0)
        m = torch.jit.load(buffer)
        new_res = m(a, b, c)
        FileCheck().check_not("aten::relu(") \
            .check("aten::_add_relu(") \
            .run(m.graph)
        torch.testing.assert_close(orig_res, new_res)

        # add, relu_
        a = torch.rand((7, 11))
        a = a * -10
        a = a + 5
        b = torch.rand((7, 11))
        c = torch.rand((7, 11))
        m = torch.jit.script(M(torch.relu_))
        orig_res = m(a, b, c)
        torch._C._jit_pass_fuse_add_relu(m.graph)
        buffer = io.BytesIO()
        torch.jit.save(m, buffer)
        buffer.seek(0)
        m = torch.jit.load(buffer)
        new_res = m(a, b, c)
        FileCheck().check_not("aten::relu_(") \
            .check("aten::_add_relu(") \
            .run(m.graph)
        torch.testing.assert_close(orig_res, new_res)

        class Madd_(torch.nn.Module):
            def __init__(self, relu_op):
                super(Madd_, self).__init__()
                self.relu_op = relu_op

            def forward(self, a, b):
                x = a.add_(b)
                x = self.relu_op(x)
                return x

        # add_, relu_
        a = torch.rand((7, 11))
        a = a * -10
        a = a + 5
        b = torch.rand((7, 11))
        # Because in place add_ will overwrite a
        a_copy = a.clone()
        m = torch.jit.script(Madd_(torch.relu_))
        orig_res = m(a, b)
        torch._C._jit_pass_fuse_add_relu(m.graph)
        buffer = io.BytesIO()
        torch.jit.save(m, buffer)
        buffer.seek(0)
        m = torch.jit.load(buffer)
        new_res = m(a_copy, b)
        FileCheck().check_not("aten::add_(") \
            .check_not("aten::relu_(") \
            .check("aten::_add_relu_(") \
            .run(m.graph)
        torch.testing.assert_close(orig_res, new_res)
        # Since _add_relu_ does inplace mutation ensure
        # a_copy is modified
        torch.testing.assert_close(orig_res, a_copy)

        class Madd_out(torch.nn.Module):
            def __init__(self, relu_op):
                super(Madd_out, self).__init__()
                self.relu_op = relu_op

            def forward(self, a, b):
                x = torch.add(a, b, out=a)
                x = self.relu_op(x)
                return x
        a = torch.rand((7, 11))
        a = a * -10
        a = a + 5
        b = torch.rand((7, 11))

        # add_out, relu_
        a = torch.rand((7, 11))
        a = a * -10
        a = a + 5
        b = torch.rand((7, 11))
        # Because in place add_ will overwrite a
        a_copy = a.clone()
        m = torch.jit.script(Madd_out(torch.relu_))
        orig_res = m(a, b)
        torch._C._jit_pass_fuse_add_relu(m.graph)
        buffer = io.BytesIO()
        torch.jit.save(m, buffer)
        buffer.seek(0)
        m = torch.jit.load(buffer)
        new_res = m(a_copy, b)
        FileCheck().check_not("aten::add(") \
            .check_not("aten::relu_(") \
            .check("aten::_add_relu(") \
            .run(m.graph)
        torch.testing.assert_close(orig_res, new_res)
        # Since _add_relu_ with out=a does inplace mutation ensure
        # a_copy is modified
        torch.testing.assert_close(orig_res, a_copy)

    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.LEGACY, "Simple executor doesn't have shape information")
    def test_peephole_optimize_shape_ops(self):
        def test_input(func, input, result):
            # if result == 2 we will trigger a bailout and
            # the unprofiled graph should return the correct result
            self.assertEqual(func(input, profile_and_replay=True), result)
            gre = func.graph_for(input)
            FileCheck().check_not("prim::If").run(gre)

        def test_dim():
            @torch.jit.script
            def func(x):
                if x.dim() == 1:
                    return 1
                else:
                    return 2

            test_input(func, torch.tensor([0.5]), 1)
            test_input(func, torch.tensor([[0.5]]), 2)
        test_dim()

        def test_size_index():
            @torch.jit.script
            def func(x):
                if x.size(0) == 1:
                    return 1
                else:
                    return 2

            test_input(func, torch.rand([1, 2]), 1)
            test_input(func, torch.rand([1, 3]), 1)

            @torch.jit.script
            def neg_index(x):
                if x.size(-2) == 1:
                    return 1
                else:
                    return 2

            test_input(neg_index, torch.rand([1, 2]), 1)
            test_input(neg_index, torch.rand([1, 3]), 1)

        if GRAPH_EXECUTOR == ProfilingMode.PROFILING:
            test_size_index()

        def test_dtype():
            @torch.jit.script
            def func(x):
                if x.dtype == torch.float32:
                    return 1
                else:
                    return 2

            test_input(func, torch.tensor(0.5, dtype=torch.float32), 1)
            test_input(func, torch.tensor(0.5, dtype=torch.int64), 2)
        test_dtype()

        def test_is_floating_poiint():
            @torch.jit.script
            def func(x):
                if x.is_floating_point():
                    return 1
                else:
                    return 2

            test_input(func, torch.tensor(0.5, dtype=torch.float32), 1)
            test_input(func, torch.tensor(0.5, dtype=torch.int64), 2)
        test_is_floating_poiint()

        def test_device():
            @torch.jit.script
            def func_1(x):
                if x.device == torch.device('cuda:0'):
                    a = 0
                else:
                    a = 1
                return a

            @torch.jit.script
            def func_2(x):
                if x.is_cuda:
                    a = 0
                else:
                    a = 1
                return a

            test_input(func_1, torch.tensor(0.5), 1)
            test_input(func_2, torch.tensor(0.5), 1)

            if RUN_CUDA:
                test_input(func_1, torch.tensor(0.5, device="cuda:0"), 0)
                test_input(func_2, torch.tensor(0.5, device="cuda:0"), 0)

        test_device()

    def test_attrs(self):
        def foo(x):
            return (
                # x.dtype, TODO: dtype long -> instance conversion
                x.device,
                x.shape,
                x.is_cuda,
                x.is_mkldnn,
                x.is_quantized,
                x.requires_grad,
                x.T,
                x.mT,
                x.H,
                x.mH
                # x.layout TODO: layout long -> instance conversion
            )

        scripted = torch.jit.script(foo)
        x = torch.rand(3, 4)
        self.assertEqual(scripted(x), foo(x))

    def test_layout(self):
        @torch.jit.script
        def check(x, y):
            return x.layout == y.layout

        x = torch.rand(3, 4)
        y = torch.rand(3, 4)

        self.assertTrue(check(x, y))

    def test_matrix_transpose(self):
        @torch.jit.script
        def check(x):
            return torch.equal(x.mT, x.transpose(-2, -1))

        x = torch.rand(3, 4)
        self.assertTrue(check(x))

    def test_transpose(self):
        @torch.jit.script
        def check(x):
            return torch.equal(x.T, x.t())

        x = torch.rand(3, 4)
        self.assertTrue(check(x))

    def test_matrix_conj_transpose(self):
        @torch.jit.script
        def check(x):
            return torch.equal(x.mH, x.transpose(-2, -1).conj())

        x = torch.rand(3, 4)
        self.assertTrue(check(x))

        x = make_tensor((3, 4), device="cpu", dtype=torch.complex64)
        self.assertTrue(check(x))

    def test_conj_transpose(self):
        @torch.jit.script
        def check(x):
            return torch.equal(x.H, x.t().conj())

        x = torch.rand(3, 4)
        self.assertTrue(check(x))

        x = make_tensor((3, 4), device="cpu", dtype=torch.complex64)
        self.assertTrue(check(x))

    def test_T_mT_H_mH(self):
        def T(x):
            return x.mT

        def mT(x):
            return x.mT

        def H(x):
            return x.H

        def mH(x):
            return x.mH

        x = torch.rand(3, 4)
        y = make_tensor((3, 4), device="cpu", dtype=torch.complex64)

        self.checkScript(T, (x, ))
        self.checkScript(mT, (x, ))
        self.checkScript(H, (x, ))
        self.checkScript(mH, (x, ))
        self.checkScript(T, (y, ))
        self.checkScript(mT, (y, ))
        self.checkScript(H, (y, ))
        self.checkScript(mH, (y, ))

    def test_nn_conv(self):
        class Mod(nn.Module):
            def __init__(self, conv):
                super().__init__()
                self.conv = conv

            def forward(self, input):
                return self.conv(input)

        inputs = [
            # Conv
            (Mod(nn.Conv1d(16, 33, 3, stride=2)), torch.randn(20, 16, 5)),
            (Mod(nn.Conv2d(16, 33, 3, stride=2)), torch.randn(20, 16, 5, 10)),
            (Mod(nn.Conv3d(16, 33, 3, stride=2)), torch.randn(20, 16, 3, 5, 4)),
            # ConvTransposed
            (Mod(nn.ConvTranspose1d(16, 33, 3, stride=2)), torch.randn(20, 16, 5)),
            (Mod(nn.ConvTranspose2d(16, 33, 3, stride=2)), torch.randn(20, 16, 5, 10)),
            (Mod(nn.ConvTranspose3d(16, 33, 3, stride=2)), torch.randn(20, 16, 3, 5, 4)),
        ]

        for m, inp in inputs:
            self.checkModule(m, (inp,))

    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING, 'Not implemented for Simple or Legacy')
    def test_debug_flush_compilation_cache(self):
        def foo(x):
            return x + 2

        class Mod(nn.Module):
            def __init__(self):
                super(Mod, self).__init__()

            def forward(self, t):
                return t + 2

        m = torch.jit.script(Mod())
        x = torch.rand(1, 10)

        with enable_profiling_mode_for_profiling_tests():
            jitted = self.checkScript(foo, (x,))
            # shouldn't throw
            states = jitted.get_debug_state()

            # after flushing there shouldn't be
            # no opt plan
            jitted._debug_flush_compilation_cache()
            with self.assertRaisesRegex(RuntimeError, "INTERNAL ASSERT FAILED"):
                states = jitted.get_debug_state()

            NUM_RUNS = 1
            with num_profiled_runs(NUM_RUNS):
                m(x)
                m(x)
                fwd = m._c._get_method("forward")
                states = m.get_debug_state()

                # after flushing there shouldn't be
                # no opt plan
                fwd._debug_flush_compilation_cache()
                with self.assertRaisesRegex(RuntimeError, "INTERNAL ASSERT FAILED"):
                    states = m.get_debug_state()

    def test_numel(self):
        @torch.jit.script
        def get_numel_script(x):
            return x.numel()

        x = torch.rand(3, 4)
        numel = get_numel_script(x)
        self.assertEqual(numel, x.numel())

    def test_element_size(self):
        @torch.jit.script
        def get_element_size_script(x):
            return x.element_size()

        x = torch.rand(3, 4)
        element_size = get_element_size_script(x)
        self.assertEqual(element_size, x.element_size())

    def test_Sequential(self):
        class Seq(nn.Module):
            def __init__(self):
                super(Seq, self).__init__()
                self.seq = nn.Sequential(nn.Linear(10, 20), nn.Linear(20, 30))

            @torch.jit.script_method
            def forward(self, x):
                for l in self.seq:
                    x = l(x)
                return x

        m = torch.jit.script(Seq())
        assert m.graph  # ensure jit was able to compile

    def test_ModuleList(self):
        class Mod(nn.Module):
            def __init__(self):
                super(Mod, self).__init__()
                self.model = nn.ModuleList([nn.Linear(10, 10) for _ in range(10)])
                self.model += (nn.Linear(10, 20),)
                self.model.append(nn.Linear(20, 30))
                self.model.extend([nn.Linear(30, 40), nn.Linear(40, 50)])

            def forward(self, v):
                for m in self.model:
                    v = m(v)
                return v

        m = torch.jit.script(Mod())
        assert m.graph  # ensure jit was able to compile

    def test_disabled(self):
        torch.jit._state.disable()
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
            torch.jit._state.enable()

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

    def test_nn_padding(self):
        class Mod(nn.Module):
            def __init__(self, padding):
                super().__init__()
                self.padding = padding

            def forward(self, input):
                return self.padding(input)

        inputs = [
            (Mod(nn.ConstantPad1d(2, 3.5)), torch.randn(1, 2, 4)),
            (Mod(nn.ConstantPad2d(2, 3.5)), torch.randn(1, 2, 2)),
            (Mod(nn.ConstantPad3d(3, 3.5)), torch.randn(16, 3, 10, 20, 30)),
            (Mod(nn.ReflectionPad1d(2)), torch.arange(8, dtype=torch.float).reshape(1, 2, 4)),
            (Mod(nn.ReflectionPad2d(2)), torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3)),
            (Mod(nn.ReflectionPad3d(3)), torch.randn(16, 3, 8, 32, 48)),
            (Mod(nn.ReplicationPad1d(2)), torch.arange(8, dtype=torch.float).reshape(1, 2, 4)),
            (Mod(nn.ReplicationPad2d(2)), torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3)),
            (Mod(nn.ReplicationPad3d(3)), torch.randn(16, 3, 8, 32, 48)),
            (Mod(nn.ZeroPad2d(2)), torch.randn(1, 1, 3, 3))
        ]

        for m, inp in inputs:
            self.checkModule(m, (inp,))

    def test_script_autograd_grad(self):
        def test_simple_grad(x, y):
            # type: (Tensor, Tensor) -> List[Optional[Tensor]]
            z = x + 2 * y + x * y
            return torch.autograd.grad((z.sum(), ), (x, y))

        def test_simple_grad_with_grad_outputs(x, y):
            # type: (Tensor, Tensor) -> List[Optional[Tensor]]
            z = x + 2 * y + x * y
            grad_outputs = torch.jit.annotate(List[Optional[torch.Tensor]], [torch.ones((2, 2)), ])
            return torch.autograd.grad((z, ), (x, y), grad_outputs)

        def test_one_output_not_requires_grad(x, y):
            # type: (Tensor, Tensor) -> List[Optional[Tensor]]
            z = 2 * y + y
            return torch.autograd.grad((z.sum(),), (x, y), allow_unused=True)

        def test_retain_graph(x, y):
            # type: (Tensor, Tensor) -> None
            z = x + 2 * y + x * y
            torch.autograd.grad((z.sum(), ), (x, y), retain_graph=True)
            torch.autograd.grad((z.sum(), ), (x, y))

        x = torch.randn(2, 2, requires_grad=True)
        y = torch.randn(2, 2, requires_grad=True)
        self.checkScript(test_simple_grad, (x, y), inputs_requires_grad=True)
        self.checkScript(test_simple_grad_with_grad_outputs, (x, y), inputs_requires_grad=True)
        self.checkScript(test_one_output_not_requires_grad, (x, y), inputs_requires_grad=True)
        self.checkScript(test_retain_graph, (x, y), inputs_requires_grad=True)

    def test_script_backward(self):
        def checkBackwardScript(fn, inputs):
            scripted_fn = torch.jit.script(fn)
            FileCheck().check("torch.autograd.backward").run(scripted_fn.code)
            recording_inputs = do_input_map(lambda t: t.detach().requires_grad_(), inputs)

            fn(*inputs)
            scripted_fn(*recording_inputs)

            for inp1, inp2 in zip(inputs, recording_inputs):
                self.assertEqual(inp1.grad, inp2.grad)

        def test_tensor_backward(input):
            # type: (Tensor) -> None
            output = torch.relu(input)
            output = output.softmax(0)
            sum_out = output.sum()
            sum_out.backward()

        def test_torch_autograd_backward(input):
            # type: (Tensor) -> None
            output = torch.relu(input)
            output = output.softmax(0)
            torch.autograd.backward(output.sum())

        def test_torch_autograd_backward_with_grad_tensors(input):
            # type: (Tensor) -> None
            output = torch.relu(input)
            output = output.softmax(0)
            grad_outputs = torch.jit.annotate(List[Optional[torch.Tensor]], [torch.ones((2, 2)), ])
            torch.autograd.backward((output,), grad_outputs)

        inp = torch.randn(2, 2, requires_grad=True)
        checkBackwardScript(test_tensor_backward, (inp,))
        checkBackwardScript(test_torch_autograd_backward, (inp,))
        checkBackwardScript(test_torch_autograd_backward_with_grad_tensors, (inp,))

    def test_script_backward_twice(self):
        def checkBackwardTwiceScript(fn, inputs, retain_graph_=False):
            torch._C._jit_set_profiling_executor(False)

            with torch.jit.optimized_execution(True):
                scripted_fn = torch.jit.script(fn, inputs)
                FileCheck().check("prim::DifferentiableGraph").run(scripted_fn.graph_for(*inputs))

                result = scripted_fn(*inputs)
                result.sum().backward(retain_graph=retain_graph_)
                if not retain_graph_:
                    self.assertRaisesRegex(RuntimeError, 'Specify retain_graph=True',
                                           lambda: result.sum().backward())
                else:
                    result.sum().backward()

        def test_script_backward_twice_with_saved_values(input1, input2):
            # type: (Tensor, Tensor) -> Tensor
            tmp1 = torch.mul(input1, input2)
            tmp2 = torch.abs(tmp1)
            if torch.equal(input1, input2):
                tmp2 = torch.acos(tmp2)
            else:
                tmp2 = torch.atan(tmp2)
            result = torch.add(tmp2, input2)
            return result

        inp1 = torch.randn(2, 2, requires_grad=True)
        inp2 = torch.randn(2, 2, requires_grad=True)
        checkBackwardTwiceScript(test_script_backward_twice_with_saved_values, (inp1, inp2), False)
        checkBackwardTwiceScript(test_script_backward_twice_with_saved_values, (inp1, inp2), True)

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

    def test_torch_sum(self):
        def fn(x):
            return torch.sum(x)

        def fn1(x, dim: int):
            return torch.sum(x, dim)

        x = torch.randn(3, 4)
        self.checkScript(fn, (x, ))
        self.checkScript(fn1, (x, 1, ))
        self.checkScript(fn1, (x, 0, ))

    def test_cse(self):
        x = torch.tensor([0.4, 0.3], requires_grad=True)
        y = torch.tensor([0.7, 0.5], requires_grad=True)

        def fn(x, y):
            w = (x + y) * (x + y) * (x + y)
            t = torch.tanh(w) + torch.tanh(w)
            z = (x + y) * (x + y) * (x + y) + t
            return z

        g, _ = torch.jit._get_trace_graph(fn, (x, y))
        self.run_pass('cse', g)
        do_exactly = True
        FileCheck().check_count("add", 1).check_count("mul", 2, do_exactly) \
            .check_count("tanh", 1, do_exactly).check_count("add", 2, do_exactly).check_next("return")  \
            .run(str(g))

        self.assertExportImport(g, (x, y))

    def test_cse_not_introduce_aliasing(self):
        @torch.jit.script
        def tensor_alias_outputs(x):
            return x + x, x + x

        self.run_pass('cse', tensor_alias_outputs.graph)
        FileCheck().check_count("aten::add", 2).run(tensor_alias_outputs.graph)

        @torch.jit.script
        def ints_alias_outputs(x):
            # type: (int) -> Tuple[int, int]
            return x + x, x + x

        # non-aliasing types can be CSEd
        self.run_pass('cse', ints_alias_outputs.graph)
        FileCheck().check_count("aten::add", 1, exactly=True).run(ints_alias_outputs.graph)

    def test_recursive_cse(self):
        input_str = """
graph(%x : Tensor,
      %y : Tensor,
      %20 : int):
  %2 : int = prim::Constant[value=1]()
  %3 : Tensor = aten::add(%x, %y, %2)
  %4 : int = aten::add(%2, %20)
  %5 : bool = aten::Bool(%4)
  %z : int = prim::If(%5)
    # CHECK: block
    block0():
      # CHECK-NOT: aten::add
      %z.1 : int = aten::add(%2, %20)
      -> (%z.1)
    block1():
      -> (%2)
  return (%z)
"""
        graph = parse_ir(input_str)
        self.run_pass('cse', graph)
        FileCheck().run(input_str, graph)

    def test_pattern_based_rewrite(self):
        # mul(mul(mul(mul(x,y),z),x),y) --> mul(mul(mulmul(x,y,z), x), y) -->
        # --> mulmul(mulmul(x,y,z), x, y)
        input_str = """
graph(%x, %y, %z):
    # CHECK-NOT: aten::mul
    # CHECK: my::fused_mulmul
    %t = aten::mul(%x, %y)
    %p = aten::mul(%t, %z)
    # CHECK: my::fused_mulmul
    %u = aten::mul(%p, %x)
    %o = aten::mul(%u, %y)
    return (%o)"""
        graph = parse_ir(input_str)
        torch._C._jit_pass_custom_pattern_based_rewrite_graph("""
graph(%a, %b, %c):
  %q = aten::mul(%a, %b)
  %r = aten::mul(%q, %c)
  return (%r)""", """
graph(%a, %b, %c):
  %r = my::fused_mulmul(%a, %b, %c)
  return (%r)""", graph)
        FileCheck().run(input_str, graph)

        # Check that overlapping matches are handled correctly
        # mul(mul(mul(x,y),z),x) --> mul(mulmul(x,y,z), x)
        input_str = """
graph(%x, %y, %z):
    # CHECK-NOT: aten::mul
    # CHECK: my::fused_mulmul
    %t = aten::mul(%x, %y)
    %p = aten::mul(%t, %z)
    # CHECK-NEXT: aten::mul
    %u = aten::mul(%p, %x)
    return (%u)"""
        graph = parse_ir(input_str)
        torch._C._jit_pass_custom_pattern_based_rewrite_graph("""
graph(%a, %b, %c):
  %q = aten::mul(%a, %b)
  %r = aten::mul(%q, %c)
  return (%r)""", """
graph(%a, %b, %c):
  %r = my::fused_mulmul(%a, %b, %c)
  return (%r)""", graph)
        FileCheck().run(input_str, graph)

        # Check add(mul(x,y),z) --> muladd(x,y,z) replacement
        input_str = """
graph(%x, %y, %z):
    # CHECK-NOT: aten::mul
    # CHECK-NOT: aten::add
    %c = prim::Const[value=1]()
    %t = aten::mul(%x, %y)
    %p = aten::add(%t, %z, %c)
    # CHECK: my::muladd
    # CHECK-NEXT: return
    return (%p)"""
        graph = parse_ir(input_str)
        torch._C._jit_pass_custom_pattern_based_rewrite_graph("""
graph(%a, %b, %c, %d):
  %q = aten::mul(%a, %b)
  %r = aten::add(%q, %c, %d)
  return (%r)""", """
graph(%a, %b, %c, %d):
  %r = my::muladd(%a, %b, %c, %d)
  return (%r)""", graph)
        FileCheck().run(input_str, graph)

        # Check add(mul(x,y),z) --> sub(add(x,y),z) replacement
        input_str = """
graph(%x, %y, %z):
    # CHECK-NOT: aten::mul
    %c = prim::Const[value=1]()
    # CHECK: aten::add
    %t = aten::mul(%x, %y)
    # CHECK-NEXT: aten::sub
    %p = aten::add(%t, %z, %c)
    # CHECK-NOT: aten::add
    # CHECK-NEXT: return
    return (%p)"""
        graph = parse_ir(input_str)
        torch._C._jit_pass_custom_pattern_based_rewrite_graph("""
graph(%a, %b, %c, %d):
  %q = aten::mul(%a, %b)
  %r = aten::add(%q, %c, %d)
  return (%r)""", """
graph(%a, %b, %c, %d):
  %q = aten::add(%a, %b, %d)
  %r = aten::sub(%q, %c, %d)
  return (%r)""", graph)
        FileCheck().run(input_str, graph)

        # Check mul(x,y) --> x replacement
        input_str = """
graph(%x, %y, %z):
    %c = prim::Const[value=1]()
    # CHECK-NOT: aten::mul
    %t = aten::mul(%x, %y)
    # CHECK: aten::add(%x, %z
    %p = aten::add(%t, %z, %c)
    # CHECK-NEXT: return
    return (%p)"""
        graph = parse_ir(input_str)
        torch._C._jit_pass_custom_pattern_based_rewrite_graph("""
graph(%Pa, %Pb):
  %Pq = aten::mul(%Pa, %Pb)
  return (%Pq)""", """
graph(%Ra, %Rb):
  return (%Ra)""", graph)
        FileCheck().run(input_str, graph)

    @_tmp_donotuse_dont_inline_everything
    def test_pattern_based_module_rewrite(self):
        # Check match::module behavior
        class Test(torch.nn.Module):
            def __init__(self):
                super(Test, self).__init__()
                self.conv = torch.nn.Conv2d(1, 20, 5, 1)
                self.bn = torch.nn.BatchNorm2d(num_features=20)

            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                return x
        m = torch.jit.script(Test())
        torch._C._jit_pass_custom_pattern_based_rewrite_graph("""
        graph(%self, %x):
                %conv = match::module[name="Conv2d"](%self)
                %y = prim::CallMethod[name="forward"](%conv, %x)
                %bn = match::module[name="BatchNorm2d"](%self)
                %z = prim::CallMethod[name="forward"](%bn, %y)
                return (%z)""", """
        graph(%self, %x):
          %z = my::matched_conv_bn(%self, %x)
          return (%z)""", m._c._get_method("forward").graph)

        FileCheck().check("my::matched_conv_bn").run(m._c._get_method("forward").graph)

    def test_pattern_based_rewrite_with_source_range_preserved(self):
        class TestModule1(torch.nn.Module):
            def __init__(self):
                super(TestModule1, self).__init__()

            def forward(self, x, y, z, w):
                x = x + y
                x = x * z
                return w - x

        input_pattern = """
        graph(%x, %y, %z, %const):
            %t = aten::add(%x, %y, %const)
            %o = aten::mul(%t, %z)
            return (%o)"""
        replacement_pattern = """
        graph(%x, %y, %z, %const):
            %o = my::add_mul(%x, %y, %z, %const)
            return (%o)"""
        scripted_model = torch.jit.script(TestModule1())
        graph = scripted_model.graph
        value_mappings = [("o", "t")]
        for node in graph.nodes():
            if node.kind() == "aten::add":
                source_range_1 = node.sourceRange()
        torch._C._jit_pass_custom_pattern_based_rewrite_graph(
            input_pattern, replacement_pattern, scripted_model.graph, value_name_pairs=value_mappings)
        graph = scripted_model.graph
        for node in graph.nodes():
            if node.kind() == "my::add_mul":
                source_range_2 = node.sourceRange()
        self.assertTrue(source_range_1 == source_range_2)

        class TestModule2(torch.nn.Module):
            def __init__(self):
                super(TestModule2, self).__init__()

            def forward(self, x, y, z, w):
                x = x + y
                x = x + z
                x = x * z
                x = x * w
                return x - 2

        # Check source range preservation for two node transforms add -> my_add
        input_pattern = """
        graph(%x, %y, %const):
            %o = aten::add(%x, %y, %const)
            return (%o)"""
        replacement_pattern = """
        graph(%x, %y, %const):
            %o = my::add(%x, %y, %const)
            return (%o)"""
        scripted_model = copy.deepcopy(torch.jit.script(TestModule2()))
        graph_copy = scripted_model.graph.copy()
        value_mappings = [("o", "o")]
        source_range_add_1 = None
        for node in graph_copy.nodes():
            if source_range_add_1 is None and node.kind() == "aten::add":
                source_range_add_1 = node.sourceRange()
            if source_range_add_1 is not None and node.kind() == "aten::add":
                source_range_add_2 = node.sourceRange()
        torch._C._jit_pass_custom_pattern_based_rewrite_graph(
            input_pattern, replacement_pattern, graph_copy, value_name_pairs=value_mappings)
        source_range_my_add_1 = None
        for node in graph_copy.nodes():
            if source_range_my_add_1 is None and node.kind() == "my::add":
                source_range_my_add_1 = node.sourceRange()
            if source_range_my_add_1 is not None and node.kind() == "my::add":
                source_range_my_add_2 = node.sourceRange()
        self.assertTrue(source_range_add_1 == source_range_my_add_1)
        self.assertTrue(source_range_add_2 == source_range_my_add_2)

        # Check source range preservation for add-add -> double_add transform
        # fuse nodes
        input_pattern = """
        graph(%x, %y, %z, %const):
            %t = aten::add(%x, %y, %const)
            %o = aten::add(%t, %z, %const)
            return (%o)"""
        replacement_pattern = """
        graph(%x, %y, %z, %const):
            %o = my::double_add(%x, %y, %z, %const)
            return (%o)"""
        scripted_model = torch.jit.script(TestModule2())
        graph_copy = scripted_model.graph.copy()
        value_mappings = [("o", "t")]
        source_range_1 = None
        source_range_2 = None
        for node in graph_copy.nodes():
            if node.kind() == "aten::add":
                source_range_1 = node.sourceRange()
                break
        torch._C._jit_pass_custom_pattern_based_rewrite_graph(
            input_pattern, replacement_pattern, graph_copy, value_name_pairs=value_mappings)
        for node in graph_copy.nodes():
            if node.kind() == "my::double_add":
                source_range_2 = node.sourceRange()
        self.assertTrue(source_range_1 == source_range_2)

        # Check source range preservation for mul -> add + add transform
        # split node
        input_pattern = """
        graph(%x, %y):
            %t = aten::mul(%x, %y)
            return (%t)"""
        replacement_pattern = """
        graph(%x, %y):
            %t = my::add(%x, %y)
            %o = my::add(%t, %y)
            return (%o)"""
        scripted_model = torch.jit.script(TestModule2())
        graph_copy = scripted_model.graph.copy()
        value_mappings = [("t", "t"), ("o", "t")]
        source_range_mul_1 = None
        for node in graph_copy.nodes():
            if source_range_mul_1 is None and node.kind() == "aten::mul":
                source_range_mul_1 = node.sourceRange()
            if source_range_mul_1 is not None and node.kind() == "aten::mul":
                source_range_mul_2 = node.sourceRange()
        torch._C._jit_pass_custom_pattern_based_rewrite_graph(
            input_pattern, replacement_pattern, graph_copy, value_name_pairs=value_mappings)
        source_range_add_1 = None
        for node in graph_copy.nodes():
            if source_range_add_1 is None and node.kind() == "my::add":
                source_range_add_1 = node.sourceRange()
            if source_range_add_1 is not None and node.kind() == "my::add":
                source_range_add_2 = node.sourceRange()
        self.assertTrue(source_range_mul_1 == source_range_add_1)
        self.assertTrue(source_range_mul_2 == source_range_add_2)

        # Check lack of source range preservation for mul-mul-> double_mul transform
        input_pattern = """
        graph(%x, %y, %z):
            %t = aten::mul(%x, %y)
            %o = aten::mul(%t, %z)
            return (%o)"""
        replacement_pattern = """
        graph(%x, %y, %z):
            %o = my::double_mul(%x, %y, %z)
            return (%o)"""
        scripted_model = torch.jit.script(TestModule2())
        graph_copy = scripted_model.graph.copy()
        for node in graph_copy.nodes():
            if node.kind() == "aten::mul":
                source_range_1 = node.sourceRange()
        torch._C._jit_pass_custom_pattern_based_rewrite_graph(input_pattern, replacement_pattern, graph_copy)
        for node in graph_copy.nodes():
            if node.kind() == "my::double_mul":
                source_range_2 = node.sourceRange()
        self.assertFalse(source_range_1 == source_range_2)

    def test_expand_quantlint(self):
        pass

    def test_expand_fold_quant_inputs(self):
        pass

    def test_shape_analysis_broadcast(self):
        def broadcast(a, b):
            return a + b

        x = torch.randn(3, 1, 5, requires_grad=True)
        y = torch.randn(4, 1, 8, 5, requires_grad=True)

        graph = torch.jit.script(broadcast).graph
        torch._C._jit_pass_complete_shape_analysis(graph, (x, y), False)
        FileCheck().check("Double(4, 3, 8, 5, strides=[120, 40, 5, 1], device=cpu)").run(str(graph))

    def test_shape_analysis_unsqueeze_in_loop(self):
        input_str = """graph(%x.1 : Tensor):
          %4 : bool = prim::Constant[value=1]()
          %1 : int = prim::Constant[value=2]()
          %7 : int = prim::Constant[value=0]()
          # CHECK: FloatTensor(requires_grad=0, device=cpu) = prim::Loop
          %x : Tensor = prim::Loop(%1, %4, %x.1)
            # CHECK: : FloatTensor(requires_grad=0, device=cpu)):
            block0(%i : int, %x.6 : Tensor):
              # CHECK: FloatTensor(requires_grad=0, device=cpu) = aten::unsqueeze
              %x.3 : Tensor = aten::unsqueeze(%x.6, %7)
              -> (%4, %x.3)
          return (%x)"""
        graph = parse_ir(input_str)
        torch._C._jit_pass_complete_shape_analysis(graph, (torch.zeros(2, 2, dtype=torch.float32),), False)
        FileCheck().run(input_str, graph)

    def test_script_tensor_type(self):
        def foo(x, t: torch.dtype):
            return x.type(t)
        scr = torch.jit.script(foo)
        x = torch.rand(3, 4)
        for t in [torch.int8, torch.float64, torch.float32,
                  torch.bfloat16, torch.complex64, torch.complex128, torch.bool]:
            self.assertEqual(scr(x, t), foo(x, t))

    def test_shape_analysis_masked_select(self):
        input_str = """graph(%0 : Float(),
          %1 : Bool()):
          # CHECK: Float(*, requires_grad=0, device=cpu) = aten::masked_select
          %2 : Tensor = aten::masked_select(%0, %1) # test/test_jit.py:15261:0
          return (%2)"""
        graph = parse_ir(input_str)
        x = torch.ones(1, dtype=torch.float32)[0]
        mask = x.ge(0.5)
        torch._C._jit_pass_complete_shape_analysis(graph, (x, mask), False)
        FileCheck().run(input_str, graph)

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

        g, _ = torch.jit._get_trace_graph(doit, (x, y))
        self.run_pass('dce', g)
        self.run_pass('canonicalize', g)
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
        for node in g.nodes():
            self.assertTrue(g2.findNode(node.kind()) is not None)

    @unittest.skipIf(IS_SANDCASTLE, "gtest runs these in sandcastle")
    @unittest.skipIf(RUN_CUDA, "covered by test_cpp_cuda")
    @unittest.skipIf(not torch._C._jit_has_cpp_tests(), "Tests were not built, use BUILD_TEST=1")
    def test_cpp(self):
        from cpp.jit import tests_setup
        tests_setup.setup()
        torch._C._jit_run_cpp_tests()
        tests_setup.shutdown()

    def test_batchnorm(self):
        x = torch.ones(2, 2, 2, 2)
        g, outputs, inputs = torch.jit._get_trace_graph(nn.BatchNorm2d(2), x,
                                                        _force_outplace=True, return_inputs=True)
        m = self.createFunctionFromGraph(g)
        self.assertEqual(outputs, m(*inputs))

    def test_dropout(self):
        x = torch.ones(2, 2)
        with torch.random.fork_rng(devices=[]):
            g, outputs, inputs = torch.jit._get_trace_graph(nn.Dropout(0.6), x, return_inputs=True)
        with torch.random.fork_rng(devices=[]):
            m = self.createFunctionFromGraph(g)
            self.assertEqual(outputs, m(*inputs))

    @slowTest
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.LEGACY, 'Testing differentiable graph')
    def test_dropout_module_requires_grad(self):
        with enable_profiling_mode_for_profiling_tests():
            class MyModule(torch.nn.Module):
                def __init__(self, M):
                    super(MyModule, self).__init__()
                    self.dropout = torch.nn.Dropout(0.5)
                    self.linear = torch.nn.Linear(M, M)

                def forward(self, input):
                    input = self.dropout(input)
                    output = self.linear(input)
                    return output

            def profile(func, X):
                with torch.autograd.profiler.profile() as prof:
                    func(X)
                return [e.name for e in prof.function_events]

            M = 1000
            scripted = torch.jit.script(MyModule(M))
            # To reduce confusion about expected behaviors:
            #   requires_grad controls whether dropout is symbolically differentiated.
            #   training controls whether bernoulli_ is called inside symbolic differentiation of dropout.
            # * When requires_grad == training, the expected behaviors are obvious.
            # * When requires_grad=True and training=False, bernoulli_ might still show up in the graph.
            #   But it's in a branch that's not called. That's why we have separate checks for autograd
            #   profiler to make sure it's not run.
            # * When requires_grad=False and training=True, bernoulli_ must be run since it's the expected
            #   behavior for the dropout layer in training mode. It's independent of whether graph requires
            #   gradient. In fact bernoulli_ comes from autograd instead of autodiff in this case.
            for training in (True, False):
                if training:
                    scripted.train()
                else:
                    scripted.eval()
                for requires_grad in (True, False):
                    X = torch.randn(M, M, requires_grad=requires_grad)
                    if requires_grad:
                        FileCheck().check("aten::bernoulli_").run(scripted.graph_for(X, profile_and_replay=True))
                    self.assertEqual(training, 'aten::bernoulli_' in profile(scripted, X))

    @unittest.skipIf(GRAPH_EXECUTOR == ProfilingMode.SIMPLE, 'Testing differentiable graph')
    def test_dropout_func_requires_grad(self):
        def dropout_training(input):
            return F.dropout(input, 0.5, training=True)

        def dropout_eval(input):
            return F.dropout(input, 0.5, training=False)

        def profile(func, X):
            with torch.autograd.profiler.profile() as prof:
                func(X)
            return [e.name for e in prof.function_events]

        M = 1000
        scripted_training = torch.jit.script(dropout_training)
        scripted_eval = torch.jit.script(dropout_eval)
        # See comments in test_dropout_module_requires_grad.
        with disable_autodiff_subgraph_inlining():
            for requires_grad in (True, False):
                X = torch.randn(M, M, requires_grad=requires_grad)
                if requires_grad:
                    FileCheck().check("aten::bernoulli_").run(scripted_training.graph_for(X, profile_and_replay=True))
                self.assertIn('aten::bernoulli_', profile(scripted_training, X))
                self.assertNotIn('aten::bernoulli_', profile(scripted_eval, X))

    @unittest.skipIf(not RUN_CUDA, "test_dropout_cuda require CUDA")
    def test_dropout_cuda(self):
        # Dropout AD is dispatched to _fused_dropout in CUDA case,
        # which is not included in TestJitGeneratedFunctional
        def _zero_rate(t):
            return torch.true_divide((t == 0).sum(), t.numel())

        x = torch.ones(1000, 1000).cuda().requires_grad_()

        with enable_profiling_mode_for_profiling_tests():
            @torch.jit.script
            def func(x):
                return torch.nn.functional.dropout(x)

            with freeze_rng_state():
                out_ref = torch.nn.functional.dropout(x)
                grad_ref = torch.autograd.grad(out_ref.sum(), x)

            with freeze_rng_state():
                out = func(x)
                grad = torch.autograd.grad(out.sum(), x)

            # TODO(#40882): previously we assert exact matches between eager and JIT result:
            #  self.assertEqual(out, out_ref)
            #  self.assertEqual(grad, grad_ref)
            # This test was disabled during legacy -> profiling executor transition.
            # Currently JIT fused results doesn't match eager result exactly due to some changes merged in between.
            # We temporarily only check statstical difference but it should be reverted once the issue is fixed.
            self.assertEqual(_zero_rate(out), _zero_rate(out_ref), rtol=1e-3, atol=1e-4)
            self.assertEqual(_zero_rate(grad[0]), _zero_rate(grad_ref[0]), rtol=1e-3, atol=1e-4)

    def test_torch_ops_overloaded(self):
        with self.assertRaisesRegex(RuntimeError, "failed to many any schema"):
            torch.ops.aten.add("a", 1)
        self.assertEqual("ab", torch.ops.aten.add("a", "b"))
        a, b = torch.rand(3, 4), torch.rand(3, 4)
        self.assertEqual(a + b, torch.ops.aten.add(a, b))
        self.assertEqual(a + 1, torch.ops.aten.add(a, 1))

    def test_torch_ops_kwonly(self):
        a, b = torch.rand(3, 4), torch.rand(3, 4)
        with self.assertRaisesRegex(RuntimeError, "positional argument"):
            torch.ops.aten.add(a, b, 2)
        # h/t Chillee for this ambiguous case
        self.assertEqual(a.prod(1), torch.ops.aten.prod(a, 1))

    def test_torch_complex(self):
        def fn(real, img):
            return torch.complex(real, img)

        def fn_out(real, img, out):
            return torch.complex(real, img, out=out)
        self.checkScript(fn, (torch.rand(3, 4), torch.rand(3, 4), ))
        self.checkScript(fn, (torch.ones(5, 1, 4), torch.ones(5, 1, 4), ))
        self.checkScript(fn, (torch.zeros(1, 6), torch.ones(6, 1), ))
        self.checkScript(fn, (torch.zeros(1, 6), torch.zeros(6, 1), ))
        self.checkScript(fn, (torch.empty(3, 4), torch.empty(3, 4), ))

        real = torch.tensor([1, 2], dtype=torch.float32)
        img = torch.tensor([3, 4], dtype=torch.float32)
        out = torch.empty([3, 4], dtype=torch.complex64)
        self.checkScript(fn_out, (real, img, out, ))

        real = torch.tensor([5, 2], dtype=torch.float64)
        img = torch.tensor([3, 4], dtype=torch.float64)
        out = torch.empty([5, 2], dtype=torch.complex128)
        self.checkScript(fn_out, (real, img, out, ))

        real = torch.ones([1, 2])
        img = torch.ones([1, 2])
        out = torch.empty([1, 2], dtype=torch.complex128)
        self.checkScript(fn_out, (real, img, out, ))

        real = torch.ones([3, 8, 7])
        img = torch.ones([3, 8, 7])
        out = torch.empty([3, 8, 7], dtype=torch.complex128)
        self.checkScript(fn_out, (real, img, out, ))

        real = torch.empty([3, 2, 6])
        img = torch.empty([3, 2, 6])
        out = torch.empty([3, 2, 6], dtype=torch.complex128)
        self.checkScript(fn_out, (real, img, out, ))

        real = torch.zeros([1, 3])
        img = torch.empty([3, 1])
        out = torch.empty([3, 3], dtype=torch.complex128)
        self.checkScript(fn_out, (real, img, out, ))

        real = torch.ones([2, 5])
        img = torch.empty([2, 1])
        out = torch.empty([2, 5], dtype=torch.complex128)
        self.checkScript(fn_out, (real, img, out, ))

        real = torch.ones([2, 5])
        img = torch.zeros([2, 1])
        out = torch.empty([2, 5], dtype=torch.complex128)
        self.checkScript(fn_out, (real, img, out, ))

    def test_einsum(self):
        def check(fn, jitted, *args):
            self.assertGraphContains(jitted.graph, kind='aten::einsum')
            self.assertEqual(fn(*args), jitted(*args))

        def equation_format(x, y):
            return torch.einsum('i,j->ij', (x, y))

        def equation_format_varargs(x, y):
            return torch.einsum('i,j->ij', x, y)

        def sublist_format(x, y):
            return torch.einsum(x, [0], y, [1], [0, 1])

        x = make_tensor((5,), 'cpu', torch.float32)
        y = make_tensor((10,), 'cpu', torch.float32)

        for fn in [equation_format, equation_format_varargs, sublist_format]:
            check(fn, torch.jit.script(fn), x, y)
            check(fn, torch.jit.trace(fn, (x, y)), x, y)

    def test_python_ivalue(self):
        # Test if pure python object can be hold as IValue and conversion
        # between IValue and PyObject are correct
        # test for numpy object
        py_array = np.arange(15)
        ret_py_obj = torch._C._ivalue_debug_python_object(py_array)
        self.assertEqual(py_array, ret_py_obj)

        # test for function object
        ret_py_obj = torch._C._ivalue_debug_python_object(F.relu)
        self.assertEqual(F.relu, ret_py_obj)

        # test for memory management
        # we need to ensure IValue correctly call incref/decref to avoid
        # dangling behavior and potential memory leaks during conversions
        def test_func_scope_helper(inp):
            # create a scope and do the conversion -> ivalue -> pyobject
            # this func return a new pyobject that refcount + 1
            inp_refcount = sys.getrefcount(inp)
            ivalue_holder = torch._C._ivalue_debug_python_object(inp)
            self.assertEqual(inp_refcount + 1, sys.getrefcount(ivalue_holder))
            return ivalue_holder + 1

        test_input = 2200
        before_count = sys.getrefcount(test_input)
        test_func_scope_helper(test_input)
        after_count = sys.getrefcount(test_input)

        # after the test_func_scope_helper_call, the refcount of
        # test_input should be equal to the original refcount
        # otherwise we get either dangling pointer or memory leak!
        self.assertEqual(before_count, after_count)

    def test_decompose_addmm(self):
        def does_decompose():
            @torch.jit.script
            def addmm(mat, mat1, mat2):
                a = mat.addmm(mat1, mat2)
                b = mat.addmm(mat1, mat2, alpha=1.0, beta=1.0)
                return a + b

            mat = torch.randn(2, 2)
            mat1 = torch.randn(2, 4)
            mat2 = torch.randn(4, 2)

            out_ref = addmm(mat, mat1, mat2)
            self.run_pass('decompose_ops', addmm.graph)
            out_test = addmm(mat, mat1, mat2)
            self.assertEqual(out_ref, out_test)
            FileCheck().check_not("addmm").run(str(addmm.graph))

        def doesnt_decompose():
            @torch.jit.script
            def addmm(mat, mat1, mat2, alpha, beta):
                a = mat.addmm(mat1, mat2, alpha=4.20, beta=2.0)
                b = mat.addmm(mat1, mat2, alpha=int(alpha), beta=int(beta))

                return a + b

            orig = str(addmm.graph)
            self.run_pass('decompose_ops', addmm.graph)
            self.assertTrue(orig == str(addmm.graph))

        does_decompose()
        doesnt_decompose()

    @suppress_warnings
    def test_sparse_tensors(self):
        @torch.jit.ignore
        def get_sparse():
            return torch.sparse.FloatTensor(2, 3)

        @torch.jit.script
        def test_is_sparse(input):
            # type: (Tensor) -> bool
            return input.is_sparse

        script_out_is_sparse = test_is_sparse(get_sparse())
        script_out_is_dense = test_is_sparse(torch.randn(2, 3))
        self.assertEqual(script_out_is_sparse, True)
        self.assertEqual(script_out_is_dense, False)

        def test_basic_sparse(input):
            output = get_sparse()
            return output, input

        self.checkScript(test_basic_sparse, (get_sparse(),))
        self.checkScript(test_basic_sparse, (torch.tensor([1]),))

        def test_sparse_sum(input):
            return torch.sparse.sum(input)

        self.checkScript(test_sparse_sum, (get_sparse(),))

        def test_sparse_mm(input1, input2):
            return torch.sparse.mm(input1, input2)

        self.checkScript(test_sparse_mm, (get_sparse(), torch.randn(3, 4)))

        def test_sparse_addmm(input, input1, input2):
            return torch.sparse.addmm(input, input1, input2)

        def test_sparse_addmm_alpha_beta(input, input1, input2):
            return torch.sparse.addmm(input, input1, input2, 1.3, 1.5)

        self.checkScript(test_sparse_addmm, (torch.randn(2, 4), get_sparse(), torch.randn(3, 4)))
        self.checkScript(test_sparse_addmm_alpha_beta, (torch.randn(2, 4), get_sparse(), torch.randn(3, 4)))

    @suppress_warnings
    def test_sparse_csr_tensors(self):
        @torch.jit.ignore
        def get_sparse_csr():
            return torch.randn(3, 3).to_sparse_csr()

        @torch.jit.script
        def test_is_sparse_csr(input):
            # type: (Tensor) -> bool
            return input.is_sparse_csr

        script_out_is_sparse_csr = test_is_sparse_csr(get_sparse_csr())
        script_out_is_dense_csr = test_is_sparse_csr(torch.randn(3, 3))

        self.assertEqual(script_out_is_sparse_csr, True)
        self.assertEqual(script_out_is_dense_csr, False)

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    def test_device_not_equal(self):

        def compare_device(x: torch.device):
            return x != torch.device("cuda:0")

        def compare_two_device(x: torch.device, y: torch.device):
            return x != y

        self.checkScript(compare_device, (torch.device("cuda:0"),))
        self.checkScript(compare_two_device, (torch.device("cuda:0"), torch.device("cuda:1"), ))

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
        FileCheck().check("prim::Constant").run(constant_prop.graph)

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

    def test_constant_prop_exception(self):
        # checking y = a[4] does not error in constant propagation
        def bad_index(x):
            # type: (bool)
            y = 0
            if x:
                a = [1, 2, 3]
                y = a[4]
            return y

        self.checkScript(bad_index, (False,))

    def test_constant_prop_aliasing_type(self):
        @torch.jit.script
        def foo():
            return len([1]), len(torch.tensor([2]))

        FileCheck().check_dag("aten::tensor").check_dag("aten::len").run(foo.graph)

        @torch.jit.script
        def fn():
            if 1 == 1:
                return 1
            else:
                return 2

        FileCheck().check_not("prim::If").run(fn.graph)

    def test_unchecked_cast(self):
        def test(cond):
            # type: (bool)
            a = torch.tensor([10])
            if cond:
                b = None
            else:
                b = a
            if b is not None:
                b[0] = 5
            return a.int()

        self.checkScript(test, (True,))
        self.checkScript(test, (False,))

    def test_constant_prop_if_constant(self):
        @torch.jit.script
        def constant_prop(a, b):
            c0 = 1
            c1 = 1
            c2 = 1
            if bool(a):  # -> c0, c1
                if bool(b):  # -> c0
                    if 1 == 1:  # -> c0
                        c0 = c0 + 1
                        if 1 == 2:
                            c1 = c1 + 1
                            c2 = c2 + 1
            else:  # -> c0, c1
                c1 = c1 + 1

            if 1 == 1:  # inlined
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
                if 1 == 2:
                    a = 10
                if i == 5:
                    b = 2
                    c = 3
            print(a, b, c)

        graph = constant_prop.graph
        self.run_pass('constant_propagation', graph)
        self.assertTrue(graph.findNode("prim::Loop").outputsSize() == 2)

    # TODO(gmagogsfm): Refactor this test to reduce complexity.
    def test_constant_insertion(self):
        funcs_template = dedent('''
        def func():
            return {constant_constructor}
        ''')

        # constants: primitives: int, double, bool, str, lists of primitives,
        # and tuples
        def check_constant(constant_constructor):
            scope = {}
            funcs_str = funcs_template.format(constant_constructor=constant_constructor)
            execWrapper(funcs_str, globals(), scope)
            cu = torch.jit.CompilationUnit(funcs_str)
            f_script = cu.func
            self.run_pass('constant_propagation', f_script.graph)
            FileCheck().check_count("prim::Constant", 1, exactly=True).run(f_script.graph)
            self.assertEqual(scope['func'](), f_script())
            imported = self.getExportImportCopy(f_script)
            self.assertEqual(imported(), f_script())

        constants = ["None", "-.5", "0", "1", "True", "False", "''", "'a'", "'b'", "torch.tensor(1)",
                     "[True, False]", "[0., .5]", "[torch.tensor(4), torch.tensor(2)]", "[0, 1]", "['0', '1']",
                     "[True, None]", "[.5, None, .2]"]

        for type in ["Tensor", "str", "int", "float", "bool"]:
            constants.append("torch.jit.annotate(List[ " + type + "], [])")

        for constant in constants:
            check_constant(constant)

        for key_type in ["str", "int", "float"]:
            for value_type in ["Tensor", "bool", "str", "int", "float"]:
                check_constant("torch.jit.annotate(Dict[ " + key_type + ", " + value_type + "], {})")
                check_constant("torch.jit.annotate(Dict[ " + key_type + ", Optional[" + value_type + "]], {})")

        for i in range(len(constants)):
            for j in range(i + 1, len(constants)):
                tup_constant = constants[i] + ", " + constants[j]
                check_constant(tup_constant)

        dict_constants = []
        for i in range(len(constants)):
            # check_constant constructs the second dict with another Tensor
            # which fails the comparison
            if not isinstance(eval(constants[i]), (str, int, float)):
                continue
            for j in range(len(constants)):
                dict_constant = "{ " + constants[i] + ": " + constants[j] + "}"
                check_constant(dict_constant)
                dict_constants.append(dict_constant)
        constants = constants + dict_constants

        # testing node hashing
        funcs_template = dedent('''
        def func():
            print({constant_constructor})
        ''')
        single_elem_tuples = ("(" + x + ",)" for x in constants)
        input_arg = ", ".join(single_elem_tuples)
        scope = {}
        funcs_str = funcs_template.format(constant_constructor=input_arg)
        execWrapper(funcs_str, globals(), scope)
        cu = torch.jit.CompilationUnit(funcs_str)
        f_script = cu.func
        self.run_pass('constant_propagation', f_script.graph)
        # prim::None return adds one constant
        self.assertEqual(len(constants) + 1, str(f_script.graph).count("prim::Constant"))
        self.run_pass('cse', f_script.graph)
        # node hashing correctly working, no CSE occurs
        self.assertEqual(len(constants) + 1, str(f_script.graph).count("prim::Constant"))

        funcs_template = dedent('''
        def func():
            a = {constant_constructor}
            print(a)
            b = {constant_constructor}
            print(b)
        ''')

        # generate dicts with built-in types (excluding torch.Tensor)
        xprod = itertools.product(constants, constants)

        # test that equal tuples and dicts correctly work with node hashing
        for tup in ("(" + x + ",)" for x in constants):
            funcs_str = funcs_template.format(constant_constructor=tup)
            scope = {}
            execWrapper(funcs_str, globals(), scope)
            cu = torch.jit.CompilationUnit(funcs_str)
            f_script = cu.func
            self.run_pass('constant_propagation_immutable_types', f_script.graph)
            num_constants = str(f_script.graph).count("prim::Constant")
            self.run_pass('cse', f_script.graph)
            FileCheck().check_count("prim::Constant", num_constants, exactly=True).run(f_script.graph)

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

    @slowTest
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

    def test_unique_state_dict(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                shared_param = torch.nn.Parameter(torch.ones(1))
                self.register_parameter('w1', shared_param)
                self.register_parameter('w2', shared_param)

            def forward(self, input):
                return input + self.w1 + self.w2

        model = MyModule()
        unittest.TestCase.assertEqual(
            self, len(torch.jit._unique_state_dict(model, keep_vars=False)), 1)
        unittest.TestCase.assertEqual(
            self, len(torch.jit._unique_state_dict(model, keep_vars=True)), 1)

    def test_export_dropout(self):
        test = torch.nn.Dropout()
        test.eval()

        traced = torch.jit.trace(test, (torch.rand(3, 4),), check_trace=False)
        imported = self.getExportImportCopy(traced)
        x = torch.randn(3, 4)
        self.assertEqual(traced(x), imported(x))

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

        @torch.jit.ignore
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

        self.assertExpected(if_test.code, "if_test")
        self.assertExpected(if_one.code, "if_one")
        self.assertExpected(while_test.code, "while_test")
        self.assertExpected(while_if_test.code, "while_if_test")
        self.assertExpected(loop_use_test.code, "loop_use_test")
        self.assertExpected(python_op_name_test.code, "python_op_name_test")
        self.assertExpected(empty_int_list_test.code, "empty_int_list_test")
        self.assertExpected(empty_float_list_test.code, "empty_float_list_test")
        self.assertExpected(print_weird_test.code, "print_weird_test")

    def test_cu_escaped_number(self):
        cu = torch.jit.CompilationUnit('''
            def foo(a):
                print("hi\016")
        ''')
        self.assertExpected(cu.foo.code)

    def test_import_method(self):
        with torch._jit_internal._disable_emit_hooks():
            class Foo(torch.jit.ScriptModule):
                def __init__(self):
                    super(Foo, self).__init__()

                @torch.jit.script_method
                def forward(self, x, y):
                    return 2 * x + y

            foo = Foo()
            buffer = io.BytesIO()
            torch.jit.save(foo, buffer)

            buffer.seek(0)
            foo_loaded = torch.jit.load(buffer)
            self.assertExpected(foo_loaded.forward.code)

    @unittest.skip("temporarily disable the test for fwd compatibility")
    def test_non_ascii_string(self):
        class Foo(torch.jit.ScriptModule):
            def __init__(self):
                super(Foo, self).__init__()
                self.a = "Over \u0e55\u0e57 57"

            @torch.jit.script_method
            def forward(self, x, y):
                return self.a + "hi\xA1"

        foo = Foo()
        buffer = io.BytesIO()
        torch.jit.save(foo, buffer)

        buffer.seek(0)
        foo_loaded = torch.jit.load(buffer)
        self.assertExpected(foo_loaded.forward.code)

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
            def hints_bad_types(x, a=10, b=0.5):  # noqa: T484
                # type: (Tensor, float, int) -> Tensor
                return x + a + b
        with self.assertRaisesRegex(RuntimeError, "Expected a default value"):
            @torch.jit.script
            def bad_no_optional(x=None):
                # type: (Dict[str, int]) -> Dict[str, int]
                return x


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

    def test_mutable_default_values(self):
        with self.assertRaisesRegex(Exception, "Mutable default parameters"):
            @torch.jit.script
            def foo(x=(1, [])):
                # type: (Tuple[int, List[Tensor]])
                return x

        class Test(torch.nn.Module):
            def forward(self, input=[]):  # noqa: B006
                return input

        with self.assertRaisesRegex(Exception, "Mutable default parameters"):
            torch.jit.script(Test())

    def test_warnings(self):
        import warnings

        def fn(x):
            if bool(x < 2):
                warnings.warn("x is less than 2")
            return x

        class M(torch.nn.Module):
            def forward(self, x):
                if bool(x < 2):
                    warnings.warn("x is less than 2")
                return x


        scripted_mod = torch.jit.script(M())
        scripted_fn = torch.jit.script(fn)

        with warnings.catch_warnings(record=True) as warns:
            fn(torch.ones(1))

        with warnings.catch_warnings(record=True) as script_warns:
            scripted_fn(torch.ones(1))

        with warnings.catch_warnings(record=True) as script_mod_warns:
            scripted_mod(torch.ones(1))

        self.assertEqual(str(warns[0]), str(script_warns[0]))
        self.assertEqual(len(script_mod_warns), 1)
        self.assertEqual(str(warns[0].message), str(script_mod_warns[0].message))

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

    @unittest.skipIf(True, "TODO: re-enable with https://github.com/pytorch/pytorch/pull/29339")
    def test_torch_load_error(self):
        class J(torch.jit.ScriptModule):
            def __init__(self):
                super(J, self).__init__()

            @torch.jit.script_method
            def forward(self, input):
                return input + 100

        j = J()
        with TemporaryFileName() as fname:
            j.save(fname)
            with self.assertRaisesRegex(RuntimeError, "is a zip"):
                torch.load(fname)

    def test_torch_load_zipfile_check(self):
        @torch.jit.script
        def fn(x):
            return x + 10

        with TemporaryFileName() as fname:
            fn.save(fname)
            with io.open(fname, 'rb') as f:
                self.assertTrue(torch.serialization._is_zipfile(f))

    def test_python_bindings(self):
        lstm_cell = torch.jit.script(LSTMCellS)

        def lstm(x, hx, cx, w_ih, w_hh, b_ih, b_hh):
            for i in range(x.size(0)):
                hx, cx = lstm_cell(x[i], hx, cx, w_ih, w_hh, b_ih, b_hh)
            return hx

        slstm = torch.jit.script(lstm)

        inputs = get_lstm_inputs('cpu', training=True, seq_length=10)
        slstm(*inputs).sum().backward()
        global fw_graph
        fw_graph = slstm.graph_for(*inputs)
        nodes = list(fw_graph.nodes())
        tested_blocks = False
        for node in nodes:
            for output in node.outputs():
                self.assertTrue(hasattr(output, 'type'))
                self.assertTrue(output.type() is not None)
            for input in node.inputs():
                self.assertTrue(hasattr(input, 'type'))
                self.assertTrue(input.type() is not None)
            for block in node.blocks():
                tested_blocks = True
                self.assertTrue(hasattr(block, 'inputs'))
                self.assertTrue(hasattr(block, 'outputs'))
                for output in block.outputs():
                    self.assertTrue(hasattr(output, 'type'))
                    self.assertTrue(output.type() is not None)
                for input in block.inputs():
                    self.assertTrue(hasattr(input, 'type'))
                    self.assertTrue(input.type() is not None)
                self.assertTrue(hasattr(block, 'returnNode'))
                self.assertTrue(type(block.returnNode()) == torch._C.Node)
                self.assertTrue(hasattr(block, 'paramNode'))
                self.assertTrue(type(block.paramNode()) == torch._C.Node)
        self.assertTrue(tested_blocks)

    def test_export_opnames(self):
        class Foo(torch.jit.ScriptModule):
            def __init__(self):
                super(Foo, self).__init__()

            def one(self, x, y):
                # type: (Tensor, Tensor) -> Tensor
                return x + y

            def two(self, x):
                # type: (Tensor) -> Tensor
                return 2 * x

            @torch.jit.script_method
            def forward(self, x):
                # type: (Tensor) -> Tensor
                return self.one(self.two(x), x)

        class Bar(torch.jit.ScriptModule):
            def __init__(self):
                super(Bar, self).__init__()
                self.sub = Foo()

            @torch.jit.script_method
            def forward(self, x):
                # type: (Tensor) -> Tensor
                return self.sub.forward(x)

        bar = Bar()
        ops = torch.jit.export_opnames(bar)
        expected = ['aten::add.Tensor', 'aten::mul.Scalar']
        self.assertTrue(set(expected).issubset(set(ops)))

    def test_pytorch_jit_env_off(self):
        import subprocess
        env = os.environ.copy()
        env['PYTORCH_JIT'] = '0'
        try:
            subprocess.check_output([sys.executable, '-c', 'import torch'], env=env)
        except subprocess.CalledProcessError as e:
            raise RuntimeError("Could not 'import torch' with PYTORCH_JIT=0") from e

    def test_print_op_module(self):
        # Issue #19351: python2 and python3 go through different paths.
        # python2 returns '<module 'torch.ops' (built-in)>'
        # python3 uses __file__ and return
        # '<module 'torch.ops' from '/scratch/ailzhang/pytorch/torch/_ops.py'>'
        s = str(torch.ops)
        self.assertRegex(s, r'ops')

    def test_print_classes_module(self):
        s = str(torch.classes)
        self.assertRegex(s, r'classes')

    def test_print_torch_ops_modules(self):
        s = str(torch._ops.ops.quantized)
        self.assertRegex(s, r'torch.ops')
        s = str(torch._ops.ops.atan)
        self.assertRegex(s, r'torch.ops')

    @unittest.skipIf(IS_WINDOWS, 'TODO: fix occasional windows failure')
    def test_profiler(self):
        prev_opt = torch._C._get_graph_executor_optimize()
        torch._C._set_graph_executor_optimize(False)

        def other_fn(x):
            return x * 2

        x = torch.rand(3, 4)
        traced_other_fn = torch.jit.trace(other_fn, x)

        def fn(x):
            y = traced_other_fn(x)
            fut = torch.jit._fork(traced_other_fn, x)
            y = torch.jit._wait(fut)
            return y

        traced_fn = torch.jit.trace(fn, x)
        with torch.autograd.profiler.profile() as prof:
            traced_fn(x)

        # expecting to see other_fn TS function call
        # with cpu time >= mul cpu time and
        # a forked other_fn

        mul_events = defaultdict(int)
        other_fn_events = defaultdict(int)
        for e in prof.function_events:
            if e.name == "aten::mul":
                self.assertTrue(e.thread not in mul_events)
                mul_events[e.thread] = e.time_range.elapsed_us()
            elif e.name == "other_fn":
                self.assertTrue(e.thread not in other_fn_events)
                other_fn_events[e.thread] = e.time_range.elapsed_us()

        self.assertTrue(len(mul_events) == 2)
        self.assertTrue(len(other_fn_events) == 2)

        for thread, mul_time in mul_events.items():
            self.assertTrue(thread in other_fn_events)
            self.assertTrue(other_fn_events[thread] >= mul_time)

        torch._C._set_graph_executor_optimize(prev_opt)

    def test_hide_source_ranges_context_manager(self):
        @torch.jit.script
        def foo(x):
            return torch.add(x, x)

        graph = foo.graph
        source_range_regex = "# .*\\.py"
        self.assertRegex(graph.__repr__(), source_range_regex)
        with torch.jit._hide_source_ranges():
            self.assertNotRegex(graph.__repr__(), source_range_regex)
            self.assertRegex(graph.str(print_source_ranges=True), source_range_regex)
        self.assertRegex(graph.__repr__(), source_range_regex)



# UBSAN per-function exclusions don't seem to work with OpenMP pragmas,
# and we have to disable the failing tests here instead.
UBSAN_DISABLED_TESTS = [
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

# known to be failing in tracer
EXCLUDE_TRACED = {
    # The following fail due to #12024.
    # A prim::ListConstruct is involved and the indices get traced as TensorType,
    # which always require_grad. This causes a crash in autodiff.
    'test___getitem___adv_index',
    'test___getitem___adv_index_beg',
    'test___getitem___adv_index_comb',
    'test___getitem___adv_index_dup',
    'test___getitem___adv_index_sub',
    'test___getitem___adv_index_sub_2',
    'test___getitem___adv_index_sub_3',
    'test___getitem___adv_index_var',

    # jit doesn't support sparse tensors.
    'test_to_sparse',
    'test_to_sparse_dim',
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
    'test_slogdet_batched_1x1_neg_det',
    'test_slogdet_batched_pos_det',
    'test_slogdet_batched_symmetric',
    'test_slogdet_batched_symmetric_pd',
    'test_slogdet_batched_distinct_singular_values'
}

# chunk returns a list in scripting and we don't unpack the list,
# Thus it won't be replaced by ConstantChunk and run AD.
# It's explicitly checked in test_chunk_constant_script_ad
# Similary for split, it's replaced by split_with_sizes in tracing,
# but we don't have AD formula for aten::split(Tensor, int[], int),
# an op registered in JIT so AD is not triggered in scripting.
EXCLUDE_SCRIPT_AD_CHECK = {
    'test_chunk',
    'test_chunk_dim',
    'test_chunk_dim_neg0',
    'test_split_size_list',
    'test_split_size_list_dim',
    'test_split_size_list_dim_neg0',
    'test_tensor_indices_sections',
    'test_tensor_indices_sections_dim',
    'test_tensor_indices_sections_dim_neg0',
    'test_tensor_split_sections',
    'test_tensor_split_sections_dim',
    'test_tensor_split_sections_dim_neg0'
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

EXCLUDE_ALIAS = {
    # aliases, which may appear in method_tests but are tested elsewhere
    'true_divide',

    # Disable tests for lu from common_methods_invocations.py
    # TODO(@nikitaved) Enable jit tests once autograd.Function does support scripting
    'lu'
}

class TestJitGeneratedModule(JitTestCase):
    pass


class TestJitGeneratedFunctional(JitTestCase):
    pass


def add_nn_functional_test(name, self_size, args, variant_name='', check_ad=(), skipTestIf=(),
                           output_process_fn=lambda x: x, kwargs=None):
    test_name = 'test_nn_' + name

    if variant_name != '':
        test_name = test_name + '_' + variant_name

    no_grad = variant_name == 'inplace'

    @suppress_warnings
    def do_test(self, name=name, args=args, test_name=test_name, check_ad=check_ad):
        torch.manual_seed(2)

        self_variable = create_input((self_size,))[0][0]

        # need to record this because methods can change the size (e.g. unsqueeze)
        args_variable, kwargs_variable = create_input(args, call_kwargs=kwargs)

        self_tensor = deepcopy(self_variable.data)
        args_tensor = deepcopy(unpack_variables(args_variable))

        if not no_grad:
            output_variable = getattr(F, name)(self_variable, *args_variable, **kwargs_variable)

        def fn(*inputs, **kwargs):
            return getattr(F, name)(*inputs, **kwargs)

        f_args_variable = (self_variable,) + args_variable
        f_args_tensor = (self_tensor,) + args_tensor
        should_autodiff_node, autodiff_nodes, fusible_nodes = normalize_check_ad(check_ad, name)

        if test_name not in EXCLUDE_SCRIPT:
            def run_test():
                # XXX: this test should always run with disable_autodiff_subgraph_inlining(True),
                #      so that we don't regress on autodiff support.
                with disable_autodiff_subgraph_inlining():
                    script_fn = create_script_fn(self, name, 'nn_functional')
                    check_against_reference(self, script_fn, fn, output_process_fn,
                                            f_args_variable, kwargs_variable, no_grad=no_grad)
                    # For tests we disabled AD subgraph inlining, make sure it's not falling back to autograd
                    if (doAutodiffCheck(test_name)):
                        self.assertAutodiffNode(script_fn.last_graph, should_autodiff_node, autodiff_nodes, fusible_nodes)

            if test_name in EXCLUDE_PYTHON_PRINT:
                with torch._jit_internal._disable_emit_hooks():
                    run_test()
            else:
                run_test()

    post_add_test(test_name, skipTestIf, do_test, TestJitGeneratedFunctional)


def add_nn_module_test(*args, **kwargs):
    no_grad = False if 'no_grad' not in kwargs else kwargs['no_grad']

    if 'desc' in kwargs and 'eval' in kwargs['desc']:
        # eval() is not supported, so skip these tests
        return

    test_name = get_nn_mod_test_name(**kwargs)

    @suppress_warnings
    def do_test(self):
        if test_name in EXCLUDE_SCRIPT_MODULES:
            return
        if not kwargs.get('check_jit', True):
            raise unittest.SkipTest('module test skipped on JIT')

        module_name = get_nn_module_name_from_kwargs(**kwargs)

        if 'constructor' in kwargs:
            nn_module = kwargs['constructor']
        else:
            nn_module = getattr(torch.nn, module_name)

        if "FunctionalModule" in str(nn_module):
            return

        if 'constructor_args_fn' in kwargs:
            constructor_args = kwargs['constructor_args_fn']()
        else:
            constructor_args = kwargs.get('constructor_args', ())

        def create_script_module(*args, **kwargs):
            """Construct a script module that passes arguments through to self.submodule"""
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

            def make_module(script):
                module = TheModule()
                # check __repr__
                str(module)
                module.define(script)
                return module

            module = make_module(script)
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
        dtype = torch.double
        if 'input_fn' in kwargs:
            input = kwargs['input_fn']()
            if isinstance(input, Tensor):
                input = (input,)

            if all(tensor.is_complex() for tensor in input):
                dtype = torch.cdouble
        else:
            input = (kwargs['input_size'],)

        if 'target_size' in kwargs:
            input = input + (kwargs['target_size'],)
        elif 'target_fn' in kwargs:
            if torch.is_tensor(input):
                input = (input,)
            input = input + (kwargs['target_fn'](),)
        elif 'target' in kwargs:
            input = input + (kwargs['target'],)

        # Extra parameters to forward()
        if 'extra_args' in kwargs:
            input = input + kwargs['extra_args']

        args_variable, kwargs_variable = create_input(input, dtype=dtype)
        f_args_variable = deepcopy(unpack_variables(args_variable))

        # TODO(issue#52052) Neither this nor no_grad should be required
        # if check_against_reference() is updated to check gradients
        # w.r.t. weights and then only check w.r.t. inputs if any
        # inputs require it.
        any_requires_grad = any(input.requires_grad for input in f_args_variable)

        # Check against Python module as reference
        check_against_reference(self, create_script_module, create_nn_module,
                                lambda x: x, f_args_variable,
                                no_grad=no_grad or not any_requires_grad)

    if 'slowTest' in kwargs:
        do_test = slowTest(do_test)

    post_add_test(test_name, (), do_test, TestJitGeneratedModule)


def post_add_test(test_name, skipTestIf, do_test, test_class):
    assert not hasattr(test_class, test_name), 'Two tests have the same name: ' + test_name

    for skip in skipTestIf:
        do_test = skip(do_test)

    if not (TEST_WITH_UBSAN and test_name in UBSAN_DISABLED_TESTS):
        setattr(test_class, test_name, do_test)


def normalize_check_ad(check_ad, name):
    # normalized check_ad is 3-element tuple: (bool, List[str], List[str])
    if len(check_ad) == 0:
        check_ad = [False, ['aten::' + name], []]
    elif len(check_ad) == 1:
        check_ad = [check_ad[0], ['aten::' + name], []]
    elif len(check_ad) == 2:
        check_ad = [check_ad[0], check_ad[1], []]
    elif len(check_ad) == 3:
        check_ad = list(check_ad)
    else:
        raise Exception('Invalid check_ad, requires (bool, str|List[str], str|List[str])')

    check_ad = [[t] if isinstance(t, str) else t for t in check_ad]

    return check_ad


class TestProducerVersion(TestCase):

    def test_version(self):
        # issue gh-32561
        self.assertTrue(torch.__version__.startswith(torch.onnx.producer_version))


for test in nn_functional_tests:
    add_nn_functional_test(*test)

for test in module_tests + new_module_tests + additional_module_tests:
    add_nn_module_test(**test)

for test in criterion_tests:
    test['no_grad'] = True
    add_nn_module_test(**test)

if __name__ == '__main__':
    run_tests()
    import test_jit_py3
    import jit.test_module_interface
    suite = unittest.findTestCases(test_jit_py3)
    unittest.TextTestRunner().run(suite)
    suite = unittest.findTestCases(jit.test_module_interface)
    unittest.TextTestRunner().run(suite)
