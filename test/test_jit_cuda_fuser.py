# Owner(s): ["oncall: jit"]

import contextlib
import unittest
import os
import random
import enum
import copy
from functools import reduce
import operator
import warnings

import torch
from torch.nn import functional
from torch.profiler import profile, ProfilerActivity

from torch.testing._internal.codegen.random_topo_test import runDefaultTestWithSeed
from torch.testing._internal.common_cuda import TEST_MULTIGPU
from torch.testing._internal.common_device_type import instantiate_device_type_tests, ops, OpDTypes
from torch.testing._internal.common_jit import JitCommonTestCase
from torch.testing._internal.common_methods_invocations import op_db, SampleInput
from torch.testing._internal.common_utils import run_tests, ProfilingMode, GRAPH_EXECUTOR, TEST_WITH_ROCM, slowTest, \
    is_iterable_of_tensors, freeze_rng_state, skipIfRocm
from torch.testing._internal.jit_utils import clone_inputs, get_traced_sample_variant_pairs, JitTestCase, RUN_CUDA
from torch.testing._internal.jit_metaprogramming_utils import create_traced_fn
from torch.testing import FileCheck

from jit.test_fuser_common import TestFuserCommon  # noqa: F401

import itertools
import numpy as np
import math

from torch.autograd.gradcheck import gradcheck

from typing import List

RUN_NVFUSER = RUN_CUDA
CUDA_MAJOR, CUDA_MINOR = 0, 0

if RUN_NVFUSER and torch.version.cuda is not None:
    CUDA_MAJOR, CUDA_MINOR = (int(x) for x in torch.version.cuda.split('.')[:2])

if 'PYTORCH_NVFUSER_ENABLE' not in os.environ:
    os.environ['PYTORCH_NVFUSER_ENABLE'] = ""
os.environ['PYTORCH_NVFUSER_ENABLE'] = 'linear_decomposition,conv_decomposition,' + os.environ['PYTORCH_NVFUSER_ENABLE']
if 'PYTORCH_NVFUSER_DISABLE' not in os.environ:
    os.environ['PYTORCH_NVFUSER_DISABLE'] = ""
os.environ['PYTORCH_NVFUSER_DISABLE'] = 'fallback,fma,' + os.environ['PYTORCH_NVFUSER_DISABLE']
os.environ['PYTORCH_NVFUSER_JIT_OPT_LEVEL'] = '0'
# TODO: enable complex when we fixes the extremal cases in OpInfo
# see issue https://github.com/csarofeen/pytorch/issues/1730"
# os.environ['PYTORCH_NVFUSER_ENABLE'] = 'complex'

if GRAPH_EXECUTOR == ProfilingMode.PROFILING:
    torch._C._jit_set_texpr_fuser_enabled(False)
    torch._C._jit_set_profiling_executor(True)
    torch._C._jit_set_profiling_mode(True)

FUSION_GROUP = 'prim::CudaFusionGroup'
FUSION_GUARD = 'prim::CudaFusionGuard'
# TODO: revert disabled alias ops
ALIAS_TEST_DISABLED = True


@contextlib.contextmanager
def nvfuser_singleton_fusion(flag):
    old_value = torch._C._jit_set_nvfuser_single_node_mode(flag)
    try:
        yield
    finally:
        torch._C._jit_set_nvfuser_single_node_mode(old_value)

@contextlib.contextmanager
def nvfuser_horizontal_fusion(flag):
    old_value = torch._C._jit_set_nvfuser_horizontal_mode(flag)
    try:
        yield
    finally:
        torch._C._jit_set_nvfuser_horizontal_mode(old_value)

def is_pre_volta():
    if not RUN_NVFUSER:
        return False
    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    return prop.major < 7

TEST_BF16 = RUN_NVFUSER and torch.cuda.is_bf16_supported()

TEST_LARGE_TENSOR = RUN_NVFUSER
if RUN_NVFUSER:
    torch.ones(1).cuda()  # initialize cuda context
    TEST_LARGE_TENSOR = torch.cuda.get_device_properties(0).total_memory >= 12e9

class CudaFuserTestOptions():
    def __init__(self):
        self.old_cpu_fuse = torch._C._jit_can_fuse_on_cpu()
        self.old_gpu_fuse = torch._C._jit_can_fuse_on_gpu()
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
        self.old_guard = torch._C._jit_set_nvfuser_guard_mode(False)
        torch._C._debug_set_autodiff_subgraph_inlining(False)
        self.old_value = torch._C._jit_set_autocast_mode(True)

        if(RUN_CUDA):
            self.old_nvfuser = torch._C._jit_set_nvfuser_enabled(True)

    def restore(self):
        if(RUN_CUDA):
            torch._C._jit_set_nvfuser_enabled(self.old_nvfuser)
        torch._C._jit_override_can_fuse_on_cpu(self.old_cpu_fuse)
        torch._C._jit_override_can_fuse_on_gpu(self.old_gpu_fuse)
        torch._C._jit_set_nvfuser_guard_mode(self.old_guard)
        torch._C._debug_set_autodiff_subgraph_inlining(True)
        torch._C._jit_set_autocast_mode(self.old_value)

class TestCudaFuser(JitTestCase):
    def assertEqual(self, *args, **kwargs):
        kwargs["exact_layout"] = True
        super(JitTestCase, self).assertEqual(*args, **kwargs)

    def _getSubgraphInFusion(self, graph):
        num_node = 0
        subgraph = None

        def count(block, ret):
            for n in block.nodes():
                if n.kind() == FUSION_GROUP:
                    ret[0] = ret[0] + 1
                    self.assertTrue(n.hasAttribute('Subgraph'))
                    ret[1] = n.g('Subgraph')
                for block in n.blocks():
                    count(block, ret)
        ret = [num_node, subgraph]
        count(graph, ret)
        self.assertEqual(ret[0], 1)
        return ret[1]

    def setUp(self):
        super(TestCudaFuser, self).setUp()

        self.skip_node_list = []
        disabled_ops = ("aten::batch_norm",
                        "aten::_batch_norm_impl_index",
                        "aten::_batch_norm_impl_index_backward",
                        "aten::native_batch_norm_backward")
        for op in disabled_ops:
            disabled_flag = torch._C._jit_set_nvfuser_skip_node_kind(op, False)
            if disabled_flag:
                torch._C._jit_set_nvfuser_skip_node_kind(op, True)
                self.skip_node_list.append(op)

        # cpu backup to avoid errors in case this is run on a CPU-only machine
        dev = 'cuda' if RUN_NVFUSER else 'cpu'
        self.special_values = torch.tensor(
            [float("-inf"), -10, -math.pi,
                -1, -0.5, 0, 1, 0.5,
                math.pi, 10, float("inf"),
                float("nan")], dtype=torch.float, device=dev)

        self.int_types = [
            torch.int8,
            torch.uint8,
            torch.int16,
            torch.int32,
            torch.int64
        ]

        self.support_tensor_dtypes = [
            torch.int32,
            torch.int64,
            torch.float16,
            torch.float32,
            torch.float64,
            torch.bool,
            torch.complex64,
            torch.complex128,
        ]
        if TEST_BF16:
            self.support_tensor_dtypes.append(torch.bfloat16)

        if(RUN_NVFUSER):
            self.cuda_fuser_options = CudaFuserTestOptions()

    def tearDown(self):
        # restoring skip node to the configuration before tests
        for op in self.skip_node_list:
            disabled_flag = torch._C._jit_set_nvfuser_skip_node_kind(op, False)
            if not disabled_flag:
                torch._C._jit_set_nvfuser_skip_node_kind(op, True)

        if(RUN_NVFUSER):
            self.cuda_fuser_options.restore()
        super(TestCudaFuser, self).tearDown()

    def _run_helper(self, jit_op, op, *args, check_stride=False, num_fusion=1, check_runs=1):
        seed = 123
        torch.cuda.manual_seed_all(seed)
        jit_o = jit_op(*args)

        for i in range(check_runs):
            torch.cuda.manual_seed_all(seed + i)
            jit_o = jit_op(*args)
            torch.cuda.manual_seed_all(seed + i)
            o = op(*args)

            if type(jit_o) is torch.Tensor:
                jit_o = [jit_o, ]
                o = [o, ]

            for oo, jit_oo in zip(o, jit_o):
                self.assertEqual(oo.dtype, jit_oo.dtype)
                self.assertEqual(oo, jit_oo)
                if check_stride:
                    self.assertEqual(oo.stride(), jit_oo.stride())

        self.assertGraphContainsExactly(jit_op.graph_for(*args), FUSION_GUARD, num_fusion, consider_subgraphs=True)

    def _run_training_helper(self, jit_op, op, grads, *args):
        torch.cuda.manual_seed_all(123)
        jit_o = jit_op(*args)
        jit_g = jit_o.backward(grads)
        torch.cuda.manual_seed_all(123)
        jit_o = jit_op(*args)
        jit_g = jit_o.backward(grads)
        torch.cuda.manual_seed_all(123)
        jit_o = jit_op(*args)
        jit_g = jit_o.backward(grads)
        torch.cuda.manual_seed_all(123)
        o = op(*args)
        g = o.backward(grads)
        self.assertEqual(o, jit_o)
        self.assertEqual(g, jit_g)
        self.assertGraphContainsExactly(jit_op.graph_for(*args), FUSION_GUARD, 1, consider_subgraphs=True)
        bwd_graph = list(
            list(jit_op.get_debug_state().execution_plans.values())[
                0].code.grad_executor_states()[0].execution_plans.values()
        )[0].graph
        self.assertGraphContainsExactly(bwd_graph, FUSION_GUARD, 1, consider_subgraphs=True)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_half(self):
        def t(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, alpha: float):
            o_16 = torch.add(x, y)
            o_32_a = torch.add(y, z, alpha=alpha)
            o_32_b = torch.add(o_16, z)
            return (o_16, o_32_a, o_32_b)

        t_jit = torch.jit.script(t)
        alpha = 0.5
        # stick to integers, this avoid the numerical difference due to our
        # promotion
        x = torch.randint(0, 256, (4, 8)).to(dtype=torch.float16, device="cuda")
        y = torch.randint(0, 256, (4, 8)).to(dtype=torch.float16, device="cuda")
        z = torch.randint(0, 256, (4, 8)).to(dtype=torch.float16, device="cuda")
        jit_o = t_jit(x, y, z, alpha)
        jit_o = t_jit(x, y, z, alpha)
        o = t(x, y, z, alpha)
        for oo, jit_oo in zip(o, jit_o):
            self.assertEqual(oo.dtype, jit_oo.dtype)
            self.assertEqual(oo, jit_oo)
        self.assertGraphContains(t_jit.graph_for(x, y, z, alpha), FUSION_GUARD)


    @unittest.skipIf(not TEST_BF16, "device does not support BFloat16")
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_bfloat(self):
        def t(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, alpha: float):
            o_16 = torch.add(x, y)
            o_32_a = torch.add(y, z, alpha=alpha)
            o_32_b = torch.add(o_16, z)
            return (o_16, o_32_a, o_32_b)

        t_jit = torch.jit.script(t)
        alpha = 0.5
        # stick to integers, this avoid the numerical difference due to our
        # promotion
        x = torch.randint(0, 256, (4, 8)).to(dtype=torch.bfloat16, device="cuda")
        y = torch.randint(0, 256, (4, 8)).to(dtype=torch.bfloat16, device="cuda")
        z = torch.randint(0, 256, (4, 8)).to(dtype=torch.bfloat16, device="cuda")
        jit_o = t_jit(x, y, z, alpha)
        jit_o = t_jit(x, y, z, alpha)
        o = t(x, y, z, alpha)
        for oo, jit_oo in zip(o, jit_o):
            self.assertEqual(oo.dtype, jit_oo.dtype)
            self.assertEqual(oo, jit_oo)
        self.assertGraphContains(t_jit.graph_for(x, y, z, alpha), FUSION_GUARD)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_const(self):
        def t(x, y):
            o = x + y
            o = o + 2.0
            return o
        t_jit = torch.jit.script(t)
        x = torch.randn(4, 8, dtype=torch.float, device="cuda")
        y = torch.randn(4, 8, dtype=torch.float, device="cuda")
        jit_o = t_jit(x, y)
        jit_o = t_jit(x, y)
        o = t(x, y)
        self.assertEqual(o, jit_o)
        self.assertGraphContains(t_jit.graph_for(x, y), FUSION_GUARD)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_chunk(self):
        def t(x, y, z, q):
            o = x + q
            x0, x1 = torch.chunk(o, 2)
            o = x0 + x1
            o = o + y
            o = o * z
            o = torch.relu(o)
            return o
        t_jit = torch.jit.script(t)
        x = torch.randn(4, 8, dtype=torch.float, device="cuda")
        y = torch.randn(2, 8, dtype=torch.float, device="cuda")
        z = torch.randn(2, 8, dtype=torch.float, device="cuda")
        q = torch.randn(4, 8, dtype=torch.float, device="cuda")
        jit_o = t_jit(x, y, z, q)
        jit_o = t_jit(x, y, z, q)
        o = t(x, y, z, q)
        self.assertEqual(o, jit_o)
        self.assertGraphContains(t_jit.graph_for(x, y, z, q), FUSION_GUARD)

    @unittest.skipIf(is_pre_volta(), "reduction not supported in pre volta device")
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_reduction_dtypes_axis(self):

        for op in [torch.sum, torch.mean, torch.amax, torch.var, torch.std]:
            for dtype in [torch.float16, torch.float32, torch.double]:
                for axis in [-1, 2, 0]:
                    def make_func(op):
                        def func(x: torch.Tensor):
                            o = torch.mul(x, 2.0)
                            o = op(o, dim=[axis])
                            return o
                        return func

                    x = torch.randn(8, 4, 16, dtype=dtype, device="cuda")
                    t = make_func(op)
                    t_jit = torch.jit.trace(t, x)
                    jit_o = t_jit(x)
                    jit_o = t_jit(x)
                    o = t(x)
                    self.assertEqual(o.dtype, jit_o.dtype)
                    self.assertTrue(self._compare("comparing output failed", o, jit_o, 1e-4))
                    self.assertGraphContains(t_jit.graph_for(x), FUSION_GUARD)

    @unittest.skipIf(is_pre_volta(), "reduction not supported in pre volta device")
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_variance(self):

        for op in [torch.var, torch.std]:
            for dtype in [torch.float16, torch.float32, torch.double]:
                for axis in [-2, -1, 2, 1]:
                    for unbiased in [False, True]:
                        def make_func(op):
                            def func(x: torch.Tensor):
                                o = torch.mul(x, 2.0)
                                o = op(o, dim=[axis])
                                return o
                            return func

                        x = torch.randn(8, 4, 16, dtype=dtype, device="cuda")
                        t = make_func(op)
                        t_jit = torch.jit.trace(t, x)
                        jit_o = t_jit(x)
                        jit_o = t_jit(x)
                        o = t(x)
                        self.assertEqual(o.dtype, jit_o.dtype)
                        self.assertTrue(self._compare("comparing output failed", o, jit_o, 1e-4))
                        self.assertGraphContains(t_jit.graph_for(x), FUSION_GUARD)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_scalar_input(self):
        def t(x: torch.Tensor, y: torch.Tensor, z: float):
            o = x + y
            o = o + z
            return o
        t_jit = torch.jit.script(t)
        x = torch.randn(4, 8, 32, 32, dtype=torch.float, device="cuda")
        y = torch.randn(4, 8, 1, 32, dtype=torch.float, device="cuda")
        y = y.expand(4, 8, 32, 32)
        jit_o = t_jit(x, y, 2.0)
        jit_o = t_jit(x, y, 2.0)
        o = t(x, y, 2.0)
        self.assertEqual(o, jit_o)
        self.assertGraphContains(t_jit.graph_for(x, y, 2.0), FUSION_GUARD)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_broadcasting_0(self):

        def t(x: torch.Tensor, y: torch.Tensor, z: float):
            o = x + y
            o = o + z
            return o
        t_jit = torch.jit.script(t)
        x = torch.randn(4, 8, 32, 32, dtype=torch.float, device="cuda")
        y = torch.randn(32, 32, dtype=torch.float, device="cuda")
        jit_o = t_jit(x, y, 2.0)
        jit_o = t_jit(x, y, 2.0)
        o = t(x, y, 2.0)
        self.assertEqual(o, jit_o)
        subgraph = self._getSubgraphInFusion(t_jit.graph_for(x, y, 2.0))
        self.assertGraphContainsExactly(subgraph, 'aten::add', 2, consider_subgraphs=False)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_broadcasting_1(self):

        def t(x: torch.Tensor, y: torch.Tensor, z: float):
            o = x + y
            o = o + z
            return o
        t_jit = torch.jit.script(t)
        x = torch.randn(4, 8, 32, 32, dtype=torch.float, device="cuda")
        y = torch.randn(1, 32, 32, dtype=torch.float, device="cuda")
        jit_o = t_jit(x, y, 2.0)
        jit_o = t_jit(x, y, 2.0)
        o = t(x, y, 2.0)
        self.assertEqual(o, jit_o)
        subgraph = self._getSubgraphInFusion(t_jit.graph_for(x, y, 2.0))
        self.assertGraphContainsExactly(subgraph, 'aten::add', 2, consider_subgraphs=False)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_broadcasting_2(self):

        def t(x: torch.Tensor, y: torch.Tensor, z: float):
            o = x + y
            o = o + z
            return o
        t_jit = torch.jit.script(t)
        x = torch.randn(4, 1, 32, 32, dtype=torch.float, device="cuda")
        y = torch.randn(8, 32, 32, dtype=torch.float, device="cuda")
        jit_o = t_jit(x, y, 2.0)
        jit_o = t_jit(x, y, 2.0)
        o = t(x, y, 2.0)
        self.assertEqual(o, jit_o)
        subgraph = self._getSubgraphInFusion(t_jit.graph_for(x, y, 2.0))
        self.assertGraphContainsExactly(subgraph, 'aten::add', 2, consider_subgraphs=False)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_broadcasting_3(self):

        def t(x: torch.Tensor, y: torch.Tensor, z: float):
            o = x + y
            o = o + z
            return o
        t_jit = torch.jit.script(t)
        x = torch.randn(8, 17, 8, dtype=torch.float, device="cuda")
        y = torch.randn(8, 17, 1, dtype=torch.float, device="cuda")
        jit_o = t_jit(x, y, 2.0)
        jit_o = t_jit(x, y, 2.0)
        o = t(x, y, 2.0)
        self.assertEqual(o, jit_o)
        subgraph = self._getSubgraphInFusion(t_jit.graph_for(x, y, 2.0))
        self.assertGraphContainsExactly(subgraph, 'aten::add', 2, consider_subgraphs=False)

    # test_broadcasting_partition_logic_X
    # Testing partition logic that is capable to avoid creating unsupported
    # broadcasting semantics in CudaFusionGroup
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_broadcasting_partition_logic_0(self):

        def t(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
            x = x + 12.0
            o1 = x + y
            o2 = x + z
            o = o1 + o2
            return o
        t_jit = torch.jit.script(t)
        x = torch.randn(4, 8, 6, 8, dtype=torch.float32, device="cuda")
        y = torch.randn(8, 6, 8, dtype=torch.float32, device="cuda")
        z = torch.randn(6, 8, dtype=torch.float32, device="cuda")
        jit_o = t_jit(x, y, z)
        jit_o = t_jit(x, y, z)
        o = t(x, y, z)
        self.assertEqual(o, jit_o)
        subgraph = self._getSubgraphInFusion(t_jit.graph_for(x, y, z))
        self.assertGraphContainsExactly(subgraph, 'aten::add', 4, consider_subgraphs=False)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_broadcasting_partition_logic_1(self):

        def t(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
            x = x + 12.0
            o1 = x + y
            o2 = x + z
            o = o1 + o2
            return o
        t_jit = torch.jit.script(t)
        x = torch.randn(8, 6, 8, dtype=torch.float32, device="cuda")
        y = torch.randn(4, 8, 6, 8, dtype=torch.float32, device="cuda")
        z = torch.randn(4, 1, 6, 8, dtype=torch.float32, device="cuda")
        jit_o = t_jit(x, y, z)
        jit_o = t_jit(x, y, z)
        o = t(x, y, z)
        self.assertEqual(o, jit_o)
        subgraph = self._getSubgraphInFusion(t_jit.graph_for(x, y, z))
        self.assertGraphContainsExactly(subgraph, 'aten::add', 4, consider_subgraphs=False)

    @unittest.skipIf(True, "Broadcast with different output not supported yet")
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_broadcasting_multiple_output_shape(self):
        def t(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
            o = x + 12
            o1 = o + y
            o2 = o + z
            oo = o1.sum() + o2.sum()
            return oo
        t_jit = torch.jit.script(t)
        x = torch.randn(32, 32, dtype=torch.float, device="cuda")
        y = torch.randn(2, 32, 32, dtype=torch.float, device="cuda")
        z = torch.randn(4, 32, 32, dtype=torch.float, device="cuda")
        jit_o = t_jit(x, y, z)
        jit_o = t_jit(x, y, z)
        o = t(x, y, z)
        self.assertEqual(o, jit_o)
        # Currently cannot fuse this
        self.assertGraphContains(t_jit.graph_for(x, y, z), FUSION_GUARD)

    @unittest.skipIf(True, "broadcast on branches can't be resolved yet")
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_broadcasting_multiple_output(self):
        def t(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
            o = x + 12
            o1 = o + y
            o2 = o + z
            oo = o1.sum() + o2.sum()
            return oo
        t_jit = torch.jit.script(t)
        x = torch.randn(32, 32, dtype=torch.float, device="cuda")
        y = torch.randn(4, 32, 32, dtype=torch.float, device="cuda")
        z = torch.randn(4, 32, 32, dtype=torch.float, device="cuda")
        jit_o = t_jit(x, y, z)
        jit_o = t_jit(x, y, z)
        o = t(x, y, z)
        self.assertEqual(o, jit_o)
        # Currently cannot fuse this
        self.assertGraphContains(t_jit.graph_for(x, y, z), FUSION_GUARD)

    def _unary_test_helper(self, operation, dtype, random_data):
        gradient_check = (dtype == torch.float64) and random_data
        shape = self.special_values.shape
        torch.cuda.manual_seed_all(211)

        # need additional def of t for boolean ops
        def t(x: torch.Tensor, y: torch.Tensor):
            o = x * y
            o = o + 5e-3
            o = operation(o)
            return o

        y = torch.rand(shape, dtype=torch.float32, device="cuda", requires_grad=gradient_check)
        y = y.to(dtype=dtype)

        if random_data:
            x = torch.rand(shape, dtype=torch.float32, device="cuda", requires_grad=gradient_check)
            if dtype in self.int_types:
                # prefer a larger variance for integer types
                x = x * 5
            x = x.to(dtype=dtype)
        else:
            x = self.special_values.to(dtype=dtype)
        try:
            ref = t(x, y)
        except Exception:
            # same way as TE checker, if eager mode throws, ignore this test
            return
        t_jit = torch.jit.script(t)
        jit_o = t_jit(x, y)
        jit_o = t_jit(x, y)
        jit_o = t_jit(x, y)
        if gradient_check:
            if jit_o.dtype != torch.bool:
                # bool dtype has no `-`
                gradcheck(t_jit, [x, y], nondet_tol=1e-5)
        elif dtype in self.support_tensor_dtypes:
            self.assertGraphContains(t_jit.graph_for(x, y), FUSION_GUARD)
        o = t(x, y)
        self.assertEqual(o.dtype, jit_o.dtype)

        if dtype == torch.bfloat16:
            # compare with the actual ground truth for
            #  bfloat16 kernels instead of eager mode
            #  implementation, since mismatch in cast
            #  adds excessive noise.
            o = t(x.to(torch.float64), y.to(torch.float64))
            if o.dtype.is_floating_point:
                o = o.to(torch.bfloat16)
        else:
            o = t(x, y)

        self.assertTrue(self._compare("failing case {}\n{}\n{}\n{}".format(dtype, operation, x, y), o, jit_o, 1e-2))

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_unary_ops(self):
        data_types = [
            *self.int_types,
            torch.float16,
            torch.float32,
            torch.float64,
            # TODO: revert this
            # see issue https://github.com/csarofeen/pytorch/issues/1730"
            # torch.cfloat,
            # torch.cdouble,
        ]
        if TEST_BF16:
            data_types.append(torch.bfloat16)
        operations = [torch.neg,
                      torch.abs,
                      torch.log,
                      torch.log10,
                      torch.log1p,
                      torch.log2,
                      torch.lgamma,
                      torch.exp,
                      torch.expm1,
                      torch.erf,
                      torch.erfc,
                      torch.cos,
                      torch.acos,
                      torch.cosh,
                      torch.sin,
                      torch.asin,
                      torch.sinh,
                      torch.tan,
                      torch.atan,
                      torch.sqrt,
                      torch.rsqrt,
                      torch.ceil,
                      torch.floor,
                      torch.round,
                      torch.trunc,
                      torch.frac,
                      torch.reciprocal,
                      torch.isfinite,
                      torch.isinf,
                      torch.isnan,
                      torch.isneginf,
                      torch.isposinf,
                      torch.isreal,
                      torch.nn.functional.softplus,
                      torch.nn.functional.gelu,
                      torch.nn.functional.leaky_relu,
                      torch.nn.functional.silu,
                      torch.relu,
                      torch.sigmoid,
                      torch.bitwise_not,
                      torch.tan,
                      torch.tanh]
        skip_complex = {torch.rsqrt, torch.reciprocal}
        for op, dtype in itertools.product(operations, data_types):
            if dtype.is_complex and op in skip_complex:
                continue
            self._unary_test_helper(op, dtype, False)  # test special numbers
            self._unary_test_helper(op, dtype, True)  # test random data

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_category_rule(self):
        def run_tensor(x, z):
            def t(x: torch.Tensor, z: torch.Tensor):
                o = x + z
                o = torch.abs(o)
                return o
            t_jit = torch.jit.script(t)
            jit_o = t_jit(x, z)
            jit_o = t_jit(x, z)
            o = t(x, z)
            self.assertEqual(o.dtype, jit_o.dtype)
            self.assertEqual(o, jit_o)
            self.assertGraphContains(t_jit.graph_for(x, z), FUSION_GUARD)

        def run_scalar(x, z):
            def t(x: torch.Tensor, z: float):
                o = x + z
                o = torch.abs(o)
                return o
            t_jit = torch.jit.script(t)
            jit_o = t_jit(x, z)
            jit_o = t_jit(x, z)
            o = t(x, z)
            self.assertEqual(o.dtype, jit_o.dtype)
            self.assertEqual(o, jit_o)
            self.assertGraphContains(t_jit.graph_for(x, z), FUSION_GUARD)

        # n-dim with 0-dim (no type-promote)
        x = torch.randn(4, 8, 32, 32, dtype=torch.float, device="cuda")
        z = torch.tensor(2.0, dtype=torch.double, device="cuda")
        run_tensor(x, z)

        # n-dim with 0-dim (type-promote)
        x = torch.randn(4, 8, 32, 32, device="cuda").to(dtype=torch.long)
        z = torch.tensor(2.0, dtype=torch.double, device="cuda")
        run_tensor(x, z)

        # n-dim with n-dim (type-promote)
        x = torch.randn(4, 8, 32, 32, dtype=torch.float, device="cuda")
        z = torch.randn(4, 8, 32, 32, dtype=torch.double, device="cuda")
        run_tensor(x, z)

        # n-dim with scalar (no type-promote)
        x = torch.randn(4, 8, 32, 32, dtype=torch.float16, device="cuda")
        z = torch.tensor(3., dtype=torch.double)
        run_scalar(x, z)
        if TEST_BF16:
            # n-dim with scalar (no type-promote)
            x = torch.randn(4, 8, 32, 32, dtype=torch.bfloat16, device="cuda")
            z = torch.tensor(3., dtype=torch.double)
            run_scalar(x, z)

        # n-dim with scalar (type-promote)
        x = torch.randn(4, 8, 32, 32, device="cuda").to(dtype=torch.long)
        z = torch.tensor(3., dtype=torch.double)
        run_scalar(x, z)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_unary_bitwise(self):
        def bit_not(x: torch.Tensor):
            return ~(x + 1)

        jitted = torch.jit.script(bit_not)
        x = torch.randn(4, 8, 32, 32, dtype=torch.float, device="cuda").mul(5).to(torch.long)
        jit_o = jitted(x)
        jit_o = jitted(x)
        o = bit_not(x)
        self.assertEqual(o, jit_o)
        jitted.graph_for(x)  # Shows up in second instance, not first
        self.assertGraphContains(jitted.graph_for(x), FUSION_GUARD)

        def bool_not(x: torch.Tensor, y: torch.Tensor):
            return ~(x & y)

        jitted = torch.jit.script(bool_not)
        x = torch.rand(4, 8, 32, 32, dtype=torch.float, device="cuda").round().to(torch.bool)
        y = torch.rand(4, 8, 32, 32, dtype=torch.float, device="cuda").round().to(torch.bool)
        jit_o = jitted(x, y)
        jit_o = jitted(x, y)
        o = bool_not(x, y)
        self.assertEqual(o, jit_o)
        jitted.graph_for(x, y)  # Shows up in second instance, not first
        self.assertGraphContains(jitted.graph_for(x, y), FUSION_GUARD)

    def _get_scalar_binary_test_fn(self, category_and_type1, category_and_type2, operation):
        category1, dtype_arg1 = category_and_type1
        category2, dtype_arg2 = category_and_type2

        def t_intx_tensory(x: int, y: torch.Tensor):
            o = operation(x, y)
            o = 2 + o
            return o

        def t_doublex_tensory(x: float, y: torch.Tensor):
            o = operation(x, y)
            o = 2 + o
            return o

        def t_cdoublex_tensory(x: complex, y: torch.Tensor):
            o = operation(x, y)
            o = 2 + o
            return o

        # Omit both scalar cases and swap cases
        assert category1 == "scalar" and category2 != "scalar"
        if dtype_arg1.is_floating_point:
            return t_doublex_tensory
        if dtype_arg1 == torch.int64 or dtype_arg1 == torch.int32:
            return t_intx_tensory
        if dtype_arg1.is_complex or dtype_arg1 == torch.int32:
            return t_cdoublex_tensory
        raise NotImplementedError

    def _binary_test_helper(self, operation, dtypes, random_data, categories="ndim"):
        if isinstance(dtypes, tuple):
            dtype_arg1, dtype_arg2 = dtypes
        else:
            dtype_arg1 = dtype_arg2 = dtypes

        if isinstance(categories, tuple) and random_data:
            category1, category2 = categories
        elif not random_data:
            category1 = category2 = "ndim"
        else:
            category1 = category2 = categories

        def is_cpu_category(x):
            return x == "0dimcpu" or x == "scalar"

        # skip unsupported cases
        if is_cpu_category(category1) and is_cpu_category(category2):
            return

        # only test cases with first operand as scalar
        if category2 == "scalar":
            return

        # skip ops that doesn't support scalar inputs in eager
        if operation in [
            torch.atan2,
            torch.max,
            torch.min,
            torch.remainder,  # unsupported in nvfuser
        ]:
            if category1 == "scalar" or category2 == "scalar":
                return

        if operation in [
            torch.fmod,
            torch.eq,
            torch.ne,
            torch.ge,
            torch.gt,
            torch.le,
            torch.lt
        ]:
            if category1 == "scalar":
                return

        # operators that does not support bfloat16
        if operation in [torch.fmod]:
            if dtype_arg1 == torch.bfloat16 or dtype_arg2 == torch.bfloat16:
                return

        def t(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
            o = operation(x, y)
            o = o + z
            return o

        shape = (4, 32, 32)

        shapex = shape if category1 == "ndim" else ()
        shapey = shape if category2 == "ndim" else ()

        if random_data:
            x = (torch.randn(shapex, dtype=torch.float, device="cuda") * 5).to(dtype_arg1)
            y = (torch.randn(shapey, dtype=torch.float, device="cuda") * 5).to(dtype_arg2)
        else:
            x = self.special_values.to(dtype=dtype_arg1)
            y = (torch.rand_like(self.special_values) * 5).to(dtype_arg2)

        r"""
            Category conversion
        """
        has_scalar = False
        if category1 == "scalar":
            has_scalar = True
            x = x.item()

        if category1 == "0dimcpu":
            x = x.to(device="cpu")

        if category2 == "scalar":
            has_scalar = True
            y = y.item()

        if category2 == "0dimcpu":
            y = y.to(device="cpu")

        z = torch.tensor([2], device="cuda").to(dtype_arg1)
        is_dtype_arg1_int = dtype_arg1 == torch.int32 or dtype_arg1 == torch.int64
        is_dtype_arg2_int = dtype_arg2 == torch.int32 or dtype_arg2 == torch.int64

        if operation in [torch.pow]:
            if is_dtype_arg1_int and is_dtype_arg2_int:
                if category2 == "scalar":
                    # RuntimeError: Integers to negative integer powers are not allowed
                    y = abs(y)
                if category2 == "0dimcpu" and y == -1:
                    # https://github.com/pytorch/pytorch/issues/73196
                    y = y - 1
                if category2 == "0dimcpu" and y == -2:
                    # avoid pow(0, -2), which gives inconsistent results on integer tensor
                    y = y - 1

        # Avoid division by zero for integer tensors
        div_like = [torch.div, torch.fmod, torch.remainder]
        if operation in div_like and (dtype_arg2 == torch.int32 or dtype_arg2 == torch.int64):
            y[y == 0] = 1

        test_value = True
        if dtype_arg1 == torch.half or dtype_arg2 == torch.half:
            test_value = False
        if dtype_arg1 == torch.bfloat16 or dtype_arg2 == torch.bfloat16:
            test_value = False

        try:
            if not has_scalar:
                o = t(x, y, z)
                t_jit = torch.jit.script(t)
                jit_o = t_jit(x, y, z)
                jit_o = t_jit(x, y, z)
                jit_o = t_jit(x, y, z)

                self.assertEqual(o.dtype, jit_o.dtype)
                if test_value:
                    self.assertEqual(o, jit_o)
                self.assertGraphContains(t_jit.graph_for(x, y, z), FUSION_GUARD)

            elif category2 != "scalar":  # only test the case where first is scalar
                test_fn = self._get_scalar_binary_test_fn((category1, dtype_arg1), (category2, dtype_arg2), operation)
                o = test_fn(x, y)
                t_jit = torch.jit.script(test_fn)
                jit_o = t_jit(x, y)
                jit_o = t_jit(x, y)
                jit_o = t_jit(x, y)

                self.assertEqual(o.dtype, jit_o.dtype)
                if test_value:
                    self.assertEqual(o, jit_o)
                self.assertGraphContains(t_jit.graph_for(x, y), FUSION_GUARD)
        except Exception as e:
            print("failing test for op: ", operation.__name__)
            print("with input\n\tx: ", x)
            print("\ty: ", y)
            print("\tz: ", z)
            raise e

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_binary_ops(self):
        data_types = [
            torch.int32,
            torch.int64,
            torch.float16,
            torch.float32,
            torch.float64,
        ]
        if TEST_BF16:
            data_types.append(torch.bfloat16)
        operations = [torch.mul,
                      torch.div,
                      torch.atan2,
                      torch.max,
                      torch.min,
                      torch.pow,
                      torch.remainder,
                      torch.fmod,
                      torch.eq,
                      torch.ne,
                      torch.ge,
                      torch.gt,
                      torch.le,
                      torch.lt]

        category_types = [
            "scalar",
            "0dim",
            "0dimcpu",
            "ndim"
        ]

        binary_dtype_combinations = list(itertools.combinations(data_types, 2))
        category_combinations = list(itertools.combinations(category_types, 2))

        for op, dtypes, categories in itertools.product(operations, binary_dtype_combinations, category_combinations):
            self._binary_test_helper(op, dtypes, True, categories)  # random data

        for op, dtypes in itertools.product(operations, binary_dtype_combinations):
            self._binary_test_helper(op, dtypes, False)  # special numbers

    # TODO: revert this
    @unittest.skipIf(True, "see issue https://github.com/csarofeen/pytorch/issues/1730")
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_binary_ops_complex(self):
        data_types = [torch.cfloat, torch.cdouble]
        operations = [torch.mul, torch.div, torch.pow, torch.eq, torch.ne]

        category_types = [
            "scalar",
            "0dim",
            "0dimcpu",
            "ndim"
        ]

        binary_dtype_combinations = list(itertools.combinations(data_types, 2))
        category_combinations = list(itertools.combinations(category_types, 2))

        for op, dtypes, categories in itertools.product(operations, binary_dtype_combinations, category_combinations):
            self._binary_test_helper(op, dtypes, True, categories)  # random data

        for op, dtypes in itertools.product(operations, binary_dtype_combinations):
            self._binary_test_helper(op, dtypes, False)  # special numbers

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_binary_bitwise(self):
        dtypes = [torch.bool, torch.int32, torch.int64]

        for dtype1, dtype2, dtype3 in itertools.product(dtypes, repeat=3):
            def jit_and(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
                return torch.bitwise_and(x, y) & z

            def jit_or(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
                return torch.bitwise_or(x, y) | z

            def jit_xor(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
                return torch.bitwise_xor(x, y) ^ z

            def jit_lshift(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
                return torch.bitwise_left_shift(x, y) << z

            def jit_rshift(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
                return torch.bitwise_right_shift(x, y) >> z

            for jit_func in [jit_and, jit_or, jit_xor, jit_lshift, jit_rshift]:
                if torch.bool in {dtype1, dtype2, dtype3} and jit_func in {jit_lshift, jit_rshift}:
                    continue
                x = torch.randn(4, 8, 32, 32, dtype=torch.float, device="cuda").mul(5).to(dtype1)
                y = torch.randn(4, 8, 32, 32, dtype=torch.float, device="cuda").mul(5).to(dtype2)
                z = torch.randn(4, 8, 32, 32, dtype=torch.float, device="cuda").mul(2).to(dtype3)

                jitted = torch.jit.script(jit_func)
                jit_o = jitted(x, y, z)
                jit_o = jitted(x, y, z)
                o = jit_func(x, y, z)
                self.assertEqual(o, jit_o)
                self.assertGraphContains(jitted.graph_for(x, y, z), FUSION_GUARD)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_type_as_op(self):
        def t(x: torch.Tensor, y: torch.Tensor, z: float):
            o = torch.lt(x, z)
            o = o.type_as(y)
            return o
        t_jit = torch.jit.script(t)
        x = torch.randn(4, 8, 32, 32, dtype=torch.float, device="cuda")
        y = torch.randn(4, 8, 32, 32, dtype=torch.float, device="cuda")
        jit_o = t_jit(x, y, 0.5)
        jit_o = t_jit(x, y, 0.5)
        o = t(x, y, 0.5)
        self.assertEqual(o, jit_o)
        self.assertGraphContains(t_jit.graph_for(x, y, 0.5), FUSION_GUARD)

    def _ternary_integer_test_helper(self, dtype_arg1):
        shape = (4, 8, 32, 32)
        magnitude = 100
        if (dtype_arg1 in self.int_types):
            x = torch.randint(-magnitude, magnitude, shape, dtype=dtype_arg1, device="cuda")
        else:
            x = torch.randn(shape, dtype=dtype_arg1, device="cuda") * magnitude
        arg2 = int(0)
        arg3 = int(magnitude * 0.1)

        def clamp0(x: torch.Tensor, f: int):
            o = 2. * torch.clamp(x, min=f)
            return o
        clamp0_jit = torch.jit.script(clamp0)
        self._run_helper(clamp0_jit, clamp0, x, arg2)

        def clamp1(x: torch.Tensor, f: int, ff: int):
            o = 2. * torch.clamp(x, min=f, max=ff)
            return o
        clamp1_jit = torch.jit.script(clamp1)
        self._run_helper(clamp1_jit, clamp1, x, arg2, arg3)

        def clamp2(x: torch.Tensor, f: float, ff: int):
            o = 2. * torch.clamp(x, min=f, max=ff)
            return o
        clamp2_jit = torch.jit.script(clamp2)
        self._run_helper(clamp2_jit, clamp2, x, float(arg2), arg3)

        def clamp3(x: torch.Tensor, f: int, ff: float):
            o = 2. * torch.clamp(x, min=f, max=ff)
            return o
        clamp3_jit = torch.jit.script(clamp3)
        self._run_helper(clamp3_jit, clamp3, x, arg2, float(arg3))

        def threshold(x: torch.Tensor, th: int, val: int):
            o = 2. * torch.threshold(x, th, val)
            return o
        threshold_jit = torch.jit.script(threshold)
        self._run_helper(threshold_jit, threshold, x, arg2, arg3)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_ternary_ops_integer_compatibility(self):
        data_types = [
            torch.float16,
            torch.float32,
            torch.float64
        ]
        for dtype in data_types:
            self._ternary_integer_test_helper(dtype)

    def _ternary_test_helper(self, operation, dtypes, random_data):
        if isinstance(dtypes, tuple):
            dtype_arg1, dtype_arg2, dtype_arg3 = dtypes
        else:
            dtype_arg1 = dtype_arg2 = dtype_arg3 = dtypes

        def t(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, alpha: torch.Tensor):
            o = operation(x, y, z)
            o = o + alpha
            return o

        shape = (4, 32, 32)
        if operation is torch.where:
            dtype_arg1 = torch.bool
            if random_data:
                x = torch.randint(0, 2, shape).to(dtype=torch.bool, device="cuda")
                y = (torch.randn(shape, dtype=torch.float, device="cuda") * 5).to(dtype_arg2)
                z = (torch.randn(shape, dtype=torch.float, device="cuda") * 5).to(dtype_arg3)
            else:
                x = torch.randint(0, 2, self.special_values.size()).to(dtype=torch.bool, device="cuda")
                y = self.special_values.to(dtype=dtype_arg2)
                z = (torch.rand_like(self.special_values) * 5).to(dtype_arg3)
        elif random_data:
            x = (torch.randn(shape, dtype=torch.float, device="cuda") * 5).to(dtype_arg1)
            y = (torch.randn(shape, dtype=torch.float, device="cuda") * 5).to(dtype_arg2)
            z = (torch.randn(shape, dtype=torch.float, device="cuda") * 5).to(dtype_arg3)
        else:
            x = self.special_values.to(dtype=dtype_arg1)
            y = (torch.rand_like(self.special_values) * 5).to(dtype_arg2)
            z = (torch.rand_like(self.special_values) * 5).to(dtype_arg3)
        alpha = torch.tensor([2], device="cuda").to(dtype_arg1)

        o = t(x, y, z, alpha)
        t_jit = torch.jit.script(t)
        jit_o = t_jit(x, y, z, alpha)
        jit_o = t_jit(x, y, z, alpha)

        self.assertEqual(o.dtype, jit_o.dtype)
        self.assertEqual(o, jit_o)
        self.assertGraphContains(t_jit.graph_for(x, y, z), FUSION_GUARD)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_ternary_ops_type_promotion(self):
        # TODO: update accuracy tolerance for bf16 / fp16 data types
        data_types = [
            # torch.float16,
            torch.float32,
            torch.float64
        ]
        '''
        if TEST_BF16:
            data_types.append(torch.bfloat16)
        '''
        # TODO: Add Tensor support for clamp
        operations = [torch.clamp]
        ternary_dtype_combinations = itertools.combinations(data_types, 3)
        for op, dtypes in itertools.product(operations, ternary_dtype_combinations):
            self._ternary_test_helper(op, dtypes, True)  # random data
            self._ternary_test_helper(op, dtypes, False)  # special numbers

    # We can't test the scalar version of rsub from python
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING, "Requires fusion optimization pass to be effective")
    def test_rsub(self):
        x = torch.randn(4, 8, 32, 32, dtype=torch.float, device="cuda")
        y = torch.randn(4, 8, 32, 32, dtype=torch.float, device="cuda")

        def rsub(x: torch.Tensor, y: torch.Tensor):
            o = torch.rsub(x, y)
            o = o * 2.
            return o

        rsub_jit = torch.jit.script(rsub)
        self._run_helper(rsub_jit, rsub, x, y)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    # legacy fuser does not work for rand_like, see issue #34361
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING, "Requires fusion optimization pass to be effective")
    def test_ternary_ops(self):
        x = torch.randn(4, 8, 32, 32, dtype=torch.float, device="cuda")
        y = torch.randn(4, 8, 32, 32, dtype=torch.float, device="cuda")
        z = torch.randn(4, 8, 32, 32, dtype=torch.float, device="cuda")
        cond = torch.randint(0, 2, (4, 8, 32, 32)).to(dtype=torch.bool, device="cuda")

        def add(x: torch.Tensor, other: torch.Tensor, alpha: float):
            o = torch.relu(x)
            o = torch.add(o, other=other, alpha=alpha)
            return o
        add_jit = torch.jit.script(add)
        self._run_helper(add_jit, add, x, y, 2.0)

        def clamp0(x: torch.Tensor, f: float):
            o = 2. * torch.clamp(x, min=f)
            return o
        clamp0_jit = torch.jit.script(clamp0)
        self._run_helper(clamp0_jit, clamp0, x, 0.5)

        def clamp1(x: torch.Tensor, f: float, ff: float):
            o = 2. * torch.clamp(x, min=f, max=ff)
            return o
        clamp1_jit = torch.jit.script(clamp1)
        self._run_helper(clamp1_jit, clamp1, x, -0.2, 0.7)

        def threshold(x: torch.Tensor, th: float, val: float):
            o = 2. * torch.threshold(x, th, val)
            return o
        threshold_jit = torch.jit.script(threshold)
        self._run_helper(threshold_jit, threshold, x, 0.2, 0.9)

        def where(x: torch.Tensor, y: torch.Tensor, cond: torch.Tensor):
            o = 2. * torch.where(cond, x, y)
            return o
        where_jit = torch.jit.script(where)
        self._run_helper(where_jit, where, x, y, cond)

        def lerp(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
            o = 2. * torch.lerp(x, y, z)
            return o
        lerp_jit = torch.jit.script(lerp)
        self._run_helper(lerp_jit, lerp, x, y, z)

        def lerp_scale(x: torch.Tensor, y: torch.Tensor, z: float):
            o = 2. * torch.lerp(x, y, z)
            return o
        lerp_scale_jit = torch.jit.script(lerp_scale)
        self._run_helper(lerp_scale_jit, lerp_scale, x, y, 0.5)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING, "Requires profiling node to run cuda fuser")
    def test_addcmul_ops(self):
        x = torch.randn(4, 8, 32, 32, dtype=torch.float, device="cuda")
        y = torch.randn(4, 8, 32, 32, dtype=torch.float, device="cuda")
        z = torch.randn(4, 8, 32, 32, dtype=torch.float, device="cuda")

        def addcmul(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, value: float):
            o = torch.add(x, 0.5)
            o = torch.addcmul(o, y, z, value=value)
            return o
        addcmul_jit = torch.jit.script(addcmul)
        self._run_helper(addcmul_jit, addcmul, x, y, z, 2.0)

        def addcmul_no_alpha(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
            o = torch.add(x, 0.5)
            o = torch.addcmul(o, y, z)
            return o
        addcmul_no_alpha_jit = torch.jit.script(addcmul_no_alpha)
        self._run_helper(addcmul_no_alpha_jit, addcmul_no_alpha, x, y, z)

        def addcmul_const_alpha(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
            o = torch.add(x, 0.5)
            o = torch.addcmul(o, y, z, value=0.75)
            return o
        addcmul_const_alpha_jit = torch.jit.script(addcmul_const_alpha)
        self._run_helper(addcmul_const_alpha_jit, addcmul_const_alpha, x, y, z)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_dynamic_size(self):
        old_guard = torch._C._jit_set_nvfuser_guard_mode(True)
        torch._C._jit_set_bailout_depth(20)

        def t(x: torch.Tensor, y: torch.Tensor, z: float):
            o = x + y
            o = o + z
            return o
        t_jit = torch.jit.script(t)
        x = torch.randn(4, 8, 32, 32, dtype=torch.float, device="cuda")
        y = torch.randn(32, 32, dtype=torch.float, device="cuda")
        jit_o = t_jit(x, y, 2.0)
        jit_o = t_jit(x, y, 2.0)
        o = t(x, y, 2.0)
        self.assertEqual(o, jit_o)
        subgraph = self._getSubgraphInFusion(t_jit.graph_for(x, y, 2.0))
        self.assertGraphContainsExactly(subgraph, 'aten::add', 2, consider_subgraphs=False)

        # this test is not ideal, as we rely on the bailout to test it and we
        # don't know a way to verify the bailout graph to validate the proper
        # fusion.
        x = torch.randn(8, 32, 16, 8, dtype=torch.float, device="cuda")
        y = torch.randn(16, 8, dtype=torch.float, device="cuda")
        jit_o = t_jit(x, y, 2.0)
        jit_o = t_jit(x, y, 2.0)
        o = t(x, y, 2.0)
        self.assertEqual(o, jit_o)
        self.assertGraphContains(t_jit.graph_for(x, y, 2.0), FUSION_GUARD)
        x = torch.randn(8, 17, 8, dtype=torch.float, device="cuda")
        y = torch.randn(8, 17, 1, dtype=torch.float, device="cuda")
        jit_o = t_jit(x, y, 2.0)
        jit_o = t_jit(x, y, 2.0)
        o = t(x, y, 2.0)
        self.assertEqual(o, jit_o)
        self.assertGraphContains(t_jit.graph_for(x, y, 2.0), FUSION_GUARD)
        torch._C._jit_set_nvfuser_guard_mode(old_guard)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_random_topo(self):
        os.environ["PYTORCH_NVFUSER_DISABLE_FALLBACK"] = "1"
        self.assertTrue(runDefaultTestWithSeed(28449))

    def _compare(self, desc, inp1, inp2, error):
        a = inp1.clone()
        b = inp2.clone()
        close = torch.allclose(a, b, rtol=error, atol=error, equal_nan=True)
        if not close:
            print(desc, close)
            z = a - b
            index = (torch.abs(z) >= error + error * torch.abs(b)).nonzero()
            print("dif    : ", z[index])
            print("inp1   : ", a[index])
            print("inp2   : ", b[index])
            print("maximum difference", z[index].max())
        return close

    # Permutation helper that applies binary operation between two tensors:
    #   1. applies separate permutation `perm0` & `perm1` to two inputs
    #   2. reduce dimension `broadcast_axis` of operand two to size 1
    # The purpose of this test is to ensure permutation works well in
    # complicated cases with arbitrary stride order and broadcasting dimensions
    def _permutation_helper(self, sizes, broadcast_axis, dtype, device, perm0, perm1):
        def t(x: torch.Tensor, y: torch.Tensor):
            o = torch.add(x, y)
            o = torch.relu(o)
            return o

        x = torch.randn([sizes[i] for i in perm0], dtype=dtype, device=device).permute(
            [perm0.index(i) for i in range(len(sizes))])
        if broadcast_axis >= 0:
            sizes[broadcast_axis] = 1
        y = torch.randn([sizes[i] for i in perm1], dtype=dtype, device=device).permute(
            [perm1.index(i) for i in range(len(sizes))])
        t_jit = torch.jit.script(t)
        jit_o = t_jit(x, y)
        jit_o = t_jit(x, y)
        o = t(x, y)
        self.assertEqual(o.dtype, jit_o.dtype)
        self.assertEqual(o, jit_o)
        self.assertEqual(o.stride(), jit_o.stride())
        self.assertGraphContains(t_jit.graph_for(x, y), FUSION_GUARD)

    # end-2-end test of permutation & contiguity handling in integration.
    # we are testing inputs with all combination of permutation order, just to
    # ensure that integration would be able to generate functionally correct
    # kernels
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_binary_ops_permutation(self):
        # note that num_dim is exclusive from len(x), so we are not reducing
        # to single element (codegen limitation at this moment)
        x = [7, 8, 12]
        b_axes = range(-1, len(x))
        for b_axis in b_axes:
            for perm0 in itertools.permutations(range(len(x))):
                for perm1 in itertools.permutations(range(len(x))):
                    x = [7, 8, 12]
                    self._permutation_helper(x, b_axis, torch.float32, "cuda", perm0, perm1)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_binary_ops_channels_last_with_bcast(self):
        device = "cuda"
        x = torch.randn([4, 3, 2, 5], device=device).to(memory_format=torch.channels_last)
        w = torch.randn([2, 5], device=device)

        def t(x: torch.Tensor, b: torch.Tensor):
            o = x + b
            return torch.relu(o)
        t_jit = torch.jit.script(t)
        jit_o = t_jit(x, w)
        jit_o = t_jit(x, w)
        jit_o = t_jit(x, w)
        o = t(x, w)
        self.assertEqual(o.dtype, jit_o.dtype)
        self.assertTrue(self._compare("comparing output failed", o, jit_o, 1e-4))
        self.assertGraphContains(t_jit.graph_for(x, w), FUSION_GUARD)

    def _reduction_helper(self, sizes, reduction_axis, dtype, device, perm0, perm1, keepdim=False):
        class MyReduction(torch.nn.Module):
            __constants__ = ['reduction_axis', 'keepdim']

            def __init__(self):
                super(MyReduction, self).__init__()
                self.reduction_axis = reduction_axis
                self.keepdim = keepdim

            def forward(self, x: torch.Tensor, y: torch.Tensor):
                o = torch.add(x, y)
                o = torch.sum(o, dim=self.reduction_axis, keepdim=self.keepdim)
                return o

        t = MyReduction()

        x = torch.randn([sizes[i] for i in perm0], dtype=dtype, device=device).permute(
            [perm0.index(i) for i in range(len(sizes))])
        y = torch.randn([sizes[i] for i in perm1], dtype=dtype, device=device).permute(
            [perm1.index(i) for i in range(len(sizes))])
        t_jit = torch.jit.script(t)
        jit_o = t_jit(x, y)
        jit_o = t_jit(x, y)
        o = t(x, y)
        self.assertEqual(o.dtype, jit_o.dtype)
        # numerical issues here due to our scheduling.
        # can't use `self.assertEqual(o, jit_o)`
        self.assertTrue(self._compare("comparing output failed", o, jit_o, 1e-4))
        self.assertGraphContains(t_jit.graph_for(x, y), FUSION_GUARD)

    @unittest.skipIf(is_pre_volta(), "reduction not supported in pre volta device")
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_reduction(self):
        for x in ([7, 8, 12], [12, 8, 7, 9, 15], [128, 16, 8, 32]):
            # note that num_dim is exclusive from len(x), so we are not reducing
            # to single element (codegen limitation at this moment)
            for num_reduce_dim in range(1, len(x)):
                for axes in itertools.combinations(range(len(x)), num_reduce_dim):
                    for keepdim in (True, False):
                        perm0 = range(len(x))
                        perm1 = range(len(x))
                        self._reduction_helper(x, axes, torch.float32, "cuda", perm0, perm1, keepdim)

    def _layer_norm_autodiff_helper(self, model, grad, shapes, args):
        jit_model = torch.jit.script(model)

        eps = np.random.random() * 1e-4
        use_cudnn = bool(np.random.randint(0, 2))

        # profile/optimization runs
        for i in range(3):
            jit_o = jit_model(shapes, *args, eps, use_cudnn)
            jit_o.backward(grad)

        ref_args = [t.detach().clone().requires_grad_() for t in args]
        [t.grad.zero_() for t in args]
        jit_o = jit_model(shapes, *args, eps, use_cudnn)
        jit_o.backward(grad)

        o = model(shapes, *ref_args, eps, use_cudnn)
        o.backward(grad)
        self.assertEqual(jit_o, o)
        for arg, ref_arg in zip(args, ref_args):
            self.assertEqual(arg.grad, ref_arg.grad)

        # check fusion in fw & bw
        g = jit_model.graph_for(shapes, *args, eps, use_cudnn)
        for node in g.nodes():
            n = node
        dbg_state = jit_model.get_debug_state()
        for val in dbg_state.execution_plans.values():
            v = val
        state2 = v.code.grad_executor_states()
        for val in state2[0].execution_plans.values():
            v2 = val
        FileCheck().check(FUSION_GUARD).run(g)
        FileCheck().check(FUSION_GUARD).run(v2.graph)

    @unittest.skipIf(is_pre_volta(), "reduction not supported in pre volta device")
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_layer_norm_autodiff(self):
        def t_wb(shapes: List[int], x, w, b, eps: float, cudnn: bool):
            o = torch.layer_norm(x, shapes, w, b, eps, cudnn)
            o = torch.relu(o)
            return o

        def t_w(shapes: List[int], x, w, eps: float, cudnn: bool):
            o = torch.layer_norm(x, shapes, w, None, eps, cudnn)
            o = torch.relu(o)
            return o

        def t_b(shapes: List[int], x, b, eps: float, cudnn: bool):
            o = torch.layer_norm(x, shapes, None, b, eps, cudnn)
            o = torch.relu(o)
            return o

        def t(shapes: List[int], x, eps: float, cudnn: bool):
            o = torch.layer_norm(x, shapes, None, None, eps, cudnn)
            o = torch.relu(o)
            return o

        model = {3: t_wb, 2: t_w, 1: t_b, 0: t}

        for w, b in itertools.product([True, False], repeat=2):
            batch = [2]
            # note: awkward shape here to avoid vectorized fast kernel, which is
            # buggy in aten
            shapes = [2, 7, 3]
            m = model[w * 2 + b]

            grad = torch.randn(batch + shapes, dtype=torch.float32, device="cuda")
            args = [torch.randn(batch + shapes, dtype=torch.float32, device="cuda").requires_grad_()]
            if w:
                args.append(torch.randn(shapes, dtype=torch.float32, device="cuda").requires_grad_())
            if b:
                args.append(torch.randn(shapes, dtype=torch.float32, device="cuda").requires_grad_())
            self._layer_norm_autodiff_helper(m, grad, shapes, args)

    @unittest.skipIf(is_pre_volta(), "reduction not supported in pre volta device")
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_layer_norm_parser(self):
        dtype = torch.float32
        device = "cuda"
        x = torch.randn([4, 4, 2], dtype=dtype, device=device)
        w = torch.randn([4, 2], dtype=dtype, device=device)
        b = torch.randn([4, 2], dtype=dtype, device=device)

        def t(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor):
            o = torch.relu(x)
            o = torch.layer_norm(o, [4, 2], w, b, 1e-5)
            return o

        o = t(x, w, b)
        t_jit = torch.jit.script(t)
        jit_o = t_jit(x, w, b)
        jit_o = t_jit(x, w, b)
        o = t(x, w, b)
        self.assertGraphContains(t_jit.graph_for(x, w, b), FUSION_GUARD)

    def _native_layer_norm_helper(self, shape, norm_shape, dtype, device, error, affine=True):
        class MyLayerNorm(torch.nn.Module):
            __constants__ = ['norm_shape']

            def __init__(self, elementwise_affine=True):
                super(MyLayerNorm, self).__init__()
                self.norm_shape = norm_shape
                if elementwise_affine:
                    self.weight = torch.randn(norm_shape, dtype=dtype, device=device)
                    self.bias = torch.randn(norm_shape, dtype=dtype, device=device)
                    with torch.no_grad():
                        self.weight.fill_(1)
                        self.bias.fill_(0)
                else:
                    self.weight = None
                    self.bias = None

            def forward(self, x: torch.Tensor):
                o = torch.relu(x)
                o = torch.native_layer_norm(o, self.norm_shape, self.weight, self.bias, 1e-5)
                return o

        t = MyLayerNorm(affine)

        x = torch.randn(shape, dtype=dtype, device=device)
        t_jit = torch.jit.script(t)
        jit_o, jit_mean, jit_rstd = t_jit(x)
        jit_o, jit_mean, jit_rstd = t_jit(x)
        o, mean, rstd = t(x)
        self.assertEqual(o.dtype, jit_o.dtype)
        # numerical issues here due to our scheduling.
        # can't use `self.assertEqual(o, jit_o)`
        self.assertTrue(self._compare("comparing output failed", o, jit_o, error))
        self.assertTrue(self._compare("comparing mean failed", mean, jit_mean, error))
        self.assertTrue(self._compare("comparing rstd failed", rstd, jit_rstd, error))
        self.assertGraphContains(t_jit.graph_for(x), FUSION_GUARD)

    @unittest.skipIf(is_pre_volta(), "reduction not supported in pre volta device")
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_native_layer_norm(self):
        dims = 4
        rnds = 3
        for idx in range(rnds):
            for offset in range(1, dims):
                for affine in (True, False):
                    input_shape = [random.randint(10, 30) for idx in range(dims)]
                    norm_shape = [input_shape[idx] for idx in range(dims - offset, dims)]
                    self._native_layer_norm_helper(input_shape, norm_shape, torch.float32, "cuda", 1e-4, affine)

    @unittest.skipIf(is_pre_volta(), "reduction not supported in pre volta device")
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_native_layer_norm_half(self):
        dims = 4
        rnds = 3
        for idx in range(rnds):
            for offset in range(1, dims):
                input_shape = [random.randint(10, 30) for idx in range(dims)]
                norm_shape = [input_shape[idx] for idx in range(dims - offset, dims)]
                self._native_layer_norm_helper(input_shape, norm_shape, torch.float16, "cuda", 5e-3)

    @unittest.skipIf(is_pre_volta(), "reduction not supported in pre volta device")
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    @unittest.skipIf(not TEST_BF16, "device does not support BFloat16")
    def test_native_layer_norm_bfloat(self):
        dims = 4
        rnds = 3
        for idx in range(rnds):
            for offset in range(1, dims):
                input_shape = [random.randint(10, 30) for idx in range(dims)]
                norm_shape = [input_shape[idx] for idx in range(dims - offset, dims)]
                self._native_layer_norm_helper(input_shape, norm_shape, torch.bfloat16, "cuda", 1e-1)

    def _norm_helper(self,
                     shape,
                     dtype,
                     device,
                     error,
                     is_batch_norm_else_instance_norm,
                     memory_format=torch.contiguous_format,
                     *,
                     layer_dtype=torch.float32):
        class MyBatchNorm(torch.nn.Module):
            def __init__(self):
                super(MyBatchNorm, self).__init__()

            def forward(self, x: torch.Tensor, r_mean: torch.Tensor, r_var: torch.Tensor):
                o = torch.nn.functional.batch_norm(x, r_mean, r_var, training=True)
                o = torch.relu(o)
                return o

        class MyInstanceNorm(torch.nn.Module):
            def __init__(self):
                super(MyInstanceNorm, self).__init__()

            def forward(self, x: torch.Tensor, r_mean: torch.Tensor, r_var: torch.Tensor):
                o = torch.nn.functional.instance_norm(x, r_mean, r_var, use_input_stats=True)
                o = torch.relu(o)
                return o

        t = MyBatchNorm() if is_batch_norm_else_instance_norm else MyInstanceNorm()

        x = torch.randn(shape, dtype=dtype, device=device).to(memory_format=memory_format)
        running_mean = torch.zeros(shape[1], dtype=layer_dtype, device=device)
        running_var = torch.ones(shape[1], dtype=layer_dtype, device=device)
        t_jit = torch.jit.script(t)

        eager_running_mean = running_mean.clone()
        eager_running_var = running_var.clone()
        jit_running_mean = running_mean.clone()
        jit_running_var = running_var.clone()

        jit_o = t_jit(x, running_mean.clone(), running_var.clone())

        self.assertTrue(self._compare("prerun comparing running_mean failed", eager_running_mean, jit_running_mean, error))
        self.assertTrue(self._compare("prerun comparing running_var failed", eager_running_var, jit_running_var, error))

        jit_o = t_jit(x, jit_running_mean, jit_running_var)
        o = t(x, eager_running_mean, eager_running_var)
        self.assertEqual(o.dtype, jit_o.dtype)
        self.assertEqual(o.stride(), jit_o.stride())
        # numerical issues here due to our scheduling.
        # can't use `self.assertEqual(o, jit_o)`
        self.assertTrue(self._compare("comparing output failed", o, jit_o, error))
        self.assertTrue(self._compare("comparing running_mean failed", eager_running_mean, jit_running_mean, error))
        self.assertTrue(self._compare("comparing running_var failed", eager_running_var, jit_running_var, error))
        self.assertGraphContains(t_jit.graph_for(x, running_mean, running_var), FUSION_GUARD)

    @unittest.skipIf(is_pre_volta(), "reduction not supported in pre volta device")
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_layer_norm_trivial_reduce_dim(self):
        def t_wb(shapes: List[int], x, w, b, eps: float, cudnn: bool):
            o = torch.layer_norm(x, shapes, w, b, eps, cudnn)
            o = torch.relu(o)
            return o

        batch = [1]
        shapes = [2, 7, 3]

        grad = torch.randn(batch + shapes, dtype=torch.float32, device="cuda")
        args = [torch.randn(batch + shapes, dtype=torch.float32, device="cuda").requires_grad_()]
        args.append(torch.randn(shapes, dtype=torch.float32, device="cuda").requires_grad_())
        args.append(torch.randn(shapes, dtype=torch.float32, device="cuda").requires_grad_())
        self._layer_norm_autodiff_helper(t_wb, grad, shapes, args)

    @unittest.skipIf(is_pre_volta(), "reduction not supported in pre volta device")
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_norm_half_layer(self):
        size = [2, 4, 2, 2]

        for is_batch_norm_else_instance_norm in [False, True]:
            for mf in [torch.channels_last, torch.contiguous_format]:
                self._norm_helper(size, torch.float16, "cuda", 1e-3, is_batch_norm_else_instance_norm,
                                  memory_format=mf, layer_dtype=torch.float16)

    @unittest.skipIf(is_pre_volta(), "reduction not supported in pre volta device")
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_norm_channels_last(self):
        size = [3, 4, 5, 6]

        with torch.backends.cudnn.flags(enabled=False):
            for is_batch_norm_else_instance_norm in [False, True]:
                for mf in [torch.channels_last, torch.contiguous_format]:
                    self._norm_helper(size, torch.float32, "cuda", 1e-4, is_batch_norm_else_instance_norm, memory_format=mf)

    @unittest.skipIf(is_pre_volta(), "reduction not supported in pre volta device")
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_norm(self):
        output_elements = 10000
        channel_sizes = [67, 457, 1024, 4096]

        with torch.backends.cudnn.flags(enabled=False):
            for is_batch_norm_else_instance_norm in [False, True]:
                for dims in range(3, 6):
                    output_size = int(pow(output_elements, 1. / (dims - 1)))
                    for C in channel_sizes:
                        x = [output_size for idx in range(dims)]
                        x[1] = C
                        self._norm_helper(x, torch.float32, "cuda", 1e-4, is_batch_norm_else_instance_norm)

    @skipIfRocm
    @unittest.skipIf(is_pre_volta(), "reduction not supported in pre volta device")
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_norm_large(self):
        output_elements = 262144
        channel_sizes = 67, 457, 1024

        for is_batch_norm_else_instance_norm in [True, False]:
            for dims in range(3, 6):
                output_size = int(pow(output_elements, 1. / (dims - 1)))
                for C in channel_sizes:
                    x = [output_size for idx in range(dims)]
                    x[1] = C
                    self._norm_helper(x, torch.float32, "cuda", 1e-4, is_batch_norm_else_instance_norm)

    @unittest.skipIf(is_pre_volta(), "reduction not supported in pre volta device")
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_norm_half(self):
        output_elements = 10000
        channel_sizes = [67, 457, 1024, 4096]

        with torch.backends.cudnn.flags(enabled=False):
            # TODO instance norm on ROCm was giving ~50% incorrect results
            for is_batch_norm_else_instance_norm in [True] if TEST_WITH_ROCM else [False, True]:
                for dims in range(3, 6):
                    output_size = int(pow(output_elements, 1. / (dims - 1)))
                    for C in channel_sizes:
                        x = [output_size for idx in range(dims)]
                        x[1] = C
                        self._norm_helper(x, torch.float16, "cuda", 5e-3, is_batch_norm_else_instance_norm)

    @unittest.skipIf(is_pre_volta(), "reduction not supported in pre volta device")
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    @unittest.skipIf(not TEST_BF16, "device does not support BFloat16")
    def test_norm_bfloat(self):
        output_elements = 10000
        channel_sizes = [67, 457, 1024, 4096]

        with torch.backends.cudnn.flags(enabled=False):
            # TODO instance norm on ROCm was giving ~50% incorrect results
            for is_batch_norm_else_instance_norm in [True] if TEST_WITH_ROCM else [False, True]:
                for dims in range(3, 6):
                    output_size = int(pow(output_elements, 1. / (dims - 1)))
                    for C in channel_sizes:
                        x = [output_size for idx in range(dims)]
                        x[1] = C
                        self._norm_helper(x, torch.bfloat16, "cuda", 1e-1, is_batch_norm_else_instance_norm)

    def _softmax_helper(self, shape, reduction_axis, is_log_softmax, dtype, device, error):
        class MySoftmax(torch.nn.Module):
            __constants__ = ['reduction_axis']

            def __init__(self):
                super(MySoftmax, self).__init__()
                self.reduction_axis = reduction_axis

            def forward(self, x: torch.Tensor, y: torch.Tensor):
                o = torch.add(x, y)
                o = torch.nn.functional.softmax(o, dim=self.reduction_axis)
                return o

        class MyLogSoftmax(torch.nn.Module):
            __constants__ = ['reduction_axis']

            def __init__(self):
                super(MyLogSoftmax, self).__init__()
                self.reduction_axis = reduction_axis

            def forward(self, x: torch.Tensor, y: torch.Tensor):
                o = torch.add(x, y)
                o = torch.nn.functional.log_softmax(o, dim=self.reduction_axis)
                return o

        gradient_check = (dtype == torch.float64)
        t = MyLogSoftmax() if is_log_softmax else MySoftmax()

        x = torch.randn(shape, dtype=dtype, device=device, requires_grad=gradient_check)
        y = torch.randn(shape, dtype=dtype, device=device, requires_grad=gradient_check)
        t_jit = torch.jit.script(t)
        jit_o = t_jit(x, y)
        jit_o = t_jit(x, y)
        jit_o = t_jit(x, y)

        if gradient_check:
            gradcheck(t_jit.forward, [x, y], nondet_tol=1e-5)
        else:
            o = t(x, y)
            self.assertEqual(o.dtype, jit_o.dtype)
            # numerical issues here due to our scheduling.
            # can't use `self.assertEqual(o, jit_o)`
            self.assertTrue(self._compare("comparing output failed", o, jit_o, error))
            self.assertGraphContains(t_jit.graph_for(x, y), FUSION_GUARD)

    @unittest.skipIf(is_pre_volta(), "reduction not supported in pre volta device")
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_softmax_dtype(self):
        def t(x: torch.Tensor, y: torch.Tensor):
            o = torch.mul(x, y)
            o = torch.nn.functional.softmax(o, dim=0, dtype=torch.float32)
            return o

        x = torch.randn([4, 4], dtype=torch.float16, device="cuda").requires_grad_()
        y = torch.randn_like(x).requires_grad_()
        grad = torch.randn_like(x).float()

        ref_x = x.detach().requires_grad_()
        ref_y = y.detach().requires_grad_()
        o = t(ref_x, ref_y)
        o.backward(grad)

        t_jit = torch.jit.script(t)
        jit_o = t_jit(x, y)
        jit_o.backward(grad)
        jit_o = t_jit(x, y)
        jit_o.backward(grad)
        jit_o = t_jit(x, y)
        jit_o.backward(grad)
        x.grad.zero_()
        y.grad.zero_()
        jit_o = t_jit(x, y)
        jit_o.backward(grad)

        self.assertEqual(o.dtype, jit_o.dtype)
        self.assertEqual(ref_x.grad, x.grad)
        self.assertEqual(ref_y.grad, y.grad)
        self.assertTrue(self._compare("comparing output failed", o, jit_o, 1e-3))
        self.assertGraphContainsExactly(t_jit.graph_for(x, y), FUSION_GUARD, 1, consider_subgraphs=True)
        bwd_graph = list(
            list(t_jit.get_debug_state().execution_plans.values())[
                0].code.grad_executor_states()[0].execution_plans.values()
        )[0].graph
        FileCheck().check(FUSION_GUARD).run(bwd_graph)

    @unittest.skipIf(is_pre_volta(), "reduction not supported in pre volta device")
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test__softmax_function(self):
        def t(x: torch.Tensor, y: torch.Tensor):
            o = torch.mul(x, y)
            o = torch._softmax(o, dim=-1, half_to_float=False)
            return o

        x = torch.randn([4, 4], dtype=torch.float16, device="cuda")
        y = torch.randn_like(x)

        o = t(x, y)

        t_jit = torch.jit.script(t)
        jit_o = t_jit(x, y)
        jit_o = t_jit(x, y)
        jit_o = t_jit(x, y)

        self.assertEqual(o.dtype, jit_o.dtype)
        self.assertTrue(self._compare("comparing output failed", o, jit_o, 1e-3))
        self.assertGraphContainsExactly(t_jit.graph_for(x, y), FUSION_GUARD, 1, consider_subgraphs=True)

    @unittest.skipIf(is_pre_volta(), "reduction not supported in pre volta device")
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test__softmax_function_half_to_float(self):
        def t(x: torch.Tensor, y: torch.Tensor):
            o = torch.mul(x, y)
            o = torch._softmax(o, dim=-1, half_to_float=True)
            return o

        x = torch.randn([4, 4], dtype=torch.float16, device="cuda")
        y = torch.randn_like(x)

        o = t(x, y)

        t_jit = torch.jit.script(t)
        jit_o = t_jit(x, y)
        jit_o = t_jit(x, y)
        jit_o = t_jit(x, y)

        self.assertEqual(o.dtype, jit_o.dtype)
        self.assertTrue(self._compare("comparing output failed", o, jit_o, 1e-3))
        self.assertGraphContainsExactly(t_jit.graph_for(x, y), FUSION_GUARD, 1, consider_subgraphs=True)

    @unittest.skipIf(is_pre_volta(), "reduction not supported in pre volta device")
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_softmax(self):
        output_size = 10000
        dims = 4
        output_size = int(pow(output_size, 1. / dims))
        reduction_sizes = [67, 256, 1024, 4096]

        # gradient check
        for reduction_dim in range(dims):
            for is_log_softmax in [False, True]:
                shape = [output_size for idx in range(dims)]
                self._softmax_helper(shape, reduction_dim, is_log_softmax, torch.float64, "cuda", 1e-4)

        for reduction_dim in range(dims):
            for reduction_size in reduction_sizes:
                x = [output_size for idx in range(dims)]
                x[reduction_dim] = reduction_size
                for is_log_softmax in [False, True]:
                    self._softmax_helper(x, reduction_dim, is_log_softmax, torch.float32, "cuda", 1e-4)

    @unittest.skipIf(is_pre_volta(), "reduction not supported in pre volta device")
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_softmax_half(self):
        output_size = 10000
        dims = 4
        output_size = int(pow(output_size, 1. / dims))
        reduction_sizes = [67, 256, 1024, 4096]

        for reduction_dim in range(dims):
            for reduction_size in reduction_sizes:
                x = [output_size for idx in range(dims)]
                x[reduction_dim] = reduction_size
                for is_log_softmax in [False, True]:
                    self._softmax_helper(x, reduction_dim, is_log_softmax, torch.float16, "cuda", 5e-3)

    @unittest.skipIf(is_pre_volta(), "reduction not supported in pre volta device")
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    @unittest.skipIf(not TEST_BF16, "device does not support BFloat16")
    def test_softmax_bfloat(self):
        output_size = 10000
        dims = 4
        output_size = int(pow(output_size, 1. / dims))
        reduction_sizes = [67, 256, 1024, 4096]

        for reduction_dim in range(dims):
            for reduction_size in reduction_sizes:
                x = [output_size for idx in range(dims)]
                x[reduction_dim] = reduction_size
                for is_log_softmax in [False, True]:
                    self._softmax_helper(x, reduction_dim, is_log_softmax, torch.bfloat16, "cuda", 1e-1)

    @unittest.skipIf(is_pre_volta(), "reduction not supported in pre volta device")
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_reduction_permutation(self):
        x = [7, 8, 12]
        # note that num_dim is exclusive from len(x), so we are not reducing
        # to single element (codegen limitation at this moment)
        for num_reduce_dim in range(1, len(x)):
            for axes in itertools.combinations(range(len(x)), num_reduce_dim):
                for perm0 in itertools.permutations(range(len(x))):
                    for perm1 in itertools.permutations(range(len(x))):
                        self._reduction_helper(x, axes, torch.float32, "cuda", perm0, perm1)

    @unittest.skipIf(is_pre_volta(), "reduction not supported in pre volta device")
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_reduction_multiple_output(self):
        old_guard = torch._C._jit_set_nvfuser_guard_mode(True)
        torch._C._jit_set_bailout_depth(20)

        def t(x: torch.Tensor, y: torch.Tensor, scale: float, z: torch.Tensor):
            o = torch.mul(x, y)
            o = torch.mul(o, scale)
            out1 = torch.mul(o, z)
            out2 = torch.sum(out1, dim=[2])
            return out1, out2

        t_jit = torch.jit.script(t)
        x = torch.randn(8, 4, 10, 16, dtype=torch.float, device="cuda")
        y = torch.randn(8, 4, 10, 16, dtype=torch.float, device="cuda")
        z = torch.randn(8, 4, 10, 16, dtype=torch.float, device="cuda")
        scale = 0.5
        jit_o = t_jit(x, y, scale, z)
        jit_o = t_jit(x, y, scale, z)
        o = t(x, y, scale, z)
        for oo, jit_oo in zip(o, jit_o):
            self.assertEqual(oo.dtype, jit_oo.dtype)
            self.assertEqual(oo, jit_oo)
        self.assertGraphContains(t_jit.graph_for(x, y, scale, z), FUSION_GUARD)

        x = x.to(memory_format=torch.channels_last)
        y = y.to(memory_format=torch.channels_last)
        z = z.to(memory_format=torch.channels_last)
        jit_o = t_jit(x, y, scale, z)
        jit_o = t_jit(x, y, scale, z)
        o = t(x, y, scale, z)
        for oo, jit_oo in zip(o, jit_o):
            self.assertEqual(oo.dtype, jit_oo.dtype)
            self.assertEqual(oo, jit_oo)
        self.assertGraphContains(t_jit.graph_for(x, y, scale, z), FUSION_GUARD)
        torch._C._jit_set_nvfuser_guard_mode(old_guard)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_channels_last_with_broadcast(self):
        # setting this true forces a new graph to be generated with a new
        # input a different broadcast shape
        torch._C._jit_set_nvfuser_guard_mode(True)

        def t(x: torch.Tensor, y: torch.Tensor):
            o = torch.mul(x, y)
            o = o + 2.0
            return o
        t_jit = torch.jit.script(t)

        # Single Channel broadcasts
        # Test 1
        x = torch.randn(8, 4, 10, 16, dtype=torch.float, device="cuda")
        x = x.to(memory_format=torch.channels_last)

        y = torch.randn(8, 4, 10, 1, dtype=torch.float, device="cuda")
        y = y.to(memory_format=torch.channels_last)

        jit_o = t_jit(x, y)
        jit_o = t_jit(x, y)
        o = t(x, y)

        self.assertEqual(o.dtype, jit_o.dtype)
        self.assertEqual(o.is_contiguous(memory_format=torch.channels_last),
                         jit_o.is_contiguous(memory_format=torch.channels_last))
        self.assertEqual(o, jit_o)

        # Test 2
        y = torch.randn(8, 4, 1, 16, dtype=torch.float, device="cuda")
        y = y.to(memory_format=torch.channels_last)

        jit_o = t_jit(x, y)
        jit_o = t_jit(x, y)
        o = t(x, y)

        self.assertEqual(o.dtype, jit_o.dtype)
        self.assertEqual(o.is_contiguous(memory_format=torch.channels_last),
                         jit_o.is_contiguous(memory_format=torch.channels_last))
        self.assertEqual(o, jit_o)

        # Test 3
        y = torch.randn(8, 1, 10, 16, dtype=torch.float, device="cuda")
        y = y.to(memory_format=torch.channels_last)

        jit_o = t_jit(x, y)
        jit_o = t_jit(x, y)
        o = t(x, y)

        self.assertEqual(o.dtype, jit_o.dtype)
        self.assertEqual(o.is_contiguous(memory_format=torch.channels_last),
                         jit_o.is_contiguous(memory_format=torch.channels_last))
        self.assertEqual(o, jit_o)

        # Test 3
        y = torch.randn(1, 4, 10, 16, dtype=torch.float, device="cuda")
        y = y.to(memory_format=torch.channels_last)

        jit_o = t_jit(x, y)
        jit_o = t_jit(x, y)
        o = t(x, y)

        self.assertEqual(o.dtype, jit_o.dtype)
        self.assertEqual(o.is_contiguous(memory_format=torch.channels_last),
                         jit_o.is_contiguous(memory_format=torch.channels_last))
        self.assertEqual(o, jit_o)

        '''
        Currently, the JIT doesn't have tensor merge logic to handle adding
        a broadcast tensor with more than one broadcast into a non-broadcast
        tensor.  Therefore, either of these tests can fail depending on the
        sort implementation.  The second test is known to fail.

        # Two Channel broadcasts
        # Test 1
        y = torch.randn(8, 4, 1, 1, dtype=torch.float, device="cuda")
        y = y.to(memory_format=torch.channels_last)

        jit_o = t_jit(x, y)
        jit_o = t_jit(x, y)
        o = t(x, y)

        self.assertEqual(o.dtype, jit_o.dtype)
        self.assertEqual(o.is_contiguous(memory_format=torch.channels_last),
                         jit_o.is_contiguous(memory_format=torch.channels_last))
        self.assertEqual(o, jit_o)

        # Test 2
        y = torch.randn(8, 4, 1, 1, dtype=torch.float, device="cuda")
        y = y.to(memory_format=torch.channels_last).transpose(2,3)
        x = x.transpose(2,3)

        jit_o = t_jit(x, y)
        jit_o = t_jit(x, y)
        o = t(x, y)

        self.assertEqual(o.dtype, jit_o.dtype)
        self.assertEqual(o.is_contiguous(memory_format=torch.channels_last),
                         jit_o.is_contiguous(memory_format=torch.channels_last))
        self.assertEqual(o, jit_o)
        '''

    @unittest.skipIf(is_pre_volta(), "reduction not supported in pre volta device")
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_pw_single_reduction_partition(self):
        sizes = [2, 2, 2]
        dtype = torch.float
        device = "cuda"
        x = torch.randn(sizes, dtype=dtype, device=device)
        y = torch.randn(sizes, dtype=dtype, device=device)
        z = torch.randn(sizes, dtype=dtype, device=device)

        def t(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
            o = torch.add(x, y)
            o = torch.sum(o, dim=[0])
            o = torch.add(o, z)
            return o
        t_jit = torch.jit.script(t)
        jit_o = t_jit(x, y, z)
        jit_o = t_jit(x, y, z)
        o = t(x, y, z)
        self.assertEqual(o.dtype, jit_o.dtype)
        self.assertEqual(o, jit_o)
        self.assertGraphContains(t_jit.graph_for(x, y, z), FUSION_GUARD)

    @unittest.skipIf(is_pre_volta(), "reduction not supported in pre volta device")
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_permutation_preservation(self):
        sizes = [2, 3, 4, 5]
        dtype = torch.float
        device = "cuda"

        with nvfuser_singleton_fusion(True):

            def t(x: torch.Tensor):
                return torch.relu(x)

            t_jit = torch.jit.script(t)
            x = torch.randn(sizes, dtype=dtype, device=device).to(memory_format=torch.channels_last)
            self._run_helper(t_jit, t, x, check_stride=True)

            def t(x: torch.Tensor, y: torch.Tensor):
                return torch.add(x, y)

            t_jit = torch.jit.script(t)
            x = torch.randn(sizes, dtype=dtype, device=device).to(memory_format=torch.channels_last)
            y = torch.randn(sizes[1:], dtype=dtype, device=device)
            self._run_helper(t_jit, t, x, y, check_stride=True)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_permutation_preservation_edge_case_0(self):
        sizes = [2, 3, 4, 5]
        dtype = torch.float
        device = "cuda"
        x = torch.randn(sizes, dtype=dtype, device=device).to(memory_format=torch.channels_last)
        # mismatch rank with *note* different permutation recognized by PE
        bias = torch.randn(3, dtype=dtype, device=device).unsqueeze(-1).unsqueeze(-1)

        def t(x, y):
            return x + y

        t_jit = torch.jit.script(t)
        with nvfuser_singleton_fusion(True):
            self._run_helper(t_jit, t, x, bias, check_stride=True)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_permutation_preservation_edge_case_1_broken(self):
        sizes = [2, 3, 4, 5]
        dtype = torch.float
        device = "cuda"
        x = torch.randn(sizes, dtype=dtype, device=device).to(memory_format=torch.channels_last)
        # in-compatible permutation, this will cause format propagation to break
        bias = torch.randn(4, 5, dtype=dtype, device=device)

        def t(x, y):
            return x + y

        t_jit = torch.jit.script(t)
        with nvfuser_singleton_fusion(True):
            for _ in range(5):
                jit_o = t_jit(x, bias)

        o = t(x, bias)
        self.assertEqual(o.dtype, jit_o.dtype)
        self.assertEqual(o, jit_o)
        try:
            # nvfuser does not support in-compatible permutation, this will throw
            self.assertEqual(o.stride(), jit_o.stride())
        except Exception as e:
            warnings.warn(
                "permutation propagation is broken, proper support should come after nvfuser permutation scheduler update")
        self.assertGraphContains(t_jit.graph_for(x, bias), FUSION_GUARD)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_permutation_preservation_edge_case_2(self):
        sizes = [2, 3, 4, 5]
        dtype = torch.float
        device = "cuda"
        x = torch.randn(sizes, dtype=dtype, device=device).to(memory_format=torch.channels_last)
        y = torch.randn(sizes, dtype=dtype, device=device).to(memory_format=torch.channels_last)
        z = torch.randn(sizes, dtype=dtype, device=device).to(memory_format=torch.channels_last)

        def t(x, y, w):
            tmp = torch.lerp(x, y, w)
            tmp = torch.clamp(tmp, -1.0, 0.5)
            tmp = torch.nn.functional.softplus(tmp)
            return torch.threshold(tmp, -2.0, 0.5)

        t_jit = torch.jit.script(t)
        self._run_helper(t_jit, t, x, y, z, check_stride=True)

    @unittest.skipIf(is_pre_volta(), "reduction not supported in pre volta device")
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_normalization_partition(self):
        sizes = [3, 8, 5]
        dtype = torch.float
        device = "cuda"
        x = torch.randn(sizes, dtype=dtype, device=device)
        y = torch.randn(sizes, dtype=dtype, device=device)
        z = torch.randn(sizes, dtype=dtype, device=device)
        r_m = torch.randn(8, dtype=dtype, device=device)
        r_v = torch.randn(8, dtype=dtype, device=device)

        def t(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, r_mean: torch.Tensor, r_var: torch.Tensor):
            o = torch.add(x, y)
            o = torch.nn.functional.softmax(o, dim=0)
            o = torch.add(o, z)
            o = torch.nn.functional.batch_norm(o, r_mean, r_var, training=True)
            return o
        t_jit = torch.jit.script(t)
        jit_o = t_jit(x, y, z, r_m, r_v)
        jit_o = t_jit(x, y, z, r_m, r_v)
        o = t(x, y, z, r_m, r_v)
        self.assertEqual(o.dtype, jit_o.dtype)
        self.assertEqual(o, jit_o)
        self.assertGraphContains(t_jit.graph_for(x, y, z, r_m, r_v), FUSION_GUARD)

    @unittest.skipIf(is_pre_volta(), "reduction not supported in pre volta device")
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_sum_to_one(self):
        dtype = torch.float
        device = "cuda"
        x = torch.randn([4, 5, 6], dtype=dtype, device=device)

        def t(x: torch.Tensor):
            o = torch.add(x, 1)
            o = torch.sum(o, dim=[0, 1, 2])
            return o
        t_jit = torch.jit.script(t)
        jit_o = t_jit(x)
        jit_o = t_jit(x)
        o = t(x)
        self.assertEqual(o.dtype, jit_o.dtype)
        self.assertEqual(o, jit_o)
        self.assertGraphContains(t_jit.graph_for(x), FUSION_GUARD)

    @unittest.skipIf(is_pre_volta(), "reduction not supported in pre volta device")
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_single_reduction_broadcast(self):
        dtype = torch.float
        device = "cuda"
        x = torch.randn([7, 4, 8], dtype=dtype, device=device)
        y = torch.randn([4, 8], dtype=dtype, device=device)
        z = torch.randn([1, 4, 8], dtype=dtype, device=device)

        def t(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
            o = torch.add(x, y)
            o = torch.add(o, z)
            o = torch.sum(o, dim=[0])
            return o
        t_jit = torch.jit.script(t)
        jit_o = t_jit(x, y, z)
        jit_o = t_jit(x, y, z)
        o = t(x, y, z)
        self.assertEqual(o.dtype, jit_o.dtype)
        self.assertEqual(o, jit_o)
        self.assertGraphContains(t_jit.graph_for(x, y, z), FUSION_GUARD)

    @unittest.skipIf(is_pre_volta(), "reduction not supported in pre volta device")
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_trivial_reduction(self):
        dtype = torch.float
        device = "cuda"
        x = torch.randn([1, 4, 8], dtype=dtype, device=device)

        def t(x: torch.Tensor):
            o = torch.add(x, 1)
            o = torch.sum(o, dim=[0])
            o = torch.sum(o, dim=[0])
            return o
        t_jit = torch.jit.script(t)
        jit_o = t_jit(x)
        jit_o = t_jit(x)
        o = t(x)
        self.assertEqual(o.dtype, jit_o.dtype)
        self.assertEqual(o, jit_o)
        self.assertGraphContains(t_jit.graph_for(x), FUSION_GUARD)

    @unittest.skip("Skipped due to rand_like behavior change")
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_profiling_node(self):
        # TODO: should we change this test to not use rand_like, or just
        # remove this test?
        dtype = torch.float
        device = "cuda"
        x = torch.randn(4, 8, 8, 8, dtype=dtype, device=device)

        def repro(x: torch.Tensor, alpha: float):
            o = torch.rand_like(x)
            o = torch.add(o, alpha)
            return o
        repro_jit = torch.jit.script(repro)
        self._run_helper(repro_jit, repro, x, 0.6)

    @unittest.skipIf(is_pre_volta(), "reduction not supported in pre volta device")
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_reduction_sizes_op(self):
        dtype = torch.float
        device = "cuda"
        x = torch.randn(2, 3, 4, 5, dtype=dtype, device=device)
        y = torch.randn(2, 3, 4, 5, dtype=dtype, device=device)

        def t(x: torch.Tensor, y: torch.Tensor):
            o = x + y
            o = torch.relu(o)
            o = o.sum((1, 3))
            return o.size()
        t_jit = torch.jit.script(t)
        jit_o = t_jit(x, y)
        jit_o = t_jit(x, y)
        o = t(x, y)
        self.assertEqual(o, jit_o)
        # since the output value is not used at all, the fusion operator should
        # have been optimized away
        self.assertGraphContainsExactly(t_jit.graph_for(x, y), FUSION_GUARD, 0)

    @unittest.skipIf(is_pre_volta(), "reduction not supported in pre volta device")
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_profile_ivalue(self):
        dtype = torch.float
        device = "cuda"
        x = torch.randn([7, 4, 7], dtype=dtype, device=device)
        y = torch.randn([7, 4, 7], dtype=dtype, device=device)

        def t(x: torch.Tensor, y: torch.Tensor, dim: List[int], keepdim: bool):
            o = torch.add(x, y)
            o = o.sum(dim, keepdim=keepdim)
            return o

        t_jit = torch.jit.script(t)
        jit_o = t_jit(x, y, (0, 1), False)
        jit_o = t_jit(x, y, (0, 1), False)
        o = t(x, y, (0, 1), False)
        self.assertEqual(o.dtype, jit_o.dtype)
        self.assertEqual(o, jit_o)
        self.assertGraphContains(t_jit.graph_for(x, y, (0, 1), False), FUSION_GUARD)

    @unittest.skipIf(is_pre_volta(), "reduction not supported in pre volta device")
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_profile_ivalue_multiple_profiles(self):
        dtype = torch.float
        device = "cuda"
        x = torch.randn([7, 4, 7], dtype=dtype, device=device)

        def t(x, num: int):
            for i in range(num):
                # varying reduction axes should break profile_ivalue
                tmp = x.sum(i, keepdim=True)
                # inplace add on input/output, can't be functionalized/fused
                x += tmp
            return x

        with nvfuser_singleton_fusion(True):
            t_jit = torch.jit.script(t)
            self._run_helper(t_jit, t, x, 3, num_fusion=0)

    @unittest.skipIf(is_pre_volta(), "reduction not supported in pre volta device")
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_sum_to_size(self):
        dtype = torch.float
        device = "cuda"
        x = torch.randn([2, 4, 4], dtype=dtype, device=device)
        y = torch.randn([2, 4, 4], dtype=dtype, device=device)

        def t(x: torch.Tensor, y: torch.Tensor, new_size: List[int]):
            o = torch.add(x, y)
            o = o.sum_to_size(new_size)
            return o

        t_jit = torch.jit.script(t)
        self._run_helper(t_jit, t, x, y, (4, 1))

        # update shape: old kernel should handle dynamic shape well without
        # recompilation
        x = torch.randn([2, 5, 8], dtype=dtype, device=device)
        y = torch.randn([2, 5, 8], dtype=dtype, device=device)
        # (TODO) check executed kernel, should extend autograd.profiler to fused
        # kernels
        self._run_helper(t_jit, t, x, y, (5, 1))

        with nvfuser_singleton_fusion(True):
            x = torch.randn([2, 5, 8], dtype=dtype, device=device)

            def t(x: torch.Tensor):
                # no-op reduction
                return x.sum_to_size((2, 5, 8))

            t_jit = torch.jit.script(t)
            self._run_helper(t_jit, t, x)

    @unittest.skipIf(is_pre_volta(), "reduction not supported in pre volta device")
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_grad_sum_to_size(self):
        dtype = torch.float
        device = "cuda"
        x = torch.randn([2, 4, 4], dtype=dtype, device=device).requires_grad_()
        y = torch.randn([4], dtype=dtype, device=device).requires_grad_()
        grad = torch.randn([2, 4, 4], dtype=dtype, device=device)

        ref_x = x.detach().clone().requires_grad_()
        ref_y = y.detach().clone().requires_grad_()

        def t(x: torch.Tensor, y: torch.Tensor):
            o = torch.add(x, y)
            o = torch.relu(o)
            return o

        # profiling runs for forward & backward
        t_jit = torch.jit.script(t)
        jit_o = t_jit(x, y)
        jit_o.backward(grad)
        jit_o = t_jit(x, y)
        jit_o.backward(grad)

        x.grad = None
        y.grad = None
        jit_o = t_jit(x, y)
        jit_o.backward(grad)
        o = t(ref_x, ref_y)
        o.backward(grad)
        self.assertEqual(o.dtype, jit_o.dtype)
        self.assertEqual(o, jit_o)
        self.assertEqual(x.grad, ref_x.grad)
        self.assertEqual(y.grad, ref_y.grad)
        bwd_graph = list(
            list(t_jit.get_debug_state().execution_plans.values())[
                0].code.grad_executor_states()[0].execution_plans.values()
        )[0].graph
        FileCheck().check(FUSION_GUARD).run(bwd_graph)

        # update shape: old kernel should handle dynamic shape well without
        # recompilation
        x = torch.randn([2, 5, 8], dtype=dtype, device=device).requires_grad_()
        y = torch.randn([8], dtype=dtype, device=device).requires_grad_()
        ref_x = x.detach().clone().requires_grad_()
        ref_y = y.detach().clone().requires_grad_()
        grad = torch.randn([2, 5, 8], dtype=dtype, device=device)
        jit_o = t_jit(x, y)
        # (TODO) check executed kernel, should extend autograd.profiler to fused
        # kernels
        jit_o.backward(grad)
        o = t(ref_x, ref_y)
        o.backward(grad)
        self.assertEqual(o.dtype, jit_o.dtype)
        self.assertEqual(o, jit_o)
        self.assertEqual(x.grad, ref_x.grad)
        self.assertEqual(y.grad, ref_y.grad)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_dropout_inference_fusion(self):
        dtype = torch.float
        device = "cuda"
        x = torch.randn([10, 4, 8], dtype=dtype, device=device)

        def t(x: torch.Tensor, p: float, train: bool):
            o = torch.nn.functional.dropout(x, p, training=train)
            o = o + 1.0
            return o

        t_jit = torch.jit.script(t)

        self._run_helper(t_jit, t, x, 0.15, False)

    @unittest.skipIf(not TEST_LARGE_TENSOR, "not enough memory")
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_dropout_train_nograd_fusion(self):
        dtype = torch.float
        device = "cuda"
        x = torch.randn([64, 128, 1024], dtype=dtype, device=device)

        def t(x: torch.Tensor, p: float, train: bool):
            o = torch.nn.functional.dropout(x, p, training=train)
            o = o + 1.0
            return o

        t_jit = torch.jit.script(t)

        self._run_helper(t_jit, t, x, 0.0, True, check_runs=20)
        self._run_helper(t_jit, t, x, 1.0, True, check_runs=20)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_dropout_train_nograd_prob_check(self):
        dtype = torch.float
        device = "cuda"
        x = torch.randn([1024, 1024], dtype=dtype, device=device)

        def t(x: torch.Tensor, p: float, train: bool):
            o = torch.nn.functional.dropout(x, p, training=train)
            o = o * 2.0
            return o

        t_jit = torch.jit.script(t)

        for prob in [0.0, 0.15, 0.5, 0.85, 1.]:
            torch.cuda.manual_seed_all(123)
            jit_o = t_jit(x, prob, True)
            torch.cuda.manual_seed_all(123)
            jit_o = t_jit(x, prob, True)

            self.assertTrue(jit_o.detach().isfinite().all().item())

            num_elems = x.numel()
            num_zeros = num_elems - jit_o.detach().count_nonzero().item()
            percent_zeros = num_zeros / num_elems

            self.assertTrue((percent_zeros >= (prob - 0.01)) and (percent_zeros <= (prob + 0.01)))
            self.assertGraphContainsExactly(t_jit.graph_for(x, prob, True), FUSION_GUARD, 1, consider_subgraphs=True)

    @unittest.skipIf(is_pre_volta(), "reduction not supported in pre volta device")
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_dropout_training_fusion(self):
        dtype = torch.float
        device = "cuda"
        sizes = [2, 3, 4, 5]

        def t(x: torch.Tensor, p: float, train: bool):
            o = torch.nn.functional.dropout(x, p, training=train)
            o = o * 2.0
            return o

        def t2(x: torch.Tensor, p: float, train: bool):
            o = torch.nn.functional.softmax(x, dim=-1)
            o = torch.nn.functional.dropout(o, p, training=train)
            return o

        # disabling cache so new inputs would generate new graph
        t.__disable_jit_function_caching__ = True
        t2.__disable_jit_function_caching__ = True

        for fn in [t, t2]:
            for m_format in [torch.contiguous_format, torch.channels_last]:
                fn_jit = torch.jit.script(fn)
                x = torch.randn(sizes, dtype=dtype, device=device, requires_grad=True).to(memory_format=m_format)
                grads = torch.randn(sizes, dtype=dtype, device=device).to(memory_format=m_format)

                # The drop probability needs to be set to zero given that the order of picking random
                # numbers between eager mode and the jit is different
                self._run_training_helper(fn_jit, fn, grads, x, 0.0, True)


    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_gelu(self):
        old_guard = torch._C._jit_set_nvfuser_guard_mode(True)
        dtype = torch.float
        device = "cuda"
        x = torch.randn([1024, 1024], dtype=dtype, device=device, requires_grad=True)
        grads = torch.randn([1024, 1024], dtype=dtype, device=device, requires_grad=False)

        def t(x: torch.Tensor, mode: str):
            o = torch.nn.functional.gelu(x, approximate=mode)
            o = o * 2.0
            return o

        t_jit = torch.jit.script(t)
        self._run_training_helper(t_jit, t, grads, x, 'none')
        self._run_training_helper(t_jit, t, grads, x, 'tanh')
        torch._C._jit_set_nvfuser_guard_mode(old_guard)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_dropout_training_prob_check(self):
        dtype = torch.float
        device = "cuda"
        x = torch.randn([1024, 1024], dtype=dtype, device=device, requires_grad=True)
        x_nograd = torch.randn([1024, 1024], dtype=dtype, device=device)

        def t(x: torch.Tensor, p: float, train: bool):
            o = torch.nn.functional.dropout(x, p, training=train)
            o = o * 2.0
            return o

        t_jit = torch.jit.script(t)

        for prob in [0.0, 0.15, 0.5, 0.85, 1.]:
            torch.cuda.manual_seed_all(123)
            jit_o = t_jit(x, prob, True)
            torch.cuda.manual_seed_all(123)
            jit_o = t_jit(x, prob, True)
            torch.cuda.manual_seed_all(123)
            jit_o = t_jit(x, prob, True)

            self.assertTrue(jit_o.detach().isfinite().all().item())

            num_elems = x.numel()
            num_zeros = num_elems - jit_o.detach().count_nonzero().item()
            percent_zeros = num_zeros / num_elems

            self.assertTrue((percent_zeros >= (prob - 0.01)) and (percent_zeros <= (prob + 0.01)))
            self.assertGraphContainsExactly(t_jit.graph_for(x, prob, True), FUSION_GUARD, 1, consider_subgraphs=True)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_linear(self):
        in_feature = 2
        out_feature = 8
        # Changing the input dims to be 3-D to avoid eager mode bias fusion
        # The bias fusion causes some precision issues with TF-32
        weight = torch.randn(out_feature, in_feature, dtype=torch.float32, device='cuda')
        bias = torch.randn(out_feature, dtype=torch.float32, device='cuda')

        def t(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
            o = torch.nn.functional.linear(x, weight, bias)
            o = torch.relu(o)
            return o

        # disabling cache so new inputs would generate new graph
        t.__disable_jit_function_caching__ = True

        sizes = [in_feature, ]
        for i in range(4):
            # increase input rank in each iteration
            sizes.insert(0, i + 2)
            x = torch.randn(*sizes, dtype=torch.float32, device='cuda')
            t_jit = torch.jit.script(t)
            # fusion only happens for input rank >= 4
            has_fusion = 0 if len(sizes) < 4 else 1
            self._run_helper(t_jit, t, x, weight, bias, check_stride=True, num_fusion=has_fusion)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_linear_symbolic_shapes(self):
        def fn(x: int):
            y = torch.zeros((3, 4, x, x + 2)).cuda()
            for i in range(2):
                inp = torch.rand((3, 4, x, x + i)).cuda()
                weight = torch.rand((x + 2, x + i)).cuda()
                bias = torch.rand((x, x + 2)).cuda()
                y += torch.sin(torch.nn.functional.linear(inp, weight, bias))
            return y

        fn_s = torch.jit.script(fn)
        fn_s(5)
        fn_s(5)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_conv2d_symbolic_shapes(self):
        def fn(x: int):
            responses = []
            for i in range(2):
                inp = torch.rand((3, 3, 32, 32)).cuda()
                weight = torch.rand((x + i, 3, 7, 7)).cuda()
                bias = torch.rand((x + i)).cuda()
                res = torch.nn.functional.conv2d(inp, weight, bias, padding=3)
                responses.append(res)
            return responses

        fn_s = torch.jit.script(fn)
        fn_s(5)
        fn_s(5)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_backward_type(self):
        # not super useful to check gradient of integer/bool, so skipping here
        type_pairs = [
            (torch.float, torch.half),
            (torch.double, torch.half),
            (torch.float, torch.double),
        ]
        if TEST_BF16:
            type_pairs += [
                (torch.float, torch.bfloat16),
                (torch.double, torch.bfloat16),
            ]
        for x_type, y_type in type_pairs:
            x = torch.randn(4, 2, dtype=x_type, device='cuda', requires_grad=True)
            y = torch.randn(4, 2, dtype=y_type, device='cuda', requires_grad=True)
            grad = torch.randn(4, 2, dtype=torch.float, device='cuda')

            def test1(x: torch.Tensor, y: torch.Tensor):
                o = torch.add(x, y)
                o = torch.add(o, y)
                o = torch.add(o, y)
                o = torch.add(o, y)
                o = o + 1.0
                return o

            test1_jit = torch.jit.script(test1)
            for i in range(3):
                jit_o = test1_jit(x, y)
                jit_o.backward(grad)

            bwd_graph = list(
                list(test1_jit.get_debug_state().execution_plans.values())[
                    0].code.grad_executor_states()[0].execution_plans.values()
            )[0].graph

            FileCheck().check(FUSION_GROUP).run(bwd_graph)
            self.assertEqual(x.grad.dtype, x.dtype)
            self.assertEqual(y.grad.dtype, y.dtype)

    @unittest.skipIf(is_pre_volta(), "reduction not supported in pre volta device")
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_autocast_1(self):
        def t(x: torch.Tensor, y: torch.Tensor):
            o = x * 2.0
            o = torch.softmax(o, dim=-1)
            o = o * 3.0
            o = torch._C._nn.linear(o, y)
            return o

        x = torch.randn(8, 4, dtype=torch.half, device='cuda', requires_grad=True)
        y = torch.randn(4, 4, dtype=torch.float, device='cuda', requires_grad=True)
        grad = torch.randn(8, 4, dtype=torch.half, device='cuda', requires_grad=False)
        t_jit = torch.jit.script(t)

        for i in range(3):
            with torch.cuda.amp.autocast():
                jit_o = t_jit(x, y)
                if i == 2:
                    fwd_graph = t_jit.graph_for(x, y)
            jit_o.backward(grad)

        self.assertGraphContainsExactly(fwd_graph, FUSION_GUARD, 1, consider_subgraphs=True)

        with torch.cuda.amp.autocast():
            bwd_graph = list(
                list(t_jit.get_debug_state().execution_plans.values())[
                    0].code.grad_executor_states()[0].execution_plans.values()
            )[0].graph
        FileCheck().check(FUSION_GROUP).run(bwd_graph)

        self.assertEqual(jit_o.dtype, torch.half)
        self.assertEqual(x.grad.dtype, x.dtype)
        self.assertEqual(y.grad.dtype, y.dtype)

    @unittest.skipIf(is_pre_volta(), "reduction not supported in pre volta device")
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_autocast_2(self):
        def t(x: torch.Tensor):
            o = x * 2.0
            o = torch.softmax(o, dim=-1)
            o = o * 3.0
            o = torch.softmax(o, dim=-1)
            o = o * 4.0
            return o

        x = torch.randn(8, 4, dtype=torch.half, device='cuda', requires_grad=True)
        grad = torch.randn(8, 4, dtype=torch.float, device='cuda', requires_grad=False)
        t_jit = torch.jit.script(t)

        for i in range(3):
            with torch.cuda.amp.autocast():
                jit_o = t_jit(x)
                if i == 2:
                    fwd_graph = t_jit.graph_for(x)
            jit_o.backward(grad)

        self.assertGraphContainsExactly(fwd_graph, FUSION_GUARD, 1, consider_subgraphs=True)

        with torch.cuda.amp.autocast():
            bwd_graph = list(
                list(t_jit.get_debug_state().execution_plans.values())[
                    0].code.grad_executor_states()[0].execution_plans.values()
            )[0].graph
        FileCheck().check(FUSION_GROUP).run(bwd_graph)

        self.assertEqual(jit_o.dtype, torch.float)
        self.assertEqual(x.grad.dtype, x.dtype)

    @unittest.skipIf(is_pre_volta(), "reduction not supported in pre volta device")
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    @unittest.skipIf(not TEST_BF16, "device does not support BFloat16")
    def test_autocast_1_bfloat(self):
        def t(x: torch.Tensor, y: torch.Tensor):
            o = x * 2.0
            o = torch.softmax(o, dim=-1)
            o = o * 3.0
            o = torch._C._nn.linear(o, y)
            return o

        x = torch.randn(8, 4, dtype=torch.bfloat16, device='cuda', requires_grad=True)
        y = torch.randn(4, 4, dtype=torch.float, device='cuda', requires_grad=True)
        grad = torch.randn(8, 4, dtype=torch.bfloat16, device='cuda', requires_grad=False)
        t_jit = torch.jit.script(t)

        for i in range(3):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                jit_o = t_jit(x, y)
                if i == 2:
                    fwd_graph = t_jit.graph_for(x, y)
            jit_o.backward(grad)

        self.assertGraphContainsExactly(fwd_graph, FUSION_GUARD, 1, consider_subgraphs=True)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            bwd_graph = list(
                list(t_jit.get_debug_state().execution_plans.values())[
                    0].code.grad_executor_states()[0].execution_plans.values()
            )[0].graph
        FileCheck().check(FUSION_GROUP).run(bwd_graph)

        self.assertEqual(jit_o.dtype, torch.bfloat16)
        self.assertEqual(x.grad.dtype, x.dtype)
        self.assertEqual(y.grad.dtype, y.dtype)

    @unittest.skipIf(is_pre_volta(), "reduction not supported in pre volta device")
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    @unittest.skipIf(not TEST_BF16, "device does not support BFloat16")
    def test_autocast_2_bfloat(self):
        def t(x: torch.Tensor):
            o = x * 2.0
            o = torch.softmax(o, dim=-1)
            o = o * 3.0
            o = torch.softmax(o, dim=-1)
            o = o * 4.0
            return o

        x = torch.randn(8, 4, dtype=torch.bfloat16, device='cuda', requires_grad=True)
        grad = torch.randn(8, 4, dtype=torch.float, device='cuda', requires_grad=False)
        t_jit = torch.jit.script(t)

        for i in range(3):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                jit_o = t_jit(x)
                if i == 2:
                    fwd_graph = t_jit.graph_for(x)
            jit_o.backward(grad)

        self.assertGraphContainsExactly(fwd_graph, FUSION_GUARD, 1, consider_subgraphs=True)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            bwd_graph = list(
                list(t_jit.get_debug_state().execution_plans.values())[
                    0].code.grad_executor_states()[0].execution_plans.values()
            )[0].graph
        FileCheck().check(FUSION_GROUP).run(bwd_graph)

        self.assertEqual(jit_o.dtype, torch.float)
        self.assertEqual(x.grad.dtype, x.dtype)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_to_dtype_fp32_to_fp16(self):
        def t(x: torch.Tensor):
            o = x * 2.0
            o = o.to(dtype=torch.half)
            o = o * 3.0
            return o

        x = torch.randn(8, 4, dtype=torch.float, device='cuda')
        t_jit = torch.jit.script(t)

        for i in range(3):
            jit_o = t_jit(x)

        self.assertGraphContainsExactly(t_jit.graph_for(x), FUSION_GUARD, 1)
        self.assertEqual(jit_o.dtype, torch.half)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_to_dtype_fp16_to_fp32(self):
        def t(x: torch.Tensor):
            o = x * 2.0
            o = o.to(dtype=torch.float)
            o = o * 3.0
            return o

        x = torch.randn(8, 4, dtype=torch.half, device='cuda')
        t_jit = torch.jit.script(t)

        for i in range(3):
            jit_o = t_jit(x)

        self.assertGraphContainsExactly(t_jit.graph_for(x), FUSION_GUARD, 1)
        self.assertEqual(jit_o.dtype, torch.float)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_to_dtype_fp16_to_fp16(self):
        def t(x: torch.Tensor):
            o = x * 2.0
            o = o.to(dtype=torch.half)
            o = o * 3.0
            return o

        x = torch.randn(8, 4, dtype=torch.half, device='cuda')
        t_jit = torch.jit.script(t)

        for i in range(3):
            jit_o = t_jit(x)

        self.assertGraphContainsExactly(t_jit.graph_for(x), FUSION_GUARD, 1)
        self.assertEqual(jit_o.dtype, torch.half)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    @unittest.skipIf(not TEST_BF16, "device does not support BFloat16")
    def test_to_dtype_fp32_to_bf16(self):
        def t(x: torch.Tensor):
            o = x * 2.0
            o = o.to(dtype=torch.bfloat16)
            o = o * 3.0
            return o

        x = torch.randn(8, 4, dtype=torch.float, device='cuda')
        t_jit = torch.jit.script(t)

        for i in range(3):
            jit_o = t_jit(x)

        self.assertGraphContainsExactly(t_jit.graph_for(x), FUSION_GUARD, 1)
        self.assertEqual(jit_o.dtype, torch.bfloat16)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    @unittest.skipIf(not TEST_BF16, "device does not support BFloat16")
    def test_to_dtype_bf16_to_fp32(self):
        def t(x: torch.Tensor):
            o = x * 2.0
            o = o.to(dtype=torch.float)
            o = o * 3.0
            return o

        x = torch.randn(8, 4, dtype=torch.bfloat16, device='cuda')
        t_jit = torch.jit.script(t)

        for i in range(3):
            jit_o = t_jit(x)

        self.assertGraphContainsExactly(t_jit.graph_for(x), FUSION_GUARD, 1)
        self.assertEqual(jit_o.dtype, torch.float)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    @unittest.skipIf(not TEST_BF16, "device does not support BFloat16")
    def test_to_dtype_bf16_to_bf16(self):
        def t(x: torch.Tensor):
            o = x * 2.0
            o = o.to(dtype=torch.bfloat16)
            o = o * 3.0
            return o

        x = torch.randn(8, 4, dtype=torch.bfloat16, device='cuda')
        t_jit = torch.jit.script(t)

        for i in range(3):
            jit_o = t_jit(x)

        self.assertGraphContainsExactly(t_jit.graph_for(x), FUSION_GUARD, 1)
        self.assertEqual(jit_o.dtype, torch.bfloat16)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(not TEST_MULTIGPU, "requires multiple CUDA device")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_multiple_device_pw(self):

        def t(x):
            o = x + 1.0
            o = torch.relu(o)
            return o

        x = torch.randn(2, dtype=torch.float32, device="cuda")
        t_jit = torch.jit.script(t)

        for i in range(3):
            jit_o = t_jit(x)

        self.assertGraphContainsExactly(t_jit.graph_for(x), FUSION_GUARD, 1)
        torch.cuda.device(1)
        x = x.to("cuda:1")
        jit_o = t_jit(x)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_graph_for_with_missing_optimized_engine(self):
        x = torch.randn(8, 4, 2, dtype=torch.float, device="cuda").requires_grad_()

        def t(x: torch.Tensor, flag: bool):
            x = x + 1.0
            x = torch.relu(x)
            if flag:
                o = x + 1.0
                o = torch.relu(o)
            else:
                o = x + 2.0
                o = torch.relu(o)
            return o

        t_jit = torch.jit.script(t)
        jit_o = t_jit(x, False)
        jit_o = t_jit(x, False)
        jit_o = t_jit(x, True)
        o = t(x, True)
        self.assertEqual(o, jit_o)
        # since the output value is not used at all, the fusion operator should
        # have been optimized away
        self.assertGraphContainsExactly(t_jit.graph_for(x, True), FUSION_GUARD, 1, True)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_branches(self):
        in_feature = 2
        out_feature = 4
        x = torch.randn(4, in_feature, dtype=torch.float32, device='cuda')
        weight = torch.randn(out_feature, in_feature, dtype=torch.float32, device='cuda')
        bias = torch.randn(out_feature, dtype=torch.float32, device='cuda')

        def t(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, flag: bool):
            if flag:
                o = torch.nn.functional.linear(x, weight, bias)
                o = o + 1.0
                o = torch.relu(o)
            else:
                o = x.sum()
                o = o + 2.0
                o = torch.relu(o)
            return o

        t_jit = torch.jit.script(t)
        jit_o = t_jit(x, weight, bias, True)
        jit_o = t_jit(x, weight, bias, True)
        o = t(x, weight, bias, True)
        self.assertEqual(o, jit_o)
        # since the output value is not used at all, the fusion operator should
        # have been optimized away
        self.assertGraphContainsExactly(t_jit.graph_for(x, weight, bias, True), FUSION_GUARD, 1)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_scalar_tensor(self):
        x = torch.empty([], device="cuda", dtype=torch.float32)

        def t(x: torch.Tensor):
            o = x + 1.0
            o = torch.nn.functional.relu(o)
            return o

        # bias set to true.
        t_jit = torch.jit.script(t)
        jit_o = t_jit(x)
        jit_o = t_jit(x)
        o = t(x)
        self.assertEqual(o, jit_o)
        # since the output value is not used at all, the fusion operator should
        # have been optimized away
        self.assertGraphContainsExactly(t_jit.graph_for(x), FUSION_GUARD, 1)

    @unittest.skipIf(os.environ.get('PYTORCH_NO_CUDA_MEMORY_CACHING') is not None,
                     "skipping graph_rng when caching allocator is disabled")
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(CUDA_MAJOR < 11, "requires CUDA11 or above")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_graph_rng(self):
        self.assertTrue(torch._C._jit_nvfuser_enabled())
        size = 10000
        a = torch.randn((size,), device="cuda", dtype=torch.float)

        def t(x):
            o = x + 1.0
            o = torch.nn.functional.dropout(o, p=0.1)
            o = o + 1.0
            o = torch.nn.functional.dropout(o, p=0.1)
            return o

        t_jit = torch.jit.script(t)

        for _ in range(3):
            t_jit(a)

        self.assertGraphContainsExactly(t_jit.graph_for(a), FUSION_GUARD, 1)

        # Control (jitted, ungraphed)
        torch.cuda.manual_seed(5)
        eager_out = a.clone()
        for _ in range(3):
            eager_out = t_jit(eager_out)

        graph_in = a.clone()
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            torch.cuda.manual_seed(5)
            g.capture_begin()
            graph_out = t_jit(graph_in)
            g.capture_end()
        torch.cuda.current_stream().wait_stream(s)
        # g is now a jitted, graphed version of t.

        # Runs a (jitted, graphed) -> (jitted, ungraphed) -> (jitted, graphed) sequence.
        # The ops in the overall sequence should be the same as Control.
        g.replay()
        # graph_out is now filled with g's result. Use it as ungraphed input.
        out = t_jit(graph_out)
        graph_in.copy_(out)
        g.replay()

        # If replay() updated RNG state correctly, graph_out should now equal eager_out
        self.assertEqual(graph_out, eager_out)

    def _test_batch_norm_impl_index_helper(self, batch, c, hw, affine=True,
                                           track_running_stats=True, train=True,
                                           dtype=torch.float32):
        # enabling inlining to avoid counter increment in BN forward
        torch._C._debug_set_autodiff_subgraph_inlining(True)

        class MyModule(torch.nn.Module):
            def __init__(self, num_features=10, affine=True, track_running_stats=True):
                super(MyModule, self).__init__()
                self.bn = torch.nn.BatchNorm2d(num_features,
                                               1e-5,
                                               affine=affine,
                                               track_running_stats=track_running_stats).to(dtype=dtype)

            def forward(self, x):
                o = self.bn(x)
                o = o * 2.0
                return o

        x = torch.randn(batch, c, hw, hw, dtype=torch.float, device="cuda").to(dtype=dtype).requires_grad_()
        grad = torch.randint(-20, 20, (batch, c, hw, hw), device="cuda").to(dtype=dtype).div(-10)

        my_module = MyModule(c, affine, track_running_stats).cuda()
        ref_module = MyModule(c, affine, track_running_stats).cuda()

        if not train:
            my_module.eval()
            ref_module.eval()

        t_jit = torch.jit.script(my_module)
        ref_module.load_state_dict(my_module.state_dict())

        ref_x = x.detach().requires_grad_()

        for i in range(0, 3):
            jit_o = t_jit(x)
            jit_o.backward(grad)

        # TODO: remove this run?
        o = ref_module(ref_x)
        o.backward(grad)

        has_affine = ref_module.bn.weight is not None
        has_running_stats = ref_module.bn.running_mean is not None

        if has_running_stats:
            my_module.bn.running_mean.zero_()
            my_module.bn.running_var.fill_(1.0)
            ref_module.bn.running_mean.zero_()
            ref_module.bn.running_var.fill_(1.0)

        # Verify that when train is False, we don't have grad for weight/bias.
        if has_affine and train:
            my_module.bn.weight.grad.zero_()
            my_module.bn.bias.grad.zero_()
            ref_module.bn.weight.grad.zero_()
            ref_module.bn.bias.grad.zero_()

        x.grad.zero_()
        ref_x.grad.zero_()

        # real runs
        jit_o = t_jit(x)
        jit_o.backward(grad)

        o = ref_module(ref_x)
        o.backward(grad)

        # assert forward graph fusion
        self.assertGraphContainsExactly(t_jit.graph_for(x), FUSION_GUARD, 1, consider_subgraphs=True)
        # assert backward graph fusion
        bwd_graph = list(
            list(t_jit.get_debug_state().execution_plans.values())[0].code.grad_executor_states()[0]
            .execution_plans.values())[0].graph
        self.assertGraphContainsExactly(bwd_graph, FUSION_GUARD, 1, consider_subgraphs=True)

        if TEST_WITH_ROCM:
            e0 = 1e-3
            e1 = 1e-2
            e2 = 1e-2
        else:
            e0 = 1e-5 if dtype is not torch.half else 1e-3
            e1 = 1e-4 if dtype is not torch.half else 1e-3
            e2 = 1e-3 if dtype is not torch.half else 1e-2

        self.assertTrue(self._compare("comparing output failed", jit_o, o, e0))
        self.assertTrue(self._compare("comparing input grad failed", x.grad, ref_x.grad, e1))
        # TODO: switch to welford and reduce this to 1e-5
        # The 1e-3 looks bad, but we don't have welford in codegen, so numeric
        # is very different between reference and codegen.
        if has_affine and train:
            self.assertTrue(self._compare("comparing weight grad failed",
                                          my_module.bn.weight.grad,
                                          ref_module.bn.weight.grad,
                                          e2))
            self.assertTrue(self._compare("comparing bias grad failed",
                                          my_module.bn.bias.grad,
                                          ref_module.bn.bias.grad,
                                          e1))
        if has_running_stats:
            self.assertTrue(self._compare("comparing running_mean failed",
                                          my_module.bn.running_mean,
                                          ref_module.bn.running_mean,
                                          e0))
            self.assertTrue(self._compare("comparing running_var failed",
                                          my_module.bn.running_var,
                                          ref_module.bn.running_var,
                                          e0))

    @unittest.skipIf(is_pre_volta(), "reduction not supported in pre volta device")
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_batch_norm_half(self):
        with torch.backends.cudnn.flags(enabled=True):
            setups = [
                [True, True],
                [False, False],
                [True, False],
                [False, True]]
            for training_and_track, affine in itertools.product(setups, [True, False]):
                training, track_running_stats = training_and_track
                self._test_batch_norm_impl_index_helper(4, 8, 5, affine, track_running_stats, training, torch.half)

    @unittest.skipIf(is_pre_volta(), "reduction not supported in pre volta device")
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_batch_norm_impl_index_inner_bcast(self):
        # the repro
        self._test_batch_norm_impl_index_helper(2, 1, 1, False, True, True)

        # running the full set
        setups = [
            [True, True],
            [False, False],
            [True, False],
            [False, True]]
        for training_and_track, affine in itertools.product(setups, [True, False]):
            training, track_running_stats = training_and_track
            self._test_batch_norm_impl_index_helper(2, 1, 1, affine, track_running_stats, training)

    @skipIfRocm
    @unittest.skipIf(is_pre_volta(), "reduction not supported in pre volta device")
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_batch_norm_impl_index_correctness(self):
        with torch.backends.cudnn.flags(enabled=True):
            batch = [2, 7, 16]
            channels = [4, 89, 19, 32]
            hw = [1, 8, 17, 32]

            # avoid tolerance failure in CI
            torch.cuda.manual_seed_all(211)

            # failing sizes (2, 1, 1, 1)
            # failing sizes (2, 89, 8, 8) training False, track True, affine: False
            for b, c, hw in itertools.product(batch, channels, hw):
                setups = [
                    [True, True],
                    [False, False],
                    [True, False],
                    [False, True]]
                for training_and_track, affine in itertools.product(setups, [True, False]):
                    training, track_running_stats = training_and_track
                    self._test_batch_norm_impl_index_helper(b, c, hw, affine, track_running_stats, training)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_softplus_fuser(self):
        def shifted_softplus(x: torch.Tensor, shift: float):
            return functional.softplus(x) - shift

        jitted = torch.jit.script(shifted_softplus)
        inp = torch.randn(4, 2, dtype=torch.float32, device="cuda").requires_grad_()
        inp_ref = inp.detach().clone().requires_grad_()
        grad = torch.randn(4, 2, dtype=torch.float32, device="cuda")

        aten_o = shifted_softplus(inp_ref, 0.693147)
        aten_o.backward(grad)
        aten_grad = inp_ref.grad

        for i in range(3):
            jit_o = jitted(inp, 0.693147)
            inp.grad = None         # avoid accumulation on grad
            jit_o.backward(grad)
            jit_grad = inp.grad

        assert torch.allclose(jit_o, aten_o)
        assert torch.allclose(jit_grad, aten_grad)
        self.assertGraphContains(jitted.graph_for(inp, 0.693147), FUSION_GROUP, True)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_inplace_removal(self):
        def t(x: torch.Tensor):
            o = torch.nn.functional.softmax(x, dim=0)
            o += x
            return o.relu_()

        jitted = torch.jit.script(t)
        inp = torch.randn(4, 2, dtype=torch.float32, device="cuda")

        for i in range(3):
            jit_o = jitted(inp)

        graph = jitted.graph_for(inp)
        self.assertGraphContains(graph, FUSION_GROUP, True)
        self.assertGraphContains(graph, 'aten::add', True)
        self.assertGraphContains(graph, 'aten::relu', True)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_conv2d_bias(self):
        def t(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor):
            o = torch.nn.functional.conv2d(x, w, bias)
            return o.relu()

        jitted = torch.jit.script(t)
        inp = torch.randn(4, 5, 3, 3, dtype=torch.float32, device="cuda")
        weight = torch.randn(2, 5, 2, 2, dtype=torch.float32, device="cuda")
        bias = torch.randn(2, dtype=torch.float32, device="cuda")

        for i in range(3):
            jit_o = jitted(inp, weight, bias)

        graph = jitted.graph_for(inp)
        self.assertGraphContains(graph, FUSION_GROUP, True)

        def t_not_fused(x: torch.Tensor, w: torch.Tensor):
            o = torch.nn.functional.conv2d(x, w)
            return o.relu()

        jitted_not_fused = torch.jit.script(t_not_fused)

        for i in range(3):
            jit_o = jitted_not_fused(inp, weight)

        graph = jitted_not_fused.graph_for(inp)
        self.assertGraphContainsExactly(graph, FUSION_GROUP, 0)
        self.assertGraphContains(graph, 'aten::relu', True)

        def t_bias(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor):
            o = torch.nn.functional.conv2d(x, w, bias)
            return o.relu()

        jitted_bias = torch.jit.script(t_bias)

        for i in range(3):
            jit_o = jitted_bias(inp, weight, bias)

        graph = jitted_bias.graph_for(inp)
        self.assertGraphContains(graph, FUSION_GROUP, True)
        self.assertGraphContains(graph, 'prim::add_optional', True)

    @unittest.skipIf(is_pre_volta(), "reduction not supported in pre volta device")
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_remove_output_used_only_in_dtype(self):
        class MyModule(torch.nn.Module):
            def __init__(self, num_features=4):
                super(MyModule, self).__init__()
                self.bn0 = torch.nn.BatchNorm2d(num_features)
                self.bn1 = torch.nn.BatchNorm2d(num_features)

            def forward(self, x, y):
                o1 = self.bn0(x)
                o2 = self.bn1(y)
                return torch.relu(o1 + o2)

        t = MyModule(4).float().cuda()

        jitted = torch.jit.script(t)
        x = torch.randn(3, 4, 2, 5, dtype=torch.float32, device="cuda")
        y = torch.randn(3, 4, 2, 5, dtype=torch.float32, device="cuda")

        with torch.cuda.amp.autocast(True):
            for i in range(5):
                jit_o = jitted(x, y)

            jit_o = jitted(x, y)
            o = t(x, y)

            self.assertTrue(torch.allclose(jit_o, o))
            graph = jitted.graph_for(x, y)
            self.assertGraphContains(graph, FUSION_GROUP, True)

    @unittest.skipIf(is_pre_volta(), "reduction not supported in pre volta device")
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_fix_shape_expression_bn(self):
        class MyModule(torch.nn.Module):
            def __init__(self, num_features=4):
                super(MyModule, self).__init__()
                self.bn = torch.nn.BatchNorm2d(num_features)

            def forward(self, x, y):
                out1 = self.bn(x)
                out2 = out1 + y
                out3 = torch.relu(out2)
                return out3

        t = MyModule(4).float().cuda()

        jitted = torch.jit.script(t)
        x = torch.randn(3, 4, 2, 5, dtype=torch.float32, device="cuda")
        y = torch.randn(3, 4, 2, 5, dtype=torch.float32, device="cuda")

        with torch.cuda.amp.autocast(True):
            for i in range(5):
                jit_o = jitted(x, y)

            jit_o = jitted(x, y)
            o = t(x, y)

            self.assertTrue(torch.allclose(jit_o, o))
            graph = jitted.graph_for(x, y)
            self.assertGraphContains(graph, FUSION_GROUP, True)

    def _run_fwd_helper(self, func, ops, *args):
        jitted = torch.jit.script(func)
        for i in range(3):
            jit_o = jitted(*args)
        jit_o = jitted(*args)
        o = func(*args)
        for oo, jit_oo in zip(o, jit_o):
            self.assertEqual(oo.dtype, jit_oo.dtype)
            self.assertEqual(oo, jit_oo)
        graph = jitted.graph_for(*args)
        self.assertGraphContains(graph, FUSION_GROUP, True)
        for op in ops:
            self.assertGraphContainsExactly(graph, op, 0)

    @unittest.skipIf(is_pre_volta(), "reduction not supported in pre volta device")
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_sibling_fusion(self):
        device = "cuda"
        dtype = torch.float
        x = torch.randn(2, 5, dtype=dtype, device=device)
        y = torch.randn(2, 5, dtype=dtype, device=device)

        def t(x: torch.Tensor):
            o1 = x + 1.0
            o2 = x * 0.5
            return o1, o2
        self._run_fwd_helper(t, ['aten::add', 'aten::mul'], x)

        def t2(x: torch.Tensor, y: torch.Tensor):
            o1 = x.sum(0)
            o2 = (x * y).sum(0)
            return o1, o2
        self._run_fwd_helper(t2, ['aten::sum', 'aten::mul'], x, y)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_clean_profile_ivalue(self):
        device = "cuda"
        dtype = torch.float
        x = torch.randn(2, 5, dtype=dtype, device=device, requires_grad=True)
        # turn on autodiff subgraph inlining
        # this is to verify that we clean up profile_ivalue node out side of
        # fusion code path.
        torch._C._debug_set_autodiff_subgraph_inlining(True)

        def t(x: torch.Tensor, flag: bool):
            return torch.dropout(x, 0.5, flag)

        jit_t = torch.jit.script(t)
        for idx in range(5):
            out = jit_t(x, True)

        graph = jit_t.graph_for(x, True)
        out = jit_t(x, False)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_sibling_fusion_no_scalar_inputs(self):
        device = "cuda"
        dtype = torch.float
        x = torch.randn(2, 5, dtype=dtype, device=device)
        y = torch.randn(3, dtype=dtype, device=device)

        # no tensor dependency between o1/o2, we shouldn't be fusing them
        def t(x: torch.Tensor, y: torch.Tensor):
            o1 = x + 1
            o2 = y - 1
            return o1, o2

        jitted = torch.jit.script(t)
        for i in range(3):
            jit_o = jitted(x, y)
        graph = jitted.graph_for(x, y)
        self.assertGraphContainsExactly(graph, FUSION_GROUP, 0)

    def _bias_view_relu_helper(self, shape, output_shape, dtype, device, error):
        class BiasViewRelu(torch.nn.Module):
            def __init__(self):
                super(BiasViewRelu, self).__init__()
                self.bias = torch.nn.Parameter(torch.randn(shape, dtype=dtype, device=device), requires_grad=False)
                with torch.no_grad():
                    self.bias.fill_(10)

            def forward(self, inputs: torch.Tensor, view_shape: List[int]):
                o = inputs + self.bias
                o = o.view(view_shape)
                return torch.relu(o)

        t = BiasViewRelu()
        x = torch.randn(shape, dtype=dtype, device=device, requires_grad=False)
        t_jit = torch.jit.script(t)

        # profiling
        jit_o = t_jit(x, output_shape)
        # optimization
        jit_o = t_jit(x, output_shape)
        # final
        jit_o = t_jit(x, output_shape)
        # eager - baseline
        o = t(x, output_shape)

        self.assertEqual(o.dtype, jit_o.dtype)
        self.assertTrue(self._compare("comparing output failed", o, jit_o, error))
        graph = t_jit.graph_for(x, output_shape)

        has_inferred_dimension = any([dim == -1 for dim in output_shape])
        if has_inferred_dimension:
            # prohibit fusing when view_shape contains an inferred dimension
            self.assertGraphContainsExactly(graph, FUSION_GROUP, 0)
            self.assertGraphContainsExactly(graph, 'prim::view_copy', 0)
        else:
            self.assertGraphContains(graph, FUSION_GUARD)
            self.assertGraphContains(graph, 'prim::view_copy', True)

    def _alias_bias_view_relu_helper(self, shape, output_shape, dtype, device, error):
        class BiasViewRelu(torch.nn.Module):
            def __init__(self):
                super(BiasViewRelu, self).__init__()
                self.bias = torch.nn.Parameter(torch.randn(shape, dtype=dtype, device=device), requires_grad=False)
                with torch.no_grad():
                    self.bias.fill_(10)

            def forward(self, inputs : torch.Tensor, bias : torch.Tensor, view_shape : List[int]):
                o = inputs.view(view_shape)
                inputs.add_(bias)
                return torch.relu(o)

        t = BiasViewRelu()
        x = torch.randn(shape, dtype=dtype, device=device, requires_grad=False)
        bias = torch.randn(shape, dtype=dtype, device=device, requires_grad=False)
        t_jit = torch.jit.script(t)

        # profiling
        jit_o = t_jit(x.clone(), bias, output_shape)
        # optimization
        jit_o = t_jit(x.clone(), bias, output_shape)
        # final
        jit_o = t_jit(x.clone(), bias, output_shape)
        # eager - baseline
        o = t(x.clone(), bias, output_shape)

        self.assertEqual(o.dtype, jit_o.dtype)
        self.assertTrue(self._compare("comparing output failed", o, jit_o, error))
        graph = t_jit.graph_for(x, bias, output_shape)
        self.assertGraphContainsExactly(graph, FUSION_GUARD, 0)
        self.assertGraphContainsExactly(graph, 'prim::view_copy', 0)

    # generate random view given original view
    def _random_view(self, original_view, max_len=8, max_views=10000):
        class Moves(enum.Enum):
            Merge = 0
            Split = 1
            Broadcast = 2
            ImplicitBroadcast = 3
            Keep = 4

        def valid(old_view, new_view):
            old_view_size = reduce(operator.mul, old_view)
            new_view_size = reduce(operator.mul, new_view)
            return old_view_size == new_view_size

        # given a random starting number, find the nearest divisor
        def find_nearest_divisor(N):
            if 2 >= (N - 1):
                return -1
            result = random.randint(2, N - 1)
            while (N % result) != 0:
                result += 1
            return result

        complete_views = set([tuple(original_view)])

        to_visit = []
        # empty new view, curent originaal view, start pos=0, move count = 0, last_move
        to_visit.append(([], original_view, 0, [], Moves.Keep))

        # depth-first search of view shapes, starting from the original view
        while len(to_visit) > 0 and len(complete_views) < max_views:
            new_view, old_view, odx, move_list, last_move = to_visit[-1]
            to_visit.pop()

            # iterate over each move type
            for idx in range(len(Moves)):
                state = Moves(idx)
                new_view_clone = copy.deepcopy(new_view)
                old_view_clone = copy.deepcopy(old_view)
                new_move_list = move_list + [state]
                new_odx = odx

                # Update state using Move state
                if state == Moves.Keep:
                    new_size = old_view_clone[odx]
                    new_view_clone.append(new_size)
                    new_odx += 1

                elif state == Moves.Merge:
                    if odx + 1 < len(old_view_clone):
                        new_size = old_view_clone[odx] * old_view_clone[odx + 1]
                        new_view_clone.append(new_size)
                        new_odx += 2
                    else:
                        continue

                elif state == Moves.Broadcast and last_move != Moves.Broadcast:
                    new_view_clone.append(1)

                elif state == Moves.Split:
                    new_size = find_nearest_divisor(old_view_clone[odx])
                    if new_size == -1:
                        continue
                    new_view_clone.append(new_size)
                    old_view_clone[odx] = int(old_view[odx] / new_size)

                    if old_view_clone[odx] == 1:
                        new_odx += 1

                elif state == Moves.ImplicitBroadcast:
                    old_view_clone.insert(odx + 1, 1)
                    new_size = old_view[odx] * 1
                    new_view_clone.append(new_size)
                    new_odx += 2

                if new_odx < len(old_view_clone) and len(new_move_list) < max_len:
                    to_visit.append((new_view_clone, old_view_clone, new_odx, new_move_list, state))
                elif (valid(original_view, new_view_clone)):
                    final_new_view = tuple(new_view_clone)
                    complete_views.add(final_new_view)
        return list(complete_views)

    # ndims - number of dimensions
    # test_fn - view test function
    def _view_test_generator(self, ndims, test_fn):
        # create random tensor
        # max value for each dimension
        max_size = 10e7
        max_value = max(int(pow(max_size, 1. / ndims)), 1)
        sizes = [random.randint(1, max_value) for idx in range(ndims)]
        x = torch.randn(sizes)

        original_sizes = list(x.size())
        all_views = self._random_view(original_sizes)
        random.shuffle(all_views)

        max_samples = 20
        max_views = min(len(all_views), max_samples)
        total = 0
        correct = 0
        # test random combinations of compatible views
        for idx in range(max_views):
            for jdx in range(idx + 1, max_views):
                total += 1
                test_fn(all_views[idx], all_views[jdx], torch.float, 'cuda', 1e-6)

    @unittest.skipIf(ALIAS_TEST_DISABLED, "skipping this test since view is disabled now")
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_view(self):
        torch._C._jit_set_nvfuser_guard_mode(True)
        self._bias_view_relu_helper([2, 3, 4, 5], [-1, 4, 5], torch.float, 'cuda', 1e-6)
        for ndims in range(1, 5):
            self._view_test_generator(ndims, self._bias_view_relu_helper)
        self._alias_bias_view_relu_helper([2, 3, 4, 5], [1, 6, 1, 2, 2, 5, 1], torch.float, 'cuda', 1e-6)

    def _bias_flatten_relu_helper(self, shape, start_dim, end_dim, dtype, device, error):
        class BiasFlattenRelu(torch.nn.Module):
            def __init__(self):
                super(BiasFlattenRelu, self).__init__()
                self.bias = torch.nn.Parameter(torch.randn(shape, dtype=dtype, device=device), requires_grad=False)
                with torch.no_grad():
                    self.bias.fill_(10)

            def forward(self, inputs : torch.Tensor, start_dim : int, end_dim : int):
                o = inputs + self.bias
                o = o.flatten(start_dim, end_dim)
                return torch.relu(o)

        t = BiasFlattenRelu()
        x = torch.randn(shape, dtype=dtype, device=device, requires_grad=False)
        t_jit = torch.jit.script(t)

        self._run_helper(t_jit, t, x, start_dim, end_dim)
        self.assertGraphContains(t_jit.graph_for(x, start_dim, end_dim), 'prim::flatten_copy', True)

    def _alias_bias_flatten_relu_helper(self, shape, start_dim, end_dim, dtype, device, error):
        class BiasFlattenRelu(torch.nn.Module):
            def __init__(self):
                super(BiasFlattenRelu, self).__init__()
                self.bias = torch.nn.Parameter(torch.randn(shape, dtype=dtype, device=device), requires_grad=False)
                with torch.no_grad():
                    self.bias.fill_(10)

            def forward(self, inputs : torch.Tensor, bias : torch.Tensor, start_dim : int, end_dim : int):
                o = inputs.flatten(start_dim, end_dim)
                inputs.add_(bias)
                return torch.relu(o)

        t = BiasFlattenRelu()
        x = torch.randn(shape, dtype=dtype, device=device, requires_grad=False)
        bias = torch.randn(shape, dtype=dtype, device=device, requires_grad=False)
        t_jit = torch.jit.script(t)

        # profiling
        jit_o = t_jit(x.clone(), bias, start_dim, end_dim)
        # optimization
        jit_o = t_jit(x.clone(), bias, start_dim, end_dim)
        # final
        jit_o = t_jit(x.clone(), bias, start_dim, end_dim)
        # eager - baseline
        o = t(x.clone(), bias, start_dim, end_dim)

        self.assertEqual(o.dtype, jit_o.dtype)
        self.assertTrue(self._compare("comparing output failed", o, jit_o, error))
        graph = t_jit.graph_for(x, bias, start_dim, end_dim)

        self.assertGraphContainsExactly(graph, FUSION_GUARD, 0)
        self.assertGraphContainsExactly(graph, 'prim::flatten_copy', 0)

    @unittest.skipIf(ALIAS_TEST_DISABLED, "skipping this test since flatten is disabled now")
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_flatten(self):
        torch._C._jit_set_nvfuser_guard_mode(True)
        self._bias_flatten_relu_helper([2, 3, 4, 5], 0, -1, torch.float, 'cuda', 1e-6)
        self._bias_flatten_relu_helper([2, 3, 4, 5], 1, -1, torch.float, 'cuda', 1e-6)
        self._bias_flatten_relu_helper([2, 3, 4, 5], 2, -1, torch.float, 'cuda', 1e-6)
        self._bias_flatten_relu_helper([2, 3, 4, 5], 0, 3, torch.float, 'cuda', 1e-6)
        self._bias_flatten_relu_helper([2, 3, 4, 5], 1, 2, torch.float, 'cuda', 1e-6)
        self._bias_flatten_relu_helper([2, 3, 4, 5], 2, 2, torch.float, 'cuda', 1e-6)
        self._alias_bias_flatten_relu_helper([2, 3, 4, 5], 0, -1, torch.float, 'cuda', 1e-6)
        self._alias_bias_flatten_relu_helper([2, 3, 4, 5], 1, -1, torch.float, 'cuda', 1e-6)
        self._alias_bias_flatten_relu_helper([2, 3, 4, 5], 2, -1, torch.float, 'cuda', 1e-6)
        self._alias_bias_flatten_relu_helper([2, 3, 4, 5], 0, 3, torch.float, 'cuda', 1e-6)
        self._alias_bias_flatten_relu_helper([2, 3, 4, 5], 1, 2, torch.float, 'cuda', 1e-6)
        self._alias_bias_flatten_relu_helper([2, 3, 4, 5], 2, 2, torch.float, 'cuda', 1e-6)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_strict_fusion(self):
        def success(x):
            with torch.jit.strict_fusion():
                return x + x + x

        scripted = self.checkScript(success, (torch.rand([4], device='cuda'),))
        g = torch.jit.last_executed_optimized_graph()
        FileCheck().check_not("aten::add").check("prim::CudaFusionGroup").run(g)

        def failure(x):
            with torch.jit.strict_fusion():
                return x + torch.mm(x, x) + x

        with self.assertRaises(Exception) as error_out:
            foo_s = torch.jit.script(failure)
            foo_s(torch.rand([4, 4]))
            foo_s(torch.rand([4, 4]))

        fc = FileCheck().check("Found unfused operators")
        fc.check("aten::mm").run(str(error_out.exception))

    def _ltc_helper(self, shape, dtype, device, error, approximate=True):
        # modeled after LTC linear layer
        class LTC(torch.nn.Module):
            def __init__(self):
                super(LTC, self).__init__()
                self.weight = torch.nn.Parameter(torch.randn([1024, 1024], dtype=dtype, device=device), requires_grad=False)
                self.bias = torch.nn.Parameter(torch.randn([1, 1024], dtype=dtype, device=device), requires_grad=False)

            def forward(self, inputs : torch.Tensor):
                o = inputs.view([32768, 1024])
                o = torch.mm(o, self.weight)
                o = o.view([256, 128, 1024])
                o = o + self.bias
                o = o.view([32768, 1024])
                o = o.view([256, 128, 1024])
                return torch.nn.functional.gelu(o)

        t = LTC()
        x = torch.randn(shape, dtype=dtype, device=device, requires_grad=False)
        t_jit = torch.jit.script(t)

        # profile/optimization runs
        for i in range(3):
            jit_o = t_jit(x)
        o = t(x)

        self.assertEqual(o.dtype, jit_o.dtype)
        self.assertTrue(self._compare("comparing output failed", o, jit_o, error))
        graph = t_jit.graph_for(x)
        self.assertGraphContains(graph, FUSION_GUARD)
        self.assertGraphContains(graph, 'prim::view_copy', True)

    @unittest.skipIf(ALIAS_TEST_DISABLED, "skipping this test since view is disabled now")
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_nested_view(self):
        self._ltc_helper([256, 128, 1024], torch.float, 'cuda', 1e-6)

    def _bias_squeeze_relu_helper(self, shape, dtype, device, error):
        class BiasSqueezeRelu(torch.nn.Module):
            def __init__(self):
                super(BiasSqueezeRelu, self).__init__()

            def forward(self, inputs: torch.Tensor, bias: torch.Tensor):
                o = inputs + bias
                o = torch.squeeze(o)
                return torch.relu(o)

        t = BiasSqueezeRelu()
        x = torch.randn(shape, dtype=dtype, device=device, requires_grad=False)
        bias = torch.randn(shape, dtype=dtype, device=device, requires_grad=False)
        t_jit = torch.jit.script(t)

        jit_o = t_jit(x, bias)
        jit_o = t_jit(x, bias)
        jit_o = t_jit(x, bias)
        o = t(x, bias)

        self.assertEqual(o.dtype, jit_o.dtype)
        self.assertTrue(self._compare("comparing output failed", o, jit_o, error))
        graph = t_jit.graph_for(x, bias)
        self.assertGraphContains(graph, FUSION_GUARD)
        self.assertGraphContains(graph, 'prim::squeeze_copy', True)

    def _alias_bias_squeeze_relu_helper(self, shape, dtype, device, error):
        class BiasSqueezeRelu(torch.nn.Module):
            def __init__(self):
                super(BiasSqueezeRelu, self).__init__()

            def forward(self, inputs: torch.Tensor, bias: torch.Tensor):
                o = torch.squeeze(inputs)
                inputs.add_(bias)
                return torch.relu(o)

        t = BiasSqueezeRelu()
        x = torch.randn(shape, dtype=dtype, device=device, requires_grad=False)
        bias = torch.randn(shape, dtype=dtype, device=device, requires_grad=False)
        t_jit = torch.jit.script(t)

        jit_o = t_jit(x.clone(), bias)
        jit_o = t_jit(x.clone(), bias)
        jit_o = t_jit(x.clone(), bias)
        o = t(x.clone(), bias)

        self.assertEqual(o.dtype, jit_o.dtype)
        self.assertTrue(self._compare("comparing output failed", o, jit_o, error))
        graph = t_jit.graph_for(x, bias)
        self.assertGraphContainsExactly(graph, FUSION_GUARD, 0)
        self.assertGraphContainsExactly(graph, 'prim::squeeze_copy', 0)

    @unittest.skipIf(ALIAS_TEST_DISABLED, "skipping this test since squeeze/unsqueeze is disabled now")
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_squeeze(self):
        self._bias_squeeze_relu_helper([1, 6, 1, 2, 2, 5, 1], torch.float, 'cuda', 1e-6)
        self._alias_bias_squeeze_relu_helper([1, 6, 1, 2, 2, 5, 1], torch.float, 'cuda', 1e-6)

    @unittest.skipIf(ALIAS_TEST_DISABLED, "skipping this test since squeeze/unsqueeze is disabled now")
    # remove this after opinfo tests are enabled
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_squeeze_zero(self):
        x = torch.tensor(1.0, dtype=torch.float, device="cuda")

        def squeeze_0(x: torch.Tensor):
            o = x + 1.
            o = torch.squeeze(o, 0)
            o = o * 2.
            return o

        def squeeze_1(x: torch.Tensor):
            o = x + 1.
            o = torch.squeeze(o, -1)
            o = o + .5
            return o

        squeeze_0_jit = torch.jit.script(squeeze_0)
        self._run_helper(squeeze_0_jit, squeeze_0, x)
        squeeze_1_jit = torch.jit.script(squeeze_1)
        self._run_helper(squeeze_1_jit, squeeze_1, x)

    def _bias_unsqueeze_relu_helper(self, shape, dtype, device, error):
        class BiasUnsqueezeRelu(torch.nn.Module):
            def __init__(self):
                super(BiasUnsqueezeRelu, self).__init__()

            def forward(self, inputs: torch.Tensor, bias: torch.Tensor):
                o = inputs + bias
                o = torch.unsqueeze(o, 0)
                return torch.relu(o)

        t = BiasUnsqueezeRelu()
        x = torch.randn(shape, dtype=dtype, device=device, requires_grad=False)
        bias = torch.randn(shape, dtype=dtype, device=device, requires_grad=False)
        t_jit = torch.jit.script(t)

        jit_o = t_jit(x, bias)
        jit_o = t_jit(x, bias)
        jit_o = t_jit(x, bias)
        o = t(x, bias)

        self.assertEqual(o.dtype, jit_o.dtype)
        self.assertTrue(self._compare("comparing output failed", o, jit_o, error))
        graph = t_jit.graph_for(x, bias)
        self.assertGraphContains(graph, FUSION_GUARD)
        self.assertGraphContains(graph, 'prim::unsqueeze_copy', True)

    def _alias_bias_unsqueeze_relu_helper(self, shape, dtype, device, error):
        class BiasUnsqueezeRelu(torch.nn.Module):
            def __init__(self):
                super(BiasUnsqueezeRelu, self).__init__()

            def forward(self, inputs : torch.Tensor, bias : torch.Tensor):
                o = torch.unsqueeze(inputs, 0)
                inputs.add_(bias)
                return torch.relu(o)

        t = BiasUnsqueezeRelu()
        x = torch.randn(shape, dtype=dtype, device=device, requires_grad=False)
        bias = torch.randn(shape, dtype=dtype, device=device, requires_grad=False)
        t_jit = torch.jit.script(t)

        jit_o = t_jit(x.clone(), bias)
        jit_o = t_jit(x.clone(), bias)
        jit_o = t_jit(x.clone(), bias)
        o = t(x.clone(), bias)

        self.assertEqual(o.dtype, jit_o.dtype)
        self.assertTrue(self._compare("comparing output failed", o, jit_o, error))
        graph = t_jit.graph_for(x, bias)
        self.assertGraphContainsExactly(graph, FUSION_GUARD, 0)
        self.assertGraphContainsExactly(graph, 'prim::unsqueeze_copy', 0)

    @unittest.skipIf(ALIAS_TEST_DISABLED, "skipping this test since squeeze/unsqueeze is disabled now")
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_unsqueeze(self):
        self._bias_unsqueeze_relu_helper([2, 3, 4, 5], torch.float, 'cuda', 1e-6)
        self._alias_bias_unsqueeze_relu_helper([2, 3, 4, 5], torch.float, 'cuda', 1e-6)

    @unittest.skipIf(ALIAS_TEST_DISABLED, "skipping this test since unsqueeze is disabled now")
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_alias_pass_fix(self):
        x = torch.randn(4, 24, 2, 2, dtype=torch.float, device="cuda")
        w = torch.randn(24, 24, 1, 1, dtype=torch.float, device="cuda")
        b = torch.randn(24, dtype=torch.float, device="cuda")

        def t(x, w, b):
            b2 = b + 1.0
            o = torch.conv2d(x, w, b2)
            return o

        t_jit = torch.jit.script(t)
        self._run_helper(t_jit, t, x, w, b)

    @unittest.skipIf(ALIAS_TEST_DISABLED, "skipping this test since squeeze/unsqueeze is disabled now")
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_squeeze_negative_dim(self):
        x = torch.randn(4, 24, 1, 2, dtype=torch.float, device="cuda")

        def t(x):
            o = x + 1.0
            o = o.squeeze(-2)
            o = o * 2.0
            return o

        t_jit = torch.jit.script(t)
        self._run_helper(t_jit, t, x)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_singleton_fusion(self):
        x = torch.randn(4, 2, device="cuda")

        with nvfuser_singleton_fusion(True):
            def t(x):
                return x.relu()

            t_jit = torch.jit.script(t)
            self._run_helper(t_jit, t, x)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_issue1445_fusion(self):
        def f(t0, t1, t2, t3):
            masked_input = torch.where(t1, t2, t3)
            total = masked_input.sum([0, 1, 2, 3])
            sizes : List[int] = []
            t10 = torch.reshape(t0, sizes)
            t7 = total / t10
            t4 = t7.to(dtype=torch.float)
            return t4

        x = torch.randn(1, 1, 1, 1, device='cuda').to(dtype=torch.long)
        y = torch.randn(3, 2, 1, 1, device='cuda').to(dtype=torch.bool).expand([3, 2, 1, 2])
        z = torch.randn(3, 2, 1, 2, device='cuda')
        w = torch.tensor(1.5, device='cuda')

        f_jit = torch.jit.script(f)
        for i in range(5):
            out_jit = f_jit(x, y, z, w)
        out = f(x, y, z, w)
        self.assertEqual(out, out_jit)
        self.assertGraphContainsExactly(f_jit.graph_for(x, y, z, w), FUSION_GROUP, 1)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_disable_sibling_fuse(self):
        x = torch.randn(4, 2, device="cuda")
        y = torch.randn(8, device="cuda")
        s = torch.tensor(1.5, device="cuda")

        with nvfuser_horizontal_fusion(False):
            def t(x, y, s):
                o1 = x + s
                o2 = y + s
                return o1, o2

            t_jit = torch.jit.script(t)
            for i in range(5):
                t_jit(x, y, s)

            # sibling fusion should be disabled with the flag
            self.assertGraphContainsExactly(t_jit.graph_for(x, y, s), FUSION_GUARD, 0)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_build_shape_expression_native_dropout(self):
        x = torch.randn(4, 2, device="cuda")

        def t(x):
            o, mask = torch.native_dropout(x, 0.0, True)
            o1 = o.sigmoid()
            o2 = mask.float().sigmoid()
            return (o1, o2)

        t_jit = torch.jit.script(t)

        jit_o = t_jit(x)
        jit_o = t_jit(x)
        o = t(x)
        for oo, jit_oo in zip(o, jit_o):
            self.assertEqual(oo.dtype, jit_oo.dtype)
            self.assertEqual(oo, jit_oo)
        self.assertGraphContains(t_jit.graph_for(x), FUSION_GUARD)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_scalar_tensor_permuted(self):
        x = torch.randn(4, 2, 3, device="cuda").permute([1, 2, 0])
        y = torch.tensor(1.0, device="cuda")

        with nvfuser_singleton_fusion(True):
            def t(x, y):
                return x + y

            t_jit = torch.jit.script(t)
            self._run_helper(t_jit, t, x, y)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_cpu_scalar(self):
        x = torch.randn(4, 2, 3, device="cuda")
        y = torch.tensor(1.0, device="cpu")
        z = torch.tensor(2.0, device="cpu")

        with nvfuser_singleton_fusion(True):
            # testing cpu scalar tensor promotion
            def t(x, y):
                return x + y

            t_jit = torch.jit.script(t)
            self._run_helper(t_jit, t, x, y)

            # scalar cpu tensor add should NOT be fused
            @torch.jit.script
            def t1(y, z):
                return y * z
            for _ in range(5):
                t1(y, z)
            self.assertGraphContainsExactly(t1.graph_for(y, z), FUSION_GUARD, 0)

            # everything, including scalar cpu tensor add should be fused
            @torch.jit.script
            def t2(x, y, z):
                tmp = y + z
                return tmp + x
            for _ in range(5):
                t2(x, y, z)
            self.assertGraphContainsExactly(t2.graph_for(x, y, z), 'aten::add', 0)
            self.assertGraphContainsExactly(t2.graph_for(x, y, z), FUSION_GUARD, 1)

            # 'cpu_tmp = y + z' shouldn't be fused.
            @torch.jit.script
            def t3(x, y, z):
                cpu_tmp = y + z
                out = x + y
                return cpu_tmp, out
            for _ in range(5):
                t3(x, y, z)
            self.assertGraphContainsExactly(t3.graph_for(x, y, z), FUSION_GUARD, 1)
            self.assertGraphContainsExactly(t3.graph_for(x, y, z), 'aten::add', 1)

    @unittest.skipIf(ALIAS_TEST_DISABLED, "skipping this test since squeeze/unsqueeze is disabled now")
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_shape_expression(self):
        x = torch.randn(4, 2, 1, 3, device="cuda")

        def t_unsqueeze(x):
            t0 = x.relu()
            t1 = t0.unsqueeze(1)
            t2 = t1 + 1.0
            t3 = t1.size()
            return t2, t3

        def t_squeeze(x):
            t0 = x.relu()
            t1 = t0.squeeze()
            t2 = t1 + 1.0
            t3 = t1.size()
            return t2, t3

        def t_squeeze_dim(x):
            t0 = x.relu()
            t1 = t0.squeeze(-2)
            t2 = t1 + 1.0
            t3 = t1.size()
            return t2, t3

        # squeezing a non-size 1 dimension should be a no op
        def t_squeeze_dim_no_op(x):
            t0 = x.relu()
            t1 = t0.squeeze(1)
            t2 = t1 + 1.0
            t3 = t1.size()
            return t2, t3

        def run(fn):
            jit_fn = torch.jit.script(fn)
            jit_o = jit_fn(x)
            jit_o = jit_fn(x)
            jit_o = jit_fn(x)
            o = fn(x)
            # output 0 is a tensor, so we check dtype and value
            self.assertEqual(o[0].dtype, jit_o[0].dtype)
            self.assertEqual(o[0], jit_o[0])
            # output 1 is shape
            self.assertEqual(o[1], jit_o[1])
            self.assertGraphContainsExactly(jit_fn.graph_for(x), FUSION_GUARD, 1)

        for t in [t_unsqueeze, t_squeeze, t_squeeze_dim, t_squeeze_dim_no_op]:
            run(t)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_scalar_cuda_tensor(self):
        x = torch.tensor(2.0, device="cuda")

        with nvfuser_singleton_fusion(True):
            def t(x):
                return x + 1.0

            t_jit = torch.jit.script(t)
            self._run_helper(t_jit, t, x)

            @torch.jit.script
            def t_jitted(x):
                return x.sum(0)

            for i in range(5):
                t_jitted(x)
            self.assertGraphContainsExactly(t_jitted.graph_for(x), FUSION_GUARD, 0)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_overlapped_input(self):
        x = torch.randn(8, device="cuda").as_strided((2, 4), (1, 1))

        with nvfuser_singleton_fusion(True):
            def t(x):
                return x + 1.0

            t_jit = torch.jit.script(t)
            self._run_helper(t_jit, t, x)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    @unittest.skipIf(is_pre_volta(), "reduction not supported in pre volta device")
    def test_reduction_empty_axes(self):
        x = torch.randn(4, 2, 3, device="cuda").permute([1, 2, 0])

        with nvfuser_singleton_fusion(True):
            def t(x):
                sizes : List[int] = []
                return x.sum(sizes)

            t_jit = torch.jit.script(t)
            self._run_helper(t_jit, t, x)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    @unittest.skipIf(is_pre_volta(), "reduction not supported in pre volta device")
    def test_int_tensor_input(self):
        x = torch.randn(4, 2, device="cuda").to(dtype=torch.int)

        with nvfuser_singleton_fusion(True):
            def t(x):
                return x.amax(dim=0)

            t_jit = torch.jit.script(t)
            self._run_helper(t_jit, t, x)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_to_boolean(self):
        x = torch.randn(4, 2, device="cuda")

        with nvfuser_singleton_fusion(True):
            def t(x):
                return x.to(dtype=torch.bool)

            t_jit = torch.jit.script(t)
            self._run_helper(t_jit, t, x)


    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_to_copy(self):
        x = torch.randn(4, 2, device="cuda")

        with nvfuser_singleton_fusion(True):
            def t(x, dtype : torch.dtype):
                o = torch.ops.aten._to_copy(x, dtype=dtype)
                return o

            t.__disable_jit_function_caching__ = True

            t_jit = torch.jit.script(t)
            for dtype in [torch.float16, torch.bool, torch.float64]:
                self._run_helper(t_jit, t, x, dtype)

            def t_none(x):
                with torch.jit.strict_fusion():
                    o = torch.ops.aten._to_copy(x, dtype=None)
                return o

            t_jit_none = torch.jit.script(t_none)
            self._run_helper(t_jit_none, t_none, x)


    @unittest.skipIf(ALIAS_TEST_DISABLED, "skipping this test since reshape is disabled now")
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_view_copy_graph_guard(self):
        x = torch.randn(4, 2, 3, device="cuda").permute([1, 2, 0])
        y = [4, 6]

        with nvfuser_singleton_fusion(True):
            def t(x, y : List[int]):
                t1 = x + 1.0
                t2 = t1 * 1.0
                out = t2.reshape(y)
                return out.relu()

            t_jit = torch.jit.script(t)
            self._run_helper(t_jit, t, x, y)

    @unittest.skipIf(ALIAS_TEST_DISABLED, "skipping this test since view is disabled now")
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_view_copy_graph_guard_double_fusion(self):
        x = torch.randn(2, 2, 5, device="cuda")
        w = torch.randn(5, 5, device="cuda")

        with nvfuser_singleton_fusion(True):
            def t(x, w):
                o = x.view([4, x.size()[-1]])
                o = torch.matmul(o, w)
                o = o.view([2, 2, o.size()[1]])
                return o

            t_jit = torch.jit.script(t)
            for i in range(3):
                jit_o = t_jit(x, w)
            o = t(x, w)
            self.assertEqual(jit_o, o)
            self.assertGraphContainsExactly(t_jit.graph_for(x, w), FUSION_GUARD, 2, consider_subgraphs=True)

    @skipIfRocm
    # see issue here on why we disabled this test https://github.com/csarofeen/pytorch/issues/2127
    @unittest.skipIf(is_pre_volta(), "permutation scheduling can be dangerous on pre-volta device")
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_view_before_permute(self):
        view_examples = [[[1, 19, 1, 12, 7, 1, 99], [1, 19, 1, 3, 2772]],
                         [[3, 17, 80, 1], [51, 1, 2, 4, 10]],
                         [[3, 17, 80, 1, 9], [51, 1, 2, 4, 10, 9]],
                         [[2, 3, 4, 5], [1, 6, 1, 2, 2, 5]],
                         [[22, 22, 2], [22, 11, 1, 1, 4]],
                         [[37, 9, 7, 6, 10], [333, 2, 2, 3, 35]],
                         [[8, 1, 1, 8, 1, 8], [8, 2, 4, 1, 8]],
                         [[1, 333, 1], [1, 37, 9]],
                         [[1, 333], [1, 1, 1, 111, 1, 3]],
                         [[1, 27454, 1, 2], [1, 7844, 1, 7]],
                         [[1, 7844, 1, 7], [1, 27454, 2]]]

        def _getTransposeAxes(sizes):
            # broadcast do not change
            # always move inner-most dim
            # random permutation of other dims
            result = []
            valid_sizes = []
            for idx, val in enumerate(sizes):
                if val > 1 and idx < len(sizes) - 1:
                    valid_sizes.append((idx, val))
                result.append(idx)
            idx, new_size = valid_sizes[random.randint(0, len(valid_sizes) - 1)]
            result[idx] = len(sizes) - 1
            result[len(sizes) - 1] = idx
            return result

        def _transposeSize(sizes, dims):
            return [sizes[old_pos] for old_pos in dims]

        for example in view_examples:
            before_view_size, after_view_size = example
            axes = _getTransposeAxes(after_view_size)
            output_size = _transposeSize(after_view_size, axes)
            self._view_before_permute_helper(before_view_size, after_view_size, output_size, axes)

    def _view_before_permute_helper(self, input_shape, view_shape, output_shape, dims):
        def t(x, y, view_shape : List[int], dims : List[int]):
            x_v = x.view(view_shape)
            x_t = torch.permute(x_v, dims)
            o = torch.add(x_t, y)
            o = torch.relu(o)
            return o

        x = torch.randn(*input_shape, device="cuda")
        y = torch.randn(*output_shape, device="cuda")
        t_jit = torch.jit.script(t)
        self._run_helper(t_jit, t, x, y, view_shape, dims)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_permute(self):
        max_dims = 4
        for ndims in range(2, max_dims + 1):
            shape = [idx + 2 for idx in range(ndims)]
            for dims in itertools.permutations(range(ndims)):
                self._permute_helper(shape, dims)

    def _permute_helper(self, shape, dims):
        def t(x, y, dims : List[int]):
            x_t = torch.permute(x, dims)
            y_t = torch.permute(y, dims)
            o = torch.add(x_t, y_t)
            o = torch.relu(o)
            return o

        x = torch.randn(*shape, device="cuda")
        y = torch.randn(*shape, device="cuda")
        t_jit = torch.jit.script(t)
        self._run_helper(t_jit, t, x, y, dims)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_transpose(self):
        max_dims = 4
        for ndims in range(2, max_dims + 1):
            shape = [idx + 2 for idx in range(ndims)]
            for idx in range(1, ndims):
                for jdx in range(idx):
                    self._transpose_helper(shape, idx, jdx)

    def _transpose_helper(self, shape, dim0, dim1):
        def t(x, y, dim0 : int, dim1 : int):
            x_t = torch.transpose(x, dim0, dim1)
            y_t = torch.transpose(y, dim0, dim1)
            o = torch.add(x_t, y_t)
            o = torch.nn.functional.gelu(o)
            return o

        x = torch.randn(*shape, device="cuda")
        y = torch.randn(*shape, device="cuda")
        t_jit = torch.jit.script(t)
        self._run_helper(t_jit, t, x, y, dim0, dim1)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_transpose_default(self):
        def t(x, y):
            x_t = torch.t(x)
            y_t = torch.t(y)
            o = torch.add(x_t, y_t)
            o = torch.nn.functional.gelu(o)
            return o

        x = torch.randn(3, 5, device="cuda")
        y = torch.randn(3, 5, device="cuda")
        t_jit = torch.jit.script(t)
        self._run_helper(t_jit, t, x, y)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_input_output_passthrough(self):
        def t(t0, t1, t2):
            mask = t1.to(dtype=torch.bool)
            masked_input = torch.where(t0, mask, t2)
            return masked_input, mask

        t_jit = torch.jit.script(t)
        # stick to integers, this avoid the numerical difference due to our
        # promotion
        x = torch.randn(4, 4, device='cuda').to(dtype=torch.bool)
        y = torch.randn(4, 4, device='cuda').to(dtype=torch.bool)
        z = torch.tensor(1.0, device='cuda').to(dtype=torch.bool)
        jit_o = t_jit(x, y, z)
        jit_o = t_jit(x, y, z)
        o = t(x, y, z)
        for oo, jit_oo in zip(o, jit_o):
            self.assertEqual(oo.dtype, jit_oo.dtype)
            self.assertEqual(oo, jit_oo)
        self.assertGraphContains(t_jit.graph_for(x, y, z), FUSION_GUARD)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_pointwise_reference_tensor(self):
        def t(input1, input2, scalar):
            _unsafe_view = torch.ops.aten._unsafe_view(input1, [2, 4, 16])
            add_ = torch.ops.aten.add_(_unsafe_view, input2)
            gelu_ = torch.ops.aten.gelu(add_)
            view_ = torch.ops.aten.view(gelu_, [8, 16])
            mul_ = torch.ops.aten.mul(add_, scalar)
            return [view_, mul_]

        x = torch.randn(8, 16, device="cuda")
        bias = torch.randn(16, device="cuda")
        scalar = torch.ones(torch.Size([]), device="cuda")

        t_jit = torch.jit.script(t)
        for i in range(3):
            jit_o = t_jit(x, bias, scalar)
        o = t(x, bias, scalar)
        self.assertEqual(jit_o, o)
        self.assertGraphContains(t_jit.graph_for(x, bias, scalar), FUSION_GUARD)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    @unittest.skipIf(is_pre_volta(), "reduction not supported in pre volta device")
    def test_native_batch_norm_backward(self):
        grad_output = torch.randn(4, 2, 3, device="cuda")
        input = torch.randn(4, 2, 3, device="cuda")
        weight = torch.randn(2, device="cuda")

        r_m = torch.randn(2, device="cuda")
        r_v = torch.randn(2, device="cuda").abs()

        save_mean = torch.randn(2, device="cuda")
        save_invstd = torch.randn(2, device="cuda").abs()

        with nvfuser_singleton_fusion(True):
            def t(grad_out, input, weight, r_m, r_v, save_mean, save_invstd, train: bool, eps: float, mask: List[bool]):
                return torch.ops.aten.native_batch_norm_backward(grad_out, input, weight, r_m, r_v, save_mean,
                                                                 save_invstd, train, eps, mask)

            t_jit = torch.jit.script(t)
            for i in range(4):
                jit_o = t_jit(grad_output, input, weight, r_m.clone(), r_v.clone(),
                              save_mean, save_invstd, True, 1e-5, [True, True, True])

            ref_m = r_m.clone()
            ref_v = r_v.clone()
            jit_o = t_jit(grad_output, input, weight, r_m, r_v, save_mean, save_invstd, True, 1e-5, [True, True, True])
            o = t(grad_output, input, weight, ref_m, ref_v, save_mean, save_invstd, True, 1e-5, [True, True, True])
            for oo, jit_oo in zip(o, jit_o):
                self.assertEqual(oo.dtype, jit_oo.dtype)
                self.assertEqual(oo, jit_oo)
            self.assertEqual(ref_m.dtype, r_m.dtype)
            self.assertEqual(ref_m, r_m)
            self.assertEqual(ref_v.dtype, r_v.dtype)
            self.assertEqual(ref_v, r_v)
            self.assertGraphContains(t_jit.graph_for(grad_output, input, weight, r_m.clone(), r_v.clone, save_mean,
                                                     save_invstd, True, 1e-5, [True, True, True]), FUSION_GUARD)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_contiguous_on_broadcasted(self):
        x = torch.randn(4, 1, device="cuda")
        y = torch.randn(4, 128, device="cuda")

        with nvfuser_singleton_fusion(True):
            def t(x, y):
                t1 = x.expand([4, 128])
                t2 = t1 * y
                return t2

            t_jit = torch.jit.script(t)
            self._run_helper(t_jit, t, x, y)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_skip_parser(self):
        x = torch.randn(4, 12, device="cuda")

        with nvfuser_singleton_fusion(True):
            def fn(x):
                t1 = x + 1.0
                return t1.relu()

            fn_jit = torch.jit.script(fn)
            self._run_helper(fn_jit, fn, x)

            # add node should have been merged into fusion
            self.assertGraphContains(fn_jit.graph_for(x), FUSION_GUARD)
            self.assertGraphContainsExactly(fn_jit.graph_for(x), 'aten::add', 0)

            # flips skip parse for `aten::add`, following fusion should skip the
            # add node
            self.assertFalse(torch._C._jit_set_nvfuser_skip_node_kind("aten::add", True))

            def fn_1(x):
                t1 = x + 2.0  # change const value so we'll not reuse plan
                return t1.relu()

            fn_1_jit = torch.jit.script(fn_1)
            self._run_helper(fn_1_jit, fn_1, x)

            # add node should have been merged into fusion
            self.assertGraphContains(fn_1_jit.graph_for(x), FUSION_GUARD)
            self.assertGraphContainsExactly(fn_1_jit.graph_for(x), 'aten::add', 1)

            # flips skip parse for `aten::add`, next fusion should fuse add node
            self.assertTrue(torch._C._jit_set_nvfuser_skip_node_kind("aten::add", True))

            def fn_2(x):
                t1 = x + 2.0  # change const value so we'll not reuse plan
                return t1.relu()

            fn_2_jit = torch.jit.script(fn_2)
            self._run_helper(fn_2_jit, fn_2, x)

            # add node should have been merged into fusion
            self.assertGraphContains(fn_2_jit.graph_for(x), FUSION_GUARD)
            self.assertGraphContainsExactly(fn_2_jit.graph_for(x), 'aten::add', 0)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_cuda_fusion_guard(self):
        old_guard = torch._C._jit_set_nvfuser_guard_mode(True)

        class ConvModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x.sin().sigmoid()

        mod = ConvModule().to(device="cuda")

        inputs = [torch.randn(20, 16, 50, 100, device="cuda", requires_grad=True)]

        def reduce_scalar(temp):
            return temp.sum()

        scripted = torch.jit.script(mod)
        with torch.no_grad():
            scripted(*inputs)
        res = scripted(*inputs)
        reduce_scalar(res).backward()
        torch._C._jit_set_nvfuser_guard_mode(old_guard)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_nvfuser_comparison_callbacks_with_fallback(self):
        try:
            fused_result = None
            unfused_result = None
            graph_ir = None

            def callback(fused_outputs, unfused_outputs, graph_str):
                nonlocal unfused_result
                nonlocal fused_result
                nonlocal graph_ir
                unfused_result = unfused_outputs[-1]
                fused_result = fused_outputs[-1]
                graph_ir = graph_str
            torch._C._jit_nvfuser_set_comparison_callback(True, callback)

            def fn(x, y):
                z = torch.add(x, y)
                return torch.relu(z)

            x = torch.rand((4, 4)).cuda() - 0.5
            y = torch.rand((4, 4)).cuda() - 0.5

            fn_s = torch.jit.script(fn)
            fn_s(x, y)
            fn_s(x, y)
            fn_s(x, y)

            expected = fn(x, y)

            self.assertEqual(expected, fused_result)
            self.assertEqual(expected, unfused_result)
            FileCheck().check("aten::add").run(graph_ir)
        finally:
            torch._C._jit_nvfuser_clear_comparison_callback()

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_nvfuser_comparison_callbacks_without_fallback(self):
        try:
            fused_result = None
            unfused_result = None
            graph_ir = None

            def callback(fused_outputs, unfused_outputs, graph_str):
                nonlocal unfused_result
                nonlocal fused_result
                nonlocal graph_ir
                if len(unfused_outputs) > 0:
                    unfused_result = unfused_outputs[-1]
                fused_result = fused_outputs[-1]
                graph_ir = graph_str
            torch._C._jit_nvfuser_set_comparison_callback(False, callback)

            def fn(x, y):
                z = torch.add(x, y)
                return torch.relu(z)

            x = torch.rand((4, 4)).cuda() - 0.5
            y = torch.rand((4, 4)).cuda() - 0.5

            fn_s = torch.jit.script(fn)
            fn_s(x, y)
            fn_s(x, y)
            fn_s(x, y)

            expected = fn(x, y)

            self.assertEqual(expected, fused_result)
            self.assertEqual(None, unfused_result)
            FileCheck().check("aten::add").run(graph_ir)
        finally:
            torch._C._jit_nvfuser_clear_comparison_callback()

    @unittest.skipIf(not RUN_NVFUSER, "requires NVFuser")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_cuda_fusion_guard_backward(self):
        old_guard = torch._C._jit_set_nvfuser_guard_mode(True)

        inp = torch.randn(10, device="cuda", requires_grad=True)
        grad = torch.randn(10, device="cuda")

        def f(x):
            a = x.cos().cos()
            return a
        scripted = torch.jit.script(f)

        with profile(activities=[ProfilerActivity.CPU]) as prof:
            for _ in range(5):
                inp.grad = None
                out = scripted(inp)
                out.backward(grad)

        # check that we do not have fallback triggered
        self.assertEqual(prof.events().table().find("fallback"), -1)
        torch._C._jit_set_nvfuser_guard_mode(old_guard)

    # TODO: generalize this
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    @unittest.skipIf(is_pre_volta(), "reduction not supported in pre volta device")
    def test_inf_quick_patch(self):
        inputs = [torch.tensor([-float('inf'), float('inf'), 4.0], device="cuda"),
                  torch.tensor([1.0, float('inf'), 4.0], device="cuda"),
                  torch.tensor([-float('inf'), -1.5, 4.0], device="cuda"),
                  torch.tensor([1.0, -3.0, float('nan')], device="cuda"),
                  torch.tensor([-float('inf'), -float('inf'), -float('inf')], device="cuda"),
                  torch.tensor([float('inf'), float('inf'), float('inf')], device="cuda"),
                  torch.tensor([float('nan'), float('nan'), float('nan')], device="cuda")]

        def fn_amax(x):
            return x.amax(dim=0)

        def fn_amin(x):
            return x.amin(dim=0)

        def fn_add_nan(x):
            return x.relu() + float('nan')

        def fn_add(x):
            return x + 1.0

        with nvfuser_singleton_fusion(True):
            for t in [fn_amax, fn_amin, fn_add, fn_add_nan]:
                for x in inputs:
                    t_jit = torch.jit.script(t)
                    self._run_helper(t_jit, t, x)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_clamp_reversed_bound(self):
        x = torch.tensor([1., -float('inf'), 2., float('inf'), float('nan')], device="cuda")

        def t(x):
            return x.clamp(min=1., max=0.5)

        with nvfuser_singleton_fusion(True):
            jit_t = torch.jit.script(t)
            self._run_helper(jit_t, t, x)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_issue_1785(self):
        class Fusion(torch.nn.Module):
            def __init__(self):
                super(Fusion, self).__init__()

            def forward(self, x, a, b):
                out = torch.mul(x.unsqueeze(-1), a)
                out = out + b
                return out

        x = torch.randn(1024, 192, 3, device='cuda')
        a = torch.randn(3, 128, device='cuda')
        b = torch.randn(3, 128, device='cuda')

        model = Fusion()
        jit_model = torch.jit.script(model)

        with torch.jit.fuser('fuser2'):
            for _ in range(4):
                out_ref = model(x, a, b)
                out_jit = jit_model(x, a, b)

        out_ref = model(x, a, b)
        out_jit = jit_model(x, a, b)
        self.assertTrue(self._compare("comparing output failed", out_ref, out_jit, 1e-5))

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_high_rank_fusion(self):
        # currently we want to limit fusion to node with input where rank <= 8
        rank_limit = 8
        shapes = [4 for i in range(rank_limit + 1)]
        x = torch.randn(shapes, device="cuda")

        with nvfuser_singleton_fusion(True):
            def t(x):
                return x.relu()

            jit_t = torch.jit.script(t)
            for i in range(5):
                jit_t(x)
                self.assertGraphContainsExactly(jit_t.graph_for(x), FUSION_GUARD, 0)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_clamp(self):
        x = torch.tensor([1., float('inf'), 2., float('nan'), float('-inf')], device="cuda")

        def clamp_max(x):
            return x.clamp(max=1.5)

        def clamp_min_max(x):
            return x.clamp(min=1.5)

        def clamp_min(x):
            return x.clamp(min=1., max=3.)

        with nvfuser_singleton_fusion(True):
            for t in [clamp_max, clamp_min, clamp_min_max]:
                t_jit = torch.jit.script(t)
                self._run_helper(t_jit, t, x)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_device_constant(self):
        x = torch.randn(4, 2, device="cuda")

        # cpu tensor shouldn't be fused
        def t_cpu(x):
            return torch.rand_like(x, device=torch.device(type='cpu'))

        with nvfuser_singleton_fusion(True):
            t_cpu_jit = torch.jit.script(t_cpu)
            for _ in range(5):
                t_cpu_jit(x)

            self.assertGraphContainsExactly(t_cpu_jit.graph_for(x), FUSION_GUARD, 0)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_expand(self):
        device = "cuda"
        x = torch.randn(3, 5, device=device)
        y = torch.randn(4, 2, 3, 5, device=device)

        def t(x, y):
            with torch.jit.strict_fusion():
                x = x.relu()
                o0 = x.expand(2, 3, 5)
                o1 = x.expand_as(y)
            return o0, o1

        t_jit = torch.jit.script(t)
        self._run_helper(t_jit, t, x, y, check_stride=True)

        def t2(x, y):
            o0 = x.expand(2, 3, 5)
            o1 = x.expand_as(y)
            x.add_(1)
            return o0, o1

        t2_jit = torch.jit.script(t2)
        self._run_helper(t2_jit, t2, x, y, check_stride=True, num_fusion=0)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_scheduler_with_polymorphic_broadcast(self):
        device = "cuda"
        x0 = torch.randn(10, 128, device=device)
        x1 = torch.rand_like(x0)
        x2 = torch.randn(10, device=device)

        def t(x0, x1, x2):
            x3 = x2.unsqueeze(-1)
            x4 = x3 + x0
            x5 = x3 + x1
            x6 = x5.sum(0)
            return x4, x6

        t_jit = torch.jit.script(t)
        self._run_helper(t_jit, t, x0, x1, x2, check_stride=True)

        x2 = torch.randn(128, device=device)

        def t2(x0, x1, x2):
            x3 = x2.unsqueeze(0)
            x4 = x3 + x0
            x5 = x3 + x1
            x6 = x5.sum(1)
            return x4, x6

        t2_jit = torch.jit.script(t2)
        self._run_helper(t2_jit, t2, x0, x1, x2, check_stride=True)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_type_inference(self):
        device = "cuda"
        x0 = torch.randn(10, 128, device=device)
        x1 = torch.rand_like(x0)
        x2 = torch.rand_like(x0)

        def t(x0, x1, x2, flag : bool = True):
            x3 = 2.0 * x0
            x4 = 2.0 * x1
            x5 = 2.0 * x2
            if flag:
                return torch.stack([x3, x4, x5], dim=-1)
            # second code path doesn't run through profiling
            # hence would utilize type inference with profiling information
            return x0 + x1 + x2

        t_jit = torch.jit.script(t)
        self._run_helper(t_jit, t, x0, x1, x2, check_stride=True)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_disable_const_chunk_propagation_for_normalization(self):
        device = "cuda"
        x0 = torch.randn(10, 12, device=device)
        x1 = torch.randn(10, 4, device=device)
        w0 = torch.randn(12, device=device)
        w1 = torch.randn(4, device=device)

        def t(x, y, w0, w1):
            ih = torch.layer_norm(x, (12,), w0)
            i_r, i_z, i_n = ih.chunk(3, dim=1)
            i_n = torch.layer_norm(i_n, (4,), w1)
            r = torch.sigmoid(i_r)
            n = torch.tanh(i_n + r * i_z)
            h = n + r * y
            return h

        t_jit = torch.jit.script(t)
        self._run_helper(t_jit, t, x0, x1, w0, w1, check_stride=True)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_no_tensor_input(self):
        device = "cuda"
        x = torch.randn(512, device=device)

        def t(x):
            tensor0 = torch.tensor(3, dtype=torch.float32, device='cuda')
            tensor1 = torch.tensor(3, dtype=torch.float32, device='cuda')
            o = torch.div(x.numel(), tensor0)
            o = torch.mul(o, tensor1)
            return o

        t_jit = torch.jit.script(t)
        self._run_helper(t_jit, t, x, check_stride=True)

        # Note that curently TS embeds constant tensor in the graph
        # this triggers memory leak check in CI
        torch.jit._state._python_cu.drop_all_functions()


class TestEnableDisableCudaFuser(JitTestCase):
    def setUp(self):
        super().setUp()
        if RUN_NVFUSER:
            self.is_enabled = torch._C._jit_set_nvfuser_enabled(False)

    def tearDown(self):
        if RUN_NVFUSER:
            torch._C._jit_set_nvfuser_enabled(self.is_enabled)
        super().tearDown()

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    def test_context_manager_test(self):
        x = torch.randn(4, 8, dtype=torch.float, device="cuda")
        y = torch.randn(4, 8, dtype=torch.float, device="cuda")
        with torch.jit.fuser('fuser2'):
            with torch.jit.fuser('fuser2'):

                def t1(x, y):
                    o = x + y
                    o = o + 2.0
                    return o
                t_jit = torch.jit.script(t1)
                t_jit(x, y)
                t_jit(x, y)
                self.assertGraphContains(t_jit.graph_for(x, y), FUSION_GUARD)

            def t2(x, y):
                o = x + y
                o = o + 3.0
                return o
            t_jit_2 = torch.jit.script(t2)
            t_jit_2(x, y)
            t_jit_2(x, y)
            self.assertGraphContains(t_jit_2.graph_for(x, y), FUSION_GUARD)

        def t3(x, y):
            o = x + y
            o = o + 4.0
            return o
        t_jit_3 = torch.jit.script(t3)
        t_jit_3(x, y)
        t_jit_3(x, y)
        self.assertGraphContainsExactly(t_jit_3.graph_for(x, y), FUSION_GUARD, 0)

    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    def test_register_fuser(self):
        self.assertFalse(torch._C._jit_set_nvfuser_enabled(True))
        self.assertTrue(torch._C._jit_nvfuser_enabled())
        self.assertTrue(torch._C._jit_set_nvfuser_enabled(True))
        self.assertTrue(torch._C._jit_nvfuser_enabled())
        self.assertTrue(torch._C._jit_set_nvfuser_enabled(False))
        self.assertFalse(torch._C._jit_nvfuser_enabled())

    @unittest.skipIf(RUN_CUDA, "Testing on CPU only")
    def test_register_fuser_cpu(self):
        with self.assertRaises(RuntimeError):
            torch._C._jit_set_nvfuser_enabled(True)
            torch._C._jit_set_nvfuser_enabled(False)

    def test_can_be_enabled_nvfuser(self):
        expected = RUN_CUDA

        self.assertEqual(expected, torch._C._jit_nvfuser_can_be_enabled())

# See TestNNCOpInfoParent
class TestCudaFuserOpInfoParent(JitCommonTestCase):
    pass

class TestCudaFuserOpInfo(TestCudaFuserOpInfoParent):
    def setUp(self):
        super(TestCudaFuserOpInfoParent, self).setUp()
        if RUN_NVFUSER:
            self.cuda_fuser_options = CudaFuserTestOptions()
            # enables guard mode since tracing could change graph to violate guard.
            torch._C._jit_set_nvfuser_guard_mode(True)
        self.nvfuser_single_node_mode = torch._C._jit_set_nvfuser_single_node_mode(True)

    def tearDown(self):
        if RUN_NVFUSER:
            self.cuda_fuser_options.restore()

        torch._C._jit_set_nvfuser_single_node_mode(self.nvfuser_single_node_mode)

        super(TestCudaFuserOpInfoParent, self).tearDown()

    @slowTest
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @ops(op_db, dtypes=OpDTypes.supported)
    def test_nvfuser_correctness(self, device, dtype, op):
        if not op.supports_tracing:
            self.skipTest("nvfuser requires tracing support")

        variant_sample_pairs = get_traced_sample_variant_pairs(device, dtype, op)

        for variant, sample in variant_sample_pairs:
            trace = create_traced_fn(self, variant, cache_traced_fn=True)
            ref = variant(*clone_inputs((sample.input, *sample.args)), **sample.kwargs)

            trace(*clone_inputs((sample.input, *sample.args)), **sample.kwargs)

            val = trace(*clone_inputs((sample.input, *sample.args)), **sample.kwargs)

            self.assertEqual(ref, val, exact_layout=True)

        # Note: Clearing CU after NVFuser tests
        # https://github.com/pytorch/pytorch/issues/35600
        # each torch.jit.trace adds state to the _python_cu compilation unit
        # since this test traces a lot of functions, out-of-memory can occur
        # if the CU is not cleared.
        torch.jit._state._python_cu.drop_all_functions()

    @slowTest
    @unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING,
                     "Requires fusion optimization pass to be effective")
    @ops(op_db, allowed_dtypes=(torch.float16, torch.bfloat16, torch.float32,
                                torch.float64, torch.complex64, torch.complex128))
    def test_nvfuser_extremal_values(self, device, dtype, op):
        if not op.supports_tracing:
            self.skipTest("nvfuser requires tracing support")

        variant_sample_pairs = get_traced_sample_variant_pairs(device, dtype, op)

        def _get_extremal_tensor(x, val, dtype):
            if x.dtype != dtype:
                return x
            return torch.full_like(x, val)

        def _get_extremal_input(x, val, dtype):
            if isinstance(x, torch.Tensor):
                return _get_extremal_tensor(x, val, dtype)
            elif is_iterable_of_tensors(x):
                return [_get_extremal_tensor(y, val, dtype) for y in x]
            return x

        def _get_extremal_sample(sample: SampleInput, val, dtype):
            extremal_sample = SampleInput(
                input=_get_extremal_input(sample.input, val, dtype),
                args=tuple(_get_extremal_input(x, val, dtype) for x in sample.args),
                kwargs={k: _get_extremal_input(v, val, dtype) for k, v in sample.kwargs.items()},
            )
            return extremal_sample

        def _get_extremal_samples(sample: SampleInput, dtype):
            vals = [float('inf'), float('-inf'), float('nan')]
            if dtype.is_complex:
                complex_vals = itertools.product(vals, vals)
                vals = tuple(map(lambda x: complex(*x), complex_vals))
            for val in vals:
                yield _get_extremal_sample(sample, val, dtype)

        variant_sample_pairs = get_traced_sample_variant_pairs(device, dtype, op)

        for variant, sample in variant_sample_pairs:

            trace = create_traced_fn(self, variant, cache_traced_fn=True)
            trace(*clone_inputs((sample.input, *sample.args)), **sample.kwargs)
            trace(*clone_inputs((sample.input, *sample.args)), **sample.kwargs)

            for extremal_sample in _get_extremal_samples(sample, dtype):
                try:
                    with freeze_rng_state():
                        ref = variant(*clone_inputs((extremal_sample.input, *extremal_sample.args)),
                                      **extremal_sample.kwargs)
                except (torch._C._LinAlgError, RuntimeError, ValueError):
                    # if eager errors out, then don't expect NVFuser to pass
                    continue

                with freeze_rng_state():
                    val = trace(*clone_inputs((extremal_sample.input, *extremal_sample.args)),
                                **extremal_sample.kwargs)

                self.assertEqual(val, ref, equal_nan=True, exact_device=True)

            # See [Note: Clearing CU after NVFuser tests]
            torch.jit._state._python_cu.drop_all_functions()

instantiate_device_type_tests(TestCudaFuserOpInfo, globals(), only_for=("cuda"))


if __name__ == '__main__':
    run_tests()
