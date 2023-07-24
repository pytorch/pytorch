import itertools
import unittest
from typing import List

import torch
import torch.nn as nn

import torch._dynamo
import torch._dynamo.test_case
from torch.fx.experimental.proxy_tensor import make_fx
from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.backends.registry import _BACKENDS, register_backend
from torch._inductor.decomposition import select_decomp_table
from torch._inductor.fx_passes.onednn_graph_fusion import (
    allow_manydim_bmm,
    onednn_graph_fuse_fx,
    remove_redundant_expand,
    replace_max_pool_with_indices,
    replace_t_matmul_to_matmul,
)

import subprocess
import os
os.environ["_DNNL_GRAPH_DISABLE_COMPILER_BACKEND"] = "1"
ONEDNN_BACKEND = "onednn"

class TestOneDNNDynamo(torch._dynamo.test_case.TestCase):
    def _compile_and_check_accuracy(self, mod, args, **kwargs):
        compiled_model = torch.compile(mod, options={"cpp.onednn_graph": True})
        with torch.no_grad():
            compiled_out = compiled_model(*args)
            expected = mod(*args)
        atol = kwargs["atol"] if "atol" in kwargs else 1e-5
        self.assertTrue(torch.allclose(compiled_out, expected, atol=atol))
        torch._dynamo.reset()

    def _check_fusion(self, mod, args, expected_fusion_names: List[str] = [], skip_graphs_before=0):
        if 'test_onednn_check_fusion_backend' in _BACKENDS:
            del _BACKENDS['test_onednn_check_fusion_backend']

        @register_backend
        def test_onednn_check_fusion_backend(gm, example_inputs):
            def fusion_wrapper(gm, example_inputs):
                gm = onednn_graph_fuse_fx(gm, False)
                fused_names = [node.target.name() for node in gm.graph.nodes if getattr(node.target, "is_opaque", False)]
                nonlocal skip_graphs_before
                if skip_graphs_before:
                    skip_graphs_before += -1
                    return gm
                self.assertTrue(len(expected_fusion_names) == len(fused_names))
                for name in expected_fusion_names:
                    self.assertTrue(name in fused_names)
                return gm

            return aot_autograd(
                fw_compiler=fusion_wrapper,
                decompositions={}
            )(gm, example_inputs)

        compiled_model = torch.compile(mod, backend='test_onednn_check_fusion_backend')
        with torch.no_grad():
            compiled_out = compiled_model(*args)
        torch._dynamo.reset()

    def test_activation(self):
        activations = {
            nn.ELU(): 'elu',
            nn.GELU(): 'gelu',
            nn.Hardsigmoid(): 'hardsigmoid',
            nn.Hardswish(): 'hardswish',
            nn.Hardtanh(0, 0.5): 'hardtanh',
            nn.LeakyReLU(): 'leaky_relu',
            nn.Mish(): 'mish',
            nn.ReLU(): 'relu',
        }
        for act, name in activations.items():
            mod = nn.Sequential(nn.Linear(8, 16), act)
            with torch.no_grad():
                x = torch.randn(32, 8)
            self._check_fusion(mod, args=(x,), expected_fusion_names=[f'addmm_{name}'])
            self._compile_and_check_accuracy(mod, args=(x,))

    def test_avgpool(self):
        mod = nn.Sequential(nn.AvgPool2d(3))
        with torch.no_grad():
            x = torch.randn(20, 16, 50, 32)
        self._check_fusion(mod, args=(x,), expected_fusion_names=["avg_pool2d"])
        self._compile_and_check_accuracy(mod, args=(x,))

    def test_mlp(self):
        """
        Test 2-layer MLP model
        """
        mod = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU())
        with torch.no_grad():
            x = torch.randn(32, 8)
        self._check_fusion(mod, args=(x,), expected_fusion_names=['addmm_relu', 'addmm_1_relu_1'])
        self._compile_and_check_accuracy(mod, args=(x,))

    def test_manydim_bmm(self):
        """
        Test execution of 4D bmm with removal of clone/expand/view
        View removal allows N-dim execution of bmm via MatMul instead of converting to 3D.
        Expand ops are redundant when output is same shape/stride as input.
        Clone ops change stride and are not redundant, but backend prefers to determine layout.
        """
        def mod(x, y):
            return x.clone(memory_format=torch.contiguous_format) @ y
        with torch.no_grad():
            x = torch.randn(3, 4, 32, 8)
            y = torch.randn(3, 4, 8, 16)
        self._check_fusion(mod, args=(x, y), expected_fusion_names=['bmm_default'])
        self._compile_and_check_accuracy(mod, args=(x,y))

    def test_no_fail_on_int64_tensor(self):
        # Testing no crash on int64 tensors
        # TODO: Does not test LLGA execution, just that graph build doesn't fail
        def mod(x, y):
            return x + y * x
        with torch.no_grad():
            x = torch.randint(1024, (4, 1024), dtype=torch.int64)
            y = torch.randint(1024, (4, 1024), dtype=torch.int64)
        self._check_fusion(mod, args=(x, y), )
        self._compile_and_check_accuracy(mod, args=(x, y))

    def test_add_scalar(self):
        def mod(x, y):
            x = x @ x.T
            return x + y
        with torch.no_grad():
            x = torch.rand((4, 1024))
            self._check_fusion(mod, args=(x, 1), expected_fusion_names=['mm_add'])
            self._check_fusion(mod, args=(x, 0.1), expected_fusion_names=['mm_add'])
            self._compile_and_check_accuracy(mod, args=(x, 1))
            self._compile_and_check_accuracy(mod, args=(x, 0.1))

    def test_conv_relu(self):
        class TestModule(nn.Module):
            def __init__(self, bias):
                super().__init__()
                self.conv = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1), bias=bias)
                self.relu = nn.ReLU()

            def forward(self, input):
                out = self.conv(input)
                out = self.relu(out)
                return out
        for bias in [True, False]:
            mod = TestModule(bias)
            with torch.no_grad():
                x = torch.randn(20, 16, 50, 100)
            self._check_fusion(mod, args=(x,), expected_fusion_names=['convolution_relu'])
            self._compile_and_check_accuracy(mod, args=(x,))


    def test_conv_channels_last(self):
        channels_last_code = """import torch
import torch.nn as nn
import os
os.environ["ONEDNN_VERBOSE"] = "1"
mod = nn.Sequential(nn.Conv2d(16,33,3), nn.ReLU())
mod = torch.compile(mod, options={"cpp.onednn_graph": True})
with torch.no_grad():
    x = torch.randn(20, 16, 50, 100)
    x = x.to(memory_format=torch.channels_last)
    out = mod(x)
    """
        proc = subprocess.Popen(["python", "-c", channels_last_code], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out = proc.communicate()
        lines = out[0].decode().split('\n')
        reorder_lines = [line for line in lines if 'reorder' in line]
        # Only reorder is tiling conv weights, not input x.
        self.assertTrue(len(reorder_lines) == 1)
        self.assertTrue('33x16x3x3' in reorder_lines[0])
        mod = nn.Sequential(nn.Conv2d(16,33,3), nn.ReLU())
        with torch.no_grad():
            x = torch.randn(20, 16, 50, 100)
            x = x.to(memory_format=torch.channels_last)
            self._compile_and_check_accuracy(mod, args=(x,))


    def test_reorder_any_layout(self):
        channels_last_code = """import torch
import torch.nn as nn
import os
os.environ["ONEDNN_VERBOSE"] = "1"
mod = nn.Sequential(nn.Conv2d(16,33,3), nn.MaxPool2d(2))
mod = torch.compile(mod, options={"cpp.onednn_graph": True})
with torch.no_grad():
    x = torch.randn(20, 16, 50, 100)
    x = x.to()
    out = mod(x)
    """
        proc = subprocess.Popen(["python", "-c", channels_last_code], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out = proc.communicate()
        lines = out[0].decode().split('\n')
        conv_ind = [ind for ind, line in enumerate(lines) if 'convolution' in line][0]
        pooling_ind = [ind for ind, line in enumerate(lines) if 'pooling' in line][0]
        self.assertTrue(conv_ind + 1 == pooling_ind)

    def test_matmul_transpose_patterns(self):
        """
        Test replacement of .t() and .permute(1,0) on addmm and mm.
        Test that .transpose() calls are removed before MatMul operation.
        """
        def mod(x, y, z):
            out1 = torch.addmm(x, y.t(), z.t())
            out2 = torch.addmm(x, y.t(), z)
            out1 += torch.mm(y.T, z.T)  # .T maps to permute([1,0])
            stack1 = torch.stack((x,y))
            stack2 = torch.stack((y,z))
            out = torch.bmm(stack1.transpose(1,2), stack2.transpose(-1,-2))
            return (out1 + out2) + out

        with torch.no_grad():
            x = torch.randn(8, 8)
            y = torch.randn(8, 8)
            z = torch.randn(8, 8)
            fusions = ['mm_default', 'addmm_1', 'bmm', 'addmm_add_add_1_add_2']
            self._check_fusion(mod, args=(x, y, z), expected_fusion_names=fusions)
            self._compile_and_check_accuracy(mod, args=(x, y, z))

    def test_max_pool2d(self):
        """
        Test aten.native_max_pool2d_with_indices is replaced by aten.max_pool2d
        and that execution is accurate.
        """
        def MaxPoolModel(kernel_size, stride, padding, dilation, return_indices, ceil_mode):
            return nn.Sequential(nn.MaxPool2d(kernel_size=kernel_size,
                                              stride=stride,
                                              padding=padding,
                                              dilation=dilation,
                                              return_indices=return_indices,
                                              ceil_mode=ceil_mode))
        for [
            kernel_size,
            stride,
            padding,
            dilation,
            return_indices,
            ceil_mode] in itertools.product(
            [5, [5, 7]],
            [2, [2, 4]],
            [0, 1, [1, 2]],
            [1, 2, [2, 1]],
            [False],
            [True, False]):

            mod = MaxPoolModel(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
            x = torch.randn(1, 3, 50, 50)
            self._check_fusion(mod, args=(x,), expected_fusion_names=['max_pool2d_default'])
            self._compile_and_check_accuracy(mod, args=(x,))

    def test_addmm_addmm_fusion(self):
        """
        Test addmm+addmm pattern with and without graph compiler
        """
        class LinLin(torch.nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.l1 = torch.nn.Linear(128,768)
                self.l2 = torch.nn.Linear(768,768)

            def forward(self, x):
                x1 = self.l1(x)
                return x1, self.l2(x1)

        for use_gc in [True, False]:
            mod = LinLin()
            if use_gc:
                del os.environ["_DNNL_GRAPH_DISABLE_COMPILER_BACKEND"]
            else:
                os.environ["_DNNL_GRAPH_DISABLE_COMPILER_BACKEND"] = "1"
            compiled_model = torch.compile(mod, options={"cpp.onednn_graph": True})
            with torch.no_grad():
                x = torch.randn(4,512,128)
                compiled_out = compiled_model(x)
                expected = mod(x)
            self.assertTrue(torch.allclose(compiled_out[0], expected[0], atol=1e-5))
            self.assertTrue(torch.allclose(compiled_out[1], expected[1], atol=1e-5))
            torch._dynamo.reset()
        os.environ["_DNNL_GRAPH_DISABLE_COMPILER_BACKEND"] = "1"

class TestOneDNNGraphRewrite(torch._dynamo.test_case.TestCase):
    def _check_graph_rewrites(self, mod, args, rewrite_calls=[], pre_rewrite_checks=[], post_rewrite_checks=[]):
        def check_helper(rewrite_checks):
            for assert_check, func in rewrite_checks:
                if isinstance(func, str):
                    result = any(func in str(n.target) for n in gm.graph.nodes)
                else:
                    result = any(n.target == func for n in gm.graph.nodes)
                self.assertTrue(result) if assert_check else self.assertFalse(result)

        decompositions = select_decomp_table()
        gm = make_fx(mod, decomposition_table=decompositions)(*args)
        gm.graph.eliminate_dead_code()
        check_helper(pre_rewrite_checks)
        for rewrite in rewrite_calls:
            rewrite(gm)
        check_helper(post_rewrite_checks)

    def test_manydim_bmm(self):
        """
        Rewrite test: remove_redundant_expand, allow_manydim_bmm
        Test removal of expand/view around bmm
        View removal allows N-dim execution of bmm via MatMul instead of converting to 3D.
        Expand ops are redundant when output is same shape/stride as input.
        """
        def mod(x, y):
            return x.clone(memory_format=torch.contiguous_format) @ y
        with torch.no_grad():
            x = torch.randn(3, 4, 32, 8)
            y = torch.randn(3, 4, 8, 16)
        self._check_graph_rewrites(mod, args=(x,y),
            rewrite_calls=[remove_redundant_expand, allow_manydim_bmm],
            post_rewrite_checks=[
                (False, torch.ops.aten.view.default),
                (False, torch.ops.aten.expand.default),
            ]
        )

    def test_matmul_transpose_patterns(self):
        """
        Rewrite test: replace_t_matmul_to_matmul
        Test replacement of .t() and .permute(1,0) on addmm and mm.
        Test that .transpose() calls are removed before MatMul operation.
        """
        def mod(x, y, z):
            out1 = torch.addmm(x, y.t(), z.t())
            out2 = torch.addmm(x, y.t(), z)
            out1 += torch.mm(y.T, z.T)  # .T maps to permute([1,0])
            stack1 = torch.stack((x,y))
            stack2 = torch.stack((y,z))
            out = torch.bmm(stack1, stack2.transpose(-1,-2))
            return (out1 + out2) + out

        with torch.no_grad():
            x = torch.randn(8, 8)
            y = torch.randn(8, 8)
            z = torch.randn(8, 8)
            self._check_graph_rewrites(mod, args=(x,y,z),
                rewrite_calls=[replace_t_matmul_to_matmul],
                pre_rewrite_checks=[
                    (True, torch.ops.aten.permute.default),
                ],
                post_rewrite_checks=[
                    (False, torch.ops.aten.t.default),
                    (False, torch.ops.aten.permute.default),
                    (False, torch.ops.aten.transpose.int),
                ]
            )

    def test_max_pool2d(self):
        """
        Rewrite test: replace_max_pool_with_indices
        Test aten.native_max_pool2d_with_indices is replaced by aten.max_pool2d
        and that execution is accurate.
        """
        def mod(kernel_size, stride, padding, dilation, return_indices, ceil_mode):
            return nn.Sequential(nn.MaxPool2d(kernel_size=kernel_size,
                                              stride=stride,
                                              padding=padding,
                                              dilation=dilation,
                                              return_indices=return_indices,
                                              ceil_mode=ceil_mode))
        for [
            kernel_size,
            stride,
            padding,
            dilation,
            return_indices,
            ceil_mode] in itertools.product(
            [5, [5, 7]],
            [2, [2, 4]],
            [0, 1, [1, 2]],
            [1, 2, [2, 1]],
            [False],
            [True, False]):

            model = mod(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
            x = torch.randn(1, 3, 50, 50)
            self._check_graph_rewrites( model, args=(x,),
                rewrite_calls=[replace_max_pool_with_indices],
                post_rewrite_checks=[
                    (False, torch.ops.aten.max_pool2d_with_indices.default),
                    (True, torch.ops.aten.max_pool2d.default),
                ]
            )


if __name__ == '__main__':
    from torch._dynamo.test_case import run_tests
    run_tests()
