# Owner(s): ["module: nvfuser"]

from copy import deepcopy
from functools import partial
import re
from typing import List
import unittest

import torch
from torch.testing._internal.common_utils import run_tests, TEST_WITH_ROCM, TestCase
from torch.testing._internal.jit_utils import RUN_CUDA
import torch._refs as refs
import torch._prims as prims

# Will only create the nvfuser module if CUDA is available
try:
    from nvfuser._C import Fusion, FusionCache, FusionDefinition, DataType
except ImportError:
    pass

RUN_NVFUSER = RUN_CUDA and not TEST_WITH_ROCM

def is_pre_volta():
    if not RUN_NVFUSER:
        return False
    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    return prop.major < 7

@unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
@unittest.skipIf(is_pre_volta(), "Only supported on Volta and newer devices.")
class TestNvFuserFrontend(TestCase):

    # Helper function to verify the nvfuser output and make sure the string
    # definition based on the FusionDefinition is executable and matches the
    # original definition
    def exec_nvfuser(self, fusion_func, inputs, new_fusion_expected=True) :
        inputs_cap = deepcopy(inputs)
        fc = FusionCache.get()
        before_fusions = fc.num_fusions()

        # Execute a fusion function and capture the string python definition
        fs = Fusion()
        fd = FusionDefinition(fs)
        with fd:
            fusion_func(fd)
        fd_str = fd.__repr__()
        out = fs.execute(inputs)

        # Execute the python definition that was captured
        func_name = re.findall('(nvfuser_fusion_id\\d+)', fd_str.split('\n')[1])[0]
        exec(fd_str)
        fs_cap = Fusion()
        with FusionDefinition(fs_cap) as fd_cap:
            eval(func_name)(fd_cap)
        out_cap = fs_cap.execute(inputs_cap)

        # Make sure the original and captured definitions match
        for idx in range(len(out)) :
            self.assertEqual(out[idx], out_cap[idx])

        self.assertEqual(fc.num_fusions() - before_fusions, int(new_fusion_expected))
        return out, fs

    def test_basic(self) :
        inputs = [
            torch.ones(2, 4, 8, device='cuda'),
            torch.ones(2, 4, 8, device='cuda'),
        ]

        def fusion_func(fd: FusionDefinition) :
            t0 = fd.define_tensor(3)
            t1 = fd.define_tensor(3)
            c0 = fd.define_constant(3.0)

            t2 = fd.ops.add(t0, t1)
            t3 = fd.ops.mul(t2, c0)
            t4 = fd.ops.sum(t3, [-1], False, DataType.Float)

            fd.add_output(t4)

        # Expected Output is a tensor of 48's
        nvf_out1, _ = self.exec_nvfuser(fusion_func, inputs)

        # Create a new fusion with the same definition, it should hit the cache!
        nvf_out2, fs2 = self.exec_nvfuser(fusion_func, inputs, new_fusion_expected=False)

        # Create a fusion from a fusion id and make sure it executes!
        fs3 = Fusion(fs2.id())
        nvf_out3 = fs3.execute(inputs)

        eager_out = torch.sum((inputs[0] + inputs[1]) * 3.0, dim=-1)
        self.assertEqual(eager_out, nvf_out1[0])
        self.assertEqual(eager_out, nvf_out2[0])
        self.assertEqual(eager_out, nvf_out3[0])

    def test_basic_fp16(self) :
        inputs = [
            torch.ones(2, 4, 8, device='cuda', dtype=torch.float16),
            torch.ones(2, 4, 8, device='cuda', dtype=torch.float16),
        ]

        def fusion_func(fd: FusionDefinition) :
            t0 = fd.define_tensor(3, DataType.Half)
            t1 = fd.define_tensor(3, DataType.Half)
            c0 = fd.define_constant(3.0)

            t2 = fd.ops.add(t0, t1)
            t3 = fd.ops.mul(t2, c0)
            t4 = fd.ops.sum(t3, [-1], False, DataType.Float)

            t5 = fd.ops.cast(t4, DataType.Half)
            fd.add_output(t5)

        # Expected Output is a tensor of 48's
        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)
        eager_out = torch.sum((inputs[0] + inputs[1]) * 3.0, dim=-1)
        self.assertEqual(eager_out, nvf_out[0])

    def test_cast_double_to_half(self) :
        inputs = [
            torch.randn(2, 4, device='cuda', dtype=torch.float64),
            torch.randn(2, 4, device='cuda', dtype=torch.float64),
        ]

        def fusion_func(fd: FusionDefinition) :
            t0 = fd.define_tensor(2, DataType.Double)
            t1 = fd.define_tensor(2, DataType.Double)

            t0h = fd.ops.cast(t0, DataType.Half)
            t1h = fd.ops.cast(t1, DataType.Half)
            t2 = fd.ops.add(t0h, t1h)
            t3 = fd.ops.relu(t2)
            t4 = fd.ops.cast(t3, DataType.Half)

            fd.add_output(t4)

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)
        eager_out = torch.relu(inputs[0].to(torch.half) + inputs[1].to(torch.half))
        self.assertEqual(eager_out, nvf_out[0])

    def test_promote_to_double(self) :
        inputs = [
            torch.randn(2, 4, device='cuda', dtype=torch.float16),
            torch.randn(2, 4, device='cuda', dtype=torch.float64),
        ]

        def fusion_func(fd: FusionDefinition) :
            t0 = fd.define_tensor(2, DataType.Half)
            t1 = fd.define_tensor(2, DataType.Double)

            t2 = fd.ops.add(t0, t1)
            t5 = fd.ops.relu(t2)

            fd.add_output(t5)

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)
        eager_out = torch.relu(inputs[0] + inputs[1])
        self.assertEqual(eager_out, nvf_out[0])

    def test_implicit_broadcast_input(self) :
        inputs = [
            torch.randn(3, device='cuda'),
            torch.randn(2, 3, 4, device='cuda'),
        ]

        def fusion_func(fd: FusionDefinition) :
            t0 = fd.define_tensor(1)
            t1 = fd.define_tensor(3)

            t0_b = fd.ops.broadcast_in_dim(t0, [2, 3, 4], [1])
            t2 = fd.ops.add(t0_b, t1)

            fd.add_output(t2)

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)
        eager_out = refs.add(prims.broadcast_in_dim(inputs[0], inputs[1].size(), [1]), inputs[1])
        self.assertEqual(eager_out, nvf_out[0])

    def test_explicit_broadcast_input(self) :
        inputs = [
            torch.randn(1, 1, 4, device='cuda'),
            torch.randn(2, 3, 4, device='cuda'),
        ]

        def fusion_func(fd: FusionDefinition) :
            t0 = fd.define_tensor(sizes=inputs[0].size(), strides=inputs[0].stride())
            t1 = fd.define_tensor(sizes=inputs[1].size(), strides=inputs[1].stride())

            t0_b = fd.ops.broadcast_in_dim(t0, inputs[1].size(), [0, 1, 2])
            t2 = fd.ops.add(t0_b, t1)

            fd.add_output(t2)

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)
        eager_out = refs.add(prims.broadcast_in_dim(inputs[0], inputs[1].size(), [0, 1, 2]), inputs[1])
        self.assertEqual(eager_out, nvf_out[0])

    def test_broadcast_mixing(self) :
        inputs = [
            torch.randn(3, 1, device='cuda'),
            torch.randn(3, device='cuda'),
        ]

        def fusion_func(fd: FusionDefinition) :
            t0 = fd.define_tensor([3, 1], [1, 1])
            t1 = fd.define_tensor(1)

            t1_b = fd.ops.broadcast_in_dim(t1, [3, 3], [0])
            t2 = fd.ops.add(t0, t1_b)

            fd.add_output(t2)

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)
        eager_out = refs.add(inputs[0], prims.broadcast_in_dim(inputs[1], [3, 3], [0]))
        self.assertEqual(eager_out, nvf_out[0])

    def test_ops_broadcast(self) :
        inputs = [
            torch.randn(3, device='cuda'),
            torch.randn(2, 3, 4, device='cuda'),
        ]

        def fusion_func(fd: FusionDefinition) :
            t0 = fd.define_tensor(1)
            t1 = fd.define_tensor(3)

            t0_b = fd.ops.broadcast(t0, [True, False, True])
            t2 = fd.ops.add(t0_b, t1)

            fd.add_output(t2)

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)
        eager_out = refs.add(prims.broadcast_in_dim(inputs[0], inputs[1].size(), [1]), inputs[1])
        self.assertEqual(eager_out, nvf_out[0])

    def test_prim_layer_norm_fwd(self) :
        input_size = [64, 128, 1024]
        dtype = torch.float32
        device = 'cuda'
        inputs = [
            torch.randn(*input_size, device=device, requires_grad=True),
            torch.nn.Parameter(torch.randn(input_size[2], dtype=dtype, device=device)),
            torch.nn.Parameter(torch.randn(input_size[2], dtype=dtype, device=device)),
        ]

        def primitive_definition(
            inputs: torch.Tensor,
            weight: torch.Tensor,
            bias: torch.Tensor,
            normalization_axis: int,
            keepdim: bool,
        ) -> torch.Tensor:
            mean = inputs.mean(normalization_axis, keepdim=keepdim)
            diff = inputs - mean
            diff_sq = diff * diff
            var = diff_sq.mean(normalization_axis, keepdim=keepdim)
            pre_shift_scale_norm_output = (inputs - mean) / torch.sqrt(var + 1e-12)
            norm_output = weight * pre_shift_scale_norm_output + bias
            return norm_output

        def nvfuser_fusion(
            fd: FusionDefinition,
            normalization_axis: int,
            norm_size: int,
            input_shape: List[int],
            eps: float,
            keepDim: bool
        ) -> None :
            inputs = fd.define_tensor(symbolic_sizes=[-1, -1, -1], contiguous=[True, True, True], dtype=DataType.Float)
            weights = fd.define_tensor(symbolic_sizes=[-1], contiguous=[True], dtype=DataType.Float)
            bias = fd.define_tensor(symbolic_sizes=[-1], contiguous=[True], dtype=DataType.Float)
            sum0 = fd.ops.sum(inputs, axes=[normalization_axis], keepdim=keepDim)
            norm_const = fd.define_constant(norm_size)
            mean = fd.ops.div(sum0, norm_const)
            diff = fd.ops.sub(inputs, mean)
            diff_sq = fd.ops.mul(diff, diff)
            sum1 = fd.ops.sum(diff_sq, axes=[normalization_axis], keepdim=keepDim)
            var = fd.ops.div(sum1, norm_const)
            eps_const = fd.define_constant(eps)
            var_eps = fd.ops.add(var, eps_const)
            invstd = fd.ops.rsqrt(var_eps)
            pre_scale_bias = fd.ops.mul(diff, invstd)
            weights_bcast = fd.ops.broadcast_in_dim(weights, output_shape=input_shape, broadcast_dims=[2])
            scale = fd.ops.mul(pre_scale_bias, weights_bcast)
            bias_bcast = fd.ops.broadcast_in_dim(bias, output_shape=input_shape, broadcast_dims=[2])
            out = fd.ops.add(scale, bias_bcast)
            fd.add_output(out)
            fd.add_output(mean)
            fd.add_output(invstd)

        def nvfuser_fusion_var_mean(
            fd: FusionDefinition,
            normalization_axis: int,
            norm_size: int,
            input_shape: List[int],
            eps: float,
            keepDim: bool
        ) -> None :
            inputs = fd.define_tensor(symbolic_sizes=[-1, -1, -1], contiguous=[True, True, True], dtype=DataType.Float)
            weights = fd.define_tensor(symbolic_sizes=[-1], contiguous=[True], dtype=DataType.Float)
            bias = fd.define_tensor(symbolic_sizes=[-1], contiguous=[True], dtype=DataType.Float)
            var, mean = fd.ops.var_mean(inputs, axes=[normalization_axis], correction=0, keepdim=keepDim)
            eps_const = fd.define_constant(eps)
            var_eps = fd.ops.add(var, eps_const)
            invstd = fd.ops.rsqrt(var_eps)
            diff = fd.ops.sub(inputs, mean)
            pre_scale_bias = fd.ops.mul(diff, invstd)
            weights_bcast = fd.ops.broadcast_in_dim(weights, output_shape=input_shape, broadcast_dims=[2])
            scale = fd.ops.mul(pre_scale_bias, weights_bcast)
            bias_bcast = fd.ops.broadcast_in_dim(bias, output_shape=input_shape, broadcast_dims=[2])
            out = fd.ops.add(scale, bias_bcast)
            fd.add_output(out)
            fd.add_output(mean)
            fd.add_output(invstd)

        fusion_func_1 = partial(nvfuser_fusion,
                                normalization_axis=2,
                                norm_size=inputs[0].size()[2],
                                input_shape=inputs[0].size(),
                                eps=1e-12,
                                keepDim=True)
        nvf_out, _ = self.exec_nvfuser(fusion_func_1, inputs)

        fusion_func_2 = partial(nvfuser_fusion_var_mean,
                                normalization_axis=2,
                                norm_size=inputs[0].size()[2],
                                input_shape=inputs[0].size(),
                                eps=1e-12,
                                keepDim=True)
        nvf_var_mean_out, _ = self.exec_nvfuser(fusion_func_2, inputs)

        eager_out = primitive_definition(inputs[0], inputs[1], inputs[2], 2, True)

        self.assertEqual(eager_out, nvf_out[0])
        self.assertEqual(eager_out, nvf_var_mean_out[0])

    def test_prim_rms_norm_fwd(self) :
        input_size = [64, 128, 1024]
        dtype = torch.float32
        device = 'cuda'
        inputs = [
            torch.randn(*input_size, device=device, requires_grad=True),
            torch.nn.Parameter(torch.randn(input_size[2], dtype=dtype, device=device)),
        ]

        def primitive_definition(
            inputs: torch.Tensor,
            weight: torch.Tensor,
            normalization_axis: int,
            keepdim: bool,
        ) -> torch.Tensor:
            var = inputs.mul(inputs).mean(normalization_axis, keepdim)
            pre_shift_scale_norm_output = inputs / torch.sqrt(var + 1e-12)
            norm_output = weight * pre_shift_scale_norm_output
            return norm_output

        def nvfuser_fusion(
            fd: FusionDefinition,
            normalization_axis: int,
            norm_size: int,
            input_shape: List[int],
            eps: float,
            keepDim: bool
        ) -> None :
            inputs = fd.define_tensor(symbolic_sizes=[-1, -1, -1], contiguous=[True, True, True], dtype=DataType.Float)
            weights = fd.define_tensor(symbolic_sizes=[-1], contiguous=[True], dtype=DataType.Float)
            inputs_sq = fd.ops.mul(inputs, inputs)
            sum0 = fd.ops.sum(inputs_sq, axes=[normalization_axis], keepdim=keepDim)
            norm_const = fd.define_constant(norm_size)
            var = fd.ops.div(sum0, norm_const)
            eps_const = fd.define_constant(eps)
            var_eps = fd.ops.add(var, eps_const)
            invstd = fd.ops.rsqrt(var_eps)
            pre_scale = fd.ops.mul(inputs, invstd)
            weights_bcast = fd.ops.broadcast_in_dim(weights, output_shape=input_shape, broadcast_dims=[2])
            out = fd.ops.mul(pre_scale, weights_bcast)
            fd.add_output(out)
            fd.add_output(invstd)

        fusion_func = partial(nvfuser_fusion,
                              normalization_axis=2,
                              norm_size=inputs[0].size()[2],
                              input_shape=inputs[0].size(),
                              eps=1e-12,
                              keepDim=True)
        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)

        eager_out = primitive_definition(inputs[0], inputs[1], 2, True)

        self.assertEqual(eager_out, nvf_out[0])

    # Testing a scenario where a broadcast requires a symbolic output shape
    def test_tensor_sizes(self) :
        def fusion_func(fd : FusionDefinition) :
            t0 = fd.define_tensor(3)
            t1 = fd.define_tensor(1)

            t0_sizes = fd.ops.tensor_sizes(t0)

            t1_b = fd.ops.broadcast_in_dim(t1, t0_sizes, [2])
            t2 = fd.ops.add(t0, t1_b)

            fd.add_output(t2)

        inputs = [
            torch.randn(2, 3, 4, device='cuda'),
            torch.randn(4, device='cuda'),
        ]

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)
        eager_out = refs.add(inputs[0], prims.broadcast_in_dim(inputs[1], inputs[0].size(), [2]))
        self.assertEqual(eager_out, nvf_out[0])

    # Testing a scenario where no broadcast is needed
    def test_tensor_sizes_nobcast(self) :
        def fusion_func(fd : FusionDefinition) :
            t0 = fd.define_tensor(symbolic_sizes=[-1, -1], contiguous=[True, True])
            t1 = fd.define_tensor(symbolic_sizes=[-1, -1], contiguous=[True, True])

            t0_sizes = fd.ops.tensor_sizes(t0)

            t1_b = fd.ops.broadcast_in_dim(t1, t0_sizes, [0, 1])
            t2 = fd.ops.add(t0, t1_b)

            fd.add_output(t2)

        inputs = [
            torch.randn(2, 3, device='cuda'),
            torch.randn(2, 3, device='cuda'),
        ]

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)
        eager_out = refs.add(inputs[0], prims.broadcast_in_dim(inputs[1], inputs[0].size(), [0, 1]))
        self.assertEqual(eager_out, nvf_out[0])

    # Testing a scenario where each arg of a binary op has broadcast.
    def test_tensor_sizes_both_args_bcast(self) :
        def fusion_func(fd : FusionDefinition) :
            t0 = fd.define_tensor(symbolic_sizes=[1, -1], contiguous=[True, True])
            t1 = fd.define_tensor(symbolic_sizes=[-1, 1], contiguous=[True, True])

            t0_sizes = fd.ops.tensor_sizes(t0)
            t1_sizes = fd.ops.tensor_sizes(t1)

            t0_b = fd.ops.broadcast_in_dim(t0, [t1_sizes[0], t0_sizes[1]], [0, 1])
            t1_b = fd.ops.broadcast_in_dim(t1, [t1_sizes[0], t0_sizes[1]], [0, 1])
            t2 = fd.ops.add(t0_b, t1_b)

            fd.add_output(t2)

        inputs = [
            torch.randn(1, 3, device='cuda'),
            torch.randn(2, 1, device='cuda'),
        ]

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)
        eager_out = refs.add(prims.broadcast_in_dim(inputs[0], [inputs[1].size()[0], inputs[0].size()[1]], [0, 1]),
                             prims.broadcast_in_dim(inputs[1], [inputs[1].size()[0], inputs[0].size()[1]], [0, 1]))
        self.assertEqual(eager_out, nvf_out[0])

    def test_broadcast_in_dim_with_dynamic_shapes(self) :
        inputs_1 = [
            torch.randn(2, 3, 4, device='cuda'),
            torch.randn(4, device='cuda'),
        ]
        inputs_2 = [
            torch.randn(2, 3, 1024, device='cuda'),
            torch.randn(1024, device='cuda'),
        ]

        def fusion_func_1(fd : FusionDefinition) :
            t0 = fd.define_tensor(symbolic_sizes=[-1, -1, -1], contiguous=[True, True, True])
            t1 = fd.define_tensor(symbolic_sizes=[-1], contiguous=[True])

            t0_sizes = fd.ops.tensor_sizes(t0)

            t1_b = fd.ops.broadcast_in_dim(t1, t0_sizes, [2])
            t2 = fd.ops.add(t0, t1_b)

            fd.add_output(t2)

        def fusion_func_2(fd : FusionDefinition) :
            t0 = fd.define_tensor(symbolic_sizes=[-1, -1, -1], contiguous=[True, True, True])
            t1 = fd.define_tensor(symbolic_sizes=[-1], contiguous=[True])

            t1_b = fd.ops.broadcast_in_dim(t1, inputs_1[0].size(), [2])
            t2 = fd.ops.add(t0, t1_b)

            fd.add_output(t2)

        def fusion_func_3(fd : FusionDefinition) :
            t0 = fd.define_tensor(symbolic_sizes=[-1, -1, -1], contiguous=[True, True, True])
            t1 = fd.define_tensor(symbolic_sizes=[-1], contiguous=[True])

            t1_b = fd.ops.broadcast_in_dim(t1, inputs_2[0].size(), [2])
            t2 = fd.ops.add(t0, t1_b)

            fd.add_output(t2)

        # Func_1 uses tensor_sizes to propagate dynamic size, therefore, it is
        # expected that test 2 should be cached based on test 2

        # Test 1
        inputs = inputs_1
        nvf_out, _ = self.exec_nvfuser(fusion_func_1, inputs)
        eager_out = refs.add(inputs[0], prims.broadcast_in_dim(inputs[1], inputs[0].size(), [2]))
        self.assertEqual(eager_out, nvf_out[0])

        # Test 2
        inputs = inputs_2
        nvf_out, _ = self.exec_nvfuser(fusion_func_1, inputs, new_fusion_expected=False)
        eager_out = refs.add(inputs[0], prims.broadcast_in_dim(inputs[1], inputs[0].size(), [2]))
        self.assertEqual(eager_out, nvf_out[0])

        # Func_2 and Func_3 are nearly identical except that have a different
        # concrete output shape for their broadcast_in_dim.  Therefore, test 4
        # should not be cached.
        # Note: It is assumed that definition will change with Tensor Size with
        # concrete shapes.

        # Test 3
        inputs = inputs_1
        nvf_out, _ = self.exec_nvfuser(fusion_func_2, inputs)
        eager_out = refs.add(inputs[0], prims.broadcast_in_dim(inputs[1], inputs[0].size(), [2]))
        self.assertEqual(eager_out, nvf_out[0])

        # Test 4
        inputs = inputs_2
        nvf_out, _ = self.exec_nvfuser(fusion_func_3, inputs)
        eager_out = refs.add(inputs[0], prims.broadcast_in_dim(inputs[1], inputs[0].size(), [2]))
        self.assertEqual(eager_out, nvf_out[0])

    # Testing a scenario where the broadcast is necessary to realize the output
    def test_tensor_sizes_with_output_bcast(self) :
        def fusion_func(fd : FusionDefinition) :
            t0 = fd.define_tensor(symbolic_sizes=[-1, -1, -1], contiguous=[True, True, True])
            t0_sizes = fd.ops.tensor_sizes(t0)

            t1 = fd.ops.sum(t0, axes=[2])
            t1_b = fd.ops.broadcast_in_dim(t1, t0_sizes, [0, 1])

            fd.add_output(t1_b)

        inputs_1 = [
            torch.randn(2, 3, 4, device='cuda'),
        ]

        inputs_2 = [
            torch.randn(4, 5, 32, device='cuda'),
        ]

        inputs = inputs_1
        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)
        eager_out = prims.broadcast_in_dim(torch.sum(inputs[0], dim=-1), inputs[0].size(), [0, 1])
        self.assertEqual(eager_out, nvf_out[0])

        # Testing Dynamic usage of same Fusion
        inputs = inputs_2
        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs, new_fusion_expected=False)
        eager_out = prims.broadcast_in_dim(torch.sum(inputs[0], dim=-1), inputs[0].size(), [0, 1])
        self.assertEqual(eager_out, nvf_out[0])

    # Testing an expand followed by a  broadcast
    def test_tensor_sizes_expand_bcast(self) :
        def fusion_func(fd : FusionDefinition) :
            t0 = fd.define_tensor(symbolic_sizes=[-1, -1, -1], contiguous=[True, True, True])
            t1 = fd.define_tensor(symbolic_sizes=[-1, 1, -1], contiguous=[True, True, True])
            t2 = fd.define_tensor(symbolic_sizes=[-1, 1, -1], contiguous=[True, True, True])
            t0_sizes = fd.ops.tensor_sizes(t0)

            t1_b = fd.ops.broadcast_in_dim(t1, t0_sizes, [0, 1, 2])
            t1_b_sizes = fd.ops.tensor_sizes(t1_b)
            t2_b = fd.ops.broadcast_in_dim(t2, t1_b_sizes, [0, 1, 2])

            fd.add_output(t2_b)

        inputs = [
            torch.randn(2, 3, 4, device='cuda'),
            torch.randn(2, 1, 4, device='cuda'),
            torch.randn(2, 1, 4, device='cuda'),
        ]

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)
        eager_out1 = prims.broadcast_in_dim(inputs[1], inputs[0].size(), [0, 1, 2])
        eager_out2 = prims.broadcast_in_dim(inputs[2], eager_out1.size(), [0, 1, 2])
        self.assertEqual(eager_out2, nvf_out[0])

    def test_alias_output_to_input(self) :
        def fusion_func(fd : FusionDefinition) :
            t0 = fd.define_tensor(2)
            s0 = fd.define_constant(1.0)
            s1 = fd.define_constant(2.0)
            s2 = fd.define_constant(3.0)
            t1 = fd.ops.add(t0, s0)
            t2 = fd.ops.add(t0, s1)
            t3 = fd.ops.add(t2, s2)
            fd.add_output(t1)
            fd.add_output(t2, alias_input=t0)
            fd.add_output(t3)

        inputs = [
            torch.ones(4, 4, device='cuda'),
        ]

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)

        eager_out1 = torch.add(torch.ones(4, 4, device='cuda'), 1.0)
        eager_out2 = torch.add(torch.ones(4, 4, device='cuda'), 2.0)
        eager_out3 = torch.add(eager_out2, 3.0)
        self.assertEqual(eager_out1, nvf_out[0])
        self.assertEqual(eager_out2, inputs[0])
        self.assertEqual(eager_out3, nvf_out[1])

    def test_gather(self):
        inputs = [
            torch.randn(8, 16, device='cuda'),
            torch.randn(8, 16, device='cuda'),
            torch.randint(0, 8, (4, 4), device="cuda").to(dtype=torch.long)
        ]

        def fusion_func(fd: FusionDefinition) :
            t0 = fd.define_tensor(2)
            t1 = fd.define_tensor(2)
            t2 = fd.define_tensor(2, DataType.Int)
            t3 = fd.ops.add(t0, t1)
            t4 = fd.ops.gather(t3, t2, 0)
            fd.add_output(t4)

        nvf_out1, _ = self.exec_nvfuser(fusion_func, inputs)
        # Create a new fusion with the same definition, it should hit the cache!
        nvf_out2, fs2 = self.exec_nvfuser(fusion_func, inputs, new_fusion_expected=False)
        # Create a fusion from a fusion id and make sure it executes!
        fs3 = Fusion(fs2.id())
        nvf_out3 = fs3.execute(inputs)

        eager_out = torch.gather(inputs[0] + inputs[1], 0, inputs[2])
        self.assertEqual(eager_out, nvf_out1[0])
        self.assertEqual(eager_out, nvf_out2[0])
        self.assertEqual(eager_out, nvf_out3[0])

if __name__ == '__main__':
    run_tests()
