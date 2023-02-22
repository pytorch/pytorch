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
    from nvfuser import FusionCache, FusionDefinition, DataType, version
    from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype
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
        with FusionDefinition() as fd:
            fusion_func(fd)
        fd_str = fd.__repr__()
        out = fd.execute(inputs)

        # Execute the python definition that was captured
        func_name = re.findall('(nvfuser_fusion_id\\d+)', fd_str.split('\n')[1])[0]
        exec(fd_str)
        with FusionDefinition() as fd_cap:
            eval(func_name)(fd_cap)
        out_cap = fd_cap.execute(inputs_cap)

        # Make sure the original and captured definitions match
        for idx in range(len(out)) :
            self.assertEqual(out[idx], out_cap[idx])

        self.assertEqual(fc.num_fusions() - before_fusions, int(new_fusion_expected))
        return out, fd

    def test_basic(self) :
        inputs = [
            torch.ones(2, 4, 8, device='cuda'),
            torch.ones(2, 4, 8, device='cuda'),
        ]

        def fusion_func(fd: FusionDefinition) :
            t0 = fd.from_pytorch(inputs[0])
            t1 = fd.from_pytorch(inputs[1])
            c0 = fd.define_constant(3.0)

            t2 = fd.ops.add(t0, t1)
            t3 = fd.ops.mul(t2, c0)
            t4 = fd.ops.sum(t3, [-1], False, DataType.Float)

            fd.add_output(t4)

        # Expected Output is a tensor of 48's
        nvf_out1, _ = self.exec_nvfuser(fusion_func, inputs)

        # Create a new fusion with the same definition, it should hit the cache!
        nvf_out2, fd2 = self.exec_nvfuser(fusion_func, inputs, new_fusion_expected=False)

        # Create a fusion from a fusion id and make sure it executes!
        fd3 = FusionDefinition(fd2.id())
        nvf_out3 = fd3.execute(inputs)

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
            t0 = fd.from_pytorch(inputs[0])
            t1 = fd.from_pytorch(inputs[1])
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
            t0 = fd.from_pytorch(inputs[0])
            t1 = fd.from_pytorch(inputs[1])

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
            t0 = fd.from_pytorch(inputs[0])
            t1 = fd.from_pytorch(inputs[1])

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
            t0 = fd.from_pytorch(inputs[0])
            t1 = fd.from_pytorch(inputs[1])

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
            t0 = fd.from_pytorch(inputs[0])
            t1 = fd.from_pytorch(inputs[1])

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
            t0 = fd.from_pytorch(inputs[0])
            t1 = fd.from_pytorch(inputs[1])

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
            t0 = fd.from_pytorch(inputs[0])
            t1 = fd.from_pytorch(inputs[1])

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
        inputs = [
            torch.randn(2, 3, 4, device='cuda'),
            torch.randn(4, device='cuda'),
        ]

        def fusion_func(fd : FusionDefinition) :
            t0 = fd.from_pytorch(inputs[0])
            t1 = fd.from_pytorch(inputs[1])

            t0_sizes = fd.ops.tensor_sizes(t0)

            t1_b = fd.ops.broadcast_in_dim(t1, t0_sizes, [2])
            t2 = fd.ops.sub(t0, t1_b)

            fd.add_output(t2)

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)
        eager_out = refs.sub(inputs[0], prims.broadcast_in_dim(inputs[1], inputs[0].size(), [2]))
        self.assertEqual(eager_out, nvf_out[0])

    # Testing a scenario where no broadcast is needed
    def test_tensor_sizes_nobcast(self) :
        inputs = [
            torch.randn(2, 3, device='cuda'),
            torch.randn(2, 3, device='cuda'),
        ]

        def fusion_func(fd : FusionDefinition) :
            t0 = fd.from_pytorch(inputs[0])
            t1 = fd.from_pytorch(inputs[1])

            t0_sizes = fd.ops.tensor_sizes(t0)

            t1_b = fd.ops.broadcast_in_dim(t1, t0_sizes, [0, 1])
            t2 = fd.ops.add(t0, t1_b)

            fd.add_output(t2)

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)
        eager_out = refs.add(inputs[0], prims.broadcast_in_dim(inputs[1], inputs[0].size(), [0, 1]))
        self.assertEqual(eager_out, nvf_out[0])

    # Testing a scenario where each arg of a binary op has broadcast.
    def test_tensor_sizes_both_args_bcast(self) :
        inputs = [
            torch.randn(1, 3, device='cuda'),
            torch.randn(2, 1, device='cuda'),
        ]

        def fusion_func(fd : FusionDefinition) :
            t0 = fd.from_pytorch(inputs[0])
            t1 = fd.from_pytorch(inputs[1])

            t0_sizes = fd.ops.tensor_sizes(t0)
            t1_sizes = fd.ops.tensor_sizes(t1)

            t0_b = fd.ops.broadcast_in_dim(t0, [t1_sizes[0], t0_sizes[1]], [0, 1])
            t1_b = fd.ops.broadcast_in_dim(t1, [t1_sizes[0], t0_sizes[1]], [0, 1])
            t2 = fd.ops.add(t0_b, t1_b)

            fd.add_output(t2)

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
        inputs = [
            torch.ones(4, 4, device='cuda'),
        ]

        def fusion_func(fd : FusionDefinition) :
            t0 = fd.from_pytorch(inputs[0])
            s0 = fd.define_constant(1.0)
            s1 = fd.define_constant(2.0)
            s2 = fd.define_constant(3.0)
            t1 = fd.ops.add(t0, s0)
            t2 = fd.ops.add(t0, s1)
            t3 = fd.ops.add(t2, s2)
            fd.add_output(t1)
            fd.add_output(t2, alias_input=t0)
            fd.add_output(t3)

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
            t0 = fd.from_pytorch(inputs[0])
            t1 = fd.from_pytorch(inputs[1])
            t2 = fd.from_pytorch(inputs[2])
            t3 = fd.ops.add(t0, t1)
            t4 = fd.ops.gather(t3, t2, 0)
            fd.add_output(t4)

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)

        eager_out = torch.gather(inputs[0] + inputs[1], 0, inputs[2])
        self.assertEqual(eager_out, nvf_out[0])

    def test_index_select(self):
        inputs = [
            torch.randn(8, 16, device='cuda'),
            torch.randn(8, 16, device='cuda'),
            torch.randint(0, 8, (6,), device="cuda").to(dtype=torch.long)
        ]

        def fusion_func(fd: FusionDefinition) :
            t0 = fd.from_pytorch(inputs[0])
            t1 = fd.from_pytorch(inputs[1])
            t2 = fd.from_pytorch(inputs[2])
            t3 = fd.ops.add(t0, t1)
            t4 = fd.ops.index_select(t3, t2, 0)
            fd.add_output(t4)

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)

        eager_out = torch.index_select(inputs[0] + inputs[1], 0, inputs[2])
        self.assertEqual(eager_out, nvf_out[0])

    def test_squeeze(self) :
        t0_sizes = [4]
        t1_sizes = [1, 4, 1]
        t2_sizes = [2, 1, 4]
        inputs = [
            torch.randn(*t0_sizes, device='cuda'),
            torch.randn(*t1_sizes, device='cuda'),
            torch.randn(*t2_sizes, device='cuda'),
        ]

        def fusion_func(fd : FusionDefinition) :
            t0 = fd.define_tensor(symbolic_sizes=[-1], contiguous=[True])
            t1 = fd.define_tensor(sizes=t1_sizes, strides=[4, 1, 1])
            t2 = fd.define_tensor(sizes=t2_sizes, strides=[4, 4, 1])
            t3 = fd.ops.squeeze(t1, t1_sizes, [0, -1])
            t4 = fd.ops.squeeze(t2, t2_sizes, [-2,])
            t5 = fd.ops.sum(t4, [0])
            t6 = fd.ops.mul(t0, t3)
            t7 = fd.ops.mul(t6, t5)
            fd.add_output(t7)

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)

        v1 = torch.sum(inputs[1], [0, -1])
        v2 = torch.sum(inputs[2], [0, 1])
        eager_out = inputs[0] * v1 * v2
        self.assertEqual(eager_out, nvf_out[0])

    def test_from_pytorch_fails_on_cpu_tensor(self):
        inputs = [
            torch.randn(4, 4, device='cpu'),
        ]

        def fusion_func(fd: FusionDefinition) :
            t0 = fd.from_pytorch(inputs[0])
            t1 = fd.ops.relu(t0)
            fd.add_output(t1)

        try:
            with FusionDefinition() as fd:
                fusion_func(fd)
            raise RuntimeError("FusionDefinition.from_pytorch should have raised an error for a CPU Tensor!")
        except ValueError:
            pass

    def test_no_definition(self):
        inputs = [
            torch.randn(4, 4, device='cpu'),
        ]

        # A FusionDefinition object is constructed but not defined, should trip an error
        try:
            fd = FusionDefinition()
            out = fd.execute(inputs)
            raise RuntimeError("Expecting an error for a lack of a child class defining a definition!")
        except NotImplementedError:
            pass

    def test_func_definition(self) :
        inputs = [
            torch.randn(4, 4, device='cuda'),
        ]

        class MyFusion(FusionDefinition):
            def definition(self) :
                t0 = self.from_pytorch(inputs[0])
                t1 = self.ops.sigmoid(t0)
                self.add_output(t1)

        fd = MyFusion()
        nvf_out = fd.execute(inputs)
        eager_out = torch.sigmoid(inputs[0])
        self.assertEqual(eager_out, nvf_out[0])

    def test_python_version_API(self):
        from nvfuser.nvfuser_version import Version
        self.assertTrue(version() > '0.0.0')
        self.assertTrue(version() > Version('0.0.0'))
    
    def test_def_and_sched_func_errors (self) :
        inputs = [
            torch.randn(4, 4, 4, device='cuda'),
        ]

        class DefError(FusionDefinition):
            def definition(self) :
                t0 = self.from_pytorch(inputs[0])
                t1 = self.ops.tanh(t0)
                self.add_output(t1)
                self.sched.merge(t1, 1)

        try:
            fd = DefError()
            out = fd.execute(inputs)
        except RuntimeError:
            pass
        
        class SchedError(FusionDefinition):
            def definition(self) :
                self.t0 = self.from_pytorch(inputs[0])
                self.t1 = self.ops.tanh(self.t0)
                self.add_output(self.t1)

            def schedule(self) :
                self.t2 = self.ops.relu(self.t1)

        try:
            fd = SchedError()
            out = fd.execute(inputs)
        except RuntimeError:
            pass
    
    def test_basic_user_schedule (self) :
        inputs = [
            torch.randn(4, 4, 4, device='cuda'),
            torch.randn(4, 4, 4, device='cuda'),
        ]

        class UserDefSched(FusionDefinition):
            def definition(self):
                self.t0 = self.from_pytorch(inputs[0])
                self.t1 = self.from_pytorch(inputs[1])
                self.t2 = self.ops.add(self.t0, self.t1)
                self.add_output(self.t2)

            def schedule(self):
                self.sched.split(self.t2, 1, 2)
                self.sched.merge(self.t2, -2)

        fd = UserDefSched()
        nvf_user_out = fd.execute(inputs)
        nvf_out = fd.execute(inputs, override_user_schedule=True)
        self.assertEqual(nvf_user_out, nvf_out)

    def test_where_dtypes(self):
        inputs = [
            torch.arange(2, device="cuda").type(torch.bool),
        ]

        def fusion_func(fd: FusionDefinition):
            t0 = fd.from_pytorch(inputs[0])

            c0 = fd.define_constant(3.0)
            c1 = fd.define_constant(5.0)
            t1 = fd.ops.where(t0, c0, c1)  # DataType.Double
            fd.add_output(t1)

            c0f = fd.define_constant(3.0, DataType.Float)
            c1f = fd.define_constant(5.0, DataType.Float)
            t1f = fd.ops.where(t0, c0f, c1f)  # DataType.Float
            fd.add_output(t1f)

            c0d = fd.define_constant(3.0, DataType.Double)
            c1d = fd.define_constant(5.0, DataType.Double)
            t1d = fd.ops.where(t0, c0d, c1d)  # DataType.Double
            fd.add_output(t1d)

            c0i = fd.define_constant(3, DataType.Int32)
            c1i = fd.define_constant(5, DataType.Int32)
            t1i = fd.ops.where(t0, c0i, c1i)  # DataType.Int32
            fd.add_output(t1i)

            c0l = fd.define_constant(3)
            c1l = fd.define_constant(5)
            t1l = fd.ops.where(t0, c0l, c1l)  # DataType.Int
            fd.add_output(t1l)

            c0c = fd.define_constant(complex(3.0))
            c1c = fd.define_constant(complex(5.0))
            t1c = fd.ops.where(t0, c0c, c1c) # DataType.ComplexDouble
            fd.add_output(t1c)

            c0cf = fd.define_constant(3.0 + 0j, DataType.ComplexFloat)
            c1cf = fd.define_constant(5.0 + 0j, DataType.ComplexFloat)
            t1cf = fd.ops.where(t0, c0cf, c1cf) # DataType.ComplexFloat
            fd.add_output(t1cf)

            c0cd = fd.define_constant(3.0 + 0j, DataType.ComplexDouble)
            c1cd = fd.define_constant(5.0 + 0j, DataType.ComplexDouble)
            t1cd = fd.ops.where(t0, c0cd, c1cd) # DataType.ComplexDouble
            fd.add_output(t1cd)

            c0b = fd.define_constant(True, DataType.Bool)
            c1b = fd.define_constant(False, DataType.Bool)
            t1b = fd.ops.where(t0, c0b, c1b)  # DataType.Bool
            fd.add_output(t1b)
        
        (
            n,
            nf,
            nd,
            ni,
            nl,
            nc,
            ncf,
            ncd,
            nb,
        ), _ = self.exec_nvfuser(fusion_func, inputs)

        eager_out = torch.where(inputs[0], 3.0, 5.0)

        # explicit Float dtype matches torch.where behavior
        self.assertEqual(eager_out, nf)

        assert n.dtype == torch.float64
        assert nf.dtype == torch.float32
        assert nd.dtype == torch.float64
        assert ni.dtype == torch.int32
        assert nl.dtype == torch.int64
        assert nc.dtype == torch.complex128
        assert ncf.dtype == torch.complex64
        assert ncd.dtype == torch.complex128
        assert nb.dtype == torch.bool

    def test_complex_constants(self):
        inputs = [
            torch.arange(2, device="cuda").type(torch.complex64),
        ]

        def fusion_func(fd: FusionDefinition):
            t0 = fd.from_pytorch(inputs[0])
            c0 = fd.define_constant(complex(3.0, 0.5))
            t1 = fd.ops.mul(t0, c0)
            fd.add_output(t1)

        (n,), _ = self.exec_nvfuser(fusion_func, inputs)

        eager_out = inputs[0] * (3.0 + 0.5j)

        self.assertEqual(eager_out, n)
        assert n.dtype == torch.complex64

    def test_where_op(self):
        def nvfuser_where(pred, a, b):
            with FusionDefinition() as fd:
                nv_pred = fd.define_tensor(sizes=pred.shape, strides=pred.stride(), dtype=DataType.Bool)
                nv_a = fd.define_tensor(sizes=a.shape, strides=a.stride(), dtype=torch_dtype_to_nvfuser_dtype(a.dtype))
                nv_b = fd.define_tensor(sizes=b.shape, strides=b.stride(), dtype=torch_dtype_to_nvfuser_dtype(b.dtype))
                result = fd.ops.where(nv_pred, nv_a, nv_b)
                fd.add_output(result)
            return fd.execute((pred, a, b))[0]
        pred = torch.testing.make_tensor((5,), device='cuda', dtype=torch.bool)
        list_of_dtype = [torch.float16, torch.bfloat16, torch.float32]
        for atype in list_of_dtype:
            for btype in list_of_dtype:
                a = torch.randn((5,), device='cuda', dtype=atype)
                b = torch.randn((5,), device='cuda', dtype=btype)
                nv_result = nvfuser_where(pred, a, b)
                torch_result = torch.where(pred, a, b)
                self.assertEqual(nv_result, torch_result)

    def test_iota(self):
        inputs = [
            (2, 0, 2, DataType.Int),
            (3, 100, 1, DataType.Int32),
            # TODO: How do I that that? I am getting the following error:
            # NameError: name 'None0' is not defined
            # (4, None, None, DataType.Int),
        ]

        def fusion_func(fd: FusionDefinition):
            for input in inputs:
                c0 = fd.define_constant(input[0])
                c1 = None if input[1] is None else fd.define_constant(input[1])
                c2 = None if input[2] is None else fd.define_constant(input[2])
                dt = input[3]
                t3 = fd.ops.iota(c0, c1, c2, dt)
                fd.add_output(t3)

        nvf_out, _ = self.exec_nvfuser(fusion_func, [])

        eager_out1 = torch.tensor([0, 2], dtype=torch.long, device="cuda")
        eager_out2 = torch.tensor([100, 101, 102], dtype=torch.int, device="cuda")
        eager_out3 = torch.tensor([0, 1, 2, 3], dtype=torch.long, device="cuda")
        self.assertEqual(eager_out1, nvf_out[0])
        self.assertEqual(eager_out2, nvf_out[1])
        # self.assertEqual(eager_out3, nvf_out[2])

    def test_complex_rsqrt(self):
        inputs = [
            torch.randn(4, device="cuda", dtype=torch.complex64),
            torch.randn(4, device="cuda", dtype=torch.complex128),
        ]

        def fusion_func(fd: FusionDefinition):
            t0 = fd.from_pytorch(inputs[0])
            t1 = fd.from_pytorch(inputs[1])
            t2 = fd.ops.rsqrt(t0)
            fd.add_output(t2)
            t3 = fd.ops.rsqrt(t1)
            fd.add_output(t3)

        (rfloat, rdouble), _ = self.exec_nvfuser(fusion_func, inputs)

        at_rfloat = inputs[0].rsqrt()
        at_rdouble = inputs[1].rsqrt()

        self.assertEqual(at_rfloat, rfloat)
        self.assertEqual(at_rdouble, rdouble)

    def test_all_dim_var_mean(self):
        inputs = [
            torch.randn(2, 2, 2, device='cuda')
        ]

        def fuser_function(correction):
            with FusionDefinition() as fd:
                t0 = fd.from_pytorch(inputs[0])
                t1,t2 = fd.ops.var_mean(t0, [0, 1, 2], correction)
                fd.add_output(t1)
                fd.add_output(t2)
            return fd.execute(inputs)
        list_of_test_cases = [0, 1]
        for correction in list_of_test_cases:
            fuser_result = fuser_function(correction)
            torch_result = torch.var_mean(inputs[0], [0, 1, 2], bool(correction))
            self.assertEqual(fuser_result, torch_result)

    def test_scalar_only_inputs(self):
        # We don't allow scalar outputs, currently,
        # so a tensor has to be returned
        def fusion_func(fd: FusionDefinition):
            s0 = fd.define_scalar()
            s1 = fd.define_scalar()
            s2 = fd.ops.add(s0, s1)
            c0 = fd.define_constant(1.0, DataType.Float)
            t3 = fd.ops.full(size=[2, 2], arg=c0, dtype=DataType.Float)
            t4 = fd.ops.mul(t3, s2)
            fd.add_output(t4)

        with FusionDefinition() as fd:
            fusion_func(fd)

        # TODO: full is broken and does not print its proper definition
        # Issue: https://github.com/csarofeen/pytorch/issues/2502
        nvf_out = fd.execute([2.0, 3.0])
        eager_out = torch.full([2, 2], 1.0) * 5.0
        self.assertEqual(eager_out, nvf_out[0]) 

if __name__ == '__main__':
    run_tests()
