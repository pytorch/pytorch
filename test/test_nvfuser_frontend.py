# Owner(s): ["module: nvfuser"]

import unittest
from typing import List

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
    def test_basic(self) :
        input1 = torch.ones(2, 4, 8, device='cuda')
        input2 = torch.ones(2, 4, 8, device='cuda')
        fc = FusionCache.get()
        before_fusions = fc.num_fusions()

        fs1 = Fusion()
        with FusionDefinition(fs1) as fd :
            t0 = fd.define_tensor(3)
            t1 = fd.define_tensor(3)
            c0 = fd.define_constant(3.0)

            t2 = fd.ops.add(t0, t1)
            t3 = fd.ops.mul(t2, c0)
            t4 = fd.ops.sum(t3, [-1], False, DataType.Float)

            fd.add_output(t4)

        # Expected Output is a tensor of 48's
        nvf_out1 = fs1.execute([input1, input2])[0]

        # Create a new fusion with the same definition, it should hit the cache!
        fs2 = Fusion()
        with FusionDefinition(fs2) as fd :
            t0 = fd.define_tensor(3)
            t1 = fd.define_tensor(3)
            c0 = fd.define_constant(3.0)

            t2 = fd.ops.add(t0, t1)
            t3 = fd.ops.mul(t2, c0)
            t4 = fd.ops.sum(t3, [-1], False, DataType.Float)

            fd.add_output(t4)

        nvf_out2 = fs2.execute([input1, input2])[0]

        # Check there is still only 1 cache entry
        fc = FusionCache.get()
        self.assertEqual(fc.num_fusions() - before_fusions, 1)

        # Create a fusion from a fusion id and make sure it executes!
        fs3 = Fusion(fs2.id())
        nvf_out3 = fs3.execute([input1, input2])[0]

        eager_out = torch.sum((input1 + input2) * 3.0, dim=-1)
        self.assertEqual(eager_out, nvf_out1)
        self.assertEqual(eager_out, nvf_out2)
        self.assertEqual(eager_out, nvf_out3)

    def test_basic_fp16(self) :
        fs = Fusion()
        with FusionDefinition(fs) as fd :
            t0 = fd.define_tensor(3, DataType.Half)
            t1 = fd.define_tensor(3, DataType.Half)
            c0 = fd.define_constant(3.0)

            t2 = fd.ops.add(t0, t1)
            t3 = fd.ops.mul(t2, c0)
            t4 = fd.ops.sum(t3, [-1], False, DataType.Float)

            t5 = fd.ops.cast(t4, DataType.Half)
            fd.add_output(t5)

        input1 = torch.ones(2, 4, 8, device='cuda', dtype=torch.float16)
        input2 = torch.ones(2, 4, 8, device='cuda', dtype=torch.float16)

        # Expected Output is a tensor of 48's
        nvf_out = fs.execute([input1, input2])[0]
        eager_out = torch.sum((input1 + input2) * 3.0, dim=-1)
        self.assertEqual(eager_out, nvf_out)

    def test_cast_double_to_half(self) :
        fs = Fusion()
        with FusionDefinition(fs) as fd :
            t0 = fd.define_tensor(2, DataType.Double)
            t1 = fd.define_tensor(2, DataType.Double)

            t0h = fd.ops.cast(t0, DataType.Half)
            t1h = fd.ops.cast(t1, DataType.Half)
            t2 = fd.ops.add(t0h, t1h)
            t3 = fd.ops.relu(t2)
            t4 = fd.ops.cast(t3, DataType.Half)

            fd.add_output(t4)

        input1 = torch.randn(2, 4, device='cuda', dtype=torch.float64)
        input2 = torch.randn(2, 4, device='cuda', dtype=torch.float64)

        nvf_out = fs.execute([input1, input2])[0]
        eager_out = torch.relu(input1.to(torch.half) + input2.to(torch.half))
        self.assertEqual(eager_out, nvf_out)

    def test_promote_to_double(self) :
        fs = Fusion()

        with FusionDefinition(fs) as fd :
            t0 = fd.define_tensor(2, DataType.Half)
            t1 = fd.define_tensor(2, DataType.Double)

            t2 = fd.ops.add(t0, t1)
            t5 = fd.ops.relu(t2)

            fd.add_output(t5)

        input1 = torch.randn(2, 4, device='cuda', dtype=torch.float16)
        input2 = torch.randn(2, 4, device='cuda', dtype=torch.float64)

        nvf_out = fs.execute([input1, input2])[0]
        eager_out = torch.relu(input1 + input2)
        self.assertEqual(eager_out, nvf_out)

    def test_implicit_broadcast_input(self) :
        fs = Fusion()
        with FusionDefinition(fs) as fd :
            t0 = fd.define_tensor(1)
            t1 = fd.define_tensor(3)

            t0_b = fd.ops.broadcast_in_dim(t0, [2, 3, 4], [1])
            t2 = fd.ops.add(t0_b, t1)

            fd.add_output(t2)

        input1 = torch.randn(3, device='cuda')
        input2 = torch.randn(2, 3, 4, device='cuda')

        nvf_out = fs.execute([input1, input2])[0]
        eager_out = refs.add(prims.broadcast_in_dim(input1, [2, 3, 4], [1]), input2)
        self.assertEqual(eager_out, nvf_out)

    def test_explicit_broadcast_input(self) :
        input1 = torch.randn(1, 1, 4, device='cuda')
        input2 = torch.randn(2, 3, 4, device='cuda')

        fs = Fusion()
        with FusionDefinition(fs) as fd :
            t0 = fd.define_tensor(sizes=input1.size(), strides=input1.stride())
            t1 = fd.define_tensor(sizes=input2.size(), strides=input2.stride())

            t0_b = fd.ops.broadcast_in_dim(t0, [2, 3, 4], [0, 1, 2])
            t2 = fd.ops.add(t0_b, t1)

            fd.add_output(t2)

        nvf_out = fs.execute([input1, input2])[0]
        eager_out = refs.add(prims.broadcast_in_dim(input1, [2, 3, 4], [0, 1, 2]), input2)
        self.assertEqual(eager_out, nvf_out)

    def test_broadcast_mixing(self) :
        fs = Fusion()
        with FusionDefinition(fs) as fd :
            t0 = fd.define_tensor([3, 1], [1, 1])
            t1 = fd.define_tensor(1)

            t1_b = fd.ops.broadcast_in_dim(t1, [3, 3], [0])
            t2 = fd.ops.add(t0, t1_b)

            fd.add_output(t2)

        input1 = torch.randn(3, 1, device='cuda')
        input2 = torch.randn(3, device='cuda')

        nvf_out = fs.execute([input1, input2])[0]
        eager_out = refs.add(input1, prims.broadcast_in_dim(input2, [3, 3], [0]))
        self.assertEqual(eager_out, nvf_out)

    def test_ops_broadcast(self) :
        fs = Fusion()
        with FusionDefinition(fs) as fd :
            t0 = fd.define_tensor(1)
            t1 = fd.define_tensor(3)

            t0_b = fd.ops.broadcast(t0, [True, False, True])
            t2 = fd.ops.add(t0_b, t1)

            fd.add_output(t2)

        input1 = torch.randn(3, device='cuda')
        input2 = torch.randn(2, 3, 4, device='cuda')

        nvf_out = fs.execute([input1, input2])[0]
        eager_out = refs.add(prims.broadcast_in_dim(input1, [2, 3, 4], [1]), input2)
        self.assertEqual(eager_out, nvf_out)

    def test_prim_layer_norm_fwd(self) :
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

        input_size = [64, 128, 1024]
        dtype = torch.float32
        device = 'cuda'
        inputs = torch.randn(*input_size, device=device, requires_grad=True)
        weights = torch.nn.Parameter(torch.randn(input_size[2], dtype=dtype, device=device))
        biases = torch.nn.Parameter(torch.randn(input_size[2], dtype=dtype, device=device))
        fc = FusionCache.get()
        before_fusions = fc.num_fusions()

        for _ in range(5) :
            nvf_fusion = Fusion()
            with FusionDefinition(nvf_fusion) as fd:
                nvfuser_fusion(fd, 2, inputs.size()[2], inputs.size(), 1e-12, True)
            nvf_out = nvf_fusion.execute([inputs, weights, biases])

        for _ in range(5) :
            nvf_var_mean_fusion = Fusion()
            with FusionDefinition(nvf_var_mean_fusion) as fd:
                nvfuser_fusion_var_mean(fd, 2, inputs.size()[2], inputs.size(), 1e-12, True)
            nvf_var_mean_out = nvf_var_mean_fusion.execute([inputs, weights, biases])

        for _ in range(5) :
            eager_out = primitive_definition(inputs, weights, biases, 2, True)

        self.assertEqual(eager_out, nvf_out[0])
        self.assertEqual(eager_out, nvf_var_mean_out[0])
        fusion_cache = FusionCache.get()
        self.assertEqual(fc.num_fusions() - before_fusions, 2)

    def test_prim_rms_norm_fwd(self) :
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

        input_size = [64, 128, 1024]
        dtype = torch.float32
        device = 'cuda'
        inputs = torch.randn(*input_size, device=device, requires_grad=True)
        weights = torch.nn.Parameter(torch.randn(input_size[2], dtype=dtype, device=device))
        fc = FusionCache.get()
        before_fusions = fc.num_fusions()

        for _ in range(5) :
            nvf_fusion = Fusion()
            with FusionDefinition(nvf_fusion) as fd:
                nvfuser_fusion(fd, 2, inputs.size()[2], inputs.size(), 1e-12, True)
            nvf_out = nvf_fusion.execute([inputs, weights])

        for _ in range(5) :
            eager_out = primitive_definition(inputs, weights, 2, True)

        self.assertEqual(eager_out, nvf_out[0])
        self.assertEqual(fc.num_fusions() - before_fusions, 1)

if __name__ == '__main__':
    run_tests()
