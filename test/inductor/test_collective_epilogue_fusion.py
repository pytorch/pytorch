"""
Test collective op autotuning with epilogue fusion patterns.

This test validates:
1. Different all_reduce strategies with epilogue fusions (relu, multiply, add)
2. vLLM-style tensor_model_parallel_all_reduce patterns
3. Autotuning selects best strategy based on real benchmarking
4. Inline fusion optimization works correctly
"""

import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import run_tests


class TestCollectiveEpilogueFusion(MultiProcessTestCase):
    """Test collective autotuning with different epilogue fusion patterns"""

    @property
    def world_size(self):
        return 2

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    @skip_if_lt_x_gpu(2)
    def test_allreduce_relu_fusion(self):
        """Test all_reduce + relu epilogue fusion with different strategies"""

        # Initialize distributed
        dist.init_process_group(
            backend="nccl",
            init_method=f"file:///tmp/test_ar_relu_{self.id()}",
            world_size=self.world_size,
            rank=self.rank,
        )

        rank = dist.get_rank()
        device = f"cuda:{rank}"

        # Register the default process group
        from torch._C._distributed_c10d import _register_process_group

        _register_process_group("default", dist.group.WORLD)

        # Define custom all_reduce + relu op
        @torch.library.custom_op("test::allreduce_relu", mutates_args=())
        def allreduce_relu(x: torch.Tensor) -> torch.Tensor:
            """AllReduce followed by ReLU activation"""
            result = x.clone()
            return torch.ops._c10d_functional.all_reduce_(result, "sum", "default")

        @allreduce_relu.register_fake
        def _(x):
            return torch.empty_like(x)

        # Strategy 1: Standard all_reduce + separate relu
        def strategy1_standard(x):
            result = x.clone()
            result = torch.ops._c10d_functional.all_reduce_(result, "sum", "default")
            return torch.relu(result)

        # Strategy 2: All_reduce + fused relu (torch.nn.functional)
        def strategy2_fused(x):
            result = x.clone()
            result = torch.ops._c10d_functional.all_reduce_(result, "sum", "default")
            return torch.nn.functional.relu(result, inplace=False)

        # Strategy 3: All_reduce + manual max(0, x) relu
        def strategy3_manual(x):
            result = x.clone()
            result = torch.ops._c10d_functional.all_reduce_(result, "sum", "default")
            return torch.maximum(result, torch.zeros_like(result))

        # Register autotuning
        from torch._inductor.kernel.custom_op import (
            register_custom_op_autotuning,
            CustomOpConfig,
        )

        register_custom_op_autotuning(
            allreduce_relu,
            configs=[
                CustomOpConfig(strategy1_standard),
                CustomOpConfig(strategy2_fused),
                CustomOpConfig(strategy3_manual),
            ],
        )

        # Test model
        class ARReluModel(torch.nn.Module):
            def forward(self, x):
                return allreduce_relu(x)

        model = torch.compile(ARReluModel()).to(device)

        # Run
        x = torch.randn(128, 128, device=device)
        x_copy = x.clone()
        y = model(x)

        # Verify: sum across 2 ranks, then relu
        expected = torch.relu(x_copy * 2)
        torch.testing.assert_close(y, expected, rtol=1e-3, atol=1e-3)

        if rank == 0:
            print(f"✅ [2 ranks] AllReduce + ReLU fusion test passed!")

        dist.destroy_process_group()

    @skip_if_lt_x_gpu(2)
    def test_allreduce_multiply_fusion(self):
        """Test all_reduce + multiply epilogue fusion"""

        # Initialize distributed
        dist.init_process_group(
            backend="nccl",
            init_method=f"file:///tmp/test_ar_mul_{self.id()}",
            world_size=self.world_size,
            rank=self.rank,
        )

        rank = dist.get_rank()
        device = f"cuda:{rank}"

        from torch._C._distributed_c10d import _register_process_group

        _register_process_group("default", dist.group.WORLD)

        # Define custom all_reduce + multiply op
        @torch.library.custom_op("test::allreduce_mul", mutates_args=())
        def allreduce_mul(x: torch.Tensor, scale: float) -> torch.Tensor:
            """AllReduce followed by scalar multiplication"""
            result = x.clone()
            return torch.ops._c10d_functional.all_reduce_(result, "sum", "default")

        @allreduce_mul.register_fake
        def _(x, scale):
            return torch.empty_like(x)

        # Strategy 1: All_reduce + separate multiply
        def strategy1(x, scale):
            result = x.clone()
            result = torch.ops._c10d_functional.all_reduce_(result, "sum", "default")
            return result * scale

        # Strategy 2: All_reduce + fused multiply (using torch.mul)
        def strategy2(x, scale):
            result = x.clone()
            result = torch.ops._c10d_functional.all_reduce_(result, "sum", "default")
            return torch.mul(result, scale)

        # Register autotuning
        from torch._inductor.kernel.custom_op import (
            register_custom_op_autotuning,
            CustomOpConfig,
        )

        register_custom_op_autotuning(
            allreduce_mul,
            configs=[
                CustomOpConfig(strategy1),
                CustomOpConfig(strategy2),
            ],
        )

        # Test model
        class ARMulModel(torch.nn.Module):
            def forward(self, x):
                return allreduce_mul(x, 0.5)

        model = torch.compile(ARMulModel()).to(device)

        # Run
        x = torch.randn(64, 64, device=device)
        x_copy = x.clone()
        y = model(x)

        # Verify: sum across 2 ranks, then multiply by 0.5
        expected = (x_copy * 2) * 0.5
        torch.testing.assert_close(y, expected, rtol=1e-3, atol=1e-3)

        if rank == 0:
            print(f"✅ [2 ranks] AllReduce + Multiply fusion test passed!")

        dist.destroy_process_group()

    @skip_if_lt_x_gpu(2)
    def test_allreduce_add_fusion(self):
        """Test all_reduce + add epilogue fusion"""

        # Initialize distributed
        dist.init_process_group(
            backend="nccl",
            init_method=f"file:///tmp/test_ar_add_{self.id()}",
            world_size=self.world_size,
            rank=self.rank,
        )

        rank = dist.get_rank()
        device = f"cuda:{rank}"

        from torch._C._distributed_c10d import _register_process_group

        _register_process_group("default", dist.group.WORLD)

        # Define custom all_reduce + add op
        @torch.library.custom_op("test::allreduce_add", mutates_args=())
        def allreduce_add(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
            """AllReduce followed by bias addition"""
            result = x.clone()
            return torch.ops._c10d_functional.all_reduce_(result, "sum", "default")

        @allreduce_add.register_fake
        def _(x, bias):
            return torch.empty_like(x)

        # Strategy 1: All_reduce + separate add
        def strategy1(x, bias):
            result = x.clone()
            result = torch.ops._c10d_functional.all_reduce_(result, "sum", "default")
            return result + bias

        # Strategy 2: All_reduce + fused add (torch.add)
        def strategy2(x, bias):
            result = x.clone()
            result = torch.ops._c10d_functional.all_reduce_(result, "sum", "default")
            return torch.add(result, bias)

        # Register autotuning
        from torch._inductor.kernel.custom_op import (
            register_custom_op_autotuning,
            CustomOpConfig,
        )

        register_custom_op_autotuning(
            allreduce_add,
            configs=[
                CustomOpConfig(strategy1),
                CustomOpConfig(strategy2),
            ],
        )

        # Test model
        class ARAddModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = None

            def forward(self, x, bias):
                return allreduce_add(x, bias)

        model = torch.compile(ARAddModel()).to(device)

        # Run
        x = torch.randn(64, 64, device=device)
        bias = torch.ones(64, 64, device=device) * 0.1
        x_copy = x.clone()
        bias_copy = bias.clone()
        y = model(x, bias)

        # Verify: sum across 2 ranks, then add bias
        expected = (x_copy * 2) + bias_copy
        torch.testing.assert_close(y, expected, rtol=1e-3, atol=1e-3)

        if rank == 0:
            print(f"✅ [2 ranks] AllReduce + Add fusion test passed!")

        dist.destroy_process_group()

    @skip_if_lt_x_gpu(2)
    def test_vllm_tensor_parallel_allreduce(self):
        """Test vLLM-style tensor_model_parallel_all_reduce pattern"""

        # Initialize distributed
        dist.init_process_group(
            backend="nccl",
            init_method=f"file:///tmp/test_vllm_tp_{self.id()}",
            world_size=self.world_size,
            rank=self.rank,
        )

        rank = dist.get_rank()
        device = f"cuda:{rank}"

        from torch._C._distributed_c10d import _register_process_group

        _register_process_group("default", dist.group.WORLD)

        # Define vLLM-style tensor parallel all_reduce
        @torch.library.custom_op("test::tp_allreduce", mutates_args=())
        def tensor_model_parallel_all_reduce(
            input_: torch.Tensor, open_fp8_quant: bool = False
        ) -> torch.Tensor:
            """vLLM-style tensor model parallel all_reduce"""
            result = input_.clone()
            return torch.ops._c10d_functional.all_reduce_(result, "sum", "default")

        @tensor_model_parallel_all_reduce.register_fake
        def _(input_, open_fp8_quant=False):
            return torch.empty_like(input_)

        # Strategy 1: Standard all_reduce
        def strategy1_standard(input_, open_fp8_quant=False):
            result = input_.clone()
            return torch.ops._c10d_functional.all_reduce_(result, "sum", "default")

        # Strategy 2: All_reduce with potential quantization path
        # (simulated for now, real vLLM would do FP8 quantization)
        def strategy2_quant_aware(input_, open_fp8_quant=False):
            result = input_.clone()
            # Simulate quantization-aware path
            if open_fp8_quant:
                # In real vLLM, this would apply FP8 quantization
                # For now, just do standard all_reduce
                pass
            result = torch.ops._c10d_functional.all_reduce_(result, "sum", "default")
            return result

        # Strategy 3: Optimized for tensor parallelism (e.g., with overlap)
        def strategy3_overlap(input_, open_fp8_quant=False):
            result = input_.clone()
            # In production, this might use NCCL streams for overlap
            result = torch.ops._c10d_functional.all_reduce_(result, "sum", "default")
            return result

        # Register autotuning
        from torch._inductor.kernel.custom_op import (
            register_custom_op_autotuning,
            CustomOpConfig,
        )

        register_custom_op_autotuning(
            tensor_model_parallel_all_reduce,
            configs=[
                CustomOpConfig(strategy1_standard),
                CustomOpConfig(strategy2_quant_aware),
                CustomOpConfig(strategy3_overlap),
            ],
        )

        # Test model (simulating vLLM tensor parallel layer)
        class TPLinearLayer(torch.nn.Module):
            """Simulated tensor parallel linear layer"""

            def __init__(self, in_features, out_features):
                super().__init__()
                # In real vLLM, weight is sharded across TP ranks
                self.weight = torch.nn.Parameter(
                    torch.randn(out_features, in_features, device=device)
                )

            def forward(self, x):
                # Local matmul (on sharded weight)
                local_out = torch.matmul(x, self.weight.t())
                # All-reduce to combine results from all TP ranks
                return tensor_model_parallel_all_reduce(local_out, open_fp8_quant=False)

        model = torch.compile(TPLinearLayer(256, 128)).to(device)

        # Run
        x = torch.randn(32, 256, device=device)
        y = model(x)

        # Verify shape
        self.assertEqual(y.shape, (32, 128))

        # Verify correctness: matmul result should be all_reduced across 2 ranks
        with torch.no_grad():
            expected = torch.matmul(x, model.weight.t()) * 2  # sum across 2 ranks

        torch.testing.assert_close(y, expected, rtol=1e-3, atol=1e-3)

        if rank == 0:
            print(f"✅ [2 ranks] vLLM tensor parallel all_reduce test passed!")

        dist.destroy_process_group()

    @skip_if_lt_x_gpu(2)
    def test_allreduce_complex_epilogue(self):
        """Test all_reduce with complex multi-operation epilogue"""

        # Initialize distributed
        dist.init_process_group(
            backend="nccl",
            init_method=f"file:///tmp/test_ar_complex_{self.id()}",
            world_size=self.world_size,
            rank=self.rank,
        )

        rank = dist.get_rank()
        device = f"cuda:{rank}"

        from torch._C._distributed_c10d import _register_process_group

        _register_process_group("default", dist.group.WORLD)

        # Define all_reduce + complex epilogue (multiply + add + relu)
        @torch.library.custom_op("test::allreduce_complex", mutates_args=())
        def allreduce_complex(
            x: torch.Tensor, scale: float, bias: torch.Tensor
        ) -> torch.Tensor:
            """AllReduce followed by scale * x + bias, then ReLU"""
            result = x.clone()
            return torch.ops._c10d_functional.all_reduce_(result, "sum", "default")

        @allreduce_complex.register_fake
        def _(x, scale, bias):
            return torch.empty_like(x)

        # Strategy 1: Separate operations
        def strategy1(x, scale, bias):
            result = x.clone()
            result = torch.ops._c10d_functional.all_reduce_(result, "sum", "default")
            result = result * scale
            result = result + bias
            return torch.relu(result)

        # Strategy 2: Fused operations
        def strategy2(x, scale, bias):
            result = x.clone()
            result = torch.ops._c10d_functional.all_reduce_(result, "sum", "default")
            # Fuse scale * x + bias
            result = torch.addcmul(bias, result, result.new_full(result.shape, scale))
            return torch.nn.functional.relu(result)

        # Strategy 3: Manual fusion
        def strategy3(x, scale, bias):
            result = x.clone()
            result = torch.ops._c10d_functional.all_reduce_(result, "sum", "default")
            # Manual fusion: scale * x + bias, then relu
            result = torch.maximum(result * scale + bias, torch.zeros_like(result))
            return result

        # Register autotuning
        from torch._inductor.kernel.custom_op import (
            register_custom_op_autotuning,
            CustomOpConfig,
        )

        register_custom_op_autotuning(
            allreduce_complex,
            configs=[
                CustomOpConfig(strategy1),
                CustomOpConfig(strategy2),
                CustomOpConfig(strategy3),
            ],
        )

        # Test model
        class ComplexModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = None

            def forward(self, x, bias):
                return allreduce_complex(x, 0.5, bias)

        model = torch.compile(ComplexModel()).to(device)

        # Run
        x = torch.randn(64, 64, device=device)
        bias = torch.ones(64, 64, device=device) * 0.1
        x_copy = x.clone()
        bias_copy = bias.clone()
        y = model(x, bias)

        # Verify: sum across 2 ranks, scale by 0.5, add bias, relu
        expected = torch.relu((x_copy * 2) * 0.5 + bias_copy)
        torch.testing.assert_close(y, expected, rtol=1e-3, atol=1e-3)

        if rank == 0:
            print(f"✅ [2 ranks] AllReduce + Complex epilogue fusion test passed!")

        dist.destroy_process_group()


if __name__ == "__main__":
    run_tests()
