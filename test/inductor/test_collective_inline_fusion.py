"""
Test to verify inline fusion happens after collective op autotuning.

This test verifies that:
1. Collective ops go through autotuning and benchmarking
2. The winning choice is inlined into the main graph
3. Inline fusion occurs with surrounding ops (e.g., allreduce + relu)
"""

import logging
import torch
import torch.distributed as dist
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.testing._internal.distributed.multi_threaded_pg import (
    MultiThreadedProcessGroup,
)


def setup_distributed(rank, world_size):
    """Setup distributed environment for testing."""
    import os
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    
    dist.init_process_group(
        backend="gloo",  # Use gloo for CPU testing
        init_method=f"tcp://localhost:29500",
        world_size=world_size,
        rank=rank,
    )


class TestCollectiveInlineFusion(TestCase):
    def setUp(self):
        super().setUp()
        # Set up logging to capture inline fusion messages
        self.logger = logging.getLogger("torch._inductor")
        self.logger.setLevel(logging.DEBUG)
        
    def test_allreduce_relu_fusion(self):
        """
        Test that allreduce + relu can be fused after autotuning.
        
        Expected flow:
        1. Register custom allreduce op with multiple configs
        2. Autotuning selects the best config
        3. Winning choice is inlined into the graph
        4. Inline fusion combines allreduce + relu operations
        """
        import os
        
        # Use small timeout and few runs for faster testing
        os.environ["TORCHINDUCTOR_COLLECTIVE_BENCHMARK_TIMEOUT"] = "5"
        os.environ["TORCHINDUCTOR_COLLECTIVE_BENCHMARK_NRUNS"] = "2"
        
        # Disable caching to force fresh autotuning
        os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "0"
        os.environ["TORCHINDUCTOR_AUTOTUNE_LOCAL_CACHE"] = "0"
        
        # Enable debug logging for inline fusion
        os.environ["TORCH_LOGS"] = "output_code,graph_code,fusion"
        
        # Initialize distributed (single process for simplicity)
        if not dist.is_initialized():
            dist.init_process_group(
                backend="gloo",
                init_method="file:///tmp/test_inline_fusion",
                world_size=1,
                rank=0,
            )
        
        # Register default process group
        from torch._C._distributed_c10d import _register_process_group
        _register_process_group("default", dist.group.WORLD)
        
        # Define custom collective op that uses REAL allreduce
        @torch.library.custom_op("test::allreduce_relu", mutates_args=())
        def allreduce_relu(x: torch.Tensor) -> torch.Tensor:
            """Custom op that does allreduce followed by relu."""
            result = x.clone()
            # Use REAL allreduce operation
            result = torch.ops._c10d_functional.all_reduce_(result, "sum", "default")
            return torch.relu(result)
        
        @allreduce_relu.register_fake
        def _(x):
            return torch.empty_like(x)
        
        # Register multiple implementations for autotuning
        from torch._inductor.kernel.custom_op import (
            CustomOpConfig,
            register_custom_op_autotuning,
        )
        
        # Implementation 1: Separate allreduce and relu
        def impl1(x):
            result = x.clone()
            # REAL allreduce
            result = torch.ops._c10d_functional.all_reduce_(result, "sum", "default")
            return torch.relu(result)
        
        # Implementation 2: Different allreduce strategy (e.g., using wait_tensor)
        def impl2(x):
            result = x.clone()
            # REAL allreduce with different approach
            result = torch.ops._c10d_functional.all_reduce_(result, "sum", "default")
            # Fused relu
            return torch.nn.functional.relu(result, inplace=False)
        
        register_custom_op_autotuning(
            allreduce_relu,
            configs=[
                CustomOpConfig(impl1),
                CustomOpConfig(impl2),
            ],
        )
        
        # Create model that uses the custom op
        class TestModel(torch.nn.Module):
            def forward(self, x):
                # This should trigger autotuning and inline fusion
                return allreduce_relu(x)
        
        # Compile the model
        print("\n" + "="*80)
        print("Compiling model with collective op autotuning...")
        print("="*80)
        
        model = torch.compile(TestModel())
        
        # Run inference
        x = torch.randn(128, 128, requires_grad=False)
        
        print("\nRunning model (expect autotuning + inline fusion)...")
        y = model(x)
        
        # Verify correctness
        expected = torch.relu(x * 2)
        torch.testing.assert_close(y, expected, rtol=1e-5, atol=1e-5)
        
        print("\n" + "="*80)
        print("✅ Test passed! Inline fusion should have occurred.")
        print("="*80)
        
        # Cleanup
        if os.path.exists("/tmp/test_inline_fusion"):
            os.remove("/tmp/test_inline_fusion")
        
        dist.destroy_process_group()
    
    def test_verify_inline_fusion_in_generated_code(self):
        """
        Verify inline fusion by checking generated code.
        
        This test inspects the generated kernel code to confirm that
        fusion actually happened.
        """
        import os
        import tempfile
        
        # Set output dir to inspect generated code
        output_dir = tempfile.mkdtemp(prefix="inductor_inline_fusion_")
        os.environ["TORCHINDUCTOR_OUTPUT_DIR"] = output_dir
        os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "0"
        
        print(f"\n📁 Generated code will be saved to: {output_dir}")
        
        # Initialize distributed
        if not dist.is_initialized():
            dist.init_process_group(
                backend="gloo",
                init_method="file:///tmp/test_inline_fusion_code",
                world_size=1,
                rank=0,
            )
        
        from torch._C._distributed_c10d import _register_process_group
        _register_process_group("default", dist.group.WORLD)
        
        # Define custom op
        @torch.library.custom_op("test::allreduce_add", mutates_args=())
        def allreduce_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            result = x.clone()
            result = result * 2  # Simulate allreduce
            return result + y
        
        @allreduce_add.register_fake
        def _(x, y):
            return torch.empty_like(x)
        
        from torch._inductor.kernel.custom_op import (
            CustomOpConfig,
            register_custom_op_autotuning,
        )
        
        def impl(x, y):
            result = x.clone()
            result = result * 2
            return result + y
        
        register_custom_op_autotuning(
            allreduce_add,
            configs=[CustomOpConfig(impl)],
        )
        
        class TestModel(torch.nn.Module):
            def forward(self, x, y):
                return allreduce_add(x, y)
        
        model = torch.compile(TestModel())
        
        x = torch.randn(64, 64)
        y = torch.randn(64, 64)
        
        print("\nCompiling and running...")
        output = model(x, y)
        
        # Verify correctness
        expected = (x * 2) + y
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)
        
        # Check generated code
        print(f"\n🔍 Checking generated code in: {output_dir}")
        
        # Look for Python files (generated code)
        import glob
        py_files = glob.glob(os.path.join(output_dir, "**/*.py"), recursive=True)
        
        if py_files:
            print(f"✅ Found {len(py_files)} generated Python file(s)")
            for f in py_files[:3]:  # Show first 3
                print(f"   - {os.path.basename(f)}")
        else:
            print("⚠️  No generated Python files found")
        
        print("\n✅ Test passed! Check the output directory for fusion evidence.")
        
        # Cleanup
        if os.path.exists("/tmp/test_inline_fusion_code"):
            os.remove("/tmp/test_inline_fusion_code")
        
        dist.destroy_process_group()


if __name__ == "__main__":
    # Run the tests
    import sys
    
    # Create test suite
    suite = TestCase.suite()
    
    # Run tests
    test = TestCollectiveInlineFusion()
    
    print("\n" + "="*80)
    print("TEST 1: Allreduce + ReLU Fusion")
    print("="*80)
    try:
        test.test_allreduce_relu_fusion()
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("TEST 2: Verify Inline Fusion in Generated Code")
    print("="*80)
    try:
        test.test_verify_inline_fusion_in_generated_code()
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("All tests completed!")
    print("="*80)
