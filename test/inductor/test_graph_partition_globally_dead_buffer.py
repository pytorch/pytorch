# Owner(s): ["module: inductor"]

import unittest

import torch
from torch.testing._internal.common_cuda import SM70OrLater
from torch.testing._internal.common_device_type import skipCUDAIf
from torch.testing._internal.common_utils import run_tests, skipIfRocm, TestCase
from torch.testing._internal.inductor_utils import HAS_CUDA


class TestGraphPartitionGloballyDeadBuffer(TestCase):
    """
    Test for issue #169232: When graph partitioning is enabled with forbidden cudagraph ops,
    buffers marked as globally dead during whole-graph dead code elimination may still be
    needed as inputs to specific partitions.
    
    Without the fix, this causes NameError when triton kernels try to reference them.
    """

    @unittest.skipIf(not HAS_CUDA, "CUDA not available")
    @skipCUDAIf(not SM70OrLater, "GPU capability is < SM70")
    @skipIfRocm
    def test_globally_dead_buffer_in_partition_with_forbidden_op(self):
        """
        Test that globally-dead buffers are kept in partition signatures when needed.
        
        This test creates a scenario where:
        1. Graph partitioning is enabled due to forbidden cudagraph ops
        2. An intermediate buffer (buf13) is created in one partition
        3. The buffer is used in a subsequent partition but not in the final graph output
        4. The buffer is marked as "globally dead" by whole-graph DCE
        5. Without the fix, the buffer is incorrectly removed from the partition signature
        6. This causes NameError when the triton kernel tries to use it
        """
        
        def model(x, y):
            # Partition 1: Create intermediate buffers
            a = x + 1.0
            b = y * 2.0
            
            # This buffer will be globally dead (not in final output)
            # but needed as input to the next partition
            intermediate = a * b
            
            # Partition 2: Use the intermediate buffer
            # clamp is a forbidden cudagraph op, forcing a partition boundary
            c = torch.clamp(intermediate, min=0.0, max=1.0)
            
            # Partition 3: Continue computation
            d = c / (c + 1e-6)
            e = d.view(-1)
            
            # Final output doesn't directly include 'intermediate',
            # making it globally dead, but it's still needed by Partition 2
            return e.sum()
        
        # Enable config that triggers the bug
        with torch._inductor.config.patch({
            "triton.cudagraphs": True,  # Enable cudagraphs to trigger partitioning
            "triton.cudagraph_support_input_mutation": True,
        }):
            x = torch.randn(10, 10, device="cuda", requires_grad=False)
            y = torch.randn(10, 10, device="cuda", requires_grad=False)
            
            # Compile the model
            compiled_model = torch.compile(model, backend="inductor")
            
            # This should not raise NameError: name 'bufXX' is not defined
            # Without the fix, triton kernel in partition would reference
            # a buffer that was incorrectly removed from the partition signature
            try:
                result = compiled_model(x, y)
                expected = model(x, y)
                
                # Verify correctness
                self.assertTrue(torch.allclose(result, expected, rtol=1e-4, atol=1e-5))
                
            except NameError as e:
                # If we get NameError about a buffer, the fix is not working
                if "buf" in str(e) and "is not defined" in str(e):
                    self.fail(
                        f"Graph partition signature fix failed: {e}\n"
                        "A globally-dead buffer was incorrectly removed from partition signature."
                    )
                else:
                    # Re-raise if it's a different NameError
                    raise

    @unittest.skipIf(not HAS_CUDA, "CUDA not available")
    @skipCUDAIf(not SM70OrLater, "GPU capability is < SM70")
    @skipIfRocm
    def test_multiple_partitions_with_intermediate_buffers(self):
        """
        More complex test with multiple partitions and intermediate buffers.
        """
        
        def complex_model(x, y, z):
            # Partition 1
            a = x + y
            b = a * 2.0
            
            # Intermediate buffer (globally dead but needed by later partitions)
            temp1 = b.expand(20, -1, -1)
            
            # Partition 2: clamp forces partition boundary
            c = torch.clamp(temp1, -1.0, 1.0)
            
            # Another intermediate (globally dead)
            temp2 = c * z
            
            # Partition 3: another forbidden op
            d = torch.clamp(temp2, 0.0, 10.0)
            
            # Partition 4: final computation
            e = d.sum(dim=0)
            result = e.view(-1)
            
            return result
        
        with torch._inductor.config.patch({
            "triton.cudagraphs": True,
            "triton.cudagraph_support_input_mutation": True,
        }):
            x = torch.randn(10, 20, device="cuda")
            y = torch.randn(10, 20, device="cuda")
            z = torch.randn(20, 10, 20, device="cuda")
            
            compiled_model = torch.compile(complex_model, backend="inductor")
            
            try:
                result = compiled_model(x, y, z)
                expected = complex_model(x, y, z)
                
                self.assertTrue(torch.allclose(result, expected, rtol=1e-4, atol=1e-5))
                
            except NameError as e:
                if "buf" in str(e) and "is not defined" in str(e):
                    self.fail(
                        f"Graph partition signature fix failed with multiple partitions: {e}"
                    )
                else:
                    raise


if __name__ == "__main__":
    run_tests()
