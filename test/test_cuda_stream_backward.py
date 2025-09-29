#!/usr/bin/env python3
"""
Test case for issue #164094: backward pass should allow CUDA stream switching
in custom autograd.Function.
"""

import torch
import torch.cuda
import unittest

class TestCudaStreamBackward(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
    
    def test_backward_stream_switching(self):
        """Test that backward() allows stream switching in custom autograd.Function."""
        
        class StreamTestFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input_tensor, target_stream):
                # Save the target stream for backward
                ctx.target_stream = target_stream
                return input_tensor.clone()
            
            @staticmethod
            def backward(ctx, grad_output):
                # Try to switch to the target stream
                original_stream = torch.cuda.current_stream()
                
                # This should now work with the fix
                torch.cuda.set_stream(ctx.target_stream)
                current_stream = torch.cuda.current_stream()
                
                # Verify that the stream actually switched
                self.assertEqual(current_stream, ctx.target_stream,
                               f"Expected stream {ctx.target_stream}, got {current_stream}")
                
                # Restore original stream
                torch.cuda.set_stream(original_stream)
                return grad_output.clone(), None
        
        # Create two different CUDA streams
        stream1 = torch.cuda.Stream()
        stream2 = torch.cuda.Stream()
        
        # Create input tensor
        x = torch.randn(2, 3, requires_grad=True, device='cuda')
        
        # Run forward pass on stream1
        with torch.cuda.stream(stream1):
            result = StreamTestFunction.apply(x, stream2)
        
        # Run backward pass - this should now switch to stream2
        result.sum().backward()
        
        # Verify that x.grad was computed
        self.assertTrue(x.grad is not None)
        self.assertEqual(x.grad.shape, x.shape)

if __name__ == "__main__":
    unittest.main()