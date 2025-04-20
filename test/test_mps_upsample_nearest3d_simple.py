import torch
import time

def test_upsample_nearest3d_vec():
    """Test upsample_nearest3d.vec implementation on MPS device"""
    # Skip if MPS is not available
    if not torch.backends.mps.is_available():
        print("MPS device not available")
        return
    
    # Create input tensor
    x = torch.randn(2, 3, 4, 5, 6, device="mps")
    
    # Check if we get a warning about CPU fallback
    with torch.no_grad():
        # Warm up
        for _ in range(3):
            y = torch.nn.functional.interpolate(x, scale_factor=(2.0, 2.0, 2.0), mode="nearest")
        
        # Measure performance
        torch.mps.synchronize()
        start_time = time.time()
        
        for _ in range(10):
            y = torch.nn.functional.interpolate(x, scale_factor=(2.0, 2.0, 2.0), mode="nearest")
            torch.mps.synchronize()
        
        end_time = time.time()
        mps_time = (end_time - start_time) / 10
        
        # Run on CPU for comparison
        x_cpu = x.to("cpu")
        start_time = time.time()
        
        for _ in range(10):
            y_cpu = torch.nn.functional.interpolate(x_cpu, scale_factor=(2.0, 2.0, 2.0), mode="nearest")
        
        end_time = time.time()
        cpu_time = (end_time - start_time) / 10
        
        # Print performance comparison
        print(f"MPS time: {mps_time:.6f} seconds")
        print(f"CPU time: {cpu_time:.6f} seconds")
        print(f"Speedup: {cpu_time / mps_time:.2f}x")
        
        # Check if results match
        y_mps = torch.nn.functional.interpolate(x, scale_factor=(2.0, 2.0, 2.0), mode="nearest")
        y_cpu = torch.nn.functional.interpolate(x_cpu, scale_factor=(2.0, 2.0, 2.0), mode="nearest")
        
        print(f"MPS output shape: {y_mps.shape}")
        print(f"CPU output shape: {y_cpu.shape}")
        print(f"Results match: {torch.allclose(y_mps.to('cpu'), y_cpu, rtol=1e-3, atol=1e-3)}")

if __name__ == "__main__":
    test_upsample_nearest3d_vec()
