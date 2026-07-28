import torch
import math
import unittest
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.testing._internal.inductor_utils import HAS_CPU, HAS_CUDA
import torch._refs.special

class TestLogNdtr(TestCase):
    def test_log_ndtr_signbit(self):
        """
        Regression test for Issue #187336.
        Ensures that torch.special.log_ndtr preserves the -0.0 signbit 
        for large inputs even when compiled with Inductor fast-math.
        """
        def func(x):
            return torch.special.log_ndtr(x)
        
        devices = ["cpu"]
        if HAS_CUDA:
            devices.append("cuda")
            
        dtypes = [torch.float32, torch.float64]
        
        compiled_at_least_once = False
        
        for device in devices:
            for dtype in dtypes:
                with self.subTest(device=device, dtype=dtype):
                    # Test large input that flushes erfc to 0.0
                    x = torch.tensor([100.0], dtype=dtype, device=device)
                    
                    # 1. Check Eager Mode (Baseline ATen kernel)
                    expected = func(x)
                    self.assertEqual(math.copysign(1.0, expected.item()), -1.0)
                    
                    # 2. Check Python Decomposition Directly (AOTAutograd path)
                    # This guarantees we test our patched torch._refs decomposition!
                    expected_decomp = torch._refs.special.log_ndtr(x)
                    self.assertEqual(math.copysign(1.0, expected_decomp.item()), -1.0)
                    
                    # 3. Check Inductor Compiled Mode (if environment supports it)
                    if (device == "cpu" and HAS_CPU) or (device == "cuda" and HAS_CUDA):
                        try:
                            cf = torch.compile(func, backend="inductor")
                            actual = cf(x)
                            compiled_at_least_once = True
                        except Exception as e:
                            # Safely skip only if compilation/execution fails, NOT on assertions
                            self.skipTest(f"Inductor compilation failed for {device}/{dtype} in this environment: {e}")
                        
                        # Assertions are strictly outside the try-except block
                        self.assertEqual(math.copysign(1.0, actual.item()), -1.0)
        
        # Ensure that the test wasn't silently skipped for all hardware combinations
        self.assertTrue(
            compiled_at_least_once, 
            "Inductor never actually compiled in any device/dtype combo — this test proved nothing!"
        )

    def test_log_ndtr_edge_cases(self):
        """
        Ensures the -abs() fix doesn't break NaNs, infinities, or transition boundaries.
        Explicitly tests both Eager ATen and the _refs decomposition.
        """
        # Test values: NaN, -inf, 1.0 (branch transition), 0.0, -1.0
        x_edge = torch.tensor([float('nan'), float('-inf'), 1.0, 0.0, -1.0], dtype=torch.float64)
        
        # Test 1: Native Eager ATen
        out_eager = torch.special.log_ndtr(x_edge)
        # Test 2: Patched Python Decomposition
        out_decomp = torch._refs.special.log_ndtr(x_edge)
        
        for out in [out_eager, out_decomp]:
            # 1. NaN remains NaN
            self.assertTrue(torch.isnan(out[0]))
            
            # 2. -inf should be -inf (verifies Issue #122426 erfcx(-inf) does not NaN here)
            self.assertTrue(math.isinf(out[1].item()) and out[1].item() < 0)
            
            # 3. Branch boundary check (x=1.0)
            self.assertTrue(out[2].item() < 0)
            
            # 4. Standard values check (x=0 -> log(0.5) ≈ -0.693; x=-1 -> log(0.158) ≈ -1.84)
            self.assertTrue(math.isclose(out[3].item(), math.log(0.5), rel_tol=1e-5))
            self.assertTrue(math.isclose(out[4].item(), math.log(0.1586552539), rel_tol=1e-5))

if __name__ == '__main__':
    run_tests()
