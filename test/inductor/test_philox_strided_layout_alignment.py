def test_philox_strided_layout_alignment(self):
        """
        Verify RNG alignment for non-contiguous views. 
        Ensures philox_rand respects strides to avoid spatial misalignment.
        """
        from torch._inductor import config
        
        def fn(x):
            # Slicing creates a view with non-standard strides
            y = x[::2, ::2]
            return torch.nn.functional.dropout(y, p=0.5, training=True)

        # 1. Ensure we are on CUDA (where Philox/Triton is used)
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
            
        x = torch.randn(20, 20, device="cuda")

        # 2. Use config.patch to ensure we hit the philox_rand lowering path
        with config.patch(fallback_random=True):
            # Eager mode (the ground truth)
            torch.manual_seed(42)
            expected = fn(x.clone())

            # Compiled mode (your fix)
            torch.manual_seed(42)
            actual = torch.compile(fn)(x)

            # 3. Use self.assertEqual for professional PyTorch error reporting
            self.assertEqual(actual, expected)