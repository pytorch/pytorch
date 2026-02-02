def test_philox_strided_layout_alignment(self):
        """
        Verify RNG alignment for non-contiguous views. 
        Ensures philox_rand respects strides to avoid spatial misalignment.
        """
        from torch._inductor import config
        
        def fn(x):
            # Slicing creates a view with non-standard physical strides
            y = x[::2, ::2]
            return torch.nn.functional.dropout(y, p=0.5, training=True)

        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
            
        x = torch.randn(20, 20, device="cuda")

        with config.patch(fallback_random=True):
            # Eager ground truth
            torch.manual_seed(42)
            expected = fn(x.clone())

            # Inductor compiled version
            torch.manual_seed(42)
            actual = torch.compile(fn)(x)

            self.assertEqual(actual, expected)