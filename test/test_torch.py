    # FIXME: find a test suite for the pdist operator
    @unittest.skipIf(IS_FBCODE and IS_REMOTE_GPU, "sandcastle OOM with current tpx gpu/re configuration")
    @onlyCUDA
    @largeTensorTest('32GB', device='cpu')
    @largeTensorTest('5GB', device='cuda')
    def test_pdist_norm_large(self, device):
        # use dim0>=46342 for forward, see:
        # https://github.com/pytorch/pytorch/issues/30583
        # Compare output using GPU with the CPU implementation
        x = torch.randn(50000, 1, dtype=torch.float32)      # 50k * 4 bytes = 200 KB
        # Will require 1249975000 float32s
        expected_cpu = torch.pdist(x, p=2)                  # ~1250M * 4 bytes = 5 GB on CPU
        actual_gpu = torch.pdist(x.to(device), p=2).cpu()   # 5 GB on GPU + 5GB on CPU
        # Use appropriate tolerance for ROCm TF32 precision differences
        if TEST_WITH_ROCM:
            self.assertTrue(torch.allclose(expected_cpu, actual_gpu, rtol=1e-3, atol=1e-3))
        else:
            # Workaround for large memory overhead of self.assertTrue (see #84944)
            self.assertTrue(torch.allclose(expected_cpu, actual_gpu))  # ~20GB in allclose
