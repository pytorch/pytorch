import torch
import gc
import ctypes
import ctypes.util

MADV_PAGEOUT = 21
libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
libc.madvise.restype = ctypes.c_int

BLOCK = 32 * 1024 * 1024  # 32 MB (> 16 MB mimalloc huge block threshold)

def test_mimalloc_zero_corruption_fix():
    """
    Regression test for PyTorch issue #200905: Silent data corruption (zeroed by kernel reclaim) on aarch64.
    
    The issue occurs when using mimalloc v2.x on ARM64 Linux, where memory allocated via torch.empty,
    freed, and reallocated can be silently zeroed by the kernel's madvise(MADV_PAGEOUT) call when the
    memory has been tagged with MADV_FREE by mimalloc's cross-thread huge block free path.
    
    The fix enables MI_HUGE_PAGE_ABANDON mode in mimalloc, which avoids the problematic
    _mi_segment_huge_page_reset function that was causing the corruption by decommitting/resetting
    memory that had been freed with MADV_FREE tags.
    
    This test verifies that after the fix, memory reuse does NOT result in massive corruption
    (from few zeros to millions of zeros) when madvise(MADV_PAGEOUT) is called.
    """
    # Allocate initial tensor and fill with specific value
    warmup = torch.empty(BLOCK, dtype=torch.uint8)
    warmup.fill_(0xCD)  # Fill with a specific pattern
    warmup_ptr = warmup.untyped_storage().data_ptr()
    
    # Delete and force garbage collection
    del warmup
    gc.collect()
    
    # Allocate new tensor (should reuse the same memory)
    tensor = torch.empty(BLOCK, dtype=torch.uint8)
    ptr = tensor.untyped_storage().data_ptr()
    size = tensor.untyped_storage().nbytes()
    reused = (ptr == warmup_ptr)
    assert reused, "Expected memory reuse for large allocation"
    
    # Count zeros before madvise
    zeros_before = (tensor == 0).sum().item()
    
    # Call madvise(MADV_PAGEOUT) - this used to corrupt the tensor
    ret = libc.madvise(ctypes.c_void_p(ptr), ctypes.c_size_t(size), ctypes.c_int(MADV_PAGEOUT))
    assert ret == 0, f"madvise failed with return code {ret}"
    
    # Count zeros after madvise - should be the same as before
    zeros_after = (tensor == 0).sum().item()
    
    # The key fix: assert that there's NO massive corruption
    # Before the fix: zeros_after >> zeros_before (corruption from few to millions of zeros)
    # After the fix: zeros_after should be approximately equal to zeros_before
    max_corruption_allowed = 1000  # Allow small variations due to memory allocator behavior
    corruption = abs(zeros_after - zeros_before)
    assert corruption <= max_corruption_allowed, \
        f"Memory corruption detected: zeros_before={zeros_before}, zeros_after={zeros_after}, " \
        f"corruption={corruption} exceeds allowed maximum {max_corruption_allowed}. " \
        f"This indicates the MI_HUGE_PAGE_ABANDON fix is not working properly."
    
    # Also verify that the tensor is not completely zeroed out (would be corruption)
    # The tensor should have mostly our original pattern, with only expected zeros
    zero_ratio = zeros_after / BLOCK
    assert zero_ratio < 0.01, \
        f"Tensor is mostly zeroed out (zero_ratio={zero_ratio}), " \
        f"indicating severe memory corruption despite the fix."
    
    # Sample some non-zero values to confirm data integrity
    # Pick some random indices that should contain our pattern (not the first few which may be zeroed by PyTorch)
    import random
    non_zero_samples = []
    for _ in range(10):
        idx = random.randint(BLOCK//4, BLOCK - BLOCK//4)  # Sample from middle to end
        non_zero_samples.append(tensor[idx].item())
    
    # Most samples should contain our original pattern (0xCD), not zero
    non_zero_count = sum(1 for val in non_zero_samples if val != 0)
    assert non_zero_count >= 8, \
        f"Too many zero values found in tensor ({non_zero_count}/10 non-zero), " \
        f"indicating data corruption despite the fix."
    
    print(f"Test passed: Memory reuse={reused}, zeros_before={zeros_before}, zeros_after={zeros_after}, "
          f"corruption={corruption}, zero_ratio={zero_ratio:.6f}")

if __name__ == "__main__":
    test_mimalloc_zero_corruption_fix()