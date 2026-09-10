#include <torch/csrc/cuda/CUDAPluggableAllocator.h>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>

namespace {
using namespace c10d::symmetric_memory;

// Alloc functor for MemPool
void* cuda_symm_alloc(size_t size, int device, void* stream) {
  static auto allocator = get_allocator(c10::DeviceType::CUDA);
  // Note: the group info is now specified at the time of rendezvous instead of
  // allocation. We thus pass `nullopt` for group here.
  return allocator->alloc(size, device, /*group_name=*/std::nullopt);
}

// Free functor for MemPool
void cuda_symm_free(void* ptr, size_t size, int device, void* stream) {
  static auto allocator = get_allocator(c10::DeviceType::CUDA);
  allocator->free(ptr);
}

// Register allocator for CUDA MemPool
struct RegisterCUDAMemPoolAllocator {
  RegisterCUDAMemPoolAllocator() {
    std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator> allocator =
        torch::cuda::CUDAPluggableAllocator::createCustomAllocator(
            cuda_symm_alloc, cuda_symm_free);
    register_mempool_allocator(c10::DeviceType::CUDA, allocator);
  }
};

static RegisterCUDAMemPoolAllocator register_cuda_mempool_allocator_;

} // namespace
