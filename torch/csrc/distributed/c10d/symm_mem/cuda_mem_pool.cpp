#include <torch/csrc/cuda/CUDAPluggableAllocator.h>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>

namespace {
using namespace c10d::symmetric_memory;

// Alloc functor for MemPool
void* cuda_symm_alloc(size_t size, int device, void* stream) {
  auto allocator = get_allocator(c10::DeviceType::CUDA);
  // Note: this alloc functor works for the NVSHMEM and NCCL backends only,
  // because only these backends takes `nullopt` for the `group` argument which
  // is not given by MemPool's invocation (actually these two backends requires
  // it to be `nullopt`).
  return allocator->alloc(size, device, /*group_name=*/std::nullopt);
}

// Free functor for MemPool
void cuda_symm_free(void* ptr, size_t size, int device, void* stream) {
  auto allocator = get_allocator(c10::DeviceType::CUDA);
  allocator->free(ptr);
}

} // namespace

namespace c10d::symmetric_memory {

// Get allocator for MemPool
std::shared_ptr<c10::Allocator> get_mempool_allocator(c10::Device device) {
  TORCH_CHECK(
      device.type() == c10::DeviceType::CUDA,
      "SymmetricMemory MemPool supports CUDA device only");
  static std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator>
      mempool_allocator =
          torch::cuda::CUDAPluggableAllocator::createCustomAllocator(
              cuda_symm_alloc, cuda_symm_free);
  return mempool_allocator;
}

} // namespace c10d::symmetric_memory
