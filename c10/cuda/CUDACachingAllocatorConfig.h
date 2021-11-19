#include <c10/cuda/CUDAMacros.h>

namespace c10 {
namespace cuda {
namespace CUDACachingAllocator {

// Currently either "native" or "cudaMallocAsync"
C10_CUDA_API const std::string& allocatorBackend();

C10_CUDA_API size_t maxSplitSize();

} // namespace CUDACachingAllocator
} // namespace cuda
} // namespace c10
