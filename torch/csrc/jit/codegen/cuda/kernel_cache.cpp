#include <torch/csrc/jit/codegen/cuda/kernel_cache.h>

/*
 */

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

at::optional<CudaKernel*> CudaKernelCache::getKernelPtr(
    c10::IntArrayRef sizes) {
  for (auto& iter : kernels_) {
    if (iter.first.matchKernelSize(sizes)) {
      return &(iter.second);
    }
  }
  return at::nullopt;
}

CudaKernel* CudaKernelCache::allocateKernelInCache(KernelArgsReq args_req) {
  kernels_.emplace_back(std::make_pair(std::move(args_req), CudaKernel()));
  return &(kernels_.back().second);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
