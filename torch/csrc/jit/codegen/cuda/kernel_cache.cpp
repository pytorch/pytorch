#include <torch/csrc/jit/codegen/cuda/kernel_cache.h>

/*
 */

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

at::optional<CudaKernel*> CudaKernelCache::getKernelPtr(
    const at::ArrayRef<IValue> inputs) {
  for (auto& iter : kernels_) {
    if (iter.first->matchKernelSize(inputs)) {
      return &(iter.second);
    }
  }
  return at::nullopt;
}

CudaKernel* CudaKernelCache::allocateKernelInCache(
    std::unique_ptr<KernelArgsReq>&& args_req) {
  kernels_.emplace_back(std::make_pair(std::move(args_req), CudaKernel()));
  return &(kernels_.back().second);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
