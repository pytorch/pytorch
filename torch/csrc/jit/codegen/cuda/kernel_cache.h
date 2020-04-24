#pragma once

#include <c10/util/ArrayRef.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/kernel.h>

/*
 */

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class CudaKernelCache {
 public:
  CudaKernelCache() = default;

  at::optional<CudaKernel*> getKernelPtr(c10::IntArrayRef sizes);
  CudaKernel* allocateKernelInCache(KernelArgsReq args_req);

  // private:
  // TODO: In theory we should assume contiguity remain constant across runs
  //       (job for BailOut node from profiling executor). In reality we might
  //       want to be safe and cache on that as well.
  // Assuming constant nDims. Cache of kernels targetting different tensor size;
  // We should flatten
  std::vector<std::pair<KernelArgsReq, CudaKernel>> kernels_;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
