#pragma once
#include <torch/csrc/jit/fuser/common/fusion.h>
#include <ATen/cuda/CUDAContext.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class CudaKernel{
public:
  CudaKernel() = default;

protected:
  int16_t device_;
  CUmodule module_;
  CUfunction function_;
};

TORCH_API void compileKernel(Fusion& fusion, CudaKernel& entry);

}}}} // namespace torch::jit::fuser::cuda
