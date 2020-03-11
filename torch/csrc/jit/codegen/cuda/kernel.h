#pragma once
#include <torch/csrc/jit/fuser/common/fusion.h>

#include <ATen/cuda/CUDAContext.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

#define STRINGIFY(...) __VA_ARGS__
#include <torch/csrc/jit/fuser/cuda/data_struct_str.h>
#undef STRINGIFY

class CudaKernel{
public:
  CudaKernel() = default;

  CUmodule& getModule() {
    return module_;
  }

  CUfunction& getFunction() {
    return function_;
  }

  int16_t device_;
  CUmodule module_;
  CUfunction function_;
  int max_blocks_;
};

#define STRINGIFY(...) #__VA_ARGS__
static auto typeinfo =
#include"data_struct_str.h"
;
#undef STRINGIFY

TORCH_API void compileKernel(Fusion& fusion, CudaKernel& entry);

TORCH_API void runKernel(
    CudaKernel& entry,
    const at::ArrayRef<IValue>& inputs,
    std::vector<at::Tensor>& outputs);

}}}} // namespace torch::jit::fuser::cuda
