#pragma once
#include <torch/csrc/jit/codegen/cuda/fusion.h>

#include <ATen/cuda/CUDAContext.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

#define STRINGIFY(...) __VA_ARGS__
#include <torch/csrc/jit/codegen/cuda/data_struct_str.h>
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

  // WARNING:
  // Block and Grid dimension setting is here for testing purposes only
  // These are not here for general use and only for use with
  // the runTestKernel() function.
  void block(uint x=1, uint y=1, uint z=1) {
	block_ = dim3(x,y,z);
  }
  void grid(uint x=1, uint y=1, uint z=1) {
	grid_ = dim3(x,y,z);
  }

  dim3 block_;
  dim3 grid_;
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

TORCH_API void runTestKernel(
    CudaKernel& entry,
    const std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs);

}}}} // namespace torch::jit::fuser::cuda
