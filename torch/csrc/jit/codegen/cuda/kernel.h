#pragma once

#include <ATen/core/ivalue.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/fusion.h>

#include <ATen/core/ivalue.h>

/*
 * The exposed APIs in this file is used by manager.h/cpp
 *
 * code here handles CUDA code generation and execution from Fusion IR.
 * NVRTC is used for kernel compilation. CUDA Driver API is used to load and
 * execute compiled kernel.
 *
 * A stringify trick is used to unify the IO data structure for kernel
 * execution. We stringify the data structure and assert it direclty in the
 * generated CUDA source to avoid runtime search of header files.
 * The header file is included twice: one time as a c++ code to allow host code
 * to prepare IO data; the other time for stringify.
 */

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

// Not checking explicit broadcasting yet.
// check only shape falls in the range;
struct KernelArgsReq {
  // We are checking accumulated output shape for now, this is a restricting
  // aproach, we should check applicability on input tensor shapes instead.
  bool matchKernelSize(const c10::IntArrayRef inputs);
  std::vector<size_t> low_;
  std::vector<size_t> hi_;
};

class CudaKernel {
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
  void block(unsigned int x = 1, unsigned int y = 1, unsigned int z = 1) {
    block_ = dim3(x, y, z);
  }
  void grid(unsigned int x = 1, unsigned int y = 1, unsigned int z = 1) {
    grid_ = dim3(x, y, z);
  }

  dim3 block_;
  dim3 grid_;
};

// compile Fusion to CUDA functions:
// 1. JIT compilation via nvrtc to generate CUDA c++ kernel code;
// 2. CUDA Drive API to load CUDA c++ kernel code as function_;
TORCH_CUDA_API void compileKernel(Fusion& fusion, CudaKernel* entry);

// run loaded kernel through Function.
// inputs/outputs is given in the sense of a PyTorch JIT ir node. This function
// wraps IO data structure for tensors on host.
TORCH_CUDA_API void runKernel(
    CudaKernel* entry,
    const at::ArrayRef<c10::IValue>& inputs,
    std::vector<at::Tensor>& outputs);

// Facility API to run kernel in tests.
TORCH_CUDA_API void runTestKernel(
    CudaKernel& entry,
    const std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
