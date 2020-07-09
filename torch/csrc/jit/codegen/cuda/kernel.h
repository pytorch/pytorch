#pragma once

#include <ATen/core/ivalue.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/kernel_cache.h>

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

// compile Fusion to CUDA functions:
// 1. JIT compilation via nvrtc to generate CUDA c++ kernel code;
// 2. CUDA Drive API to load CUDA c++ kernel code as function_;
TORCH_CUDA_API void compileKernel(CudaKernel* entry);

// run loaded kernel through Function.
// inputs/outputs is given in the sense of a PyTorch JIT ir node. This function
// wraps IO data structure for tensors on host.
TORCH_CUDA_API void runKernel(
    CudaKernel* entry,
    const at::ArrayRef<c10::IValue> inputs,
    const std::vector<at::Tensor>& outputs,
    const c10::optional<at::IntArrayRef>& broadcasted_size = c10::nullopt);

// Facility API to run kernel in tests.
TORCH_CUDA_API void runTestKernel(
    CudaKernel* entry,
    const at::ArrayRef<c10::IValue> inputs,
    const std::vector<at::Tensor>& outputs);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
