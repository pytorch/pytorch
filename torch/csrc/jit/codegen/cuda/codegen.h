#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/codegen/cuda/kernel.h>

#include <string>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace codegen {

//! Generates a CUDA kernel definition for the given kernel
TORCH_CUDA_CU_API std::string generateCudaKernel(
    const Kernel* kernel,
    const std::string& kernel_name = "CUDAGeneratedKernel");

} // namespace codegen
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
