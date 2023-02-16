#pragma once

#include <c10/macros/Export.h>
#include <kernel.h>

#include <string>

namespace nvfuser {
namespace codegen {

//! Generates a CUDA kernel definition for the given kernel
TORCH_CUDA_CU_API std::string generateCudaKernel(
    const kir::Kernel* kernel,
    const std::string& kernel_name = "CUDAGeneratedKernel");

} // namespace codegen
} // namespace nvfuser
