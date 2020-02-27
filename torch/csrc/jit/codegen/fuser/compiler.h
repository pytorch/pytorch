#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/fuser/arg_spec.h>
#include <torch/csrc/jit/fuser/fused_kernel.h>
#include <torch/csrc/jit/fuser/interface.h>
#include <torch/csrc/jit/fuser/kernel_spec.h>
#include <torch/csrc/jit/ir.h>
#include <ATen/core/stack.h>

#include <cstdint>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {

// Performs device-independent "upfront" compilation of the given fusion_group,
// if it has not been registered already.
// Returns a key that can be used to run the fusion later
TORCH_API int64_t registerFusion(const Node* fusion_group);

// Performs device-specific "runtime" compilation of the given kernel
//  with the runtime arguments specified in ArgSpec.
//  Outputs are allocated using map_size on the specified device.
TORCH_API std::shared_ptr<FusedKernel> compileKernel(
    const KernelSpec& spec,
    const ArgSpec& arg_spec,
    const std::vector<int64_t>& map_size,
    const at::Device device);

TORCH_API size_t nCompiledKernels();

TORCH_API int debugFuser();

using FusedKernelConstructor = std::function<std::shared_ptr<FusedKernel>(
    int16_t device,
    std::string name,
    std::string code,
    std::vector<TensorDesc> input_desc,
    std::vector<TensorDesc> output_desc,
    std::vector<PartitionDesc> chunk_desc,
    std::vector<PartitionDesc> concat_desc,
    bool has_random)>;

TORCH_API void registerFusionBackend(
    at::Device::Type backend_type,
    FusedKernelConstructor ctor);
TORCH_API bool hasFusionBackend(at::Device::Type backend_type);
struct TORCH_API RegisterFusionBackend {
  RegisterFusionBackend(
      at::Device::Type backend_type,
      FusedKernelConstructor ctor) {
    registerFusionBackend(backend_type, std::move(ctor));
  }
};

} // namespace fuser
} // namespace jit
} // namespace torch
