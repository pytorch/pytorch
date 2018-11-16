#pragma once
#include "torch/csrc/jit/fuser/config.h"
#if USE_CPU_FUSER

#include "ATen/ATen.h"
#include "torch/csrc/WindowsTorchApiMacro.h"
#include "torch/csrc/utils/disallow_copy.h"
#include "torch/csrc/jit/fuser/cpu/dynamic_library.h"
#include "torch/csrc/jit/fuser/fused_kernel.h"

#include <string>
#include <cstdint>
#include <memory>

namespace torch { namespace jit { namespace fuser { namespace cpu {

// Represents a compiled CPU kernel and the metadata necessary to run it
struct TORCH_API FusedKernelCPU : public ::torch::jit::fuser::FusedKernel {
  FusedKernelCPU(
      std::string name,
      std::string code,
      std::vector<TensorDesc> input_desc,
      std::vector<TensorDesc> output_desc,
      std::vector<PartitionDesc> chunk_desc,
      std::vector<PartitionDesc> concat_desc,
      bool has_random);

  at::Backend backend() const override {
    return at::Backend::CPU;
  }

  void launch_raw(const uint32_t numel, std::vector<void*>& arguments)
      const override {
    kernel(numel, arguments.data());
  }

private:
  std::unique_ptr<DynamicLibrary> so_lib;
  void (*kernel)(uint32_t, void**) = nullptr;
};

} // namespace cpu
} // namespace fuser
} // namespace jit
} // namespace torch

#endif // USE_CPU_FUSER
