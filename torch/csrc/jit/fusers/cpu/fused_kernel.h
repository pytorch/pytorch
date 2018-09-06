#include "torch/csrc/jit/fusers/Config.h"
#if USE_CPU_FUSER
#pragma once

#include "torch/csrc/jit/fusers/cpu/fusion_compiler.h"
#include "torch/csrc/jit/fusers/cpu/dynamic_library.h"
#include "torch/csrc/jit/fusers/common/fused_kernel.h"
#include "torch/csrc/jit/fusers/common/annotated_graph.h"

#include "ATen/ATen.h"

#include <string>
#include <cstdint>
#include <memory>

namespace torch { namespace jit { namespace cpufuser {

struct CPUFusedKernel : public ::torch::jit::FusedKernel {
  CPUFusedKernel(
    const std::string& name
  , AnnotatedGraph& agraph
  , CPUFusionCompilerConfig& config);

protected:
  virtual at::Backend backend() const override {
    return at::Backend::CPU;
  }

  virtual uint64_t get_rand_offset(uint32_t numel) override {
    return numel;
  }

  virtual void launch_raw(uint32_t numel, void** arguments) override {
    kernel(numel, arguments);
  }

  std::unique_ptr<DynamicLibrary> so_lib;
  void (*kernel)(uint32_t, void**) = nullptr;
};

} // namespace cpufuser
} // namespace jit 
} // namespace torch

#endif // USE_CPU_FUSER
