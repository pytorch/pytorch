#pragma once

#include <ATen/Context.h>
#include <torch/csrc/jit/codegen/cuda/interface.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/pass_manager.h>
#include <string>
#include <utility>

namespace torch {
namespace jit {

// Register CudaFuseGraph in custom passes
struct TORCH_API RegisterCudaFuseGraph
    : public PassManager<RegisterCudaFuseGraph> {
  static bool registerPass(bool enabled, bool check_gpu = true) {
    bool old_flag = PassManager::isRegistered();
    if (enabled) {
      // TODO: this might not be the right place to put the ROCm logic.
      // i.e. since this is built as part of libtorch cpu, USE_ROCM flag
      // might not be available while building this lib.
#ifdef USE_ROCM
      bool has_rocm = true;
#else
      bool has_rocm = false;
#endif
      if (check_gpu) {
        TORCH_CHECK(
            at::globalContext().hasCUDA() && !has_rocm,
            "Running CUDA fuser is only supported on CUDA builds.");
      }
      PassManager::registerPass(fuser::cuda::fuseGraph);
    } else {
      PassManager::clearPass();
    }
    return old_flag;
  }

  static bool isRegistered() {
    return PassManager::isRegistered();
  }
};

struct CudaFuserComparisonCallback {
  using callback_type =
      std::function<void(const Stack&, const Stack&, const std::string&)>;
  bool run_fallback;
  callback_type callback;
};

TORCH_API CudaFuserComparisonCallback getCudaFuserComparisonCallback();
TORCH_API void setCudaFuserComparisonCallback(CudaFuserComparisonCallback);

} // namespace jit
} // namespace torch
