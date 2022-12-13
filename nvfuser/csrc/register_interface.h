#pragma once
#include <manager.h>
#include <transform_view.h>

#include <c10/macros/Export.h>
#include <torch/csrc/jit/codegen/cuda/interface.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/pass_manager.h>
#include <torch/csrc/jit/runtime/profiling_record.h>

/*
 * This file contains APIs for cuda fuser;
 *
 * We use an empty static struct to hold the function pointers, which are
 * registered separately. This is to support cpu-only compilation.
 * Registration is done in torch/csrc/jit/codegen/cuda/register_interface.cpp
 */

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

TORCH_CUDA_CU_API bool complyWith(
    const at::Tensor& tensor,
    const c10::TensorTypePtr& guard_tensor_type);

struct TORCH_CUDA_CU_API NVFuserPassManager
    : public PassManager<NVFuserPassManager> {
  static bool registerPass(bool enabled) {
    bool old_value = PassManager::isRegistered();
    if (enabled) {
      PassManager::registerPass(fuseGraph);
    } else {
      PassManager::clearPass();
    }
    return old_value;
  }

  static bool isRegistered() {
    return PassManager::isRegistered();
  }
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
