#pragma once

#include <torch/csrc/jit/backends/backend_debug_handler.h>
#include <torch/custom_class.h>

namespace torch {
namespace jit {

/*
 * This custom class exists for only 1 reason.
 * 1. Register an instance of this class in LoweredModule as
 * __backend_debug_info
 * 2. During serialization extract __backend_debug_info class to obtain debug
 * map
 *
 * Notes:
 * - This class should only be available at build time for preprocess/compile
 * - It should not be necessary at run time.
 *   This is the case for lite interpreter at the moment. Not so for JIT.
 */
class TORCH_API PyTorchBackendDebugInfo : public torch::CustomClassHolder {
 public:
  PyTorchBackendDebugInfo() = default;

  c10::optional<BackendDebugInfoMapType>& getDebugInfoMap() {
    return debug_info_map_;
  }

  void setDebugInfoMap(BackendDebugInfoMapType&& debug_info_map) {
    debug_info_map_ = std::move(debug_info_map);
  }

 private:
  c10::optional<BackendDebugInfoMapType> debug_info_map_;
};
} // namespace jit
} // namespace torch
