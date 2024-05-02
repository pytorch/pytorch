#pragma once

#ifndef BUILD_LITE_INTERPRETER
#include <torch/csrc/jit/backends/backend_debug_handler.h>
#endif
#include <torch/custom_class.h>

namespace torch {
namespace jit {

constexpr static auto kBackendUtilsNamespace = "backendutils";
constexpr static auto kBackendDebugInfoClass = "BackendDebugInfo";

#ifndef BUILD_LITE_INTERPRETER
/*
 * Custom class for holding debug information in lowered modules, intended
 * purely for keeping this information to be later serialized outside of the
 * lowered module itself.
 * Its usage pattern is:
 * 1. LoweredModule declares an instance of this class in __backend_debug_info
 * 2. During serialization, __backend_debug_info is used to obtain the debug
 *    information.
 * 3. The contents of LoweredModule.__backend_debug_info are not serialized
 *    within the LoweredModule itself.
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

#else

/*
 * Dummy instance exists for the following reason:
 * __backend_debug_info is of type BackendDebugInfo which is a torchbind'
 * class backed by cpp class PyTorchBackendDebugInfo.
 * PyTorchBackendDebugInfo, depends on ir.h., scope.h, source_range etc.
 * We dont include this on lite interpreter side. Thus on lite interpreter side
 * we cannot have valid definition of PyTorchBackendDebugInfo. However we do not
 * need valid instance of __backend_debug_info in lite interpreter anyway as we
 * dont serialize this info as part of LowerdModule as mentioned ealrier.
 * However since LoweredModule has registered attribute of __backend_debug_info
 * we still need to make sure that BackendDebugInfo is registered with
 * TorchScript. However in this instance it does not have to be backed by
 * PyTorchBackendDebugInfo, so we create a dummy PyTorchBackendDebugInfoDummy
 * just for this purpose.
 */
class PyTorchBackendDebugInfoDummy : public torch::CustomClassHolder {
 public:
  PyTorchBackendDebugInfoDummy() = default;
};
#endif
} // namespace jit
} // namespace torch
