#pragma once

#ifdef USE_XPU
// WARNING: Be careful when adding new includes here. This header will be used
// in model.so, and should not refer to any aten/c10 headers except the stable
// C ABI defined in torch/csrc/inductor/aoti_torch/c/shim.h. The same rule
// applies to other files under torch/csrc/inductor/aoti_runtime/.
#include <torch/csrc/inductor/aoti_runtime/utils.h>

namespace torch::aot_inductor {

inline void delete_xpu_guard(void* ptr) {
  AOTI_TORCH_ERROR_CODE_CHECK(
      aoti_torch_delete_xpu_guard(reinterpret_cast<XPUGuardHandle>(ptr)));
}

inline void delete_xpu_stream_guard(void* ptr) {
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_delete_xpu_stream_guard(
      reinterpret_cast<XPUStreamGuardHandle>(ptr)));
}

class AOTIXpuGuard {
 public:
  AOTIXpuGuard(int32_t device_index) : guard_(nullptr, delete_xpu_guard) {
    XPUGuardHandle ptr = nullptr;
    AOTI_TORCH_ERROR_CODE_CHECK(
        aoti_torch_create_xpu_guard(device_index, &ptr));
    guard_.reset(ptr);
  }

  void set_index(int32_t device_index) {
    AOTI_TORCH_ERROR_CODE_CHECK(
        aoti_torch_xpu_guard_set_index(guard_.get(), device_index));
  }

 private:
  std::unique_ptr<XPUGuardOpaque, DeleterFnPtr> guard_;
};

class AOTIXpuStreamGuard {
 public:
  AOTIXpuStreamGuard(void* stream, int32_t device_index)
      : guard_(nullptr, delete_xpu_stream_guard) {
    XPUStreamGuardHandle ptr = nullptr;
    AOTI_TORCH_ERROR_CODE_CHECK(
        aoti_torch_create_xpu_stream_guard(stream, device_index, &ptr));
    guard_.reset(ptr);
  }

 private:
  std::unique_ptr<XPUStreamGuardOpaque, DeleterFnPtr> guard_;
};

} // namespace torch::aot_inductor
#endif // USE_XPU
