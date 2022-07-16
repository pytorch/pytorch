#pragma once

#include <torch/csrc/utils/torch_dispatch_mode.h>

namespace at {
namespace impl {

struct TORCH_API DispatchContextSnapshot {
  c10::impl::LocalDispatchKeySet key_set_;
  std::shared_ptr<c10::SafePyObject> dispatch_mode_;
};

struct TORCH_API RestorePythonTLSSnapshot {
  RestorePythonTLSSnapshot();
  ~RestorePythonTLSSnapshot();

private:
  DispatchContextSnapshot saved_;
  c10::impl::ForceDispatchKeyGuard key_guard_;
  torch::torch_dispatch_mode::StashTorchDispatchModeGuard mode_guard_;
};


// RAII guard to make working with the above TLS safer.
struct TORCH_API MaybeSetTLSOnEntryGuard {
public:
  MaybeSetTLSOnEntryGuard();
  ~MaybeSetTLSOnEntryGuard();

private:
  bool value_set_;
};

} // namespace impl
} // namespace at
