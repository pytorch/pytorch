#pragma once
#include <ATen/core/TorchDispatchUtils.h>


namespace at::impl {

struct TORCH_API RestorePythonTLSSnapshot {
  RestorePythonTLSSnapshot();
  ~RestorePythonTLSSnapshot();

private:
  c10::impl::LocalDispatchKeySet saved_;
  c10::impl::ForceDispatchKeyGuard guard_;
};


// RAII guard to make working with the above TLS safer.
struct TORCH_API MaybeSetTLSOnEntryGuard {
public:
  MaybeSetTLSOnEntryGuard();
  ~MaybeSetTLSOnEntryGuard();

private:
  bool value_set_;
};

} // namespace at::impl
