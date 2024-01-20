#pragma once
#include <ATen/core/TorchDispatchUtils.h>

namespace at {
namespace impl {

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

TORCH_API void set_nested_tensor_cls(std::shared_ptr<c10::SafePyObject> t);

TORCH_API std::shared_ptr<c10::SafePyObject> get_nested_tensor_cls();

} // namespace impl
} // namespace at
