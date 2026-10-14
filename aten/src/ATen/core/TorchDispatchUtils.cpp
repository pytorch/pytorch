#include <ATen/core/TorchDispatchUtils.h>


namespace at::impl {

bool tensor_has_dispatch(const at::Tensor& t) {
  // True if operators on this tensor are intercepted and implemented before
  // reaching a backend kernel: Python __torch_dispatch__ subclasses
  // (Python/PythonTLSSnapshot) and C++ FakeTensors (handled by the Fake
  // fallback). Fake does not need a dispatch_mode_enabled() analog: under a
  // C++ FakeTensorMode every tensor an operator produces has the Fake key.
  DispatchKeySet key_set(
      {DispatchKey::Python,
       DispatchKey::PythonTLSSnapshot,
       DispatchKey::Fake});
  return t.key_set().has_any(key_set);
}

bool tensorlist_has_dispatch(at::ITensorListRef li) {
  for (const auto& t : li) {
    if (tensor_has_dispatch(t)) {
      return true;
    }
  }
  return false;
}

bool tensorlist_has_dispatch(const c10::List<std::optional<at::Tensor>>& li) {
  for (auto i : c10::irange(li.size())) {
    auto t = li.get(i);
    if (t && tensor_has_dispatch(*t)) {
      return true;
    }
  }
  return false;
}

} // namespace at::impl
