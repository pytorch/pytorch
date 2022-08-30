#include <ATen/core/TorchDispatchUtils.h>

namespace at {
namespace impl {

bool tensor_has_dispatch(const at::Tensor& t) {
  DispatchKeySet key_set({DispatchKey::Python, DispatchKey::PythonTLSSnapshot});
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

bool tensorlist_has_dispatch(at::IOptTensorListRef li) {
  for (const auto& t : li) {
    if (t.has_value() && tensor_has_dispatch(*t)) {
      return true;
    }
  }
  return false;
}

} // namespace impl
} // namespace at
