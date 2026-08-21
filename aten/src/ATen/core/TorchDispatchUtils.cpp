#include <ATen/core/TorchDispatchUtils.h>


namespace at::impl {

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

bool tensorlist_has_dispatch(const c10::List<std::optional<at::Tensor>>& li) {
  for (const auto& element : li) {
    const c10::IValue& ivalue = element.get();
    if (!ivalue.isNone() && tensor_has_dispatch(ivalue.toTensor())) {
      return true;
    }
  }
  return false;
}

} // namespace at::impl
