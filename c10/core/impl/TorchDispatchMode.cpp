#include <c10/core/DispatchKeySet.h>
#include <c10/core/SafePyObject.h>
#include <c10/core/impl/TorchDispatchMode.h>

namespace c10 {
namespace impl {

std::shared_ptr<SafePyObject> TorchDispatchMode::torchDispatchModeState;

void TorchDispatchMode::set_state(std::shared_ptr<SafePyObject> state) {
  if (state) {
    c10::add_to_default_include_set(
        {DispatchKey::Python, DispatchKey::PythonTLSSnapshot});
  } else {
    c10::remove_from_default_include_set(
        {DispatchKey::Python, DispatchKey::PythonTLSSnapshot});
  }
  std::atomic_store(&torchDispatchModeState, std::move(state));
}

std::shared_ptr<SafePyObject> TorchDispatchMode::get_state() {
  return std::atomic_load(&torchDispatchModeState);
}

bool dispatch_mode_enabled() {
  return static_cast<bool>(c10::impl::TorchDispatchMode::get_state());
}

} // namespace impl
} // namespace c10
