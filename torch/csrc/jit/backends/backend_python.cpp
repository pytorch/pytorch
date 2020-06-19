#include <torch/csrc/jit/backends/backend_python.h>
#include <ATen/core/builtin_function.h>
#include <torch/csrc/jit/backends/backend_detail.h>

namespace torch {
namespace jit {
namespace detail {

namespace {
// This struct combines a callback with the last backend (by index) that it has
// been called on. This helps figure out which backends this callback still
// needs to be called on.
struct Callback {
  Callback(BackendRegistrationCallback cb)
      : callback(std::move(cb)), last_called_on(0) {}
  BackendRegistrationCallback callback;
  unsigned last_called_on;
};

// Get a reference to the list of all callbacks that should be called when a
// backend is registered. This is primarily used for creating Python bindings
// for lowering to backends from Python.
std::vector<Callback>& getBackendRegistrationCallbacks() {
  static std::vector<Callback> callbacks;
  return callbacks;
}
} // namespace

void addBackendRegistrationCallback(BackendRegistrationCallback callback) {
  // Add the callback to the callback registry.
  getBackendRegistrationCallbacks().emplace_back(std::move(callback));

  // Call all callbacks on backends that they haven't been called on yet.
  auto& backends = getBackendRegistry();
  unsigned num_backends = backends.size();
  for (auto& cb : getBackendRegistrationCallbacks()) {
    for (unsigned i = cb.last_called_on; i < num_backends; ++i) {
      cb.callback(backends[i]);
    }
    cb.last_called_on = num_backends - 1;
  }
}
} // namespace detail
} // namespace jit
} // namespace torch
