#pragma once

#include <ATen/core/dispatch/Dispatcher.h>

// TODO: this can probably live in c10


namespace at::impl {

// Manages the single Python interpreter instance for PyTorch.
// Since torch/deploy and multipy are deprecated, we now only support
// one Python interpreter per process.
class TORCH_API PythonOpRegistrationTrampoline final {
  static c10::impl::PyInterpreter* interpreter_;

public:
  // Register the Python interpreter. Returns true on first registration,
  // false if an interpreter was already registered.
  // In single-interpreter mode, only the first registration succeeds.
  static bool registerInterpreter(c10::impl::PyInterpreter*);

  // Returns the registered interpreter via the global PyInterpreter hooks.
  // Returns nullptr if no interpreter has been registered yet.
  static c10::impl::PyInterpreter* getInterpreter();
};

} // namespace at::impl
