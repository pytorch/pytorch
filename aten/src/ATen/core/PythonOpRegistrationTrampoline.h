#pragma once

#include <ATen/core/dispatch/Dispatcher.h>

// TODO: We can get rid of this


namespace at::impl {

// Manages the single Python interpreter instance for PyTorch.
class TORCH_API PythonOpRegistrationTrampoline final {
  static c10::impl::PyInterpreter* interpreter_;

public:
  // Register the Python interpreter. Returns true on first registration,
  // false if an interpreter was already registered.
  static bool registerInterpreter(c10::impl::PyInterpreter*);

  // Returns the registered interpreter via the global PyInterpreter hooks.
  // Returns nullptr if no interpreter has been registered yet.
  static c10::impl::PyInterpreter* getInterpreter();
};

} // namespace at::impl
