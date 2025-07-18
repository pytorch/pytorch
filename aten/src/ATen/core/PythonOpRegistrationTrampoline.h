#pragma once

#include <ATen/core/dispatch/Dispatcher.h>

// TODO: this can probably live in c10


namespace at::impl {

// Simplified version that only supports a single Python interpreter
// since torch deploy/multipy is no longer used
class TORCH_API PythonOpRegistrationTrampoline final {
  static c10::impl::PyInterpreter* interpreter_;

public:
  // Register the single Python interpreter
  // Returns true on first registration, false if already registered
  static bool registerInterpreter(c10::impl::PyInterpreter*);

  // Returns nullptr if no interpreter has been registered yet.
  static c10::impl::PyInterpreter* getInterpreter();
};

} // namespace at::impl
