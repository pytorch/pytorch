#pragma once

#include <c10/macros/Export.h>

namespace c10::impl {

// This TLS was originally for controlling PyObject hermetic behavior with
// multiple interpreters (torchdeploy/multipy). Since we now only support
// a single interpreter, this is greatly simplified.
//
// In single-interpreter mode, we always permanently associate PyObject
// with Tensor the first time it is allocated (non-hermetic behavior).
struct C10_API HermeticPyObjectTLS {
  static void set_state(bool state);

  // In single-interpreter mode, always returns false (non-hermetic)
  static bool get_state() {
    // With only one interpreter, we never need hermetic PyObject behavior
    return false;
  }

  // No-op in single-interpreter mode
  static void init_state() {}

 private:
  // Kept for backward compatibility but unused in single-interpreter mode
  static bool get_tls_state();
};

} // namespace c10::impl
