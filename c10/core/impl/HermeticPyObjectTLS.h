#pragma once

#include <c10/macros/Macros.h>

namespace c10 {
namespace impl {

// This TLS controls whether or not we permanently associate PyObject
// with Tensor the first time it is allocated.  When hermetic PyObject
// TLS is enabled (state is true), we DO NOT save PyObjects to Tensor,
// meaning you get a distinct PyObject whenever you execute the code in
// question.
struct C10_API HermeticPyObjectTLS {
  static void set_state(bool state);
  static bool get_state() {
    // Fastpath if torchdeploy isn't used.  In header so it inlines
    if (!haveState_)
      return false;
    return get_tls_state();
  }
  // Call this from the multipy/torchdeploy top level
  static void init_state();

 private:
  // This only flipped once from false to true during torchdeploy/multipy
  // initialization, and never again.
  static bool haveState_;
  static bool get_tls_state();
};

struct C10_API EnableHermeticPyObject {
  EnableHermeticPyObject() : old_(HermeticPyObjectTLS::get_state()) {
    HermeticPyObjectTLS::set_state(true);
  }
  ~EnableHermeticPyObject() {
    HermeticPyObjectTLS::set_state(old_);
  }
  bool old_;
};

} // namespace impl
} // namespace c10
