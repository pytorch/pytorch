#pragma once

#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/pythoncapi_compat.h>

namespace torch::utils {

// Set a Python object as immortal, i.e. living for the same lifetime as the
// entire runtime.
//
// Reference counting is expensive on the free-threaded build when threads
// incref/decref objects that they don't own, there's an atomic
// read-modify-write that leads to contended cache lines.  This particularly
// hurts common shared objects like singletons.
//
// Return true if the object was successfully immortalized.
inline bool set_immortal_if_possible([[maybe_unused]] PyObject* obj) {
  // The unstable API for immortalizing objects is added in Python 3.15, but the
  // compat header includes support for it starting with Python 3.13.
#if PY_VERSION_HEX >= 0x030D0000
  return PyUnstable_SetImmortal(obj) > 0;
#else
  return false;
#endif
}

} // namespace torch::utils
