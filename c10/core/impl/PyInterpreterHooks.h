#pragma once

#include <c10/core/impl/PyInterpreter.h>
#include <c10/macros/Export.h>
#include <c10/util/Registry.h>
#include <memory>

namespace c10::impl {

// Minimal interface for PyInterpreter hooks
struct C10_API PyInterpreterHooksInterface {
  virtual ~PyInterpreterHooksInterface() = default;

  // Get the PyInterpreter instance
  // Stub implementation throws error when Python is not available
  // We return nullptr rather than throwing an error since there are bits of c10
  // that expect an empty PyObjectSlot when python is not available.
  virtual PyInterpreter* getPyInterpreter() const {
    return nullptr;
  }
};

struct C10_API PyInterpreterHooksArgs{};

C10_DECLARE_REGISTRY(
    PyInterpreterHooksRegistry,
    PyInterpreterHooksInterface,
    PyInterpreterHooksArgs);

#define REGISTER_PYTHON_HOOKS(clsname) \
  C10_REGISTER_CLASS(PyInterpreterHooksRegistry, clsname, clsname)

// Get the global PyInterpreter hooks instance
C10_API const PyInterpreterHooksInterface& getPyInterpreterHooks();

C10_API PyInterpreter* getGlobalPyInterpreter();

} // namespace c10::impl
