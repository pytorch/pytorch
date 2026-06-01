#include <c10/core/impl/PyInterpreterHooks.h>

namespace c10::impl {

// Define the registry
C10_DEFINE_REGISTRY(PyInterpreterHooksRegistry, PyInterpreterHooksInterface)

const PyInterpreterHooksInterface& getPyInterpreterHooks() {
  auto create_impl = [] {
#if !defined C10_MOBILE
    auto hooks = PyInterpreterHooksRegistry()->Create("PyInterpreterHooks");
    if (hooks) {
      return hooks;
    }
#endif
    // Return stub implementation that will throw errors when methods are called
    return std::make_unique<PyInterpreterHooksInterface>();
  };
  static auto hooks = create_impl();
  return *hooks;
}

// Main function to get global PyInterpreter
PyInterpreter* getGlobalPyInterpreter() {
  static PyInterpreter* cached = getPyInterpreterHooks().getPyInterpreter();
  return cached;
}

} // namespace c10::impl
