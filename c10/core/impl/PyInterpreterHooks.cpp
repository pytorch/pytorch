#include <c10/core/impl/PyInterpreterHooks.h>

namespace c10::impl {

// Define the registry
C10_DEFINE_REGISTRY(
    PyInterpreterHooksRegistry,
    PyInterpreterHooksInterface,
    PyInterpreterHooksArgs)

const PyInterpreterHooksInterface& getPyInterpreterHooks() {
  auto create_impl = [] {
#if !defined C10_MOBILE
    auto hooks = PyInterpreterHooksRegistry()->Create(
        "PyInterpreterHooks", PyInterpreterHooksArgs{});
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
  return getPyInterpreterHooks().getPyInterpreter();
}

} // namespace c10::impl
