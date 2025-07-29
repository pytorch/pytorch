#include <torch/csrc/PyInterpreter.h>
#include <torch/csrc/PyInterpreterHooks.h>

namespace torch::detail {

PyInterpreterHooks::PyInterpreterHooks(c10::impl::PyInterpreterHooksArgs) {}

c10::impl::PyInterpreter* PyInterpreterHooks::getPyInterpreter() const {
  // Delegate to the existing implementation
  return ::getPyInterpreter();
}

} // namespace torch::detail

// Sigh, the registry doesn't support namespaces :(
using c10::impl::PyInterpreterHooksRegistry;
using c10::impl::RegistererPyInterpreterHooksRegistry;
using PyInterpreterHooks = torch::detail::PyInterpreterHooks;
// Register the implementation
REGISTER_PYTHON_HOOKS(PyInterpreterHooks)
