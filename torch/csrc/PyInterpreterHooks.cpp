#include <torch/csrc/PyInterpreterHooks.h>
#include <torch/csrc/PyInterpreter.h>

namespace torch::detail {

PyInterpreterHooks::PyInterpreterHooks(c10::impl::PyInterpreterHooksArgs) {}

c10::impl::PyInterpreter* PyInterpreterHooks::getPyInterpreter() const {
  // Delegate to the existing implementation
  return ::getPyInterpreter();
}

} // namespace torch::detail

// Register the implementation
REGISTER_PYTHON_HOOKS(torch::detail::PyInterpreterHooks);