#include <torch/csrc/autograd/python_mode.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/autograd/python_variable.h>
#include <c10/core/PythonDispatcher.h>

namespace torch { namespace autograd {

void PythonMode::enter(PyObject* dispatcher_type) {
  TORCH_CHECK(
    !c10::hasPythonDispatcher(),
    "python mode has already been set. We do not yet support nested python ",
    "mode. Please file us an issue and reset it before setting it again.")

  auto* interpreter = getPyInterpreter();

  // PythonDispatcher steals a reference, See NOTE [What is PythonDispatcher?]
  Py_INCREF(dispatcher_type);

  c10::setPythonDispatcher(
      std::make_shared<c10::PythonDispatcher>(dispatcher_type, interpreter));
}

void PythonMode::exit() {
  TORCH_CHECK(c10::hasPythonDispatcher(), "exiting Python Mode but it wasn't set!");

  c10::popPythonDispatcher();
}

bool PythonMode::enabled() noexcept {
  return c10::hasPythonDispatcher();
}

}}
