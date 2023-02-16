#include <python_frontend/python_bindings.h>
#include <torch/extension.h>

PYBIND11_MODULE(EXTENSION_NAME, m) {
  m.doc() = "nvfuser C API python binding"; // optional module docstring
  nvfuser::python_frontend::initNvFuserPythonBindings(m.ptr());
}
