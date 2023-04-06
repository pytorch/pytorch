#include <python_frontend/python_bindings.h>
#include <torch/extension.h>

PYBIND11_MODULE(EXTENSION_NAME, m) {
  m.doc() = "nvfuser C API python binding"; // optional module docstring
  torch::jit::initNvFuserPythonBindings(m.ptr());
}
