#include <pybind11/pybind11.h>
#include <torch/csrc/jit/codegen/cuda/python_frontend/python_bindings.h>
// PYBIND11_MODULE(nvfuser, m) {
//     m.doc() = "nvfuser python API"; // optional module docstring
//     torch::jit::initNvFuserPythonBindings(m.ptr());
// }

PyObject* module;

#define ASSERT_TRUE(cmd) \
  if (!(cmd))            \
  return nullptr

extern "C"
#ifdef _WIN32
    __declspec(dllexport)
#endif
        TORCH_API PyObject* initModule();
// separate decl and defn for msvc error C2491
PyObject* initModule() {
  static struct PyModuleDef nvfusermodule = {
    PyModuleDef_HEAD_INIT, "nvfuser", nullptr, -1, nullptr};
  ASSERT_TRUE(module = PyModule_Create(&nvfusermodule));
  torch::jit::initNvFuserPythonBindings(module);
  return module;
}
