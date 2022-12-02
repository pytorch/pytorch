#include <pybind11/pybind11.h>
#include <torch/csrc/jit/codegen/cuda/python_frontend/python_bindings.h>
#include <torch/csrc/utils/pybind.h>
// PYBIND11_MODULE(nvfuser, m) {
//     m.doc() = "nvfuser python API"; // optional module docstring
//     torch::jit::initNvFuserPythonBindings(m.ptr());
// }

#define ASSERT_TRUE(cmd) \
  if (!(cmd))            \
  return nullptr

extern "C"
#ifdef _WIN32
    __declspec(dllexport)
#endif
        TORCH_API PyObject* initModuleNvfuser();
// separate decl and defn for msvc error C2491
PyObject* initModuleNvfuser() {
  PyMethodDef m[] = {{nullptr, nullptr, 0, nullptr}}; // NOLINT
  static struct PyModuleDef torchmodule = {
      PyModuleDef_HEAD_INIT, "torch._C_nvfuser", nullptr, -1, m};
  PyObject* module;
  ASSERT_TRUE(module = PyModule_Create(&torchmodule));
  torch::jit::initNvFuserPythonBindings(module);
  return module;
}
