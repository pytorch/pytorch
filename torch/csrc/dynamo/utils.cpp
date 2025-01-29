#include <torch/csrc/dynamo/utils.h>

namespace torch::dynamo {

// random utilities for C dynamo

PyObject* system_random_getstate() {
  return py::module_::import("random").attr("getstate")().release().ptr();
}

void system_random_setstate(PyObject* state) {
  py::module_::import("random").attr("setstate")(py::handle(state));
}

static std::array<PyMethodDef, 1> _methods = {{
    {nullptr,
     nullptr,
     0,
     nullptr} // Sentinel value indicating the end of the array
}};

bool is_instancemethod(py::object obj) {
  return PyInstanceMethod_Check(obj.ptr());
}

static struct PyModuleDef _module = {
    PyModuleDef_HEAD_INIT,
    "torch._C._dynamo.utils",
    "Module containing C utils",
    -1,
    _methods.data()};

PyObject* torch_c_dynamo_utils_init() {
  auto m = PyModule_Create(&_module);
  if (m == nullptr)
    return nullptr;

#ifdef Py_GIL_DISABLED
  PyUnstable_Module_SetGIL(m, Py_MOD_GIL_NOT_USED);
#endif

  auto py_m = py::handle(m).cast<py::module>();
  py_m.def("is_instancemethod", is_instancemethod);
  return m;
}

} // namespace torch::dynamo
