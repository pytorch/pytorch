#include <torch/csrc/dynamo/utils.h>

namespace torch::dynamo {

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

  auto py_m = py::handle(m).cast<py::module>();
  py_m.def("is_instancemethod", is_instancemethod);
  return m;
}

} // namespace torch::dynamo
