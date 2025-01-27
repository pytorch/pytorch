#include <torch/csrc/dynamo/utils.h>

namespace torch::dynamo {

// random utilities for C dynamo

// random module reference
py::object _random{py::none()};

PyObject* random_module() {
  if (_random.is_none()) {
    _random = py::module_::import("random");
  }
  return _random.ptr();
}

PyObject* random_getstate(PyObject* rng) {
  py::handle rng_h(rng);
  py::object state = rng_h.attr("getstate")();
  return state.release().ptr();
}

void random_setstate(PyObject* rng, PyObject* state) {
  py::handle rng_h(rng), state_h(state);
  rng_h.attr("setstate")(state_h);
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
