#include <Python.h>

#include <torch/csrc/stable/pyobject.h>
#include <torch/csrc/stable/tensor.h>

#include <exception>

// Importable (abi3) module functions that exercise the 2.16 Python interop
// stable shims: a raw PyObject arrives straight from Python (no dispatcher
// boxing), so the GIL is naturally held. The module links only libtorch; the
// conversion is serviced by the vtable that libtorch_python registers.

namespace {

// Runs fn, turning a C++ exception into a Python RuntimeError. A Python error
// that is already pending is left alone, so a conversion that leaks the Python
// error indicator surfaces as a different exception type.
template <typename F>
PyObject* call(F&& fn) {
  try {
    return static_cast<PyObject*>(fn());
  } catch (const std::exception& e) {
    if (!PyErr_Occurred()) {
      PyErr_SetString(PyExc_RuntimeError, e.what());
    }
    return nullptr;
  }
}

PyObject* layout_roundtrip(PyObject* /*self*/, PyObject* obj) {
  return call([&] {
    return torch::stable::layout_to_pyobject(
        torch::stable::layout_from_pyobject(obj));
  });
}

PyObject* tensor_layout(PyObject* /*self*/, PyObject* obj) {
  return call([&] {
    auto tensor = torch::stable::tensor_from_pyobject(obj);
    return torch::stable::layout_to_pyobject(tensor.layout());
  });
}

PyMethodDef methods[] = {
    {"layout_roundtrip", layout_roundtrip, METH_O, nullptr},
    {"tensor_layout", tensor_layout, METH_O, nullptr},
    {nullptr, nullptr, 0, nullptr}};

PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_interop",
    nullptr,
    -1,
    methods,
    nullptr,
    nullptr,
    nullptr,
    nullptr};

} // namespace

// PyMODINIT_FUNC (not a bare extern "C") so the init symbol is dllexported on
// Windows; without it the .pyd omits PyInit__interop and the import fails.
PyMODINIT_FUNC PyInit__interop(void) {
  return PyModule_Create(&moduledef);
}
