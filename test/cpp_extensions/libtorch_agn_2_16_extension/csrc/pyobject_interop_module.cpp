#include <Python.h>

#include <torch/csrc/stable/pyobject.h>
#include <torch/csrc/stable/tensor.h>

#include <exception>

// Importable (abi3) module functions that exercise the 2.16 Python interop
// stable shims: a raw PyObject arrives straight from Python (no dispatcher
// boxing), so the GIL is naturally held. The module links only libtorch; the
// conversion is serviced by the vtable that libtorch_python registers.

namespace {

// Runs fn, preserving a pending CPython exception on failure or setting a
// RuntimeError from the C++ diagnostic when no Python error is pending.
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

PyObject* memory_format_roundtrip(PyObject* /*self*/, PyObject* obj) {
  return call([&] {
    return torch::stable::memory_format_to_pyobject(
        torch::stable::memory_format_from_pyobject(obj));
  });
}

PyObject* bool_roundtrip(PyObject* /*self*/, PyObject* obj) {
  return call([&] {
    return torch::stable::bool_to_pyobject(
        torch::stable::bool_from_pyobject(obj));
  });
}

PyObject* int64_roundtrip(PyObject* /*self*/, PyObject* obj) {
  return call([&] {
    return torch::stable::int64_to_pyobject(
        torch::stable::int64_from_pyobject(obj));
  });
}

PyObject* double_roundtrip(PyObject* /*self*/, PyObject* obj) {
  return call([&] {
    return torch::stable::double_to_pyobject(
        torch::stable::double_from_pyobject(obj));
  });
}

PyObject* double_error_message(PyObject* /*self*/, PyObject* obj) {
  try {
    torch::stable::double_from_pyobject(obj);
  } catch (const std::exception& error) {
    // Model a C++ caller handling the failure rather than propagating PyErr.
    PyErr_Clear();
    return PyUnicode_FromString(error.what());
  }
  PyErr_SetString(PyExc_ValueError, "expected double conversion to fail");
  return nullptr;
}

PyMethodDef methods[] = {
    {"layout_roundtrip", layout_roundtrip, METH_O, nullptr},
    {"tensor_layout", tensor_layout, METH_O, nullptr},
    {"memory_format_roundtrip", memory_format_roundtrip, METH_O, nullptr},
    {"bool_roundtrip", bool_roundtrip, METH_O, nullptr},
    {"int64_roundtrip", int64_roundtrip, METH_O, nullptr},
    {"double_roundtrip", double_roundtrip, METH_O, nullptr},
    {"double_error_message", double_error_message, METH_O, nullptr},
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
