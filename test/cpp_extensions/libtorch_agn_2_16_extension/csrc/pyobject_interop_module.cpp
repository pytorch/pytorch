#include <Python.h>

#include <torch/csrc/stable/pyobject.h>
#include <torch/csrc/stable/tensor.h>

#include <exception>
#include <optional>
#include <string>
#include <vector>

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

PyObject* string_roundtrip(PyObject* /*self*/, PyObject* obj) {
  return call([&] {
    return torch::stable::string_to_pyobject(
        torch::stable::string_from_pyobject(obj));
  });
}

// PyObject -> T -> PyObject through the generic from_pyobject / to_pyobject.
template <typename T>
PyObject* roundtrip(PyObject* /*self*/, PyObject* obj) {
  return call([&] {
    return torch::stable::to_pyobject(torch::stable::from_pyobject<T>(obj));
  });
}

// Python sequence -> std::vector<int64_t> -> HeaderOnlyArrayRef<int64_t> ->
// Python list.
PyObject* int64_arrayref_roundtrip(PyObject* /*self*/, PyObject* obj) {
  return call([&] {
    const auto values = torch::stable::from_pyobject<std::vector<int64_t>>(obj);
    return torch::stable::to_pyobject(
        torch::headeronly::HeaderOnlyArrayRef<int64_t>(values));
  });
}

PyMethodDef methods[] = {
    {"layout_roundtrip", layout_roundtrip, METH_O, nullptr},
    {"tensor_layout", tensor_layout, METH_O, nullptr},
    {"memory_format_roundtrip", memory_format_roundtrip, METH_O, nullptr},
    {"bool_roundtrip", bool_roundtrip, METH_O, nullptr},
    {"int64_roundtrip", int64_roundtrip, METH_O, nullptr},
    {"double_roundtrip", double_roundtrip, METH_O, nullptr},
    {"string_roundtrip", string_roundtrip, METH_O, nullptr},
    {"optional_bool_roundtrip",
     roundtrip<std::optional<bool>>,
     METH_O,
     nullptr},
    {"optional_int64_roundtrip",
     roundtrip<std::optional<int64_t>>,
     METH_O,
     nullptr},
    {"optional_string_roundtrip",
     roundtrip<std::optional<std::string>>,
     METH_O,
     nullptr},
    {"optional_tensor_roundtrip",
     roundtrip<std::optional<torch::stable::Tensor>>,
     METH_O,
     nullptr},
    {"optional_device_roundtrip",
     roundtrip<std::optional<torch::stable::Device>>,
     METH_O,
     nullptr},
    {"double_list_roundtrip",
     roundtrip<std::vector<double>>,
     METH_O,
     nullptr},
    {"bool_list_roundtrip", roundtrip<std::vector<bool>>, METH_O, nullptr},
    {"string_list_list_roundtrip",
     roundtrip<std::vector<std::vector<std::string>>>,
     METH_O,
     nullptr},
    {"optional_tensor_list_roundtrip",
     roundtrip<
         std::optional<std::vector<std::optional<torch::stable::Tensor>>>>,
     METH_O,
     nullptr},
    {"int64_arrayref_roundtrip", int64_arrayref_roundtrip, METH_O, nullptr},
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
