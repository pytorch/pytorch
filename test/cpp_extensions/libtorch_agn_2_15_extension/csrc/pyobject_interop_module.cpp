#include <Python.h>

#include <torch/csrc/stable/pyobject.h>
#include <torch/csrc/stable/tensor.h>

// Importable (abi3) module functions that exercise the 2.15 Python interop
// stable shims: a raw PyObject arrives straight from Python (no dispatcher
// boxing), so the GIL is naturally held. The module links only libtorch; the
// conversion is serviced by the vtable that libtorch_python registers.

using torch::stable::Tensor;

namespace {

// Whether obj is a torch.Tensor (or a subclass), via the stable probe.
PyObject* is_tensor(PyObject* /*self*/, PyObject* obj) {
  try {
    return PyBool_FromLong(torch::stable::is_tensor_pyobject(obj));
  } catch (const std::exception& e) {
    if (!PyErr_Occurred()) {
      PyErr_SetString(PyExc_RuntimeError, e.what());
    }
    return nullptr;
  }
}

// PyObject -> ScalarType -> PyObject; torch.dtype objects are singletons, so
// the result is the input object.
PyObject* dtype_roundtrip(PyObject* /*self*/, PyObject* obj) {
  try {
    auto dtype = torch::stable::dtype_from_pyobject(obj);
    return static_cast<PyObject*>(torch::stable::dtype_to_pyobject(dtype));
  } catch (const std::exception& e) {
    if (!PyErr_Occurred()) {
      PyErr_SetString(PyExc_RuntimeError, e.what());
    }
    return nullptr;
  }
}

// PyObject -> stable Device -> PyObject.
PyObject* device_roundtrip(PyObject* /*self*/, PyObject* obj) {
  try {
    auto device = torch::stable::device_from_pyobject(obj);
    return static_cast<PyObject*>(torch::stable::device_to_pyobject(device));
  } catch (const std::exception& e) {
    if (!PyErr_Occurred()) {
      PyErr_SetString(PyExc_RuntimeError, e.what());
    }
    return nullptr;
  }
}

// Returns the tensor's dtype as a Python torch.dtype, going through the stable
// Tensor accessors end to end.
PyObject* tensor_dtype(PyObject* /*self*/, PyObject* obj) {
  try {
    Tensor t = torch::stable::tensor_from_pyobject(obj);
    return static_cast<PyObject*>(
        torch::stable::dtype_to_pyobject(t.scalar_type()));
  } catch (const std::exception& e) {
    if (!PyErr_Occurred()) {
      PyErr_SetString(PyExc_RuntimeError, e.what());
    }
    return nullptr;
  }
}

// Returns the tensor's device as a Python torch.device.
PyObject* tensor_device(PyObject* /*self*/, PyObject* obj) {
  try {
    Tensor t = torch::stable::tensor_from_pyobject(obj);
    return static_cast<PyObject*>(
        torch::stable::device_to_pyobject(t.device()));
  } catch (const std::exception& e) {
    if (!PyErr_Occurred()) {
      PyErr_SetString(PyExc_RuntimeError, e.what());
    }
    return nullptr;
  }
}

PyMethodDef methods[] = {
    {"is_tensor", is_tensor, METH_O, nullptr},
    {"dtype_roundtrip", dtype_roundtrip, METH_O, nullptr},
    {"device_roundtrip", device_roundtrip, METH_O, nullptr},
    {"tensor_dtype", tensor_dtype, METH_O, nullptr},
    {"tensor_device", tensor_device, METH_O, nullptr},
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
