#include <Python.h>

#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/pyobject.h>
#include <torch/csrc/stable/tensor.h>

// Importable (abi3) module functions that exercise the use case for the
// PyObject<->Tensor stable shims: a raw PyObject arrives straight from Python
// (no dispatcher boxing), so the GIL is naturally held. This is how a consumer
// (e.g. a python-registered custom op) would actually call tensor_from_pyobject
// / tensor_to_pyobject. The module links only libtorch; the conversion is
// serviced by
// the vtable that libtorch_python registers.

using torch::stable::Tensor;

namespace {

// PyObject -> stable Tensor -> PyObject; the result shares storage with input.
PyObject* pyobject_roundtrip(PyObject* /*self*/, PyObject* obj) {
  try {
    Tensor t = torch::stable::tensor_from_pyobject(obj);
    return static_cast<PyObject*>(torch::stable::tensor_to_pyobject(t));
  } catch (const std::exception& e) {
    if (!PyErr_Occurred()) {
      PyErr_SetString(PyExc_RuntimeError, e.what());
    }
    return nullptr;
  }
}

// args = (tensor, py_type); forces the result's exact Python type via
// tensor_to_pyobject's py_type argument (e.g. torch.nn.Parameter).
PyObject* pyobject_to_type(PyObject* /*self*/, PyObject* args) {
  PyObject* obj = nullptr;
  PyObject* py_type = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &obj, &py_type)) {
    return nullptr;
  }
  try {
    // Clone so the result wraps a fresh TensorImpl: tensor_to_pyobject with an explicit
    // py_type fails if the TensorImpl already has a Python object of a different
    // (non-subclass) type, which is the case for `obj` itself.
    Tensor t = torch::stable::clone(torch::stable::tensor_from_pyobject(obj));
    return static_cast<PyObject*>(torch::stable::tensor_to_pyobject(t, py_type));
  } catch (const std::exception& e) {
    if (!PyErr_Occurred()) {
      PyErr_SetString(PyExc_RuntimeError, e.what());
    }
    return nullptr;
  }
}

// Real work through the stable ABI on a tensor obtained from tensor_from_pyobject.
PyObject* pyobject_sum(PyObject* /*self*/, PyObject* obj) {
  try {
    Tensor t = torch::stable::tensor_from_pyobject(obj);
    return static_cast<PyObject*>(
        torch::stable::tensor_to_pyobject(torch::stable::sum(t)));
  } catch (const std::exception& e) {
    if (!PyErr_Occurred()) {
      PyErr_SetString(PyExc_RuntimeError, e.what());
    }
    return nullptr;
  }
}

PyMethodDef methods[] = {
    {"pyobject_roundtrip", pyobject_roundtrip, METH_O, nullptr},
    {"pyobject_to_type", pyobject_to_type, METH_VARARGS, nullptr},
    {"pyobject_sum", pyobject_sum, METH_O, nullptr},
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
