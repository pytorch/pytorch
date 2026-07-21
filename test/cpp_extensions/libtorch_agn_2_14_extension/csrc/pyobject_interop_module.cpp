#include <Python.h>

#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/pyobject.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/version.h>

// Importable (abi3) module functions that exercise the real use case for the
// PyObject<->Tensor stable shims: a raw PyObject arrives straight from Python
// (no dispatcher boxing), so the GIL is naturally held. This is how a consumer
// (e.g. a python-registered custom op) would actually call from_pyobject /
// to_pyobject. The module links only libtorch; the conversion is serviced by
// the vtable that libtorch_python registers.

#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_14_0

using torch::stable::Tensor;

namespace {

// PyObject -> stable Tensor -> PyObject; the result shares storage with input.
PyObject* pyobject_roundtrip(PyObject* /*self*/, PyObject* obj) {
  try {
    Tensor t = torch::stable::from_pyobject(obj);
    return static_cast<PyObject*>(torch::stable::to_pyobject(t));
  } catch (const std::exception& e) {
    if (!PyErr_Occurred()) {
      PyErr_SetString(PyExc_RuntimeError, e.what());
    }
    return nullptr;
  }
}

// args = (tensor, py_type); forces the result's exact Python type via
// to_pyobject's py_type argument (e.g. torch.nn.Parameter).
PyObject* pyobject_to_type(PyObject* /*self*/, PyObject* args) {
  PyObject* obj = nullptr;
  PyObject* py_type = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &obj, &py_type)) {
    return nullptr;
  }
  try {
    // Clone so the result wraps a fresh TensorImpl: to_pyobject with an explicit
    // py_type fails if the TensorImpl already has a Python object of a different
    // (non-subclass) type, which is the case for `obj` itself.
    Tensor t = torch::stable::clone(torch::stable::from_pyobject(obj));
    return static_cast<PyObject*>(torch::stable::to_pyobject(t, py_type));
  } catch (const std::exception& e) {
    if (!PyErr_Occurred()) {
      PyErr_SetString(PyExc_RuntimeError, e.what());
    }
    return nullptr;
  }
}

// Real work through the stable ABI on a tensor obtained from from_pyobject.
PyObject* pyobject_sum(PyObject* /*self*/, PyObject* obj) {
  try {
    Tensor t = torch::stable::from_pyobject(obj);
    return static_cast<PyObject*>(
        torch::stable::to_pyobject(torch::stable::sum(t)));
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
    "_C",
    nullptr,
    -1,
    methods,
    nullptr,
    nullptr,
    nullptr,
    nullptr};

} // namespace

extern "C" PyObject* PyInit__C() {
  return PyModule_Create(&moduledef);
}

#else

extern "C" PyObject* PyInit__C() {
  static PyModuleDef moduledef = {
      PyModuleDef_HEAD_INIT,
      "_C",
      nullptr,
      -1,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr};
  return PyModule_Create(&moduledef);
}

#endif // TORCH_FEATURE_VERSION >= TORCH_VERSION_2_14_0
