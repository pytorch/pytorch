#include "OpenReg.h"

#include <ATen/Context.h>

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/python_numbers.h>

static PyObject* _initExtension(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS

  at::globalContext().lazyInitDevice(c10::DeviceType::PrivateUse1);

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* _getDefaultGenerator(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      THPUtils_checkLong(arg),
      "_get_default_generator expects an int, but got ",
      THPUtils_typename(arg));
  auto idx = static_cast<int>(THPUtils_unpackLong(arg));

  return THPGenerator_initDefaultGenerator(
      at::globalContext().defaultGenerator(
          c10::Device(c10::DeviceType::PrivateUse1, idx)));

  END_HANDLE_TH_ERRORS
}

static PyMethodDef methods[] = {
    {"_init", _initExtension, METH_NOARGS, nullptr},
    {"_get_default_generator", _getDefaultGenerator, METH_O, nullptr},
    {nullptr, nullptr, 0, nullptr}
};

static struct PyModuleDef openreg_C_module =
    {PyModuleDef_HEAD_INIT, "pytorch_openreg._C", nullptr, -1, methods};

PyMODINIT_FUNC PyInit__C(void) {
  PyObject* mod = PyModule_Create(&openreg_C_module);

  py::object openreg_mod = py::module_::import("pytorch_openreg");
  // Only borrowed from the python side!
  openreg::set_impl_factory(openreg_mod.attr("impl_factory").ptr());

  return mod;
}
