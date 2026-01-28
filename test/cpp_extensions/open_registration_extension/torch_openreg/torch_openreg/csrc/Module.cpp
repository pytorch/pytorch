#include <ATen/Context.h>

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils.h>
#include <torch/csrc/utils/device_lazy_init.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/python_numbers.h>

#include <runtime/OpenRegFunctions.h>

static PyObject* _initExtension(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS

  torch::utils::register_fork_handler_for_device_init(at::kPrivateUse1);
  at::globalContext().lazyInitDevice(c10::DeviceType::PrivateUse1);

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* _isInBadFork(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  return PyBool_FromLong(torch::utils::is_device_in_bad_fork(at::kPrivateUse1));
  END_HANDLE_TH_ERRORS
}

// LITERALINCLUDE START: OPENREG GET DEFAULT GENERATOR
static PyObject* _getDefaultGenerator(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      THPUtils_checkLong(arg),
      "_get_default_generator expects an int, but got ",
      THPUtils_typename(arg));
  auto idx = static_cast<int>(THPUtils_unpackLong(arg));

  torch::utils::register_fork_handler_for_device_init(at::kPrivateUse1);
  return THPGenerator_initDefaultGenerator(
      at::globalContext().defaultGenerator(
          c10::Device(c10::DeviceType::PrivateUse1, idx)));

  END_HANDLE_TH_ERRORS
}
// LITERALINCLUDE END: OPENREG GET DEFAULT GENERATOR

// LITERALINCLUDE START: MODULE SET DEVICE HELPER

PyObject* _setDevice(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(arg), "invalid argument to setDevice");
  auto device = THPUtils_unpackDeviceIndex(arg);
  torch::utils::device_lazy_init(at::kPrivateUse1);
  c10::openreg::set_device(device);

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// LITERALINCLUDE END: MODULE SET DEVICE HELPER

PyObject* _exchangeDevice(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(arg), "invalid argument to exchangeDevice");
  auto device_index = THPUtils_unpackDeviceIndex(arg);
  if (device_index < 0) {
    return THPUtils_packInt32(-1);
  }

  torch::utils::device_lazy_init(at::kPrivateUse1);
  auto current_device = c10::openreg::ExchangeDevice(device_index);

  return THPUtils_packDeviceIndex(current_device);
  END_HANDLE_TH_ERRORS
}

PyObject* _getDevice(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  torch::utils::device_lazy_init(at::kPrivateUse1);
  auto device = static_cast<int32_t>(c10::openreg::current_device());
  return THPUtils_packInt32(device);
  END_HANDLE_TH_ERRORS
}

PyObject* _getDeviceCount(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  torch::utils::register_fork_handler_for_device_init(at::kPrivateUse1);
  return THPUtils_packUInt64(c10::openreg::device_count());
  END_HANDLE_TH_ERRORS
}

// LITERALINCLUDE START: OPENREG MODULE METHODS
static PyMethodDef methods[] = {
    {"_init", _initExtension, METH_NOARGS, nullptr},
    {"_isInBadFork", _isInBadFork, METH_NOARGS, nullptr},
    {"_get_default_generator", _getDefaultGenerator, METH_O, nullptr},
    {"_get_device", _getDevice, METH_NOARGS, nullptr},
    {"_set_device", _setDevice, METH_O, nullptr},
    {"_exchangeDevice", _exchangeDevice, METH_O, nullptr},
    {"_get_device_count", _getDeviceCount, METH_NOARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}};
// LITERALINCLUDE END: OPENREG MODULE METHODS
/*
 * When ASAN is enabled, PyTorch modifies the dlopen flag during import,
 * causing all global and weak symbols in _C.so and its dependent libraries
 * to be exposed to the global symbol scope, which in turn causes
 * subsequent symbols with the same name in other libraries to be intercepted.
 * Therefore, it cannot be named initModule here, otherwise initModule
 * in torch/csrc/Module.cpp will be called, resulting in failure.
 */
extern "C" OPENREG_EXPORT PyObject* initOpenRegModule(void) {
  static struct PyModuleDef openreg_C_module = {
      PyModuleDef_HEAD_INIT, "torch_openreg._C", nullptr, -1, methods};
  PyObject* mod = PyModule_Create(&openreg_C_module);

  return mod;
}
