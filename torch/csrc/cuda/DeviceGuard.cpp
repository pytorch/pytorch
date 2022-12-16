#include <Python.h>

#include <torch/csrc/utils/cuda_lazy_init.h>
#include <torch/csrc/utils/python_arg_parser.h>

#include <c10/cuda/CUDAGuard.h>

namespace {
struct PyDeviceGuard {
  PyObject_HEAD

  c10::DeviceIndex idx_;
  c10::cuda::OptionalCUDAGuard guard_;

  void enter() {
    torch::utils::cuda_lazy_init();
    if (idx_ > 0) {
      guard_.set_index(idx_);
    }
  }

  void exit() {
    guard_.reset();
  }

  int get_idx() const {
    return static_cast<int>(idx_);
  }
  void set_idx(int idx) {
    TORCH_CHECK(
        !guard_.current_device().has_value(),
        "Cannot change index while guard is active");
    idx_ = static_cast<c10::DeviceIndex>(idx);
  }
  int get_prev_idx() const {
    auto od = guard_.original_device();
    return od.has_value() ? od->index() : -1;
  }
};

PyObject* PyCUDADeviceGuard_new(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwargs) {
  auto self = reinterpret_cast<PyDeviceGuard*>(type->tp_alloc(type, 0));
  if (self == nullptr)
    return nullptr;

  self = new (self) PyDeviceGuard;
  return reinterpret_cast<PyObject*>(self);
}

void PyCUDADeviceGuard_dealloc(PyDeviceGuard* self) {
  self->~PyDeviceGuard();
  Py_TYPE(self)->tp_free(self);
}

int PyCUDADeviceGuard_init(
    PyDeviceGuard* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser({"_DeviceGuard(int64_t idx)"});
  torch::ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  self->set_idx(static_cast<c10::DeviceIndex>(r.toInt64(0)));
  return 0;
  END_HANDLE_TH_ERRORS_RET(-1);
}

PyObject* PyCUDADeviceGuard_enter(PyDeviceGuard* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  self->enter();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* PyCUDADeviceGuard_exit(PyDeviceGuard* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  self->exit();
  Py_RETURN_FALSE;
  END_HANDLE_TH_ERRORS
}

PyMethodDef PyCUDADeviceGuard_methods[] = {
    {"__enter__", (PyCFunction)&PyCUDADeviceGuard_enter, METH_NOARGS, nullptr},
    {"__exit__", (PyCFunction)&PyCUDADeviceGuard_exit, METH_VARARGS, nullptr},
    {nullptr},
};

PyObject* PyCUDADeviceGuard_get_idx(PyDeviceGuard* self, void*) {
  HANDLE_TH_ERRORS
  return THPUtils_packInt64(self->get_idx());
  END_HANDLE_TH_ERRORS
}

int PyCUDADeviceGuard_set_idx(PyDeviceGuard* self, PyObject* arg, void*) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(arg), "invalid argument to setDevice");
  int64_t device = THPUtils_unpackLong(arg);
  self->set_idx(device);
  return 0;
  END_HANDLE_TH_ERRORS_RET(-1)
}

PyObject* PyCUDADeviceGuard_get_prev_idx(PyDeviceGuard* self, void*) {
  HANDLE_TH_ERRORS
  return THPUtils_packInt64(self->get_prev_idx());
  END_HANDLE_TH_ERRORS
}

PyGetSetDef PyCUDADeviceGuard_getset[] = {
    {"idx",
     (getter)&PyCUDADeviceGuard_get_idx,
     (setter)&PyCUDADeviceGuard_set_idx},
    {"prev_idx", (getter)&PyCUDADeviceGuard_get_prev_idx, nullptr},
    {nullptr},
};

PyType_Slot PyCUDADeviceGuard_Type_slots[] = {
    {Py_tp_methods, PyCUDADeviceGuard_methods},
    {Py_tp_getset, (void*)&PyCUDADeviceGuard_getset},
    {Py_tp_init, (void*)&PyCUDADeviceGuard_init},
    {Py_tp_new, (void*)&PyCUDADeviceGuard_new},
    {Py_tp_dealloc, (void*)&PyCUDADeviceGuard_dealloc},
    {0, 0},
};

PyType_Spec PyCUDADeviceGuard_spec{
    "torch._C._cuda_DeviceGuard",
    sizeof(PyDeviceGuard),
    0,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    PyCUDADeviceGuard_Type_slots};

}  // namespace (anonymous)

namespace torch {
namespace cuda {

void initDeviceGuard(PyObject* module) {
  PyObject* DeviceGuardType = PyType_FromSpec(&PyCUDADeviceGuard_spec);
  if (!DeviceGuardType) {
    throw python_error();
  }
  int err = PyModule_AddObject(module, "_cuda_DeviceGuard", DeviceGuardType);
  if (err) {
    throw python_error();
  }
}

} // namespace cuda
} // namespace torch
