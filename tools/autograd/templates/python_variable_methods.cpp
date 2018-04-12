// ${generated_comment}

#include <Python.h>

#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/Size.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/autograd/utils/wrap_outputs.h"
#ifdef WITH_CUDA
#include "torch/csrc/cuda/Stream.h"
#endif
#include "torch/csrc/utils/cuda_lazy_init.h"
#include "torch/csrc/utils/object_ptr.h"
#include "torch/csrc/utils/python_arg_parser.h"
#include "torch/csrc/utils/python_numbers.h"
#include "torch/csrc/utils/python_strings.h"
#include "torch/csrc/utils/python_tuples.h"
#include "torch/csrc/utils/tensor_apply.h"
#include "torch/csrc/utils/tensor_conversion_dispatch.h"
#include "torch/csrc/utils/tensor_list.h"
#include "torch/csrc/utils/tensor_new.h"
#include "torch/csrc/utils/tensor_numpy.h"
#include "torch/csrc/utils/tensor_types.h"

#include "python_variable_methods_dispatch.h"

using at::Tensor;
using at::Scalar;
using at::ScalarType;
using at::Backend;
using namespace torch::autograd::utils;

namespace torch { namespace autograd {

static PyObject * THPVariable_apply_(PyObject* self, PyObject* arg)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  if (self_.requires_grad()) {
    throw std::runtime_error(
        "Can't call apply_() on Variable that requires grad. Use "
        "var.detach().apply_() instead.");
  }
  return THPVariable_Wrap(torch::utils::apply_(self_, arg));
  END_HANDLE_TH_ERRORS
}

static Tensor dispatch_clamp(const Tensor & self, Scalar min, Scalar max) {
  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.clamp(min, max);
}
static Tensor dispatch_clamp_min(const Tensor & self, Scalar min) {
  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.clamp_min(min);
}
static Tensor dispatch_clamp_max(const Tensor & self, Scalar max) {
  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.clamp_max(max);
}

static PyObject * THPVariable_clamp(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "clamp(Scalar min=None, Scalar max=None)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (!r.isNone(0) && !r.isNone(1)) {
    return THPVariable_Wrap(dispatch_clamp(self_, r.scalar(0), r.scalar(1)));
  } else if (!r.isNone(0)) {
    return THPVariable_Wrap(dispatch_clamp_min(self_, r.scalar(0)));
  } else if (!r.isNone(1)) {
    return THPVariable_Wrap(dispatch_clamp_max(self_, r.scalar(1)));
  } else {
    throw std::runtime_error("At least one of 'min' or 'max' must not be None");
  }
  END_HANDLE_TH_ERRORS
}

static Tensor & dispatch_clamp_(Tensor & self, Scalar min, Scalar max) {
  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.clamp_(min, max);
}
static Tensor & dispatch_clamp_min_(Tensor & self, Scalar min) {
  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.clamp_min_(min);
}
static Tensor & dispatch_clamp_max_(Tensor & self, Scalar max) {
  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.clamp_max_(max);
}

static PyObject * THPVariable_clamp_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "clamp_(Scalar min=None, Scalar max=None)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (!r.isNone(0) && !r.isNone(1)) {
    return THPVariable_Wrap(dispatch_clamp_(self_, r.scalar(0), r.scalar(1)));
  } else if (!r.isNone(0)) {
    return THPVariable_Wrap(dispatch_clamp_min_(self_, r.scalar(0)));
  } else if (!r.isNone(1)) {
    return THPVariable_Wrap(dispatch_clamp_max_(self_, r.scalar(1)));
  } else {
    throw std::runtime_error("At least one of 'min' or 'max' must not be None");
  }
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_size(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "size(int64_t dim)",
    "size()",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(self_.size(r.toInt64(0)));
  } else if (r.idx == 1) {
    // Yes, this is called sizes in ATen
    IntList sizes = self_.sizes();
    // we can't do the normal wrapping here because IntList maps to both
    // torch.Size and tuple in python.
    return THPSize_New(sizes.size(), sizes.data());
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_stride(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "stride(int64_t dim)",
    "stride()",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(self_.stride(r.toInt64(0)));
  } else if (r.idx == 1) {
    // yes, this is called strides in ATen.
    IntList strides = self_.strides();
    // we can't do the normal wrapping here because IntList maps to both
    // torch.Size and tuple in python
    return THPUtils_packInt64Array(strides.size(), strides.data());
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_dim(PyObject* self, PyObject* args)
{
   HANDLE_TH_ERRORS
   auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
   return THPUtils_packInt64(self_.dim());
   END_HANDLE_TH_ERRORS
}

static Tensor dispatch_contiguous(const Tensor & self) {
  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.contiguous();
}

static PyObject * THPVariable_contiguous(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  // avoids touching the GIL or current device if self is already contiguous
  if (self_.is_contiguous()) {
    Py_INCREF(self);
    return self;
  }
  return THPVariable_Wrap(dispatch_contiguous(self_));
  END_HANDLE_TH_ERRORS
}

static Tensor dispatch_copy_(Tensor & self, const Tensor & other, bool non_blocking) {
  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.copy_(other, non_blocking);
}

static PyObject * THPVariable_copy_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "copy_(Tensor other, bool non_blocking=False)",
    "copy_(Tensor other, bool async=False)|deprecated"
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  return THPVariable_Wrap(dispatch_copy_(self_, r.tensor(0), r.toBool(1)));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_detach(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return THPVariable_Wrap(self_.detach());
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_detach_(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  reinterpret_cast<THPVariable*>(self)->cdata.detach_();
  Py_INCREF(self);
  return self;
  END_HANDLE_TH_ERRORS
}

static double dispatch_to_CDouble(const Tensor & self) {
  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  if (self.numel() != 1) {
    throw ValueError("only one element tensors can be converted to Python scalars");
  }
  return self.toCDouble();
}

static int64_t dispatch_to_CLong(const Tensor & self) {
  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  if (self.numel() != 1) {
    throw ValueError("only one element tensors can be converted to Python scalars");
  }
  return self.toCLong();
}

static PyObject * THPVariable_float_scalar(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_to_CDouble(self_));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_integral_scalar(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  if (isFloatingType(self_.type().scalarType())) {
    // we can't dispatch to toCLong here because we want to avoid ATen overflow checks;
    // the python integral type (long in python2) can't overflow.
    return THPUtils_packDoubleAsInt(dispatch_to_CDouble(self_));
  } else {
    return wrap(dispatch_to_CLong(self_));
  }
  END_HANDLE_TH_ERRORS
}

// This is the __index__ function in Python which is similar to __int__, but
// called when used as a slice.
static PyObject * THPVariable_index_scalar(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  // TODO: change the condition to `self_.dim() != 0` once we expose scalars
  // in PyTorch.
  if (!isIntegralType(self_.type().scalarType()) || self_.numel() != 1) {
    throw TypeError("only integer tensors of a single element can be converted to an index");
  }
  return wrap(dispatch_to_CLong(self_));
  END_HANDLE_TH_ERRORS
}

static Tensor dispatch_invert(const Tensor & self) {
  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return 1 - self;
}

static PyObject * THPVariable_invert(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  if (self_.type().scalarType() != at::kByte) {
    throw TypeError("~ (operator.invert) is only implemented on byte tensors");
  }
  return THPVariable_Wrap(dispatch_invert(self_));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_cpu(PyObject* self, PyObject* args)
{
   HANDLE_TH_ERRORS
   auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
   auto backend = self_.is_sparse() ? Backend::SparseCPU : Backend::CPU;
   auto& type = self_.type().toBackend(backend);
   return wrap(torch::utils::dispatch_type_conversion(self_, type));
   END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_cuda(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cuda(Device? device=None, bool non_blocking=False)",
    "cuda(Device? device=None, bool async=False)|deprecated"
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto backend = self_.is_sparse() ? at::kSparseCUDA : at::kCUDA;
  auto& type = self_.type().toBackend(backend);
  auto device_obj = r.device(0);
  if (!r.isNone(0) && device_obj.type != DeviceType::CUDA) {
    throw std::runtime_error("Invalid device, must be cuda device");
  }
  auto device = (device_obj.is_default || device_obj.type == DeviceType::CPU) ? -1 : device_obj.index;
  return THPVariable_Wrap(torch::utils::dispatch_type_conversion(self_, type, device, r.toBool(1)));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_to_type(PyObject* self, ScalarType scalarType) {
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  auto& type = self_.type().toScalarType(scalarType);
  return THPVariable_Wrap(torch::utils::dispatch_type_conversion(self_, type));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_byte(PyObject* self, PyObject* args) {
  return THPVariable_to_type(self, ScalarType::Byte);
}

static PyObject * THPVariable_char(PyObject* self, PyObject* args) {
  return THPVariable_to_type(self, ScalarType::Char);
}

static PyObject * THPVariable_double(PyObject* self, PyObject* args) {
  return THPVariable_to_type(self, ScalarType::Double);
}

static PyObject * THPVariable_float(PyObject* self, PyObject* args) {
  return THPVariable_to_type(self, ScalarType::Float);
}

static PyObject * THPVariable_half(PyObject* self, PyObject* args) {
  return THPVariable_to_type(self, ScalarType::Half);
}

static PyObject * THPVariable_int(PyObject* self, PyObject* args) {
  return THPVariable_to_type(self, ScalarType::Int);
}

static PyObject * THPVariable_long(PyObject* self, PyObject* args) {
  return THPVariable_to_type(self, ScalarType::Long);
}

static PyObject * THPVariable_short(PyObject* self, PyObject* args) {
  return THPVariable_to_type(self, ScalarType::Short);
}

static PyObject * THPVariable_element_size(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  size_t element_size = self_.type().elementSizeInBytes();
  return THPUtils_packInt64(element_size);
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_numpy(PyObject* self, PyObject* arg)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  if (self_.requires_grad()) {
    throw std::runtime_error(
        "Can't call numpy() on Variable that requires grad. "
        "Use var.detach().numpy() instead.");
  }
  return torch::utils::tensor_to_numpy(self_.data());
  END_HANDLE_TH_ERRORS
}

// TODO: move this to ATen. We would need to expose Stream objects in ATen.
static PyObject * THPVariable_record_stream(PyObject* self, PyObject* arg)
{
  HANDLE_TH_ERRORS
#ifdef WITH_CUDA
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  if (!THCPStream_Check(arg)) {
    return PyErr_Format(PyExc_TypeError, "expected Stream object");
  }
  void* data = self_.data_ptr();
  THCCachingAllocator_recordStream(data, ((THCPStream*)arg)->cdata);
  Py_RETURN_NONE;
#else
  throw std::runtime_error("PyTorch compiled without CUDA support");
#endif
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_item(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  if (self_.is_floating_point()) {
    return wrap(dispatch_to_CDouble(self_));
  } else {
    return wrap(dispatch_to_CLong(self_));
  }
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_map_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({ "map_(Tensor other, PyObject* callable)" });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  Variable other = r.tensor(0);
  if (self_.requires_grad() || other.requires_grad()) {
    throw std::runtime_error(
        "Can't call map_() on Variable that requires grad. Use "
        "var.detach().map_() instead.");
  }
  return THPVariable_Wrap(torch::utils::map_(self_, other, r.pyobject(1)));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_map2_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({ "map2_(Tensor x, Tensor y, PyObject* callable)" });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  Variable x = r.tensor(0);
  Variable y = r.tensor(1);
  if (self_.requires_grad() || x.requires_grad() || y.requires_grad()) {
    throw std::runtime_error(
        "Can't call map2_() on Variable that requires grad. Use "
        "var.detach().map2_() instead.");
  }
  return THPVariable_Wrap(torch::utils::map2_(self_, x, y, r.pyobject(2)));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_new(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  AutoGPU auto_gpu(self_);
  return THPVariable_Wrap(torch::utils::legacy_tensor_new(self_.type(), args, kwargs));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_new_empty(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  AutoGPU auto_gpu(self_);
  return THPVariable_Wrap(torch::utils::new_empty(self_.type(), args, kwargs));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_new_full(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  AutoGPU auto_gpu(self_);
  return THPVariable_Wrap(torch::utils::new_full(self_.type(), args, kwargs));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_new_ones(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  AutoGPU auto_gpu(self_);
  return THPVariable_Wrap(torch::utils::new_ones(self_.type(), args, kwargs));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_new_tensor(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  AutoGPU auto_gpu(self_);
  return THPVariable_Wrap(torch::utils::new_tensor(self_.type(), args, kwargs));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_new_zeros(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  AutoGPU auto_gpu(self_);
  return THPVariable_Wrap(torch::utils::new_zeros(self_.type(), args, kwargs));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_storage(PyObject* self, PyObject* arg)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return createPyObject(*self_.storage());
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_storage_type(PyObject* self, PyObject* arg)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  auto storage = THPObjectPtr(createPyObject(*self_.storage()));
  auto storage_type = (PyObject*)Py_TYPE(storage);
  Py_INCREF(storage_type);
  return storage_type;
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_tolist(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return torch::utils::tensor_to_list(self_.data());
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_type(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "type(PyObject* dtype=None, bool non_blocking=False)",
    "type(PyObject* dtype=None, bool async=False)|deprecated"
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.isNone(0)) {
    return THPUtils_packString(torch::utils::type_to_string(self_.type()));
  }
  auto obj = r.pyobject(0);
  std::string type_name;
  bool is_dtype = false;
  if (PyType_Check(obj)) {
    if (obj == THPVariableClass) {
      type_name = "torch.Tensor";
    } else {
      type_name = ((PyTypeObject*)obj)->tp_name;
    }
  } else if (THPUtils_checkString(obj)) {
    type_name = THPUtils_unpackString(obj);
  } else if (THPDtype_Check(obj)) {
    is_dtype = true;
  } else {
    throw TypeError("dtype must be a type, str, or dtype object");
  }
  auto self_device_type = torch::getDeviceType(self_.type());
  auto& type = is_dtype ? torch::getType(r.scalartype(0), *torch::getLayout(self_.type().backend()), self_device_type) :
                          torch::utils::type_from_string(type_name);
  return THPVariable_Wrap(torch::utils::dispatch_type_conversion(self_, type, -1, r.toBool(1)));
  END_HANDLE_TH_ERRORS
}

// generated methods start here

${py_methods}

PyMethodDef variable_methods[] = {
  {"__add__", (PyCFunction)THPVariable_add, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__radd__", (PyCFunction)THPVariable_add, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__iadd__", (PyCFunction)THPVariable_add_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__rmul__", (PyCFunction)THPVariable_mul, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__mul__", (PyCFunction)THPVariable_mul, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__imul__", (PyCFunction)THPVariable_mul_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__sub__", (PyCFunction)THPVariable_sub, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__isub__", (PyCFunction)THPVariable_sub_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__div__", (PyCFunction)THPVariable_div, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__truediv__", (PyCFunction)THPVariable_div, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__idiv__", (PyCFunction)THPVariable_div_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__mod__", (PyCFunction)THPVariable_remainder, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__bool__", (PyCFunction)THPVariable_is_nonzero, METH_NOARGS, NULL},
  {"__float__", (PyCFunction)THPVariable_float_scalar, METH_NOARGS, NULL},
  {"__int__", (PyCFunction)THPVariable_integral_scalar, METH_NOARGS, NULL},
  {"__long__", (PyCFunction)THPVariable_integral_scalar, METH_NOARGS, NULL},
  {"__index__", (PyCFunction)THPVariable_index_scalar, METH_NOARGS, NULL},
  {"__invert__", (PyCFunction)THPVariable_invert, METH_NOARGS, NULL},
  {"__nonzero__", (PyCFunction)THPVariable_is_nonzero, METH_NOARGS, NULL},
  {"__matmul__", (PyCFunction)THPVariable_matmul, METH_VARARGS | METH_KEYWORDS, NULL},
  {"apply_", (PyCFunction)THPVariable_apply_, METH_O, NULL},
  {"byte", (PyCFunction)THPVariable_byte, METH_NOARGS, NULL},
  {"char", (PyCFunction)THPVariable_char, METH_NOARGS, NULL},
  {"clamp", (PyCFunction)THPVariable_clamp, METH_VARARGS | METH_KEYWORDS, NULL},
  {"clamp_", (PyCFunction)THPVariable_clamp_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"contiguous", (PyCFunction)THPVariable_contiguous, METH_NOARGS, NULL},
  {"copy_", (PyCFunction)THPVariable_copy_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"cpu", (PyCFunction)THPVariable_cpu, METH_NOARGS, NULL},
  {"cuda", (PyCFunction)THPVariable_cuda, METH_VARARGS | METH_KEYWORDS, NULL},
  {"dim", (PyCFunction)THPVariable_dim, METH_NOARGS, NULL},
  {"detach", (PyCFunction)THPVariable_detach, METH_NOARGS, NULL},
  {"detach_", (PyCFunction)THPVariable_detach_, METH_NOARGS, NULL},
  {"double", (PyCFunction)THPVariable_double, METH_NOARGS, NULL},
  {"element_size", (PyCFunction)THPVariable_element_size, METH_NOARGS, NULL},
  {"float", (PyCFunction)THPVariable_float, METH_NOARGS, NULL},
  {"half", (PyCFunction)THPVariable_half, METH_NOARGS, NULL},
  {"int", (PyCFunction)THPVariable_int, METH_NOARGS, NULL},
  {"item", (PyCFunction)THPVariable_item, METH_NOARGS, NULL},
  {"long", (PyCFunction)THPVariable_long, METH_NOARGS, NULL},
  {"map_", (PyCFunction)THPVariable_map_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"map2_", (PyCFunction)THPVariable_map2_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"ndimension", (PyCFunction)THPVariable_dim, METH_NOARGS, NULL},
  {"nelement", (PyCFunction)THPVariable_numel, METH_NOARGS, NULL},
  {"new", (PyCFunction)THPVariable_new, METH_VARARGS | METH_KEYWORDS, NULL},
  {"new_empty", (PyCFunction)THPVariable_new_empty, METH_VARARGS | METH_KEYWORDS, NULL},
  {"new_full", (PyCFunction)THPVariable_new_full, METH_VARARGS | METH_KEYWORDS, NULL},
  {"new_ones", (PyCFunction)THPVariable_new_ones, METH_VARARGS | METH_KEYWORDS, NULL},
  {"new_tensor", (PyCFunction)THPVariable_new_tensor, METH_VARARGS | METH_KEYWORDS, NULL},
  {"new_zeros", (PyCFunction)THPVariable_new_zeros, METH_VARARGS | METH_KEYWORDS, NULL},
  {"numpy", (PyCFunction)THPVariable_numpy, METH_NOARGS, NULL},
  {"record_stream", (PyCFunction)THPVariable_record_stream, METH_O, NULL},
  {"short", (PyCFunction)THPVariable_short, METH_NOARGS, NULL},
  {"size", (PyCFunction)THPVariable_size, METH_VARARGS | METH_KEYWORDS, NULL},
  {"storage", (PyCFunction)THPVariable_storage, METH_NOARGS, NULL},
  {"storage_type", (PyCFunction)THPVariable_storage_type, METH_NOARGS, NULL},
  {"stride", (PyCFunction)THPVariable_stride, METH_VARARGS | METH_KEYWORDS, NULL},
  {"tolist", (PyCFunction)THPVariable_tolist, METH_NOARGS, NULL},
  {"type", (PyCFunction)THPVariable_type, METH_VARARGS | METH_KEYWORDS, NULL},
  ${py_method_defs}
  {NULL}
};

}} // namespace torch::autograd
