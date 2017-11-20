// ${generated_comment}

#include <Python.h>

#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/Size.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/autograd/utils/wrap_outputs.h"
#include "torch/csrc/utils/object_ptr.h"
#include "torch/csrc/utils/python_arg_parser.h"
#include "torch/csrc/utils/python_numbers.h"
#include "torch/csrc/utils/python_tuples.h"

#include "python_variable_methods_dispatch.h"

using at::Tensor;
using at::Scalar;
using at::ScalarType;
using namespace torch::autograd::utils;

namespace torch { namespace autograd {

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

PyObject * THPVariable_clamp(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "clamp(Scalar min=None, Scalar max=None)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
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

PyObject * THPVariable_clamp_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "clamp_(Scalar min=None, Scalar max=None)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  PyObject* parsed_args[2];
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
  PyObject* parsed_args[3];
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
  PyObject* parsed_args[3];
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

static Tensor dispatch_to_type(const Tensor & self, ScalarType scalarType) {
  AutoNoGIL no_gil;
  AutoGPU auto_gpu(self);
  return self.toType(scalarType);
}

static PyObject * THPVariable_to_type(PyObject* self, ScalarType scalarType) {
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return THPVariable_Wrap(dispatch_to_type(self_, scalarType));
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

static PyObject * THPVariable_storage(PyObject* self, PyObject* arg)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return createPyObject(*self_.storage());
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
  {"byte", (PyCFunction)THPVariable_byte, METH_NOARGS, NULL},
  {"char", (PyCFunction)THPVariable_char, METH_NOARGS, NULL},
  {"clamp", (PyCFunction)THPVariable_clamp, METH_VARARGS | METH_KEYWORDS, NULL},
  {"clamp_", (PyCFunction)THPVariable_clamp_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"dim", (PyCFunction)THPVariable_dim, METH_NOARGS, NULL},
  {"contiguous", (PyCFunction)THPVariable_contiguous, METH_NOARGS, NULL},
  {"detach", (PyCFunction)THPVariable_detach, METH_NOARGS, NULL},
  {"detach_", (PyCFunction)THPVariable_detach_, METH_NOARGS, NULL},
  {"double", (PyCFunction)THPVariable_double, METH_NOARGS, NULL},
  {"element_size", (PyCFunction)THPVariable_element_size, METH_NOARGS, NULL},
  {"float", (PyCFunction)THPVariable_float, METH_NOARGS, NULL},
  {"half", (PyCFunction)THPVariable_half, METH_NOARGS, NULL},
  {"int", (PyCFunction)THPVariable_int, METH_NOARGS, NULL},
  {"long", (PyCFunction)THPVariable_long, METH_NOARGS, NULL},
  {"ndimension", (PyCFunction)THPVariable_dim, METH_NOARGS, NULL},
  {"nelement", (PyCFunction)THPVariable_numel, METH_NOARGS, NULL},
  {"short", (PyCFunction)THPVariable_short, METH_NOARGS, NULL},
  {"size", (PyCFunction)THPVariable_size, METH_VARARGS | METH_KEYWORDS, NULL},
  {"storage", (PyCFunction)THPVariable_storage, METH_NOARGS, NULL},
  {"stride", (PyCFunction)THPVariable_stride, METH_VARARGS | METH_KEYWORDS, NULL},
  ${py_method_defs}
  {NULL}
};

}} // namespace torch::autograd
