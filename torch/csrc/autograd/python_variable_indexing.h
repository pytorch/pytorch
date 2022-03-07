#pragma once

#include <torch/csrc/python_headers.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/utils/python_compat.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/tensor_new.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/utils/tensor_types.h>

#include <ATen/TensorIndexing.h>
#include <ATen/TracerMode.h>
#include <c10/core/TensorOptions.h>
#include <ATen/DeviceGuard.h>
#include <ATen/ExpandUtils.h>
#include <c10/util/irange.h>
#include <ATen/core/LegacyTypeDispatch.h>

using namespace at;
using namespace torch::autograd::utils;

namespace torch { namespace autograd {

Py_ssize_t THPVariable_length(PyObject* self);
PyObject* THPVariable_getitem(PyObject* self, PyObject* index);
int THPVariable_setitem(PyObject* self, PyObject* index, PyObject* value);

static inline Variable valueToTensor(c10::TensorOptions options, PyObject* value, const at::Device& device) {
  if (THPVariable_Check(value)) {
    return THPVariable_Unpack(value);
  }
  at::AutoDispatchBelowADInplaceOrView guard;  // TODO: remove
  at::tracer::impl::NoTracerDispatchMode tracer_guard;
  if (THPUtils_checkLong(value) || PyBool_Check(value)) {
    return at::indexing::scalarToTensor(Scalar(THPUtils_unpackLong(value)), options, device);
  }
  if (PyFloat_Check(value)) {
    return at::indexing::scalarToTensor(Scalar(THPUtils_unpackDouble(value)), options, device);
  }
  if (PyComplex_Check(value)) {
    return at::indexing::scalarToTensor(Scalar(THPUtils_unpackComplexDouble(value)), options, device);
  }
  throw TypeError(
    "can't assign a %s to a %s",
    Py_TYPE(value)->tp_name,
    torch::utils::options_to_string(options).c_str());
}

}} // namespace torch::autograd
