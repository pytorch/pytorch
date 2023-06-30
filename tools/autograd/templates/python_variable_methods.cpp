#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// ${generated_comment}

#include <Python.h>

// Undefine the copysign macro so that at::copysign works as intended with MSVC
// https://github.com/python/cpython/blob/c60394c7fc9cc09b16e9675a3eeb5844b6d8523f/PC/pyconfig.h#L196
#ifdef _MSC_VER
#undef copysign
#endif // _MSC_VER

#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/Size.h"
#include "torch/csrc/autograd/generated/VariableType.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/autograd/utils/python_arg_parsing.h"
#include "torch/csrc/autograd/utils/error_messages.h"
#include "torch/csrc/autograd/utils/wrap_outputs.h"
#include "torch/csrc/jit/frontend/tracer.h"
#ifdef USE_CUDA
#include "torch/csrc/cuda/Event.h"
#endif
#include "torch/csrc/utils/cuda_lazy_init.h"
#include "torch/csrc/utils/object_ptr.h"
#include "torch/csrc/utils/pycfunction_helpers.h"
#include "torch/csrc/utils/python_arg_parser.h"
#include "torch/csrc/utils/python_numbers.h"
#include "torch/csrc/utils/python_strings.h"
#include "torch/csrc/utils/python_tuples.h"
#include "torch/csrc/utils/tensor_apply.h"
#include "torch/csrc/utils/tensor_list.h"
#include "torch/csrc/utils/tensor_new.h"
#include "torch/csrc/utils/tensor_numpy.h"
#include "torch/csrc/utils/tensor_types.h"
#include "torch/csrc/utils/structseq.h"
#include "torch/csrc/autograd/python_return_types.h"

#include <ATen/core/Tensor.h>
#include <ATen/FuncTorchTLS.h>
#include "c10/util/Optional.h"
#include "c10/core/Stream.h"

#include <stdexcept>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
$ops_headers
#include <ATen/ops/_local_scalar_dense.h>
#endif

using at::DeviceGuard;
using at::device_of;
using at::OptionalDeviceGuard;
using at::Backend;
using at::Scalar;
using at::ScalarType;
using at::Tensor;
using c10::Stream;
using namespace torch::autograd::utils;

namespace torch { namespace autograd {

static PyObject * THPVariable__is_view(PyObject *self, PyObject* args)
{
  HANDLE_TH_ERRORS
  if (check_has_torch_function(self)) {
    return handle_torch_function(self, "_is_view", args);
  }
  auto& self_ = THPVariable_Unpack(self);
  if (self_.is_view()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

// implemented on the python object bc no support for first-class functions in native_functions.yaml
// See: ATen/native/README.md for more context
static PyObject * THPVariable_apply_(PyObject* self, PyObject* arg)
{
  HANDLE_TH_ERRORS
  if (check_has_torch_function(self)) {
    auto args = py::make_tuple(py::handle(arg));
    return handle_torch_function(self, "apply_", args.ptr());
  }
  auto& self_ = THPVariable_Unpack(self);
  if (self_.requires_grad()) {
    throw std::runtime_error(
        "Can't call apply_() on Variable that requires grad. Use "
        "var.detach().apply_() instead.");
  }
  return THPVariable_Wrap(torch::utils::apply_(self_, arg));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_size(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "size(int64_t dim)",
    "size()",
    "size(Dimname dim)",
  });
  auto& self_ = THPVariable_Unpack(self);
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  if(r.has_torch_function()){
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  if (r.idx == 0) {
    if (jit::tracer::isTracing()) {
      // will error out if a tensor has symints
      return wrap(jit::tracer::getSizeOf(self_, r.toInt64(0)));
    } else {
      return torch::toPyObject(self_.sym_size(r.toInt64(0)));
    }
  } else if (r.idx == 1) {
    return THPSize_NewFromSymSizes(self_);
  }
  else if (r.idx == 2) {
    if (jit::tracer::isTracing()) {
      TORCH_INTERNAL_ASSERT(false, "NYI: Named tensors w/ JIT");
    }
    return wrap(self_.size(r.dimname(0)));
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
    "stride(Dimname dim)",
  });
  auto& self_ = THPVariable_Unpack(self);
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  if(r.has_torch_function()){
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  if (r.idx == 0) {
    return torch::toPyObject(self_.sym_stride(r.toInt64(0)));
  } else if (r.idx == 1) {
    // yes, this is called strides in ATen.
    at::SymIntArrayRef strides = self_.sym_strides();
    // we can't do the normal wrapping here because IntArrayRef maps to both
    // torch.Size and tuple in python
    // TODO: consider factoring this out
    THPObjectPtr tuple(PyTuple_New(strides.size()));
    if (!tuple) throw python_error();
    for (size_t i = 0; i != strides.size(); i++) {
      PyObject* s = torch::toPyObject(strides[i]);
      if (!s) throw python_error();
      PyTuple_SET_ITEM(tuple.get(), i, s);
    }
    return tuple.release();
  }
  else if (r.idx == 2) {
    return wrap(self_.stride(r.dimname(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// implemented on the python object to avoid dispatch overhead
static PyObject * THPVariable_get_device(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  if (check_has_torch_function(self_)) {
    return handle_torch_function(self_, "get_device", args, nullptr);
  }
  auto& self = THPVariable_Unpack(self_);
  return wrap(self.get_device());
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_has_names(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  if (check_has_torch_function(self_)) {
    return handle_torch_function(self_, "has_names", args);
  }
  auto& self = THPVariable_Unpack(self_);
  return wrap(self.has_names());
  END_HANDLE_TH_ERRORS
}

// implemented on the python object to avoid dispatch overhead
static PyObject * THPVariable_data_ptr(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  if (check_has_torch_function(self_)) {
    return handle_torch_function(self_, "data_ptr", args);
  }
  auto& self = THPVariable_Unpack(self_);
  return wrap(self.data_ptr());
  END_HANDLE_TH_ERRORS
}

// implemented on the python object to avoid dispatch overhead
static PyObject * THPVariable_storage_offset(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  if (check_has_torch_function(self_)) {
    return handle_torch_function(self_, "storage_offset");
  }
  auto& self = THPVariable_Unpack(self_);
  return py::cast(self.sym_storage_offset()).release().ptr();
  END_HANDLE_TH_ERRORS
}

// implemented on the python object to avoid dispatch overhead
static PyObject * THPVariable_dim(PyObject* self, PyObject* args)
{
   HANDLE_TH_ERRORS
   if (check_has_torch_function(self)) {
     return handle_torch_function(self, "dim", args);
   }
   auto& self_ = THPVariable_Unpack(self);
   return THPUtils_packInt64(self_.dim());
   END_HANDLE_TH_ERRORS
}

// implemented on the python object to avoid dispatch overhead
static PyObject * THPVariable_numel(PyObject* self, PyObject* args)
{
   HANDLE_TH_ERRORS
   if (check_has_torch_function(self)) {
     return handle_torch_function(self, "numel", args);
   }
   auto& self_ = THPVariable_Unpack(self);
   if (jit::tracer::isTracing()) {
     return wrap(jit::tracer::getNumelOf(self_));
   } else {
     return py::cast(self_.sym_numel()).release().ptr();
   }
   END_HANDLE_TH_ERRORS
}

static Tensor dispatch_contiguous(const Tensor & self, at::MemoryFormat memory_format) {
  pybind11::gil_scoped_release no_gil;
  OptionalDeviceGuard device_guard(device_of(self));
  return self.contiguous(memory_format);
}

static PyObject * THPVariable_contiguous(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "contiguous(*, MemoryFormat memory_format=contiguous_format)",
  });
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  if(r.has_torch_function()){
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  auto& self_ = THPVariable_Unpack(self);
  auto memory_format = r.memoryformat(0);
  // avoids touching the GIL or current device if self is already contiguous
  if (self_.is_contiguous(memory_format)) {
    // NOTE: this logic is duplicated from VariableType.cpp. Since we need to
    // record this call to contiguous() in the trace regardless of whether
    // we actually call contiguous here, we need to record this information
    // manually.
    if (jit::tracer::isTracing()) {
      auto tracer_state = jit::tracer::getTracingState();
      auto op_name = c10::Symbol::fromQualString("aten::contiguous");
      auto node = tracer_state->createNode(op_name, /*num_outputs=*/0);
      jit::tracer::recordSourceLocation(node);
      jit::tracer::addInputs(node, "self", self_);
      jit::tracer::addInputs(node, "memory_format", memory_format);
      tracer_state->insertNode(node);
      jit::tracer::addOutput(node, self_);
    }
    Py_INCREF(self);
    return self;
  }
  return THPVariable_Wrap(dispatch_contiguous(self_, memory_format));
  END_HANDLE_TH_ERRORS
}

static Tensor dispatch_copy_(const Tensor & self, const Tensor & other, bool non_blocking) {
  pybind11::gil_scoped_release no_gil;
  OptionalDeviceGuard device_guard(device_of(self));
  return self.copy_(other, non_blocking);
}

 static PyObject * THPVariable_copy_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "copy_(Tensor other, bool non_blocking=False)",
    "copy_(Tensor other, bool async=False)|deprecated"
  });
  auto& self_ = THPVariable_Unpack(self);
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  if(r.has_torch_function()){
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  return THPVariable_Wrap(dispatch_copy_(self_, r.tensor(0), r.toBool(1)));
  END_HANDLE_TH_ERRORS
}

static double dispatch_to_CDouble(const Tensor & self) {
  pybind11::gil_scoped_release no_gil;
  OptionalDeviceGuard device_guard(device_of(self));
  if (self.sym_numel() != 1) {
    throw ValueError("only one element tensors can be converted to Python scalars");
  }
  return self.item<double>();
}

static c10::complex<double> dispatch_to_CComplexDouble(const Tensor & self) {
  pybind11::gil_scoped_release no_gil;
  OptionalDeviceGuard device_guard(device_of(self));
  if (self.sym_numel() != 1) {
    throw ValueError("only one element tensors can be converted to Python scalars");
  }
  return self.item<c10::complex<double>>();
}

static int64_t dispatch_to_CLong(const Tensor & self) {
  pybind11::gil_scoped_release no_gil;
  OptionalDeviceGuard device_guard(device_of(self));
  if (self.sym_numel() != 1) {
    throw ValueError("only one element tensors can be converted to Python scalars");
  }
  return self.item<int64_t>();
}

static PyObject * THPVariable_float_scalar(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function(self)) {
    return handle_torch_function(self, "__float__", args);
  }
  jit::tracer::warn("Converting a tensor to a Python float", jit::tracer::WARN_PYTHON_DATAFLOW);
  auto& self_ = THPVariable_Unpack(self);
  return wrap(dispatch_to_CDouble(self_));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_complex_scalar(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function(self)) {
    return handle_torch_function(self, "__complex__", args);
  }
  jit::tracer::warn("Converting a tensor to a Python complex", jit::tracer::WARN_PYTHON_DATAFLOW);
  auto& self_ = THPVariable_Unpack(self);
  return wrap(dispatch_to_CComplexDouble(self_));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_integral_scalar(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function(self)) {
    return handle_torch_function(self, "__int__", args);
  }
  jit::tracer::warn("Converting a tensor to a Python integer", jit::tracer::WARN_PYTHON_DATAFLOW);
  auto& self_ = THPVariable_Unpack(self);
  if (isFloatingType(self_.scalar_type())) {
    // we can't dispatch to item<int64_t> here because we want to avoid ATen overflow checks;
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
  if (check_has_torch_function(self)) {
    return handle_torch_function(self, "__index__", args);
  }
  auto& self_ = THPVariable_Unpack(self);
  // TODO: change the condition to `self_.dim() != 0` once we expose scalars
  // in PyTorch.
  if (!isIntegralType(self_.scalar_type(), /*includeBool=*/true) || self_.sym_numel() != 1) {
    throw TypeError("only integer tensors of a single element can be converted to an index");
  }
  return wrap(dispatch_to_CLong(self_));
  END_HANDLE_TH_ERRORS
}

static Tensor dispatch_invert(const Tensor & self) {
  pybind11::gil_scoped_release no_gil;
  OptionalDeviceGuard device_guard(device_of(self));
  return self.bitwise_not();
}

static PyObject * THPVariable_invert(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function(self)) {
    return handle_torch_function(self, "__invert__", args);
  }
  auto& self_ = THPVariable_Unpack(self);
  if (!isIntegralType(self_.scalar_type(), /*includeBool=*/true)) {
    throw TypeError("~ (operator.invert) is only implemented on integer and Boolean-type tensors");
  }
  return THPVariable_Wrap(dispatch_invert(self_));
  END_HANDLE_TH_ERRORS
}

static Tensor dispatch_to(const Tensor & self, Device device, bool non_blocking, bool copy, c10::optional<c10::MemoryFormat> optional_memory_format) {
  pybind11::gil_scoped_release no_gil;
  // NOTE: this is where we record aten::to in the graph during tracing. However, the behavior of aten::to
  // is different with respect to TensorOptions fields that are not present: aten::to inherits fields that
  // are missing from the self argument while the tracer assumes that they should be populated with the
  // default values (eg. float for scalar type). By explicitly copying over the tensor options here we fully
  // specify all tensor options and thus record the proper trace
  return self.to(self.options().device(device).memory_format(optional_memory_format), non_blocking, copy);
}

static Tensor dispatch_to(const Tensor & self, bool non_blocking, bool copy, c10::optional<c10::MemoryFormat> optional_memory_format) {
  AutoNoGIL no_gil;
  return self.to(self.options().memory_format(optional_memory_format), non_blocking, copy);
}

static Tensor dispatch_to(const Tensor & self, ScalarType dtype, bool non_blocking, bool copy, c10::optional<c10::MemoryFormat> optional_memory_format) {
  pybind11::gil_scoped_release no_gil;
  // TODO: Make this call the TensorOptions version, maybe?
  return self.to(dtype, non_blocking, copy, optional_memory_format);
}

static Tensor dispatch_to(const Tensor & self, Device device, ScalarType dtype, bool non_blocking, bool copy, c10::optional<c10::MemoryFormat> optional_memory_format) {
  pybind11::gil_scoped_release no_gil;
  // TODO: Make this call the TensorOptions version, maybe?
  return self.to(device, dtype, non_blocking, copy, optional_memory_format);
}

static PyObject * THPVariable_cpu(PyObject* self, PyObject* args, PyObject* kwargs)
{
   HANDLE_TH_ERRORS
   static PythonArgParser parser({
     "cpu(*, MemoryFormat? memory_format=None)"
   });
   auto& self_ = THPVariable_Unpack(self);
   ParsedArgs<1> parsed_args;
   auto r = parser.parse(self, args, kwargs, parsed_args);

   if(r.has_torch_function()){
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
    }

   auto opt_memory_format = r.memoryformatOptional(0);
   return THPVariable_Wrap(dispatch_to(self_, at::Device(at::DeviceType::CPU), false, false, opt_memory_format));
   END_HANDLE_TH_ERRORS
}

static Tensor dispatch_nonzero(const Tensor & self) {
  pybind11::gil_scoped_release no_gil;
  OptionalDeviceGuard device_guard(device_of(self));
  return self.nonzero();
}

static std::vector<Tensor> dispatch_nonzero_numpy(const Tensor & self) {
  pybind11::gil_scoped_release no_gil;
  OptionalDeviceGuard device_guard(device_of(self));
  return self.nonzero_numpy();
}

static PyObject * THPVariable_nonzero(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "nonzero()",
    "nonzero(*, bool as_tuple)",
  });
  auto& self_ = THPVariable_Unpack(self);
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  if(r.has_torch_function()){
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  if (r.idx == 0 || (r.idx == 1 && !r.toBool(0))) {
    return wrap(dispatch_nonzero(self_));
  } else {
    return wrap(dispatch_nonzero_numpy(self_));
  }
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_cuda(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cuda(Device? device=None, bool non_blocking=False, *, MemoryFormat? memory_format=None)",
    "cuda(Device? device=None, bool async=False, *, MemoryFormat? memory_format=None)|deprecated"
  });
  auto& self_ = THPVariable_Unpack(self);
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  if(r.has_torch_function()){
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  auto device = r.isNone(0) ? at::Device(at::DeviceType::CUDA) : r.device(0);
  auto opt_memory_format = r.memoryformatOptional(2);
  TORCH_CHECK(device.is_cuda(), "Invalid device, must be cuda device");
  torch::utils::cuda_lazy_init();
  return THPVariable_Wrap(dispatch_to(self_, device, r.toBool(1), false, opt_memory_format));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_xpu(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "xpu(Device? device=None, bool non_blocking=False, *, MemoryFormat? memory_format=None)",
    "xpu(Device? device=None, bool async=False, *, MemoryFormat? memory_format=None)|deprecated"
  });
  auto& self_ = THPVariable_Unpack(self);
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  if (r.has_torch_function()) {
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  auto device = r.isNone(0) ? at::Device(at::DeviceType::XPU) : r.device(0);
  auto opt_memory_format = r.memoryformatOptional(2);
  TORCH_CHECK(device.is_xpu(), "Invalid device, must be xpu device");
  return THPVariable_Wrap(dispatch_to(self_, device, r.toBool(1), false, opt_memory_format));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_ipu(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "ipu(Device? device=None, bool non_blocking=False, *, MemoryFormat? memory_format=None)",
    "ipu(Device? device=None, bool async=False, *, MemoryFormat? memory_format=None)|deprecated"
  });
  auto& self_ = THPVariable_Unpack(self);
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  if (r.has_torch_function()) {
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  auto device = r.isNone(0) ? at::Device(at::DeviceType::IPU) : r.device(0);
  auto opt_memory_format = r.memoryformatOptional(2);
  TORCH_CHECK(device.is_ipu(), "Invalid device, must be ipu device");
  return THPVariable_Wrap(dispatch_to(self_, device, r.toBool(1), false, opt_memory_format));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_to_type(PyObject* self, ScalarType scalarType, c10::optional<c10::MemoryFormat> optional_memory_format) {
  HANDLE_TH_ERRORS
  auto& self_ = THPVariable_Unpack(self);
  return THPVariable_Wrap(dispatch_to(self_, scalarType, false, false, optional_memory_format));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_byte(PyObject* self, PyObject* args, PyObject* kwargs)  {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "byte(*, MemoryFormat? memory_format=None)"
  });
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  if(r.has_torch_function()){
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  auto opt_memory_format = r.memoryformatOptional(0);
  return THPVariable_to_type(self, ScalarType::Byte, opt_memory_format);
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_char(PyObject* self, PyObject* args, PyObject* kwargs)  {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "char(*, MemoryFormat? memory_format=None)"
  });
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  if(r.has_torch_function()){
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  auto opt_memory_format = r.memoryformatOptional(0);
  return THPVariable_to_type(self, ScalarType::Char, opt_memory_format);
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_double(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "double(*, MemoryFormat? memory_format=None)"
  });
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  if(r.has_torch_function()){
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  auto opt_memory_format = r.memoryformatOptional(0);
  return THPVariable_to_type(self, ScalarType::Double, opt_memory_format);
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_float(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "float(*, MemoryFormat? memory_format=None)"
  });
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  if(r.has_torch_function()){
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  auto opt_memory_format = r.memoryformatOptional(0);
  return THPVariable_to_type(self, ScalarType::Float, opt_memory_format);
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_cdouble(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cdouble(*, MemoryFormat? memory_format=None)"
  });
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  if(r.has_torch_function()){
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  auto opt_memory_format = r.memoryformatOptional(0);
  return THPVariable_to_type(self, ScalarType::ComplexDouble, opt_memory_format);
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_cfloat(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cfloat(*, MemoryFormat? memory_format=None)"
  });
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  if(r.has_torch_function()){
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  auto opt_memory_format = r.memoryformatOptional(0);
  return THPVariable_to_type(self, ScalarType::ComplexFloat, opt_memory_format);
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_half(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "half(*, MemoryFormat? memory_format=None)"
  });
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  if(r.has_torch_function()){
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  auto opt_memory_format = r.memoryformatOptional(0);
  return THPVariable_to_type(self, ScalarType::Half, opt_memory_format);
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_int(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "int(*, MemoryFormat? memory_format=None)"
  });
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  if(r.has_torch_function()){
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  auto opt_memory_format = r.memoryformatOptional(0);
  return THPVariable_to_type(self, ScalarType::Int, opt_memory_format);
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_long(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "long(*, MemoryFormat? memory_format=None)"
  });
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  if(r.has_torch_function()){
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  auto opt_memory_format = r.memoryformatOptional(0);
  return THPVariable_to_type(self, ScalarType::Long, opt_memory_format);
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_short(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "short(*, MemoryFormat? memory_format=None)"
  });
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  if(r.has_torch_function()){
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  auto opt_memory_format = r.memoryformatOptional(0);
  return THPVariable_to_type(self, ScalarType::Short, opt_memory_format);
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_bool(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "bool(*, MemoryFormat? memory_format=None)"
  });
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  if(r.has_torch_function()){
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  auto opt_memory_format = r.memoryformatOptional(0);
  return THPVariable_to_type(self, ScalarType::Bool, opt_memory_format);
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_bfloat16(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "bfloat16(*, MemoryFormat? memory_format=None)"
  });
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  if(r.has_torch_function()){
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  auto opt_memory_format = r.memoryformatOptional(0);
  return THPVariable_to_type(self, ScalarType::BFloat16, opt_memory_format);
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_element_size(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  if (check_has_torch_function(self)) {
    return handle_torch_function(self, "element_size", args);
  }
  auto& self_ = THPVariable_Unpack(self);
  return THPUtils_packInt64(self_.element_size());
  END_HANDLE_TH_ERRORS
}

// implemented on the python object bc PyObjects not declarable in native_functions.yaml
// See: ATen/native/README.md for more context
static PyObject * THPVariable_numpy(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "numpy(*, bool force=False)"
  });
  auto& self_ = THPVariable_Unpack(self);
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  if (r.has_torch_function()) {
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  jit::tracer::warn("Converting a tensor to a NumPy array", jit::tracer::WARN_PYTHON_DATAFLOW);
  return torch::utils::tensor_to_numpy(self_, r.toBool(0));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_requires_grad_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "requires_grad_(bool requires_grad=True)",
  });
  auto& self_ = THPVariable_Unpack(self);
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  if(r.has_torch_function()){
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  // temporary hack to improve functorch UX.
  const auto& functorch_tls = at::functorch::functorchTLSAccessor();
  if (functorch_tls) {
    functorch_tls->checkSupportsInplaceRequiresGrad();
  }

  auto requires_grad = r.toBool(0);
  // should we throw if requires_grad is true?  var.requires_grad = True throws here
  // but it's nice to let this be a no-op.
  if (!self_.is_leaf() && !requires_grad) {
    throw std::runtime_error(autograd::utils::requires_grad_leaf_error(requires_grad));
  }
  if (requires_grad && ! isDifferentiableType(at::typeMetaToScalarType(self_.dtype()))) {
    throw std::runtime_error("only Tensors of floating point dtype can require gradients");
  }
  self_.set_requires_grad(requires_grad);
  return THPVariable_Wrap(self_);
  END_HANDLE_TH_ERRORS
}

inline bool dispatch_is_contiguous(const Tensor & self, MemoryFormat memory_format) {
  return self.is_contiguous(memory_format);
}

// implemented on the python object to avoid dispatch overhead
static PyObject * THPVariable_is_contiguous(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "is_contiguous(*, MemoryFormat memory_format=contiguous_format)",
  });
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(self_, args, kwargs, parsed_args);

  if(r.has_torch_function()){
    return handle_torch_function(r, self_, args, kwargs, PyObject_Type(self_), "torch.Tensor");
  }

  auto memory_format = r.memoryformat(0);
  auto& self = THPVariable_Unpack(self_);
  return wrap(dispatch_is_contiguous(self, memory_format));
  END_HANDLE_TH_ERRORS
}

// implemented on the python object to avoid dispatch overhead
static PyObject * THPVariable_item(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  if (check_has_torch_function(self)) {
    return handle_torch_function(self, "item", args);
  }
  jit::tracer::warn("Converting a tensor to a Python number", jit::tracer::WARN_PYTHON_DATAFLOW);
  auto& self_ = THPVariable_Unpack(self);
  auto dispatch_item_ = [](const Tensor& self) -> at::Scalar {
    pybind11::gil_scoped_release no_gil;
    return self.item();
  };
  return py::cast(dispatch_item_(self_)).release().ptr();
  END_HANDLE_TH_ERRORS
}

// implemented on the python object bc no support for first class functions in native_functions.yaml
// See: ATen/native/README.md for more context
static PyObject * THPVariable_map_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({ "map_(Tensor other, PyObject* callable)" });
  auto& self_ = THPVariable_Unpack(self);
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  if(r.has_torch_function()){
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  Variable other = r.tensor(0);
  if (self_.requires_grad() || other.requires_grad()) {
    throw std::runtime_error(
        "Can't call map_() on Variable that requires grad. Use "
        "var.detach().map_() instead.");
  }
  TORCH_CHECK(
      !self_.unsafeGetTensorImpl()->is_python_dispatch() && !other.unsafeGetTensorImpl()->is_python_dispatch(),
      ".map_ is not supported for tensor subclasses.");

  return THPVariable_Wrap(torch::utils::map_(self_, other, r.pyobject(1)));
  END_HANDLE_TH_ERRORS
}

// implemented on the python object bc no support for first class functions in native_functions.yaml
// See: ATen/native/README.md for more context
static PyObject * THPVariable_map2_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({ "map2_(Tensor x, Tensor y, PyObject* callable)" });
  auto& self_ = THPVariable_Unpack(self);
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  if(r.has_torch_function()){
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  Variable x = r.tensor(0);
  Variable y = r.tensor(1);
  if (self_.requires_grad() || x.requires_grad() || y.requires_grad()) {
    throw std::runtime_error(
        "Can't call map2_() on Variable that requires grad. Use "
        "var.detach().map2_() instead.");
  }
  TORCH_CHECK(
      !x.unsafeGetTensorImpl()->is_python_dispatch() && !y.unsafeGetTensorImpl()->is_python_dispatch(),
      ".map2_ is not supported for tensor subclasses.");
  return THPVariable_Wrap(torch::utils::map2_(self_, x, y, r.pyobject(2)));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_new(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  if (check_has_torch_function(self)) {
    return handle_torch_function(self, "new", args, kwargs);
  }
  auto& self_ = THPVariable_Unpack(self);
  OptionalDeviceGuard device_guard(device_of(self_));
  return THPVariable_Wrap(torch::utils::legacy_tensor_new(legacyExtractDispatchKey(self_), self_.scalar_type(), args, kwargs));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_new_tensor(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  if (check_has_torch_function(self)) {
    return handle_torch_function(self, "new_tensor", args, kwargs);
  }
  auto& self_ = THPVariable_Unpack(self);
  OptionalDeviceGuard device_guard(device_of(self_));
  return THPVariable_Wrap(torch::utils::new_tensor(legacyExtractDispatchKey(self_), self_.scalar_type(), args, kwargs));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_storage(PyObject* self, PyObject* arg)
{
  HANDLE_TH_ERRORS
  if (check_has_torch_function(self)) {
    return handle_torch_function(self, "untyped_storage");
  }

  // For tensor subclasses that are traceable,
  // we need to be able to track aliasing of subclasses in AOTAutograd.
  // __tensor_flatten__/__tensor_unflatten__ are part of the contract
  // for "declaring your subclass to be traceable",
  // So we detect this situation and allow python storage objects to be returned.
  // In the future we could think about making subclass storages always visible in python,
  // although it has some footguns (you can use them to loosely track aliasing,
  // but they don't have a valid data pointer).
  py::object maybe_tensor_flatten = PyObject_FastGetAttrString(self, "__tensor_flatten__");
  py::object maybe_tensor_unflatten = PyObject_FastGetAttrString(self, "__tensor_unflatten__");
  auto is_traceable_subclass = maybe_tensor_flatten && maybe_tensor_unflatten;

  auto& self_ = THPVariable_Unpack(self);
  return createPyObject(self_.storage(), /*always_create_storage=*/is_traceable_subclass);
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_to(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "to(Device device=None, ScalarType dtype=None, bool non_blocking=False, bool copy=False, *, MemoryFormat? memory_format=None)",
    "to(ScalarType dtype, bool non_blocking=False, bool copy=False, *, MemoryFormat? memory_format=None)",
    "to(Tensor tensor, bool non_blocking=False, bool copy=False, *, MemoryFormat? memory_format=None)",
  });
  ParsedArgs<5> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);
  if (r.has_torch_function()) {
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  auto parsed = parse_to_conversion(r, /*allow_copy*/ true);
  auto& device = std::get<0>(parsed);
  auto& scalarType = std::get<1>(parsed);
  auto non_blocking = std::get<2>(parsed);
  auto copy = std::get<3>(parsed);
  auto opt_memory_format = std::get<4>(parsed);
  auto& self_ = THPVariable_Unpack(self);
  if (device && device->is_cuda()) {
    torch::utils::cuda_lazy_init();
  }
  if (!device && !scalarType && !copy && !opt_memory_format.has_value()) {
    Py_INCREF(self);
    return self;
  } else if (!device && !scalarType) {
    return THPVariable_Wrap(
        dispatch_to(self_, non_blocking, copy, opt_memory_format));
  } else if (!device) {
    return THPVariable_Wrap(dispatch_to(self_, *scalarType, non_blocking, copy, opt_memory_format));
  } else if (!scalarType) {
    return THPVariable_Wrap(dispatch_to(self_, *device, non_blocking, copy, opt_memory_format));
  } else {
    return THPVariable_Wrap(dispatch_to(self_, *device, *scalarType, non_blocking, copy, opt_memory_format));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// implemented on the python object b/c arbitrarily nested list not declarable in native_functions.yaml
// See: ATen/native/README.md for more context
static PyObject * THPVariable_tolist(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  if (check_has_torch_function(self)) {
    return handle_torch_function(self, "tolist", args);
  }
  jit::tracer::warn("Converting a tensor to a Python list", jit::tracer::WARN_PYTHON_DATAFLOW);
  auto self_ = THPVariable_Unpack(self);
  return torch::utils::tensor_to_list(self_);
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_type(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "type(PyObject* dtype=None, bool non_blocking=False, *, MemoryFormat? memory_format=None)",
    "type(PyObject* dtype=None, bool async=False, *, MemoryFormat? memory_format=None)|deprecated"
  });
  auto& self_ = THPVariable_Unpack(self);
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  if(r.has_torch_function()){
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  if (r.isNone(0)) {
    return THPUtils_packString(torch::utils::options_to_string(self_.options()));
  }
  auto obj = r.pyobject(0);
  auto opt_memory_format = r.memoryformatOptional(2);
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
  ScalarType scalar_type;
  Device device = self_.device();
  if (is_dtype) {
    scalar_type = r.scalartype(0);
  } else {
    at::TensorOptions options = torch::utils::options_from_string(type_name);
    scalar_type = at::typeMetaToScalarType(options.dtype());
    auto device_type = options.device().type();
    if (device_type != device.type()) {
      device = at::Device(device_type);
    }
  }
  if (device.is_cuda()) {
    torch::utils::cuda_lazy_init();
  }
  return THPVariable_Wrap(dispatch_to(self_, device, scalar_type, /*non_blocking=*/ r.toBool(1), /*copy=*/ false, opt_memory_format));
  END_HANDLE_TH_ERRORS
}

// generated methods start here

${py_methods}

static PyObject * THPVariable_bool_scalar(PyObject* self, PyObject* args) {
  if (check_has_torch_function(self)) {
    HANDLE_TH_ERRORS
    return handle_torch_function(self, "__bool__", args);
    END_HANDLE_TH_ERRORS
  }
  jit::tracer::warn("Converting a tensor to a Python boolean", jit::tracer::WARN_PYTHON_DATAFLOW);
  return THPVariable_is_nonzero(self, args);
}

// Wrapper converts a raised TypeError into returning NotImplemented
// Used to implement binary arithmetic operators
template <PyObject* (*Func)(PyObject*, PyObject*, PyObject*)>
static PyObject * TypeError_to_NotImplemented_(PyObject* self, PyObject* args, PyObject* kwargs) {

  PyObject* ret = Func(self, args, kwargs);
  if (!ret && PyErr_ExceptionMatches(PyExc_TypeError)) {
    PyErr_Clear();
    Py_INCREF(Py_NotImplemented);
    ret = Py_NotImplemented;
  }
  return ret;
}

// set_ has to be defined in the template because the c10::Storage object
// does not have a type, and we need to make sure the Python storage object's
// type matches the tensor's type
static PyObject* THPVariable_set_(
    PyObject* self_,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser(
      {
          "set_()",
          "set_(Storage source)",
          "set_(Storage source, SymInt storage_offset, SymIntArrayRef size, SymIntArrayRef stride=None)",
          "set_(Tensor source)",
          "set_(Tensor source, SymInt storage_offset, SymIntArrayRef size, SymIntArrayRef stride=None)",
      },
      /*traceable=*/false);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::set_(Tensor(a!) self) -> Tensor(a!)
      auto dispatch_set_ = [](const Tensor& self) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.set_();
      };
      return wrap(dispatch_set_(self));
    }
    case 1: {
      // aten::set_.source_Storage(Tensor(a!) self, Storage source) ->
      // Tensor(a!)
      at::ScalarType storage_scalar_type;
      bool is_typed_storage = true;
      at::Storage storage = _r.storage(0, storage_scalar_type, is_typed_storage);
      TORCH_CHECK(storage_scalar_type == self.dtype() || !is_typed_storage,
        "Expected a Storage of type ", self.dtype(),
        " or an UntypedStorage, but got type ", storage_scalar_type,
        " for argument 1 'storage'");
      auto dispatch_set_ = [](const Tensor& self, Storage source) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.set_(source);
      };
      return wrap(dispatch_set_(self, storage));
    }
    case 2: {
      // aten::set_.source_Storage_storage_offset(Tensor(a!) self, Storage
      // source, int storage_offset, int[] size, int[] stride=[]) -> Tensor(a!)
      at::ScalarType storage_scalar_type;
      bool is_typed_storage = true;
      at::Storage storage = _r.storage(0, storage_scalar_type, is_typed_storage);
      TORCH_CHECK(storage_scalar_type == self.dtype() || !is_typed_storage,
        "Expected a Storage of type ", self.dtype(),
        " or an UntypedStorage, but got type ", storage_scalar_type,
        " for argument 1 'storage'");
      auto dispatch_set_ = [](const Tensor& self,
                              Storage source,
                              c10::SymInt storage_offset,
                              c10::SymIntArrayRef size,
                              c10::SymIntArrayRef stride) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.set__symint(source, storage_offset, size, stride);
      };
      return wrap(dispatch_set_(
          self, storage, _r.toSymInt(1), _r.symintlist(2), _r.symintlist(3)));
    }
    case 3: {
      // aten::set_.source_Tensor(Tensor(a!) self, Tensor source) -> Tensor(a!)
      auto dispatch_set_ = [](const Tensor& self, const Tensor& source) -> Tensor {
        TORCH_CHECK(source.dtype() == self.dtype(), "Could not set tensor of type ", source.dtype(), " to a tensor of type ", self.dtype());
        pybind11::gil_scoped_release no_gil;
        return self.set_(source);
      };
      return wrap(dispatch_set_(self, _r.tensor(0)));
    }
    case 4: {
      // aten::set_.source_Tensor_storage_offset(Tensor(a!) self, Tensor
      // source, int storage_offset, int[] size, int[] stride=[]) -> Tensor(a!)
      at::Tensor storage = _r.tensor(0);
      auto dispatch_set_ = [](const Tensor& self,
                              const Tensor& source,
                              c10::SymInt storage_offset,
                              c10::SymIntArrayRef size,
                              c10::SymIntArrayRef stride) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.set__symint(source, storage_offset, size, stride);
      };
      return wrap(dispatch_set_(
          self, storage, _r.toSymInt(1), _r.symintlist(2), _r.symintlist(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// XXX: ops that are bound here are not exposed to the C++ api nor the JIT.
// Any new ops added here should be accompanied with a comment why they are not
// being registered through native_functions.yaml, and be tagged cpp / JIT
PyMethodDef variable_methods[] = {
  // These magic methods are all implemented on python object to wrap NotImplementedError
  {"__add__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable_add>), METH_VARARGS | METH_KEYWORDS, NULL},
  {"__radd__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable_add>), METH_VARARGS | METH_KEYWORDS, NULL},
  {"__iadd__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable_add_>), METH_VARARGS | METH_KEYWORDS, NULL},
  {"__rmul__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable_mul>), METH_VARARGS | METH_KEYWORDS, NULL},
  {"__mul__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable_mul>), METH_VARARGS | METH_KEYWORDS, NULL},
  {"__imul__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable_mul_>), METH_VARARGS | METH_KEYWORDS, NULL},
  {"__sub__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable_sub>), METH_VARARGS | METH_KEYWORDS, NULL},
  {"__isub__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable_sub_>), METH_VARARGS | METH_KEYWORDS, NULL},
  {"__div__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable_div>), METH_VARARGS | METH_KEYWORDS, NULL},
  {"__truediv__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable_div>), METH_VARARGS | METH_KEYWORDS, NULL},
  {"__floordiv__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable_floor_divide>), METH_VARARGS | METH_KEYWORDS, NULL},
  {"__idiv__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable_div_>), METH_VARARGS | METH_KEYWORDS, NULL},
  {"__ifloordiv__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable_floor_divide_>), METH_VARARGS | METH_KEYWORDS, NULL},
  {"__mod__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable_remainder>), METH_VARARGS | METH_KEYWORDS, NULL},
  {"__imod__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable_remainder_>), METH_VARARGS | METH_KEYWORDS, NULL},
  {"__eq__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable_eq>), METH_VARARGS | METH_KEYWORDS, NULL},
  {"__ne__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable_ne>), METH_VARARGS | METH_KEYWORDS, NULL},
  {"__lt__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable_lt>), METH_VARARGS | METH_KEYWORDS, NULL},
  {"__le__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable_le>), METH_VARARGS | METH_KEYWORDS, NULL},
  {"__gt__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable_gt>), METH_VARARGS | METH_KEYWORDS, NULL},
  {"__ge__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable_ge>), METH_VARARGS | METH_KEYWORDS, NULL},
  {"__rand__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable_bitwise_and>), METH_VARARGS | METH_KEYWORDS, NULL},
  {"__ror__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable_bitwise_or>), METH_VARARGS | METH_KEYWORDS, NULL},
  {"__rxor__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable_bitwise_xor>), METH_VARARGS | METH_KEYWORDS, NULL},
  {"__bool__", THPVariable_bool_scalar, METH_NOARGS, NULL},
  {"__float__", THPVariable_float_scalar, METH_NOARGS, NULL},
  {"__complex__", THPVariable_complex_scalar, METH_NOARGS, NULL},
  {"__int__", THPVariable_integral_scalar, METH_NOARGS, NULL},
  {"__long__", THPVariable_integral_scalar, METH_NOARGS, NULL},
  {"__index__", THPVariable_index_scalar, METH_NOARGS, NULL},
  {"__nonzero__", THPVariable_bool_scalar, METH_NOARGS, NULL},
  {"__invert__", THPVariable_invert, METH_NOARGS, NULL},
  {"__matmul__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable_matmul>), METH_VARARGS | METH_KEYWORDS, NULL},
  {"_is_view", THPVariable__is_view, METH_NOARGS, NULL},
  {"apply_", THPVariable_apply_, METH_O, NULL},
  {"bfloat16", castPyCFunctionWithKeywords(THPVariable_bfloat16), METH_VARARGS | METH_KEYWORDS, NULL},
  {"byte", castPyCFunctionWithKeywords(THPVariable_byte), METH_VARARGS | METH_KEYWORDS, NULL},
  {"char", castPyCFunctionWithKeywords(THPVariable_char), METH_VARARGS | METH_KEYWORDS, NULL},
  {"contiguous", castPyCFunctionWithKeywords(THPVariable_contiguous), METH_VARARGS | METH_KEYWORDS, NULL},
  {"copy_", castPyCFunctionWithKeywords(THPVariable_copy_), METH_VARARGS | METH_KEYWORDS, NULL},
  {"cpu", castPyCFunctionWithKeywords(THPVariable_cpu), METH_VARARGS | METH_KEYWORDS, NULL},
  {"cuda", castPyCFunctionWithKeywords(THPVariable_cuda), METH_VARARGS | METH_KEYWORDS, NULL},
  {"xpu", castPyCFunctionWithKeywords(THPVariable_xpu), METH_VARARGS | METH_KEYWORDS, NULL},
  {"ipu", castPyCFunctionWithKeywords(THPVariable_ipu), METH_VARARGS | METH_KEYWORDS, NULL},
  {"data_ptr", THPVariable_data_ptr, METH_NOARGS, NULL},
  {"dim", THPVariable_dim, METH_NOARGS, NULL},
  {"has_names", THPVariable_has_names, METH_NOARGS, NULL},
  {"double", castPyCFunctionWithKeywords(THPVariable_double), METH_VARARGS | METH_KEYWORDS, NULL},
  {"cdouble", castPyCFunctionWithKeywords(THPVariable_cdouble), METH_VARARGS | METH_KEYWORDS, NULL},
  {"element_size", THPVariable_element_size, METH_NOARGS, NULL},
  {"float", castPyCFunctionWithKeywords(THPVariable_float), METH_VARARGS | METH_KEYWORDS, NULL},
  {"cfloat", castPyCFunctionWithKeywords(THPVariable_cfloat), METH_VARARGS | METH_KEYWORDS, NULL},
  {"get_device", THPVariable_get_device, METH_NOARGS, NULL},
  {"bool", castPyCFunctionWithKeywords(THPVariable_bool), METH_VARARGS | METH_KEYWORDS, NULL},
  {"half", castPyCFunctionWithKeywords(THPVariable_half), METH_VARARGS | METH_KEYWORDS, NULL},
  {"int", castPyCFunctionWithKeywords(THPVariable_int), METH_VARARGS | METH_KEYWORDS, NULL},
  {"is_contiguous", castPyCFunctionWithKeywords(THPVariable_is_contiguous), METH_VARARGS | METH_KEYWORDS, NULL},
  {"item", THPVariable_item, METH_NOARGS, NULL},
  {"long", castPyCFunctionWithKeywords(THPVariable_long), METH_VARARGS | METH_KEYWORDS, NULL},
  {"map_", castPyCFunctionWithKeywords(THPVariable_map_), METH_VARARGS | METH_KEYWORDS, NULL},
  {"map2_", castPyCFunctionWithKeywords(THPVariable_map2_), METH_VARARGS | METH_KEYWORDS, NULL},
  {"ndimension", THPVariable_dim, METH_NOARGS, NULL},
  {"nelement", THPVariable_numel, METH_NOARGS, NULL},
  {"new", castPyCFunctionWithKeywords(THPVariable_new), METH_VARARGS | METH_KEYWORDS, NULL},
  {"new_tensor", castPyCFunctionWithKeywords(THPVariable_new_tensor), METH_VARARGS | METH_KEYWORDS, NULL},
  {"nonzero", castPyCFunctionWithKeywords(THPVariable_nonzero), METH_VARARGS | METH_KEYWORDS, NULL},
  {"numel", THPVariable_numel, METH_NOARGS, NULL},
  {"numpy", castPyCFunctionWithKeywords(THPVariable_numpy), METH_VARARGS | METH_KEYWORDS, NULL},
  {"requires_grad_", castPyCFunctionWithKeywords(THPVariable_requires_grad_), METH_VARARGS | METH_KEYWORDS, NULL},
  {"set_", castPyCFunctionWithKeywords(THPVariable_set_), METH_VARARGS | METH_KEYWORDS, NULL},
  {"short", castPyCFunctionWithKeywords(THPVariable_short), METH_VARARGS | METH_KEYWORDS, NULL},
  {"size", castPyCFunctionWithKeywords(THPVariable_size), METH_VARARGS | METH_KEYWORDS, NULL},
  {"untyped_storage", THPVariable_storage, METH_NOARGS, NULL},
  {"storage_offset", THPVariable_storage_offset, METH_NOARGS, NULL},
  {"stride", castPyCFunctionWithKeywords(THPVariable_stride), METH_VARARGS | METH_KEYWORDS, NULL},
  {"to", castPyCFunctionWithKeywords(THPVariable_to), METH_VARARGS | METH_KEYWORDS, NULL},
  {"tolist", THPVariable_tolist, METH_NOARGS, NULL},
  {"type", castPyCFunctionWithKeywords(THPVariable_type), METH_VARARGS | METH_KEYWORDS, NULL},
  ${py_method_defs}
  {NULL}
};

}} // namespace torch::autograd
